from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from scope_server import ScopePolicy

from data_guard import PARTITION, STAGE1, check_episode, partition, sha256, write_json
from features import CAMERAS, CurrentFeatureBank, Geometry, SpatialAdapter, observation_key
from policy import RecurrentSession
from training import Decision, Transition, build_trainer
from trajectory import ExecutionFeedback, cost_aware_reward


class Stage2Policy(ScopePolicy):
    def __init__(self, *, output, regime="C", seed=101, spatial_adapter=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        write_json(self.output / "source_hashes.json", {path.name: sha256(path) for path in ROOT.glob("*.py")})
        manifest = json.loads((STAGE1 / "EXECUTION_MANIFEST.json").read_text())
        checkpoint = manifest["checkpoint"]
        if sha256(checkpoint["path"]) != checkpoint["sha256"]:
            raise ValueError("frozen checkpoint changed")
        super().__init__(checkpoint=checkpoint["path"], legacy_root=str(ROOT.parent / "legacy_runtime"),
                         qwen_root=checkpoint["qwen_root"], telemetry=self.output / "scope.jsonl")
        self.model.requires_grad_(False)
        self.trainer = build_trainer(regime, "cuda", seed)
        self.spatial_adapter = None
        if spatial_adapter is not None:
            self.spatial_adapter = SpatialAdapter().to("cuda")
            self.spatial_adapter.load_state_dict(torch.load(spatial_adapter, map_location="cuda", weights_only=True))
            self.spatial_adapter.requires_grad_(False).eval()
        self.encoder_hash = checkpoint["sha256"]
        self.provenance = {"backbone": self.encoder_hash, "regime": regime, "seed": seed,
                           "partition_sha256": sha256(PARTITION)}
        self.regime = regime
        self.active_bank = None
        self.current_blocks = []
        self.visual_calls = []
        self.encode_phase = "current"
        self.native_pixel_chunks = []
        self.action_generator = torch.Generator(device="cuda").manual_seed(seed)
        original_build = self.model._build_qwen35_inputs
        original_encode = self.model._encode_online_postmerge
        original_instruction = self.model._build_instruction

        def build(*args, **kwargs):
            inputs, blocks = original_build(*args, **kwargs)
            self.current_blocks = blocks
            self.last_context_grids = inputs["_nav_context_image_grid_thw"].detach().cpu().tolist()
            return inputs, blocks

        def encode(inputs):
            if self.active_bank is None:
                return original_encode(inputs)
            from starVLA.model.modules.qwen35_vision import encode_qwen35_postmerge_batched
            indices = inputs["_nav_online_indices"]
            grids = inputs["_nav_online_image_grid_thw"]
            chunks = torch.split(inputs["pixel_values"], grids.prod(-1).tolist())
            output, missing, missing_positions = [], [], []
            for position, block_index in enumerate(indices):
                block = self.current_blocks[block_index]
                if block.get("is_history", False):
                    output.append(None)
                    missing.append(position)
                    missing_positions.append(len(output) - 1)
                    continue
                camera = CAMERAS.index(block["camera_name"])
                geometry = (Geometry.focus(int(block["candidate_id"])) if block.get("scope_kind") == "focus"
                            else Geometry.glance(camera))
                feature = self.active_bank.read(geometry)
                if tuple(grids[position].tolist()) != feature.grid:
                    raise ValueError("cached current grid differs from native online grid")
                output.append(feature.tokens if self.spatial_adapter is None else self.spatial_adapter(feature))
            if missing:
                visual = self.model.qwen35_vl_interface.model.model.visual
                self.encode_phase = "history"
                encoded = encode_qwen35_postmerge_batched(visual, torch.cat([chunks[index] for index in missing]).to(visual.dtype),
                                                          grids[missing])
                self.encode_phase = "current"
                for position, value in zip(missing_positions, encoded, strict=True):
                    output[position] = value
            self.reused_current += sum(not self.current_blocks[index].get("is_history", False) for index in indices)
            if self.verify_reencode:
                self.encode_phase = "diagnostic_reencode"
                batched_reference = original_encode(inputs)
                current_positions = [position for position, index in enumerate(indices)
                                     if not self.current_blocks[index].get("is_history", False)]
                reference = list(output)
                canonical_groups = [current_positions[:4]] + [[position] for position in current_positions[4:]]
                visual = self.model.qwen35_vl_interface.model.model.visual
                for positions in canonical_groups:
                    encoded = encode_qwen35_postmerge_batched(visual, torch.cat([chunks[position] for position in positions]).to(visual.dtype),
                                                              grids[positions])
                    for position, tokens in zip(positions, encoded, strict=True):
                        reference[position] = tokens
                self.encode_phase = "current"
                errors = []
                current_cursor = 0
                for position, block_index in enumerate(indices):
                    if self.current_blocks[block_index].get("is_history", False):
                        continue
                    difference = output[position].float() - reference[position].float()
                    errors.append({"max_abs": float(difference.abs().max()),
                                   "relative_l2": float(difference.norm() / reference[position].float().norm().clamp_min(1e-8)),
                                   "pixel_max_abs": float((chunks[position].float() - self.native_pixel_chunks[current_cursor].float()).abs().max()),
                                   "old_batched_relative_l2": float((output[position].float() - batched_reference[position].float()).norm() /
                                                                   batched_reference[position].float().norm().clamp_min(1e-8))})
                    current_cursor += 1
                self.reencode_errors = errors
                if any(record["max_abs"] != 0 or record["pixel_max_abs"] != 0 for record in errors):
                    write_json(self.output / "native_feature_mismatch.json", errors)
                    raise AssertionError(f"native cached/scatter feature mismatch: {errors}")
                if self.use_native_features:
                    return reference
            return output

        def instruction(sample):
            text = original_instruction(sample)
            if getattr(self, "annotate", False):
                labels = [f"{CAMERAS[Geometry.focus(candidate).camera]} ROI pixels={Geometry.focus(candidate).roi} of 512x512"
                          for candidate in self.active_bank.selected]
                text += "\nCurrent images end with four full-view Glances in front,left,right,rear order; then cropped Focus images: " + "; ".join(labels)
            return text

        self.model._build_qwen35_inputs = build
        self.model._encode_online_postmerge = encode
        self.model._build_instruction = instruction
        visual = self.model.qwen35_vl_interface.model.model.visual

        def capture(module, args, kwargs):
            self.visual_calls.append({"phase": self.encode_phase, "grids": kwargs["grid_thw"].detach().cpu().tolist()})

        visual.register_forward_pre_hook(capture, with_kwargs=True)
        write_json(self.output / "model_contract.json", {"backbone": self.encoder_hash, "regime": regime,
                   "actor_parameters": sum(parameter.numel() for parameter in self.trainer.actor.parameters()),
                   "critic_parameters": sum(parameter.numel() for parameter in self.trainer.critic.parameters()),
                   "frozen_parameters": sum(parameter.numel() for parameter in self.model.parameters()),
                   "backbone_frozen": all(not parameter.requires_grad for parameter in self.model.parameters()),
                   "visual_class": type(visual).__name__, "dtype": str(visual.dtype)})

    def new_session(self, **kwargs):
        session = super().new_session(**kwargs)
        session.update(recurrent=RecurrentSession("cuda"), pending=None, rollout=[], purpose=None, records=[])
        return session

    @torch.inference_mode()
    def native_encode(self, images):
        from starVLA.model.modules.qwen35_vision import encode_qwen35_postmerge_batched
        interface = self.model.qwen35_vl_interface
        processed = interface.processor.image_processor(images=images, do_resize=False, return_tensors="pt")
        visual = interface.model.model.visual
        grids = processed["image_grid_thw"].to("cuda")
        self.native_pixel_chunks.extend(chunk.to("cuda") for chunk in torch.split(processed["pixel_values"], grids.prod(-1).tolist()))
        tokens = encode_qwen35_postmerge_batched(visual, processed["pixel_values"].to(device="cuda", dtype=visual.dtype), grids)
        return list(zip(tokens, grids.cpu().tolist(), strict=True))

    @torch.no_grad()
    def language(self, instruction):
        tokenizer = self.model.qwen35_vl_interface.processor.tokenizer
        identifiers = tokenizer(instruction, add_special_tokens=False)["input_ids"]
        if len(identifiers) > 256:
            identifiers = identifiers[:128] + identifiers[-128:]
        if not identifiers:
            raise ValueError("empty instruction")
        embedding = self.model.qwen35_vl_interface.model.get_input_embeddings()
        return embedding(torch.tensor(identifiers, device="cuda")).float().mean(0).detach()

    def bank_for(self, frame):
        from starVLA.model.modules.input_context import prepare_scope_images
        raw, compatible, glance, candidates = prepare_scope_images(frame["images"])
        bank = CurrentFeatureBank(observation_key(frame["images"], frame["body_pose"]), self.encoder_hash, self.native_encode)
        bank.acquire([glance[camera] for camera in CAMERAS], [Geometry.glance(camera) for camera in range(4)])
        return bank

    def predict_session(self, session, *, frames, instruction):
        from starVLA.model.modules import input_context
        if session["pending"] is not None:
            raise RuntimeError("must supply actual action feedback before another policy prediction")
        frame = frames[-1]
        purpose = frame["stage2_purpose"]
        check_episode(session["episode_id"], frame["stage2_scene"], purpose)
        session["purpose"] = purpose
        self.reused_current = 0
        self.visual_calls = []
        self.native_pixel_chunks = []
        self.annotate = bool(frame.get("stage2_annotate", False))
        self.diffusion_seed = frame.get("stage2_diffusion_seed")
        self.verify_reencode = bool(frame.get("stage2_verify_reencode", False))
        self.use_native_features = bool(frame.get("stage2_use_native_features", False))
        self.reencode_errors = []
        if frame.get("stage2_no_history", False):
            session["history"] = self.EpisodeHistory()
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        self.active_bank = self.bank_for(frame)
        language = self.language(instruction)
        actor = self.trainer.actor
        critic = self.trainer.critic
        initial_actor = session["recurrent"].observation(self.active_bank, language).snapshot()
        initial_critic = session["recurrent"].observation(self.active_bank, language, critic=True).snapshot()
        with torch.no_grad():
            value = critic(initial_critic).detach()
        if session["rollout"] and not session["rollout"][-1].terminated:
            session["rollout"][-1].next_value = float(value)
        decisions, trace = [], []
        selector_end = None

        def select(mode, *, candidates, **kwargs):
            nonlocal selector_end
            forced = frame.get("stage2_forced")
            for focus_index in range(2):
                observation = session["recurrent"].observation(self.active_bank, language).snapshot()
                with torch.no_grad():
                    distribution = actor(observation)
                    if forced is not None:
                        action = forced[focus_index] if focus_index < len(forced) else 16
                    elif purpose == "dev" or frame.get("stage2_greedy", False):
                        action = int(distribution.logits.argmax())
                    else:
                        action = int(torch.multinomial(distribution.probs, 1, generator=self.action_generator))
                    mask = actor.action_mask(observation.selected, distribution.logits.device)
                    if not bool(mask[action]) and forced is None:
                        raise ValueError("invalid sampled perception action")
                    log_probability = distribution.log_prob(torch.tensor(action, device="cuda")).detach()
                trace.append(action)
                if forced is None:
                    decisions.append(Decision(observation, action, mask, log_probability))
                if action == 16:
                    break
                if frame.get("stage2_low_detail", False):
                    from PIL import Image
                    candidates[action]["image"] = candidates[action]["image"].resize((64, 64), Image.Resampling.BICUBIC).resize((256, 256), Image.Resampling.BICUBIC)
                self.active_bank.select(action, candidates[action]["image"])
            self.torch.cuda.synchronize()
            selector_end = time.perf_counter()
            return [{**candidates[candidate], "focus_index": index} for index, candidate in enumerate(self.active_bank.selected)], trace

        original_select = input_context.select_focus
        input_context.select_focus = select
        for current in frames:
            current["policy_mode"] = "learned_adaptive"
        try:
            response = super().predict_session(session, frames=frames, instruction=instruction)
            self.torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            perception_ms = ((selector_end or started) - started) * 1000
            bank = self.active_bank
            audit = bank.audit()
            current_count = 4 + len(bank.selected)
            if bank.encoder_images != current_count or self.reused_current != current_count:
                raise AssertionError("current visual features were not encoded exactly once and reused")
            visual_summary = torch.cat([bank.glance().reshape(-1, 2560)] +
                                       [bank.read_focus(candidate).tokens for candidate in bank.selected]).float().mean(0)
            record = {"episode_id": session["episode_id"], "step": session["prediction_index"] - 1,
                      "purpose": purpose, "trace": trace, "bank": audit, "reused_current": self.reused_current,
                      "encoder_calls": self.visual_calls, "context_grids": self.last_context_grids,
                      "perception_ms": perception_ms, "model_ms": elapsed,
                      "navigation_completed_before": session["recurrent"].completed_steps,
                      "history_visual_tokens": response["metadata"]["scope"]["total_visual_tokens"] - (64 + 64 * len(bank.selected)),
                      "annotated": self.annotate, "no_history": bool(frame.get("stage2_no_history", False)),
                      "spatial_adapter_enabled": self.spatial_adapter is not None,
                      "low_detail": bool(frame.get("stage2_low_detail", False)),
                      "diffusion_seed": self.diffusion_seed,
                      "diagnostic_reencode": self.verify_reencode, "reencode_errors": self.reencode_errors,
                      "value": float(value), "forced": frame.get("stage2_forced")}
            session["pending"] = {"actor": initial_actor, "critic": initial_critic, "decisions": decisions,
                                  "value": value, "visual": visual_summary, "record": record,
                                  "glance_summary": bank.glance().float().mean((0, 1)).cpu().tolist(),
                                  "focus_summaries": [bank.read_focus(candidate).tokens.float().mean(0).cpu().tolist()
                                                      for candidate in bank.selected],
                                  "actions": response["actions"].tolist()}
            session["pending"]["before_pose"] = list(frame["body_pose"])
            response["metadata"]["stage2"] = record
            return response
        finally:
            input_context.select_focus = original_select
            self.active_bank = None

    def _seeded_prediction(self, sample, seed):
        return super()._seeded_prediction(sample, seed if self.diffusion_seed is None else int(self.diffusion_seed))

    def complete_step(self, session, *, execution, teacher=None):
        pending = session["pending"]
        if pending is None:
            raise RuntimeError("no pending navigation action")
        feedback = ExecutionFeedback(**execution)
        if not np.array_equal(np.asarray(feedback.predicted_waypoints), np.asarray(pending["actions"])):
            raise ValueError("feedback must refer to the actual predicted action chunk")
        if not np.allclose(feedback.before, pending["before_pose"], rtol=0, atol=1e-5):
            raise ValueError("feedback must start at the observed prediction pose")
        record = pending["record"] | feedback.record()
        record["selected_roi_sequence"] = pending["record"]["bank"]["selected"]
        reward = None
        time_limit = feedback.navigation_done and feedback.termination == "environment_time_limit"
        if session["purpose"] == "train" and pending["decisions"]:
            if teacher is None or bool(teacher["terminal"]) != (feedback.navigation_done and not time_limit):
                raise ValueError("consistent measured training-only teacher is required")
            reward = cost_aware_reward(**teacher, focus_count=len(record["selected_roi_sequence"]),
                                       latency_ms=record["perception_ms"],
                                       token_weight=0.01 if self.regime == "C" else 0,
                                       latency_weight=0.001 if self.regime == "C" else 0)
        vector = torch.tensor(feedback.vector(), device="cuda")
        session["recurrent"].complete(self.trainer.actor, self.trainer.critic, pending["actor"], pending["critic"],
                                      pending["visual"], vector)
        record["actor_state_norm_after_feedback"] = float(session["recurrent"].actor_state.norm())
        record["critic_state_norm_after_feedback"] = float(session["recurrent"].critic_state.norm())
        record["visual_summary"] = pending["visual"].cpu().tolist()
        record["glance_summary"] = pending["glance_summary"]
        record["focus_summaries"] = pending["focus_summaries"]
        if session["purpose"] == "train" and pending["decisions"]:
            transition = Transition(pending["decisions"], pending["critic"], pending["value"], reward,
                                    feedback.navigation_done and not time_limit, time_limit, 0.0, session["episode_id"])
            session["rollout"].append(transition)
            record["reward"] = reward
            record["reward_teacher"] = teacher | {"collision_source": "native step_filter requested-versus-filtered displacement"}
        session["pending"] = None
        session["records"].append(record)
        with (self.output / "trajectories.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        return {"completed_steps": session["recurrent"].completed_steps, "reward": record.get("reward")}

    def finish(self, session, *, bootstrap_frame=None, instruction=None, update=False):
        if session["pending"] is not None:
            raise RuntimeError("cannot finish before actual feedback")
        rollout = session["rollout"]
        if rollout and not rollout[-1].terminated:
            if bootstrap_frame is None or instruction is None:
                raise ValueError("truncation requires an observed next-state bootstrap")
            bank = self.bank_for(bootstrap_frame)
            observation = session["recurrent"].observation(bank, self.language(instruction), critic=True)
            with torch.no_grad():
                rollout[-1].next_value = float(self.trainer.critic(observation))
            rollout[-1].truncated = True
        report = {"transitions": len(rollout), "updated": False}
        if update:
            if session["purpose"] != "train" or not rollout:
                raise PermissionError("PPO update requires train-only real rollout")
            before = torch.cat([parameter.detach().flatten() for parameter in self.trainer.actor.parameters()])
            torch.save(rollout, self.output / f"real_train_rollout_{self.regime}_{session['episode_id']}.pt")
            began = time.perf_counter()
            statistics = self.trainer.update(rollout, epochs=2)
            torch.cuda.synchronize()
            after = torch.cat([parameter.detach().flatten() for parameter in self.trainer.actor.parameters()])
            report.update(updated=True, statistics=statistics, update_seconds=time.perf_counter() - began,
                          actor_parameter_l2_change=float((after - before).norm()),
                          backbone_gradients_none=all(parameter.grad is None for parameter in self.model.parameters()))
            checkpoint = self.output / f"actor_critic_{self.regime}_update_{self.trainer.updates:03d}.pt"
            cursor = {"finished_episode": session["episode_id"], "action_generator": self.action_generator.get_state(),
                      "next_training_episode_index": partition()["training_episode_ids"].index(session["episode_id"]) + 1}
            self.trainer.save(checkpoint, provenance=self.provenance, cursor=cursor)
            report["checkpoint"] = str(checkpoint)
        session["recurrent"].reset()
        session["rollout"] = []
        write_json(self.output / f"finish_{session['purpose']}_{self.regime}_{session['episode_id']}_{len(session['records'])}_{self.trainer.updates}.json", report)
        return report

    def configure_regime(self, regime):
        seed = self.provenance["seed"]
        self.trainer = build_trainer(regime, "cuda", seed)
        self.regime = regime
        self.provenance = {"backbone": self.encoder_hash, "regime": regime, "seed": seed,
                           "partition_sha256": sha256(PARTITION)}
        self.action_generator.manual_seed(seed)
        return {"regime": regime, "reset": True}
