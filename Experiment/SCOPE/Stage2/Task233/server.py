from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task21.runtime import FusionPolicy
from Experiment.SCOPE.Stage2.Task22.actor import ActorState, LatentAction, SpatialQueryActor
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save
from Experiment.SCOPE.Stage2.Task231.server import fingerprint
from Experiment.SCOPE.Stage2.runtime import Stage2Policy
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json
from Experiment.SCOPE.Stage2.Task233.teacher_candidates import geometry


def weight_fingerprint(module):
    return fingerprint({name: {"shape": list(value.shape), "values": value.reshape(-1)}
                        for name, value in module.state_dict().items()})


class TeacherPolicy(FusionPolicy):
    language = Stage2Policy.language

    def __init__(self, output, checkpoint, manifest):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        adapter = ROOT.parent / "Task21/artifacts/release/adapter.pt"
        super().__init__(output=output, checkpoint=adapter)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload["provenance"]["backbone_sha256"] != self.backbone_hash or payload["provenance"]["adapter_sha256"] != sha256(adapter):
            raise ValueError("checkpoint provenance mismatch")
        self.actor = SpatialQueryActor().cuda().eval().requires_grad_(False)
        self.actor.load_state_dict(payload["trainer"]["old_actor"])
        self.allowed = {record["source_episode_id"]: record for record in json.loads(Path(manifest).read_text())["train"]}
        self.versions = {name: parameter._version for name, parameter in self.model.named_parameters()}
        self.adapter_hash = weight_fingerprint(self.fusion)
        self.actor_hash = weight_fingerprint(self.actor)
        self.calls = 0
        self.focus_calls = 0
        self.language_cache = {}
        write_json(self.output / "provenance.json", {"backbone_sha256": self.backbone_hash,
                   "checkpoint_sha256": sha256(checkpoint), "adapter_sha256": sha256(adapter),
                   "manifest_sha256": sha256(manifest), "optimizer_created": False})

    @torch.no_grad()
    def predict_session(self, session, *, frames, instruction):
        expected = self.allowed.get(str(session["episode_id"]))
        if expected is None or expected["instruction"].strip() != instruction.strip():
            raise PermissionError("train-only teacher episodes")
        if any(frame.get("stage2_purpose") != "train" or frame.get("stage2_scene") != expected["scene_id"] for frame in frames):
            raise PermissionError("split/scene mismatch")
        request = frames[-1]["task233"]
        if set(request) != {"prefix", "candidate", "learned", "anchor_id", "save_state"}:
            raise ValueError("teacher GT/reward must not enter policy inputs")
        prefix = request["prefix"]
        if len(prefix) > 3:
            raise ValueError("illegal microstep")
        if instruction not in self.language_cache:
            self.language_cache[instruction] = self.language(instruction)
        language = self.language_cache[instruction]
        diagnostic = {}

        def controller(perception):
            selected = []
            conditionals = []
            for index in range(len(prefix) + 1):
                state = ActorState(perception.glance, language, perception.current, perception.coverage,
                                   perception.history, len(perception.selected))
                state.validate()
                distribution = self.actor(state)
                conditionals.append({"j": state.count, "camera": distribution.camera.probs.cpu().tolist(),
                                     "p_stop": float(distribution.stop.probs)})
                if index == len(prefix):
                    diagnostic["anchor_state_sha256"] = fingerprint(state.packed())
                    diagnostic["conditionals"] = conditionals
                    if request["save_state"]:
                        path = self.output / "states" / (request["anchor_id"] + ".pt")
                        if path.exists():
                            raise FileExistsError(path)
                        atomic_save(path, {"state": state.packed(), "data_kind": "teacher_candidate",
                                           "ppo_eligible": False, "prefix": prefix})
                        diagnostic["state_path"] = str(path)
                    if request["learned"]:
                        camera = int(distribution.camera.probs.argmax())
                        record = {"camera": camera, "latent": distribution.position.loc[camera].cpu().tolist()}
                    else:
                        record = request["candidate"]
                    if record is None:
                        break
                else:
                    record = prefix[index]
                latent = torch.tensor(record["latent"], device="cuda", dtype=torch.float32)
                choice = LatentAction(False, record["camera"], latent)
                choice.validate()
                perception.acquire(choice.focus())
                selected.append({"camera": choice.camera, "latent": latent.cpu().tolist(), "roi": list(choice.focus().roi)})
            diagnostic.update(selected=selected, focus_count=len(selected), **geometry(selected))

        began = time.perf_counter()
        response = super().predict_session(session, frames=frames, instruction=instruction, controller=controller)
        if tuple(self.last_current.shape) != (4, 16, 2560) or response["metadata"]["scope"]["current_visual_tokens"] != 64:
            raise ValueError("64 current visual tokens invariant violated")
        self.calls += 1
        self.focus_calls += diagnostic["focus_count"]
        diagnostic.update(context_sha256=fingerprint(self.last_blocks), context_grids=self.last_grids,
                          current_tokens=64, wall_seconds=time.perf_counter() - began,
                          data_kind="teacher_candidate", ppo_eligible=False)
        response["metadata"]["task233"] = diagnostic
        return response

    def audit(self):
        checks = {"backbone_frozen": all(not parameter.requires_grad and parameter.grad is None and parameter._version == self.versions[name]
                                        for name, parameter in self.model.named_parameters()),
                  "adapter_unchanged": weight_fingerprint(self.fusion) == self.adapter_hash,
                  "actor_unchanged": weight_fingerprint(self.actor) == self.actor_hash}
        if not all(checks.values()):
            raise AssertionError(checks)
        return {"checks": checks, "model_predictions": self.calls, "focus_encoder_calls": self.focus_calls,
                "optimizer_updates": 0, "critic_training_samples": 0}


async def serve(policy, port):
    import websockets.asyncio.server
    from deployment.model_server.tools import msgpack_numpy

    async def handle(websocket):
        sessions = {}
        await websocket.send(msgpack_numpy.packb({"protocol": "simplenav.session.v1", "mode": "teacher-audit-only"}))
        async for raw in websocket:
            request = msgpack_numpy.unpackb(raw)
            method, identifier = request["method"], request["session_id"]
            payload = request.get("payload", {})
            try:
                if method == "reset":
                    if sessions:
                        raise RuntimeError("close previous candidate session")
                    sessions[identifier] = policy.new_session(**payload)
                    result = None
                elif method == "predict":
                    result = policy.predict_session(sessions[identifier], **payload)
                elif method == "close":
                    del sessions[identifier]
                    result = None
                elif method == "audit":
                    result = policy.audit()
                else:
                    raise PermissionError("Gate0 forbids optimizer, feedback, group and training RPC")
                response = {"ok": True, "data": result}
            except Exception as error:
                traceback.print_exc()
                response = {"ok": False, "error": repr(error)}
            await websocket.send(msgpack_numpy.packb(response))

    async with websockets.asyncio.server.serve(handle, "127.0.0.1", port, compression=None, max_size=None, ping_interval=None):
        write_json(policy.output / "ready.json", {"port": port})
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    policy = TeacherPolicy(args.output, args.checkpoint, args.manifest)
    asyncio.run(serve(policy, args.port))


if __name__ == "__main__":
    main()
