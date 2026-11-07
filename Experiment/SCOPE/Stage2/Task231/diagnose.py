from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import torch
from torch.distributions import kl_divergence

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task22.actor import ActorState, SpatialQueryActor
from Experiment.SCOPE.Stage2.Task22.trajectory import replay_decisions
from Experiment.SCOPE.Stage2.Task23.learning import NavigationTransition, gae, navigation_reward, sequence_objective

CHECKPOINTS = ((0, 0), (14, 1033), (28, 2054), (43, 3100), (56, 4096))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def interval(values, seed=231):
    values = np.asarray(values, dtype=float)
    samples = np.random.default_rng(seed).choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(), "episodes": len(values)}


def paired(first, second):
    assert [row["episode_uid"] for row in first] == [row["episode_uid"] for row in second]
    lost = sum(before["SR"] == 1 and after["SR"] == 0 for before, after in zip(first, second))
    gained = sum(before["SR"] == 0 and after["SR"] == 1 for before, after in zip(first, second))
    discordant = lost + gained
    probability = min(1., 2 * sum(math.comb(discordant, count) for count in range(min(lost, gained) + 1)) / 2 ** discordant)
    return {"success_to_failure": lost, "failure_to_success": gained, "exact_mcnemar_p": probability,
            "delta_SR": interval([after["SR"] - before["SR"] for before, after in zip(first, second)]),
            "delta_SPL": interval([after["SPL"] - before["SPL"] for before, after in zip(first, second)]),
            "delta_nDTW": interval([after["nDTW"] - before["nDTW"] for before, after in zip(first, second)])}


def verify(source, output):
    manifest = json.loads((source.parent / "ARTIFACT_MANIFEST.json").read_text())
    expected = {record["path"]: record for record in manifest["files"]}
    paths = [source / "DATA_MANIFEST.json", source / "config.json"]
    paths += [source / f"runtime/checkpoints/update_{update:04d}.pt" for update, _ in CHECKPOINTS]
    paths += sorted((source / "episodes").glob("val_*/result.json"))
    records = []
    for path in paths:
        relative = str(path.relative_to(ROOT.parents[3]))
        reference = expected.get(relative)
        actual = digest(path)
        records.append({"path": relative, "sha256": actual, "expected_sha256": reference["sha256"] if reference else None,
                        "verified": bool(reference and actual == reference["sha256"])})
    if not all(record["verified"] for record in records):
        write_json(output / "provenance.json", records)
        raise AssertionError("original artifact SHA mismatch or missing reference")
    versions = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda}
    for package in ("numpy", "transformers", "websockets", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    write_json(output / "provenance.json", {"artifacts": records, "model_environment": versions,
                                           "role_document": "agents/exec-agent.md missing; user explicitly assigned Exec role",
                                           "source_manifest_sha256": digest(source.parent / "ARTIFACT_MANIFEST.json")})
    return expected


def history(source, output):
    rounds = []
    rows = []
    manifest = json.loads((source / "DATA_MANIFEST.json").read_text())
    summaries = []
    for round_index, (update, transitions) in enumerate(CHECKPOINTS):
        current = [json.loads(path.read_text()) for path in sorted((source / "episodes").glob(f"val_{round_index:03d}_*/result.json"))]
        assert len(current) == 32
        assert [row["episode_uid"] for row in current] == [record["episode_uid"] for record in manifest["val_seen"]]
        rounds.append(current)
        for row in current:
            summary = row["training"]["episode_summary"]
            collision_count = 0
            minimum_distance = summary["start_geodesic_distance"]
            for path in row["training"]["trace_paths"]:
                trace = load(path)
                teacher = trace["teacher_for_reward_only"]
                minimum_distance = min(minimum_distance, teacher["distance_after"])
                collision_count += int(teacher["collision"])
            rows.append({"transitions": transitions, "update": update, "episode_uid": row["episode_uid"], "scene": row["scene_id"],
                         "SR": row["SR"], "SPL": row["SPL"], "nDTW": row["nDTW"], "final_distance": row["final_distance"],
                         "path_length": row["path_length"], "steps": row["steps"], "oracle_success": row["oracle_success"],
                         "minimum_chunk_boundary_distance": minimum_distance, "collision_chunks": collision_count,
                         "terminated": summary["terminated"], "truncated": summary["truncated"],
                         "termination_reason": row["termination_reason"], "focus_mean": row["focus_mean"], "reward_sum": row["reward_sum"]})
        summaries.append({"transitions": transitions, "SR": interval([row["SR"] for row in current]),
                          "OSR": float(np.mean([row["oracle_success"] for row in current])),
                          "oracle_but_failed": sum(row["oracle_success"] and not row["SR"] for row in current),
                          "truncated": sum(row["training"]["episode_summary"]["truncated"] for row in current)})
    write_csv(output / "paired_episodes.csv", rows)
    comparisons = {f"{CHECKPOINTS[first][1]}->{CHECKPOINTS[second][1]}": paired(rounds[first], rounds[second])
                   for first, second in itertools.combinations(range(5), 2)}
    write_json(output / "paired_statistics.json", {"rounds": summaries, "comparisons": comparisons,
                                                  "bootstrap": "20000 episode-paired percentile draws, seed231, exploratory unadjusted comparisons"})
    return rounds


def probe(source, output, rounds, device, states):
    train = [json.loads(path.read_text()) for path in sorted((source / "episodes").glob("train_*/result.json"))]
    selected = []
    for split, episodes in (("train", train), ("val_seen", rounds[0])):
        paths = []
        offset = 0
        while len(paths) < states:
            previous = len(paths)
            for row in episodes:
                if offset < len(row["training"]["trace_paths"]):
                    paths.append(row["training"]["trace_paths"][offset])
                    if len(paths) == states:
                        break
            if previous == len(paths):
                raise ValueError("not enough real states")
            offset += 1
        for path in paths:
            trace = load(path)
            state = ActorState.unpack(trace["decisions"][0]["state"], device)
            assert state.count == 0 and set(state.packed()) == {"glance", "fused", "language", "coverage", "history", "count"}
            selected.append((split, path, trace["scene_id"], state))
    actor = SpatialQueryActor().to(device).eval()
    initial = SpatialQueryActor().to(device).eval()
    initial.load_state_dict(load(source / "runtime/snapshots/actor_0000.pt")["actor"])
    rows = []
    summaries = []
    with torch.no_grad():
        references = [initial(state) for _, _, _, state in selected]
        for update, transitions in CHECKPOINTS:
            checkpoint = torch.load(source / f"runtime/checkpoints/update_{update:04d}.pt", map_location="cpu", weights_only=False)
            actor.load_state_dict(checkpoint["trainer"]["actor"])
            for (split, path, scene, state), reference in zip(selected, references):
                distribution = actor(state)
                probability = float(distribution.stop.probs)
                row = {"transitions": transitions, "split": split, "state_path": path, "scene": scene, "p_stop": probability,
                       "stop_ge_045": int(probability >= .45), "stop_ge_050": int(probability >= .5),
                       "stationary_p_expected_focus_proxy": sum((1 - probability) ** count for count in range(1, 5)),
                       "first_decision_deterministic_focus_proxy": 0 if probability >= .5 else 4,
                       "camera_entropy": float(distribution.camera.entropy()), "camera_top1": int(distribution.camera.probs.argmax()),
                       "camera_top1_probability": float(distribution.camera.probs.max()),
                       "stop_KL_initial_to_current": float(kl_divergence(reference.stop, distribution.stop)),
                       "camera_KL_initial_to_current": float(kl_divergence(reference.camera, distribution.camera)),
                       "position_KL_initial_to_current": float((reference.camera.probs * kl_divergence(reference.position, distribution.position).sum(-1)).sum())}
                for camera in range(4):
                    row[f"camera_p_{camera}"] = float(distribution.camera.probs[camera])
                    for axis in range(2):
                        row[f"mean_{camera}_{axis}"] = float(distribution.position.loc[camera, axis])
                        row[f"sigma_{camera}_{axis}"] = float(distribution.position.scale[camera, axis])
                rows.append(row)
            for split in ("train", "val_seen"):
                group = [row for row in rows if row["transitions"] == transitions and row["split"] == split]
                summaries.append({"transitions": transitions, "split": split, "states": len(group),
                                  "p_stop_P10_P50_P90": np.quantile([row["p_stop"] for row in group], [.1, .5, .9]).tolist(),
                                  "stop_ge_045": float(np.mean([row["stop_ge_045"] for row in group])),
                                  "stop_ge_050": float(np.mean([row["stop_ge_050"] for row in group])),
                                  "camera_top1_fractions": {str(camera): sum(row["camera_top1"] == camera for row in group) / len(group) for camera in range(4)},
                                  "camera_entropy": float(np.mean([row["camera_entropy"] for row in group]))})
    write_csv(output / "checkpoint_policy_probe.csv", rows)
    write_json(output / "policy_probe_summary.json", {"summaries": summaries,
        "focus_count_limitation": "Actor-only pre-Focus states cannot determine future fused observations. Count columns are explicitly stationary-p/first-decision proxies, not actual sequential expected counts; actual deterministic counts come from rollouts.",
        "inputs": "Only detached ActorState whitelist; no reward, GT, teacher, scene or success inputs."})
    print("policy probe complete", flush=True)


def numerical(source, output, expected, device):
    config = json.loads((source / "config.json").read_text())
    actor = SpatialQueryActor().to(device).eval()
    old_actor = SpatialQueryActor().to(device).eval()
    rows, gradients, fusion = [], [], []
    reward_components = Counter()
    errors = Counter()
    audits = []
    for update_path in sorted((source / "runtime/updates").glob("update_*.json")):
        report = json.loads(update_path.read_text())
        update = report["update"]
        old_actor.load_state_dict(load(source / f"runtime/snapshots/actor_{update-1:04d}.pt")["actor"])
        actor.load_state_dict(load(source / f"runtime/snapshots/actor_{update:04d}.pt")["actor"])
        transitions = []
        trace_paths = []
        for boundary_path in report["boundary_paths"]:
            boundary = load(boundary_path)
            for trace_path in boundary["report"]["trace_paths"]:
                absolute = Path(trace_path).resolve()
                relative = str(absolute.relative_to(ROOT.parents[3]))
                if digest(absolute) != expected[relative]["sha256"]:
                    raise AssertionError(f"trace hash mismatch: {trace_path}")
                trace = load(absolute)
                teacher = trace["teacher_for_reward_only"]
                reward = navigation_reward(teacher, trace["focus_count"], terminated=trace["terminated"], truncated=trace["truncated"])
                errors["reward_max_abs"] = max(errors["reward_max_abs"], abs(reward - trace["navigation_reward"]))
                reward_components["progress"] += .2 * np.clip((teacher["distance_before"] - teacher["distance_after"]) / 3, -1, 1)
                reward_components["focus_cost"] -= .01 * trace["focus_count"]
                reward_components["step_cost"] -= .01
                reward_components["terminal"] += (2 if teacher["success"] else -.5) if trace["terminated"] else 0
                transitions.append(NavigationTransition(trace, trace["old_value"], trace["old_next_value"], reward,
                                                        trace["transition_id"].rsplit(":", 1)[0]))
                trace_paths.append(trace_path)
        advantages, returns = gae(transitions, config["gamma"], config["gae_lambda"])
        normalized = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        targets = load(source / f"runtime/updates/targets_{update:04d}.pt")
        errors["return_max_abs"] = max(errors["return_max_abs"], float((returns - targets["returns"]).abs().max()))
        errors["normalized_advantage_max_abs"] = max(errors["normalized_advantage_max_abs"], float((normalized - targets["advantages"]).abs().max()))
        selected_counts = set()
        all_ratios = []
        all_logratios = []
        for index, (transition, path) in enumerate(zip(transitions, trace_paths)):
            trace = transition.trace
            decisions = replay_decisions(trace, device)
            with torch.no_grad():
                for decision in decisions:
                    old_probability, _ = old_actor.evaluate(decision.observation, decision.action)
                    errors["old_logprob_max_abs"] = max(errors["old_logprob_max_abs"], abs(float(old_probability - decision.old_log_probability)))
                loss, ratios, logratios, _ = sequence_objective(actor, decisions, normalized[index].to(device), config)
            all_ratios.extend(ratios.cpu().tolist())
            all_logratios.extend(logratios.cpu().tolist())
            rows.append({"update": update, "trace_path": path, "focus_count": trace["focus_count"], "decision_count": len(decisions),
                         "advantage": float(advantages[index]), "normalized_advantage": float(normalized[index]), "return": float(returns[index]),
                         "old_value": transition.old_value, "reward": transition.reward, "sequence_loss": float(loss),
                         "ratio_mean": float(ratios.mean()), "approximate_KL": float(((ratios - 1) - logratios).mean()),
                         "clip_fraction": float(((ratios - 1).abs() > config["clip"]).float().mean()),
                         "progress": trace["teacher_for_reward_only"]["distance_before"] - trace["teacher_for_reward_only"]["distance_after"]})
            if trace["focus_count"] not in selected_counts:
                selected_counts.add(trace["focus_count"])
                actor.zero_grad(set_to_none=True)
                loss, _, _, _ = sequence_objective(actor, decisions, normalized[index].to(device), config)
                loss.backward()
                gradient = {"update": update, "trace_path": path, "focus_count": trace["focus_count"], "decision_count": len(decisions),
                            "advantage": float(normalized[index]), "sequence_loss": float(loss.detach())}
                for name, module in (("stop", actor.stop_head), ("camera", actor.camera_head), ("position", actor.position_head)):
                    gradient[f"{name}_gradient_norm"] = math.sqrt(sum(float(parameter.grad.square().sum()) for parameter in module.parameters() if parameter.grad is not None))
                gradient["log_std_gradient_norm"] = float(actor.log_std.grad.norm()) if actor.log_std.grad is not None else 0.
                gradients.append(gradient)
            final = trace["final_state"]
            uncovered = final["coverage"].squeeze(-1) == 0
            delta = (final["fused"].float() - final["glance"].float()).square().mean(-1)
            uncovered_error = float((final["fused"][uncovered].float() - final["glance"][uncovered].float()).abs().max()) if uncovered.any() else 0.
            errors["uncovered_token_max_abs"] = max(errors["uncovered_token_max_abs"], uncovered_error)
            fusion.append({"scene": trace["scene_id"], "focus_count": trace["focus_count"], "coverage_mean": float(final["coverage"].mean()),
                           "changed_token_mse": float(delta.mean()), "uncovered_max_abs": uncovered_error,
                           "progress": rows[-1]["progress"], "collision": trace["teacher_for_reward_only"]["collision"]})
        ratio_array, logratio_array = np.asarray(all_ratios), np.asarray(all_logratios)
        audits.append({"update": update, "transitions": len(transitions), "post_update_approx_KL": float(np.mean(ratio_array - 1 - logratio_array)),
                       "post_update_clip_fraction": float(np.mean(abs(ratio_array - 1) > config["clip"])),
                       "logged_minibatch_KL_max": max(row["approximate_kl"] for row in report["minibatches"]),
                       "logged_explained_variance_in_sample": report["explained_variance"], "kl_early_stop": report["kl_early_stop"]})
        print(f"numerical update {update}: {len(transitions)} transitions", flush=True)
    write_csv(output / "numerical_transitions.csv", rows)
    write_csv(output / "head_gradients.csv", gradients)
    write_csv(output / "fusion_audit.csv", fusion)
    groups = []
    for count in range(5):
        group = [row for row in rows if row["focus_count"] == count]
        groups.append({"focus_count": count, "samples": len(group), "advantage_mean": float(np.mean([row["advantage"] for row in group])),
                       "advantage_variance": float(np.var([row["advantage"] for row in group])),
                       "sequence_loss_sum": sum(row["sequence_loss"] for row in group),
                       "absolute_sequence_loss_sum": sum(abs(row["sequence_loss"]) for row in group)})
    result = {"transitions": len(rows), "errors": dict(errors), "reward_components": dict(reward_components), "focus_groups": groups,
              "updates": audits, "gradient_sampling": "first observed sample per Focus count per update; full head modules, unclipped post-update gradients; not optimizer-time causal attribution",
              "limitations": ["logged EV is in-sample, not held-out", "teacher features were not retained; cross-scene teacher-MSE requires new encoding",
                              "token-change/progress association is observational, not causal action improvement"],
              "feature_change_progress_correlation": float(np.corrcoef([row["changed_token_mse"] for row in fusion], [row["progress"] for row in fusion])[0, 1])}
    write_json(output / "numerical_audit.json", result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent / "Task23/artifacts/ppo4096")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/offline")
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--states", type=int, default=128)
    args = parser.parse_args()
    args.source, args.output = args.source.resolve(), args.output.resolve()
    if not args.output.is_relative_to(ROOT) or args.states < 64:
        parser.error("Task231-only outputs and at least 64 states per split required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    expected = verify(args.source, args.output)
    rounds = history(args.source, args.output)
    probe(args.source, args.output, rounds, args.device, args.states)
    numerical(args.source, args.output, expected, args.device)
    write_json(args.output / "COMPLETE.json", {"status": "offline diagnostics complete", "training_updates": 0})


if __name__ == "__main__":
    main()
