from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch

from data_guard import write_json
from policy import ROIActor


def run(rollout_path, output):
    torch.set_num_threads(1)
    torch.manual_seed(777)
    rollout = torch.load(rollout_path, map_location="cpu", weights_only=False)
    if any(not transition.episode_id for transition in rollout):
        raise ValueError("real train provenance required")
    decisions = [transition.decisions[0] for transition in rollout]
    actor = ROIActor()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=1e-3)

    def loss():
        return torch.stack([-actor(decision.observation, decision.mask).log_prob(torch.tensor(decision.action))
                            for decision in decisions]).mean()

    initial = float(loss().detach())
    started = time.perf_counter()
    for iteration in range(30):
        optimizer.zero_grad(set_to_none=True)
        current = loss()
        current.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
        optimizer.step()
    final = float(loss().detach())
    report = {"kind": "train-only real-observation overfit sanity; not navigation or on-policy PPO",
              "source": str(rollout_path), "samples": len(decisions), "iterations": 30,
              "initial_nll": initial, "final_nll": final, "seconds": time.perf_counter() - started,
              "passed": final < initial * 0.5}
    write_json(output, report)
    if not report["passed"]:
        raise AssertionError(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.rollout, args.output)
