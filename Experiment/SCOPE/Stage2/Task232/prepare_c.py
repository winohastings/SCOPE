from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task23.audit import compare_nested
from Experiment.SCOPE.Stage2.Task22.trajectory import atomic_save
from Experiment.SCOPE.Stage2.data_guard import sha256, write_json


def main():
    frozen = ROOT / "frozen"
    assert not any((frozen / f"fork_{branch}.pt").exists() or (frozen / f"config_{branch}.json").exists() for branch in ("C", "C0"))
    audit = json.loads((frozen / "FORK_AUDIT.json").read_text())
    assert sha256(audit["parent"]) == audit["parent_sha256"]
    original = torch.load(audit["parent"], map_location="cpu", weights_only=False)
    base = torch.load(frozen / "fork_U.pt", map_location="cpu", weights_only=False)
    settings = json.loads((frozen / "config_U.json").read_text())
    records = {}
    for branch, beta in (("C", .01), ("C0", 0.)):
        config = settings | {"task232_branch": branch, "coverage_beta": beta, "continuation_budgets": [],
                             "protocol_sha256": sha256(ROOT / "C_PROTOCOL.md")}
        write_json(frozen / f"config_{branch}.json", config)
        fork = copy.deepcopy(base)
        fork["trainer"]["config"] = config
        fork["provenance"]["task232_config_sha256"] = sha256(frozen / f"config_{branch}.json")
        for key in ("actor", "critic", "old_actor", "old_critic", "actor_optimizer", "critic_optimizer", "python_rng", "numpy_rng", "torch_rng", "cuda_rng", "minibatch_rng"):
            compare_nested(original["trainer"][key], fork["trainer"][key], key)
        assert fork["collected"] == 1033 and fork["trainer"]["transitions_seen"] == 1033
        assert not fork["buffer_boundaries"] and fork["cursor"]["train_episode_index"] == 70
        atomic_save(frozen / f"fork_{branch}.pt", fork)
        records[branch] = {"beta": beta, "config_sha256": sha256(frozen / f"config_{branch}.json"),
                           "checkpoint_sha256": sha256(frozen / f"fork_{branch}.pt")}
    write_json(frozen / "C_FORK_AUDIT.json", {"passed": True, "parent_sha256": audit["parent_sha256"],
               "full_state_bitwise_equal": True, "branches": records, "protocol_sha256": sha256(ROOT / "C_PROTOCOL.md"),
               "source_hashes": {name: sha256(ROOT / name) for name in ("runtime.py", "run.py", "novelty.py", "prepare_c.py")}})
    print(json.dumps(records))


if __name__ == "__main__":
    main()
