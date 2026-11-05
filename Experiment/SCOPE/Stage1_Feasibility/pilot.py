from __future__ import annotations

import json
import statistics
import sys
import time

import numpy as np

from compile_configs import ROOT, SIMPLENAV, write_json
from policies import MODES

sys.path.insert(0, str(SIMPLENAV))
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def main():
    client = WebsocketClientPolicy(uri="ws://127.0.0.1:10095", timeout=120)
    fixture = np.load(ROOT / "pilot_observation.npz", allow_pickle=False)
    images = {name: fixture[name] for name in ("front", "left", "right", "rear")}
    records = []
    try:
        for group, mode in MODES.items():
            warm = []
            stable = False
            for index in range(200):
                client.reset(session_id="pilot", episode_id="9791", seed=42, statistics_key="vln_train_train")
                begin = time.perf_counter()
                response = client.predict(frames=[{"images": images, "body_pose": fixture["body_pose"],
                                                  "frame_id": 0, "timestamp_s": 0.0, "policy_mode": mode}],
                                          instruction=str(fixture["instruction"]))
                record = {**response["metadata"]["scope"], "group": group,
                          "rpc_wall_ms": (time.perf_counter() - begin) * 1000}
                client.close_session("pilot")
                if not stable:
                    warm.append(record["model_wall_ms"])
                    if len(warm) >= 20 and len(warm) % 10 == 0:
                        drift = abs(statistics.median(warm[-10:]) / statistics.median(warm[-20:-10]) - 1)
                        stable = drift <= 0.05
                    record["measurement_phase"] = "warmup"
                else:
                    record["measurement_phase"] = "timed"
                records.append(record)
                if stable and sum(row["group"] == group and row["measurement_phase"] == "timed" for row in records) == 100:
                    break
                if not stable and len(warm) == 100:
                    break
            print(json.dumps({"group": group, "warmup": len(warm), "stable": stable}), flush=True)
            write_json(ROOT / "PILOT.json", records)
            if not stable:
                raise RuntimeError("platform_unstable; do not start formal evaluation")
    finally:
        client.close()


if __name__ == "__main__":
    main()
