from __future__ import annotations

import sys

import numpy as np

from compile_configs import ROOT, SIMPLENAV, write_json
from policies import MODES

sys.path.insert(0, str(SIMPLENAV))
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def main():
    fixture = np.load(ROOT / "pilot_observation.npz", allow_pickle=False)
    clients = [WebsocketClientPolicy(uri="ws://127.0.0.1:10095", timeout=120) for index in range(2)]
    results = []
    try:
        for group, mode in MODES.items():
            for client in clients:
                client.reset(session_id="isolation", episode_id="9791", seed=42, statistics_key="vln_train_train")
            for step in range(3):
                responses = [client.predict(frames=[{"images": {name: fixture[name] for name in ("front", "left", "right", "rear")},
                                                     "body_pose": fixture["body_pose"], "frame_id": step,
                                                     "timestamp_s": float(step), "policy_mode": mode}],
                                            instruction=str(fixture["instruction"])) for client in clients]
                np.testing.assert_array_equal(responses[0]["actions"], responses[1]["actions"])
                if responses[0]["metadata"]["scope"]["selection_trace"] != responses[1]["metadata"]["scope"]["selection_trace"]:
                    raise ValueError("session cross-contamination of selector")
                results.append({"group": group, "step": step, "max_abs_action_difference": 0.0})
            for client in clients:
                client.close_session("isolation")
        write_json(ROOT / "SESSION_ISOLATION.json", {"passed": True, "two_clients_interleaved": True, "comparisons": results})
    finally:
        for client in clients:
            client.close()


if __name__ == "__main__":
    main()
