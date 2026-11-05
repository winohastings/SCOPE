from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/data/czh/Third_Party/baselines/SimpleNav")
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--observations")
    args = parser.parse_args()
    generator = np.random.default_rng(42)
    raw = {camera: generator.integers(0, 256, (512, 512, 3), dtype=np.uint8)
           for camera in ("front", "left", "right", "rear")}
    instruction = "Walk forward and stop at the door."
    pose = np.zeros(4, dtype=np.float32)
    if args.observations:
        with np.load(args.observations, allow_pickle=False) as observation:
            raw = {camera: observation[camera] for camera in raw}
            instruction = str(observation["instruction"])
            pose = observation["body_pose"]
    compatible = {camera: np.asarray(Image.fromarray(image).resize((256, 256), Image.Resampling.BICUBIC))
                  for camera, image in raw.items()}
    clients = [WebsocketClientPolicy(uri=f"ws://127.0.0.1:{port}") for port in (10093, 10094)]
    report = {"fixture": args.observations or "seeded synthetic fixed RGB observation", "comparisons": []}
    try:
        for client in clients:
            client.reset(session_id="scope-fixed", episode_id="fixed", seed=42, statistics_key="vln_train_train")
        for step in range(2):
            responses = []
            for client, images in zip(clients, (compatible, raw)):
                responses.append(client.predict(frames=[{"images": images, "body_pose": pose,
                                                          "frame_id": step, "timestamp_s": float(step), "policy_mode": "original"}],
                                                instruction=instruction))
            delta = float(np.abs(responses[0]["actions"] - responses[1]["actions"]).max())
            np.testing.assert_allclose(responses[0]["actions"], responses[1]["actions"], atol=1e-5, rtol=1e-5)
            report["comparisons"].append({"step": step, "max_abs_difference": delta,
                                           "shape": list(responses[0]["actions"].shape)})
        client = clients[1]
        for mode in ("glance_only", "fixed_2", "random_2", "learned_2", "learned_adaptive"):
            client.reset(session_id="scope-fixed", episode_id="fixed", seed=42, statistics_key="vln_train_train")
            for step in range(2):
                response = client.predict(frames=[{"images": raw, "body_pose": pose,
                                                    "frame_id": step, "timestamp_s": float(step), "policy_mode": mode}],
                                          instruction=instruction)
                report.setdefault("modes", []).append(response["metadata"]["scope"])
        Path(args.output).write_text(json.dumps(report, indent=2))
    finally:
        for client in clients:
            client.close_session("scope-fixed")
            client.close()


if __name__ == "__main__":
    main()
