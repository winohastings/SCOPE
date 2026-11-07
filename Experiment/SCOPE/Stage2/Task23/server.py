from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from Experiment.SCOPE.Stage2.Task23.runtime import PPOPolicy
from Experiment.SCOPE.Stage2.data_guard import write_json


async def serve(policy, port):
    import websockets.asyncio.server
    from deployment.model_server.tools import msgpack_numpy

    active = set()

    async def handle(websocket):
        sessions = {}
        await websocket.send(msgpack_numpy.packb({"protocol": "simplenav.session.v1", "training": "PPO-only"}))
        try:
            async for raw in websocket:
                request = msgpack_numpy.unpackb(raw)
                method, identifier = request["method"], request["session_id"]
                payload = request.get("payload", {})
                try:
                    if method == "reset":
                        if active:
                            raise RuntimeError("single on-policy environment; close previous episode first")
                        sessions[identifier] = policy.new_session(**payload)
                        active.add((id(websocket), identifier))
                        result = None
                    elif method == "predict":
                        result = policy.predict_session(sessions[identifier], **payload)
                    elif method == "feedback":
                        result = policy.complete_step(sessions[identifier], **payload)
                    elif method == "finish":
                        result = policy.finish(sessions[identifier], **payload)
                    elif method == "close":
                        if sessions[identifier]["pending"] is not None:
                            raise RuntimeError("cannot discard incomplete feedback")
                        sessions.pop(identifier)
                        active.discard((id(websocket), identifier))
                        result = None
                    elif method == "status":
                        result = policy.status()
                    elif method == "audit":
                        result = policy.freeze_audit()
                    else:
                        if active:
                            raise RuntimeError("control/update/checkpoint only at episode boundaries")
                        if method == "configure":
                            split = payload["split"]
                            if split not in ("train", "val_seen") or (split == "val_seen" and policy.buffer):
                                raise ValueError("forbidden split or nonempty on-policy buffer")
                            policy.split = split
                            policy.actor_mode = "training-sampling" if split == "train" else "deterministic"
                            result = {"split": split, "mode": policy.actor_mode}
                        elif method in ("update", "save", "restore", "validation_result"):
                            result = getattr(policy, method)(**payload)
                        else:
                            raise ValueError(method)
                    response = {"ok": True, "data": result}
                except Exception as error:
                    traceback.print_exc()
                    response = {"ok": False, "error": repr(error)}
                await websocket.send(msgpack_numpy.packb(response))
        finally:
            for identifier in sessions:
                active.discard((id(websocket), identifier))

    async with websockets.asyncio.server.serve(handle, "127.0.0.1", port, compression=None, max_size=None,
                                               ping_interval=None):
        write_json(policy.output / "ready.json", {"port": port})
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10123)
    args = parser.parse_args()
    policy = PPOPolicy(output=args.output, config=args.config, manifest=args.manifest)
    asyncio.run(serve(policy, args.port))


if __name__ == "__main__":
    main()
