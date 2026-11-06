from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))
from Experiment.SCOPE.Stage2.Task22.actor_runtime import ActorPolicy
from Experiment.SCOPE.Stage2.data_guard import write_json


async def serve(policy, port):
    import websockets.asyncio.server
    from deployment.model_server.tools import msgpack_numpy

    active = set()

    async def handle(websocket):
        sessions = {}
        await websocket.send(msgpack_numpy.packb({"protocol": "simplenav.session.v1", "stage2_feedback": True,
                                                 "agm_schema": "agm-task22-v1"}))
        try:
            async for raw in websocket:
                request = msgpack_numpy.unpackb(raw)
                identifier, method = request["session_id"], request["method"]
                payload = request.get("payload", {})
                try:
                    if method == "reset":
                        if identifier in sessions:
                            raise RuntimeError("close the previous session before resetting")
                        sessions[identifier] = policy.new_session(**payload)
                        active.add((id(websocket), identifier))
                        result = None
                    elif method == "predict":
                        result = policy.predict_session(sessions[identifier], **payload)
                    elif method == "feedback":
                        result = policy.complete_step(sessions[identifier], **payload)
                    elif method == "finish":
                        result = policy.finish(sessions[identifier], **payload)
                    elif method == "configure":
                        if active or payload.get("mode") not in ("deterministic", "training-sampling"):
                            raise ValueError("mode changes require no live sessions")
                        policy.actor_mode = payload["mode"]
                        result = {"mode": policy.actor_mode}
                    elif method == "close":
                        if sessions[identifier]["pending"] is not None:
                            raise RuntimeError("pending action feedback cannot be discarded")
                        sessions.pop(identifier)
                        active.discard((id(websocket), identifier))
                        result = None
                    elif method == "audit":
                        result = policy.freeze_audit()
                    else:
                        raise ValueError(f"unsupported method {method}; no RL update endpoint")
                    response = {"ok": True, "data": result}
                except Exception as error:
                    traceback.print_exc()
                    response = {"ok": False, "error": repr(error)}
                await websocket.send(msgpack_numpy.packb(response))
        finally:
            for identifier in sessions:
                active.discard((id(websocket), identifier))

    async with websockets.asyncio.server.serve(handle, "127.0.0.1", port, compression=None, max_size=None):
        write_json(policy.output / "ready.json", {"port": port, "mode": policy.actor_mode})
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10122)
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--mode", choices=("deterministic", "training-sampling"), default="training-sampling")
    parser.add_argument("--actor-checkpoint", type=Path)
    args = parser.parse_args()
    policy = ActorPolicy(output=args.output, seed=args.seed, mode=args.mode, actor_checkpoint=args.actor_checkpoint)
    asyncio.run(serve(policy, args.port))


if __name__ == "__main__":
    main()
