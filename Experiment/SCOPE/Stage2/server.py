from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import traceback

from runtime import Stage2Policy


async def serve(policy, port):
    import websockets.asyncio.server
    from deployment.model_server.tools import msgpack_numpy

    async def handle(websocket):
        sessions = {}
        await websocket.send(msgpack_numpy.packb({"protocol": "simplenav.session.v1", "stage2_feedback": True}))
        async for raw in websocket:
            request = msgpack_numpy.unpackb(raw)
            identifier, method = request["session_id"], request["method"]
            payload = request.get("payload", {})
            try:
                if method == "reset":
                    sessions[identifier] = policy.new_session(**payload)
                    result = None
                elif method == "predict":
                    result = policy.predict_session(sessions[identifier], **payload)
                elif method == "feedback":
                    result = policy.complete_step(sessions[identifier], **payload)
                elif method == "finish":
                    result = policy.finish(sessions[identifier], **payload)
                elif method == "restore":
                    if sessions:
                        raise RuntimeError("restore only with no active episodes")
                    cursor = policy.trainer.restore(payload["checkpoint"], provenance=policy.provenance)
                    policy.action_generator.set_state(cursor.pop("action_generator").cpu())
                    result = cursor
                elif method == "configure":
                    if sessions:
                        raise RuntimeError("change training regime only with no active episodes")
                    result = policy.configure_regime(payload["regime"])
                elif method == "close":
                    sessions.pop(identifier, None)
                    result = None
                else:
                    raise ValueError(method)
                response = {"ok": True, "data": result}
            except Exception as error:
                traceback.print_exc()
                response = {"ok": False, "error": repr(error)}
            await websocket.send(msgpack_numpy.packb(response))

    async with websockets.asyncio.server.serve(handle, "127.0.0.1", port, compression=None, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=10096)
    parser.add_argument("--regime", choices=("A", "B", "C"), default="C")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--spatial-adapter", type=Path)
    args = parser.parse_args()
    if (args.output / "model_contract.json").exists():
        raise FileExistsError("use a new output directory; do not overwrite a previous run")
    policy = Stage2Policy(output=args.output, regime=args.regime, seed=args.seed, spatial_adapter=args.spatial_adapter)
    asyncio.run(serve(policy, args.port))
