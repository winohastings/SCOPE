from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[3]))

from Experiment.SCOPE.Stage2.Task232.runtime import RepairPolicy
from Experiment.SCOPE.Stage2.Task23.server import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("Task232-only output")
    asyncio.run(serve(RepairPolicy(output=args.output, config=args.config, manifest=args.manifest), args.port))


if __name__ == "__main__":
    main()
