from __future__ import annotations

import math

import numpy as np


def coverage_novelty(decisions, beta):
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative")
    covered = np.zeros((4, 512, 512), dtype=bool)
    rows = []
    for decision in decisions:
        if decision["action"]["stop"]:
            continue
        camera = int(decision["action"]["camera"])
        left, top, right, bottom = map(int, decision["roi"])
        if not (0 <= camera < 4 and 0 <= left < right <= 512 and 0 <= top < bottom <= 512):
            raise ValueError("invalid ROI")
        if right - left != 192 or bottom - top != 192:
            raise ValueError("coverage reward requires a 192x192 ROI")
        area = covered[camera, top:bottom, left:right]
        new_pixels = int(area.size - area.sum())
        fraction = new_pixels / (192 ** 2)
        rows.append({"camera": camera, "roi": [left, top, right, bottom], "new_pixels": new_pixels,
                     "novelty": fraction, "reward": beta * fraction})
        area[:] = True
    return {"beta": beta, "focus": rows, "new_pixels": int(covered.sum()),
            "reward": sum(row["reward"] for row in rows)}


def augment_transitions(transitions, beta):
    for transition in transitions:
        novelty = coverage_novelty(transition.trace["decisions"], beta)
        transition.trace["task232_coverage_novelty"] = novelty
        transition.reward += novelty["reward"]
        transition.trace["navigation_reward"] = transition.reward
    return transitions
