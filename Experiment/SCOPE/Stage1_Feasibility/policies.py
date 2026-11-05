from __future__ import annotations

import hashlib

import numpy as np

CYCLE = ((0, 4), (8, 12), (1, 5), (9, 13), (2, 6), (10, 14), (3, 7), (11, 15))
TOKENS = {"G0": 256, "G1": 64, "G2": 192, "G3": 192}
MODES = {"G0": "original", "G1": "glance_only", "G2": "fixed_2", "G3": "random_2"}


def select_pair(mode, *, seed, episode_id, step):
    if step < 0:
        raise ValueError("negative navigation step")
    digest = hashlib.sha256(f"{seed}:{episode_id}:{step}".encode()).digest()
    if mode == "fixed_2":
        pair = list(CYCLE[step % len(CYCLE)])
    elif mode == "random_2":
        generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        first = int(generator.choice(16))
        second = int(generator.choice([candidate for candidate in range(16) if candidate // 4 != first // 4]))
        pair = [first, second]
    elif mode in ("original", "glance_only"):
        pair = []
    else:
        raise ValueError(f"unauthorized mode: {mode}")
    if pair and (len(set(pair)) != 2 or pair[0] // 4 == pair[1] // 4):
        raise ValueError("invalid camera/ROI pair")
    return pair, digest.hex()
