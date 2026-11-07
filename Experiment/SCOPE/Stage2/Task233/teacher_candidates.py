from __future__ import annotations

import math
import numpy as np


def action(camera, latent):
    values = np.asarray(latent, dtype=np.float64)
    if type(camera) is not int or camera not in range(4) or values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("invalid camera/continuous latent")
    return {"camera": camera, "latent": values.tolist()}


def candidates(seed):
    generator = np.random.default_rng(seed)
    result = [action(camera, [0., 0.]) for camera in range(4)]
    for camera in range(4):
        normalized = generator.uniform(.1, .9, size=2)
        result.append(action(camera, np.log(normalized / (1 - normalized))))
    return result


def random_action(seed):
    generator = np.random.default_rng(seed)
    normalized = generator.uniform(1e-6, 1 - 1e-6, size=2)
    return action(int(generator.integers(4)), np.log(normalized / (1 - normalized)))


def roi(record):
    camera = record["camera"]
    validated = action(camera, record["latent"])
    center = [96 + 320 / (1 + math.exp(-value)) for value in validated["latent"]]
    start = [min(320, max(0, math.floor(value - 96 + .5))) for value in center]
    return [start[0], start[1], start[0] + 192, start[1] + 192]


def geometry(records):
    masks = np.zeros((4, 512, 512), dtype=bool)
    repeated = 0
    previous = []
    for record in records:
        bounds = roi(record)
        for old in previous:
            if old["camera"] != record["camera"]:
                continue
            other = roi(old)
            area = max(0, min(bounds[2], other[2]) - max(bounds[0], other[0])) * max(0, min(bounds[3], other[3]) - max(bounds[1], other[1]))
            if area / (2 * 192 ** 2 - area) >= .8:
                repeated += 1
                break
        masks[record["camera"], bounds[1]:bounds[3], bounds[0]:bounds[2]] = True
        previous.append(record)
    return {"union_coverage": float(masks.mean()), "repeat_count": repeated,
            "repeat_denominator": max(0, len(records) - 1)}
