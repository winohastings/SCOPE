from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


def body_displacement(before, after):
    delta = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
    yaw = float(before[3])
    return np.array([math.cos(yaw) * delta[0] + math.sin(yaw) * delta[1],
                     -math.sin(yaw) * delta[0] + math.cos(yaw) * delta[1], delta[2],
                     (delta[3] + math.pi) % (2 * math.pi) - math.pi], dtype=np.float32)


@dataclass(frozen=True)
class ExecutionFeedback:
    before: list[float]
    after: list[float]
    predicted_waypoints: list[list[float]]
    executed_poses: list[list[float]]
    current_tokens: int
    perception_ms: float
    model_ms: float
    navigation_done: bool
    termination: str
    action_executed: bool

    def __post_init__(self):
        if len(self.before) != 4 or len(self.after) != 4 or not self.predicted_waypoints:
            raise ValueError("complete pose and predicted waypoints required")
        if self.action_executed and not self.executed_poses:
            raise ValueError("actual execution poses are required after action execution")
        if not self.action_executed and not self.navigation_done:
            raise ValueError("no-action feedback is valid only for confirmed navigation STOP")
        if not np.isfinite(np.asarray(self.before + self.after)).all():
            raise ValueError("nonfinite pose")

    def vector(self):
        actual = body_displacement(self.before, self.after)
        predicted = np.asarray(self.predicted_waypoints, dtype=np.float32)[-1, :4]
        error = predicted - actual
        error[3] = (error[3] + math.pi) % (2 * math.pi) - math.pi
        return np.concatenate((actual, predicted, error,
                               [self.current_tokens / 192, self.perception_ms / 1000, self.model_ms / 1000,
                                float(self.navigation_done), float(self.action_executed), len(self.executed_poses) / 8])).astype(np.float32)

    def record(self):
        return asdict(self) | {"actual_body_displacement": self.vector()[:4].tolist(),
                              "prediction_execution_error": self.vector()[8:12].tolist()}


def cost_aware_reward(*, distance_before, distance_after, collision, terminal, success,
                      spl, focus_count, latency_ms, token_weight=0.01, latency_weight=0.001):
    values = [distance_before, distance_after, collision, spl, focus_count, latency_ms]
    if any(value is None for value in values) or not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError("teacher signals must be measured, never silently filled")
    if not 0 <= focus_count <= 2 or distance_before < 0 or distance_after < 0 or latency_ms < 0:
        raise ValueError("invalid teacher/cost signal")
    reward = 0.2 * float(np.clip((distance_before - distance_after) / 3, -1, 1)) - 0.01 - 0.02 * bool(collision)
    reward -= token_weight * focus_count * 64 / 192 + latency_weight * latency_ms / 1000
    if terminal:
        reward += 2 * bool(success) + 0.5 * spl - 0.5 * (not success)
    return float(reward)
