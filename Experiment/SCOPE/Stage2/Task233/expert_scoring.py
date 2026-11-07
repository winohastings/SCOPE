from __future__ import annotations

import numpy as np


def wrap_angle(values):
    return (np.asarray(values) + np.pi) % (2 * np.pi) - np.pi


def sample_path(points, distances):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError("invalid reachable path")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 1e-8))
    points = points[keep]
    if len(points) < 2:
        raise ValueError("degenerate nonterminal expert path")
    cumulative = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    targets = np.minimum(np.asarray(distances), cumulative[-1])
    return np.stack([np.interp(targets, cumulative, points[:, axis]) for axis in range(3)], axis=1)


def world_to_body(points, anchor, headings):
    points, anchor = np.asarray(points), np.asarray(anchor)
    delta = points - anchor[:3]
    cosine, sine = np.cos(anchor[3]), np.sin(anchor[3])
    return np.column_stack((-sine * delta[:, 0] + cosine * delta[:, 1],
                            -cosine * delta[:, 0] - sine * delta[:, 1], delta[:, 2],
                            wrap_angle(np.asarray(headings) - anchor[3]))).astype(np.float32)


def expert_at_pose(backend, observation, step_m=.25):
    import habitat_sim

    environment = backend.runtime.env
    position = np.asarray(environment.sim.get_agent_state().position)
    paths = []
    for goal in environment.current_episode.goals:
        query = habitat_sim.ShortestPath()
        query.requested_start = position
        query.requested_end = np.asarray(goal.position)
        if environment.sim.pathfinder.find_path(query) and np.isfinite(query.geodesic_distance):
            paths.append((float(query.geodesic_distance), np.asarray(query.points)))
    if not paths:
        raise ValueError("no reachable current-pose goal path")
    distance, points = min(paths, key=lambda record: record[0])
    anchor = observation["pose"].as_array()
    stop = distance < 3.0
    if stop:
        actions = np.zeros((8, 4), dtype=np.float32)
        error = 0.
    else:
        targets = sample_path(points, np.arange(1, 9) * step_m)
        world = np.column_stack((-targets[:, 2], targets[:, 0], -targets[:, 1]))
        deltas = np.diff(np.vstack((anchor[:3], world)), axis=0)
        headings = np.arctan2(-deltas[:, 0], deltas[:, 1])
        for index in range(8):
            if np.linalg.norm(deltas[index, :2]) < 1e-8:
                headings[index] = anchor[3] if index == 0 else headings[index - 1]
        actions = world_to_body(world, anchor, headings)
        projected = backend.project_action_to_world(observation["pose"], actions)
        error = float(np.max(np.abs(projected[:, :3] - world)))
        if error > 1e-5:
            raise ValueError("expert/action adapter coordinate round trip failed")
    return {"actions": actions.tolist(), "stop": stop, "distance": distance,
            "source": "current-pose shortest geodesic path to R2R train goal; not time-index GT replay",
            "step_m": step_m, "waypoints": 8, "coordinate_roundtrip_max_abs": error,
            "habitat_path": points.tolist(), "pose": anchor.tolist()}


def score(actions, stopped, expert):
    predicted = np.asarray(actions, dtype=np.float64)
    target = np.asarray(expert["actions"], dtype=np.float64)
    if predicted.shape != (8, 4) or target.shape != (8, 4) or not np.isfinite(predicted).all():
        raise ValueError("expected finite released 8x4 action chunk")
    translation = float(np.mean(np.sum((predicted[:, :3] - target[:, :3]) ** 2, axis=1)))
    heading = float(np.mean(wrap_angle(predicted[:, 3] - target[:, 3]) ** 2))
    stop_error = float(bool(stopped) != expert["stop"])
    loss = translation + .1 * heading + stop_error
    return {"score_gt": -loss, "translation_loss": translation, "heading_loss": heading,
            "stop_loss": stop_error}
