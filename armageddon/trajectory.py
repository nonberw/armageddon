"""A target that loops through a figure-eight in front of the robot, gripper pointing down."""

import numpy as np

from .kinematics import rotation

CENTER = np.array([0.42, 0.0, 0.30])
PERIOD = 6.0  # seconds per loop


def figure_eight(t: float) -> tuple[np.ndarray, np.ndarray]:
    w = 2 * np.pi * t / PERIOD
    pos = CENTER + np.array([0.08 * np.sin(2 * w), 0.24 * np.sin(w), 0.10 * np.cos(2 * w)])
    yaw = np.arctan2(pos[1], pos[0])
    tilt = 0.35 * np.sin(w)  # wobble the tool a bit so orientation tracking is visible
    rot = rotation(np.array([0.0, 0.0, 1.0]), yaw) @ rotation(np.array([0.0, 1.0, 0.0]), np.pi + tilt)
    return pos, rot
