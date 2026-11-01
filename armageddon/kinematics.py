"""Serial-chain kinematics in pure NumPy: SO(3) helpers, forward kinematics, geometric Jacobian.

Convention: every joint is revolute. Joint i sits at `offset` in its parent's frame
(no fixed rotation between frames), rotates about `axis` given in its own frame.
This is exactly how a MuJoCo <body pos=...><joint axis=.../> nest behaves, which
lets the tests check this code against MuJoCo's own FK.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- SO(3) ----

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues: rotation matrix about a unit `axis` by `angle` radians."""
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    """Log map SO(3) -> R^3 (axis * angle). Stable near 0 and near pi."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-6:
        return 0.5 * vee
    if np.pi - angle < 1e-4:
        # sin(angle) ~ 0, so read the axis off the symmetric part instead.
        B = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(B)))
        axis = B[:, i] / np.sqrt(B[i, i])
        if vee @ axis < 0:  # keep sign consistent with the antisymmetric part
            axis = -axis
        return axis * angle
    return angle / (2.0 * np.sin(angle)) * vee


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z), MuJoCo order."""
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------- chain ----

@dataclass(frozen=True)
class Joint:
    name: str
    offset: np.ndarray  # from parent joint frame, meters
    axis: np.ndarray    # unit vector in this joint's frame
    lower: float
    upper: float


@dataclass(frozen=True)
class Chain:
    joints: tuple[Joint, ...]
    ee_offset: np.ndarray  # tool point in the last joint's frame

    @property
    def dof(self) -> int:
        return len(self.joints)

    @property
    def lower(self) -> np.ndarray:
        return np.array([j.lower for j in self.joints])

    @property
    def upper(self) -> np.ndarray:
        return np.array([j.upper for j in self.joints])

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)


@dataclass(frozen=True)
class Pose:
    origins: np.ndarray  # (n, 3) joint positions in world
    axes: np.ndarray     # (n, 3) joint axes in world
    pos: np.ndarray      # (3,) end-effector position
    rot: np.ndarray      # (3, 3) end-effector orientation


def default_arm() -> Chain:
    """6-DoF arm: base yaw, shoulder, elbow, wrist roll, wrist pitch, tool roll."""
    X, Y, Z = np.eye(3)
    d = np.deg2rad
    return Chain(
        joints=(
            Joint("base_yaw",    np.array([0, 0, 0.10]), Z, d(-170), d(170)),
            Joint("shoulder",    np.array([0, 0, 0.05]), Y, d(-100), d(100)),
            Joint("elbow",       np.array([0, 0, 0.35]), Y, d(-150), d(150)),
            Joint("wrist_roll",  np.array([0, 0, 0.30]), Z, d(-170), d(170)),
            Joint("wrist_pitch", np.array([0, 0, 0.08]), Y, d(-120), d(120)),
            Joint("tool_roll",   np.array([0, 0, 0.06]), Z, d(-170), d(170)),
        ),
        ee_offset=np.array([0, 0, 0.08]),
    )


HOME = np.array([0.0, 0.4, 1.2, 0.0, 1.0, 0.0])


def forward(chain: Chain, q: np.ndarray) -> Pose:
    n = chain.dof
    origins = np.empty((n, 3))
    axes = np.empty((n, 3))
    R = np.eye(3)
    p = np.zeros(3)
    for i, (joint, qi) in enumerate(zip(chain.joints, q)):
        p = p + R @ joint.offset
        origins[i] = p
        axes[i] = R @ joint.axis
        R = R @ rotation(joint.axis, qi)
    return Pose(origins, axes, p + R @ chain.ee_offset, R)


def jacobian(chain: Chain, q: np.ndarray) -> tuple[np.ndarray, Pose]:
    """Geometric Jacobian (6 x n): rows 0-2 linear velocity, rows 3-5 angular velocity."""
    pose = forward(chain, q)
    Jv = np.cross(pose.axes, pose.pos - pose.origins).T
    Jw = pose.axes.T
    return np.vstack([Jv, Jw]), pose
