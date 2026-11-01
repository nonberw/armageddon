import mujoco
import numpy as np
import pytest

from armageddon.kinematics import (
    HOME, default_arm, forward, jacobian, quat_from_matrix, rotation, rotvec_from_matrix,
)
from armageddon.mjcf import load_model

CHAIN = default_arm()
RNG = np.random.default_rng(0)
RANDOM_QS = [RNG.uniform(CHAIN.lower, CHAIN.upper) for _ in range(25)]


@pytest.fixture(scope="module")
def sim():
    model = load_model(CHAIN)
    return model, mujoco.MjData(model)


@pytest.mark.parametrize("q", RANDOM_QS)
def test_fk_matches_mujoco(sim, q):
    model, data = sim
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    site = model.site("ee").id
    pose = forward(CHAIN, q)
    np.testing.assert_allclose(pose.pos, data.site_xpos[site], atol=1e-9)
    np.testing.assert_allclose(pose.rot, data.site_xmat[site].reshape(3, 3), atol=1e-9)


@pytest.mark.parametrize("q", RANDOM_QS)
def test_jacobian_matches_mujoco(sim, q):
    model, data = sim
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, model.site("ee").id)
    J, _ = jacobian(CHAIN, q)
    np.testing.assert_allclose(J, np.vstack([jacp, jacr]), atol=1e-9)


def test_jacobian_matches_finite_differences():
    q, h = HOME + 0.1, 1e-6
    J, pose = jacobian(CHAIN, q)
    for i in range(CHAIN.dof):
        dq = np.zeros(CHAIN.dof)
        dq[i] = h
        p = forward(CHAIN, q + dq)
        np.testing.assert_allclose((p.pos - pose.pos) / h, J[:3, i], atol=1e-5)
        dR = rotvec_from_matrix(p.rot @ pose.rot.T) / h
        np.testing.assert_allclose(dR, J[3:, i], atol=1e-5)


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.5, 2.0, np.pi - 1e-6, np.pi])
def test_log_map_roundtrip(angle):
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    w = rotvec_from_matrix(rotation(axis, angle))
    np.testing.assert_allclose(rotation(w / max(np.linalg.norm(w), 1e-12), np.linalg.norm(w)),
                               rotation(axis, angle), atol=1e-6)


@pytest.mark.parametrize("q", RANDOM_QS[:10])
def test_quat_matches_mujoco(q):
    R = forward(CHAIN, q).rot
    expected = np.zeros(4)
    mujoco.mju_mat2Quat(expected, R.ravel())
    got = quat_from_matrix(R)
    assert min(np.abs(got - expected).max(), np.abs(got + expected).max()) < 1e-9
