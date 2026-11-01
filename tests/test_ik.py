import numpy as np
import pytest

from armageddon import HOME, default_arm, forward, solve

CHAIN = default_arm()


def reachable_targets(n, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        q = rng.uniform(CHAIN.lower * 0.8, CHAIN.upper * 0.8)
        pose = forward(CHAIN, q)
        if pose.pos[2] > 0.05:
            out.append(pose)
    return out


def test_dls_position_only():
    targets = reachable_targets(100)
    ok = sum(solve(CHAIN, t.pos, q0=HOME).success for t in targets)
    assert ok >= 97


def test_dls_full_pose_with_restarts():
    targets = reachable_targets(100, seed=2)
    rng = np.random.default_rng(0)
    results = [solve(CHAIN, t.pos, t.rot, q0=HOME, restarts=10, rng=rng) for t in targets]
    assert sum(r.success for r in results) >= 99
    for r, t in zip(results, targets):
        if r.success:
            assert np.linalg.norm(forward(CHAIN, r.q).pos - t.pos) < 1e-4


@pytest.mark.parametrize("method", ["transpose", "pinv", "dls"])
def test_solution_respects_joint_limits(method):
    for t in reachable_targets(20, seed=3):
        q = solve(CHAIN, t.pos, t.rot, q0=HOME, method=method).q
        assert np.all(q >= CHAIN.lower - 1e-12) and np.all(q <= CHAIN.upper + 1e-12)


def test_unreachable_target_fails_gracefully():
    res = solve(CHAIN, np.array([5.0, 0.0, 0.5]), q0=HOME)
    assert not res.success
    assert np.all(np.isfinite(res.q))
    # Best effort: the arm should be stretched toward the target.
    assert forward(CHAIN, res.q).pos[0] > 0.7


def test_warm_start_converges_fast():
    t = forward(CHAIN, HOME + 0.02)
    res = solve(CHAIN, t.pos, t.rot, q0=HOME)
    assert res.success and res.iterations <= 5


def test_nullspace_pulls_toward_rest_pose():
    t = reachable_targets(1, seed=4)[0]
    plain = solve(CHAIN, t.pos, q0=HOME)
    biased = solve(CHAIN, t.pos, q0=HOME, q_rest=HOME, nullspace_gain=0.1)
    assert plain.success and biased.success
    assert np.linalg.norm(biased.q - HOME) <= np.linalg.norm(plain.q - HOME) + 1e-9
