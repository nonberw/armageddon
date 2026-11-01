"""Numerical inverse kinematics, from scratch.

Three classic solvers, so the README can show why the third one wins:
  * "transpose" - Jacobian transpose with Buss's optimal step size
  * "pinv"      - Moore-Penrose pseudo-inverse (Gauss-Newton)
  * "dls"       - damped least squares with adaptive Levenberg-Marquardt damping,
                  null-space bias toward a rest pose, and optional random restarts
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .kinematics import Chain, jacobian, rotvec_from_matrix

METHODS = ("transpose", "pinv", "dls")
STALL_WINDOW = 25  # iterations without a 1% improvement before giving up on this attempt


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iterations: int
    pos_error: float  # meters
    rot_error: float  # radians
    history: list[float] = field(default_factory=list)  # weighted task error per iteration


def _residual(chain, q, target_pos, target_rot, rot_weight):
    J, pose = jacobian(chain, q)
    e_pos = target_pos - pose.pos
    if target_rot is None:
        return e_pos, J[:3], float(np.linalg.norm(e_pos)), 0.0
    e_rot = rotvec_from_matrix(target_rot @ pose.rot.T)
    e = np.concatenate([e_pos, rot_weight * e_rot])
    J = np.vstack([J[:3], rot_weight * J[3:]])
    return e, J, float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_rot))


def _step(method, J, e, lam):
    if method == "transpose":
        g = J.T @ e
        JJg = J @ g
        denom = JJg @ JJg
        return g * ((e @ JJg) / denom if denom > 1e-12 else 0.0)
    if method == "pinv":
        return np.linalg.pinv(J, rcond=1e-4) @ e
    if method == "dls":
        m = J.shape[0]
        return J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(m), e)
    raise ValueError(f"unknown method {method!r}, expected one of {METHODS}")


def solve(
    chain: Chain,
    target_pos: np.ndarray,
    target_rot: np.ndarray | None = None,
    q0: np.ndarray | None = None,
    method: str = "dls",
    max_iters: int = 200,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    rot_weight: float = 0.3,
    max_step: float = 0.25,
    damping: float = 0.05,
    q_rest: np.ndarray | None = None,
    nullspace_gain: float = 0.0,
    restarts: int = 0,
    rng: np.random.Generator | None = None,
) -> IKResult:
    """Find joint angles that put the end effector at `target_pos` (and `target_rot`, if given).

    Starts from `q0` (warm start) or the middle of the joint range. With `restarts > 0`
    a failed solve is retried from random configurations and the best attempt is kept.
    """
    q0 = (chain.lower + chain.upper) / 2 if q0 is None else np.asarray(q0, dtype=float)
    best = _solve_once(chain, target_pos, target_rot, q0, method, max_iters, pos_tol, rot_tol,
                       rot_weight, max_step, damping, q_rest, nullspace_gain)
    rng = rng or np.random.default_rng()
    total_iters, history = best.iterations, best.history
    for _ in range(restarts):
        if best.success:
            break
        q_rand = rng.uniform(chain.lower, chain.upper)
        res = _solve_once(chain, target_pos, target_rot, q_rand, method, max_iters, pos_tol, rot_tol,
                          rot_weight, max_step, damping, q_rest, nullspace_gain)
        total_iters += res.iterations
        history = history + res.history
        if res.success or res.history[-1] < best.history[-1]:
            best = res
    best.iterations, best.history = total_iters, history
    return best


def _solve_once(chain, target_pos, target_rot, q0, method, max_iters, pos_tol, rot_tol,
                rot_weight, max_step, damping, q_rest, nullspace_gain) -> IKResult:
    q = chain.clip(q0.copy())
    lam = damping
    e, J, pos_err, rot_err = _residual(chain, q, target_pos, target_rot, rot_weight)
    err = float(np.linalg.norm(e))
    history = [err]

    for it in range(1, max_iters + 1):
        if pos_err < pos_tol and rot_err < rot_tol:
            return IKResult(q, True, it - 1, pos_err, rot_err, history)
        # Stuck in a local minimum: bail out early so a restart gets the budget instead.
        if len(history) > STALL_WINDOW and err > 0.99 * history[-STALL_WINDOW - 1]:
            return IKResult(q, False, it - 1, pos_err, rot_err, history)

        # Joints that would be pushed through their limit get frozen for this step and the
        # remaining ones re-solve the task; plain clipping would silently eat the progress.
        free = np.ones(chain.dof, dtype=bool)
        for _ in range(chain.dof):
            Jf = J * free
            dq = _step(method, Jf, e, lam)
            if nullspace_gain > 0 and q_rest is not None:
                N = np.eye(chain.dof) - np.linalg.pinv(Jf, rcond=1e-4) @ Jf
                dq += nullspace_gain * (N @ (q_rest - q)) * free
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm
            q_try = q + dq
            hit = free & ((q_try < chain.lower) | (q_try > chain.upper))
            if not hit.any() or hit.sum() == free.sum():
                break
            free &= ~hit

        q_new = chain.clip(q + dq)
        e_new, J_new, pos_new, rot_new = _residual(chain, q_new, target_pos, target_rot, rot_weight)
        err_new = float(np.linalg.norm(e_new))

        if method == "dls":
            # Levenberg-Marquardt: trust the linear model more when it pays off, less when it doesn't.
            if err_new < err:
                lam = max(lam * 0.5, 1e-4)
            else:
                lam = min(lam * 2.0, 10.0)
                history.append(err)
                continue

        q, e, J, pos_err, rot_err, err = q_new, e_new, J_new, pos_new, rot_new, err_new
        history.append(err)

    success = pos_err < pos_tol and rot_err < rot_tol
    return IKResult(q, success, max_iters, pos_err, rot_err, history)
