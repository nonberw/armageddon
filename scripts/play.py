"""Interactive demo with real physics: the arm is driven by position motors, IK picks the setpoints.

    python scripts/play.py          # drag the target yourself
    python scripts/play.py --auto   # target flies a figure-eight on its own

Grab the target: double-click the pink sphere, then Ctrl + right-drag to move it,
Ctrl + left-drag to rotate it.
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from armageddon import HOME, default_arm, solve  # noqa: E402
from armageddon.kinematics import quat_from_matrix  # noqa: E402
from armageddon.mjcf import load_model  # noqa: E402
from armageddon.trajectory import figure_eight  # noqa: E402

CONTROL_HZ = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", action="store_true", help="move the target along a figure-eight")
    args = ap.parse_args()

    chain = default_arm()
    model = load_model(chain)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    q_des = HOME.copy()
    steps_per_control = max(1, round(1 / (CONTROL_HZ * model.opt.timestep)))
    rot = np.zeros(9)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.25, 0.0, 0.3]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.5, 145, -18
        while viewer.is_running():
            tick = time.perf_counter()

            if args.auto:
                pos, R = figure_eight(data.time)
                data.mocap_pos[0], data.mocap_quat[0] = pos, quat_from_matrix(R)
            mujoco.mju_quat2Mat(rot, data.mocap_quat[0])

            # Warm start from the last setpoint: a handful of iterations is enough at 60 Hz.
            res = solve(chain, data.mocap_pos[0].copy(), rot.reshape(3, 3), q0=q_des,
                        max_iters=15, q_rest=HOME, nullspace_gain=0.02)
            q_des = res.q
            data.ctrl[:] = q_des

            for _ in range(steps_per_control):
                # Gravity compensation: feed forward the torque gravity/coriolis would cost us,
                # so the position motors only fix the remaining error (~4x less tracking lag).
                data.qfrc_applied[:] = data.qfrc_bias
                mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(max(0.0, steps_per_control * model.opt.timestep - (time.perf_counter() - tick)))


if __name__ == "__main__":
    main()
