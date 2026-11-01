"""Render assets/demo.gif: the arm tracks a moving target, IK solved from scratch every frame.

    python scripts/render_gif.py
"""

import argparse
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from armageddon import HOME, default_arm, forward, solve  # noqa: E402
from armageddon.kinematics import quat_from_matrix  # noqa: E402
from armageddon.mjcf import load_model  # noqa: E402
from armageddon.trajectory import PERIOD, figure_eight  # noqa: E402

TRAIL = 90


def add_sphere(scene, pos, size, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0]), pos, np.eye(3).ravel(), np.asarray(rgba, dtype=np.float32))
    scene.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "assets" / "demo.gif"))
    ap.add_argument("--width", type=int, default=520)
    ap.add_argument("--height", type=int, default=340)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    chain = default_arm()
    model = load_model(chain)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    renderer = mujoco.Renderer(model, args.height, args.width)

    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.24, 0.0, 0.33]
    cam.distance, cam.azimuth, cam.elevation = 1.2, 145, -16

    try:
        font = ImageFont.truetype("consola.ttf", 15)
    except OSError:
        font = ImageFont.load_default()

    q = HOME.copy()
    trail = []
    frames = []
    n = int(PERIOD * args.fps)
    for k in range(n):
        target_pos, target_rot = figure_eight(k / args.fps)
        t0 = time.perf_counter()
        res = solve(chain, target_pos, target_rot, q0=q, max_iters=50)
        ms = (time.perf_counter() - t0) * 1e3
        q = res.q

        data.qpos[:] = q
        data.mocap_pos[0] = target_pos
        data.mocap_quat[0] = quat_from_matrix(target_rot)
        mujoco.mj_forward(model, data)

        trail.append(forward(chain, q).pos)
        trail = trail[-TRAIL:]
        renderer.update_scene(data, cam)
        for i, p in enumerate(trail):
            a = (i + 1) / len(trail)
            add_sphere(renderer.scene, p, 0.004 + 0.004 * a, (0.2 + 0.8 * a, 0.95 - 0.5 * a, 1.0, 0.25 + 0.7 * a))

        img = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(img)
        hud = f"pos err {res.pos_error * 1e3:6.3f} mm   rot err {np.rad2deg(res.rot_error):5.3f} deg\n" \
              f"iters {res.iterations:2d}   solve {ms:5.2f} ms   pure numpy IK"
        draw.multiline_text((12, 10), hud, fill=(235, 230, 255), font=font, spacing=4)
        frames.append(np.asarray(img))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, frames, duration=1000 / args.fps, loop=0)
    print(f"wrote {args.out}  ({len(frames)} frames, {Path(args.out).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
