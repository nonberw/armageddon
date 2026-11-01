"""Builds the MuJoCo scene (MJCF XML) from a `Chain`, so the sim and the math never drift apart.

The arm pieces (`arm_xml`, `arm_actuators`, `ARM_MATERIALS`) are public so other scenes can
reuse the exact same robot, e.g. with `gripper="parallel"` for real two-finger grasping.
"""

from __future__ import annotations

import numpy as np

from .kinematics import HOME, Chain, forward, quat_from_matrix

GRIPPERS = ("fixed", "parallel")
FINGER_TRAVEL = 0.045  # meters each finger can open from the center line

ARM_MATERIALS = """\
    <material name="link_a" rgba="0.62 0.36 1 1" specular="0.7" shininess="0.7"/>
    <material name="link_b" rgba="0.15 0.9 0.95 1" specular="0.7" shininess="0.7"/>
    <material name="housing" rgba="0.1 0.1 0.13 1" specular="0.4"/>
    <material name="finger" rgba="1 0.35 0.65 1" specular="0.5"/>"""


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a, dtype=float).ravel())


def _gripper(chain: Chain, gripper: str, indent: str) -> list[str]:
    z = chain.ee_offset[2]
    if gripper == "fixed":
        return [
            f'{indent}<geom type="box" pos="0 0 {z - 0.045:.4g}" size="0.012 0.04 0.01" material="housing"/>',
            f'{indent}<geom type="box" pos="0 0.028 {z - 0.015:.4g}" size="0.008 0.006 0.028" material="finger"/>',
            f'{indent}<geom type="box" pos="0 -0.028 {z - 0.015:.4g}" size="0.008 0.006 0.028" material="finger"/>',
        ]
    # Two slide-jointed fingers. At q=0 their inner faces meet on the center line,
    # so the finger joint value is exactly the half-gap between the pads.
    pad = 'condim="4" friction="1.5 0.02 0.002" solref="0.005 1"'
    lines = [f'{indent}<geom type="box" pos="0 0 {z - 0.05:.4g}" size="0.014 0.06 0.008" material="housing"/>']
    for side, sign in (("l", 1), ("r", -1)):
        lines += [
            f'{indent}<body name="finger_{side}" pos="0 0 {z - 0.015:.4g}">',
            f'{indent}  <joint name="finger_{side}" type="slide" axis="0 {sign} 0" range="0 {FINGER_TRAVEL}" damping="2" armature="0.001"/>',
            f'{indent}  <geom type="box" pos="0 {sign * 0.006} 0" size="0.012 0.006 0.028" material="finger" {pad}/>',
            f"{indent}</body>",
        ]
    return lines


def _body(chain: Chain, i: int, indent: str, gripper: str) -> str:
    joint = chain.joints[i]
    last = i == chain.dof - 1
    nxt = chain.ee_offset if last else chain.joints[i + 1].offset
    radius = 0.035 - 0.003 * i
    material = "link_a" if i % 2 == 0 else "link_b"
    a = joint.axis * 0.035

    lines = [
        f'{indent}<body name="link{i + 1}" pos="{_v(joint.offset)}">',
        f'{indent}  <joint name="{joint.name}" axis="{_v(joint.axis)}" range="{joint.lower:.6g} {joint.upper:.6g}"/>',
        f'{indent}  <geom type="cylinder" fromto="{_v(-a)} {_v(a)}" size="{radius + 0.012:.4g}" material="housing"/>',
    ]
    if last:
        # With real fingers, stop the wrist link short so it can't poke through a grasped object.
        stub = nxt * 0.25 if gripper == "parallel" else nxt
        lines.append(f'{indent}  <geom type="capsule" fromto="0 0 0 {_v(stub)}" size="{radius:.4g}" material="{material}"/>')
        lines += _gripper(chain, gripper, indent + "  ")
        lines.append(f'{indent}  <site name="ee" pos="{_v(chain.ee_offset)}" size="0.008" rgba="1 1 1 0.8"/>')
    else:
        lines.append(f'{indent}  <geom type="capsule" fromto="0 0 0 {_v(nxt)}" size="{radius:.4g}" material="{material}"/>')
        lines.append(_body(chain, i + 1, indent + "  ", gripper))
    lines.append(f"{indent}</body>")
    return "\n".join(lines)


def arm_xml(chain: Chain, gripper: str = "fixed", indent: str = "    ") -> str:
    """Base + kinematic chain bodies, to drop inside <worldbody>. Uses ARM_MATERIALS."""
    if gripper not in GRIPPERS:
        raise ValueError(f"gripper must be one of {GRIPPERS}, got {gripper!r}")
    base = f'{indent}<geom name="base" type="cylinder" pos="0 0 0.05" size="0.09 0.05" material="housing" contype="0"/>'
    return base + "\n" + _body(chain, 0, indent, gripper)


def arm_actuators(chain: Chain, kp: float = 300.0, gripper: str = "fixed", indent: str = "    ") -> str:
    lines = [
        f'{indent}<position name="act_{j.name}" joint="{j.name}" kp="{kp:g}" ctrlrange="{j.lower:.6g} {j.upper:.6g}"/>'
        for j in chain.joints
    ]
    if gripper == "parallel":
        lines += [
            f'{indent}<position name="act_finger_{s}" joint="finger_{s}" kp="200" ctrlrange="0 {FINGER_TRAVEL}"/>'
            for s in ("l", "r")
        ]
    return "\n".join(lines)


def build_mjcf(chain: Chain, home: np.ndarray = HOME, kp: float = 300.0) -> str:
    pose = forward(chain, home)
    return f"""<mujoco model="armageddon">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast"/>

  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.35 0.35 0.4" diffuse="0.5 0.5 0.55" specular="0.1 0.1 0.1"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.16 0.09 0.32" rgb2="0.01 0.01 0.04" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.13 0.12 0.2" rgb2="0.08 0.08 0.13" width="512" height="512" mark="edge" markrgb="0.45 0.3 0.9"/>
    <material name="floor" texture="grid" texrepeat="10 10" reflectance="0.2"/>
{ARM_MATERIALS}
  </asset>

  <default>
    <joint damping="3" armature="0.01"/>
    <geom contype="1" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0.6 -0.6 2.2" dir="-0.3 0.3 -1" diffuse="0.9 0.85 1" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor" contype="0" conaffinity="1"/>
{arm_xml(chain)}
    <body name="target" mocap="true" pos="{_v(pose.pos)}" quat="{_v(quat_from_matrix(pose.rot))}">
      <geom type="sphere" size="0.022" rgba="1 0.2 0.45 0.45" contype="0"/>
      <geom type="capsule" fromto="0 0 0 0.07 0 0" size="0.004" rgba="1 0.2 0.2 1" contype="0"/>
      <geom type="capsule" fromto="0 0 0 0 0.07 0" size="0.004" rgba="0.2 1 0.3 1" contype="0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.07" size="0.004" rgba="0.3 0.5 1 1" contype="0"/>
    </body>
  </worldbody>

  <actuator>
{arm_actuators(chain, kp)}
  </actuator>

  <keyframe>
    <key name="home" qpos="{_v(home)}" ctrl="{_v(home)}" mpos="{_v(pose.pos)}" mquat="{_v(quat_from_matrix(pose.rot))}"/>
  </keyframe>
</mujoco>
"""


def load_model(chain: Chain | None = None):
    import mujoco

    from .kinematics import default_arm

    return mujoco.MjModel.from_xml_string(build_mjcf(chain or default_arm()))
