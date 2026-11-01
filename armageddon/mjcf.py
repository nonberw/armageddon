"""Builds the MuJoCo scene (MJCF XML) from a `Chain`, so the sim and the math never drift apart."""

from __future__ import annotations

import numpy as np

from .kinematics import HOME, Chain, forward, quat_from_matrix


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.asarray(a, dtype=float).ravel())


def _body(chain: Chain, i: int, indent: str) -> str:
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
    if np.linalg.norm(nxt) > 1e-6:
        lines.append(f'{indent}  <geom type="capsule" fromto="0 0 0 {_v(nxt)}" size="{radius:.4g}" material="{material}"/>')
    if last:
        z = chain.ee_offset[2]
        lines += [
            f'{indent}  <geom type="box" pos="0 0 {z - 0.045:.4g}" size="0.012 0.04 0.01" material="housing"/>',
            f'{indent}  <geom type="box" pos="0 0.028 {z - 0.015:.4g}" size="0.008 0.006 0.028" material="finger"/>',
            f'{indent}  <geom type="box" pos="0 -0.028 {z - 0.015:.4g}" size="0.008 0.006 0.028" material="finger"/>',
            f'{indent}  <site name="ee" pos="{_v(chain.ee_offset)}" size="0.008" rgba="1 1 1 0.8"/>',
        ]
    else:
        lines.append(_body(chain, i + 1, indent + "  "))
    lines.append(f"{indent}</body>")
    return "\n".join(lines)


def build_mjcf(chain: Chain, home: np.ndarray = HOME, kp: float = 300.0) -> str:
    pose = forward(chain, home)
    actuators = "\n".join(
        f'    <position name="act_{j.name}" joint="{j.name}" kp="{kp:g}" ctrlrange="{j.lower:.6g} {j.upper:.6g}"/>'
        for j in chain.joints
    )
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
    <material name="link_a" rgba="0.62 0.36 1 1" specular="0.7" shininess="0.7"/>
    <material name="link_b" rgba="0.15 0.9 0.95 1" specular="0.7" shininess="0.7"/>
    <material name="housing" rgba="0.1 0.1 0.13 1" specular="0.4"/>
    <material name="finger" rgba="1 0.35 0.65 1" specular="0.5"/>
  </asset>

  <default>
    <joint damping="3" armature="0.01"/>
    <geom contype="1" conaffinity="0"/>
  </default>

  <worldbody>
    <light pos="0.6 -0.6 2.2" dir="-0.3 0.3 -1" diffuse="0.9 0.85 1" castshadow="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor" contype="0" conaffinity="1"/>
    <geom name="base" type="cylinder" pos="0 0 0.05" size="0.09 0.05" material="housing" contype="0"/>
{_body(chain, 0, "    ")}
    <body name="target" mocap="true" pos="{_v(pose.pos)}" quat="{_v(quat_from_matrix(pose.rot))}">
      <geom type="sphere" size="0.022" rgba="1 0.2 0.45 0.45" contype="0"/>
      <geom type="capsule" fromto="0 0 0 0.07 0 0" size="0.004" rgba="1 0.2 0.2 1" contype="0"/>
      <geom type="capsule" fromto="0 0 0 0 0.07 0" size="0.004" rgba="0.2 1 0.3 1" contype="0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.07" size="0.004" rgba="0.3 0.5 1 1" contype="0"/>
    </body>
  </worldbody>

  <actuator>
{actuators}
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
