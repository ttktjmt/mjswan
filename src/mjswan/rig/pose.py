"""A skin measured against its robot: the turns and heights that make the avatar's rest
pose the robot's ``rest_pose``, where its hips sit, and how far its limbs are from the
robot's lengths. Everything here runs once, when the skin is added.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import mujoco
import numpy as np


def resolve_body(model: mujoco.MjModel, name: str) -> str | None:
    """The model's body *name* names: itself, else the one ``<entity>/<name>``.

    The browser matches the same way, so what passes here is found there.
    """
    names = [model.body(b).name for b in range(model.nbody)]
    if name in names:
        return name
    prefixed = [n for n in names if n.endswith(f"/{name}")]
    return prefixed[0] if len(prefixed) == 1 else None


def resolve_joint(model: mujoco.MjModel, name: str) -> int | None:
    """The model's joint *name* names: itself, else the one ``<entity>/<name>``."""
    names = [model.joint(j).name for j in range(model.njnt)]
    if name in names:
        return names.index(name)
    prefixed = [j for j, n in enumerate(names) if n.endswith(f"/{name}")]
    return prefixed[0] if len(prefixed) == 1 else None


def pose_at_rest(
    model: mujoco.MjModel, rest_pose: str | Mapping[str, float] | None
) -> mujoco.MjData:
    """The robot posed as the avatar rests: a keyframe by name, or hinge and slide joint
    -> value (radians, metres) over the zero pose, which ``None`` is."""
    data = mujoco.MjData(model)
    if isinstance(rest_pose, str):
        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, rest_pose)
        if key < 0:
            keys = [model.key(k).name for k in range(model.nkey)]
            raise ValueError(
                f"rest_pose {rest_pose!r} is not a keyframe; there are {keys}."
            )
        mujoco.mj_resetDataKeyframe(model, data, key)
    elif rest_pose:
        for name, value in rest_pose.items():
            joint = resolve_joint(model, name)
            if joint is None:
                raise ValueError(
                    f"rest_pose names {name!r}, which is not a joint of the model."
                )
            if model.jnt_type[joint] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                raise ValueError(
                    f"rest_pose sets hinge and slide joints; {name!r} is neither."
                )
            data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_kinematics(model, data)
    return data


def standing_height(model: mujoco.MjModel, body: int, data: mujoco.MjData) -> float:
    """How high *body* is above the lowest point of the colliding geoms of its tree: the
    robot's root over its soles, wherever *data* stands it."""
    tree = model.body_rootid[body]
    bottoms = [
        _geom_bottom(model, data, g)
        for g in range(model.ngeom)
        if model.body_rootid[model.geom_bodyid[g]] == tree
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    bottom = min(bottoms, default=math.inf)
    return float(data.xpos[body][2] - (bottom if math.isfinite(bottom) else 0.0))


def rest_offsets(
    model: mujoco.MjModel, bodies: Mapping[str, int], root: str, rest: mujoco.MjData
) -> dict[str, list[float]]:
    """Each bone's rest offset (``[x, y, z, w]``, the avatar's frame) that has one: the
    turn undoing its body's from the zero pose to *rest*, both relative to the root body,
    so a keyframe's own tilt is not carried in."""
    zero = mujoco.MjData(model)
    mujoco.mj_kinematics(model, zero)
    offsets = {}
    for bone, body in bodies.items():
        turn = np.empty(4)
        mujoco.mju_mulQuat(
            turn,
            _relative(rest, bodies[root], body),
            _inverse(_relative(zero, bodies[root], body)),
        )
        w, x, y, z = turn
        if abs(w) < 1 - 1e-9:
            # MuJoCo's x, y, z are the avatar's z, x, y; inverted, it undoes the turn.
            offsets[bone] = [-y, -z, -x, w]
    return offsets


def hips_offset(
    model: mujoco.MjModel,
    bodies: Mapping[str, int],
    root: str,
    rest: mujoco.MjData,
    *,
    limb_roots: Mapping[str, str],
    positions: Mapping[str, np.ndarray],
) -> tuple[float, float, float]:
    """Where the avatar's root bone sits on the root body (its frame, metres): across the
    floor, the centre of its limbs' first bones over the centre of the robot's limb
    roots. *limb_roots* maps each limb's first bone to the bone above it."""
    frame = rest.xmat[bodies[root]].reshape(3, 3)
    robot, avatar = [], []
    for bone, above in limb_roots.items():
        start = limb_root(model, bodies[above], bodies[bone])
        robot.append(frame.T @ (rest.xpos[start] - rest.xpos[bodies[root]]))
        x, _, z = positions[bone] - positions[root]
        avatar.append((z, x))
    if not robot:
        return (0.0, 0.0, 0.0)
    (rx, ry, _), (ax, ay) = np.mean(robot, axis=0), np.mean(avatar, axis=0)
    # + 0.0 so a -0.0 is written as 0.0.
    return (round(float(rx - ax), 4) + 0.0, round(float(ry - ay), 4) + 0.0, 0.0)


def limb_ratios(
    model: mujoco.MjModel,
    bodies: Mapping[str, int],
    limbs: Sequence[tuple[str, str, str]],
    *,
    limb_roots: Mapping[str, str],
    positions: Mapping[str, np.ndarray],
) -> dict[tuple[str, str, str], float]:
    """Each limb's length over the robot's, from its first joint to its end, at the zero
    pose. *limb_roots* maps each limb's first bone to the bone above it. Where the end
    rides the lower bone's body (a Go1 has no foot body), the robot's limb ends at that
    body's farthest colliding geom."""
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    out = {}
    for upper, lower, end in limbs:
        start = data.xpos[limb_root(model, bodies[limb_roots[upper]], bodies[upper])]
        if bodies[end] == bodies[lower]:
            stop = _farthest_geom(model, data, bodies[lower])
        else:
            stop = data.xpos[bodies[end]]
        robot = float(np.linalg.norm(stop - start))
        avatar = float(
            np.linalg.norm(positions[lower] - positions[upper])
            + np.linalg.norm(positions[end] - positions[lower])
        )
        if robot > 0:
            out[(upper, lower, end)] = avatar / robot
    return out


def euler_to_quat(degrees: Sequence[float]) -> list[float]:
    """Turns about the fixed x, then y, then z axis, as ``[x, y, z, w]``."""
    q = [0.0, 0.0, 0.0, 1.0]
    for axis, angle in enumerate(degrees):
        half = math.radians(angle) / 2
        r = [0.0, 0.0, 0.0, math.cos(half)]
        r[axis] = math.sin(half)
        q = quat_mul(r, q)
    return q


def quat_mul(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """``a · b`` for ``[x, y, z, w]`` quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def _relative(data: mujoco.MjData, root: int, body: int) -> np.ndarray:
    """*body*'s rotation in *root*'s frame, MuJoCo's ``(w, x, y, z)``."""
    out = np.empty(4)
    mujoco.mju_mulQuat(out, _inverse(data.xquat[root]), data.xquat[body])
    return out


def _inverse(quat: np.ndarray) -> np.ndarray:
    out = np.empty(4)
    mujoco.mju_negQuat(out, quat)
    return out


def limb_root(model: mujoco.MjModel, above: int, body: int) -> int:
    """The first body below *above* on the way down to *body*: where the limb leaves the
    trunk, at its first joint. *body* itself if *above* is not over it."""
    start = body
    while body > 0 and model.body_parentid[body] != above:
        body = model.body_parentid[body]
    return body if body > 0 else start


def _geom_bottom(model: mujoco.MjModel, data: mujoco.MjData, g: int) -> float:
    """How low geom *g* reaches (world z), as the browser's ``geomBottom`` finds it."""
    kind = mujoco.mjtGeom(model.geom_type[g])
    # The world z of each of the geom's axes, and its sizes along them.
    zx, zy, zz = data.geom_xmat[g].reshape(3, 3)[2]
    a, b, c = model.geom_size[g]
    z = data.geom_xpos[g][2]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return z - a
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return z - abs(zz) * b - a
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return z - abs(zz) * b - a * math.sqrt(max(0.0, 1 - zz * zz))
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return z - math.hypot(zx * a, zy * b, zz * c)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return z - abs(zx) * a - abs(zy) * b - abs(zz) * c
    if kind in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
        return math.inf
    cx, cy, cz, hx, hy, hz = model.geom_aabb[g]
    return z + zx * cx + zy * cy + zz * cz - abs(zx) * hx - abs(zy) * hy - abs(zz) * hz


def _farthest_geom(model: mujoco.MjModel, data: mujoco.MjData, body: int) -> np.ndarray:
    geoms = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == body
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    if not geoms:
        return data.xpos[body]
    return max(
        (data.geom_xpos[g] for g in geoms),
        key=lambda p: np.linalg.norm(p - data.xpos[body]),
    )


__all__ = [
    "euler_to_quat",
    "hips_offset",
    "limb_ratios",
    "limb_root",
    "pose_at_rest",
    "quat_mul",
    "resolve_body",
    "resolve_joint",
    "rest_offsets",
    "standing_height",
]
