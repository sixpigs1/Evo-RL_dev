"""Collision geometry and predicates for precision and multi-stage tabletop tasks."""

from itertools import pairwise
from xml.etree import ElementTree as ET

import numpy as np

# Coupon dimensions in metres. X is the insertion axis in both mating frames.
PIN_Y = (-0.008, 0.008)
PIN_HALF = 0.0025
SEAT_X = 0.017
PIN_TIP_X = 0.037
PIN_DEPTH = PIN_TIP_X - SEAT_X
DRAWER_TRAVEL = 0.125
TRAY_X = 0.061
TRAY_Y = 0.048
TRAY_FLOOR = 0.005
TRAY_RIM = 0.034
ITEM_HALF = np.array([0.012, 0.011, 0.011])
DISTRACTOR_HALF = np.array([0.014, 0.012, 0.010])
CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])


def _vec(values):
    return " ".join(f"{value:.8g}" for value in values)


def _box(parent, name, pos, size, color, **kwargs):
    return ET.SubElement(
        parent, "geom", name=name, type="box", pos=_vec(pos), size=_vec(size), rgba=color, **kwargs
    )


def _site(body, name, pos):
    ET.SubElement(body, "site", name=name, pos=_vec(pos), size="0.001", rgba="0 0 0 0")


def _plug(world, y):
    plug = ET.SubElement(world, "body", name="object", pos=_vec((0.188, y, 0.061)))
    ET.SubElement(plug, "freejoint", name="object_free")
    _box(
        plug,
        "plug_housing",
        (0, 0, 0),
        (SEAT_X, 0.016, 0.013),
        ".18 .32 .61 1",
        density="730",
        friction="1.05 .006 .0005",
    )
    for i, offset in enumerate(PIN_Y):
        _box(
            plug,
            f"pin_{i}",
            (0.026, offset, 0),
            (0.009, PIN_HALF, PIN_HALF),
            ".68 .61 .36 1",
            density="7100",
            friction=".28 .004 .0001",
        )
        _box(
            plug,
            f"pin_tip_{i}",
            (0.036, offset, 0),
            (0.001, 0.002, 0.002),
            ".68 .61 .36 1",
            density="7100",
            friction=".28 .004 .0001",
        )
        _site(plug, f"pin_profile_{i}", (0.034, offset, 0))
        _site(plug, f"pin_tip_{i}", (PIN_TIP_X, offset, 0))
    _box(plug, "plug_key", (0.021, -0.003, 0.009), (0.004, 0.0025, 0.0016), ".12 .21 .42 1", density="730")
    _site(plug, "plug_key_tip", (0.025, -0.003, 0.009))
    _site(plug, "plug_seat", (SEAT_X, 0, 0))
    _site(plug, "plug_grasp", (-0.006, 0, 0))


def _socket(world, *, mobile, clearance):
    attrs = {} if mobile else {"mocap": "true"}
    pos = (0.288, 0.010, 0.092) if mobile else (0.292, 0.15, 0.092)
    socket = ET.SubElement(world, "body", name="socket", pos=_vec(pos), **attrs)
    if mobile:
        ET.SubElement(
            socket,
            "joint",
            name="fixture_slide",
            type="slide",
            axis="0 1 0",
            range="-.045 .045",
            damping="1.8",
            armature=".002",
        )
        ET.SubElement(
            socket,
            "joint",
            name="fixture_yaw",
            type="hinge",
            axis="0 0 1",
            range="-20 20",
            damping=".025",
            stiffness=".012",
            armature=".0004",
        )
        _box(
            socket,
            "fixture_handle",
            (-0.023, 0.085, 0.034),
            (0.013, 0.015, 0.011),
            ".85 .42 .14 1",
            density="650",
            friction="1.1 .01 .001",
        )
        _box(
            socket,
            "fixture_handle_stem",
            (-0.018, 0.049, 0.013),
            (0.009, 0.041, 0.009),
            ".35 .37 .38 1",
            density="650",
        )
        _site(socket, "fixture_grasp", (-0.023, 0.085, 0.034))
        _box(
            world,
            "fixture_rail",
            (pos[0] + 0.012, pos[1], pos[2] - 0.023),
            (0.024, 0.096, 0.004),
            ".28 .3 .33 1",
            contype="0",
            conaffinity="0",
        )
    else:
        _box(socket, "socket_stand", (0.012, 0, -0.055), (0.022, 0.033, 0.036), ".28 .30 .33 1")

    # A lower shroud lip bears the housing weight after the jaws release.
    _box(
        socket,
        "socket_lower_lip",
        (-0.007, 0, -0.0164),
        (0.007, 0.018, 0.003),
        ".78 .80 .77 1",
        density="700",
        friction=".45 .004 .0001",
    )

    # Tile the actual passage cross-section around three apertures.
    radius = PIN_HALF + clearance
    holes = [(y - radius, y + radius, -radius, radius) for y in PIN_Y]
    holes.append(
        (
            -0.003 - 0.0025 - clearance,
            -0.003 + 0.0025 + clearance,
            0.009 - 0.0016 - clearance,
            0.009 + 0.0016 + clearance,
        )
    )
    ys = sorted({-0.025, 0.025, *(v for h in holes for v in h[:2])})
    zs = sorted({-0.018, 0.018, *(v for h in holes for v in h[2:])})
    # Merge runs along z to avoid dozens of redundant coplanar contacts.
    for iy, (y0, y1) in enumerate(pairwise(ys)):
        start = None
        for iz, (z0, z1) in enumerate(pairwise(zs)):
            y, z = (y0 + y1) / 2, (z0 + z1) / 2
            blocked = not any(a < y < b and c < z < d for a, b, c, d in holes)
            if blocked and start is None:
                start = z0
            if start is not None and (not blocked or iz == len(zs) - 2):
                end = z1 if blocked else z0
                _box(
                    socket,
                    f"socket_wall_{iy}_{iz}",
                    (0.011, y, (start + end) / 2),
                    (0.011, (y1 - y0) / 2, (end - start) / 2),
                    ".78 .80 .77 1",
                    density="700",
                    friction=".28 .004 .0001",
                )
                start = None
    _box(socket, "socket_backstop", (0.024, 0, 0), (0.002, 0.025, 0.018), ".55 .57 .55 1", density="850")
    _site(socket, "socket_mouth", (0, 0, 0))


def build_plug(world, difficulty, *, mobile=False):
    y = -0.15 if mobile else 0.17
    # The narrow cradle supports the housing while leaving the gripping sides accessible.
    _box(world, "plug_cradle", (0.188, y, 0.024), (0.015, 0.008, 0.024), ".27 .29 .32 1")
    _plug(world, y)
    _socket(world, mobile=mobile, clearance=0.0012 if difficulty == "nominal" else 0.0006)


def build_mobile_plug(world, difficulty):
    build_plug(world, difficulty, mobile=True)


def reset_plug(sim):
    qpos = sim.data.joint("object_free").qpos
    precise = sim.cfg.task_difficulty == "precision"
    shift = min(sim.cfg.randomization, 0.004 if precise else 0.0025)
    qpos[:2] += sim.rng.uniform(-shift, shift, 2)
    if shift:
        yaw = sim.rng.uniform(-0.12 if precise else -0.045, 0.12 if precise else 0.045)
        qpos[3:] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
    if sim.cfg.task == "bimanual_stabilized_insert":
        sim.data.joint("fixture_slide").qpos[0] = sim.rng.uniform(-shift, shift)
        sim.data.joint("fixture_yaw").qpos[0] = (
            sim.rng.uniform(-6, 6) * np.pi / 180 if precise and shift else 0
        )
    elif shift:
        mocap = sim.model.body("socket").mocapid[0]
        sim.data.mocap_pos[mocap, 1] += sim.rng.uniform(-2 * shift, 2 * shift)
        yaw = sim.rng.uniform(-0.12 if precise else -0.035, 0.12 if precise else 0.035)
        sim.data.mocap_quat[mocap] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))


def socket_local(sim, site):
    mouth = sim.data.site("socket_mouth")
    return (sim.data.site(site).xpos - mouth.xpos) @ mouth.xmat.reshape(3, 3)


def plug_seated(sim):
    plug = sim.data.body("object").xmat.reshape(3, 3)
    socket = sim.data.site("socket_mouth").xmat.reshape(3, 3)
    alignment = socket.T @ plug
    if np.trace(alignment) < 1 + 2 * np.cos(np.deg2rad(5)):
        return False
    clearance = 0.0012 if sim.cfg.task_difficulty == "nominal" else 0.0006
    seat = socket_local(sim, "plug_seat")
    if not -0.0025 <= seat[0] <= 0.0008:
        return False
    for index, lateral in enumerate(PIN_Y):
        tip = socket_local(sim, f"pin_tip_{index}")
        profile = socket_local(sim, f"pin_profile_{index}")
        if not PIN_DEPTH - 0.003 <= tip[0] <= 0.0224:
            return False
        if max(abs(profile[1] - lateral), abs(profile[2])) > clearance:
            return False
    key = socket_local(sim, "plug_key_tip")
    return bool(key[0] >= 0.0055 and abs(key[1] + 0.003) <= clearance and abs(key[2] - 0.009) <= clearance)


def _quiet(sim, name):
    velocity = sim.data.joint(f"{name}_free").qvel
    return np.linalg.norm(velocity[:3]) < 0.015 and np.linalg.norm(velocity[3:]) < 0.3


def plug_released(sim):
    return plug_seated(sim) and not sim.gripper_touches(0, "object") and _quiet(sim, "object")


def plug_aligned(sim):
    seat = socket_local(sim, "plug_seat")
    return bool(-0.045 < seat[0] < 0.005 and np.linalg.norm(seat[1:]) < 0.006)


def fixture_held(sim):
    slide, yaw = (sim.data.joint(name) for name in ("fixture_slide", "fixture_yaw"))
    return (
        abs(slide.qpos[0]) < 0.012
        and abs(yaw.qpos[0]) < np.deg2rad(8)
        and abs(slide.qvel[0]) < 0.035
        and abs(yaw.qvel[0]) < 0.35
        and sim.grasping(0, "socket", "fixture_handle")
    )


def mobile_plug_released(sim):
    return (
        fixture_held(sim)
        and plug_seated(sim)
        and not sim.gripper_touches(1, "object")
        and _quiet(sim, "object")
    )


def build_drawer(world, difficulty):
    drawer = ET.SubElement(world, "body", name="drawer", pos=".365 0 .045")
    ET.SubElement(
        drawer,
        "joint",
        name="drawer_slide",
        type="slide",
        axis="-1 0 0",
        range=f"0 {DRAWER_TRAVEL}",
        damping="1.5" if difficulty == "nominal" else "2.2",
        frictionloss=".24",
        armature=".008",
    )
    _box(drawer, "drawer_floor", (0, 0, 0.002), (TRAY_X, TRAY_Y, 0.003), ".66 .62 .52 1", density="450")
    for sign, side in ((-1, "near"), (1, "far")):
        _box(
            drawer,
            f"drawer_{side}_end",
            (sign * (TRAY_X + 0.003), 0, 0.019),
            (0.003, TRAY_Y + 0.003, 0.015),
            ".66 .62 .52 1",
            density="450",
        )
        _box(
            drawer,
            f"drawer_{side}_wall",
            (0, sign * (TRAY_Y + 0.003), 0.019),
            (TRAY_X, 0.003, 0.015),
            ".66 .62 .52 1",
            density="450",
        )
        y = sign * 0.026
        ET.SubElement(
            drawer,
            "geom",
            name=f"handle_post_{side}",
            type="capsule",
            size=".005",
            fromto=_vec((-0.097, y, 0.046, -0.063, y, 0.023)),
            density="600",
        )
    ET.SubElement(
        drawer,
        "geom",
        name="drawer_handle",
        type="capsule",
        size=".008",
        fromto="-.101 -.026 .046 -.101 .026 .046",
        density="600",
    )
    _site(drawer, "drawer_handle_center", (-0.101, 0, 0.046))
    _box(world, "cabinet_top", (0.380, 0, 0.103), (0.078, 0.065, 0.009), ".28 .30 .34 1")
    for sign, side in ((-1, "near"), (1, "far")):
        _box(
            world,
            f"cabinet_{side}",
            (0.380, sign * 0.06, 0.047),
            (0.078, 0.006, 0.047),
            ".28 .30 .34 1",
        )
    _box(world, "cabinet_back", (0.456, 0, 0.047), (0.006, 0.052, 0.047), ".28 .30 .34 1")
    _box(world, "pickup_riser", (0.216, -0.16, 0.011), (0.011, 0.009, 0.011), ".32 .34 .35 1")
    for name, pos, size, color in (
        ("object", (0.216, -0.16, 0.022 + ITEM_HALF[2]), ITEM_HALF, ".79 .16 .12 1"),
        ("distractor", (0.286, -0.105, DISTRACTOR_HALF[2]), DISTRACTOR_HALF, ".12 .34 .77 1"),
    ):
        body = ET.SubElement(world, "body", name=name, pos=_vec(pos))
        ET.SubElement(body, "freejoint", name=f"{name}_free")
        _box(body, name, (0, 0, 0), size, color, density="540", friction="1.05 .006 .0005")


def drawer_open(sim):
    return bool(sim.data.joint("drawer_slide").qpos[0] >= 0.100)


def _corners(sim, name, half):
    body = sim.data.body(name)
    return (CORNERS * half) @ body.xmat.reshape(3, 3).T + body.xpos


def item_inside(sim):
    drawer = sim.data.body("drawer")
    corners = (_corners(sim, "object", ITEM_HALF) - drawer.xpos) @ drawer.xmat.reshape(3, 3)
    return bool(
        np.all(np.abs(corners[:, 0]) < TRAY_X - 0.001)
        and np.all(np.abs(corners[:, 1]) < TRAY_Y - 0.001)
        and min(corners[:, 2]) >= TRAY_FLOOR - 0.002
        and max(corners[:, 2]) < TRAY_RIM
    )


def item_stowed(sim):
    return (
        item_inside(sim)
        and not any(sim.gripper_touches(arm, "object") for arm in range(2))
        and _quiet(sim, "object")
    )


def open_item_stowed(sim):
    return drawer_open(sim) and item_stowed(sim)


def drawer_closed(sim):
    slide = sim.data.joint("drawer_slide")
    other = _corners(sim, "distractor", DISTRACTOR_HALF)
    return bool(
        slide.qpos[0] <= 0.005
        and abs(slide.qvel[0]) < 0.012
        and item_stowed(sim)
        and -0.002 <= min(other[:, 2]) <= 0.004
        and _quiet(sim, "distractor")
        and np.all(np.abs(other[:, 1]) <= 0.495)
        and np.all(np.abs(other[:, 0] - 0.16) <= 0.645)
    )


def drawer_recovery(sim):
    # Reopen before trying to stow again if the payload left the tray.
    if sim.phase == 2 and not item_inside(sim):
        return int(drawer_open(sim))
    return sim.phase


def fixture_recovery(sim):
    return 0 if sim.phase and not fixture_held(sim) else sim.phase
