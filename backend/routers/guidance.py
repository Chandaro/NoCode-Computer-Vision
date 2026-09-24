"""
Steering guidance ("which way should I move the wheel").

Turns detections into a single glanceable instruction: go straight, keep left,
keep right, or stop. Built for a helmet-mounted camera + HUD: the camera looks
where the rider is heading, so the bottom-centre of the frame is the path ahead.

How it works (deliberately simple so it is fast and predictable on an edge
device):
  * Only obstacles low in the frame matter. Anything high up is far away or
    in the sky, so it is ignored (PATH_TOP).
  * Each obstacle gets a proximity score from how much of the frame it covers
    (a big box means close) and from how low it sits (lower = nearer).
  * The frame is split into three corridors (left / centre / right). We add up
    the proximity of the obstacles in each one.
  * The centre corridor is where the rider is actually going. If it is clear,
    the cue is GO. If it is blocked we point at the freer side. If everything is
    blocked and something is very close, the cue is STOP.
"""
from __future__ import annotations

import os
import tempfile
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from sqlmodel import Session

from database import get_session

router = APIRouter(tags=["guidance"])

VIDEO_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "video_out")
os.makedirs(VIDEO_OUT_DIR, exist_ok=True)

_jobs: dict = {}

# Below this fraction of image height, an object is treated as part of the path.
PATH_TOP = 0.45
# Corridor boundaries (left | centre | right) in x.
C_LEFT, C_RIGHT = 0.35, 0.65
# A box covering this fraction of the frame is "right on top of you".
AREA_VERY_CLOSE = 0.22
# Centre-corridor proximity above this means the path is blocked.
BLOCKED = 0.25
# ...and above this it is an emergency.
DANGER = 0.7


def _corridor_blocks(obstacles: list) -> tuple[float, float, float]:
    """Total proximity of obstacles in the left / centre / right corridor."""
    left = centre = right = 0.0
    for o in obstacles:
        cx = o["cx"]
        if cx < C_LEFT:
            left += o["prox"]
        elif cx <= C_RIGHT:
            centre += o["prox"]
        else:
            right += o["prox"]
    return left, centre, right


def _img_to_ground(H, pts: list) -> list:
    """Inverse of _proj: image pixels -> ground metres (x lateral, z ahead)."""
    import cv2
    import numpy as np
    if not pts:
        return []
    # np.linalg.inv gives float64; OpenCV's perspectiveTransform wants float32
    Hinv = np.linalg.inv(H).astype(np.float32)
    arr = np.array([[[float(p[0]), float(p[1])]] for p in pts], dtype=np.float32)
    out = cv2.perspectiveTransform(arr, Hinv)
    return [(float(p[0][0]), float(p[0][1])) for p in out]


def _ground_obstacles(H, obstacles: list, w: int, h: int, road_half: float = 1.8) -> list:
    """Where each obstacle actually sits on the road, in metres.

    The bottom edge of a box is where the object meets the ground, so we take
    the bottom-centre of the box and push it back through the road homography.
    """
    pixels = []
    for o in obstacles:
        bcx = (o["x"] + o["w"] / 2.0) * w
        bcy = (o["y"] + o["h"]) * h
        pixels.append((bcx, bcy))                       # contact point
        pixels.append((o["x"] * w, bcy))                # left foot
        pixels.append(((o["x"] + o["w"]) * w, bcy))     # right foot
    ground = _img_to_ground(H, pixels)

    out = []
    for i, o in enumerate(obstacles):
        cx, cz = ground[3 * i]
        lx, _ = ground[3 * i + 1]
        rx, _ = ground[3 * i + 2]
        if cz <= 0.2 or cz > 15.0:
            # behind us, or past the edge of the ground mapping: it cannot be
            # steered around, so it counts as off-road rather than vanishing
            o["on_road"] = False
            out.append({"class_name": o["class_name"], "x": cx, "z": cz,
                        "half": 0.3, "prox": o["prox"],
                        "track_id": o.get("track_id"), "on_road": False})
            continue
        on_road = _on_road(cx, cz, road_half)
        o["on_road"] = on_road                          # let the HUD style it too
        out.append({
            "class_name": o["class_name"],
            "x": cx, "z": cz,
            # clamp to a realistic half-width: the inverse projection can blow
            # up for boxes near the horizon and make everything look like a bus
            "half": min(1.6, max(0.18, abs(rx - lx) / 2.0)),
            "prox": o["prox"],
            "track_id": o.get("track_id"),
            # on the road path = allowed to influence the cue; off it = ignored
            "on_road": on_road,
        })
    return out


# The cue's own half-width on the road, and how far ahead we plan.
PATH_HALF = 0.5          # planning margin: how much room the cue needs to be safe
RIBBON_HALF = 0.3        # visual width of the painted ribbon (thinner reads better)
Z_START, Z_END = 1.2, 8.0
Z_MIN, Z_MAX = 0.8, 12.0

# ── The road path ─────────────────────────────────────────────────────────────
# Everything the cue cares about happens on the road ahead. This band of tarmac
# is the only place obstacles can influence the decision: a pedestrian on the
# pavement, a shop sign, a parked car on the verge must not steer the rider.
ROAD_HALF = 1.8                  # metres each side of the rider
ROAD_Z_NEAR, ROAD_Z_FAR = 0.4, 12.0


def _road_path_polygon(half: float = ROAD_HALF) -> list:
    """Ground polygon of the drivable path ahead, in (x, z) metres."""
    half = max(1.2, min(4.0, float(half)))
    return [(-half, ROAD_Z_NEAR), (half, ROAD_Z_NEAR),
            (half, ROAD_Z_FAR), (-half, ROAD_Z_FAR)]


def auto_calibrate(frame_bgr) -> dict:
    """Work out the camera knobs from the picture itself - no manual tuning.

    Finds the road surface (the low-saturation, low-texture region connected to
    the bottom-centre of the frame) and measures it:
      * road_bottom - nearest visible road row (bonnet/dash sits below it)
      * horizon     - where the road's two edges converge
      * near_l/r    - the road's left/right edges at the near (bottom) end
      * far_l/r     - the road's left/right edges at the horizon
      * road_half   - metres each side, from lane-marking spacing when we can
                      find it, otherwise a standard lane half-width prior

    Returns the knobs plus a confidence score, the basis used for road_half and
    the detected road quad in normalised coords so the UI can show it for a
    sanity check before the rider trusts it.
    """
    import cv2
    import numpy as np

    LANE_W = 3.5                    # metres: assumed lane width for scaling

    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    grey = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)

    # tarmac: washed-out colour, mid brightness, not much texture
    mask = ((S < 80) & (V > 30) & (V < 215) & (grad < 110)).astype(np.uint8) * 255
    mask[: int(0.28 * h), :] = 0                     # sky/skyline is never road
    k = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    # keep the blob touching the bottom-centre: that is the road we are on
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    votes = {}
    for y in range(int(0.72 * h), h, 3):
        for x in range(int(0.30 * w), int(0.70 * w), 4):
            lab = int(labels[y, x])
            if lab:
                votes[lab] = votes.get(lab, 0) + 1
    lab = max(votes, key=votes.get) if votes else 0
    if not lab and n > 1:
        lab = max(range(1, n), key=lambda i: stats[i, 4])

    fails = {"ok": False, "reason": "no road surface found", "horizon": 0.47,
             "road_bottom": 0.99, "road_half": 1.75, "near_l": 0.05, "near_r": 0.95,
             "far_l": 0.42, "far_r": 0.58, "confidence": 0.0, "basis": "default"}
    if not lab:
        return fails

    road = (labels == lab).astype(np.uint8)

    # Walk UP from the nearest road row, always following the run that contains
    # our own centre column. That keeps us on the road we are actually on and
    # stops the mask jumping onto a same-coloured wall or verge.
    ys = [y for y in range(h - 1, -1, -1) if road[y].sum() >= 4]
    if not ys:
        return fails
    y_near = ys[0]
    xs = np.where(road[y_near] > 0)[0]
    cx = int((int(xs.min()) + int(xs.max())) / 2)

    ext = []                                  # (y, x0, x1, width) near -> far
    for y in range(y_near, -1, -1):
        rowm = road[y]
        if rowm[cx] == 0:
            # lane markings are bright so they fall outside the tarmac mask and
            # punch holes in the run - hop over a small gap instead of quitting
            LO, HI = max(0, cx - int(0.025 * w)), min(w, cx + int(0.025 * w) + 1)
            sub = np.where(rowm[LO:HI] > 0)[0]
            if sub.size == 0:
                break
            cx = LO + int(sub[len(sub) // 2])
        x0 = cx
        while x0 > 0 and rowm[x0 - 1] > 0:
            x0 -= 1
        x1 = cx
        while x1 < w - 1 and rowm[x1 + 1] > 0:
            x1 += 1
        width = x1 - x0 + 1
        if width < 4:
            break
        ext.append((y, x0, x1, width))
        cx = int((x0 + x1) / 2)
    if len(ext) < 6:
        return fails

    near_row = ext[0]                         # closest visible road
    # far end: the last row (going away) that is still a believable road width.
    # Past that the road has converged into the horizon.
    far_row = near_row
    for r in ext:
        if r[3] >= max(10, 0.035 * w):
            far_row = r

    road_bottom = float(np.clip(near_row[0] / h, 0.30, 0.995))
    near_l, near_r = near_row[1] / w, near_row[2] / w
    far_l, far_r = far_row[1] / w, far_row[2] / w

    # horizon: just above where the road stops being wide enough to matter
    horizon = float(np.clip(far_row[0] / h - 0.015, 0.22, 0.62))
    horizon = min(horizon, road_bottom - 0.06)

    # road_half: scale from lane-marking spacing when lines are clearly visible
    band_top, band_bot = int(0.68 * h), int(0.96 * h)
    bright = ((V > 195) & (S < 60)).astype(np.uint8)
    bright[road == 0] = 0                    # markings must be ON the road
    cols = bright[band_top:band_bot].sum(axis=0)
    cand = np.where(cols > 0.10 * (band_bot - band_top))[0]
    clusters = []
    for c in cand:
        if clusters and c - clusters[-1][-1] <= 8:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    # keep only thin lines (a lane marking is narrow)
    centres = [sum(c) / len(c) for c in clusters if len(c) <= 0.05 * w]
    road_half, basis = 1.75, "standard lane prior"
    if len(centres) >= 2:
        gaps = [centres[i + 1] - centres[i] for i in range(len(centres) - 1)]
        gaps = [g for g in gaps if 0.10 * w < g < 0.60 * w]
        if gaps:
            gap = float(np.median(gaps))
            half_px = (near_row[2] - near_row[1]) / 2.0
            road_half = float(np.clip((half_px / gap) * (LANE_W / 2.0) * 2.0, 1.2, 4.0))
            basis = "lane markings"

    # confidence: how much of the lower frame the road explains
    cover = sum(r[3] for r in ext) / float(max(1, (h - int(0.28 * h)) * w))
    conf = float(np.clip(min(1.0, cover * 2.2), 0.0, 1.0))
    if basis == "standard lane prior":
        conf *= 0.8

    return {
        "ok": True,
        "horizon": round(horizon, 3),
        "road_bottom": round(road_bottom, 3),
        "road_half": round(road_half, 2),
        "near_l": round(float(np.clip(near_l, 0.0, 0.6)), 3),
        "near_r": round(float(np.clip(near_r, 0.4, 1.0)), 3),
        "far_l": round(float(np.clip(far_l, 0.0, 0.6)), 3),
        "far_r": round(float(np.clip(far_r, 0.4, 1.0)), 3),
        "confidence": round(conf, 2),
        "basis": basis,
    }


def _point_in_poly(x: float, z: float, poly: list) -> bool:
    """Ray-cast point-in-polygon test, poly as a list of (x, z)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, zi = poly[i]
        xj, zj = poly[j]
        dz = zj - zi
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (dz if dz else 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def _on_road(x: float, z: float, half: float = ROAD_HALF) -> bool:
    """Is this ground point on the road path we are steering along?"""
    return _point_in_poly(x, z, _road_path_polygon(half))

# ── Smoothness / safety tuning (research-backed) ──────────────────────────────
# A rider cannot throw their weight sideways instantly, so the cue must respect
# the same limits. Bounding velocity, acceleration AND jerk is what makes a
# motion feel smooth (Hogan 1984: smoothness is low jerk, the 3rd derivative).
DT            = 1.0 / 30.0     # assumed frame interval
MAX_LAT_VEL   = 2.5            # m/s    - how fast the cue may slide sideways
MAX_LAT_ACC   = 5.0            # m/s^2
MAX_LAT_JERK  = 45.0           # m/s^3
# A new non-urgent cue must hold for this many frames before it is shown, so the
# display cannot flicker between GO / LEFT / RIGHT (AR-HUD HMI guidance).
ACTION_DWELL  = 6
# Forward-collision-warning research puts useful lead time at about 3 seconds.
TTC_LEAD      = 3.0


class PathPlanner:
    """Jerk-limited aim point, flicker-free cue, and per-track ground memory.

    Three separate smoothing jobs:
      * smooth_target  - move the aim point with bounded vel/acc/jerk
      * commit_action  - only change the shown cue once it has settled
      * smooth_ground  - de-jitter obstacle positions and estimate closing speed
    """

    def __init__(self):
        self.x = 0.0            # aim point (metres lateral)
        self.v = 0.0            # its velocity
        self.a = 0.0            # its acceleration
        self.action = "straight"
        self.pending = "straight"
        self.pending_n = 0
        self.tracks = {}        # track_id -> last ground position

    def smooth_ground(self, ground: list, dt: float = DT) -> list:
        """Ease obstacle positions and work out how fast each one is closing.

        Positions are filtered per track id so a jittery detector does not make
        the plan twitch. `vz` is the rate of change of distance ahead of us, so
        a negative value means the obstacle is coming towards the rider.
        """
        out = []
        for o in ground:
            tid = o.get("track_id")
            if tid is None:
                out.append(o)
                continue
            prev = self.tracks.get(tid)
            if prev is None:
                sx, sz, vz = o["x"], o["z"], 0.0
            else:
                alpha = 0.5                                  # gentle de-jitter
                sx = prev["x"] + alpha * (o["x"] - prev["x"])
                sz = prev["z"] + alpha * (o["z"] - prev["z"])
                vz = (o["z"] - prev["z"]) / dt
            self.tracks[tid] = {"x": sx, "z": sz}
            oo = dict(o)
            oo["x"], oo["z"] = sx, sz
            closing = max(0.0, -vz)                          # m/s towards us
            oo["ttc"] = round(sz / closing, 2) if closing > 0.05 else None
            out.append(oo)
        if len(self.tracks) > 300:                           # forget stale ids
            self.tracks.clear()
        return out

    def smooth_target(self, target: float, dt: float = DT) -> float:
        """Slide the aim point toward `target` within vel / acc / jerk limits."""
        kp, kd = 12.0, 6.0
        a_des = kp * (target - self.x) - kd * self.v          # PD pull
        a_des = max(-MAX_LAT_ACC, min(MAX_LAT_ACC, a_des))
        step = MAX_LAT_JERK * dt
        self.a += max(-step, min(step, a_des - self.a))       # jerk limit
        self.v += self.a * dt
        self.v = max(-MAX_LAT_VEL, min(MAX_LAT_VEL, self.v))  # velocity limit
        self.x += self.v * dt
        return round(self.x, 3)

    def commit_action(self, action: str, dwell: int = ACTION_DWELL) -> str:
        """De-flicker: hold a new cue until it has been asked for consistently.

        STOP is exempt - a safety-critical cue must never be delayed.
        """
        if action == "stop":
            self.action = self.pending = action
            self.pending_n = 0
            return self.action
        if action == self.action:
            self.pending, self.pending_n = action, 0
            return self.action
        if action == self.pending:
            self.pending_n += 1
        else:
            self.pending, self.pending_n = action, 1
        if self.pending_n >= dwell:
            self.action = action
        return self.action


def _clearance(target: float, obstacles: list) -> float:
    """Smallest lateral gap between the cue path and any obstacle footprint.

    Negative means the path would clip the obstacle; positive means it passes
    cleanly with that many metres to spare.
    """
    worst = 9.9
    for o in obstacles:
        gap = abs(target - o["x"]) - (o["half"] + PATH_HALF)
        worst = min(worst, gap)
    return worst


def _plan_path(obstacles: list) -> tuple:
    """Choose the aim point that steers around the obstacles most comfortably.

    Returns (target_x, used_obstacles, fully_blocked).
    """
    relevant = [o for o in obstacles if Z_MIN <= o["z"] <= Z_MAX]
    if not relevant:
        return 0.0, [], False

    # Straight ahead first: if the centre is already clear, hold the line.
    if _clearance(0.0, relevant) >= 0.15:
        return 0.0, relevant, False

    # Otherwise sweep candidate offsets and take the one with the best
    # clearance, with a small pull back toward the centre so we only veer
    # as far as we need to. Candidates stay inside the road width, so the cue
    # can never tell the rider to ride off the tarmac.
    best, best_score = 0.0, -99.0
    for i in range(-12, 13):
        t = i * 0.2                      # -2.4 .. +2.4 m (road width)
        score = _clearance(t, relevant) - 0.10 * abs(t)
        if score > best_score:
            best, best_score = t, score

    # Only call it blocked when nothing close to a gap exists. A slightly tight
    # squeeze is still offered as a "least bad" swerve rather than a hard stop.
    fully_blocked = _clearance(best, relevant) < -0.35
    return round(best, 3), relevant, fully_blocked


def _build_path(target: float) -> list:
    """A curvature-continuous ground path out to the aim point.

    Uses a minimum-jerk (quintic) blend, so the path leaves straight and arrives
    straight with zero curvature at both ends - no kink where it starts or stops
    bending, which is the same reason road designers use clothoids.
    """
    pts = []
    for i in range(9):
        t = i / 8.0
        z = Z_START + t * (Z_END - Z_START)
        ease = 10 * t**3 - 15 * t**4 + 6 * t**5      # minimum jerk
        pts.append((target * ease, z))
    return pts


def decide(objects: list, image_h: int, w: int | None = None, h: int | None = None,
           planner: "PathPlanner | None" = None, horizon: float = 0.47,
           road_half: float = ROAD_HALF, road_bottom: float = 0.99) -> dict:
    """Turn detections into one steering instruction, routed around obstacles."""
    obstacles = []
    for o in objects:
        bottom = o["y"] + o["h"]
        if bottom < PATH_TOP:
            continue
        if bottom > road_bottom:
            # below the road-start line: bonnet, dashboard or handlebars
            continue
        area = o["w"] * o["h"]
        prox_area = min(1.0, area / AREA_VERY_CLOSE)
        prox_low = max(0.0, (bottom - PATH_TOP) / (1.0 - PATH_TOP))
        prox = round(min(1.0, 0.7 * prox_area + 0.3 * prox_low), 3)
        obstacles.append({
            "class_name": o["class_name"],
            "cx": o["x"] + o["w"] / 2.0,
            "x": o["x"], "y": o["y"], "w": o["w"], "h": o["h"],
            "prox": prox,
            "conf": o.get("conf", 0),
            "track_id": o.get("track_id"),
        })

    left, centre, right = _corridor_blocks(obstacles)
    closest = max((o["prox"] for o in obstacles), default=0.0)

    # ── plan the avoidance path on the road ───────────────────────────────────
    if w and h:
        H = _ground_homography(w, h, horizon, road_half, road_bottom)
        ground = _ground_obstacles(H, obstacles, w, h, road_half)
        if planner is not None:
            ground = planner.smooth_ground(ground)   # de-jitter + closing speed
        # The cue only answers to obstacles ON the road path. Anything on the
        # pavement or the verge is scenery, not something to steer around.
        on_road = [o for o in ground if o.get("on_road")]
        wanted, relevant, blocked = _plan_path(on_road)
    else:
        ground, on_road = [], []
        wanted, relevant, blocked = 0.0, [], False
    off_road = len(ground) - len(on_road)

    # ease the aim point within vel/acc/jerk limits
    target = planner.smooth_target(wanted) if planner is not None else wanted

    if blocked:
        raw_action = "stop"
    elif not relevant:
        # nothing in the path at all: the cue is GO even while the ribbon
        # glides back to the centre from a previous swerve
        raw_action = "straight"
    elif target <= -0.45:
        raw_action = "left"
    elif target >= 0.45:
        raw_action = "right"
    else:
        raw_action = "straight"

    # A clear path is shown immediately: holding a stale KEEP LEFT after the road
    # has opened up again is worse than a little chatter. STOP also passes
    # through at once. Only the directional cues are de-flickered.
    if planner is not None and not relevant and not blocked:
        planner.action = planner.pending = "straight"
        planner.pending_n = 0
        action = "straight"
    else:
        action = planner.commit_action(raw_action) if planner is not None else raw_action

    # Urgency only counts obstacles actually in the rider's path. A pedestrian
    # safely off to the side must not raise the alarm.
    in_path = [o for o in relevant if abs(o["x"]) < 1.4]
    ttcs = [o["ttc"] for o in in_path if o.get("ttc")]
    if ttcs:
        urgency = round(min(1.0, max(0.0, 1.0 - min(ttcs) / TTC_LEAD)), 3)
    else:
        urgency = round(min(1.0, max((o["prox"] for o in in_path), default=0.0)), 3)

    hud = {
        "straight": {"arrow": 0,    "text": "GO",         "colour": "#22c55e"},
        "left":     {"arrow": -90,  "text": "KEEP LEFT",  "colour": "#f59e0b"},
        "right":    {"arrow": 90,   "text": "KEEP RIGHT", "colour": "#f59e0b"},
        "stop":     {"arrow": None, "text": "STOP",       "colour": "#ef4444"},
    }[action]

    reasons = []
    if blocked:
        reasons.append("no clear gap between obstacles")
    elif action == "straight":
        reasons.append("road path clear")
    else:
        gap = _clearance(target, relevant)
        reasons.append(f"steering {target:+.1f} m for {gap:.1f} m clearance")
    if off_road:
        reasons.append(f"{off_road} object(s) off the road path ignored")

    return {
        "action": action,
        "target": target,
        "path": _build_path(target),
        "urgency": urgency,
        "hud": hud,
        "corridors": {"left": round(left, 3), "centre": round(centre, 3), "right": round(right, 3)},
        "obstacles": sorted(obstacles, key=lambda o: -o["prox"]),
        "reasons": reasons,
        "centre_clear": not relevant,
        "on_road_count": len(on_road),
        "off_road_count": off_road,
    }


# ── Drawing ───────────────────────────────────────────────────────────────────

def _hex_to_bgr(h: str) -> tuple:
    """'#f59e0b' -> (B, G, R) because OpenCV wants BGR, not RGB."""
    r = int(h[1:3], 16); g = int(h[3:5], 16); b = int(h[5:7], 16)
    return (b, g, r)


def _bezier(p0, p1, p2, p3, n=40):
    """Sample a cubic bezier so we can draw a smooth curved arrow."""
    pts = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        x = (u**3) * p0[0] + 3 * (u**2) * t * p1[0] + 3 * u * (t**2) * p2[0] + (t**3) * p3[0]
        y = (u**3) * p0[1] + 3 * (u**2) * t * p1[1] + 3 * u * (t**2) * p2[1] + (t**3) * p3[1]
        pts.append((int(x), int(y)))
    return pts


def _ground_homography(w: int, h: int, horizon: float = 0.47, road_half: float = 2.6,
                       road_bottom: float = 0.99,
                       near_l: float = 0.02, near_r: float = 0.98,
                       far_l: float = 0.40, far_r: float = 0.60):
    """Map a flat rectangle of road (in metres) onto the image.

    A rectangle on the ground shows up as a trapezoid in a forward-facing
    camera, so we build the homography from that trapezoid. Anything defined in
    ground coordinates then lands on the road with correct perspective - narrow
    far away, wide up close. That is what makes the cue look painted on.

    `horizon`     - where the road meets the sky (fraction of image height).
    `road_half`   - how many metres each side of the rider the near edge covers.
    `road_bottom` - where the nearest visible road starts. On a helmet cam that
                    is the bottom of the frame, but a dashcam looking over a
                    bonnet (or a bike over the handlebars) has the vehicle body
                    filling the lower part, so the road only begins higher up.
    `near_l/r`    - image x fractions of the road's near (bottom) edge.
    `far_l/r`     - image x fractions of the road's far edge.

    All of these are normally produced by auto_calibrate(); they stay overridable
    so an installer can nudge them by hand.
    """
    import cv2
    import numpy as np

    horizon = min(0.75, max(0.20, float(horizon)))
    road_bottom = max(horizon + 0.05, min(0.995, float(road_bottom)))
    z_near, z_far = 0.6, 9.0
    near_w = float(road_half)
    # converge the far edge so the trapezoid matches a realistic lens
    far_w = max(0.25, near_w * 0.30)

    world = np.float32([[-near_w, z_near], [near_w, z_near],
                        [far_w, z_far], [-far_w, z_far]])
    # near edge rides on the road-start line, far edge on the horizon line
    img = np.float32([[near_l * w, road_bottom * h], [near_r * w, road_bottom * h],
                      [far_r * w, horizon * h], [far_l * w, horizon * h]])
    return cv2.getPerspectiveTransform(world, img)


def _proj(H, pts: list) -> list:
    """Project ground points (x, z) into image pixels through the homography."""
    import cv2
    import numpy as np
    arr = np.array([[[float(p[0]), float(p[1])]] for p in pts], dtype=np.float32)
    out = cv2.perspectiveTransform(arr, H)
    return [(int(round(p[0][0])), int(round(p[0][1]))) for p in out]


def _ground_band(center: list, half_w: float) -> list:
    """Widen a centreline into a ribbon of constant ground width."""
    import math
    left, right = [], []
    n = len(center)
    for i, (x, z) in enumerate(center):
        if i == 0:
            dx, dz = center[1][0] - x, center[1][1] - z
        elif i == n - 1:
            dx, dz = x - center[i - 1][0], z - center[i - 1][1]
        else:
            dx, dz = center[i + 1][0] - center[i - 1][0], center[i + 1][1] - center[i - 1][1]
        L = math.hypot(dx, dz) or 1.0
        nx, nz = -dz / L, dx / L
        left.append((x + nx * half_w, z + nz * half_w))
        right.append((x - nx * half_w, z - nz * half_w))
    return left + right[::-1]


def _image_arrowhead(tip: tuple, prev: tuple, half_px: float) -> list:
    """Build a triangle head in image pixels so it always looks like an arrow.

    Drawing the head in ground space fights perspective (depth compresses much
    harder than width, which flattens it into a bar). Sizing it in pixels keeps
    the shaft perspective-correct while the head stays a clean pointed triangle.
    """
    import math
    dx, dy = tip[0] - prev[0], tip[1] - prev[1]
    L = math.hypot(dx, dy) or 1.0
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    head_half = max(9.0, half_px * 1.6)
    head_len = max(14.0, half_px * 2.8)
    base = (tip[0] - ux * head_len, tip[1] - uy * head_len)
    return [
        (int(tip[0]), int(tip[1])),
        (int(base[0] + px * head_half), int(base[1] + py * head_half)),
        (int(base[0] - px * head_half), int(base[1] - py * head_half)),
    ]


def _draw_ground_arrow(frame, action: str, path: list, colour_rgb: str, w: int, h: int,
                       horizon: float = 0.47, road_bottom: float = 0.99):
    """Paint the steering cue flat on the road, following the planned path.

    `path` is a list of ground points (x metres, z metres) worked out by the
    planner, so the ribbon bends around whatever is actually in the way. A
    constant width in metres means perspective narrows it toward the horizon
    by itself, which is what makes it read as lying on the asphalt.
    """
    import cv2
    import numpy as np

    colour = _hex_to_bgr(colour_rgb)
    H = _ground_homography(w, h, horizon, RIBBON_HALF * 8.6, road_bottom)

    if action == "stop":
        # a barrier band across the lane, also on the ground
        band = [(-2.2, 3.0), (2.2, 3.0), (2.2, 3.9), (-2.2, 3.9)]
        poly = np.array([_proj(H, band)], dtype=np.int32)
        overlay = frame.copy()
        cv2.fillPoly(overlay, poly, colour)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.polylines(frame, poly, True, (255, 255, 255), 3, cv2.LINE_AA)
        return frame

    if not path:
        return frame

    half_w = RIBBON_HALF
    ribbon = _proj(H, _ground_band(path, half_w))

    poly = np.array([ribbon], dtype=np.int32)
    overlay = frame.copy()
    cv2.fillPoly(overlay, poly, colour)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # crisp edge so it stays readable on a busy road
    cv2.polylines(frame, poly, True, (255, 255, 255), 2, cv2.LINE_AA)

    # arrow head sized in pixels so it always reads as a pointed triangle
    tip = _proj(H, [path[-1]])[0]
    prev = _proj(H, [path[-2]])[0]
    cx, cz = path[-1]
    left_px = _proj(H, [(cx - half_w, cz)])[0]
    right_px = _proj(H, [(cx + half_w, cz)])[0]
    half_px = max(4.0, abs(right_px[0] - left_px[0]) / 2.0)

    head_poly = np.array([_image_arrowhead(tip, prev, half_px)], dtype=np.int32)
    cv2.fillPoly(frame, head_poly, colour)
    cv2.polylines(frame, head_poly, True, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _draw_hud(frame, result, w, h, horizon: float = 0.47, road_bottom: float = 0.99):
    """Draw the road path, obstacle boxes, the on-road arrow and the cue bar."""
    import cv2
    import numpy as np

    # corridor guide lines (where the rider is heading)
    for x in (int(C_LEFT * w), int(C_RIGHT * w)):
        cv2.line(frame, (x, int(PATH_TOP * h)), (x, h), (90, 90, 90), 1, cv2.LINE_AA)
    cv2.line(frame, (0, int(PATH_TOP * h)), (w, int(PATH_TOP * h)), (90, 90, 90), 1, cv2.LINE_AA)

    # road path region: the only part of the scene the cue answers to
    road = np.array([_proj(_ground_homography(w, h, horizon, 1.8, road_bottom),
                           _road_path_polygon())], dtype=np.int32)
    road_overlay = frame.copy()
    cv2.fillPoly(road_overlay, road, (70, 70, 70))
    cv2.addWeighted(road_overlay, 0.16, frame, 0.84, 0, frame)
    cv2.polylines(frame, road, True, (120, 120, 120), 1, cv2.LINE_AA)

    # obstacle boxes: on-road ones get colour, off-road ones are muted
    for o in result["obstacles"]:
        x1 = int(o["x"] * w); y1 = int(o["y"] * h)
        x2 = int((o["x"] + o["w"]) * w); y2 = int((o["y"] + o["h"]) * h)
        if o.get("on_road", True):
            col = (60, 60, 240) if o["prox"] >= DANGER else (60, 170, 240)
            label = f"{o['class_name']} {o['prox']:.2f}"
        else:
            col = (120, 120, 120)               # off the road path: ignored
            label = f"{o['class_name']} (off road)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
        cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # the cue, painted on the road along the planned path
    _draw_ground_arrow(frame, result["action"], result.get("path") or [],
                       result["hud"]["colour"], w, h, horizon, road_bottom)

    # cue bar on top
    hud = result["hud"]
    colour = _hex_to_bgr(hud["colour"])
    bar_h = 46
    cv2.rectangle(frame, (0, 0), (w, bar_h), (20, 20, 24), -1)
    cv2.putText(frame,
                f"on road {result.get('on_road_count', 0)}   off road {result.get('off_road_count', 0)}"
                f"   urgency {result['urgency']:.2f}",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
    cv2.putText(frame, hud["text"], (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, colour, 2, cv2.LINE_AA)
    return frame


def _analyse_fast(model, rgb, class_names, conf, imgsz):
    """Detection without tracking - the cheapest path for the realtime loop.

    Tracking adds ~7 ms/frame; for live use we drop it, which means no per-track
    de-jitter or time-to-collision. The planner's own smoothing still applies.
    """
    from routers.road_safety import _analyse_frame
    return _analyse_frame(model, rgb, class_names, conf, 0.45, imgsz, False, False)


# ── Live (realtime) mode ──────────────────────────────────────────────────────
# For a helmet device the cue has to arrive fast enough to act on. Two levers
# make that work: run detection on a smaller frame, and if needed detect only
# every Nth frame while the planner keeps easing the cue on every single frame.
_rt_stats: dict = {}
_rt_lock = __import__("threading").Lock()


@router.get("/projects/{project_id}/guidance/live/stats")
def guidance_live_stats(project_id: int):
    """Latest measured FPS / latency and the current cue from the live loop."""
    return _rt_stats.get(project_id, {"running": False})


@router.get("/projects/{project_id}/guidance/live")
def guidance_live(
    project_id: int,
    source: str = "0",              # webcam index, or a video file path
    run_id: int | None = None,
    conf: float = 0.3,
    imgsz: int = 320,
    horizon: float = 0.47,
    road_half: float = 1.8,
    road_bottom: float = 0.99,
    every: int = 1,                 # run detection every Nth frame
    session: Session = Depends(get_session),
):
    """Realtime MJPEG stream: the steering cue drawn live on every frame.

    Open it straight in a browser (`<img src=...>`) or on the helmet device.
    `/guidance/live/stats` reports the achieved FPS and latency.
    """
    import cv2
    from fastapi.responses import StreamingResponse
    from routers.road_safety import _load_model_and_classes

    model, class_names, note = _load_model_and_classes(project_id, run_id, session)
    src = int(source) if source.isdigit() else source
    every = max(1, int(every))

    def gen():
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            return
        is_file = not str(source).isdigit()
        planner = PathPlanner()
        last_res = None
        i = 0
        t_prev = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_file:                       # loop a file so it keeps streaming
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            h, w = frame.shape[:2]
            # detection runs on a smaller working copy: big speed win, and the
            # cue only needs the road layout, not full resolution detail
            if w > 640:
                work = cv2.resize(frame, (640, int(round(640 * h / w))), interpolation=cv2.INTER_AREA)
            else:
                work = frame
            if i % every == 0:
                rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
                with _rt_lock:
                    objects, _ = _analyse_fast(model, rgb, class_names, conf, imgsz)
                last_res = decide(objects, h, w=w, h=h, planner=planner,
                                  horizon=horizon, road_half=road_half,
                                  road_bottom=road_bottom)
            if last_res is None:
                _, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            else:
                drawn = _draw_hud(frame, last_res, w, h, horizon)
                _, jpg = cv2.imencode(".jpg", drawn, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

            now = time.time()
            dt = max(1e-6, now - t_prev)
            t_prev = now
            _rt_stats[project_id] = {
                "running": True, "fps": round(1.0 / dt, 1),
                "latency_ms": int(dt * 1000), "frame": i,
                "action": last_res["action"] if last_res else None,
                "urgency": last_res["urgency"] if last_res else None,
                "target": last_res["target"] if last_res else None,
                "detect_every": every, "imgsz": imgsz, "model": note,
            }
            i += 1
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
        cap.release()
        _rt_stats[project_id] = {"running": False}

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

def _run_guidance(model, arr, class_names, conf, imgsz, planner=None,
                  horizon: float = 0.47, road_half: float = ROAD_HALF,
                  road_bottom: float = 0.99):
    """Detect road users in a frame and return the guidance decision.

    Tracking is on so every obstacle keeps a stable id between frames - that id
    is what lets the planner de-jitter its position and estimate closing speed.
    """
    from routers.road_safety import _analyse_frame
    objects, _ = _analyse_frame(model, arr, class_names, conf, 0.45, imgsz, False, True)
    h, w = arr.shape[:2]
    return decide(objects, h, w=w, h=h, planner=planner, horizon=horizon,
                  road_half=road_half, road_bottom=road_bottom)


@router.post("/projects/{project_id}/guidance/calibrate")
async def guidance_calibrate(
    project_id: int,
    file: UploadFile = File(...),
):
    """Auto-calibrate the camera knobs from a picture (or a video's mid frame).

    No model needed - this is geometry. Feed it a typical frame from the camera
    and it returns horizon / road_bottom / road_half plus the road quad it found
    and a confidence score, so the installer can accept or nudge the values.
    """
    import cv2
    import numpy as np

    name = (file.filename or "frame.jpg").lower()
    data = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(name)[1] or ".jpg")
    try:
        tmp.write(data)
        tmp.close()
        arr = None
        if name.endswith((".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v")):
            cap = cv2.VideoCapture(tmp.name)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            picks = [int(total * f) for f in (0.3, 0.45, 0.6, 0.75, 0.9)] if total else [0]
            best, best_conf = None, -1.0
            for p in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, p))
                ok, fr = cap.read()
                if not ok:
                    continue
                cand = auto_calibrate(fr)
                if cand.get("ok") and cand.get("confidence", 0) > best_conf:
                    best, best_conf = cand, cand.get("confidence", 0)
                    best["frame_at"] = round(p / max(1, total), 2)
            cap.release()
            if best is not None:
                best["source"] = name
                return best
            arr = None
        else:
            img = cv2.imread(tmp.name)
            arr = img
        if arr is None:
            raise HTTPException(status_code=400, detail="Could not read that image or video")
        result = auto_calibrate(arr)
        result["source"] = name
        return result
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@router.post("/projects/{project_id}/guidance/analyze")
async def guidance_analyze(
    project_id: int,
    file: UploadFile = File(...),
    run_id: int | None = Form(None),
    conf: float = Form(0.3),
    imgsz: int = Form(640),
    horizon: float = Form(0.47),
    road_half: float = Form(1.8),
    road_bottom: float = Form(0.99),
    session: Session = Depends(get_session),
):
    """Single frame: returns the steering cue for a helmet/dash camera view."""
    from routers.road_safety import _load_model_and_classes

    model, class_names, note = _load_model_and_classes(project_id, run_id, session)

    suffix = Path(file.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        import numpy as np
        from PIL import Image as PILImage
        arr = np.array(PILImage.open(tmp_path).convert("RGB"))
        t0 = time.time()
        result = _run_guidance(model, arr, class_names, conf, imgsz,
                               horizon=horizon, road_half=road_half, road_bottom=road_bottom)
        result["meta"] = {"model": note, "horizon": horizon, "road_half": road_half,
                          "time_ms": int((time.time() - t0) * 1000)}
        return result
    finally:
        os.unlink(tmp_path)


def _process_guidance_video(job_id, in_path, model, class_names, conf, imgsz,
                            horizon=0.47, road_half=ROAD_HALF, road_bottom=0.99):
    job = _jobs[job_id]
    try:
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(in_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        job["total_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_path = os.path.join(VIDEO_OUT_DIR, f"guide_{job_id}.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        tally = {"straight": 0, "left": 0, "right": 0, "stop": 0}
        worst = 0.0
        events = []
        planner = PathPlanner()          # keeps the cue smooth across frames
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = _run_guidance(model, rgb, class_names, conf, imgsz, planner,
                                horizon, road_half, road_bottom)
            tally[res["action"]] += 1
            worst = max(worst, res["urgency"])
            if res["action"] in ("left", "right", "stop") and len(events) < 40:
                events.append({"frame": i, "action": res["action"],
                               "urgency": res["urgency"], "reasons": res["reasons"]})
            writer.write(_draw_hud(frame, res, w, h, horizon, road_bottom))
            i += 1
            job["processed"] = i
        cap.release()
        writer.release()
        job["out_path"] = out_path
        job["status"] = "done"
        job["summary"] = {"frames": i, "tally": tally,
                          "worst_urgency": round(worst, 3),
                          "events": events[:15]}
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    finally:
        if os.path.exists(in_path):
            try:
                os.unlink(in_path)
            except OSError:
                pass


@router.post("/projects/{project_id}/guidance/analyze-video")
async def guidance_analyze_video(
    project_id: int,
    file: UploadFile = File(...),
    run_id: int | None = Form(None),
    conf: float = Form(0.3),
    imgsz: int = Form(640),
    horizon: float = Form(0.47),
    road_half: float = Form(1.8),
    road_bottom: float = Form(0.99),
    session: Session = Depends(get_session),
):
    """Video: per-frame steering cue drawn on each frame, plus a summary."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        raise HTTPException(500, "OpenCV (cv2) is not installed on this server.")

    from routers.road_safety import _load_model_and_classes
    model, class_names, note = _load_model_and_classes(project_id, run_id, session)

    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    in_path = os.path.join(VIDEO_OUT_DIR, f"guide_in_{uuid.uuid4().hex}{suffix}")
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex
    _jobs[job_id] = {"status": "running", "processed": 0, "total_frames": 0,
                     "out_path": None, "error": None, "model": note}
    threading.Thread(target=_process_guidance_video,
                     args=(job_id, in_path, model, class_names, conf, imgsz, horizon, road_half, road_bottom),
                     daemon=True).start()
    return {"job_id": job_id, "model": note}


@router.get("/projects/{project_id}/guidance/video/{job_id}/status")
def guidance_status(project_id: int, job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"status": job["status"], "processed": job["processed"],
            "total_frames": job["total_frames"], "error": job.get("error"),
            "summary": job.get("summary")}


@router.get("/projects/{project_id}/guidance/video/{job_id}/download")
def guidance_download(project_id: int, job_id: str):
    job = _jobs.get(job_id)
    if not job or not job.get("out_path") or not os.path.exists(job["out_path"]):
        raise HTTPException(404, "Result not ready")
    return FileResponse(job["out_path"], media_type="video/mp4",
                        filename=f"guidance_{job_id}.mp4")
