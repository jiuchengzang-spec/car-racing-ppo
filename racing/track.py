"""Procedural closed-loop race track, authored in *curvature space*.

Instead of scattering (x, y) control points, a lap is written as an alternating
sequence of straights (curvature 0) and circular corners (curvature ±1/radius).
Integrating curvature → heading → position turns that sequence into a centreline,
which is then offset by half the track width to get the left/right boundaries.
Corner signs are picked so the loop nets exactly one full turn (+2π) while still
mixing left and right; with the corners fixed, position is *linear* in the
straight lengths, so closing the loop is a tiny linear solve. ``TRACK_PROFILES``
sets the "character" (corner count, sharpness, straight length); the seed rolls
the dice within those rules. Deterministic for a given (seed, profile), so the
human game and the RL agent always race the same circuit.

The track also answers the questions the env needs every tick:
  * ``project`` — where am I along the lap, and how far off the centre?
  * ``cast_ray`` — how far to the nearest wall along a heading (the car's sensors)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Projection:
    s: float  # arc-length progress along the centreline (m)
    lateral: float  # signed offset from centreline (m); + is left of travel dir
    heading: float  # centreline tangent heading at the projection (rad)
    segment: int  # index of the nearest centreline segment


class Track:
    def __init__(
        self,
        centerline: np.ndarray,
        width: float,
    ) -> None:
        self.centerline = centerline  # (N, 2), first point != last (open ring)
        self.width = width
        self.half = width / 2.0

        # Closed-loop segment geometry (wrap last->first).
        nxt = np.roll(centerline, -1, axis=0)
        self._seg_vec = nxt - centerline  # (N, 2)
        seg_len = np.hypot(self._seg_vec[:, 0], self._seg_vec[:, 1])
        seg_len = np.where(seg_len < 1e-9, 1e-9, seg_len)
        self._seg_len = seg_len
        self._seg_len2 = seg_len * seg_len
        self.length = float(seg_len.sum())
        # Cumulative arc length at the start of each segment.
        self._s0 = np.concatenate([[0.0], np.cumsum(seg_len)[:-1]])
        # Unit normals (left of travel direction): rotate tangent +90deg.
        tangents = self._seg_vec / seg_len[:, None]
        self._normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        self._tangents = tangents

        # Per-segment curvature (turn rate, rad/m): how fast the heading swings.
        # Near zero on straights, large in tight corners. The renderer uses it to
        # kerb the corners and just line the straights.
        ang = np.arctan2(tangents[:, 1], tangents[:, 0])
        dang = (np.roll(ang, -1) - ang + math.pi) % (2.0 * math.pi) - math.pi
        self.curvature = np.abs(dang) / seg_len

        self.left = centerline + self._normals * self.half
        self.right = centerline - self._normals * self.half
        # Boundary segments as (P, Q) arrays for ray casting (closed loops).
        self._walls_p = np.concatenate([self.left, self.right], axis=0)
        self._walls_q = np.concatenate(
            [np.roll(self.left, -1, axis=0), np.roll(self.right, -1, axis=0)], axis=0
        )

    @property
    def start_pose(self) -> tuple[float, float, float]:
        x, y = self.centerline[0]
        return float(x), float(y), float(math.atan2(*self._tangents[0][::-1]))

    def pose_at(self, s: float) -> tuple[float, float, float]:
        """Centreline pose (x, y, yaw) at arc-length ``s`` (wrapped to the lap)."""
        s = s % self.length
        i = int(np.searchsorted(self._s0, s, side="right") - 1)
        i = max(0, min(i, len(self._s0) - 1))
        frac = (s - self._s0[i]) / self._seg_len[i]
        p = self.centerline[i] + self._seg_vec[i] * frac
        return float(p[0]), float(p[1]), float(math.atan2(self._tangents[i, 1], self._tangents[i, 0]))

    def project(self, x: float, y: float) -> Projection:
        """Nearest point on the centreline to (x, y), vectorised over segments."""
        p = np.array([x, y], dtype=np.float64)
        ap = p[None, :] - self.centerline  # (N, 2)
        t = np.einsum("ij,ij->i", ap, self._seg_vec) / self._seg_len2
        t = np.clip(t, 0.0, 1.0)
        proj = self.centerline + self._seg_vec * t[:, None]
        d2 = np.einsum("ij,ij->i", p - proj, p - proj)
        i = int(np.argmin(d2))

        s = float(self._s0[i] + t[i] * self._seg_len[i])
        rel = p - proj[i]
        lateral = float(np.dot(rel, self._normals[i]))
        heading = float(math.atan2(self._tangents[i, 1], self._tangents[i, 0]))
        return Projection(s=s, lateral=lateral, heading=heading, segment=i)

    def is_inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        return abs(self.project(x, y).lateral) <= self.half + margin

    def cast_ray(self, x: float, y: float, angle: float, max_dist: float) -> float:
        """Distance from (x, y) along ``angle`` to the nearest wall, capped."""
        o = np.array([x, y], dtype=np.float64)
        d = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)

        p = self._walls_p
        q = self._walls_q
        e = q - p  # wall direction vectors (M, 2)
        # Solve o + t d = p + u e  for each wall; t>=0, 0<=u<=1.
        denom = d[0] * e[:, 1] - d[1] * e[:, 0]
        safe = np.abs(denom) > 1e-12
        diff = p - o  # (M, 2)
        t = np.full(p.shape[0], np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_all = (diff[:, 0] * e[:, 1] - diff[:, 1] * e[:, 0]) / denom
            u_all = (diff[:, 0] * d[1] - diff[:, 1] * d[0]) / denom
        valid = safe & (t_all >= 0.0) & (u_all >= 0.0) & (u_all <= 1.0)
        t[valid] = t_all[valid]
        hit = float(t.min()) if np.any(valid) else max_dist
        return min(hit, max_dist)


# Track "character" profiles. Each sets the *rules of the dice*; the seed rolls
# them. Fields: n = corner count range; angle = per-corner turn range (deg);
# radius = corner radius range (m, small = slow hairpin, large = fast sweep);
# straight = base straight-length range (m); longs = how many straights to
# stretch into flat-out runs (each gets the tightest following corner -> a heavy
# braking zone). Balance (left vs right) is a *consequence* of corner count: a
# closed loop must net one full turn (+2pi), so only corner-rich tracks approach
# 50/50 — few-corner "power" tracks are naturally more one-directional.
TRACK_PROFILES: dict[str, dict] = {
    "balanced":  dict(n=(16, 22), angle=(30, 95),  radius=(35, 150), straight=(45, 140), longs=(1, 2)),
    "power":     dict(n=(6, 9),   angle=(50, 150), radius=(30, 110), straight=(95, 250), longs=(2, 3)),
    "flowing":   dict(n=(9, 13),  angle=(28, 95),  radius=(90, 230), straight=(55, 150), longs=(1, 2)),
    "technical": dict(n=(15, 21), angle=(38, 140), radius=(22, 80),  straight=(26, 80),  longs=(0, 1)),
}


def make_track(
    seed: int = 0,
    profile: str = "balanced",
    width: float = 16.0,
) -> Track:
    """Build a closed circuit in *curvature space* (see the module docstring).

    Rather than scatter (x, y) points, we author the lap as an alternating
    sequence of straights (curvature 0) and circular corners (curvature
    ±1/radius), then integrate curvature → heading → position. Corner signs are
    chosen so the total turning is exactly one lap (+2π) while still mixing left
    and right — something the old radial-star generator structurally could not
    do. With the corners fixed, position is *linear* in the straight lengths, so
    closing the loop is a tiny linear solve (no fragile optimiser). Draws that
    self-intersect or fall out of size bounds are rejected and resampled, so the
    result is always a clean, drivable loop. Deterministic for a given
    (seed, profile).
    """
    if profile not in TRACK_PROFILES:
        raise ValueError(
            f"unknown track profile {profile!r}; choose from {sorted(TRACK_PROFILES)}"
        )
    spec = TRACK_PROFILES[profile]
    rng = np.random.default_rng(seed)
    for _ in range(400):  # rejection sampling for a clean closed loop
        cl = _try_curvature_track(rng, spec)
        if cl is not None:
            return Track(centerline=_roll_to_straight(cl), width=width)
    raise RuntimeError(
        f"could not generate a clean {profile!r} track for seed {seed}"
    )


def _try_curvature_track(rng: np.random.Generator, spec: dict) -> np.ndarray | None:
    """One attempt at a closed curvature-space loop; None if it isn't clean."""
    n = int(rng.integers(spec["n"][0], spec["n"][1] + 1))
    amin, amax = math.radians(spec["angle"][0]), math.radians(spec["angle"][1])
    mag = rng.uniform(amin, amax, n)  # per-corner turn magnitudes (rad)
    total = float(mag.sum())
    if total < 2.0 * math.pi + math.radians(15):
        return None  # not enough turning to close a lap

    # Assign right-hand (negative) corners up to ~(total - 2π)/2 so the signed
    # sum lands near +2π; the leftover residual is nudged onto one left corner.
    t_right = (total - 2.0 * math.pi) / 2.0
    sign = np.ones(n)
    acc = 0.0
    for idx in rng.permutation(n):
        if acc + mag[idx] <= t_right:
            sign[idx] = -1.0
            acc += mag[idx]
    angles = sign * mag
    resid = 2.0 * math.pi - float(angles.sum())
    lefts = np.where(sign > 0.0)[0]
    if len(lefts) == 0:
        return None
    j = int(lefts[np.argmax(mag[lefts])])
    angles[j] += resid
    if not (amin <= abs(angles[j]) <= amax):
        return None  # the closure nudge broke that corner -> resample

    radii = rng.uniform(spec["radius"][0], spec["radius"][1], n)
    straight = rng.uniform(spec["straight"][0], spec["straight"][1], n)

    # Stretch a few straights into flat-out runs and drop the tightest corners
    # at their ends, so a long straight always pours into a heavy braking zone.
    lo, hi = spec["longs"]
    n_long = int(rng.integers(lo, hi + 1)) if hi > 0 else 0
    if n_long:
        long_idx = rng.choice(n, size=min(n_long, n), replace=False)
        straight[long_idx] *= rng.uniform(1.8, 2.6, len(long_idx))
        longest = long_idx[np.argsort(-straight[long_idx])]
        tightest = np.sort(radii)[: len(longest)]
        radii[longest] = tightest  # smallest radius after the longest straight

    # Heading on straight_k (and entering corner_k) = sum of prior corner angles.
    theta = np.concatenate([[0.0], np.cumsum(angles)[:-1]])
    sgn = np.sign(angles)
    a_disp = np.stack(
        [
            radii * sgn * (np.sin(theta + angles) - np.sin(theta)),
            radii * sgn * (np.cos(theta) - np.cos(theta + angles)),
        ],
        axis=1,
    )  # each corner's (dx, dy)

    # Close position: nudge the straight lengths so Σ_k L_k·(cosθ_k, sinθ_k)
    # cancels the corner displacements, keeping every straight above s_min.
    u = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (n, 2)
    straight = _close_straights(u, a_disp.sum(axis=0), straight, spec["straight"][0] * 0.3)
    if straight is None:
        return None  # couldn't close without a too-short straight -> resample

    cl = _assemble(theta, angles, radii, straight)
    if _self_intersects(cl):
        return None
    if not (600.0 <= _loop_len(cl) <= 2800.0):
        return None
    return cl


def _close_straights(
    u: np.ndarray, a_disp_sum: np.ndarray, straight: np.ndarray, s_min: float,
) -> np.ndarray | None:
    """Adjust straight lengths so the loop closes, keeping each ≥ ``s_min``.

    We need ``u.T @ straight = -a_disp_sum`` (2 equations, n unknowns). The
    minimum-change solution can drive an individual straight negative, so we
    solve, clamp any straight that fell below ``s_min`` to that floor, freeze it,
    and re-solve over the rest — a small active-set loop. ``None`` if the loop
    can't close without violating the floor (e.g. the free straights end up
    parallel and can't span the residual).
    """
    target = -a_disp_sum
    s = straight.astype(np.float64).copy()
    fixed = np.zeros(len(s), dtype=bool)
    for _ in range(8):
        free = ~fixed
        uf = u[free]
        rhs = target - u[fixed].T @ s[fixed]
        a = uf.T @ uf + 1e-9 * np.eye(2)
        resid = rhs - uf.T @ s[free]
        s[free] = s[free] + uf @ np.linalg.solve(a, resid)
        below = free & (s < s_min)
        if not below.any():
            closed = np.linalg.norm(u.T @ s - target) < 1e-3
            return s if closed else None
        s[below] = s_min
        fixed |= below
        if fixed.all():
            return None
    return None


# Centreline point spacing. Straights are geometrically a line, so they only
# need enough points for the renderer's depth-tint bands — coarse is fine and
# keeps the polygon count (hence frame cost) down. Corners carry the shape and
# the kerbs, so they stay finely sampled to read as smooth arcs.
STRAIGHT_DS = 11.0
CORNER_DS = 2.6


def _assemble(
    theta: np.ndarray, angles: np.ndarray, radii: np.ndarray, straight: np.ndarray,
) -> np.ndarray:
    """Walk the straight/corner sequence into a closed centreline polyline.

    Each straight and arc emits its points up to *and including* its own end
    point, and the next primitive resumes from that same point without re-emitting
    it. So a join contributes exactly one point — no duplicate (zero-length)
    segment, which would otherwise give a degenerate boundary normal there (a
    spike in the wall, seen as a gap and a render-cost blowup). The lap's start
    point isn't emitted explicitly; the final arc lands back on it (closure), so
    the wrap segment from the last point to the first is a normal step.

    Straights are sampled coarsely and corners finely (see ``STRAIGHT_DS`` /
    ``CORNER_DS``) so flat-out runs don't bury the renderer in tiny quads.
    """
    pts: list[np.ndarray] = []
    pos = np.zeros(2)
    head = 0.0
    for k in range(len(angles)):
        # Straight_k along the current heading; emit offsets up to Lk (incl. end).
        Lk = float(straight[k])
        d = np.array([math.cos(head), math.sin(head)])
        m = max(int(round(Lk / STRAIGHT_DS)), 1)
        for j in range(1, m + 1):
            pts.append(pos + d * (Lk * j / m))
        pos = pos + d * Lk
        # Corner_k: circular arc of radius radii[k] turning by angles[k]; emit
        # arc points 1..ma (the last one is the arc's exact end point).
        ak = float(angles[k])
        Rk = float(radii[k])
        s = math.copysign(1.0, ak)
        ma = max(int(round(Rk * abs(ak) / CORNER_DS)), 2)
        for j in range(1, ma + 1):
            psi = head + ak * (j / ma)
            pts.append(pos + np.array([
                Rk * s * (math.sin(psi) - math.sin(head)),
                Rk * s * (math.cos(head) - math.cos(psi)),
            ]))
        pos = pts[-1].copy()
        head = head + ak
    return np.array(pts, dtype=np.float64)


def _loop_len(cl: np.ndarray) -> float:
    d = np.roll(cl, -1, axis=0) - cl
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _self_intersects(cl: np.ndarray, step: int = 3) -> bool:
    """True if the (down-sampled) closed polyline crosses itself.

    Standard segment-crossing test: segments AB and CD cross iff A,B fall on
    opposite sides of CD *and* C,D fall on opposite sides of AB.
    """
    p = cl[::step]
    q = np.roll(p, -1, axis=0)
    n = len(p)

    def _orient(ax, ay, bx, by, cx, cy):
        # Sign of cross product (B-A) x (C-A); >0 means C is left of A->B.
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    for i in range(n):
        ax, ay = p[i]
        bx, by = q[i]
        idx = np.arange(i + 2, n)
        if i == 0:
            idx = idx[idx != n - 1]  # skip the segment that wraps back to A
        if len(idx) == 0:
            continue
        cx, cy = p[idx, 0], p[idx, 1]
        dx, dy = q[idx, 0], q[idx, 1]
        d1 = (_orient(ax, ay, bx, by, cx, cy) > 0) != (_orient(ax, ay, bx, by, dx, dy) > 0)
        d2 = (_orient(cx, cy, dx, dy, ax, ay) > 0) != (_orient(cx, cy, dx, dy, bx, by) > 0)
        if np.any(d1 & d2):
            return True
    return False


def _roll_to_straight(cl: np.ndarray) -> np.ndarray:
    """Re-index the closed loop so it starts at the top of its longest straight."""
    nxt = np.roll(cl, -1, axis=0)
    t = nxt - cl
    seg = np.hypot(t[:, 0], t[:, 1])
    ang = np.arctan2(t[:, 1], t[:, 0])
    dang = np.abs((np.roll(ang, -1) - ang + math.pi) % (2.0 * math.pi) - math.pi)
    curv = dang / np.maximum(seg, 1e-6)
    straight = curv < 0.6 * curv.mean()  # below-average bend = "straight enough"

    n = len(cl)
    best_start, best_len, run_start, run_len = 0, 0, 0, 0
    for k in range(2 * n):  # wrap once to catch runs across the seam
        if straight[k % n]:
            if run_len == 0:
                run_start = k
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0
    return np.roll(cl, -(best_start % n), axis=0)
