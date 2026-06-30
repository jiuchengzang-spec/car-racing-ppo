"""A software-rendered 3D *hood camera* for the racing env.

No GPU or extra dependencies — just pygame. The camera is mounted at the car's
nose, raised to bonnet height and pitched slightly down, looking straight ahead;
it rotates rigidly with the car, so steering swings the whole world the way an
in-car view does. The flat track is drawn with a perspective projection: the
tarmac as depth-sorted quads (painter's algorithm), kerbs and the start/finish
line on top, a sky/grass split at the horizon, and a cosmetic bonnet along the
bottom edge.

Same interface as :class:`racing.render.PygameRenderer` (``draw`` / ``close``)
so the env can swap between the top-down and hood views freely. The HUD and the
mouse steering-zone overlay are shared from :mod:`racing.render`.
"""
from __future__ import annotations

import math
import os
import random
from collections import deque

import numpy as np
import pygame

from .car import Car, CarParams
from .render import SCREEN_H, SCREEN_W, SLIP_PEAK, SPIN_PEAK, draw_hud, draw_minimap
from .track import Track

# Camera rig (metres / radians).
CAM_HEIGHT = 1.15  # bonnet height above the road
CAM_FWD = 1.30  # mounted this far ahead of the CG (out at the nose)
CAM_PITCH = math.radians(9.0)  # look slightly down at the road
FOV = math.radians(74.0)
NEAR = 1.0  # clip anything closer than this (m)

# Road feel. The bump *phase* is integrated frame-by-frame (so changing the bump
# rate on a kerb never makes the wave jump), the amplitude is eased in/out (so
# mounting a kerb ramps the shake rather than teleporting the camera), and weight
# transfer is a real nose *pitch* (with a matching dynamic horizon) so braking
# leans forward instead of dropping the whole rig. Distances in m, angles in rad.
DT = 1.0 / 60.0
RUMBLE_BASE = 0.003  # bob amplitude on smooth tarmac (scaled by speed) — gentle
RUMBLE_SPEED_REF = 45.0  # m/s at which the base rumble reaches full amplitude
BUMP_FREQ = 0.5  # primary bump rate (rad per metre travelled) — low so it stays smooth
KERB_RUMBLE = 0.014  # extra bob amplitude on a kerb
GRASS_RUMBLE = 0.025  # extra bob amplitude off on the grass
CHATTER = 0.020  # lateral chatter amplitude on kerbs / grass
ENV_SMOOTH = 0.16  # how fast the bump amplitude eases in/out (per frame)
PITCH_COUP = 0.5  # how much the vertical bob couples into nose pitch
# Dive/squat amplitude and settle speed are per-car (CarParams.dive_pitch /
# squat_pitch / pitch_smooth) so a handling preset can carry its own pitch feel.
LEAN_GAIN = 0.0016  # sideways shift per unit of lateral load (r * vx)
SEAM_SPACING = 8.0  # transverse tarmac seam every N metres (speed cue)
SEAM_MAX_DEPTH = 70.0  # don't draw seams past this depth (they'd just be noise)
SEAM = (80, 82, 92)
# Don't draw tarmac/kerbs past this depth: on infield-crossing views nearly the
# whole far side of the track projects in front of the camera, and filling those
# hundreds of tiny distant quads is what tanks the frame rate. The last
# FAR_FADE metres blend to grass so the cut-off isn't a hard line.
FAR_CLIP = 155.0
FAR_FADE = 35.0
SCN_FAR_TREE = 200.0  # cull trees past here (small billboards, not worth drawing)
SCN_FAR_BOX = 340.0  # keep distant grandstands/skyline further out for depth

SKY_TOP = (96, 140, 190)
SKY_BOT = (168, 196, 220)
CLOUD = (252, 253, 255)  # sunlit cloud
CLOUD_SHADE = (196, 209, 224)  # cooler, greyer base
# Panoramic sky baked once and sampled by heading (an HDRI-style backdrop). Soft
# fbm-noise clouds live in a vertical band that fades out before the horizon.
# Tunable: lower COVER = more cloud; higher SOFT = wispier, more feathered edges.
PANO_W = 2048  # full 360deg of azimuth (tileable)
PANO_H = 360
CLOUD_COVER = 0.50  # fbm threshold (0..1): below this stays clear sky
CLOUD_SOFT = 0.34  # threshold width: how soft/feathered the cloud edges are
# How much the sky pans with heading. Under pure rotation a distant backdrop
# should move at the *same* on-screen rate as the far scenery (there is no
# rotational parallax — only translation gives parallax), so a cloud scrolls off
# with the trees instead of looking glued in place. 0.86 = FOV/(2*tan(FOV/2)) is
# the factor that matches our linear-azimuth sky to the perspective scenery at
# screen centre. Lower it only if you deliberately want a lagging, far-off feel.
# Overridable live for tuning: `SKY_PARALLAX=0.6 python play.py`.
SKY_PARALLAX = float(os.environ.get("SKY_PARALLAX", "0.6"))
GRASS = (38, 92, 46)
TARMAC = (58, 60, 66)
TARMAC_FAR = (44, 46, 52)
KERB_A = (212, 64, 58)
KERB_B = (235, 235, 238)
KERB_W = 1.2  # kerb band width (m), outside the painted edge (= the valid runoff)
KERB_H = 0.16  # kerb crest height (m): flush at the edge, ramping up to this outside
KERB_RAMP = 5.0  # arc length (m) over which a kerb ramps up at its ends (then holds)
LIMIT = (188, 190, 196)  # track-limit line painted on the boundary (grey-white)
START = (240, 220, 90)
HOOD = (34, 36, 42)
HOOD_SHINE = (70, 74, 84)
SENSOR = (96, 210, 150)       # rangefinder beam (what the agent "sees")
SENSOR_HIT = (244, 232, 120)  # marker where a beam meets the wall

# Scenery palette.
TRUNK = (74, 52, 36)
TREES = ((36, 86, 44), (46, 102, 52), (30, 78, 40), (54, 112, 60))
BUILDINGS = (
    (120, 124, 132), (104, 108, 118), (134, 127, 120), (92, 98, 110),
    (126, 118, 108), (112, 120, 124), (98, 102, 112),
)
STAND_BODY = (66, 70, 78)
STAND_ROOF = (44, 47, 54)
STAND_SEATS = ((196, 72, 64), (74, 126, 200), (224, 206, 96), (208, 210, 214))

# Fixed sun direction for shading box faces (points from surface toward light).
_LIGHT = np.array([-0.45, -0.5, 0.74])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


class HoodCamRenderer:
    def __init__(self, track: Track, params: CarParams, mode: str = "human") -> None:
        self.track = track
        self.p = params
        self.mode = mode
        pygame.init()
        pygame.display.set_caption("car-racing-rl  ·  hood cam")
        if mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        else:
            self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 18)
        self.big = pygame.font.SysFont("consolas,menlo,monospace", 26, bold=True)
        self.clock = pygame.time.Clock()

        self.cx = SCREEN_W / 2.0
        self.cy = SCREEN_H / 2.0
        self.focal = (SCREEN_W / 2.0) / math.tan(FOV / 2.0)
        # Horizon of a flat ground plane is fixed for a fixed pitch.
        self.horizon_y = int(self.cy - self.focal * math.tan(CAM_PITCH))

        # Boundary rings as 3D points on the road (z = 0), built once.
        n = len(track.centerline)
        self._nxt = (np.arange(n) + 1) % n
        self.left3d = _to3d(track.left)
        self.right3d = _to3d(track.right)
        # Kerb the corners, just paint a line down the straights.
        self._kerb = track.curvature > max(0.010, 0.9 * float(track.curvature.mean()))
        # Longitudinal kerb profile: 0 outside a run, ramping up over KERB_RAMP at
        # entry, holding 1 through the middle, ramping back down at exit (a trapezoid
        # by distance into the run). Drives the drawn width, the crest height and the
        # rumble, so a kerb rises, stays raised, then drops — not a single wedge.
        self._kerb_frac = _kerb_profile(self._kerb, track._seg_len, KERB_RAMP)
        # The kerb sits *outside* the painted edge, on what would be runoff: a second
        # ring offset outward by the (tapered) kerb width and *raised* to a crest, so
        # it ramps up from the road — a visible reason for the rumble. Each corner
        # segment is a red/white quad from the (flush) boundary out to this ring.
        w = (KERB_W * self._kerb_frac)[:, None]
        h = KERB_H * self._kerb_frac
        self._kerb_l_out = np.column_stack([track.left + track._normals * w, h])
        self._kerb_r_out = np.column_stack([track.right - track._normals * w, h])

        # Transverse tarmac seams every few metres so the ground streams past and
        # you feel the speed even on a bare straight. Built once, by arc length.
        seams_l, seams_r = [], []
        for k in range(max(int(track.length / SEAM_SPACING), 1)):
            x, y, yaw = track.pose_at(k * SEAM_SPACING)
            nx, ny = -math.sin(yaw), math.cos(yaw)
            e = track.half * 0.92
            seams_l.append([x + nx * e, y + ny * e, 0.0])
            seams_r.append([x - nx * e, y - ny * e, 0.0])
        self._seam_l = np.array(seams_l)
        self._seam_r = np.array(seams_r)
        # Road-feel state, all integrated/eased frame-to-frame so nothing jumps
        # when the surface changes under you.
        self._phase = 0.0  # bump phase (rad), integrated from speed*freq
        self._env = 0.0  # eased bump amplitude (m)
        self._chat = 0.0  # eased lateral chatter amplitude (m)
        self._dive = 0.0  # eased weight-transfer pitch (rad)

        # Trees / grandstands / distant buildings scattered beyond the verges, so
        # the world reads as a place and you get a sense of speed off the scenery.
        self._props, self._scn_anchors = self._build_scenery()

        # Panoramic sky baked once: gradient + soft fbm-noise clouds, tileable in
        # azimuth. At runtime we blit the heading's slice (rotational parallax), so
        # the sky pans as you turn. Far softer and cheaper than drawn cloud shapes.
        self._pano = self._build_sky_panorama()
        self._pano_span = int(math.ceil(FOV / (2.0 * math.pi) * PANO_W))
        self._pano_slice = pygame.Surface((self._pano_span, PANO_H))
        # Continuous (unwrapped) heading for sampling the sky: atan2 jumps at +/-pi,
        # and with parallax != 1 that jump isn't a whole panorama wrap, so it would
        # teleport the sky at one heading. Accumulate deltas to keep it smooth.
        self._sky_head = 0.0
        self._sky_head_raw: float | None = None

        # The horizon moves with nose pitch (weight transfer); the sky is scaled to
        # meet it every frame, so we just track its current value.
        self._horizon_y = self.horizon_y

        # Screen FX: a speed vignette (edges darken as you go faster), built once as a
        # radial per-pixel-alpha overlay and blitted at speed-scaled strength each
        # frame. Plus rising tyre-smoke puffs when a tyre breaks traction.
        self._vignette = _radial_overlay((6, 7, 10))
        self._smoke: deque = deque(maxlen=48)

        self.show_beams = True  # draw the agent's rangefinder beams (toggle: B in play.py)
        self._cam = np.zeros(3)
        self._fwd = np.array([1.0, 0.0, 0.0])
        self._right = np.array([0.0, -1.0, 0.0])
        self._up = np.array([0.0, 0.0, 1.0])

    # -- camera + projection ---------------------------------------------

    def _update_camera(self, car: Car, info: dict) -> None:
        s = car.s
        dz, dx, dpitch = self._road_feel(s, info)
        hd = np.array([math.cos(s.yaw), math.sin(s.yaw), 0.0])
        cam = np.array([s.x, s.y, 0.0]) + hd * CAM_FWD + np.array([0.0, 0.0, CAM_HEIGHT])
        # Nose pitch = rest pitch + weight transfer + a little bump coupling, so the
        # whole view tips (and the horizon with it) instead of the rig dropping.
        pitch = CAM_PITCH + dpitch
        cp, sp = math.cos(pitch), math.sin(pitch)
        fwd = np.array([hd[0] * cp, hd[1] * cp, -sp])
        self._fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(self._fwd, np.array([0.0, 0.0, 1.0]))
        self._right = right / np.linalg.norm(right)
        self._up = np.cross(self._right, self._fwd)
        self._cam = cam + np.array([0.0, 0.0, dz]) + self._right * dx
        # Horizon of a flat plane follows pitch only; keep it on-screen.
        hy = int(self.cy - self.focal * math.tan(pitch))
        self._horizon_y = int(np.clip(hy, 40, SCREEN_H - 40))

    def _road_feel(self, s, info: dict) -> tuple[float, float, float]:
        """Road-surface bob + chatter and weight-transfer pitch, all continuous.

        Phase is integrated from speed (so the bump rate can change on a kerb
        without the wave jumping), amplitudes are eased toward their target each
        frame (so mounting a kerb ramps in), and weight transfer is returned as a
        nose *pitch* rather than a drop. Returns ``(dz, dx, dpitch)``."""
        speed = math.hypot(s.vx, s.vy)
        spd_f = min(speed / RUMBLE_SPEED_REF, 1.0)

        proj = self.track.project(s.x, s.y)
        edge = abs(proj.lateral) / self.track.half  # 1.0 at the painted limit line
        kerb_span = KERB_W / self.track.half
        kf = float(self._kerb_frac[proj.segment])  # kerb strength here (tapered)
        amp_t, chat_t, freq = RUMBLE_BASE * spd_f, 0.0, BUMP_FREQ
        if edge > 1.0:  # past the painted edge
            if kf > 0.05 and edge <= 1.0 + kerb_span * kf:  # riding the kerb: buzz
                amp_t += KERB_RUMBLE * kf * (0.4 + 0.6 * spd_f)
                chat_t = CHATTER * 0.6 * kf * spd_f
                freq *= 1.3
            else:  # off on the grass (no kerb, or past it): heavy, chunky thuds
                amp_t += GRASS_RUMBLE * (0.5 + 0.5 * spd_f)
                chat_t = CHATTER * (0.5 + 0.5 * spd_f)
                freq *= 0.7

        # Integrate phase from distance travelled this frame (continuous across
        # surface/freq changes), and ease the amplitudes in/out.
        self._phase += speed * DT * freq
        self._env += (amp_t - self._env) * ENV_SMOOTH
        self._chat += (chat_t - self._chat) * ENV_SMOOTH
        ph = self._phase
        bob = math.sin(ph) + 0.45 * math.sin(2.3 * ph + 1.1) + 0.22 * math.sin(4.7 * ph + 2.3)
        dz = self._env * bob
        dx = self._chat * math.sin(1.9 * ph + 0.5) + LEAN_GAIN * (s.r * s.vx)

        # Weight transfer -> eased nose pitch; a touch of the vertical bob couples
        # into pitch too, so a bump pitches the nose rather than lifting the rig.
        pitch_t = self.p.dive_pitch * info.get("brake_app", 0.0) - self.p.squat_pitch * info.get("throttle_app", 0.0)
        self._dive += (pitch_t - self._dive) * self.p.pitch_smooth
        dpitch = self._dive + PITCH_COUP * (dz / CAM_HEIGHT)
        return (
            float(np.clip(dz, -0.2, 0.2)),
            float(np.clip(dx, -0.15, 0.15)),
            float(np.clip(dpitch, -0.08, 0.08)),
        )

    def _project(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rel = pts - self._cam
        depth = rel @ self._fwd
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = self.cx + (rel @ self._right) / depth * self.focal
            sy = self.cy - (rel @ self._up) / depth * self.focal
        return np.stack([sx, sy], axis=1), depth

    # -- drawing ----------------------------------------------------------

    def draw(self, car: Car, beam_angles: np.ndarray, info: dict):
        self._update_camera(car, info)

        # Panoramic sky: blit the slice for the current heading (it pans as you
        # turn), scaled down to meet the horizon. Grass fills everything below.
        self.screen.fill(GRASS)
        self._draw_sky()

        # Scenery sits behind the track: drawn after the backdrop, before the
        # tarmac (which is always nearer and correctly paints over their bases).
        self._draw_scenery()

        ls, ld = self._project(self.left3d)
        rs, rd = self._project(self.right3d)
        # Camera-space lateral/up of the edge rings, kept so quads straddling the
        # near plane can be *clipped* to it rather than dropped — dropping the quad
        # right under the nose leaves a hole that flashes grass through the floor.
        rel_l, rel_r = self.left3d - self._cam, self.right3d - self._cam
        lat_l, up_l = rel_l @ self._right, rel_l @ self._up
        lat_r, up_r = rel_r @ self._right, rel_r @ self._up
        nxt = self._nxt
        d4 = np.stack([ld, rd, ld[nxt], rd[nxt]], axis=1)
        valid = np.all(d4 > NEAR, axis=1)
        cdepth = d4.mean(axis=1)
        nearest = d4.min(axis=1)  # closest corner of each quad (for far-clip)
        # Which quads are worth drawing: in front of the near plane, inside the
        # far-clip, and actually on screen. The on-screen bbox only means anything
        # when all four corners are in front (else the projection is degenerate),
        # so near-straddling quads bypass it and get clipped below. On winding,
        # infield-crossing views most in-range quads project off to the sides or
        # above the horizon — skipping them here is the bulk of the speed-up.
        sx = np.stack([ls[:, 0], rs[:, 0], ls[nxt, 0], rs[nxt, 0]], axis=1)
        sy = np.stack([ls[:, 1], rs[:, 1], ls[nxt, 1], rs[nxt, 1]], axis=1)
        onscreen = ((sx.max(1) > 0) & (sx.min(1) < SCREEN_W)
                    & (sy.max(1) > 0) & (sy.min(1) < SCREEN_H))
        drawable = (d4.max(axis=1) > NEAR) & (nearest <= FAR_CLIP) & (onscreen | ~valid)
        order = np.argsort(-cdepth)  # far first (painter's algorithm)
        order = order[drawable[order]]  # keep only what we'll actually draw

        # Tarmac ribbon. Distant quads are tinted darker for depth and fade to
        # grass as they near the far-clip, so the cut-off reads as haze not a wall.
        near = NEAR
        for i in order:
            j = nxt[i]
            t = float(np.clip((cdepth[i] - near) / (FAR_CLIP - near), 0.0, 1.0))
            color = _mix(TARMAC, TARMAC_FAR, t)
            fade = float(np.clip((cdepth[i] - (FAR_CLIP - FAR_FADE)) / FAR_FADE, 0.0, 1.0))
            if fade > 0.0:
                color = _mix(color, GRASS, fade)
            if valid[i]:
                poly = [tuple(ls[i]), tuple(rs[i]), tuple(rs[j]), tuple(ls[j])]
            else:
                poly = self._clip_quad(
                    [(lat_l[i], up_l[i], ld[i]), (lat_r[i], up_r[i], rd[i]),
                     (lat_r[j], up_r[j], rd[j]), (lat_l[j], up_l[j], ld[j])]
                )
                if poly is None:
                    continue
            pygame.draw.polygon(self.screen, color, poly)

        # Transverse seams streaming toward you — the main cue that you're moving.
        sl, sld = self._project(self._seam_l)
        sr, srd = self._project(self._seam_r)
        for k in range(len(sld)):
            if NEAR < sld[k] < SEAM_MAX_DEPTH and NEAR < srd[k] < SEAM_MAX_DEPTH:
                col = _mix(SEAM, TARMAC, sld[k] / SEAM_MAX_DEPTH)
                pygame.draw.line(self.screen, col, tuple(sl[k]), tuple(sr[k]), 2)

        # Wide red/white kerb bands sit *outside* the painted edge through the
        # corners (tapering in/out), with a grey-white track-limit line painted on
        # the boundary everywhere on top — so you always see where "out" begins.
        los, lod = self._project(self._kerb_l_out)
        ros, rod = self._project(self._kerb_r_out)
        for i in order:
            j = nxt[i]
            if self._kerb_frac[i] > 0.01 or self._kerb_frac[j] > 0.01:
                col = KERB_A if i % 2 == 0 else KERB_B
                if ld[i] > NEAR and ld[j] > NEAR and lod[i] > NEAR and lod[j] > NEAR:
                    pygame.draw.polygon(self.screen, col, [tuple(ls[i]), tuple(los[i]), tuple(los[j]), tuple(ls[j])])
                if rd[i] > NEAR and rd[j] > NEAR and rod[i] > NEAR and rod[j] > NEAR:
                    pygame.draw.polygon(self.screen, col, [tuple(rs[i]), tuple(ros[i]), tuple(ros[j]), tuple(rs[j])])
        for i in order:
            j = nxt[i]
            if ld[i] > NEAR and ld[j] > NEAR:
                pygame.draw.aaline(self.screen, LIMIT, tuple(ls[i]), tuple(ls[j]))  # AA: crisper edge
            if rd[i] > NEAR and rd[j] > NEAR:
                pygame.draw.aaline(self.screen, LIMIT, tuple(rs[i]), tuple(rs[j]))

        # Start/finish line across the track.
        if ld[0] > NEAR and rd[0] > NEAR:
            pygame.draw.line(self.screen, START, tuple(ls[0]), tuple(rs[0]), 4)

        if self.show_beams:
            self._draw_beams(car, beam_angles, info)

        self._draw_hood()
        self._draw_speed_fx(info)
        draw_hud(self.screen, self.font, self.big, info)
        draw_minimap(self.screen, self.track, car, info)

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(60)
            return None
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    # -- scenery ----------------------------------------------------------

    def _build_scenery(self) -> tuple[list, np.ndarray]:
        """Lay out trees, grandstands and a distant skyline beyond the verges.

        Trees are camera-facing billboards (fine for foliage); grandstands and
        buildings are real **world-fixed boxes** with shaded faces, so they read
        as solid volumes and turn as you pass them. Everything is set well back
        from the tarmac so you never drive through it. Deterministic for a given
        track shape. Returns the props and their ground anchors (for depth sort).
        """
        t, cl = self.track, self.track.centerline
        n = len(cl)
        rng = np.random.default_rng(int(abs(cl.sum()) * 1000.0) % (2**32))
        props: list = []
        anchors: list = []

        # True distance from a point to the whole centreline loop. A prop offset
        # outward from one segment can land *on* another segment where the track
        # winds back near itself (infield crossings); checking the nearest point
        # over the entire loop is what keeps buildings off the road there.
        def dist_to_track(p: np.ndarray) -> float:
            ap = p[None, :] - cl
            tt = np.clip(np.einsum("ij,ij->i", ap, t._seg_vec) / t._seg_len2, 0.0, 1.0)
            foot = cl + t._seg_vec * tt[:, None]
            return float(np.sqrt(np.min(np.einsum("ij,ij->i", p - foot, p - foot))))

        def add(anchor2d, prop: dict, footprint: float) -> bool:
            p = np.array([float(anchor2d[0]), float(anchor2d[1])])
            if dist_to_track(p) < t.half + footprint + 4.0:
                return False  # would overlap or crowd the track somewhere on the loop
            props.append(prop)
            anchors.append([p[0], p[1], 0.0])
            return True

        def outward(i: int) -> np.ndarray:
            d = t.left[i] - cl[i]
            return d / max(float(np.hypot(*d)), 1e-6)

        for i in range(0, n, 7):  # trees in loose clusters, well off the verge
            nrm = outward(i)
            tang = np.array([-nrm[1], nrm[0]])
            for side in (1.0, -1.0):
                if rng.random() < 0.4:
                    continue
                for _ in range(int(rng.integers(1, 3))):
                    off = t.half + float(rng.uniform(14.0, 60.0))
                    base = cl[i] + side * nrm * off + tang * float(rng.uniform(-12.0, 12.0))
                    add(base, {"kind": "tree", "base": base, "layers": _tree_layers(rng)}, 4.0)

        for i in (0, n // 5, (2 * n) // 5, (3 * n) // 5, (4 * n) // 5):  # grandstands
            side = 1.0 if rng.random() < 0.5 else -1.0
            nrm = side * outward(i)
            tang = np.array([-nrm[1], nrm[0]])
            yaw = math.atan2(tang[1], tang[0])  # box +x runs along the track
            hx, hy = float(rng.uniform(16.0, 26.0)), float(rng.uniform(5.0, 8.0))
            h = float(rng.uniform(9.0, 13.0))
            center = cl[i] + nrm * (t.half + float(rng.uniform(8.0, 14.0)) + hy)
            add(center, _make_stand(center, yaw, nrm, hx, hy, h, rng), max(hx, hy))

        for b in range(10):  # distant skyline, spread around the lap so it reads as depth
            i = int(b / 10.0 * n + rng.integers(-2, 3)) % n
            side = 1.0 if rng.random() < 0.5 else -1.0
            nrm = side * outward(i)
            yaw = math.atan2(nrm[1], nrm[0]) + float(rng.uniform(-0.6, 0.6))
            hx, hy = float(rng.uniform(12.0, 24.0)), float(rng.uniform(12.0, 24.0))
            h = float(rng.uniform(22.0, 58.0))
            center = cl[i] + nrm * (t.half + float(rng.uniform(95.0, 240.0)))
            add(center, _make_building(center, yaw, hx, hy, h, rng), max(hx, hy))

        return props, np.array(anchors, dtype=np.float64) if anchors else np.zeros((0, 3))

    def _build_sky_panorama(self) -> pygame.Surface:
        """Bake the 360deg sky once: gradient + soft fbm-noise clouds.

        Clouds are fractal value noise (fbm) thresholded into a soft density and
        blended toward white, confined to a vertical band that fades out before
        the horizon. The noise tiles in x so the panorama wraps seamlessly when
        sampled by heading. Deterministic for a given track.
        """
        cl = self.track.centerline
        rng = np.random.default_rng((int(abs(cl.sum()) * 1000.0) + 7) % (2**32))
        w, h = PANO_W, PANO_H
        fbm = _fbm(w, h, rng)
        fbm = (fbm - fbm.min()) / max(float(np.ptp(fbm)), 1e-6)
        v = np.linspace(0.0, 1.0, h)[:, None]  # 0 at the top, 1 at the horizon
        # Cloud band: fade in below the very top, fade out before the horizon.
        vmask = _smoothstep01((v - 0.06) / 0.26) * (1.0 - _smoothstep01((v - 0.56) / 0.30))
        cover = _smoothstep01((fbm - CLOUD_COVER) / CLOUD_SOFT)
        density = cover * vmask
        lit = np.clip(cover * 1.05 - 0.20 * v, 0.0, 1.0)  # tops lighter than bases
        base = np.empty((h, w, 3))
        cloud = np.empty((h, w, 3))
        for c in range(3):
            base[:, :, c] = SKY_TOP[c] + (SKY_BOT[c] - SKY_TOP[c]) * v
            cloud[:, :, c] = CLOUD_SHADE[c] + (CLOUD[c] - CLOUD_SHADE[c]) * lit
        out = base * (1.0 - density[:, :, None]) + cloud * density[:, :, None]
        arr = np.transpose(np.clip(out, 0, 255).astype(np.uint8), (1, 0, 2))  # (w,h,3)
        return pygame.surfarray.make_surface(arr)

    def _draw_sky(self) -> None:
        """Blit the heading's slice of the panorama, scaled to meet the horizon."""
        h = max(self._horizon_y, 1)
        raw = math.atan2(self._fwd[1], self._fwd[0])
        if self._sky_head_raw is None:
            self._sky_head = raw
        else:
            d = (raw - self._sky_head_raw + math.pi) % (2.0 * math.pi) - math.pi
            self._sky_head += d
        self._sky_head_raw = raw
        # Negated: the perspective projection puts world points at sx = cx - tan(a-h),
        # so turning left (heading up) sweeps scenery *right*. The sky must do the
        # same, which means the sampled column moves the opposite way to heading.
        x0 = -self._sky_head * SKY_PARALLAX / (2.0 * math.pi) * PANO_W - self._pano_span / 2.0
        xm = x0 % PANO_W
        sl = self._pano_slice
        sl.blit(self._pano, (-xm, 0))  # main copy
        sl.blit(self._pano, (-xm + PANO_W, 0))  # wrap copy for the seam
        scaled = pygame.transform.scale(sl, (SCREEN_W, h))
        self.screen.blit(scaled, (0, 0))

    def _draw_scenery(self) -> None:
        """Collect every scenery polygon, near-clip it, then paint far-to-near.

        One global depth sort across all props (not per object) so overlapping
        buildings can't paint over nearer ones, and near-plane *clipping* (rather
        than dropping a whole face) so faces never wink out as you drive past.
        """
        if not self._props:
            return
        rel = self._scn_anchors - self._cam
        anchor_depth = rel @ self._fwd
        anchor_lat = rel @ self._right
        half_fov_tan = math.tan(FOV / 2.0) * 1.35  # a little margin for prop width
        polys: list = []  # (depth, screen_pts, fill, outline)
        for k in range(len(self._props)):
            dz = anchor_depth[k]
            prop = self._props[k]
            far = SCN_FAR_TREE if prop["kind"] == "tree" else SCN_FAR_BOX
            if dz <= 0.5 or dz > far:
                continue  # behind the camera or past the scenery far-cull
            if abs(anchor_lat[k]) > dz * half_fov_tan + 30.0:
                continue  # well outside the view frustum sideways
            if prop["kind"] == "tree":
                self._collect_billboard(prop["base"], prop["layers"], polys)
            else:
                for corners, color in prop["boxes"]:
                    self._collect_box(corners, color, polys)
        for _, pts, fill, outline in sorted(polys, key=lambda it: -it[0]):
            pygame.draw.polygon(self.screen, fill, pts)
            if outline is not None and len(pts) >= 3:
                pygame.draw.polygon(self.screen, outline, pts, 1)

    def _cam_coords(self, world: np.ndarray) -> list:
        rel = world - self._cam
        return list(zip(rel @ self._right, rel @ self._up, rel @ self._fwd))

    def _collect_billboard(self, base: np.ndarray, layers: list, polys: list) -> None:
        # Ground-projected camera-right axis -> the card faces the camera, so its
        # vertical edges share a depth and project as true verticals (no lean).
        rh = np.array([self._right[0], self._right[1], 0.0])
        rh = rh / max(float(np.linalg.norm(rh)), 1e-6)
        ax, ay = float(base[0]), float(base[1])
        for w, z0, z1, color, is_tri in layers:
            hw = rh * (w * 0.5)
            if is_tri:
                verts = np.array([
                    [ax - hw[0], ay - hw[1], z0], [ax + hw[0], ay + hw[1], z0], [ax, ay, z1],
                ])
            else:
                verts = np.array([
                    [ax - hw[0], ay - hw[1], z0], [ax + hw[0], ay + hw[1], z0],
                    [ax + hw[0], ay + hw[1], z1], [ax - hw[0], ay - hw[1], z1],
                ])
            cam = self._cam_coords(verts)
            scr = self._clip_quad(cam)
            if scr is not None:
                depth = float(np.mean([max(c[2], NEAR) for c in cam]))
                polys.append((depth, scr, color, None))

    def _collect_box(self, corners: np.ndarray, color: tuple, polys: list) -> None:
        faces = ((4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
        center = corners.mean(axis=0)
        cam = self._cam_coords(corners)
        for f in faces:
            idx = list(f)
            v = corners[idx]
            fc = v.mean(axis=0)
            nrm = np.cross(v[1] - v[0], v[2] - v[0])
            nl = float(np.linalg.norm(nrm))
            if nl < 1e-9:
                continue
            nrm /= nl
            if np.dot(nrm, fc - center) < 0:
                nrm = -nrm
            if np.dot(nrm, fc - self._cam) > 0:
                continue  # face points away from the camera
            scr = self._clip_quad([cam[i] for i in idx])
            if scr is None:
                continue
            bright = 0.55 + 0.45 * max(0.0, float(np.dot(nrm, _LIGHT)))
            fill = tuple(int(np.clip(c * bright, 0, 255)) for c in color)
            outline = tuple(int(c * 0.62) for c in fill)
            depth = float(np.mean([max(cam[i][2], NEAR) for i in idx]))
            polys.append((depth, scr, fill, outline))

    def _clip_quad(self, cam_quad: list) -> list | None:
        """Clip a road quad to the near plane (depth >= NEAR), then project it.

        Each input vertex is ``(lateral, up, depth)`` in camera space. Returns the
        clipped polygon's screen-space points (Sutherland-Hodgman against the near
        plane), or ``None`` if no part of it is in front of the camera.
        """
        out = []
        n = len(cam_quad)
        for k in range(n):
            cur, nxt = cam_quad[k], cam_quad[(k + 1) % n]
            if cur[2] >= NEAR:
                out.append(cur)
            if (cur[2] >= NEAR) != (nxt[2] >= NEAR):
                f = (NEAR - cur[2]) / (nxt[2] - cur[2])
                out.append((cur[0] + f * (nxt[0] - cur[0]), cur[1] + f * (nxt[1] - cur[1]), NEAR))
        if len(out) < 3:
            return None
        return [(self.cx + c[0] / c[2] * self.focal, self.cy - c[1] / c[2] * self.focal) for c in out]

    def _draw_beams(self, car: Car, beam_angles: np.ndarray, info: dict) -> None:
        """Draw the agent's rangefinder beams along the road (near-plane clipped).

        Uses the env's *smoothed* beam distances (``info["beam_dists_m"]``) so the
        lines are as steady as the values the policy reacts to — no grazing-ray
        shake — anti-aliased and faded toward the grass with distance for a clean
        look. Falls back to a live raycast if the env didn't supply distances.
        """
        s = car.s
        z = 0.06  # lift a touch off the tarmac so the lines aren't depth-fought away
        origin = np.array([s.x, s.y, z])
        dists = info.get("beam_dists_m")
        for i, a in enumerate(beam_angles):
            ang = s.yaw + float(a)
            if dists is not None and i < len(dists):
                dist = float(dists[i])
            else:
                dist = self.track.cast_ray(s.x, s.y, ang, FAR_CLIP)
            end = np.array([s.x + dist * math.cos(ang), s.y + dist * math.sin(ang), z])
            seg = self._clip_segment(origin, end)
            if seg is None:
                continue
            (p0, _), (p1, d1) = seg
            col = _mix(SENSOR, GRASS, 0.6 * min(dist / FAR_CLIP, 1.0))  # fade with range
            pygame.draw.aaline(self.screen, col, p0, p1)
            if d1 > NEAR:  # mark where the beam hits the wall, if it's in front
                pygame.draw.circle(self.screen, SENSOR_HIT, (int(p1[0]), int(p1[1])), 3)

    def _clip_segment(self, p0_world: np.ndarray, p1_world: np.ndarray):
        """Clip a world segment to depth >= NEAR and project; None if fully behind.

        Returns ``((p0, depth0), (p1, depth1))`` of screen points.
        """
        r0, r1 = p0_world - self._cam, p1_world - self._cam
        a = [float(r0 @ self._right), float(r0 @ self._up), float(r0 @ self._fwd)]
        b = [float(r1 @ self._right), float(r1 @ self._up), float(r1 @ self._fwd)]
        if a[2] < NEAR and b[2] < NEAR:
            return None
        if (a[2] < NEAR) != (b[2] < NEAR):  # one endpoint behind: clip it to the plane
            t = (NEAR - a[2]) / (b[2] - a[2])
            cross = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), NEAR]
            if a[2] < NEAR:
                a = cross
            else:
                b = cross
        p0 = (self.cx + a[0] / a[2] * self.focal, self.cy - a[1] / a[2] * self.focal)
        p1 = (self.cx + b[0] / b[2] * self.focal, self.cy - b[1] / b[2] * self.focal)
        return (p0, a[2]), (p1, b[2])

    def _draw_hood(self) -> None:
        w, h = SCREEN_W, SCREEN_H
        hood = [(w * 0.20, h), (w * 0.80, h), (w * 0.62, h - 86), (w * 0.38, h - 86)]
        pygame.draw.polygon(self.screen, HOOD, hood)
        pygame.draw.line(self.screen, HOOD_SHINE, (w * 0.5, h), (w * 0.5, h - 86), 2)

    def _draw_speed_fx(self, info: dict) -> None:
        """Speed vignette + tyre smoke — drawn over the world, under the HUD, so the
        instruments stay crisp on top."""
        speed = float(info.get("speed", 0.0))
        k = min(speed / 75.0, 1.0)              # ~270 km/h ≈ full vignette
        if k > 0.02:
            v = self._vignette.copy()
            v.fill((255, 255, 255, int(255 * 0.5 * k)), special_flags=pygame.BLEND_RGBA_MULT)
            self.screen.blit(v, (0, 0))
        self._draw_smoke(info)

    def _draw_smoke(self, info: dict) -> None:
        """Translucent puffs that rise from the lower edge when a tyre breaks traction."""
        slide = max(abs(info.get("slip_r", 0.0)) / SLIP_PEAK,
                    abs(info.get("slip_ratio", 0.0)) / SPIN_PEAK,
                    abs(info.get("slip_f", 0.0)) / SLIP_PEAK)
        if slide > 1.05 and info.get("speed", 0.0) > 4.0:
            for _ in range(2):
                x = SCREEN_W * (0.5 + random.uniform(-0.26, 0.26))
                y = SCREEN_H - random.uniform(0.0, 40.0)
                self._smoke.append([x, y, 0.0, random.uniform(14.0, 26.0)])
        if not self._smoke:
            return
        alive = []
        for px, py, age, rad in self._smoke:
            age += 0.045
            if age >= 1.0:
                continue
            py -= 1.8            # rise
            rad += 0.9           # billow out
            a = int(110 * (1.0 - age))
            d = int(rad * 2)
            puff = pygame.Surface((d, d), pygame.SRCALPHA)
            pygame.draw.circle(puff, (205, 206, 210, a), (int(rad), int(rad)), int(rad))
            self.screen.blit(puff, (px - rad, py - rad))
            alive.append([px, py, age, rad])
        self._smoke = deque(alive, maxlen=48)

    def close(self) -> None:
        pygame.quit()


def _radial_overlay(color: tuple) -> pygame.Surface:
    """A full-screen per-pixel-alpha overlay of `color`: transparent in the centre,
    ramping opaque toward the edges. Reused for the speed vignette and brake glow."""
    yy, xx = np.mgrid[0:SCREEN_H, 0:SCREEN_W]
    cx, cy = SCREEN_W / 2.0, SCREEN_H * 0.52
    d = np.sqrt(((xx - cx) / (SCREEN_W * 0.60)) ** 2 + ((yy - cy) / (SCREEN_H * 0.60)) ** 2)
    a = (np.clip((d - 0.45) / 0.55, 0.0, 1.0) ** 1.6 * 255).astype(np.uint8)
    surf = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    rgb = pygame.surfarray.pixels3d(surf)
    rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2] = color
    del rgb
    al = pygame.surfarray.pixels_alpha(surf)
    al[:, :] = a.T
    del al
    return surf


def _tree_layers(rng) -> list:
    """A tree as a trunk rect + two stacked green triangles (camera-facing)."""
    h = float(rng.uniform(4.5, 9.5))
    w = h * float(rng.uniform(0.5, 0.7))
    col = TREES[int(rng.integers(len(TREES)))]
    return [
        (w * 0.16, 0.0, 0.34 * h, TRUNK, False),
        (w, 0.28 * h, 0.70 * h, col, True),
        (w * 0.66, 0.52 * h, h, col, True),
    ]


def _box_corners(cx: float, cy: float, yaw: float, hx: float, hy: float, z0: float, z1: float) -> np.ndarray:
    """8 world corners of a box: indices 0-3 the base ring, 4-7 the top ring."""
    c, s = math.cos(yaw), math.sin(yaw)
    base = []
    for lx, ly in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
        base.append((cx + lx * c - ly * s, cy + lx * s + ly * c))
    return np.array([(x, y, z0) for x, y in base] + [(x, y, z1) for x, y in base], dtype=np.float64)


def _make_building(center, yaw: float, hx: float, hy: float, h: float, rng) -> dict:
    """A distant tower: one shaded box."""
    col = BUILDINGS[int(rng.integers(len(BUILDINGS)))]
    return {"kind": "box", "boxes": [(_box_corners(center[0], center[1], yaw, hx, hy, 0.0, h), col)]}


def _make_stand(center, yaw: float, nrm: np.ndarray, hx: float, hy: float, h: float, rng) -> dict:
    """A grandstand: a seat-coloured body box under an overhanging roof box.

    ``hx`` runs along the track, ``hy`` is the depth; ``nrm`` points away from the
    track, so the roof is nudged the other way to overhang the seats trackside.
    """
    seat = STAND_SEATS[int(rng.integers(len(STAND_SEATS)))]
    body = _box_corners(center[0], center[1], yaw, hx, hy, 0.0, h * 0.82)
    roof_c = (center[0] - nrm[0] * hy * 0.35, center[1] - nrm[1] * hy * 0.35)
    roof = _box_corners(roof_c[0], roof_c[1], yaw, hx * 1.04, hy * 1.35, h * 0.82, h)
    return {"kind": "box", "boxes": [(body, seat), (roof, STAND_ROOF)]}


def _to3d(pts2d: np.ndarray) -> np.ndarray:
    out = np.zeros((len(pts2d), 3))
    out[:, :2] = pts2d
    return out


def _mix(a, b, t: float):
    return tuple(int(a[k] + (b[k] - a[k]) * t) for k in range(3))


def _kerb_profile(flag: np.ndarray, seg_len: np.ndarray, ramp: float) -> np.ndarray:
    """Trapezoidal 0..1 profile along a closed ring of kerb on/off flags.

    For each kerbed segment, measure the arc distance to the nearer end of its run
    (forward to the next gap, backward to the previous one) and ramp it up over
    ``ramp`` metres. The result is 0 at a run's ends, climbs to 1 within ``ramp``,
    and holds at 1 through the middle — so the kerb rises, plateaus, then falls.
    """
    n = len(flag)
    fwd = np.zeros(n)
    acc = 0.0
    for _ in range(2):  # twice around the ring so wrap-around runs accumulate
        for i in range(n):
            if not flag[i]:
                acc = 0.0
            else:
                fwd[i] = acc
                acc += float(seg_len[i])
    bwd = np.zeros(n)
    acc = 0.0
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            if not flag[i]:
                acc = 0.0
            else:
                bwd[i] = acc
                acc += float(seg_len[i])
    depth = np.minimum(fwd, bwd)
    return np.clip(depth / max(ramp, 1e-6), 0.0, 1.0)


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _value_noise(w: int, h: int, gx: int, gy: int, rng) -> np.ndarray:
    """Smooth value noise on a (gy x gx) lattice, bilinearly upsampled to (h, w).

    The lattice wraps in x (last column == first), and the sample positions span
    [0, gx) without the endpoint, so the result tiles seamlessly horizontally.
    """
    g = rng.random((gy + 1, gx + 1))
    g[:, -1] = g[:, 0]
    xs = np.linspace(0.0, gx, w, endpoint=False)
    ys = np.linspace(0.0, gy, h, endpoint=False)
    x0 = np.floor(xs).astype(int); fx = xs - x0; x1 = x0 + 1
    y0 = np.floor(ys).astype(int); fy = ys - y0; y1 = y0 + 1
    sx = (fx * fx * (3.0 - 2.0 * fx))[None, :]
    sy = (fy * fy * (3.0 - 2.0 * fy))[:, None]
    gy0, gy1 = g[y0], g[y1]
    top = gy0[:, x0] * (1.0 - sx) + gy0[:, x1] * sx
    bot = gy1[:, x0] * (1.0 - sx) + gy1[:, x1] * sx
    return top * (1.0 - sy) + bot * sy


def _fbm(w: int, h: int, rng, octaves: int = 5, gx0: int = 5, gy0: int = 2) -> np.ndarray:
    """Fractal Brownian motion: stacked octaves of value noise, each finer/fainter."""
    out = np.zeros((h, w))
    amp, total, gx, gy = 1.0, 0.0, gx0, gy0
    for _ in range(octaves):
        out += amp * _value_noise(w, h, gx, gy, rng)
        total += amp
        amp *= 0.5
        gx *= 2
        gy *= 2
    return out / total
