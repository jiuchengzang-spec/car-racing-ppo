"""Pygame renderer for the racing env.

Imported lazily by ``RacingEnv.render`` so headless training never needs pygame
or a display. Draws the tarmac, kerbs, the car as a rotated body, its sensor
beams, and a small HUD with speed / lap time / progress. The camera follows the
car and the world is scaled metres -> pixels.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import pygame

from .car import Car, CarParams
from .track import Track

SCREEN_W, SCREEN_H = 1100, 760
PPM = 4.2  # pixels per metre (zoom)

# Mouse steering: the cursor's horizontal offset from centre sets the lock, with
# a small dead strip around centre that still counts as straight. The on-screen
# readout lives in the bottom telemetry bar; play.py shares this deadzone.
STEER_ZONE_DEAD = 18  # px either side of centre that still counts as straight

GRASS = (28, 42, 30)
TARMAC = (54, 56, 60)
KERB = (210, 210, 215)
KERB_A = (212, 64, 58)
KERB_B = (235, 235, 238)
EDGE = (220, 222, 226)
CENTER = (90, 92, 98)
CAR_BODY = (224, 86, 72)
CAR_NOSE = (250, 240, 210)
BEAM = (90, 170, 120)
HUD = (235, 238, 240)
HUD_DIM = (150, 156, 162)
WARN = (235, 90, 80)
GOOD = (120, 210, 140)
PURPLE = (190, 130, 235)
START = (240, 220, 90)
ZONE = (90, 150, 200)
TRAIL_COL = (96, 176, 232)   # racing-line trail (fades with age)
SKID_COL = (20, 18, 22)      # rubber laid down when a tyre slides past grip


class PygameRenderer:
    def __init__(self, track: Track, params: CarParams, mode: str = "human") -> None:
        self.track = track
        self.p = params
        self.mode = mode
        pygame.init()
        pygame.display.set_caption("car-racing-rl")
        if mode == "human":
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        else:
            self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 18)
        self.big = pygame.font.SysFont("consolas,menlo,monospace", 26, bold=True)
        self.clock = pygame.time.Clock()
        self._cam = np.array([0.0, 0.0])
        self._trail: deque = deque(maxlen=240)  # recent car positions (racing-line trail)
        self._skids: deque = deque(maxlen=700)  # world points where a tyre slid past grip
        self._last_pos = None                   # to drop skids only when actually moving

    def _to_screen(self, pts: np.ndarray) -> np.ndarray:
        rel = (pts - self._cam) * PPM
        sx = rel[..., 0] + SCREEN_W / 2
        sy = SCREEN_H / 2 - rel[..., 1]  # flip y for screen coords
        return np.stack([sx, sy], axis=-1)

    def draw(self, car: Car, beam_angles: np.ndarray, info: dict):
        s = car.s
        self._cam = np.array([s.x, s.y])
        self._record_trail(car)
        self.screen.fill(GRASS)

        # Tarmac: filled ribbon between the two boundaries.
        left = self._to_screen(self.track.left)
        right = self._to_screen(self.track.right)
        poly = np.concatenate([left, right[::-1]], axis=0)
        pygame.draw.polygon(self.screen, TARMAC, poly.tolist())

        # Edges: red/white kerbs through the corners, a painted line on straights.
        kerb = self.track.curvature > max(0.010, 0.9 * float(self.track.curvature.mean()))
        n = len(left)
        for i in range(n):
            j = (i + 1) % n
            if kerb[i]:
                col, w = (KERB_A if i % 2 == 0 else KERB_B), 3
            else:
                col, w = EDGE, 1
            pygame.draw.line(self.screen, col, left[i], left[j], w)
            pygame.draw.line(self.screen, col, right[i], right[j], w)
        center = self._to_screen(self.track.centerline)
        pygame.draw.lines(self.screen, CENTER, True, center.tolist(), 1)

        # Start/finish line across the track at the first centreline point.
        sl = self._to_screen(np.stack([self.track.left[0], self.track.right[0]]))
        pygame.draw.line(self.screen, START, sl[0], sl[1], 4)

        # Skid marks (rubber laid where a tyre slid) then the fading racing-line trail.
        self._draw_skids_and_trail()

        # Sensor beams: the env's smoothed distances (capped for this view), anti-aliased.
        beam_d = info.get("beam_dists_m")
        for i, a in enumerate(beam_angles):
            ang = s.yaw + a
            if beam_d is not None and i < len(beam_d):
                d = min(float(beam_d[i]), 60.0)
            else:
                d = self.track.cast_ray(s.x, s.y, ang, 60.0)
            end = np.array([s.x + d * math.cos(ang), s.y + d * math.sin(ang)])
            seg = self._to_screen(np.stack([[s.x, s.y], end]))
            pygame.draw.aaline(self.screen, BEAM, seg[0], seg[1])

        self._draw_car(car)
        self._draw_hud(info)
        draw_minimap(self.screen, self.track, car, info)

        if self.mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata_fps())
            return None
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def metadata_fps(self) -> int:
        return 60

    def _draw_car(self, car: Car) -> None:
        s = car.s
        L, W = self.p.length, self.p.width
        corners = np.array(
            [[L * 0.5, W * 0.5], [L * 0.5, -W * 0.5], [-L * 0.5, -W * 0.5], [-L * 0.5, W * 0.5]]
        )
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        rot = np.array([[c, -sn], [sn, c]])
        world = (corners @ rot.T) + np.array([s.x, s.y])
        scr = self._to_screen(world)
        pygame.draw.polygon(self.screen, CAR_BODY, scr.tolist())
        # Nose marker so heading is obvious.
        nose = np.array([[L * 0.5, W * 0.32], [L * 0.5, -W * 0.32], [L * 0.2, 0.0]])
        nose_w = (nose @ rot.T) + np.array([s.x, s.y])
        pygame.draw.polygon(self.screen, CAR_NOSE, self._to_screen(nose_w).tolist())

    def _record_trail(self, car: Car) -> None:
        """Append the car position to the trail and lay skid points where a tyre is
        sliding past grip. Clears on a big position jump (episode reset/restage)."""
        s = car.s
        p = (s.x, s.y)
        if self._last_pos is not None:
            dx, dy = p[0] - self._last_pos[0], p[1] - self._last_pos[1]
            if dx * dx + dy * dy > 400.0:   # >20 m in one tick = teleport/reset
                self._trail.clear()
                self._skids.clear()
        self._last_pos = p
        self._trail.append(p)
        if s.speed < 3.0:
            return
        L, W = self.p.length, self.p.width
        c, sn = math.cos(s.yaw), math.sin(s.yaw)

        def wheel(fx: float, fy: float):
            return (s.x + fx * c - fy * sn, s.y + fx * sn + fy * c)

        kappa = abs((s.wheel_v_r - s.vx) / max(abs(s.vx), self.p.slip_vx_floor))
        if abs(s.slip_f) > SLIP_PEAK:                       # front sliding (understeer)
            self._skids.append(wheel(L * 0.35, W * 0.42))
            self._skids.append(wheel(L * 0.35, -W * 0.42))
        if abs(s.slip_r) > SLIP_PEAK or kappa > SPIN_PEAK:  # rear sliding / wheelspin
            self._skids.append(wheel(-L * 0.35, W * 0.42))
            self._skids.append(wheel(-L * 0.35, -W * 0.42))

    def _draw_skids_and_trail(self) -> None:
        if self._skids:
            for p in self._to_screen(np.array(self._skids, dtype=float)):
                x, y = int(p[0]), int(p[1])
                if -4 <= x <= SCREEN_W + 4 and -4 <= y <= SCREEN_H + 4:
                    pygame.draw.circle(self.screen, SKID_COL, (x, y), 2)
        if len(self._trail) > 1:
            pts = self._to_screen(np.array(self._trail, dtype=float))
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            n = len(pts) - 1
            for i in range(n):
                a = int(30 + 170 * (i + 1) / n)   # newest brightest, older fades out
                pygame.draw.line(overlay, (*TRAIL_COL, a), pts[i], pts[i + 1], 2)
            self.screen.blit(overlay, (0, 0))

    def _draw_hud(self, info: dict) -> None:
        draw_hud(self.screen, self.font, self.big, info)

    def close(self) -> None:
        pygame.quit()


CARD_X, CARD_Y, CARD_W, CARD_PAD = 14, 14, 230, 16
CARD_BG = (22, 24, 30)
DIVIDER = (52, 56, 66)
PILL_BG = (32, 35, 43)
PILL_LIVE = (50, 92, 146)
PILL_PEND = (24, 26, 32)
LABEL = (150, 156, 168)
SEC_W, SEC_H, SEC_GAP = 100, 52, 8

_FONTS: dict[tuple[int, bool], "pygame.font.Font"] = {}


def _get_font(size: int, bold: bool = False):
    key = (size, bold)
    f = _FONTS.get(key)
    if f is None:
        f = pygame.font.SysFont("consolas,menlo,monospace", size, bold=bold)
        _FONTS[key] = f
    return f


def _blit_center(screen, font, text: str, color, cx: int, cy: int) -> None:
    surf = font.render(text, True, color)
    screen.blit(surf, surf.get_rect(center=(cx, cy)))


def draw_hud(screen, font, big, info: dict) -> None:
    """Shared HUD: a compact lap-times card (top-left), a top-centre sector strip,
    and a bottom-centre speed readout."""
    pad = CARD_PAD
    x = CARD_X + pad
    sub_y = CARD_Y + 14
    div1_y = sub_y + 24
    rows_y = div1_y + 12
    rows = [
        ("THIS", _fmt_time(info.get("current_lap_time")), HUD),
        ("LAST", _fmt_lap(info.get("last_lap_time"), info.get("last_lap_valid", True)),
         HUD if info.get("last_lap_valid", True) else WARN),
        ("BEST", _fmt_time(info.get("best_lap_time")), GOOD if info.get("best_lap_time") else HUD_DIM),
        ("THEO", _fmt_time(info.get("theoretical_best")), PURPLE if info.get("theoretical_best") else HUD_DIM),
    ]
    card_h = (rows_y + len(rows) * 24 + 12) - CARD_Y

    # --- solid card: lap subline + time rows ---
    pygame.draw.rect(screen, CARD_BG, pygame.Rect(CARD_X, CARD_Y, CARD_W, card_h), border_radius=14)
    pygame.draw.line(screen, DIVIDER, (CARD_X + pad - 2, div1_y), (CARD_X + CARD_W - pad + 2, div1_y))

    f_sub = _get_font(13)
    sub = f"LAP {info.get('lap_count', 0)}  ·  {info.get('lap_fraction', 0) * 100:.0f}%  ·  {info.get('valid_laps', 0)} VALID"
    screen.blit(f_sub.render(sub, True, LABEL), (x, sub_y))

    f_lbl = _get_font(14)
    for k, (label, value, color) in enumerate(rows):
        ry = rows_y + k * 24
        screen.blit(f_lbl.render(label, True, LABEL), (x, ry + 2))
        vs = font.render(value.strip(), True, color)
        screen.blit(vs, (CARD_X + CARD_W - pad - vs.get_width(), ry))

    _draw_sectors(screen, info)
    _draw_telemetry(screen, info)
    _draw_racing_telemetry(screen, info)

    if not info.get("lap_valid", True):
        _draw_banner(screen, big, "LAP INVALID", WARN)
    if info.get("off_track"):
        screen.blit(big.render("OFF TRACK", True, WARN), (SCREEN_W // 2 - 90, 16))

    hint = "mouse / arrows / WASD to steer   ·   W·S throttle/brake   ·   R restage   ·   Esc quit"
    screen.blit(font.render(hint, True, HUD_DIM), (18, SCREEN_H - 28))


def _draw_sectors(screen, info: dict) -> None:
    """Three solid sector pills along the top centre: completed splits show a
    green/red delta vs your best sector, the live one counts up, the rest
    preview your last lap."""
    splits = info.get("sector_splits", [None, None, None])
    last_s = info.get("last_sectors", [None, None, None])
    deltas = info.get("sector_delta", [None, None, None])
    cur = info.get("cur_sector", 0)
    armed = info.get("timing_armed", False)
    clt = info.get("current_lap_time", 0.0) or 0.0
    done = sum(s for s in splits[:cur] if s) if armed else 0.0

    total = 3 * SEC_W + 2 * SEC_GAP
    x0 = SCREEN_W // 2 - total // 2
    y = 14
    f_sl = _get_font(12)
    f_st = _get_font(20, bold=True)
    f_sd = _get_font(12, bold=True)
    for i in range(3):
        px = x0 + i * (SEC_W + SEC_GAP)
        if splits[i] is not None:
            fill, tcol = PILL_BG, HUD
            time_txt = f"{splits[i]:.2f}"
            d = deltas[i]
            dtxt = "" if d is None else f"{'+' if d > 0 else '-'}{abs(d):.3f}"
            dcol = LABEL if d is None else (GOOD if d <= 0 else WARN)
        elif armed and i == cur:
            fill, tcol = PILL_LIVE, HUD
            time_txt = f"{max(clt - done, 0.0):.2f}"
            dtxt, dcol = "LIVE", (210, 224, 240)
        else:
            fill, tcol = PILL_PEND, HUD_DIM
            time_txt = f"{last_s[i]:.2f}" if last_s[i] is not None else "--.--"
            dtxt, dcol = "", LABEL
        pygame.draw.rect(screen, fill, pygame.Rect(px, y, SEC_W, SEC_H), border_radius=10)
        cx = px + SEC_W // 2
        _blit_center(screen, f_sl, f"S{i + 1}", LABEL, cx, y + 11)
        _blit_center(screen, f_st, time_txt, tcol, cx, y + 29)
        if dtxt:
            _blit_center(screen, f_sd, dtxt, dcol, cx, y + 44)

    if not armed:
        _blit_center(screen, _get_font(13), "cross the line to start your lap", START,
                     SCREEN_W // 2, y + SEC_H + 12)


THR_COL = (104, 206, 132)
BRK_COL = (236, 96, 86)
STEER_COL = (96, 176, 232)
TRACK_BG = (40, 43, 52)
TELE_W, TELE_H, TELE_PAD = 312, 116, 14
TELE_BOTTOM = SCREEN_H - 40


def _draw_telemetry(screen, info: dict) -> None:
    """Bottom-centre cluster: throttle/brake bars, big speed, and a steering bar."""
    px = SCREEN_W // 2 - TELE_W // 2
    py = TELE_BOTTOM - TELE_H
    pygame.draw.rect(screen, CARD_BG, pygame.Rect(px, py, TELE_W, TELE_H), border_radius=16)

    ix, iy = px + TELE_PAD, py + TELE_PAD
    bar_w, bar_h, bar_gap = 22, 58, 9
    f_tiny = _get_font(11, bold=True)

    # Throttle + brake: vertical bars that fill from the bottom up.
    for k, (label, frac, col) in enumerate((
        ("T", info.get("throttle_app", 0.0), THR_COL),
        ("B", info.get("brake_app", 0.0), BRK_COL),
    )):
        bx = ix + k * (bar_w + bar_gap)
        rect = pygame.Rect(bx, iy, bar_w, bar_h)
        pygame.draw.rect(screen, TRACK_BG, rect, border_radius=5)
        fh = int(bar_h * float(np.clip(frac, 0.0, 1.0)))
        if fh > 0:
            pygame.draw.rect(screen, col, pygame.Rect(bx, iy + bar_h - fh, bar_w, fh), border_radius=5)
        _blit_center(screen, f_tiny, label, LABEL, bx + bar_w // 2, iy + bar_h + 9)

    # Speed: the big readout, to the right of the pedal bars.
    f_num = _get_font(44, bold=True)
    f_unit = _get_font(15)
    ns = f_num.render(f"{info.get('speed_kmh', 0):.0f}", True, HUD)
    us = f_unit.render("km/h", True, LABEL)
    pedals_right = ix + 2 * bar_w + bar_gap
    region_l, region_r = pedals_right, px + TELE_W - TELE_PAD
    block_w = ns.get_width() + 7 + us.get_width()
    bx0 = (region_l + region_r) // 2 - block_w // 2
    screen.blit(ns, (bx0, iy + (bar_h - ns.get_height()) // 2 - 2))
    screen.blit(us, (bx0 + ns.get_width() + 7, iy + bar_h - us.get_height() - 8))

    # Steering: a horizontal bar; the fill runs from centre toward the lock, with
    # a notch at dead-ahead. Left lock is positive steer (our world convention).
    sb_h = 14
    sb_y = iy + bar_h + 18
    sb_x0, sb_x1 = ix, px + TELE_W - TELE_PAD
    pygame.draw.rect(screen, TRACK_BG, pygame.Rect(sb_x0, sb_y, sb_x1 - sb_x0, sb_h), border_radius=7)
    cx = (sb_x0 + sb_x1) // 2
    steer = float(np.clip(info.get("steer_cmd", 0.0), -1.0, 1.0))
    half = (sb_x1 - sb_x0) // 2 - 2
    pos = int(cx - steer * half)
    lo, hi = min(cx, pos), max(cx, pos)
    if hi - lo > 0:
        pygame.draw.rect(screen, STEER_COL, pygame.Rect(lo, sb_y, hi - lo, sb_h), border_radius=7)
    pygame.draw.line(screen, HUD, (cx, sb_y - 1), (cx, sb_y + sb_h + 1), 1)
    pygame.draw.circle(screen, HUD, (pos, sb_y + sb_h // 2), 6)


# --- racing telemetry cluster (gear/RPM tach, tyre grip, friction circle) ------
REDLINE_RPM = 8200.0      # matches CarParams.redline_rpm
SHIFT_RPM = 7800.0        # matches CarParams.shift_up_rpm (tach redline zone)
SLIP_PEAK = math.radians(6.5)   # slip angle at grip peak; beyond ≈ sliding
SPIN_PEAK = 0.18                # rear slip ratio at grip peak; beyond ≈ wheelspin
G_SCALE = 1.6                   # friction-circle full-scale (g)
GRIP_OK = (110, 200, 130)
GRIP_WARN = (236, 200, 92)
GRIP_OVER = (236, 96, 84)
RPM_COL = (120, 180, 235)


def _grip_color(load: float):
    return GRIP_OVER if load >= 1.0 else GRIP_WARN if load >= 0.8 else GRIP_OK


def _draw_racing_telemetry(screen, info: dict) -> None:
    """Bottom-left instruments straight from the car physics: gear + RPM tach,
    front/rear tyre grip bars (green→red as a tyre nears/exceeds its grip limit, so
    you see understeer/oversteer/wheelspin), and a friction-circle g-meter."""
    w, h, pad = 286, 122, 12
    x0, y0 = 16, SCREEN_H - 44 - h
    pygame.draw.rect(screen, CARD_BG, pygame.Rect(x0, y0, w, h), border_radius=14)
    ix, iy = x0 + pad, y0 + pad
    f_lbl = _get_font(12)
    f_big = _get_font(32, bold=True)
    f_sm = _get_font(12, bold=True)
    circle_d = 86
    left_w = w - pad * 3 - circle_d

    # GEAR (big) + RPM tach to its right.
    gear = int(info.get("gear", 0)) + 1
    screen.blit(f_lbl.render("GEAR", True, LABEL), (ix, iy))
    screen.blit(f_big.render(str(gear), True, HUD), (ix, iy + 11))
    gw = 30
    tx, ty, tw, th = ix + gw + 6, iy + 4, left_w - gw - 6, 11
    pygame.draw.rect(screen, TRACK_BG, pygame.Rect(tx, ty, tw, th), border_radius=6)
    rpm = float(info.get("rpm", 0.0))
    fw = int(tw * min(rpm / REDLINE_RPM, 1.0))
    if fw > 0:
        pygame.draw.rect(screen, GRIP_OVER if rpm >= SHIFT_RPM else RPM_COL,
                         pygame.Rect(tx, ty, fw, th), border_radius=6)
    rl_x = tx + int(tw * SHIFT_RPM / REDLINE_RPM)
    pygame.draw.line(screen, GRIP_OVER, (rl_x, ty - 1), (rl_x, ty + th + 1), 2)
    screen.blit(f_lbl.render(f"{rpm:>4.0f} rpm", True, HUD_DIM), (tx, ty + th + 3))

    # Tyre grip bars — rear load is the worse of slip-angle and wheelspin.
    fl = abs(float(info.get("slip_f", 0.0))) / SLIP_PEAK
    rr = max(abs(float(info.get("slip_r", 0.0))) / SLIP_PEAK,
             abs(float(info.get("slip_ratio", 0.0))) / SPIN_PEAK)
    bx, bw = ix + 16, left_w - 16
    for k, (lab, load) in enumerate((("F", fl), ("R", rr))):
        ry = iy + 52 + k * 22
        screen.blit(f_sm.render(lab, True, LABEL), (ix, ry))
        pygame.draw.rect(screen, TRACK_BG, pygame.Rect(bx, ry, bw, 13), border_radius=6)
        bw_fill = int(bw * min(load, 1.0))
        col = _grip_color(load)
        if bw_fill > 0:
            pygame.draw.rect(screen, col, pygame.Rect(bx, ry, bw_fill, 13), border_radius=6)
        if load >= 1.0:
            s = f_sm.render("SLIP", True, GRIP_OVER)
            screen.blit(s, (bx + bw - s.get_width() - 4, ry))

    # Friction-circle g-meter: a dot at (lateral, longitudinal) g; out to the ring
    # = at the limit. Inner ring = 1 g. Dot reddens as combined g rises.
    cx, cy = x0 + w - pad - circle_d // 2, y0 + pad + circle_d // 2
    R = circle_d // 2
    pygame.draw.circle(screen, TRACK_BG, (cx, cy), R)
    pygame.draw.circle(screen, HUD_DIM, (cx, cy), R, 1)
    pygame.draw.circle(screen, (72, 76, 86), (cx, cy), max(int(R / G_SCALE), 2), 1)
    pygame.draw.line(screen, (72, 76, 86), (cx - R, cy), (cx + R, cy), 1)
    pygame.draw.line(screen, (72, 76, 86), (cx, cy - R), (cx, cy + R), 1)
    lat, lon = float(info.get("lat_g", 0.0)), float(info.get("long_g", 0.0))
    dx = int(np.clip(lat / G_SCALE, -1.0, 1.0) * R)
    dy = int(np.clip(lon / G_SCALE, -1.0, 1.0) * R)
    gmag = math.hypot(lat, lon)
    pygame.draw.circle(screen, _grip_color(gmag / 1.1), (cx + dx, cy - dy), 5)
    gs = f_lbl.render(f"{gmag:.1f}g", True, HUD)
    screen.blit(gs, (cx - gs.get_width() // 2, cy + R - 13))


MINIMAP_SIZE = 170
MINIMAP_MARGIN = 16
SECTOR_COLS = ((196, 72, 64), (74, 126, 200), (224, 206, 96))


def draw_minimap(screen, track, car, info: dict) -> None:
    """A small overhead track map (top-right) with a sector-tinted position dot."""
    size, pad = MINIMAP_SIZE, 12
    x0, y0 = SCREEN_W - size - MINIMAP_MARGIN, MINIMAP_MARGIN
    bg = pygame.Surface((size, size), pygame.SRCALPHA)
    bg.fill((10, 14, 18, 150))
    screen.blit(bg, (x0, y0))
    pygame.draw.rect(screen, ZONE, pygame.Rect(x0, y0, size, size), 1, border_radius=6)

    cl = track.centerline
    minx, maxx = float(cl[:, 0].min()), float(cl[:, 0].max())
    miny, maxy = float(cl[:, 1].min()), float(cl[:, 1].max())
    span = max(maxx - minx, maxy - miny, 1.0)
    scale = (size - 2 * pad) / span
    ox = (size - 2 * pad - (maxx - minx) * scale) / 2.0
    oy = (size - 2 * pad - (maxy - miny) * scale) / 2.0

    def to_px(pts: np.ndarray) -> np.ndarray:
        px = x0 + pad + ox + (pts[..., 0] - minx) * scale
        py = y0 + pad + oy + (maxy - pts[..., 1]) * scale  # flip y for screen
        return np.stack([px, py], axis=-1)

    for ring in (track.left, track.right):
        pygame.draw.lines(screen, (120, 124, 132), True, to_px(ring).tolist(), 1)
    sl = to_px(np.stack([track.left[0], track.right[0]]))
    pygame.draw.line(screen, START, sl[0], sl[1], 2)

    dot = to_px(np.array([car.s.x, car.s.y]))
    col = SECTOR_COLS[int(info.get("cur_sector", 0)) % 3]
    pygame.draw.circle(screen, col, (int(dot[0]), int(dot[1])), 4)
    pygame.draw.circle(screen, HUD, (int(dot[0]), int(dot[1])), 4, 1)


def _draw_banner(screen, big, text: str, color) -> None:
    """A big centred message with a translucent backdrop, near the top."""
    surf = big.render(text, True, color)
    w, h = surf.get_size()
    x, y = SCREEN_W // 2 - w // 2, 108
    bg = pygame.Surface((w + 44, h + 18), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    screen.blit(bg, (x - 22, y - 9))
    pygame.draw.rect(screen, color, pygame.Rect(x - 22, y - 9, w + 44, h + 18), 2, border_radius=6)
    screen.blit(surf, (x, y))


def _fmt_time(t: float | None) -> str:
    if t is None:
        return "  --.---"
    m, s = divmod(t, 60.0)
    return f"{int(m)}:{s:06.3f}" if m else f"  {s:6.3f}"


def _fmt_lap(t: float | None, valid: bool) -> str:
    if t is None:
        return "  --.---"
    return _fmt_time(t) + ("" if valid else "  (void)")
