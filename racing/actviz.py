"""A separate pygame window that visualises the policy's activations live.

Opened by ``enjoy.py --viz`` alongside the driving view. Each frame it shows, top
to bottom: the input observation (rangefinders + state + curvature preview + tyre
slip), both actor hidden layers as heatmaps, the action mean/std it's outputting,
the value estimate, a per-term reward-contribution breakdown (what the last step
actually paid/charged), and both critic hidden layers. Hidden units are tanh
outputs in [-1, 1] — drawn red (positive) / blue (negative), brightness =
magnitude — so you can watch which units light up going into a corner. The heatmap
cells auto-size to the network width, so a wider net (e.g. hidden 512) stays in
the window instead of overflowing.

Uses pygame-ce's multi-window ``pygame.Window`` API so it's a real separate OS
window; the main driving window stays untouched. If that API isn't available it
raises, and enjoy.py just skips the viz rather than crashing the drive.
"""
from __future__ import annotations

import numpy as np
import pygame

W, H = 560, 880
MARGIN = 20
BG = (16, 18, 22)
FG = (208, 212, 220)
DIM = (120, 126, 138)


def _heat(v: float) -> tuple[int, int, int]:
    """Diverging colour for a value in [-1, 1]: blue (neg) .. grey (0) .. red (pos)."""
    v = -1.0 if v < -1.0 else 1.0 if v > 1.0 else float(v)
    base = 28
    if v >= 0.0:
        return (base + int((255 - base) * v), base, base)
    return (base, base, base + int((255 - base) * (-v)))


# Observation layout (must match racing.env._obs): 11 beams, 5 car-state, 3
# curvature-preview, 4 tyre-slip. Used only to label the obs row.
_OBS_GROUPS = (("beams (15)", 15), ("state (5)", 5), ("curv (3)", 3), ("slip (4)", 4))


class ActivationViz:
    def __init__(self) -> None:
        if not hasattr(pygame, "Window"):
            raise RuntimeError("pygame.Window unavailable — need pygame-ce >= 2.4 for a separate viz window")
        pygame.init()
        pygame.font.init()
        self.window = pygame.Window("model activations", (W, H))
        self.surf = self.window.get_surface()
        self.f = pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.fs = pygame.font.SysFont("consolas,menlo,monospace", 11)
        self.fb = pygame.font.SysFont("consolas,menlo,monospace", 15, bold=True)
        self._open = True

    # -- drawing helpers --------------------------------------------------

    def _label(self, text: str, x: int, y: int, color=DIM) -> None:
        self.surf.blit(self.fs.render(text, True, color), (x, y))

    def _grid(self, vec: np.ndarray, x: int, y: int, label: str) -> int:
        """Draw a vector as a heatmap grid sized to FIT the window; return y below it.

        Cells auto-shrink with the layer width (a 512-unit hidden layer packs into the
        same band a 256-unit one used to), so wider nets don't blow past the window.
        """
        n = len(vec)
        avail = W - 2 * MARGIN
        rows = 8 if n > 128 else max(1, (n + 15) // 16)  # ~8 rows for big layers
        cols = (n + rows - 1) // rows
        cell = max(4, min(15, avail // cols))            # fill width, capped 4..15px
        self._label(f"{label}  ({n}, tanh)", x, y, FG)
        y += 15
        for i, v in enumerate(vec):
            cx = x + (i % cols) * cell
            cy = y + (i // cols) * cell
            pygame.draw.rect(self.surf, _heat(v), (cx, cy, cell - 1, cell - 1))
        return y + rows * cell + 10

    def _center_label(self, text: str, cx: float, y: float, color=DIM) -> None:
        surf = self.fs.render(text, True, color)
        self.surf.blit(surf, surf.get_rect(midtop=(int(cx), int(y))))

    def _obs_row(self, obs: np.ndarray, x: int, y: int) -> int:
        self._label("observation  (policy input)", x, y, FG)
        y += 16
        cw = (W - 2 * MARGIN) / len(obs)
        for i, v in enumerate(obs):
            pygame.draw.rect(self.surf, _heat(v), (int(x + i * cw), y, max(int(cw) - 1, 1), 22))
        # group dividers + SHORT centred labels (so they never collide)
        gx = float(x)
        for name, n in _OBS_GROUPS:
            gw = cw * n
            if gx > x:
                pygame.draw.line(self.surf, BG, (int(gx), y), (int(gx), y + 22), 2)
            self._center_label(name, gx + gw / 2.0, y + 25, DIM)
            gx += gw
        return y + 46

    def _bar(self, name: str, val: float, std: float, x: int, y: int) -> None:
        """A centred [-1, 1] bar (steer/throttle mean) with its value and sigma."""
        self.surf.blit(self.f.render(name, True, FG), (x, y))
        bx, by, bw, bh = x + 74, y + 1, 200, 14
        pygame.draw.rect(self.surf, (40, 43, 52), (bx, by, bw, bh), border_radius=4)
        cx = bx + bw // 2
        pos = int(cx + max(-1.0, min(1.0, val)) * (bw // 2 - 2))
        lo, hi = min(cx, pos), max(cx, pos)
        pygame.draw.rect(self.surf, (96, 176, 232), (lo, by, max(hi - lo, 1), bh), border_radius=4)
        pygame.draw.line(self.surf, FG, (cx, by - 1), (cx, by + bh + 1), 1)
        self.surf.blit(self.f.render(f"{val:+.2f}", True, FG), (bx + bw + 12, y))
        self._label(f"σ {std:.2f}", bx + bw + 78, y + 2)

    # Fixed display order for the reward terms (see racing.env step()).
    _TERM_ORDER = ("progress", "speed", "time", "slip", "comfort",
                   "sector", "corner_brake", "grass", "crash", "lap")

    def _reward_breakdown(self, terms: dict, x: int, y: int) -> int:
        """Per-term reward contributions as signed bars (green +, red −); bars are
        scaled to the largest-magnitude term this step so you can see what dominates."""
        total = float(sum(terms.values()))
        self.surf.blit(self.fb.render("reward", True, FG), (x, y))
        tcol = (150, 210, 150) if total >= 0 else (212, 150, 150)
        self.surf.blit(self.f.render(f"Σ {total:+.2f}", True, tcol), (x + 64, y + 1))
        y += 22
        shown = [(k, float(terms.get(k, 0.0))) for k in self._TERM_ORDER]
        scale = max(1e-6, max(abs(v) for _, v in shown))
        bx, bw = x + 96, 150
        cx = bx + bw // 2
        for name, v in shown:
            active = abs(v) > 1e-6
            self._label(name, x, y + 1, FG if active else DIM)
            pygame.draw.rect(self.surf, (40, 43, 52), (bx, y, bw, 11), border_radius=3)
            pygame.draw.line(self.surf, DIM, (cx, y - 1), (cx, y + 12), 1)
            if active:
                w = max(1, int((abs(v) / scale) * (bw // 2 - 2)))
                if v >= 0:
                    pygame.draw.rect(self.surf, (96, 176, 120), (cx, y, w, 11), border_radius=3)
                else:
                    pygame.draw.rect(self.surf, (200, 110, 110), (cx - w, y, w, 11), border_radius=3)
                self.surf.blit(self.fs.render(f"{v:+.2f}", True, FG), (bx + bw + 8, y + 1))
            y += 14
        return y + 8

    # -- public -----------------------------------------------------------

    def update(self, acts: dict, info: dict | None = None) -> None:
        if not self._open:
            return
        s = self.surf
        s.fill(BG)
        x = MARGIN
        s.blit(self.fb.render("policy activations", True, FG), (x, 8))
        y = 36
        y = self._obs_row(acts["obs"], x, y)
        y = self._grid(acts["actor_hidden"][0], x, y, "actor hidden 1")
        y = self._grid(acts["actor_hidden"][1], x, y, "actor hidden 2")

        # outputs
        s.blit(self.fb.render("outputs", True, FG), (x, y))
        y += 22
        mean, std = acts["mean"], acts["std"]
        self._bar("steer", float(mean[0]), float(std[0]), x, y)
        y += 24
        self._bar("throttle", float(mean[1]), float(std[1]), x, y)
        y += 28
        vcol = (150, 210, 150) if acts["value"] >= 0 else (212, 150, 150)
        s.blit(self.f.render(f"value  {acts['value']:+.1f}", True, vcol), (x, y))
        y += 26

        # reward breakdown (what the last env step actually paid / charged)
        terms = (info or {}).get("reward_terms")
        if terms:
            y = self._reward_breakdown(terms, x, y)
        else:
            self._label("reward breakdown — start driving to populate", x, y, DIM)
            y += 22

        y = self._grid(acts["critic_hidden"][0], x, y, "critic hidden 1")
        y = self._grid(acts["critic_hidden"][1], x, y, "critic hidden 2")
        self._label("red = +   blue = −   brightness = |activation|", x, H - 18)
        self.window.flip()

    def closed(self, events) -> bool:
        """True if the user closed the viz window (pass pygame.event.get())."""
        for e in events:
            if e.type == pygame.WINDOWCLOSE and getattr(e, "window", None) == self.window:
                return True
        return False

    def close(self) -> None:
        if self._open:
            try:
                self.window.destroy()
            except Exception:
                pass
            self._open = False
