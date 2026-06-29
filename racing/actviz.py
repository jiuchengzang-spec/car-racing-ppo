"""A separate pygame window that visualises the policy's activations live.

Opened by ``enjoy.py --viz`` alongside the driving view. Each frame it shows, top
to bottom: the input observation (rangefinders + state + curvature preview + tyre
slip), both actor hidden layers as heatmaps, the action mean/std it's outputting,
the value estimate, and both critic hidden layers. Hidden units are tanh outputs
in [-1, 1] — drawn red (positive) / blue (negative), brightness = magnitude — so
you can watch which units light up going into a corner.

Uses pygame-ce's multi-window ``pygame.Window`` API so it's a real separate OS
window; the main driving window stays untouched. If that API isn't available it
raises, and enjoy.py just skips the viz rather than crashing the drive.
"""
from __future__ import annotations

import numpy as np
import pygame

W, H = 560, 840
MARGIN = 20
COLS = 32           # heatmap columns for a 256-unit layer (-> 8 rows)
CELL = 15
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
        """Draw a vector as a COLS-wide heatmap grid; return the y below it."""
        self._label(label, x, y, FG)
        y += 15
        rows = (len(vec) + COLS - 1) // COLS
        for i, v in enumerate(vec):
            cx = x + (i % COLS) * CELL
            cy = y + (i // COLS) * CELL
            pygame.draw.rect(self.surf, _heat(v), (cx, cy, CELL - 1, CELL - 1))
        return y + rows * CELL + 10

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

    # -- public -----------------------------------------------------------

    def update(self, acts: dict) -> None:
        if not self._open:
            return
        s = self.surf
        s.fill(BG)
        x = MARGIN
        s.blit(self.fb.render("policy activations", True, FG), (x, 8))
        y = 36
        y = self._obs_row(acts["obs"], x, y)
        y = self._grid(acts["actor_hidden"][0], x, y, "actor hidden 1  (256, tanh)")
        y = self._grid(acts["actor_hidden"][1], x, y, "actor hidden 2  (256, tanh)")

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

        y = self._grid(acts["critic_hidden"][0], x, y, "critic hidden 1  (256, tanh)")
        y = self._grid(acts["critic_hidden"][1], x, y, "critic hidden 2  (256, tanh)")
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
