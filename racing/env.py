"""Gymnasium racing environment shared by the human game and the RL agent.

Observation (all roughly normalised to ~[-1, 1]):
    * N rangefinder beams: distance to the wall ahead at fanned-out angles
    * forward speed vx, lateral speed vy, yaw rate r
    * heading error vs the track tangent, current steering angle
    * curvature preview: signed centreline curvature at fixed look-ahead
      distances, so the agent reads the upcoming corner before the beams do
    * tyre state: front/rear slip angles, rear slip ratio, vehicle sideslip —
      what a driver feels at the limit and uses to catch a slide

Action (Box, continuous):
    * steer in [-1, 1]      (left .. right)
    * throttle in [-1, 1]   (full brake/reverse .. full throttle)

Reward is racing-oriented: forward progress along the track spline (Frenet ds)
plus a speed term, so the optimum is the fast geometric line, not the centreline.
Performance drifting is allowed (only *excess* sideslip is penalised); running
wide onto the grass is a recoverable cost, and only a full track exit ends the
episode with a heavy crash penalty.
"""
from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .car import Car, CarParams
from .track import Track, make_track

# 15 rangefinder beams, packed densely toward the front (4-10 deg apart near 0)
# and sparser to the sides — fine forward vision is what matters for placing the
# car into a corner; the wide beams just need to catch a wall.
BEAM_ANGLES_DEG = (-90, -60, -42, -28, -18, -10, -4, 0, 4, 10, 18, 28, 42, 60, 90)
MAX_BEAM = 155.0  # m, matched to the 3D renderer's track draw distance (FAR_CLIP)
# so the agent "sees" the same distance of track ahead that a human driver does —
# essential once tracks are randomised and it must react to unseen corners.
MAX_SPEED_REF = 60.0  # m/s, used to normalise speed in the observation
# Asymmetric per-tick rate limit on the beams (normalised units, [0,1]). The wave
# the policy saws on comes from a beam clearing a track edge and jumping near->far
# in one tick. That spurious LENGTHENING is non-urgent, so cap how fast a beam may
# grow; SHORTENING (a wall appearing — react now) is left instant. Caps the wobble
# without dulling collision reaction. ~0.05/tick = a full 0->155 m sweep in ~20 ticks.
BEAM_RISE_MAX = 0.05

# Curvature-preview look-ahead distances (m) and observation normalisers.
CURV_PREVIEW_M = (20.0, 50.0, 100.0)
CURV_REF = 25.0  # curvature normaliser: a ~25 m-radius corner saturates to ~1
SLIP_REF = math.radians(30.0)  # slip-angle normaliser (~30 deg = a big slide)
BETA_REF = math.radians(45.0)  # vehicle-sideslip normaliser


class RacingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: str | None = None,
        track_seed: int = 8,
        track_profile: str = "balanced",
        dt: float = 1.0 / 60.0,
        max_steps: int = 4000,
        grass_margin: float = 0.5,
        exit_margin: float = 2.5,
        randomize_track: bool = False,
        track_pool: int = 0,
        car_params: CarParams | None = None,
        terminate_off_track: bool = True,
        terminate_on_lap: bool = True,
        off_track_grip: float = 0.4,
        off_track_rolling: float = 6.0,
        view: str = "hood",
        stage_dist: float = 12.0,
        beam_smooth: float = 0.4,
        progress_w: float = 0.5,
        speed_w: float = 0.003,
        time_cost: float = 0.05,
        slip_w: float = 0.04,
        slip_threshold: float = math.radians(22.0),
        comfort_w: float = 0.003,
        cte_w: float = 0.0,
        heading_w: float = 0.0,
        grass_penalty: float = 0.5,
        crash_penalty: float = 10.0,
        crash_speed_w: float = 0.1,
        lap_bonus: float = 100.0,
        sector_bonus: float = 0.0,
        corner_brake_w: float = 0.0,
        corner_lookahead: float = 25.0,
        corner_alat_safe: float = 12.0,
        spawn_speed: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.view = view  # "hood" (3D in-car) or "top" (overhead)
        self.dt = dt
        self.max_steps = max_steps
        # Graduated track limits: past the painted edge by ``grass_margin`` puts
        # wheels on the grass (a recoverable, lap-voiding cost); past ``exit_margin``
        # is a full track exit (terminal). No instant reset for a minor clip.
        self.grass_margin = grass_margin
        self.exit_margin = exit_margin
        self.randomize_track = randomize_track
        self._base_seed = track_seed
        self.track_profile = track_profile
        # When randomizing, optionally draw from a fixed pool of N seeds (so the
        # agent sees each track many times) instead of an unbounded new seed each
        # episode. Derived deterministically from track_seed so the pool is the
        # same across machines/runs.
        self.track_pool = track_pool
        self._pool_seeds: list[int] = []
        if randomize_track and track_pool > 0:
            rng = np.random.default_rng(track_seed)
            self._pool_seeds = [int(s) for s in rng.integers(0, 1_000_000, size=track_pool)]
        # RL wants an episode per lap (terminate on lap / off-track); the human
        # game keeps driving through both, so it flips these off.
        self.terminate_off_track = terminate_off_track
        self.terminate_on_lap = terminate_on_lap
        self.off_track_grip = off_track_grip
        self.off_track_rolling = off_track_rolling
        self.stage_dist = stage_dist  # spawn this far before the line; clock waits

        # Rangefinder beams are filtered before the policy sees them: a beam grazing
        # a wall during a turn jumps near<->far, which was making the policy saw the
        # wheel left-right. We rate-limit how fast a beam may LENGTHEN (the spurious
        # clear-to-far jump that drives the wobble) while letting it SHORTEN instantly
        # (a wall appearing — react now), then EMA-smooth the residual. The renderer
        # draws the filtered values too, so the beams you see match what the agent
        # reacts to. beam_smooth in [0, 1): 0 = rate-limit only, higher = more EMA/lag.
        self.beam_smooth = beam_smooth
        self._beam_dist: np.ndarray | None = None  # filtered normalised beam distances

        # Reward shaping (RL only — the human game ignores reward). The philosophy
        # is *exploit the limits, minimise lap time*, not lane-keeping: progress
        # along the spline (Frenet ds) plus a speed term is the dense driver, so the
        # optimum is the fast geometric line. ``slip_w``/``slip_threshold`` allow
        # performance drifting and only punish the excess sideslip of a spinout;
        # ``comfort_w`` is small so aggressive limit-of-grip counter-steer isn't
        # taxed. The centre/heading terms (``cte_w``/``heading_w``) default to 0 —
        # kept as knobs for a "civilian" tune, but off for racing.
        self.progress_w = progress_w
        self.speed_w = speed_w
        self.time_cost = time_cost
        self.slip_w = slip_w
        self.slip_threshold = slip_threshold
        self.comfort_w = comfort_w
        self.cte_w = cte_w
        self.heading_w = heading_w
        self.grass_penalty = grass_penalty
        self.crash_penalty = crash_penalty
        self.crash_speed_w = crash_speed_w
        self.lap_bonus = lap_bonus
        # Checkpoint reward: paid each time the car reaches a new track third (sector
        # 1/3, 2/3). A denser "get all the way around" gradient on top of the finish
        # bonus, so the agent is pulled toward closing the lap instead of banking
        # progress and crashing. 0 keeps the original behaviour.
        self.sector_bonus = sector_bonus
        # Corner-braking: penalise predicted over-speed (v^2 * upcoming-curvature
        # beyond corner_alat_safe) for the corner corner_lookahead metres ahead.
        self.corner_brake_w = corner_brake_w
        self.corner_lookahead = corner_lookahead
        self.corner_alat_safe = corner_alat_safe
        self.spawn_speed = spawn_speed  # (lo, hi) m/s rolling start, or None

        self.track: Track = make_track(seed=track_seed, profile=track_profile)
        self.car = Car(car_params)

        self._beams = np.radians(np.array(BEAM_ANGLES_DEG, dtype=np.float64))
        # beams + [vx, vy, r, heading_err, steer] + curvature preview + tyre state
        n_obs = len(self._beams) + 5 + len(CURV_PREVIEW_M) + 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(n_obs,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        self._sec_len = self.track.length / 3.0  # arc length of one sector

        # Per-lap timing state (wiped by reset / restage).
        self._steps = 0
        self._prev_vx = 0.0   # previous forward speed, for the longitudinal-g telemetry
        self._long_g = 0.0    # last longitudinal g (accel +, brake -), for the HUD g-meter
        self._prev_s = 0.0
        self._lap_progress = 0.0  # unwrapped distance travelled along the track
        self._lap_start_step = 0
        self._lap_valid = True
        self._timing_armed = False  # clock starts on the first finish-line crossing
        self._cur_sector = 0
        self._sector_start_time = 0.0
        self._cur_sector_splits: list[float | None] = [None, None, None]
        self._sector_delta: list[float | None] = [None, None, None]

        # Records that survive a restage (only reset() wipes them).
        self._lap_count = 0
        self._last_lap_time: float | None = None
        self._last_lap_valid = True
        self._last_sectors: list[float | None] = [None, None, None]
        self._best_lap_time: float | None = None
        self._best_sectors: list[float | None] = [None, None, None]
        self._lap_times: list[float] = []  # every valid lap, in order

        # Set by the human game (mouse mode) so the renderer can draw the steering
        # zone + knob; None means no overlay. Purely a UI concern.
        self.steer_overlay: float | None = None
        self._last_action = (0.0, 0.0)  # (steer, throttle) — for the HUD input bars
        self._renderer = None

    # -- gym API ----------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if self.randomize_track:
            if self._pool_seeds:
                s = int(self.np_random.choice(self._pool_seeds))
            else:
                s = int(self.np_random.integers(0, 1_000_000))
            self.track = make_track(seed=s, profile=self.track_profile)
            self._sec_len = self.track.length / 3.0
        self._records_reset()
        self._stage_and_reset_lap()
        return self._obs(), self._info()

    def restage(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Re-place the car at the staging spot and reset only the *current* lap.

        Best lap, best sectors and the full valid-lap list are kept — the human
        game's R key uses this so a reset costs you position, not your records.
        """
        self._stage_and_reset_lap()
        return self._obs(), self._info()

    def _records_reset(self) -> None:
        self._lap_count = 0
        self._last_lap_time = None
        self._last_lap_valid = True
        self._last_sectors = [None, None, None]
        self._best_lap_time = None
        self._best_sectors = [None, None, None]
        self._lap_times = []

    def _stage_and_reset_lap(self) -> None:
        # Spawn a little before the start/finish line; the clock stays at zero
        # until the car first crosses it (see the arming branch in step()).
        x, y, yaw = self.track.pose_at(self.track.length - self.stage_dist)
        self.car.reset(x, y, yaw)
        if self.spawn_speed is not None:
            # Rolling start at racing speed so the policy must handle high momentum
            # from the first tick (the "Grandma Driver" fix — no creeping to safety).
            lo, hi = self.spawn_speed
            self.car.rolling_start(float(self.np_random.uniform(lo, hi)))
        self._last_action = (0.0, 0.0)  # no phantom action-delta on the first tick
        self._prev_vx = self.car.s.vx   # seed from spawn speed so first-frame long_g isn't a spike
        self._long_g = 0.0
        self._beam_dist = None  # fresh start: no carry-over smoothing from last episode
        self._update_beams()
        self._steps = 0
        self._prev_s = self.track.project(x, y).s
        self._lap_progress = 0.0
        self._lap_start_step = 0
        self._lap_valid = True
        self._timing_armed = False
        self._begin_sectors()

    def _begin_sectors(self) -> None:
        self._cur_sector = 0
        self._sector_start_time = 0.0
        self._cur_sector_splits = [None, None, None]
        self._sector_delta = [None, None, None]

    def _complete_sector(self, i: int, end_time: float) -> None:
        """Record sector ``i``'s split and its delta vs the best so far."""
        split = end_time - self._sector_start_time
        self._cur_sector_splits[i] = split
        ref = self._best_sectors[i]
        self._sector_delta[i] = (split - ref) if ref is not None else None
        self._sector_start_time = end_time

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        steer, throttle = float(action[0]), float(action[1])
        prev_steer, prev_throttle = self._last_action  # last tick's command (anti-weave)
        self._last_action = (steer, throttle)
        s = self.car.s

        # The surface under the car right now sets grip for this tick: grass off
        # the track is slippery and draggy, so leaving the tarmac costs time.
        surface = self.track.project(s.x, s.y)
        on_grass = abs(surface.lateral) > self.track.half + self.grass_margin
        grip_mult = self.off_track_grip if on_grass else 1.0
        rolling_mult = self.off_track_rolling if on_grass else 1.0
        self.car.step(steer, throttle, self.dt, grip_mult=grip_mult, rolling_mult=rolling_mult)
        # Longitudinal g from the speed change this tick (for the HUD friction circle).
        self._long_g = (self.car.s.vx - self._prev_vx) / self.dt / 9.81
        self._prev_vx = self.car.s.vx
        self._steps += 1
        self._update_beams()  # cast + EMA-smooth the rangefinders for this tick

        proj = self.track.project(s.x, s.y)

        # Progress along the lap, handling the start/finish wrap.
        ds = proj.s - self._prev_s
        crossed_finish = False
        if ds < -self.track.length * 0.5:
            ds += self.track.length
            crossed_finish = True  # passed the start/finish line going forward
        elif ds > self.track.length * 0.5:
            ds -= self.track.length  # crossed it backwards
        self._prev_s = proj.s
        self._lap_progress += ds

        on_grass_now = abs(proj.lateral) > self.track.half + self.grass_margin
        fully_off = abs(proj.lateral) > self.track.half + self.exit_margin
        if on_grass_now:
            self._lap_valid = False  # a wheel on the grass voids the lap

        clt = (self._steps - self._lap_start_step) * self.dt  # current-lap time

        lap_done = False
        sectors_gained = 0  # new track-thirds reached this step (for the checkpoint bonus)
        if crossed_finish and not self._timing_armed:
            # First crossing out of the staging spot: this is where the clock and
            # sector 1 actually start — the launch run before it doesn't count.
            self._timing_armed = True
            self._lap_start_step = self._steps
            self._lap_progress = 0.0
            self._lap_valid = not on_grass_now
            self._begin_sectors()
        elif crossed_finish and self._lap_progress > self.track.length * 0.5:
            # A full lap: close out the final sector, bank the lap, start the next.
            lap_done = True
            lap_time = clt
            self._complete_sector(2, lap_time)
            self._lap_count += 1
            self._last_lap_time = lap_time
            self._last_lap_valid = self._lap_valid
            self._last_sectors = list(self._cur_sector_splits)
            if self._lap_valid:
                self._lap_times.append(lap_time)
                if self._best_lap_time is None or lap_time < self._best_lap_time:
                    self._best_lap_time = lap_time
                for k in range(3):
                    sp = self._cur_sector_splits[k]
                    if sp is not None and (
                        self._best_sectors[k] is None or sp < self._best_sectors[k]
                    ):
                        self._best_sectors[k] = sp
            self._lap_start_step = self._steps
            self._lap_progress -= self.track.length
            self._lap_valid = not on_grass_now
            self._begin_sectors()
        elif self._timing_armed:
            # Mid-lap: close out sector(s) as we pass the arc-length thirds. Each new
            # third reached pays sector_bonus (below) — a denser "get all the way
            # around" signal that pulls the agent toward closing the lap.
            sec = min(int(proj.s / self._sec_len), 2)
            while self._cur_sector < sec and self._cur_sector < 2:
                self._complete_sector(self._cur_sector, clt)
                self._cur_sector += 1
                sectors_gained += 1

        # Reward, tracked term-by-term so `enjoy.py --viz` can show what is driving
        # (or penalising) the agent this step. The step reward is the sum of terms.
        terms = {}
        # Progress along the spline (Frenet ds) is the dense driver, with a speed
        # term so going fast pays directly — together they make the fast geometric
        # line the optimum, not the centreline.
        terms["progress"] = self.progress_w * ds
        terms["speed"] = self.speed_w * max(s.vx, 0.0)
        # Time cost: standing still loses; doubled while crawling backwards.
        terms["time"] = -self.time_cost * (2.0 if s.vx < 0.0 else 1.0)
        # Sideslip stability: allow performance drifting up to slip_threshold, then
        # penalise the excess — discourages catastrophic spinouts without killing a
        # controllable slide. Keyed on real speed so a parked car stays quiet.
        slip_pen = 0.0
        if s.speed > 2.0:
            beta = abs(math.atan2(s.vy, abs(s.vx) + 1e-3))
            if beta > self.slip_threshold:
                slip_pen = -self.slip_w * (beta - self.slip_threshold)
        terms["slip"] = slip_pen
        # Relaxed comfort term (anti-weave), small so limit-of-grip counter-steer
        # isn't taxed.
        terms["comfort"] = -self.comfort_w * (abs(steer - prev_steer) + abs(throttle - prev_throttle))
        # Checkpoint bonus for reaching a new track third this step (see sector_bonus).
        terms["sector"] = self.sector_bonus * sectors_gained
        # Corner-braking: penalise carrying too much speed into the corner AHEAD —
        # predicted lateral accel (v^2 * upcoming curvature) beyond a safe budget. This
        # teaches the agent to BRAKE for deep-angle curves (match speed to the corner)
        # instead of plowing in and running wide — a generic skill that helps on
        # unseen/sharp corners. 0 = off.
        cb_pen = 0.0
        if self.corner_brake_w > 0.0 and s.vx > 0.0:
            cur_curv = abs(self.track.curvature_at(proj.s))
            upcoming_curv = abs(self.track.curvature_at(proj.s + self.corner_lookahead))
            # Only brake for a corner that is still TIGHTENING ahead (entry). Once the
            # curve is opening up (upcoming <= current), release the penalty so the
            # agent gets back on power on EXIT instead of coasting out of the corner.
            if upcoming_curv > cur_curv:
                a_lat_pred = s.vx * s.vx * upcoming_curv  # predicted lateral accel into the corner
                cb_pen = -self.corner_brake_w * max(0.0, a_lat_pred - self.corner_alat_safe)
        terms["corner_brake"] = cb_pen
        # Graduated track limits: a grass clip is a recoverable per-step cost; a full
        # exit is handled by the terminal crash term below (no grass/lane double-count).
        terms["grass"] = 0.0
        terms["lane"] = 0.0
        if fully_off:
            pass  # the crash term below handles a full exit
        elif on_grass_now:
            terms["grass"] = -self.grass_penalty  # ran wide onto the grass — recoverable cost
        else:
            # Optional centre/heading shaping — 0 by default for racing (the racing
            # line is not the centreline), but kept tunable for a civilian feel.
            heading_err = _wrap(proj.heading - s.yaw)
            cte = proj.lateral / self.track.half  # ~[-1, 1] across the tarmac
            terms["lane"] = -(self.cte_w * cte * cte + self.heading_w * heading_err * heading_err)
        terminated = False
        terms["crash"] = 0.0
        if fully_off and self.terminate_off_track:
            # Speed-scaled crash penalty: plowing off at corner-entry speed should
            # cost more than the progress banked on the straight before it (so
            # "floor it and crash" stays net-negative vs braking), but not so much
            # that the agent refuses to move at all. The v^2 term keeps it a
            # gradient ("shed speed for the corner"), gentle for low-speed offs.
            terms["crash"] = -(self.crash_penalty + self.crash_speed_w * s.vx * abs(s.vx))
            terminated = True
        terms["lap"] = 0.0
        if lap_done and self.terminate_on_lap:
            terms["lap"] = self.lap_bonus
            terminated = True
        reward = float(sum(terms.values()))

        truncated = self._steps >= self.max_steps

        info = self._info()
        info["off_track"] = fully_off  # full track exit (terminal); not a grass clip
        info["on_grass"] = on_grass_now
        info["lap_done"] = lap_done
        info["reward"] = reward
        info["reward_terms"] = terms  # per-term breakdown for the --viz HUD

        if self.render_mode == "human":
            self.render()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self._renderer is None:
            if self.view == "top":
                from .render import PygameRenderer

                self._renderer = PygameRenderer(self.track, self.car.p, self.render_mode)
            else:
                from .render3d import HoodCamRenderer

                self._renderer = HoodCamRenderer(self.track, self.car.p, self.render_mode)
        return self._renderer.draw(self.car, self._beams, self._info())

    def toggle_view(self) -> str:
        """Swap between the 3D hood cam and the top-down overhead view, live.

        The renderer is just dropped (rebuilt on the next render, re-using the same
        window) — we don't ``close()`` it, since that calls ``pygame.quit()`` and
        would tear down the whole session (including any activation-viz window).
        Returns the new view name.
        """
        self.view = "top" if self.view == "hood" else "hood"
        self._renderer = None
        return self.view

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- helpers ----------------------------------------------------------

    def _cast_beams(self) -> np.ndarray:
        """Raw normalised rangefinder distances at the current pose ([0, 1])."""
        s = self.car.s
        return np.array(
            [self.track.cast_ray(s.x, s.y, s.yaw + a, MAX_BEAM) / MAX_BEAM for a in self._beams],
            dtype=np.float32,
        )

    def _update_beams(self) -> None:
        """Cast the beams, rate-limit lengthening to kill the clear-to-far wobble, EMA.

        A beam clearing a track edge jumps near->far in one tick — a spurious step that
        sawed the wheel. We cap how fast each beam may GROW (BEAM_RISE_MAX/tick) but let
        it SHRINK instantly so a wall appearing still reads immediately; the EMA then
        smooths the residual. On a fresh episode the filter is primed from the first cast.
        """
        raw = self._cast_beams()
        if self._beam_dist is None:  # fresh episode: no carry-over from the last one
            self._beam_dist = raw
            return
        delta = raw - self._beam_dist
        delta = np.where(delta > 0.0, np.minimum(delta, BEAM_RISE_MAX), delta)  # cap rise only
        target = self._beam_dist + delta
        if self.beam_smooth <= 0.0:
            self._beam_dist = target.astype(np.float32)
        else:
            k = self.beam_smooth
            self._beam_dist = (k * self._beam_dist + (1.0 - k) * target).astype(np.float32)

    def _obs(self) -> np.ndarray:
        s = self.car.s
        dists = list(self._beam_dist)  # smoothed (see _update_beams)
        proj = self.track.project(s.x, s.y)
        heading_err = _wrap(proj.heading - s.yaw)
        extra = [
            np.clip(s.vx / MAX_SPEED_REF, -1.0, 1.0),
            np.clip(s.vy / 15.0, -1.0, 1.0),
            np.clip(s.r / 3.0, -1.0, 1.0),
            np.clip(heading_err / math.pi, -1.0, 1.0),
            np.clip(s.steer / self.car.p.max_steer, -1.0, 1.0),
        ]
        # Curvature preview: signed centreline curvature at fixed look-ahead
        # distances, so the policy can read the corner (direction + tightness) and
        # set up the line/brake point before the wall beams pick it up.
        curv = [
            np.clip(self.track.curvature_at(proj.s + d) * CURV_REF, -1.0, 1.0)
            for d in CURV_PREVIEW_M
        ]
        # Tyre state at the limit: front/rear slip angles, rear slip ratio, and the
        # vehicle sideslip beta — the cues a driver uses to feel and catch a slide.
        kappa = (s.wheel_v_r - s.vx) / max(abs(s.vx), self.car.p.slip_vx_floor)
        beta = math.atan2(s.vy, abs(s.vx) + 1e-3)
        tyre = [
            np.clip(s.slip_f / SLIP_REF, -1.0, 1.0),
            np.clip(s.slip_r / SLIP_REF, -1.0, 1.0),
            np.clip(kappa, -1.0, 1.0),
            np.clip(beta / BETA_REF, -1.0, 1.0),
        ]
        return np.array(dists + extra + curv + tyre, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        s = self.car.s
        bs = self._best_sectors
        theo = float(sum(bs)) if all(v is not None for v in bs) else None
        return {
            "speed": s.speed,
            "speed_kmh": s.speed * 3.6,
            "vx": s.vx,
            "progress": self._lap_progress,
            "lap_fraction": float(np.clip(self._lap_progress / self.track.length, 0, 1)),
            "time": self._steps * self.dt,
            "current_lap_time": (self._steps - self._lap_start_step) * self.dt if self._timing_armed else 0.0,
            "timing_armed": self._timing_armed,
            "cur_sector": self._cur_sector,
            "sector_splits": list(self._cur_sector_splits),
            "sector_delta": list(self._sector_delta),
            "last_sectors": list(self._last_sectors),
            "best_sectors": list(bs),
            "theoretical_best": theo,
            "last_lap_time": self._last_lap_time,
            "last_lap_valid": self._last_lap_valid,
            "best_lap_time": self._best_lap_time,
            "lap_count": self._lap_count,
            "valid_laps": len(self._lap_times),
            "lap_valid": self._lap_valid,
            "steer_overlay": self.steer_overlay,
            # Smoothed beam distances (metres) so the renderer draws the stable
            # values the agent sees, not a fresh (jittery) raycast.
            "beam_dists_m": None if self._beam_dist is None else (self._beam_dist * MAX_BEAM).tolist(),
            "steer_cmd": self._last_action[0],
            "throttle_app": max(self._last_action[1], 0.0),
            "brake_app": max(-self._last_action[1], 0.0),
            # Live physics telemetry for the racing HUD (gear/rpm tach, per-axle tyre
            # grip/slip bars, friction-circle g-meter) — data the sim already computes.
            "gear": s.gear,
            "rpm": s.engine_rpm,
            "slip_f": s.slip_f,                 # front slip angle (rad)
            "slip_r": s.slip_r,                 # rear slip angle (rad)
            "slip_ratio": (s.wheel_v_r - s.vx) / max(abs(s.vx), self.car.p.slip_vx_floor),  # rear wheelspin
            "beta": math.atan2(s.vy, abs(s.vx) + 1e-3),  # vehicle sideslip (rad)
            "lat_g": s.vx * s.r / 9.81,         # lateral (centripetal) g
            "long_g": self._long_g,             # longitudinal g (accel/brake), tracked in step()
            "vy": s.vy,
        }


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
