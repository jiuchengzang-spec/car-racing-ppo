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

BEAM_ANGLES_DEG = (-90, -60, -40, -25, -12, 0, 12, 25, 40, 60, 90)
MAX_BEAM = 155.0  # m, matched to the 3D renderer's track draw distance (FAR_CLIP)
# so the agent "sees" the same distance of track ahead that a human driver does —
# essential once tracks are randomised and it must react to unseen corners.
MAX_SPEED_REF = 60.0  # m/s, used to normalise speed in the observation

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
        self._steps += 1

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
            # Mid-lap: close out sector(s) as we pass the arc-length thirds.
            sec = min(int(proj.s / self._sec_len), 2)
            while self._cur_sector < sec and self._cur_sector < 2:
                self._complete_sector(self._cur_sector, clt)
                self._cur_sector += 1

        # Progress along the spline (Frenet ds) is the dense driver, with a speed
        # term so going fast pays directly — together they make the fast geometric
        # line the optimum, not the centreline.
        reward = self.progress_w * ds + self.speed_w * max(s.vx, 0.0)
        reward -= self.time_cost  # time cost: standing still loses
        if s.vx < 0.0:
            reward -= self.time_cost  # discourage crawling backwards

        # Sideslip stability: allow performance drifting up to slip_threshold, then
        # penalise the excess — discourages catastrophic spinouts without killing a
        # controllable slide. Keyed on real speed so a parked car stays quiet.
        if s.speed > 2.0:
            beta = abs(math.atan2(s.vy, abs(s.vx) + 1e-3))
            if beta > self.slip_threshold:
                reward -= self.slip_w * (beta - self.slip_threshold)
        # Relaxed comfort term (anti-weave), small so limit-of-grip counter-steer
        # isn't taxed.
        reward -= self.comfort_w * (abs(steer - prev_steer) + abs(throttle - prev_throttle))

        if fully_off:
            pass  # the crash term below handles a full exit
        elif on_grass_now:
            reward -= self.grass_penalty  # ran wide onto the grass — recoverable cost
        else:
            # Optional centre/heading shaping — 0 by default for racing (the racing
            # line is not the centreline), but kept tunable for a civilian feel.
            heading_err = _wrap(proj.heading - s.yaw)
            cte = proj.lateral / self.track.half  # ~[-1, 1] across the tarmac
            reward -= self.cte_w * cte * cte
            reward -= self.heading_w * heading_err * heading_err
        terminated = False
        if fully_off and self.terminate_off_track:
            # Speed-scaled crash penalty: plowing off at corner-entry speed should
            # cost more than the progress banked on the straight before it (so
            # "floor it and crash" stays net-negative vs braking), but not so much
            # that the agent refuses to move at all. The v^2 term keeps it a
            # gradient ("shed speed for the corner"), gentle for low-speed offs.
            reward -= self.crash_penalty + self.crash_speed_w * s.vx * abs(s.vx)
            terminated = True
        if lap_done and self.terminate_on_lap:
            reward += self.lap_bonus
            terminated = True

        truncated = self._steps >= self.max_steps

        info = self._info()
        info["off_track"] = fully_off  # full track exit (terminal); not a grass clip
        info["on_grass"] = on_grass_now
        info["lap_done"] = lap_done

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

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # -- helpers ----------------------------------------------------------

    def _obs(self) -> np.ndarray:
        s = self.car.s
        dists = [
            self.track.cast_ray(s.x, s.y, s.yaw + a, MAX_BEAM) / MAX_BEAM
            for a in self._beams
        ]
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
            "steer_cmd": self._last_action[0],
            "throttle_app": max(self._last_action[1], 0.0),
            "brake_app": max(-self._last_action[1], 0.0),
        }


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
