"""Vehicle dynamics: a dynamic bicycle model with sim-style grip.

Grip is modelled the way racing games / sims do it:

  * **Pacejka "magic formula" tyres** — lateral force rises with slip angle to a
    peak (~6 deg) then falls off as the tyre slides, instead of a straight line.
  * **Friction circle (combined slip)** — each axle has one grip budget shared
    between cornering and drive/brake. The driven rear has a spinning wheel: its
    longitudinal force comes from *slip ratio* (wheel surface speed vs road) on
    the same Pacejka curve as the lateral force, so flooring it lights up the
    rears, bleeds cornering grip and steps the car out — and self-limits, so a
    lift hooks it back. The front shares its budget via a friction ellipse, so
    you can trail-brake and rotate.
  * **Aero downforce** — vertical tyre load, and therefore grip, grows with
    speed squared. This is the F1 signature: slippery when slow, planted when
    fast.
  * **Longitudinal weight transfer** — braking loads the front (turn-in),
    accelerating loads the rear (traction).
  * **Geared engine** — a torque curve over rpm driving an automatic 6-speed
    gearbox: force steps down on each upshift then recovers as revs climb, so
    the car "catches its breath" between gears instead of pulling endlessly.

State is the car in the global plane (x, y, yaw) plus body-frame velocities
(vx forward, vy lateral, r yaw-rate). Integrated with sub-steps for stability,
with lateral grip faded out near standstill so the car doesn't jitter when parked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

G = 9.81

# Pacejka lateral coefficients (stiffness / shape / curvature). Peak near ~6 deg.
PAC_B = 9.5
PAC_C = 1.5
PAC_E = 0.97

# Engine torque curve (Nm vs rpm): builds off idle, fat plateau ~4-5k, tapers to
# the limiter. Peak power ~462 kW (~620 hp) near 7000 rpm (630 Nm · 733 rad/s).
_TQ_RPM = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 8200], dtype=np.float64)
_TQ_NM = np.array([360, 520, 650, 710, 715, 690, 630, 540, 500], dtype=np.float64)
_RPM_PER_RADS = 60.0 / (2.0 * math.pi)  # rad/s -> rpm


def _engine_torque(rpm: float) -> float:
    """Engine torque (Nm) at the given rpm (np.interp clamps to the curve ends)."""
    return float(np.interp(rpm, _TQ_RPM, _TQ_NM))


@dataclass
class CarParams:
    # A downforce-equipped sports-racer: ~1000 kg, ~620 hp, ~270 km/h, 0-100 ~2.0s.
    mass: float = 1000.0  # kg
    inertia_z: float = 1650.0  # kg m^2 (yaw); higher -> yaw builds, less go-kart darting
    lf: float = 1.70  # m, CG to front axle (larger -> more static load on rear)
    lr: float = 1.55  # m, CG to rear axle (lf+lr ~3.3m wheelbase, F1-long)
    h_cg: float = 0.32  # m, CG height (drives weight transfer)
    width: float = 1.9  # m
    mu: float = 1.7  # peak tyre-road friction (slicks)
    # Rear tyres are wider than the fronts (an F1 staple: ~405 vs 305 mm), so the
    # rear axle has more grip. With the tyre relaxation model now catching slides
    # naturally, this can sit closer to neutral for a livelier, pointier car
    # without the snap-spins that used to force it higher.
    rear_grip_bias: float = 1.25  # multiplies rear-axle grip

    max_engine_force: float = 18000.0  # N, used only for braking/reverse scaling now
    # Automatic 6-speed gearbox. Wheel force = engine_torque · total_ratio /
    # wheel_radius, where total_ratio = gear_ratios[gear] · final_drive; engine
    # rpm = (road wheel speed / wheel_radius) · total_ratio. The discrete ratios
    # give stepped delivery: each upshift drops rpm back into the torque plateau,
    # so thrust dips then recovers (the "catch its breath" feel) instead of the
    # old gearless Power/speed hyperbola. Ratios fall from a punchy 1st to a
    # long 6th that runs out against aero drag at ~270 km/h — well short of the
    # 6th-gear rev limiter (~340 km/h), so the top end is drag-limited, not revs.
    gear_ratios: tuple[float, ...] = (3.44, 2.56, 1.97, 1.53, 1.19, 0.94)
    final_drive: float = 3.2
    idle_rpm: float = 1200.0  # rpm floor for torque lookup (off-idle launch)
    redline_rpm: float = 8200.0  # rpm ceiling (limiter)
    shift_up_rpm: float = 7800.0  # auto-upshift threshold
    shift_down_rpm: float = 3200.0  # auto-downshift threshold (wide hysteresis -> no hunting)
    shift_time: float = 0.15  # s of torque cut during a gearchange
    # Driven-wheel rotational model, so flooring it can light up the rears. The
    # tyre's longitudinal force comes from *slip ratio* (wheel surface speed vs
    # road speed) exactly as the lateral force comes from slip angle; spinning
    # the wheel bleeds thrust AND lateral grip, so the rear steps out and the
    # spin self-limits — lift and it hooks back up.
    wheel_radius: float = 0.33  # m, driven wheel rolling radius
    wheel_inertia: float = 2.4  # kg m^2, effective (wheels + drivetrain, geared)
    slip_vx_floor: float = 2.5  # m/s, floor on road speed in the slip-ratio denominator
    max_brake_force: float = 32000.0  # N at full brake (grip-limited ~2g low, ~3.8g high)
    brake_bias: float = 0.55  # fraction of brake force to the front axle (forward = stable)

    # Traction control: caps driven-wheel slip ratio so binary keyboard throttle
    # can't instantly light up the rears from low speed. It's speed-tapered — full
    # authority off the line (where a spin is uncontrollable and just kills the
    # launch), releasing as speed builds so mid-corner throttle-steer stays lively
    # and slidey. Real driver aid, toggleable; off = full wheelspin physics.
    traction_control: bool = True
    tc_slip: float = 0.12  # slip-ratio ceiling under full TC (just under the grip peak ~0.18)
    tc_full_speed: float = 8.0  # m/s; below this TC is at full authority
    tc_off_speed: float = 22.0  # m/s; above this TC has fully released

    drag_coef: float = 1.05  # N / (m/s)^2 — profile drag, sets top speed (~270 km/h)
    downforce_coef: float = 2.6  # N / (m/s)^2 — added vertical load with speed
    aero_rear_bias: float = 0.50  # fraction of downforce on the rear axle (forward bias = sharper high-speed turn-in)
    rolling_resistance: float = 120.0  # N, constant roll drag while moving

    max_steer: float = math.radians(30.0)  # rad at full lock
    # Steering-wheel speed scales with car speed: eager near standstill (snappy
    # hairpins, easy to provoke rotation) easing to calm at speed (no go-kart
    # darting at 250 km/h). A flat rate can't do both at once.
    steer_rate_lo: float = math.radians(300.0)  # rad/s near standstill
    steer_rate_hi: float = math.radians(110.0)  # rad/s at steer_speed_ref and above
    # Full lock at a crawl, easing to this fraction at speed for stable turn-in.
    high_speed_steer: float = 0.36
    steer_speed_ref: float = 55.0  # m/s at which the easing reaches its floor
    # Tyre relaxation length: the contact-patch side force does not appear the
    # instant the slip angle changes — it builds up as the tyre rolls through
    # ~this distance. Modelled as a first-order lag on each axle's slip angle.
    # This is real physics (the point-contact bicycle model omits it); it catches
    # a slide smoothly, which is what lets the rear be lively yet recoverable —
    # the job the old artificial yaw-damping moment used to fake.
    relax_len: float = 0.55  # m, slip-angle relaxation length

    # Camera weight-transfer pitch (visual only): how the hood-cam view dives
    # under braking and squats under power, and how fast it settles. Stiff and
    # quick reads as an F1 car; soft and slow reads as an old GT cruiser.
    dive_pitch: float = math.radians(1.3)  # nose-down under full braking
    squat_pitch: float = math.radians(0.6)  # nose-up under full throttle
    pitch_smooth: float = 0.28  # how fast the pitch settles (per frame)

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @property
    def length(self) -> float:
        return self.wheelbase + 0.9  # a little overhang, for drawing only


# Named handling presets — switch with `play.py --handling NAME`. Each returns a
# fresh CarParams so callers can mutate without disturbing the registry. "nimble"
# is the default tune (quick, planted at speed, traction control on); "boat" is
# the original validated feel (understeery, soft slow weight transfer, no TC).
def _nimble() -> CarParams:
    return CarParams()


def _boat() -> CarParams:
    return CarParams(
        rear_grip_bias=1.35,
        downforce_coef=1.9,
        aero_rear_bias=0.56,
        traction_control=False,
        dive_pitch=math.radians(2.6),
        squat_pitch=math.radians(1.2),
        pitch_smooth=0.14,
    )


# A *fair* oversteer/understeer pair: the same fundamental car as `nimble` (same
# tyres, grip, engine, brakes, traction control) with only the race-engineer
# setup knobs moved — weight split (lf/lr at fixed wheelbase), aero balance, and
# brake bias — symmetrically about the nimble baseline (lf/lr 1.70/1.55,
# aero_rear_bias 0.50, brake_bias 0.55). So a head-to-head is decided by setup,
# not a hidden hardware advantage.
def _understeer() -> CarParams:
    return CarParams(  # weight & aero & brakes forward -> front gives up first, stable
        lf=1.55, lr=1.70,  # 47.7% rear static load (forward of neutral)
        aero_rear_bias=0.56,
        brake_bias=0.62,
    )


def _oversteer() -> CarParams:
    return CarParams(  # weight & aero & brakes rearward -> rotates, tail-happy on entry
        lf=1.85, lr=1.40,  # 56.9% rear static load (rearward of neutral)
        aero_rear_bias=0.44,
        brake_bias=0.48,
    )


HANDLING_PRESETS: dict[str, "callable[[], CarParams]"] = {
    "nimble": _nimble,
    "boat": _boat,
    "un": _understeer,
    "ov": _oversteer,
}


def handling_preset(name: str) -> CarParams:
    """Return a fresh CarParams for a named preset (see HANDLING_PRESETS)."""
    try:
        return HANDLING_PRESETS[name]()
    except KeyError:
        raise ValueError(
            f"unknown handling preset {name!r}; choose from {sorted(HANDLING_PRESETS)}"
        ) from None


@dataclass
class CarState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0  # body-frame forward velocity (m/s)
    vy: float = 0.0  # body-frame lateral velocity (m/s)
    r: float = 0.0  # yaw rate (rad/s)
    steer: float = 0.0  # current front-wheel angle (rad)
    wheel_v_r: float = 0.0  # driven rear-wheel surface speed (m/s); >vx == wheelspin
    slip_f: float = 0.0  # relaxed front slip angle (rad); lags the geometric angle
    slip_r: float = 0.0  # relaxed rear slip angle (rad)
    gear: int = 0  # current gear index (0 = 1st)
    shift_timer: float = 0.0  # s remaining of torque-cut during a gearchange
    engine_rpm: float = 0.0  # last engine rpm (road-speed derived; for HUD/debug)

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.vx, self.vy, self.r], dtype=np.float64)


class Car:
    """A single car. Call :meth:`step` with normalised controls each tick."""

    def __init__(self, params: CarParams | None = None) -> None:
        self.p = params or CarParams()
        self.s = CarState()

    def reset(self, x: float, y: float, yaw: float) -> None:
        self.s = CarState(x=x, y=y, yaw=yaw)

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        self.s.x, self.s.y, self.s.yaw = x, y, yaw

    def rolling_start(self, speed: float) -> None:
        """Begin already rolling straight ahead at ``speed`` m/s (for RL inits).

        Sets the forward velocity and matches the driven-wheel surface speed (no
        launch wheelspin), then selects the lowest gear whose rpm sits at/under the
        upshift point — so the first throttle isn't bouncing off the limiter. Used
        to spawn episodes at racing speed and break the "drive 5 km/h to never
        crash" local optimum.
        """
        s, p = self.s, self.p
        s.vx = float(speed)
        s.vy = 0.0
        s.r = 0.0
        s.wheel_v_r = float(speed)
        s.gear = len(p.gear_ratios) - 1
        for g in range(len(p.gear_ratios)):
            if self._engine_rpm(speed, g) <= p.shift_up_rpm:
                s.gear = g
                break
        s.engine_rpm = self._engine_rpm(speed, s.gear)

    def step(
        self,
        steer_cmd: float,
        throttle: float,
        dt: float,
        substeps: int = 10,
        grip_mult: float = 1.0,
        rolling_mult: float = 1.0,
    ) -> None:
        """Advance the car.

        steer_cmd, throttle in [-1, 1]. Positive throttle drives, negative brakes
        (and reverses once stopped). ``grip_mult`` / ``rolling_mult`` scale tyre
        grip and rolling drag for the current surface (e.g. grass off-track).
        """
        steer_cmd = float(np.clip(steer_cmd, -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))
        speed_frac = min(abs(self.s.vx) / self.p.steer_speed_ref, 1.0)
        scale = 1.0 - (1.0 - self.p.high_speed_steer) * speed_frac
        target_steer = steer_cmd * self.p.max_steer * scale
        steer_rate = self.p.steer_rate_lo + (self.p.steer_rate_hi - self.p.steer_rate_lo) * speed_frac
        self._auto_shift(dt)
        h = dt / substeps
        for _ in range(substeps):
            self._integrate(target_steer, throttle, h, steer_rate, grip_mult, rolling_mult)

    # -- internals --------------------------------------------------------

    def _engine_rpm(self, vx: float, gear: int) -> float:
        total = self.p.gear_ratios[gear] * self.p.final_drive
        return abs(vx) / self.p.wheel_radius * total * _RPM_PER_RADS

    def _auto_shift(self, dt: float) -> None:
        """Pick a gear once per tick (road-speed rpm + wide hysteresis)."""
        p, s = self.p, self.s
        s.engine_rpm = self._engine_rpm(s.vx, s.gear)
        if s.shift_timer > 0.0:
            s.shift_timer = max(0.0, s.shift_timer - dt)
            return
        if s.engine_rpm > p.shift_up_rpm and s.gear < len(p.gear_ratios) - 1:
            s.gear += 1
            s.shift_timer = p.shift_time
        elif s.engine_rpm < p.shift_down_rpm and s.gear > 0:
            s.gear -= 1
            s.shift_timer = p.shift_time

    def _integrate(
        self, target_steer: float, throttle: float, h: float, steer_rate: float,
        grip_mult: float, rolling_mult: float
    ) -> None:
        p, s = self.p, self.s

        # Rate-limit the steering toward the target.
        max_d = steer_rate * h
        s.steer += float(np.clip(target_steer - s.steer, -max_d, max_d))
        delta = s.steer

        # Longitudinal force command: geared drive, or braking/reverse.
        if throttle >= 0.0:
            total = p.gear_ratios[s.gear] * p.final_drive
            rpm = min(max(self._engine_rpm(s.vx, s.gear), p.idle_rpm), p.redline_rpm)
            wheel_force = _engine_torque(rpm) * total / p.wheel_radius
            if s.shift_timer > 0.0:
                wheel_force = 0.0  # clutch/torque cut while the gear changes
            fx_drive = throttle * wheel_force
        elif s.vx > 0.5:
            fx_drive = throttle * p.max_brake_force
        else:
            fx_drive = throttle * p.max_engine_force * 0.5  # gentle reverse

        # Resistive longitudinal forces (profile drag + rolling).
        drag = p.drag_coef * s.vx * abs(s.vx)
        rr = (
            p.rolling_resistance * rolling_mult * math.copysign(1.0, s.vx)
            if abs(s.vx) > 0.05
            else 0.0
        )

        # Static axle loads + aero downforce (grows with speed^2).
        downforce = p.downforce_coef * s.vx * s.vx
        fz_f = p.mass * G * p.lr / p.wheelbase + downforce * (1.0 - p.aero_rear_bias)
        fz_r = p.mass * G * p.lf / p.wheelbase + downforce * p.aero_rear_bias

        # Longitudinal weight transfer (estimate accel from the commanded force).
        a_long = (fx_drive - drag - rr) / p.mass
        d_fz = p.mass * a_long * p.h_cg / p.wheelbase
        fz_f = max(fz_f - d_fz, 100.0)
        fz_r = max(fz_r + d_fz, 100.0)

        grip_f = p.mu * grip_mult * fz_f
        grip_r = p.mu * grip_mult * p.rear_grip_bias * fz_r

        # Slip angles. Guard forward speed and fade lateral grip out near standstill.
        vx_safe = max(abs(s.vx), 1.0)
        alpha_f_geo = math.atan2(s.vy + p.lf * s.r, vx_safe) - delta
        alpha_r_geo = math.atan2(s.vy - p.lr * s.r, vx_safe)
        # Relaxation length: the slip angle the tyre actually "feels" lags the
        # geometric one by a first-order filter over distance travelled (v·h/σ).
        relax = min(vx_safe * h / p.relax_len, 1.0)
        s.slip_f += (alpha_f_geo - s.slip_f) * relax
        s.slip_r += (alpha_r_geo - s.slip_r) * relax
        alpha_f = s.slip_f
        alpha_r = s.slip_r
        # Fade grip out only at a true standstill (to stop parked jitter). Key it
        # on actual planar speed, not forward speed — a car sliding fully sideways
        # has vx≈0 but is moving fast, and must still feel the tyres scrubbing it.
        grip_fade = float(np.clip((math.hypot(s.vx, s.vy) - 0.5) / 1.5, 0.0, 1.0))

        fy_f = _pacejka(alpha_f, grip_f) * grip_fade
        if fx_drive >= 0.0:
            # Driven rear from COMBINED slip: the wheel's slip ratio (spin) and
            # slip angle share one Pacejka grip budget. Flooring it spins the
            # wheel up, which bleeds lateral grip and steps the rear out — biggest
            # at low speed (little downforce) and self-limiting, since the spin
            # also caps the thrust. Lift off and the wheel re-hooks, so the slide
            # is catchable.
            fx_f = 0.0
            kappa = (s.wheel_v_r - s.vx) / max(abs(s.vx), p.slip_vx_floor)
            sy = math.tan(alpha_r)
            sigma = math.hypot(kappa, sy)
            if sigma < 1e-6:
                fx_r = fy_r = 0.0
            else:
                fmag = _pacejka_mag(sigma, grip_r)
                fx_r = fmag * (kappa / sigma)
                fy_r = -fmag * (sy / sigma) * grip_fade
            # Spin up the rear wheel: I·dω/dt = (engine - road) torque. Stiff at
            # this step size, so integrate the wheel semi-implicitly.
            a = h * p.wheel_radius * p.wheel_radius / p.wheel_inertia
            k = grip_r * PAC_B * PAC_C / max(abs(s.vx), p.slip_vx_floor)
            s.wheel_v_r += a * (fx_drive - fx_r) / (1.0 + a * k)
            if p.traction_control:
                # Speed-tapered slip ceiling: tight off the line, releasing toward
                # free-spin as speed builds, so slides stay alive in faster corners.
                t = (abs(s.vx) - p.tc_full_speed) / (p.tc_off_speed - p.tc_full_speed)
                t = min(max(t, 0.0), 1.0)
                slip_allow = p.tc_slip + (1.0 - p.tc_slip) * t
                ceiling = s.vx + slip_allow * max(abs(s.vx), p.slip_vx_floor)
                if s.wheel_v_r > ceiling:
                    s.wheel_v_r = ceiling
        else:
            fy_r = _pacejka(alpha_r, grip_r) * grip_fade
            # Braking: the FRONT shares its grip budget between brake and cornering
            # via a friction ellipse — straight-line braking gets the whole budget,
            # but braking *and* turning trims both proportionally (you can still
            # trail-brake and rotate, just with less of each). This curbs the front
            # over-biting and snapping the unloaded rear, without locking you
            # straight. The rear keeps lateral-first so it stays planted.
            bias = p.brake_bias
            fx_f_want = fx_drive * bias
            mag = math.hypot(fx_f_want, fy_f)
            if mag > grip_f:
                scl = grip_f / mag
                fx_f, fy_f = fx_f_want * scl, fy_f * scl
            else:
                fx_f = fx_f_want
            fx_r = _friction_clamp(fx_drive * (1.0 - bias), grip_r, fy_r)
            s.wheel_v_r = s.vx  # ABS-style: no stored spin while braking

        fx_long = fx_f + fx_r - drag - rr

        # Body-frame accelerations (bicycle model).
        ax = fx_long / p.mass + s.vy * s.r - (fy_f * math.sin(delta)) / p.mass
        ay = (fy_f * math.cos(delta) + fy_r) / p.mass - s.vx * s.r
        r_dot = (p.lf * fy_f * math.cos(delta) - p.lr * fy_r) / p.inertia_z

        s.vx += ax * h
        s.vy += ay * h
        s.r += r_dot * h

        if throttle < 0.0 and -0.5 < s.vx < 0.5:
            s.vx = 0.0  # don't let braking drag the car into spurious reverse

        # Integrate global pose.
        s.x += (s.vx * math.cos(s.yaw) - s.vy * math.sin(s.yaw)) * h
        s.y += (s.vx * math.sin(s.yaw) + s.vy * math.cos(s.yaw)) * h
        s.yaw = _wrap_angle(s.yaw + s.r * h)


def _pacejka(alpha: float, grip: float) -> float:
    """Lateral tyre force (N) opposing slip angle ``alpha``, peak magnitude ~grip."""
    return -grip * math.sin(PAC_C * math.atan(PAC_B * alpha - PAC_E * (PAC_B * alpha - math.atan(PAC_B * alpha))))


def _pacejka_mag(slip: float, grip: float) -> float:
    """Combined-slip tyre force magnitude (N) for a normalised slip, peak ~grip."""
    return grip * math.sin(PAC_C * math.atan(PAC_B * slip - PAC_E * (PAC_B * slip - math.atan(PAC_B * slip))))


def _friction_clamp(value: float, f_max: float, used: float) -> float:
    """Clamp ``value`` to the budget left on the friction circle after ``used``."""
    budget = f_max * f_max - used * used
    if budget <= 0.0:
        return 0.0
    allowed = math.sqrt(budget)
    return float(np.clip(value, -allowed, allowed))


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
