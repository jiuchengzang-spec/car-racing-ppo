"""Drive the car yourself.

    python play.py [--track-seed N] [--mouse]

Controls: arrow keys or WASD to steer; W/Up throttle, S/Down brake (reverses once
stopped). With ``--mouse``, left/right mouse position steers and the mouse buttons
(or W/S) drive and brake. R restages (keeping your records), B toggles the sensor
beams, V swaps the hood cam <-> top-down view, Esc / window-close quits.

You keep driving through off-track moments and finish lines — the HUD tracks your
current, last and best lap, and going off the track voids the lap in progress. You
drive the exact same env the RL agent trains on, so a fast human lap and a fast
agent lap are directly comparable.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import pygame

from racing.car import HANDLING_PRESETS, handling_preset
from racing.env import RacingEnv
from racing.render import SCREEN_W, STEER_ZONE_DEAD
from racing.track import TRACK_PROFILES


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track-seed", type=int, default=8)
    ap.add_argument("--track", choices=sorted(TRACK_PROFILES), default="balanced",
                    help="track character: balanced, power (long straights, few big corners), "
                         "flowing (fast sweepers), or technical (tight, twisty)")
    ap.add_argument("--mouse", action="store_true", help="steer with horizontal mouse position (anywhere on screen)")
    ap.add_argument("--top-down", action="store_true", help="overhead view instead of the 3D hood cam")
    ap.add_argument("--handling", choices=sorted(HANDLING_PRESETS), default="nimble",
                    help="handling preset: nimble (neutral F1-like, default), boat (soft legacy), "
                         "un/ov (fair understeer/oversteer pair — same car, setup-only)")
    args = ap.parse_args()

    # Human mode keeps rolling: don't end the episode on a lap or an off-track.
    env = RacingEnv(
        render_mode="human",
        track_seed=args.track_seed,
        track_profile=args.track,
        max_steps=10**9,
        terminate_off_track=False,
        terminate_on_lap=False,
        view="top" if args.top_down else "hood",
        car_params=handling_preset(args.handling),
    )
    env.reset()
    env.render()  # bring up the pygame window before we poll input

    steer = 0.0
    throttle_cmd = 0.0  # analog: ramps up while held, eases off when released
    brake_cmd = 0.0
    last_count = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                # R re-stages your position only — best lap, best sectors and your
                # lap history are kept, and the clock waits at the line again.
                env.restage()
                steer = 0.0
                throttle_cmd = 0.0
                brake_cmd = 0.0
                last_count = env._info()["lap_count"]
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_b:
                # Toggle the rangefinder-beam overlay (the agent's "vision").
                r = env._renderer
                if r is not None and hasattr(r, "show_beams"):
                    r.show_beams = not r.show_beams
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_v:
                env.toggle_view()  # swap hood cam <-> top-down overhead

        keys = pygame.key.get_pressed()
        up = keys[pygame.K_UP] or keys[pygame.K_w]
        down = keys[pygame.K_DOWN] or keys[pygame.K_s]

        if args.mouse:
            # The whole window is the wheel: the cursor's horizontal offset from
            # centre sets the lock (with a small straight-ahead deadzone). The band
            # drawn near the bottom is just a visual readout of the wheel — steering
            # is enforced anywhere on screen, so you never have to hunt for a zone.
            mx, _ = pygame.mouse.get_pos()
            cx = SCREEN_W / 2
            dx = cx - mx  # left of centre -> positive -> turn left
            mag = max(abs(dx) - STEER_ZONE_DEAD, 0.0)
            steer = float(np.clip(math.copysign(mag, dx) / (SCREEN_W / 2 - STEER_ZONE_DEAD), -1.0, 1.0))
            env.steer_overlay = steer
            buttons = pygame.mouse.get_pressed(3)
            up = up or buttons[0]
            down = down or buttons[2]
        else:
            left = keys[pygame.K_LEFT] or keys[pygame.K_a]
            right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            # Smooth toward the held direction; self-centre when released.
            # Left = positive steer (counter-clockwise / left turn in world frame).
            target = (1.0 if left else 0.0) + (-1.0 if right else 0.0)
            steer += np.clip(target - steer, -0.15, 0.15)
            if target == 0.0:
                steer *= 0.8

        # Analog throttle/brake: ramp toward full while the key is held and ease
        # back when released, instead of an on/off pedal. A quick tap of the brake
        # is now a light dab — so braking at speed no longer instantly spins you;
        # you can squeeze and trail off the brake like a real pedal.
        throttle_cmd = float(np.clip(throttle_cmd + (0.035 if up else -0.08), 0.0, 1.0))
        brake_cmd = float(np.clip(brake_cmd + (0.030 if down else -0.10), 0.0, 1.0))
        throttle = throttle_cmd - brake_cmd

        _, _, _, _, info = env.step(np.array([steer, throttle], dtype=np.float32))
        if info.get("lap_count", 0) > last_count:
            last_count = info["lap_count"]
            tag = "" if info.get("last_lap_valid", True) else "  (INVALID — went off track)"
            print(f"Lap {last_count}: {info['last_lap_time']:.3f}s{tag}")

    env.close()


if __name__ == "__main__":
    main()
