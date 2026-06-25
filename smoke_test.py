"""Headless sanity check — no display, no pygame needed.

    python smoke_test.py

Verifies the env steps cleanly, observations stay finite and in-bounds, the
physics actually move the car under throttle, and the track geometry / sensors
behave. Good enough to catch a broken model before you open a window or train.
"""
from __future__ import annotations

import math

import numpy as np

from racing.env import RacingEnv
from racing.track import make_track


def check_track() -> None:
    t = make_track(seed=1)
    assert t.length > 100.0, "track suspiciously short"
    x, y, heading = t.start_pose
    assert t.is_inside(x, y), "start pose should be on the track"
    # From the centre, a ray perpendicular to the track reaches a wall at ~half
    # the width on each side.
    left = t.cast_ray(x, y, heading + math.pi / 2, 100.0)
    right = t.cast_ray(x, y, heading - math.pi / 2, 100.0)
    assert left + right <= t.width + 1.0, (
        f"cross-track ray span {left + right:.2f} > width {t.width}"
    )
    print(f"track ok: length={t.length:.1f}m width={t.width}m")


def check_env_runs() -> None:
    env = RacingEnv(track_seed=1)
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert np.all(np.isfinite(obs))

    # Pin the throttle, steer straight: the car must gain forward speed.
    speeds = []
    for _ in range(180):  # ~3s
        obs, reward, terminated, truncated, info = env.step(np.array([0.0, 1.0], dtype=np.float32))
        assert np.all(np.isfinite(obs)), "non-finite observation"
        assert env.observation_space.contains(obs), "observation out of declared bounds"
        assert math.isfinite(reward)
        speeds.append(info["speed"])
        if terminated or truncated:
            break
    assert max(speeds) > 5.0, f"car never accelerated (max {max(speeds):.2f} m/s)"
    print(f"env ok: reached {max(speeds) * 3.6:.1f} km/h under full throttle")

    # A hard lock at speed should bend the path (yaw rate becomes non-trivial).
    env.reset(seed=0)
    for _ in range(120):
        env.step(np.array([0.0, 1.0], dtype=np.float32))
    for _ in range(60):
        env.step(np.array([1.0, 0.6], dtype=np.float32))
    assert abs(env.car.s.r) > 0.05, "steering produced no yaw rate"
    print(f"steering ok: yaw rate {env.car.s.r:.3f} rad/s under lock")


def check_random_rollout() -> None:
    env = RacingEnv(track_seed=2, randomize_track=True)
    for ep in range(3):
        env.reset(seed=ep)
        for _ in range(200):
            a = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(a)
            assert np.all(np.isfinite(obs))
            if term or trunc:
                break
    print("random rollouts ok: 3 episodes, finite throughout")


if __name__ == "__main__":
    check_track()
    check_env_runs()
    check_random_rollout()
    print("\nALL CHECKS PASSED")
