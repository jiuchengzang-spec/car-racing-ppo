"""A realistic-ish car racing environment you can drive or train an RL agent on."""
from __future__ import annotations

from gymnasium.envs.registration import register

from .car import HANDLING_PRESETS, Car, CarParams, CarState, handling_preset
from .env import RacingEnv
from .track import TRACK_PROFILES, Track, make_track

register(id="Racing-v0", entry_point="racing.env:RacingEnv", max_episode_steps=4000)

__all__ = [
    "Car", "CarParams", "CarState", "RacingEnv", "Track", "make_track",
    "HANDLING_PRESETS", "handling_preset", "TRACK_PROFILES",
]
