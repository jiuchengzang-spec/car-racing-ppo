"""Watch a trained agent drive.

    python enjoy.py --model ppo_racing [--track-seed N]
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_racing")
    ap.add_argument("--track-seed", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    from racing.env import RacingEnv

    env = RacingEnv(render_mode="human", track_seed=args.track_seed)
    model = PPO.load(args.model)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        info: dict = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        tag = "lap!" if info.get("lap_done") else ("crash" if info.get("off_track") else "timeout")
        print(f"episode {ep}: {tag}  time={info.get('time', 0):.2f}s  progress={info.get('lap_fraction', 0) * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
