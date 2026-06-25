"""Watch a trained agent drive.

    python enjoy.py --model ppo_racing.pt [--track-seed N] [--track PROFILE]

Loads a policy saved by train.py (pure-PyTorch PPO) and drives the same env you
play by hand, so a fast agent lap and a fast human lap are directly comparable.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_racing.pt")
    ap.add_argument("--track-seed", type=int, default=8)
    ap.add_argument("--track", default="balanced", help="track character: balanced | power | flowing | technical")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    args = ap.parse_args()

    from racing.env import RacingEnv
    from racing.ppo import load_policy

    env = RacingEnv(render_mode="human", track_seed=args.track_seed, track_profile=args.track)
    model, meta = load_policy(args.model)
    if meta:
        print(f"loaded {args.model}  (trained to step {meta.get('step', '?')})")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        info: dict = {}
        while not done:
            action = model.act(obs, deterministic=not args.stochastic)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        tag = "lap!" if info.get("lap_done") else ("crash" if info.get("off_track") else "timeout")
        print(f"episode {ep}: {tag}  time={info.get('time', 0):.2f}s  progress={info.get('lap_fraction', 0) * 100:.1f}%")

    env.close()


if __name__ == "__main__":
    main()
