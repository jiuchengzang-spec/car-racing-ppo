"""Watch a trained agent drive.

    python enjoy.py --model ppo_racing.pt [--track-seed N] [--track PROFILE]
    python enjoy.py --model ppo_single_best.pt --viz   # + a model-activation window

Loads a policy saved by train.py (pure-PyTorch PPO) and drives the same env you
play by hand, so a fast agent lap and a fast human lap are directly comparable.
With ``--viz`` a second window shows the network's live activations as it drives.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ppo_racing.pt")
    ap.add_argument("--track-seed", type=int, default=8)
    ap.add_argument("--track", default="balanced", help="track character: balanced | power | flowing | technical")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=8000,
                    help="episode step limit before a 'timeout' truncation (env runs at 60 Hz, so "
                         "8000 ≈ 133 s — enough for a full lap; the 4000 env default truncates long laps)")
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of using the mean")
    ap.add_argument("--viz", action="store_true", help="open a separate window visualising the model's activations")
    ap.add_argument("--smooth-steer", type=float, default=0.6, metavar="A",
                    help="EMA-smooth the steering output to kill jitter from noisy long-range beams "
                         "(0 = raw policy, ~0.6 = stable, higher = smoother but laggier)")
    args = ap.parse_args()

    import pygame

    from racing.env import RacingEnv
    from racing.ppo import load_policy

    env = RacingEnv(render_mode="human", track_seed=args.track_seed, track_profile=args.track,
                    max_steps=args.max_steps)
    model, meta = load_policy(args.model)
    if meta:
        print(f"loaded {args.model}  (trained to step {meta.get('step', '?')})")
    print("controls:  V swap view (hood / top-down)   ·   Esc / window-close quit")

    env.reset()
    env.render()  # bring up the driving window first (before the optional viz window)
    viz = None
    if args.viz:
        try:
            from racing.actviz import ActivationViz

            viz = ActivationViz()
        except Exception as e:
            print(f"activation viz unavailable ({e}); continuing without it")
            viz = None

    running = True
    for ep in range(args.episodes):
        if not running:
            break
        obs, _ = env.reset()
        prev_steer = None  # for steering EMA (resets each episode)
        done = False
        info: dict = {}
        while not done and running:
            # Always poll events so V (swap view) and quitting work with or without --viz.
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    running = False
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_v:
                    env.toggle_view()
            if viz is not None:
                if viz.closed(events):
                    running = False
                else:
                    viz.update(model.activations(obs))
            if not running:
                break
            action = model.act(obs, deterministic=not args.stochastic)
            if args.smooth_steer > 0.0:
                # Low-pass the steering command so noisy long-range beam readings
                # don't translate into a visible left-right saw on the wheel.
                if prev_steer is None:
                    prev_steer = float(action[0])
                action = action.copy()
                action[0] = args.smooth_steer * prev_steer + (1.0 - args.smooth_steer) * float(action[0])
                prev_steer = float(action[0])
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        tag = "lap!" if info.get("lap_done") else ("crash" if info.get("off_track") else "timeout")
        print(f"episode {ep}: {tag}  time={info.get('time', 0):.2f}s  progress={info.get('lap_fraction', 0) * 100:.1f}%")

    if viz is not None:
        viz.close()
    env.close()


if __name__ == "__main__":
    main()
