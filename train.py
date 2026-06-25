"""Train an RL agent to lap as fast as it can, using PPO (stable-baselines3).

    pip install "stable-baselines3>=2.3"
    python train.py --timesteps 1_000_000

The agent sees the rangefinder + speed observation and outputs continuous
steering/throttle. Reward is forward progress along the track minus a time cost,
with a crash penalty for leaving the tarmac and a bonus for completing the lap —
so the optimum is "as fast as possible within bounds".

Progress is logged to TensorBoard (``--tb-log``); watch it live with::

    tensorboard --logdir ./tb_logs

The model is snapshotted every ``--save-freq`` steps into ``./checkpoints`` and
the best-by-evaluation policy is kept as ``checkpoints/best_model.zip``. While
training runs you can watch any snapshot in another terminal::

    python enjoy.py --model checkpoints/ppo_racing_200000_steps
    python enjoy.py --model checkpoints/best_model
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--track-seed", type=int, default=8)
    ap.add_argument("--randomize-track", action="store_true",
                    help="vary the track each episode for a more general driver")
    ap.add_argument("--track-pool", type=int, default=10,
                    help="with --randomize-track, cycle a fixed pool of this many "
                         "seeds instead of an unbounded new track every episode "
                         "(0 = unbounded)")
    ap.add_argument("--save-freq", type=int, default=200_000,
                    help="save a checkpoint every this many env steps (total across envs)")
    ap.add_argument("--tb-log", default="./tb_logs", help="TensorBoard log dir")
    ap.add_argument("--out", default="ppo_racing")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.env_util import make_vec_env

    from racing.env import RacingEnv

    def make() -> RacingEnv:
        return RacingEnv(track_seed=args.track_seed, randomize_track=args.randomize_track,
                         track_pool=args.track_pool)

    venv = make_vec_env(make, n_envs=args.n_envs)
    eval_env = make_vec_env(make, n_envs=1)

    # Callback counters tick once per rollout step (which advances all n_envs at
    # once), so divide the requested *total* step frequency by n_envs.
    freq = max(args.save_freq // args.n_envs, 1)
    checkpoint_cb = CheckpointCallback(
        save_freq=freq, save_path="./checkpoints", name_prefix=args.out
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./checkpoints",
        log_path="./checkpoints",
        eval_freq=freq,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=2048,
        batch_size=2048,
        gae_lambda=0.95,
        gamma=0.995,
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log=args.tb_log,
        verbose=1,
    )
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=True,
        callback=CallbackList([checkpoint_cb, eval_cb]),
    )
    model.save(args.out)
    print(f"Saved model to {args.out}.zip — watch it with:  python enjoy.py --model {args.out}")


if __name__ == "__main__":
    main()
