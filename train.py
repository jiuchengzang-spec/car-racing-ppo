"""Train an RL agent to lap as fast as it can — self-contained PyTorch PPO.

    pip install torch
    python train.py --timesteps 2_000_000
    python enjoy.py --model ppo_racing.pt        # watch it drive

No stable-baselines3 and no TensorBoard — just numpy + gymnasium + torch. The
agent is a small actor-critic MLP over the env's low-dimensional observation
(11 rangefinders + forward/lateral speed + yaw rate + heading error + steer).
Updates are textbook PPO: clipped surrogate, GAE(lambda), advantage
normalisation, a few epochs of minibatch SGD, optional KL early-stop.

**Curriculum learning** (on by default): training starts on easy *flowing*
tracks (fast, open sweepers — forgiving), then auto-advances to *balanced* and
finally tight *technical* circuits once the agent reliably finishes laps. Pass
``--no-curriculum`` to train on a single (optionally randomised) setting instead.

Progress prints to the console each update; pass ``--csv path.csv`` to also log a
row per update for plotting. Checkpoints land in ``./checkpoints`` and the final
policy in ``--out`` (default ``ppo_racing.pt``).
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from racing.env import RacingEnv
from racing.ppo import ActorCritic, save_policy

# Curriculum stages, easy -> hard. Each randomises the track across a small pool
# of seeds (derived from --track-seed) drawn from one "character" profile: flowing
# sweepers are the gentlest to keep on, technical hairpins the most punishing.
CURRICULUM = [
    {"name": "easy", "profile": "flowing", "pool": 4},
    {"name": "medium", "profile": "balanced", "pool": 8},
    {"name": "hard", "profile": "technical", "pool": 16},
]


class SyncVec:
    """A tiny synchronous vector env: step N RacingEnvs, auto-reset on done.

    Deliberately not gymnasium's VectorEnv — its autoreset semantics shift between
    versions, and a few lines here keep the rollout/bootstrap logic explicit and
    under our control. On a done step it returns the *reset* observation as the
    next obs (so the rollout never stalls) and hands back the terminal observation
    separately so PPO can bootstrap the value of a time-limit truncation.
    """

    def __init__(self, make_env, n: int, base_seed: int) -> None:
        self.envs = [make_env(i) for i in range(n)]
        self.n = n
        self.obs_dim = self.envs[0].observation_space.shape[0]
        self.act_dim = self.envs[0].action_space.shape[0]
        self._seed0 = base_seed

    def reset(self) -> np.ndarray:
        obs = np.empty((self.n, self.obs_dim), dtype=np.float32)
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=self._seed0 + i)  # distinct seeds -> varied pool draws
            obs[i] = o
        return obs

    def step(self, actions: np.ndarray):
        next_obs = np.empty((self.n, self.obs_dim), dtype=np.float32)
        rewards = np.empty(self.n, dtype=np.float32)
        terminated = np.zeros(self.n, dtype=bool)
        truncated = np.zeros(self.n, dtype=bool)
        term_obs: dict[int, np.ndarray] = {}
        infos: list[dict] = []
        for i, e in enumerate(self.envs):
            o, r, term, trunc, info = e.step(actions[i])
            if term or trunc:
                term_obs[i] = np.asarray(o, dtype=np.float32)  # final obs pre-reset
                o, _ = e.reset()
            next_obs[i] = o
            rewards[i] = r
            terminated[i] = term
            truncated[i] = trunc
            infos.append(info)
        return next_obs, rewards, terminated, truncated, term_obs, infos


def make_env_fn(profile: str, base_seed: int, randomize: bool, pool: int):
    def thunk(idx: int) -> RacingEnv:
        return RacingEnv(
            track_seed=base_seed,
            track_profile=profile,
            randomize_track=randomize,
            track_pool=pool,
        )

    return thunk


def build_vec(stage_cfg: dict, args, device_seed: int) -> SyncVec:
    fn = make_env_fn(stage_cfg["profile"], args.track_seed, stage_cfg["randomize"], stage_cfg["pool"])
    return SyncVec(fn, args.n_envs, base_seed=args.seed + device_seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=2048, help="rollout horizon per env")
    ap.add_argument("--epochs", type=int, default=10, help="PPO epochs per update")
    ap.add_argument("--minibatches", type=int, default=32)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--target-kl", type=float, default=0.03, help="early-stop epochs past this KL (<=0 disables)")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--track-seed", type=int, default=8)
    # Curriculum control.
    ap.add_argument("--no-curriculum", action="store_true", help="train on one setting instead of easy->hard stages")
    ap.add_argument("--randomize-track", action="store_true", help="(--no-curriculum) vary the track each episode")
    ap.add_argument("--track-pool", type=int, default=10, help="(--no-curriculum) pool size for --randomize-track (0 = unbounded)")
    ap.add_argument("--track-profile", default="balanced", help="(--no-curriculum) track character")
    ap.add_argument("--advance-lap-rate", type=float, default=0.5, help="advance a stage once this fraction of recent episodes finish a lap")
    ap.add_argument("--stage-min-steps", type=int, default=300_000, help="minimum env steps before a stage may advance")
    # IO.
    ap.add_argument("--out", default="ppo_racing.pt")
    ap.add_argument("--save-freq", type=int, default=200_000, help="checkpoint every N env steps")
    ap.add_argument("--csv", default="", help="also append a row per update to this CSV file")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # Weights & Biases experiment tracking (opt-in; wandb is only imported when used).
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-entity", default="jiucheng-zang-venuiti-solutions")
    ap.add_argument("--wandb-project", default="racing-car-ppo training")
    ap.add_argument("--wandb-name", default="", help="W&B run name (blank = auto-generated)")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Curriculum schedule (single fixed stage when --no-curriculum).
    if args.no_curriculum:
        stages = [{
            "name": args.track_profile,
            "profile": args.track_profile,
            "randomize": args.randomize_track,
            "pool": args.track_pool,
        }]
    else:
        stages = [dict(s, randomize=True) for s in CURRICULUM]

    stage_i = 0
    vec = build_vec(stages[stage_i], args, device_seed=0)
    obs_dim, act_dim = vec.obs_dim, vec.act_dim
    print(f"obs_dim={obs_dim} act_dim={act_dim} device={device} stages={[s['name'] for s in stages]}")

    model = ActorCritic(obs_dim, act_dim, args.hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    n_envs, n_steps = args.n_envs, args.n_steps
    batch = n_envs * n_steps
    minibatch = max(batch // args.minibatches, 1)

    # Rollout buffers (T, N, ...).
    obs_buf = torch.zeros((n_steps, n_envs, obs_dim), device=device)
    act_buf = torch.zeros((n_steps, n_envs, act_dim), device=device)
    logp_buf = torch.zeros((n_steps, n_envs), device=device)
    rew_buf = torch.zeros((n_steps, n_envs), device=device)
    val_buf = torch.zeros((n_steps, n_envs), device=device)
    done_buf = torch.zeros((n_steps, n_envs), device=device)  # episode-boundary flag at step t

    next_obs = torch.tensor(vec.reset(), device=device)
    next_done = torch.zeros(n_envs, device=device)

    # Episode bookkeeping (for logging + curriculum advancement).
    ep_return = np.zeros(n_envs, dtype=np.float64)
    ep_len = np.zeros(n_envs, dtype=np.int64)
    ret_hist: deque[float] = deque(maxlen=100)
    len_hist: deque[float] = deque(maxlen=100)
    lap_hist: deque[int] = deque(maxlen=100)  # 1 if the episode finished a lap

    csv_writer = None
    csv_file = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["step", "stage", "ep_return", "ep_len", "lap_rate",
                             "pg_loss", "v_loss", "entropy", "approx_kl", "fps"])

    use_wandb = args.wandb
    if use_wandb:
        import wandb  # lazy: only a dependency when --wandb is passed

        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name or None,
            config={**vars(args), "stages": [s["name"] for s in stages],
                    "batch": batch, "minibatch": minibatch},
        )

    os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    stage_step0 = 0
    next_save = args.save_freq
    best_return = -1e18
    start = time.time()
    update = 0

    while global_step < args.timesteps:
        update += 1
        # --- collect a rollout ------------------------------------------------
        for t in range(n_steps):
            obs_buf[t] = next_obs
            done_buf[t] = next_done
            with torch.no_grad():
                action, logp, _, value = model.get_action_and_value(next_obs)
            val_buf[t] = value
            act_buf[t] = action
            logp_buf[t] = logp

            clipped = action.clamp(-1.0, 1.0).cpu().numpy()
            nobs, reward, term, trunc, term_obs, infos = vec.step(clipped)
            global_step += n_envs

            reward = reward.astype(np.float32)
            # Bootstrap the value of a *truncated* (time-limit) episode so the
            # cut-off doesn't look like a real terminal state. True terminals
            # (crash / lap done) get no bootstrap — the episode genuinely ended.
            trunc_only = [i for i in range(n_envs) if trunc[i] and not term[i]]
            if trunc_only:
                tobs = torch.tensor(np.stack([term_obs[i] for i in trunc_only]), device=device)
                with torch.no_grad():
                    tv = model.get_value(tobs).cpu().numpy()
                for k, i in enumerate(trunc_only):
                    reward[i] += args.gamma * float(tv[k])

            rew_buf[t] = torch.tensor(reward, device=device)
            next_obs = torch.tensor(nobs, device=device)
            done = np.logical_or(term, trunc)
            next_done = torch.tensor(done.astype(np.float32), device=device)

            ep_return += reward
            ep_len += 1
            for i in range(n_envs):
                if done[i]:
                    ret_hist.append(float(ep_return[i]))
                    len_hist.append(float(ep_len[i]))
                    lap_hist.append(1 if infos[i].get("lap_done") else 0)
                    ep_return[i] = 0.0
                    ep_len[i] = 0

        # --- GAE(lambda) advantages + returns ---------------------------------
        with torch.no_grad():
            next_value = model.get_value(next_obs)
        advantages = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(n_envs, device=device)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_nonterminal = 1.0 - next_done
                next_val = next_value
            else:
                next_nonterminal = 1.0 - done_buf[t + 1]
                next_val = val_buf[t + 1]
            delta = rew_buf[t] + args.gamma * next_val * next_nonterminal - val_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        # --- PPO update -------------------------------------------------------
        b_obs = obs_buf.reshape(-1, obs_dim)
        b_act = act_buf.reshape(-1, act_dim)
        b_logp = logp_buf.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)

        idx = np.arange(batch)
        pg_loss = v_loss = entropy = approx_kl = torch.tensor(0.0)
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            stop = False
            for start_i in range(0, batch, minibatch):
                mb = idx[start_i:start_i + minibatch]
                mb_t = torch.as_tensor(mb, device=device)
                _, new_logp, ent, new_val = model.get_action_and_value(b_obs[mb_t], b_act[mb_t])
                log_ratio = new_logp - b_logp[mb_t]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()

                adv = b_adv[mb_t]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # per-minibatch norm
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((new_val - b_ret[mb_t]) ** 2).mean()
                entropy = ent.mean()
                loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl > 0 and approx_kl.item() > args.target_kl:
                stop = True
            if stop:
                break

        # --- logging ----------------------------------------------------------
        mean_ret = float(np.mean(ret_hist)) if ret_hist else float("nan")
        mean_len = float(np.mean(len_hist)) if len_hist else float("nan")
        lap_rate = float(np.mean(lap_hist)) if lap_hist else 0.0
        fps = int(global_step / max(time.time() - start, 1e-9))
        # Until the first episode ends, the history is empty — show a placeholder
        # rather than "nan" (a near-idle fresh policy just runs to the time limit).
        ret_s = f"{mean_ret:8.1f}" if ret_hist else "      --"
        len_s = f"{mean_len:6.0f}" if len_hist else "    --"
        print(
            f"upd {update:4d} | step {global_step:>9,} | {stages[stage_i]['name']:<7} "
            f"| ret {ret_s} | len {len_s} | lap {lap_rate:4.0%} "
            f"| pg {pg_loss.item():+.3f} | v {v_loss.item():.3f} | ent {entropy.item():.3f} "
            f"| kl {approx_kl.item():.3f} | {fps} fps"
        )
        if csv_writer:
            csv_writer.writerow([global_step, stages[stage_i]["name"], f"{mean_ret:.3f}",
                                 f"{mean_len:.1f}", f"{lap_rate:.4f}", f"{pg_loss.item():.4f}",
                                 f"{v_loss.item():.4f}", f"{entropy.item():.4f}",
                                 f"{approx_kl.item():.4f}", fps])
            csv_file.flush()
        if use_wandb:
            log_dict = {
                "losses/pg_loss": pg_loss.item(),
                "losses/v_loss": v_loss.item(),
                "losses/entropy": entropy.item(),
                "losses/approx_kl": approx_kl.item(),
                "charts/fps": fps,
                "curriculum/stage_idx": stage_i,
            }
            if ret_hist:  # skip episode stats until the first episode has finished
                log_dict["charts/ep_return"] = mean_ret
                log_dict["charts/ep_len"] = mean_len
                log_dict["charts/lap_rate"] = lap_rate
            wandb.log(log_dict, step=global_step)

        # --- checkpoints ------------------------------------------------------
        meta = {"step": global_step, "stage": stages[stage_i]["name"], "mean_return": mean_ret}
        if global_step >= next_save:
            path = os.path.join("checkpoints", f"ppo_racing_{global_step}.pt")
            save_policy(path, model, meta)
            next_save += args.save_freq
        if ret_hist and mean_ret > best_return:
            best_return = mean_ret
            save_policy(os.path.join("checkpoints", "ppo_racing_best.pt"), model, meta)

        # --- curriculum advancement ------------------------------------------
        if (stage_i < len(stages) - 1
                and (global_step - stage_step0) >= args.stage_min_steps
                and len(lap_hist) >= 20 and lap_rate >= args.advance_lap_rate):
            stage_i += 1
            stage_step0 = global_step
            print(f"  >> curriculum: advancing to stage '{stages[stage_i]['name']}' "
                  f"(lap rate {lap_rate:.0%} on '{stages[stage_i - 1]['name']}')")
            vec = build_vec(stages[stage_i], args, device_seed=stage_i)
            next_obs = torch.tensor(vec.reset(), device=device)
            next_done = torch.zeros(n_envs, device=device)
            ep_return[:] = 0.0
            ep_len[:] = 0
            ret_hist.clear(); len_hist.clear(); lap_hist.clear()

    save_policy(args.out, model, {"step": global_step, "final": True})
    if csv_file:
        csv_file.close()
    if use_wandb:
        wandb.finish()
    print(f"\nSaved final policy to {args.out} — watch it with:  python enjoy.py --model {args.out}")


if __name__ == "__main__":
    main()
