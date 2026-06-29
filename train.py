"""Train an RL agent to lap as fast as it can — self-contained PyTorch PPO.

    pip install torch
    python train.py --timesteps 2_000_000
    python enjoy.py --model ppo_racing.pt        # watch it drive

No stable-baselines3 and no TensorBoard — just numpy + gymnasium + torch. The
agent is a small actor-critic MLP over the env's low-dimensional observation
(rangefinders + speeds + heading + steer + curvature preview + tyre slip).
Updates are textbook PPO: clipped surrogate, GAE(lambda), advantage
normalisation, a few epochs of minibatch SGD, optional KL early-stop, AdamW with
linear learning-rate annealing.

Racing-specific defaults: episodes start at racing speed (`--spawn-speed-kmh`,
60-150 km/h rolling starts) so the policy must manage high momentum from tick one
rather than creeping along to dodge crash penalties.

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
    # Unbounded pools (a fresh track every episode) so the policy must GENERALISE,
    # not memorise a fixed set — the only way to score is to read each new corner and
    # set speed for it. Eval is on separate held-out seeds (see eval_envs).
    {"name": "easy", "profile": "flowing", "pool": 0},
    {"name": "medium", "profile": "balanced", "pool": 0},
    {"name": "hard", "profile": "technical", "pool": 0},
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


def make_env_fn(profile: str, base_seed: int, randomize: bool, pool: int,
                spawn_speed: tuple[float, float] | None, reward_kwargs: dict):
    def thunk(idx: int) -> RacingEnv:
        return RacingEnv(
            track_seed=base_seed,
            track_profile=profile,
            randomize_track=randomize,
            track_pool=pool,
            spawn_speed=spawn_speed,
            **reward_kwargs,
        )

    return thunk


def build_vec(stage_cfg: dict, args, device_seed: int) -> SyncVec:
    reward_kwargs = dict(lap_bonus=args.lap_bonus, sector_bonus=args.sector_bonus,
                         speed_w=args.speed_w, crash_penalty=args.crash_penalty,
                         max_steps=args.max_steps, beam_smooth=args.beam_smooth,
                         corner_brake_w=args.corner_brake_w)
    fn = make_env_fn(stage_cfg["profile"], args.track_seed, stage_cfg["randomize"],
                     stage_cfg["pool"], args.spawn_speed, reward_kwargs)
    return SyncVec(fn, args.n_envs, base_seed=args.seed + device_seed)


@torch.no_grad()
def evaluate_deterministic(model: ActorCritic, env, device: str, n_episodes: int) -> float:
    """Lap-completion rate of the *deterministic* (mean-action) policy.

    THIS is the metric that matters for deployment: the stochastic rollout
    lap_rate can be wildly optimistic when the policy leans on action noise — a
    policy can score 95% stochastically yet 0% with the mean. Cheap eval on a
    single env with fixed seeds so the number is comparable across updates.
    """
    laps = 0
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done = False
        info: dict = {}
        while not done:
            t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            action = model.actor_mean(t).squeeze(0).clamp(-1.0, 1.0).cpu().numpy()
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        laps += int(bool(info.get("lap_done")))
    return laps / max(n_episodes, 1)


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
    ap.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay")
    ap.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True,
                    help="linearly decay the learning rate to 0 over training (standard PPO; --no-anneal-lr to disable)")
    ap.add_argument("--anneal-ent", action=argparse.BooleanOptionalAction, default=True,
                    help="linearly decay the entropy coef to 0 — keeps a converged policy from being diffused by a constant entropy bonus")
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
    ap.add_argument("--spawn-speed-kmh", type=float, nargs=2, default=[60.0, 150.0],
                    metavar=("LO", "HI"),
                    help="rolling-start speed range (km/h) for RL episodes; '0 0' starts from rest")
    # Reward shaping (env weights). Defaults match RacingEnv; raise --lap-bonus /
    # --sector-bonus and drop --speed-w to push the agent to *close* laps.
    ap.add_argument("--lap-bonus", type=float, default=100.0, help="reward for completing a lap")
    ap.add_argument("--sector-bonus", type=float, default=0.0, help="reward per new track third reached (denser lap-closing signal)")
    ap.add_argument("--speed-w", type=float, default=0.003, help="reward weight on forward speed (0 = rely on progress only)")
    ap.add_argument("--crash-penalty", type=float, default=10.0, help="penalty for a full track exit")
    ap.add_argument("--max-steps", type=int, default=4000, help="episode truncation limit (raise so a clean lap doesn't time out before finishing)")
    ap.add_argument("--beam-smooth", type=float, default=0.4, help="EMA smoothing on the rangefinder beams (0=raw, higher=steadier obs)")
    ap.add_argument("--corner-brake-w", type=float, default=0.0, help="penalty weight for over-speeding into the upcoming corner (teaches braking for deep-angle curves; 0=off)")
    ap.add_argument("--advance-lap-rate", type=float, default=0.5, help="advance a stage once this fraction of recent episodes finish a lap")
    ap.add_argument("--stage-min-steps", type=int, default=300_000, help="minimum env steps before a stage may advance")
    # IO.
    ap.add_argument("--out", default="ppo_racing.pt")
    ap.add_argument("--init-from", default="", help="warm-start the policy from this .pt checkpoint (must match --hidden / obs dims)")
    ap.add_argument("--save-freq", type=int, default=200_000, help="checkpoint every N env steps")
    ap.add_argument("--eval-freq", type=int, default=200_000, help="deterministic eval every N env steps")
    ap.add_argument("--eval-episodes", type=int, default=5, help="deterministic eval episodes (0 = disable; this is the real deployable metric)")
    ap.add_argument("--csv", default="", help="also append a row per update to this CSV file")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    # Weights & Biases experiment tracking (opt-in; wandb is only imported when used).
    ap.add_argument("--wandb", action="store_true", help="log metrics to Weights & Biases")
    ap.add_argument("--wandb-entity", default="jiucheng-zang-venuiti-solutions")
    ap.add_argument("--wandb-project", default="racing-car-ppo training")
    ap.add_argument("--wandb-name", default="", help="W&B run name (blank = auto-generated)")
    args = ap.parse_args()

    # Rolling-start speed range (km/h -> m/s); "0 0" means spawn from rest.
    lo_kmh, hi_kmh = args.spawn_speed_kmh
    args.spawn_speed = None if (lo_kmh <= 0 and hi_kmh <= 0) else (lo_kmh / 3.6, hi_kmh / 3.6)

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
    if args.init_from:
        # Warm-start: load policy+value weights from a prior checkpoint to continue
        # training (e.g. focus an already-decent agent on a harder track) instead of
        # starting from scratch. Optimizer state isn't carried (a clean LR schedule).
        from racing.ppo import load_policy
        src, _ = load_policy(args.init_from, device=device)
        if (src.obs_dim, src.act_dim, src.hidden) != (obs_dim, act_dim, args.hidden):
            raise SystemExit(
                f"--init-from net dims {(src.obs_dim, src.act_dim, src.hidden)} "
                f"!= current {(obs_dim, act_dim, args.hidden)}"
            )
        model.load_state_dict(src.state_dict())
        print(f"warm-started policy from {args.init_from}")
    # AdamW (decoupled weight decay) — a better default than Adam for this policy.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay)

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
        csv_writer.writerow(["step", "stage", "ep_return", "ep_len", "lap_rate", "det_lap",
                             "pg_loss", "v_loss", "entropy", "approx_kl", "lr", "fps"])

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
    ckpt_prefix = os.path.splitext(os.path.basename(args.out))[0]  # so runs don't clobber
    # Deterministic-eval envs (the real deployable metric). For a curriculum run the
    # GOAL is generalization, so eval on HELD-OUT tracks (randomize + unbounded pool,
    # fixed eval seeds 10000+) across ALL profiles — not just the easy stage-0 — so
    # det_lap reflects "all tracks". A --no-curriculum run evals its single configured
    # setting, as before. Reward kwargs are irrelevant to eval (it only checks laps).
    eval_reward = dict(lap_bonus=args.lap_bonus, sector_bonus=args.sector_bonus,
                       speed_w=args.speed_w, crash_penalty=args.crash_penalty,
                       max_steps=args.max_steps, beam_smooth=args.beam_smooth,
                       corner_brake_w=args.corner_brake_w)
    eval_envs = {}
    if args.eval_episodes > 0:
        if args.no_curriculum:
            eval_envs[stages[0]["profile"]] = make_env_fn(
                stages[0]["profile"], args.track_seed, stages[0]["randomize"],
                stages[0]["pool"], args.spawn_speed, eval_reward)(0)
        else:
            for _prof in ("flowing", "balanced", "technical"):
                eval_envs[_prof] = make_env_fn(_prof, args.track_seed, True, 0,
                                               args.spawn_speed, eval_reward)(0)
    global_step = 0
    stage_step0 = 0
    next_save = args.save_freq
    next_eval = args.eval_freq
    best_return = -1e18
    best_det = -1.0  # best deterministic lap rate seen (the deployable best)
    last_det = float("nan")  # latest deterministic eval value (carried into the CSV)
    start = time.time()
    update = 0

    while global_step < args.timesteps:
        update += 1
        # Linear LR anneal to 0 over training — a standard PPO stabiliser that
        # tightens the policy as it converges (fraction of budget remaining * lr).
        if args.anneal_lr:
            cur_lr = max(0.0, 1.0 - global_step / args.timesteps) * args.lr
            for g in optimizer.param_groups:
                g["lr"] = cur_lr
        else:
            cur_lr = args.lr
        # Anneal the entropy bonus to 0 too: it's vital for exploration early, but a
        # constant bonus inflates (diffuses) the policy once it has converged — which
        # collapsed two earlier runs. Decaying it lets the policy settle and hold.
        if args.anneal_ent:
            cur_ent = max(0.0, 1.0 - global_step / args.timesteps) * args.ent_coef
        else:
            cur_ent = args.ent_coef
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
                loss = pg_loss - cur_ent * entropy + args.vf_coef * v_loss

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
            f"| kl {approx_kl.item():.3f} | lr {cur_lr:.1e} | {fps} fps"
        )
        if csv_writer:
            csv_writer.writerow([global_step, stages[stage_i]["name"], f"{mean_ret:.3f}",
                                 f"{mean_len:.1f}", f"{lap_rate:.4f}", f"{last_det:.4f}", f"{pg_loss.item():.4f}",
                                 f"{v_loss.item():.4f}", f"{entropy.item():.4f}",
                                 f"{approx_kl.item():.4f}", f"{cur_lr:.6f}", fps])
            csv_file.flush()
        if use_wandb:
            log_dict = {
                "losses/pg_loss": pg_loss.item(),
                "losses/v_loss": v_loss.item(),
                "losses/entropy": entropy.item(),
                "losses/approx_kl": approx_kl.item(),
                "charts/fps": fps,
                "charts/lr": cur_lr,
                "charts/ent_coef": cur_ent,
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
            path = os.path.join("checkpoints", f"{ckpt_prefix}_{global_step}.pt")
            save_policy(path, model, meta)
            next_save += args.save_freq
        if ret_hist and mean_ret > best_return:
            best_return = mean_ret
            save_policy(os.path.join("checkpoints", f"{ckpt_prefix}_best.pt"), model, meta)

        # --- deterministic eval (the real, deployable metric) -----------------
        if eval_envs and global_step >= next_eval:
            next_eval += args.eval_freq
            per_profile = {p: evaluate_deterministic(model, e, device, args.eval_episodes)
                           for p, e in eval_envs.items()}
            det_lap = sum(per_profile.values()) / len(per_profile)  # mean over profiles = "all tracks"
            last_det = det_lap
            std = float(torch.exp(model.log_std.detach()).mean())
            brk = "  ".join(f"{p[:4]} {v:.0%}" for p, v in per_profile.items())
            print(f"  [eval] det lap {det_lap:5.0%}  ({brk})  (action std {std:.2f})")
            if use_wandb:
                wd = {"charts/det_lap_rate": det_lap, "charts/action_std": std}
                wd.update({f"charts/det_lap_{p}": v for p, v in per_profile.items()})
                wandb.log(wd, step=global_step)
            if det_lap > best_det:  # keep the best DEPLOYABLE (most general) policy separately
                best_det = det_lap
                save_policy(os.path.join("checkpoints", f"{ckpt_prefix}_detbest.pt"), model,
                            {"step": global_step, "det_lap_rate": det_lap, "per_profile": per_profile})

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
