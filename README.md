# car-racing-rl

A small but realistic top-down car racing environment. Drive it yourself, then
train an RL agent to lap the same track as fast as it can without leaving the
tarmac. The human game and the agent use the **exact same environment**, so your
best lap and the agent's best lap are directly comparable.

## What's "realistic" here

Grip is modelled the way racing games and sims do it (`racing/car.py`):

- **Dynamic bicycle model** with body-frame longitudinal/lateral velocities and
  yaw rate.
- **Pacejka "magic formula" tyres** — lateral force rises with slip angle to a
  peak (~6°) then falls off as the tyre slides, instead of a straight line.
- **Friction circle (combined slip)** — each axle has one grip budget shared
  between cornering and drive/brake. The *driven* rear has a spinning wheel: its
  drive force comes from the wheel's **slip ratio** (how much faster the tyre is
  turning than the road) on the same Pacejka curve as cornering, so flooring it
  out of a slow corner lights up the rears — the spin bleeds lateral grip and the
  back steps out (power-on oversteer). It self-limits (the spin caps the thrust)
  and a lift re-hooks the tyre, so the slide is catchable. Up high, downforce
  loads the rears so hard they won't spin — planted on power. The front shares
  its budget between braking and cornering via a friction ellipse, so you can
  trail-brake and rotate the car into a corner — but braking at the very limit
  leaves nothing for steering, so you can't do both flat-out at once.
- **Aero downforce** — vertical tyre load (and thus grip) grows with speed²: the
  F1 signature of slippery-when-slow, planted-when-fast. Braking too: ~2 g at low
  speed (grip-limited) building to ~3.8 g up high as downforce loads the tyres.
- **Longitudinal weight transfer** — braking loads the front (turn-in),
  accelerating loads the rear (traction).
- **Wider rear tyres** — the rear axle has more grip than the front (`rear_grip_bias`),
  the F1 staple that keeps a rear-weight-biased car stable. Combined with the
  brake going front-first, the car stays planted under hard braking: brake in a
  straight line, then turn — exactly the technique a real car rewards.
- **Geared engine** — a torque curve over rpm through an automatic 6-speed
  gearbox: thrust steps down on each upshift then climbs back as the revs
  recover, so the car "catches its breath" between gears instead of pulling on
  one smooth hyperbola. The default profile is a ~1000 kg, ~620 hp downforce
  racer with more crank force than the rear tyres can hold at low speed (hence
  the wheelspin): a clean straight launch is ~2.0 s to 100 km/h, and it runs out
  against aero drag at ~270 km/h (drag-limited, well below the 6th-gear limiter).
- Off the tarmac the grass is low-grip and draggy, so running wide costs you.

Tune any of it in `CarParams` (mass, grip `mu`, downforce, engine/brake force,
steering lock/rate, …).

## Install

```bash
cd car-racing
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # core + RL
# or just the core to drive:  pip install numpy gymnasium pygame-ce
```

> Uses **pygame-ce** (a drop-in fork of pygame — code still does `import pygame`)
> because upstream pygame has no wheel for Python 3.14 yet and falls back to a
> source build that needs MSVC. If you ever have both installed, uninstall plain
> `pygame` first to avoid a clash: `pip uninstall pygame`.

## Drive it

```bash
python play.py                 # arrows / WASD; W·S throttle/brake; R restage; B beams; V view; Esc quit
python play.py --mouse         # steer inside the on-screen steering zone
python play.py --top-down      # overhead 2D view instead of the 3D hood cam
python play.py --track-seed 7  # a different circuit
python play.py --track power   # track character: balanced | power | flowing | technical
python play.py --handling ov   # handling preset: nimble | boat | un | ov (fair under/oversteer pair)
```

By default you drive from a **3D hood camera** — a perspective view mounted at the
car's nose that rotates with the car (software-rendered in pygame, no GPU or extra
dependencies). Pass `--top-down` for the overhead view, which is handy for seeing
the whole corner — or **swap between the hood cam and the overhead view live with
V** (works in `enjoy.py` too, so you can flip to the top-down to check the agent's
overall position). Both views draw the agent's **rangefinder sensor beams** (the
distances it observes) — toggle them with **B**.

With `--mouse`, the cursor's left/right position steers **anywhere on screen**
(the centre strip is a straight-ahead deadzone) — the **steering zone** drawn near
the bottom of the window is a visual readout of the wheel, not a region you have to
keep the cursor inside. Mouse buttons (or W/S) drive and brake.

Throttle and brake are **analog pedals**: holding W/Up (or the left mouse button)
ramps the throttle up and releasing eases it off, and S/Down (or the right button)
does the same for the brake — so a quick dab of the brake at speed is a light
squeeze, not an instant lock-up.

A yellow start/finish line marks the lap. You keep driving through laps and
off-track moments — the HUD shows speed plus your **current, last and best lap**
times. Putting a wheel on the grass **voids the lap in progress** (shown as
`LAP INVALID`), so only clean laps count toward your best.

## Quick sanity check (no display needed)

```bash
python smoke_test.py
```

## Train an agent

A self-contained **PyTorch PPO** — no stable-baselines3, no TensorBoard, just
`numpy + gymnasium + torch`. Training uses **curriculum learning** by default: it
starts on easy *flowing* circuits and auto-advances to *balanced* then tight
*technical* ones once the agent reliably finishes laps. Progress prints to the
console each update (add `--csv run.csv` to also log a row per update for plots).

```bash
pip install torch                              # the only extra dep for training
python train.py --timesteps 2_000_000          # PPO, 8 parallel envs, curriculum on
python train.py --no-curriculum --randomize-track   # one randomised setting instead
python enjoy.py --model ppo_racing.pt          # watch it drive
python enjoy.py --model ppo_racing.pt --viz    # + a separate model-activation window
```

`--viz` opens a second window showing the policy's live activations as it drives —
the input observation, both actor/critic hidden layers as heatmaps (red = +,
blue = −), and the action mean/std + value it's producing — handy for seeing
which units fire going into a corner. Uses pygame-ce's multi-window support; if
that's unavailable it just skips the viz.

Checkpoints land in `./checkpoints` (`ppo_racing_<steps>.pt` plus a best-so-far
`ppo_racing_best.pt`); the final policy is written to `--out` (default
`ppo_racing.pt`).

Training **auto-detects a CUDA GPU** (`--device cpu` / `--device cuda` to force).
Heads-up: the policy is a tiny MLP and the physics env runs on the CPU, so env
stepping dominates wall-clock. On a laptop GPU the default 64-wide net actually
ran *faster on CPU* (~1080 vs ~740 FPS); the two only break even around
`--hidden 256`. So for the default config CPU is the better pick — reach for the
GPU once you scale the network (or batch) up.

**Optional — Weights & Biases.** Console/CSV logging needs nothing extra, but you
can mirror every update to [W&B](https://wandb.ai) with `--wandb` (`wandb` is only
imported when you pass it):

```bash
pip install wandb && wandb login
python train.py --wandb                         # logs to the preset entity/project
python train.py --wandb --wandb-project my-proj --wandb-name run1   # override
```

It logs episode return, length, lap rate, the PPO losses, KL, FPS and the current
curriculum stage. Defaults: entity `jiucheng-zang-venuiti-solutions`, project
`racing-car-ppo training`.

## The RL problem

The setup is built for **racing — exploit the limits, minimise lap time** — not
lane-keeping, so the agent is free to use the whole track width and find the
out-in-out line.

- **Observation** (`Box`, ~[-1, 1], 27-dim): 15 rangefinder beams — packed densely
  toward the front (fine corner vision) and **EMA-smoothed** so a beam grazing a
  wall doesn't jitter the input (and saw the steering); + forward speed, lateral
  speed, yaw rate, heading error, steering angle + a **curvature preview**
  (continuous signed curvature at 20 / 50 / 100 m ahead) + **tyre state**
  (front/rear slip angles, rear slip ratio, vehicle sideslip — the cues a driver
  uses to catch a slide).
- **Action** (`Box`): `[steer, throttle]`, each in `[-1, 1]` (throttle negative =
  brake/reverse).
- **Reward**: forward progress along the track **spline** (Frenet Δs) plus a speed
  term — so the optimum is the fast geometric line, not the centre. Performance
  **drifting is allowed**: only sideslip past a threshold (`slip_threshold`) is
  penalised, to discourage spinouts without killing a controllable slide. Control
  penalties are light (aggressive limit-of-grip counter-steer shouldn't be taxed).
  **Graduated track limits**: running wide onto the grass is a recoverable per-step
  cost (and voids the lap), and only a *full* track exit ends the episode with a
  heavy, speed-scaled crash penalty — no instant reset for a minor clip. All
  weights are tunable `RacingEnv` args (`progress_w`, `speed_w`, `slip_w`,
  `grass_penalty`, …; the civilian `cte_w`/`heading_w` default to 0).
- **Training interventions**: episodes start at **racing speed** (60–150 km/h
  rolling starts, `--spawn-speed-kmh`) so the policy manages high momentum from
  tick one instead of creeping to dodge crashes; the optimiser is **AdamW**. (The
  human game keeps rolling through laps and off-tracks; only the RL episode resets,
  on a lap or a full track exit.)

Registered as a Gymnasium id too:

```python
import racing                       # registers "Racing-v0"
import gymnasium as gym
env = gym.make("Racing-v0", render_mode="human")
```

## Layout

```
racing/car.py     vehicle dynamics (the physics)
racing/track.py   procedural closed-loop track, progress + ray casting
racing/env.py     Gymnasium env (obs / action / reward) — shared by human & agent
racing/ppo.py     PPO actor-critic network (pure PyTorch) — shared by train & enjoy
racing/actviz.py  live activation-visualiser window (enjoy.py --viz)
racing/render.py  top-down 2D pygame rendering (lazy-imported; headless training needs no display)
racing/render3d.py  3D hood-cam renderer (perspective, software-rendered in pygame)
play.py           drive it yourself
train.py          PPO training (pure PyTorch, curriculum learning)
enjoy.py          watch a trained model (--viz for a model-activation window)
smoke_test.py     headless checks
```

## License

GPL-3.0 — see [LICENSE](LICENSE). Copyright (C) 2026 Jiucheng Zang, Venuiti Solutions.
