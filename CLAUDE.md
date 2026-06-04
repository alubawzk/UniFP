# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Note: a different `CLAUDE.md` exists one level up in `mini3_lab/` for the **MINI3-Train** project (IsaacLab/IsaacSim). It is unrelated to this repo. **UniFP** is a standalone IsaacGym Preview 4 / legged_gym project — ignore the parent file's commands, env IDs, and architecture when working here.

## Project Overview

**UniFP** (Unified Force and Position control, CoRL 2025 Best Paper — [website](https://unified-force.github.io/), [arxiv](https://arxiv.org/pdf/2505.20829)) is a whole-body RL framework for the **B2Z1** robot: a Unitree B2 quadruped (12 leg DoF) with a Unitree Z1 arm + gripper mounted on top. The single policy learns to track **both** velocity/position commands and **external force** commands at the end-effector (gripper) and the base simultaneously.

It is a fork of ETH/Nikita Rudin's `legged_gym` + `rsl_rl`, restructured around B2Z1 and a custom PPO variant. Only the B2Z1 task is implemented; README mentions of G1/H1/humanoidgym tasks are aspirational and **not present** in this repo.

## Environment & Dependencies

- Ubuntu 20.04/22.04, **Python 3.8** (hard requirement — IsaacGym needs ≤3.8), CUDA 11.2+
- IsaacGym Preview 4 (download separately from NVIDIA; install via `cd isaacgym/python && pip install -e .`)
- PyTorch 2.3.1, `numpy==1.23`, `mujoco==3.2.3`, `params_proto`, `pydelatin`, `wandb`, `tensorboard`
- Install this package: `pip install -e .` (see `setup.py`)
- libpython error fix: `export LD_LIBRARY_PATH=<conda_env>/lib:$LD_LIBRARY_PATH`

## Key Commands

All scripts run from `legged_gym/scripts/`. There is **one** registered task: `b2z1_pos_force`.

**Train:**
```bash
cd legged_gym/scripts
python train_b2z1posforce.py --task=b2z1_pos_force --headless
# train.py forces args.headless = True regardless of the flag
```

**Play / evaluate** (also exports JIT policy modules):
```bash
python play_b2z1posforce.py --task=b2z1_pos_force --load_run=<run_name>
```

**Common CLI args** (defined in `legged_gym/utils/helpers.py`, `get_args`):
`--task` (default `go2`, use `b2z1_pos_force`), `--resume`, `--experiment_name`, `--run_name`, `--load_run` (`-1` = latest), `--checkpoint` (int, `-1` = latest), `--num_envs`, `--seed`, `--max_iterations`, `--flat_terrain` (sets `terrain.height=[0,0]`), `--rl_device`, `--headless`, `--observe_gait_commands`.

**Logs / checkpoints:** saved to `logs/<experiment_name>/<MonDD_HH-MM-SS>_<run_name>/` (experiment_name is `b2z1_pos_force`). `logs/` is gitignored. Exported JIT policies go to `logs/<experiment_name>/exported/policies/`.

No test suite, linter, or CI exists in this repo.

## Architecture

### Three-layer structure (legged_gym lineage)

```
legged_gym/
  envs/
    base/      legged_robot.py, base_task.py + *_config.py  — generic IsaacGym quadruped base
    b2/        legged_robot_b2z1_pos_force.py (2300+ lines) + b2z1_pos_force_config.py  — the UniFP task
  b2_gym_learn/   custom RL library (the "rsl_rl" of this repo), package ppo_cse_pf
    ppo_cse_pf/  actor_critic.py, ppo.py, on_policy_runner.py, rollout_storage.py
    env/         vec_env.py (VecEnv interface)
    utils/
  scripts/     train_b2z1posforce.py, play_b2z1posforce.py
  utils/        task_registry.py, helpers.py, terrain.py, math.py, logger.py
resources/robots/b2z1/  b2z1.urdf + meshes
```

Task wiring happens in `legged_gym/envs/__init__.py`, which imports the env + config classes and calls `task_registry.register("b2z1_pos_force", LeggedRobot_b2z1_pos_force, B2Z1PosForceRoughCfg(), B2Z1PosForceRoughCfgPPO())`. `TaskRegistry` (in `utils/task_registry.py`) builds the env (`make_env`) and the PPO runner (`make_alg_runner`).

`LeggedRobot_b2z1_pos_force` subclasses the base `LeggedRobot`; almost all UniFP-specific behavior (force commands, EE goals, gripper/base pushes, the full reward set) lives in `legged_gym/envs/b2/legged_robot_b2z1_pos_force.py`. The reward methods are `_reward_*` and selected/weighted by `rewards.scales` in the config.

### Config system

Two `params_proto`-style nested-class configs per task (subclassing the base configs in `envs/base/legged_robot_config.py`):
- `B2Z1PosForceRoughCfg` — scene, robot init pose, domain randomization, **commands** (incl. force-push schedules), terrain, control gains, rewards & scales, env dims.
- `B2Z1PosForceRoughCfgPPO` — `policy` (actor/critic `[512,256,128]`), `algorithm`, `runner` (`experiment_name`, `run_name`). Class names selected at runtime: `policy_class_name='ActorCritic'`, `algorithm_class_name='PPO'`, `runner_class_name='OnPolicyRunner'`.

Key env dims (`config.env`): `num_actions=17` (12 legs + 5 arm; gripper handled separately, `num_gripper_joints=2`), `num_single_obs=73`, `frame_stack=32` → `num_observations=32*73`, privileged obs `single_num_privileged_obs=149` with `c_frame_stack=3`, `num_pred_obs=12`. `control.decimation=4`, `action_scale=0.25`.

### The "cse_pf" PPO variant — what makes UniFP different

`ppo_cse_pf` is a teacher/student asymmetric-PPO with an **adaptation (estimation) module**. In `actor_critic.py`:

- **`adaptation_encoder_module`**: maps the full stacked observation history (`num_obs = frame_stack*num_single_obs`) → a latent of size `(frame_stack/num_single_obs gives ...)` `num_latent_dim`. The actor consumes `current_obs (num_single_obs) + latent`.
- **`adaptation_decoder_module`**: decodes the latent → `num_pred_obs (12)` predicted privileged quantities. Supervised via `AC_Args.adaptation_labels = ["base_velocity_loss", "gripper_pos_loss", "force_ee_loss", "force_base_loss"]`, dims `[3,3,3,3]`, weights `[.2,.2,1.0,1.0]`. **This is the core mechanism**: the policy estimates EE force, base force, base velocity, and gripper position from proprioceptive history, so external forces are observable at deploy time without a force sensor.
- **critic**: takes privileged obs directly.

The three JIT modules exported by `play` (`adaptation_module.pt` = encoder, `adaptation_decoder.pt`, `actor_body.pt`) are the deployable artifacts. At inference, `policy_info["latents"]` carries the decoded predictions (e.g. indices `6:9` = ee_force, `9:12` = base_force — see `play_b2z1posforce.py` `VISUAL_PRED`).

The runner threads three observation streams everywhere: `obs`, `privileged_obs`, `obs_pred` (see `on_policy_runner.py` / `rollout_storage.py`). When editing obs/action dims, all three plus the config `env.*` sizes must stay consistent.

### Force/position control mechanics (in `legged_robot_b2z1_pos_force.py`)

- **Force commands** are realized by *applying* external forces in sim and asking the policy to maintain its command while resisting/exploiting them. `_push_gripper` and `_push_robot_base` apply randomized force profiles (kp/kd "force gains", interval/duration/probability schedules in `config.commands`); `forces_local` holds the per-body external force in the base-yaw frame, scaled into observations by `obs_scales.ee_force` / `base_force`.
- Force training is curriculum-gated: external base force only starts after `commands.force_start_step` (8000) iterations — see the `global_steps` check in `step()`.
- **EE goals** are sampled in a sphere around the base (`goal_ee` / `arm` configs, `_resample_ee_goal*`); reward terms `tracking_ee_force_world`, `tracking_lin_vel_force_world` drive unified force+velocity tracking.
- `commands.num_commands = 15` packs velocity, force, and EE-goal channels together.
- `env.teleop_mode` switches command sourcing to external teleop input (off for training).

## Gotchas

- **Python must be 3.8** and IsaacGym must be importable *before* torch in scripts (note `import isaacgym` precedes `import torch` in every entry point) — reordering breaks the IsaacGym/torch interop.
- The README references files that don't exist here (`task_registry_b2z1posforce.py`, G1/H1 tasks). Trust the code, not the README task list.
- Changing any observation/action/privileged/pred dimension requires updating `config.env` sizes **and** the corresponding `nn.Linear` input sizes flow automatically from those configs — verify `num_single_obs`, `num_latent_dim` math in `ActorCritic.__init__` after edits.
- `train.py` hard-codes `args.headless = True`; pass through the GUI only via `play`.
