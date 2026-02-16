# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics-informed flow matching for robot imitation learning. Combines flow matching with the Affine Geometric Heat Flow (AGHF) to learn trajectory generation from demonstrations that may include dynamically infeasible examples. Built on NVIDIA Isaac Lab / Isaac Sim for simulation and trajectory collection.

**Core idea**: decompose a vector field on trajectory space into two components:
- `v_phys(γ)` — fixed AGHF vector field that drives trajectories toward dynamic feasibility (not learned)
- `v_learn(γ; θ)` — neural network that captures demonstration preferences (learned)

The combined field `v_θ(γ) = v_phys(γ) + v_learn(γ; θ)` is trained via flow matching loss, and inference is a single ODE integration from noise to trajectory.

**Current state**: the codebase was generated from the Isaac Lab extension template and currently implements a cartpole balancing task with RSL-RL PPO. The pendulum environment conversion and flow matching pipeline are the next steps.

### Research Context

- See `notes/writeup_v3.pdf` for the theoretical framework (AGHF, flow matching, loss formulation)
- See `notes/writeup_isaaclab_v2.pdf` for the Isaac Lab implementation plan (single pendulum, discretization, training pseudocode)
- ROAHM Lab, University of Michigan

### High-Level Pipeline

1. **Expert RL training** — train PPO policy in Isaac Lab to produce expert demonstrations
2. **Trajectory collection** — record state trajectories `{[θ_k, θ̇_k]}` from expert rollouts (Isaac Lab is a non-differentiable black box here)
3. **Flow matching training** — train `v_learn` neural network using the decomposed vector field loss: `L = ||v_phys(γ_s) + v_learn(γ_s; θ) - v*||²`
4. **Inference** — generate trajectories by integrating `dγ/ds = -v_θ(γ)` from `s=1` (noise) to `s=0` (data)

### Key Mathematical Objects

| Object | Description | Source |
|--------|-------------|--------|
| State `x_k = [θ_k, θ̇_k]` | Joint position + velocity | `joint_pos`, `joint_vel` from Isaac Lab |
| Demo trajectories `γ_demo` | Expert rollout states at each control step | Collected from trained RL policy |
| Noise trajectories `γ_noise` | Synthetic (e.g., linear interpolation between boundary states) | Generated |
| Interpolated `γ_s` | `(1−s)·γ_demo + s·γ_noise` | Vectorized computation |
| Target velocity `v*` | `γ_demo − γ_noise` | Pointwise subtraction |
| Physical action `A_phys` | `k_d · Σ ||(q_{k+1}−q_k)/Δt − q̇_k||²·Δt` | Forward finite differences on trajectory |
| `v_phys(γ)` | `−∇_γ A_phys(γ)` (with `G=I` for single pendulum) | `torch.autograd.grad` |
| `v_learn(γ; θ)` | Neural network output | Standard forward pass + backprop |

### First Experiment: Single Pendulum

- Configuration: `q = θ ∈ R` (angle from vertical), state `x = [θ, θ̇] ∈ R²`
- Adapted from cartpole by removing the sliding cart (keep only revolute joint)
- Control input: joint position delta `Δq`, tracked by virtual PD controller
- Simulation: `Δt = 1/120s`, decimation 2 → control period `Δt_ctrl = 1/60s`
- No backpropagation through the simulator is needed

## Prerequisites

Requires a working Isaac Lab installation (Isaac Sim 4.5.0–5.1.0). Python 3.10+ with the `env_isaaclab` conda environment active. Isaac Lab is installed locally at `~/src/IsaacLab/`.

## Common Commands

```bash
# Install the extension (editable mode)
python -m pip install -e source/physics_lab

# List registered environments
python scripts/list_envs.py

# Train with RSL-RL
python scripts/rsl_rl/train.py --task=Template-Physics-Lab-Direct-v0

# Play/evaluate a trained checkpoint
python scripts/rsl_rl/play.py --task=Template-Physics-Lab-Direct-v0

# Test with dummy agents
python scripts/zero_agent.py --task=Template-Physics-Lab-Direct-v0
python scripts/random_agent.py --task=Template-Physics-Lab-Direct-v0

# Lint and format
pre-commit run --all-files
```

## Architecture

### Environment Registration Flow

Environments are registered as Gymnasium tasks. The entry point chain:

1. `source/physics_lab/physics_lab/__init__.py` — imports `tasks.*`, triggering registration
2. `tasks/direct/physics_lab/__init__.py` — calls `gym.register()` with id `Template-Physics-Lab-Direct-v0`
3. The gym entry maps to `PhysicsLabEnv` (env class) + `PhysicsLabEnvCfg` (config dataclass) + `PPORunnerCfg` (agent config)

Scripts must `import physics_lab.tasks` before calling `gym.make()` to trigger registration.

### Key Source Locations

- **Env logic**: `source/physics_lab/physics_lab/tasks/direct/physics_lab/physics_lab_env.py` — `PhysicsLabEnv(DirectRLEnv)` with reward computation via `@torch.jit.script`
- **Env config**: `...physics_lab_env_cfg.py` — `PhysicsLabEnvCfg(DirectRLEnvCfg)` with sim params, reward scales, reset conditions
- **Agent config**: `...agents/rsl_rl_ppo_cfg.py` — `PPORunnerCfg` (hyperparameters for RSL-RL PPO)
- **Training/play scripts**: `scripts/rsl_rl/train.py` and `play.py` — handle AppLauncher, Hydra config, RSL-RL runner setup
- **Isaac Lab (upstream)**: `~/src/IsaacLab/` — local installation, docs at `~/src/IsaacLab/docs/`

### Adding a New Task

1. Create env + config classes under `source/physics_lab/physics_lab/tasks/direct/<name>/`
2. Register via `gym.register()` in that directory's `__init__.py` (use `Template-` prefix for `list_envs.py` discovery)
3. Provide an agent config (e.g., RSL-RL runner cfg) referenced in the gym kwargs

### Extension Packaging

`source/physics_lab/config/extension.toml` defines Omniverse extension metadata and Isaac Lab dependencies. `source/physics_lab/setup.py` reads from this TOML for pip install.

### Isaac Lab Patterns

- **Direct RL envs** inherit from `DirectRLEnv` with a `DirectRLEnvCfg` config. User implements `_setup_scene`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`.
- **Articulation data** is accessed via `robot.data.joint_pos[:, idx]`, `robot.data.joint_vel[:, idx]`, `robot.data.root_pos_w`, etc.
- **Scene setup** uses `{ENV_REGEX_NS}` placeholder in prim paths for per-environment cloning.
- **AppLauncher** must be initialized before importing any Isaac Lab modules (hence E402 being ignored).
- **Actuators**: `ImplicitActuatorCfg` for physics-engine PD control, explicit actuators for custom motor models.

## Code Style

- **Ruff** for linting/formatting (line length 120, Python 3.10 target)
- Custom isort section ordering: stdlib → third-party → omniverse → isaaclab → first-party → local
- `E402` (module-level import order) is intentionally ignored — Isaac Lab scripts require `AppLauncher` initialization before most imports
- License headers are auto-inserted by pre-commit on `.py` and `.yaml` files
- Uses `@configclass` decorator (Isaac Lab's dataclass variant) for all configuration classes

## Training Output

Logs go to `logs/rsl_rl/<experiment_name>/<timestamp>/` (gitignored). Supports TensorBoard, W&B, and Neptune loggers via `--logger` flag.
