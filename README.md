# humanoid_jump - Three-Stage Jump Optimization (H1-2)

Implementation of the T-ASE three-stage jump framework (SRMP TO → per-timestep QP → symmetric-body NLP) on the Unitree H1-2, in MuJoCo.
Roadmap: Obsidian note "Three-Stage Jump Optimization Roadmap (H1-2)".

## Setup (Phase 0)

```bash
chmod +x setup.sh
./setup.sh            # adds packages to the conda env "mujoco", downloads H1-2, runs the checks
```

Or step by step:

```bash
conda env update -n mujoco -f environment.yml
git clone --depth 1 --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git third_party/unitree_ros
git -C third_party/unitree_ros sparse-checkout set robots/h1_2_description
conda activate mujoco
python scripts/check_env.py
```

The Phase 0 gate passes when all 12 checks pass. Pinocchio prints `Visual material must contain a name attribute` errors while it loads the URDF. They only concern visual meshes and are harmless.

## Spawn the robot

```bash
conda activate mujoco
mjpython scripts/spawn_robot.py            # viewer (macOS needs mjpython)
mjpython scripts/spawn_robot.py --a 0.35   # deeper crouch: hip=-a, knee=2a, ankle=-a
python scripts/spawn_robot.py --headless   # 5 s stand test, prints PASS/FAIL
```

The H1-2 spawns with both soles flat on the floor and holds the pose with joint PD
(hips/knees/torso kp 1000, ankles 600, arms 100). Model fixes come from `h1_2_cop_cam_jump`:
sole boxes instead of the foot mesh, no damping on the free joint, no self-collision.

## Layout

```
humanoid_jump/
├── environment.yml          # packages added to the "mujoco" env
├── setup.sh                 # one-time setup (Phase 0)
├── config/
│   ├── robot.yaml           # model paths + Phase 0 tolerances
│   └── joint_map.yaml       # H1-2 -> 7-DOF symmetric model mapping (draft)
├── scripts/
│   ├── check_env.py         # Phase 0 gate
│   └── spawn_robot.py       # spawn H1-2 and hold a standing pose
├── toys/                    # Phase 1: collocation toy problems
├── humanoid_jump/           # Python package
│   ├── collocation.py       # shared CasADi collocation helpers (Phase 1)
│   ├── models/              # srmp.py (Stage 1), symmetric.py (Stage 3), full.py (Stage 2)
│   ├── stages/              # stage1_srmp.py, stage2_qp.py, stage3_nlp.py
│   └── sim/                 # robot.py (model + spawn + PD), mujoco_runner.py (Phase 5)
├── assets/h1_2/             # H1-2 MJCF + scene + meshes (copied from h1_2_cop_cam_jump)
├── results/                 # plots, logs (git-ignored)
└── third_party/             # downloaded H1-2 model (git-ignored)
```

## Stack

| Paper | Here |
| --- | --- |
| GPOPS-II | CasADi (own direct collocation) |
| SNOPT | IPOPT (ships with CasADi) |
| QP solver | OSQP / ProxQP |
| Robot model | Pinocchio (conda-forge, with `pinocchio.casadi`) |
| Robot | MuJoCo, `h1_2_handless` (about 67 kg) |
# jump
# jump
