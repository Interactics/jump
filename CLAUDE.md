# CLAUDE.md — humanoid_jump

Re-implementation of the T-ASE paper **"A Three-Step Optimization Framework With Hybrid Models for a Humanoid Robot's Jump Motion"** on the **Unitree H1-2** in MuJoCo. One-semester project (KIT, winter 2026/27). Goal: fast optimization of humanoid jump + landing, and improving on the paper's gaps.

- Papers (PDFs): `../../Papers/` (the three-step paper, and its landing reference [2]: Qi et al., T-RO 2023 "Vertical Jump ... CoP-Guided Angular Momentum Control and Impact Absorption")
- Roadmap: Obsidian note `~/Documents/Obsidian Vault/Project - 2layer jump/Three-Stage Jump Optimization Roadmap (H1-2).md`
- Sister project: `../../h1_2_cop_cam_jump/`, my earlier re-implementation of the T-RO paper on H1-2 (QP-based CDC, CoP-guided CAM, IACVM landing). Reuse its model handling and landing controller; don't modify it from here.
- Remote: `github.com/Interactics/jump` (branch `main`)

## The paper in one screen

Three stages, each = one model + one solver. Launch phase makes momentum; flight phase uses inertia (L is constant in flight, ω = L / I).

| Stage | Model | Solver | Output |
|---|---|---|---|
| 1 | SRMP, 4 DOF (leg angle, leg length, barbell angle, barbell radius ϕ; ρ = 2·m_p·ϕ²) | NLP (TO) | CoM/CAM/inertia references, CoP constraint, takeoff state |
| 2 | Full 3D H1-2 | QP per timestep (no iterations) | Initial guess only |
| 3 | 7-DOF symmetric planar (base x, z, pitch + hip pitch, knee, ankle pitch, shoulder pitch; legs and arms merged) | NLP (TO) with CAM constraint (Eq. 47) | Final joint trajectory |

Execution in the paper: open-loop position playback for launch and flight; landing uses [2]'s online WBC. Control input in Stages 1 and 3 is **jerk**. Reference tracking goes in the cost (soft); targets such as takeoff velocity and landing posture are hard constraints.

Known gaps in the paper that this project aims to fill: no baseline for the claimed speed-up, no success rates, open-loop launch/flight, landing outside the optimization, sagittal + symmetric only, hand-tuned weights (0.1 to 1e8).

## Plan and status

Build order is **Stage 1 → Stage 3 → Stage 2** (Stage 2 only warm-starts Stage 3; solving Stage 3 from a naive guess first gives the baseline). Every phase = Learn → Build → Gate.

| Phase | Work | Gate | Status |
|---|---|---|---|
| 0 | Env, H1-2 model, joint map, spawn | `check_env.py` 12/12; robot stands | **Done** 2026-11-12: 12/12 (mass/CoM/CAM errors 0); stand 5 s: CoM drift 5.8 mm, tilt 0.6° |
| 1 | CasADi collocation toys in `toys/` (cart-pole trapezoid, jerk input, two-phase hop) | Two-phase toy converges from a poor guess | **Done** 2026-11-12: 20/20 random guesses reach the same optimum (median 35 iters, 78 ms); 432 rows tallied |
| 2 | Stage 1 SRMP TO (`models/srmp.py`, `stages/stage1_srmp.py`) | CoP in foot, inertia shaping in flight, landing angle hit | **Done** 2026-11-15 (joint mode, h 0.10 m, l 0.40 m, W_t = 5e7): 49 iters, 0.33 s; t_sl 0.844 s; CoP 0.008..0.048 m; ρ 7.0 → 6.1 in flight; landing angle −8.28° hit. Paper's sequential mode: flight infeasible |
| 3 | Stage 3 symmetric NLP, naive guess first (baseline: time, iterations, success) | Converges for one target; baseline logged | **Done** 2026-11-16 (h 0.10 m, l 0.40 m, naive guess = hold the crouch): **982 iters, 44.6 s**, 2139 variables, 2809 rows; lands at 0.400 m; peak torque (L+R) hip 156/400, knee 241/600, ankle 58/120, shoulder 80/80 N·m; CoP 0.008..0.048 m. Stage 1 adds 0.33 s |
| 4 | Stage 2 QP mapping + warm start | Yes/no with numbers: does it speed up Stage 3? (baseline to beat: 982 iters, 44.6 s) | Next |
| 5 | MuJoCo execution; landing with own WBC | Full jump without falling | **Launch + flight playback done** 2026-11-16 (`sim/mujoco_runner.py`, PD + torque feedforward): take-off v (0.439, 1.404) vs plan (0.463, 1.401) m/s, apex 0.964 vs 0.967 m, touch-down 1.124 vs 1.129 s at ankle x 0.370 vs 0.400 m, foot pitch 0.9°. Falls after touch-down (impact 5.3 BW): **landing controller still missing** |
| 6 | Evaluation: speed-up vs baseline, success rate, model-error robustness | One table + one figure | |

Start with modest jump targets (0.3–0.5 m forward). H1-2 is about 67 kg (handless) with the same 360 N·m legs as H1, so the paper's 1.0 m is not a first target.

## What exists now (through Phase 3)

One target runs end to end in optimization: **h = 0.10 m, l = 0.40 m**. Played back in MuJoCo up to touch-down (matches the plan); no landing controller yet, so the robot falls after landing.

1. `scripts/srmp_numbers.py` turns the H1-2 model into SRMP parameters (CoM height, pitch inertia, CoP limits).
2. Stage 1 (`models/srmp.py`, `stages/stage1_srmp.py`): SRMP launch + flight in one NLP → references for CoM velocity, CAM rate, inertia, and the phase durations (t_sl 0.844 s, t_f 0.286 s). 49 iters, 0.33 s.
3. Stage 3 (`models/symmetric.py`, `stages/stage3_nlp.py`): 7-DOF whole-body launch + flight in one NLP, from a naive guess → joint trajectory in `results/stage3_nlp.npz`. 982 iters, 44.6 s.

4. Playback (`sim/mujoco_runner.py`): joint PD + planned torque feedforward in the full MuJoCo model; plot and snapshots in `results/playback.png`.

Stage 2 (QP initial guess) is skipped so far by design: its only job is to speed up Stage 3, and the numbers above are the baseline it has to beat.

## Differences from the paper

Three kinds: **(G)** the paper gives no information, so a choice was needed; **(E)** the paper's formulation fails or contradicts itself when implemented; **(R)** a deliberate change for H1-2 or for physical consistency.

| # | Where | Paper | Ours | Why |
|---|---|---|---|---|
| 1 | Tools | MATLAB, GPOPS-II, SNOPT | Python, CasADi Opti with own trapezoidal collocation, IPOPT | (R) open tools. Iteration counts are not comparable across solvers |
| 2 | Robot | 42 kg, 1.5 m, 14 DOF, 1-DOF arms | Unitree H1-2 handless, 66.98 kg, 27 joints (4 pitch joints per side used, rest locked) | (R) our platform. Targets are ours (0.10 m, 0.40 m), not the paper's 1.0 m |
| 3 | SRMP dynamics (Eq. 2) | M, H not written out | Derived from the Lagrangian with CasADi; energy check 6e-16 | (G) |
| 4 | SRMP numbers (bounds in Eqs. 19, 23) | "manually designed", no values | Measured on H1-2: φ 0.60–0.93 m, ρ 6.1–13.6 kg·m², CoP −0.066..0.154 m | (G) |
| 5 | Stage 1 CoP constraint (Eq. 22) | Fraction with Fz in the denominator | Multiplied through by Fz, plus Fz ≥ 0 | (E) 0/0 at take-off, where Eq. 18 sets the force to 0 |
| 6 | Stage 1 solve order | Launch NLP, then flight NLP | Launch + flight in one NLP (`joint=True`) | (E) sequential flight is infeasible: launch ends with L = +1.2, flight needs L ≈ −6.5 |
| 7 | Stage 1 time weight W_t (Table I) | 5e5 | 5e7 | (E) with 5e5 the launch time runs to any upper bound (3.4 s unbounded) |
| 8 | Internal inertia ρ^I (Eq. 26) | Used, never given a value | 0 (ρ^R is the whole inertia) | (G) |
| 9 | Stage 1 initial state | Not stated | At rest in a crouch (a = 0.8 rad), barbell aligned with the leg | (G) |
| 10 | 7-DOF model (Sec. III-C) | Single-leg 5-link model, leg and arm parameters aggregated | Full URDF in Pinocchio with a planar root and a left = right matrix | (R) nothing merged by hand; verified against MuJoCo to 1e-6 |
| 11 | Stage 3 solve order | Launch, then flight | One NLP | (R) same reason as #6 |
| 12 | Foot contact in launch (Eq. 50) | Toe fixed, heel may lift, ground torque applied at the toe | Whole sole fixed and flat; CoP inside the sole | (R) a point contact cannot transmit that torque. Cost: no heel-lift push-off |
| 13 | Limits in Stage 3 | "Boundary constraints", none listed | Joint torque limits, joint ranges, friction (μ 0.7), CoP, pelvis pitch ±1 rad | (R) H1-2's ankle is weak (120 N·m for both) |
| 14 | Heel velocity cost (Eq. 46) | Weight 1.7e5 | Dropped | Follows from #12: the heel does not move |
| 15 | Take-off | Only CoM velocity (Eq. 51) | Also ground wrench = 0 at the last launch node | (R) acceleration is a continuous state across the phase link (Eq. 55) |
| 16 | Landing (Eq. 56) | Foot − CoM offset mirrors the **initial** posture | Foot lands at x = l, offset mirrors the **take-off**, sole flat on the floor | (E) Eq. 56 contradicts Eq. 17; as written a 0.40 m target gives 0.23–0.31 m |
| 17 | Flight weights (Table IV) | ρ 0.1, ρ̇ 1e8, θ̈ 2e5, jerk 1, ankle 1e8 | ρ 0.1, ρ̇ 1e3, θ̈ 1 (joints only), jerk 1e-3, ankle 1e3 | (E) no convergence in 2000 iters with the paper's values on H1-2 |
| 18 | Take-off node constraints | Not discussed | Duplicated dynamics and inequalities on the zero wrench removed | (E) otherwise the constraints are degenerate and IPOPT does not converge |
| 19 | Scaling | Not discussed | Forces in body weights, objective × 1e-4 (Stage 1) and × 1e-6 (Stage 3) | (R) IPOPT needs it with weights up to 1e8 |
| 20 | Stage 2 | QP guess feeds Stage 3 | Not built yet; Stage 3 starts from "hold the crouch" | (R) build order: baseline first |
| 21 | Execution of launch + flight | Open-loop joint position playback | Joint PD + planned torque as feedforward | (E) position-only playback overshoots take-off vz by 28 % in MuJoCo (1.80 vs 1.40 m/s) |

Used as printed: Table I except W_t; Table III (launch, minus the heel term); jerk as the input in both stages; the hard CAM constraint (Eq. 47); soft tracking of CoM velocity and inertia; hard take-off velocity.

Findings about the paper so far (candidates for the report):

- Stage 1's two NLPs cannot be solved one after the other as described (#6).
- The launch duration is not determined by the published cost (#7).
- The inertia reference is not followed by the whole body: Stage 3 tracks L exactly but ρ is 11.1 vs 7.0 at take-off (arms swing up). The backward L that Stage 1 asks for comes from the SRMP's massless leg; a real robot can also swing its legs forward.
- Eqs. 17 and 56 disagree on the landing posture (#16).
- The "under 10 s" claim has no baseline. Ours from a naive guess: 44.6 s (982 iters) on a newer CPU than the paper's i5-9400F, so hardware does not explain the gap; the initial guess, the solver and the extra limits (#13) do. Phase 4 measures the first of these.

## Environment and commands

Conda env **`mujoco`** (Python 3.11). On this Mac the env is **`mujoco_env`** (MuJoCo 3.9.0, CasADi 3.7.2, Pinocchio 4.0.0 with `pinocchio.casadi`): use `./setup.sh mujoco_env`. macOS, Apple Silicon.

```bash
conda activate mujoco
./setup.sh                                 # adds packages from environment.yml, downloads the H1-2 URDF, runs checks
python scripts/check_env.py                # Phase 0 gate
mjpython scripts/spawn_robot.py            # viewer; on macOS the passive viewer needs mjpython
python scripts/spawn_robot.py --headless   # 5 s stand test, PASS/FAIL
```

- Pinocchio must come from **conda-forge**. The pip wheel `pin` has no `pinocchio.casadi`.
- IPOPT ships with CasADi. QP: OSQP or ProxQP.
- Paper tool → ours: GPOPS-II → CasADi (own direct collocation, Opti stack); SNOPT → IPOPT.

## Layout

```
config/robot.yaml        model paths + check tolerances
config/joint_map.yaml    H1-2 → 7-DOF symmetric mapping (draft; elbow angle is an open question)
scripts/                 check_env.py, spawn_robot.py, srmp_numbers.py (H1-2 → SRMP parameters)
toys/                    Phase 1 collocation exercises
humanoid_jump/
  collocation.py         shared multi-phase collocation helpers
  models/                srmp.py (Stage 1), symmetric.py (Stage 3), full.py (Pinocchio, Stage 2)
  stages/                stage1_srmp.py, stage2_qp.py, stage3_nlp.py
  sim/robot.py           MuJoCo model build, spawn, joint PD
  sim/mujoco_runner.py   execution + landing (Phase 5)
assets/h1_2/             MJCF + scene + meshes (unitree_mujoco copy, identical to upstream)
third_party/             unitree_ros h1_2_description (URDF for Pinocchio), git-ignored
results/                 plots and logs
```

`collocation.py` has `trapezoid` (defect rows), `trapz_cost`, `solve`. `models/srmp.py` (Lagrangian SRMP, self-check: `python -m humanoid_jump.models.srmp`) and `stages/stage1_srmp.py` (`launch(h, l, joint=...)`, `flight(la)`; run `python -m humanoid_jump.stages.stage1_srmp [h l]`, plot in `results/stage1_srmp.png`) are done. `models/symmetric.py` (7-DOF planar model from the URDF via Pinocchio, planar root + left = right matrix `S`; self-check against MuJoCo: `python -m humanoid_jump.models.symmetric`) and `stages/stage3_nlp.py` (`solve(ref, hard_cam=..., guess=...)`; run `python -m humanoid_jump.stages.stage3_nlp [h l]`; plot `results/stage3_nlp.png`, trajectory `results/stage3_nlp.npz`) are done. `sim/mujoco_runner.py` plays `results/stage3_nlp.npz` back (`python -m humanoid_jump.sim.mujoco_runner [--no-ff]`, viewer: `mjpython -m humanoid_jump.sim.mujoco_runner --view`). `models/full.py` and `stages/stage2_qp.py` are still docstring-only placeholders.

**Solve time is a tracked result**: every stage prints iterations and wall time; record them here whenever a formulation changes.

Toys: `python toys/<name>.py`, each ends with an assert self-check.

## Robot model facts (verified)

- Use `h1_2_handless`: 27 actuated joints, 66.98 kg. The URDF (unitree_ros) and the MJCF (unitree_mujoco) match exactly in mass, CoM and CAM.
- Legs have 6 DOF: hip yaw/pitch/roll, knee, ankle pitch/roll. Arms have 7: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw. Plus `torso_joint`. Joint names follow `left_hip_pitch_joint` etc.
- Actuators are torque motors. ctrlrange: hips ±200, knee ±300, **ankle pitch ±60, ankle roll ±40** N·m. The weak ankle limits CoP and toe-off, so put torque limits into Stage 3 early.
- `sim/robot.py` model fixes (from the sister project): foot mesh collision replaced by a sole box (x ∈ [-0.086, 0.174], y ∈ ±0.043, z = -0.045 in the ankle-roll frame); no damping, armature or friction on the free joint; no self-collision.
- SRMP parameters (`scripts/srmp_numbers.py`, support point = ankle projected on the floor): CoM height 0.62 m (a = 1.0) to 0.93 m (standing, arms down); pitch inertia ρ from 6.1 (deep crouch, arms down) to 14.0 kg·m² (standing, arms up); raising the arms adds about 4.5 kg·m². CoP limits −0.066..0.154 m (sole minus 2 cm).
- 7-DOF symmetric model vs MuJoCo over 20 random states: CoM 1e-8 m, CAM 1.5e-6, ρ 9e-7, bias forces 8.5e-6. Limits used (L+R sums): hip 400, knee 600, ankle 120, shoulder 80 N·m; ranges hip −3.14..2.5, knee −0.12..2.19, ankle −0.897..0.524, shoulder −3.14..1.57 rad. Locked joints keep `ARMS` (elbow 0.6, shoulder roll ±0.15).
- Free joint: MuJoCo quaternion is (w, x, y, z), Pinocchio's is (x, y, z, w). MuJoCo free-joint angular velocity is in the local frame.

## Lessons learned (don't repeat)

- **Standing with joint PD needs stiff gains**: hips/knees/torso 1000, ankles 600, arms 100 (kd = 1.6·√kp). Ankle kp must exceed m·g·h ≈ 600 N·m/rad; with 300/80 the knees fold and the robot falls backward in about 1 s.
- `qfrc_bias` on a floating base is **not** standing gravity compensation (the free body is in free fall), so it doesn't help a PD stand.
- The CoM can't be centred over the feet by shifting the base: the feet move too. At a = 0.2 rad the CoM sits about 24 mm behind the sole centre (still inside the foot).
- urdfdom prints "Visual material must contain a name attribute" for every link when Pinocchio loads the URDF. It concerns visuals only; inertias load fine (check_env 12/12).
- **Trapezoid error compounds in open-loop playback.** Cart-pole, N = 20/40/80/160: open-loop RK4 end error 89/24/5.8/1.4° (O(h²)). The paper plays launch and flight back open-loop, so check the grid density in the final model, or close the loop.
- A point mass on a point foot can lean freely (no CoP limit): in the hop toy, D = 0.4 m is reached with 0.87 s of stance and only 0.07 s of flight. The SRMP's CoP constraint (Stage 1) is what rules this out.
- CasADi Opti: `vertcat(opti.bounded(...), opti.bounded(...))` is rejected ("Constraint type unknown"); add each `bounded` separately.
- **Stage 1, paper's sequential order fails here.** The launch NLP minimises ‖L_end‖² (Eq. 14) and gives L = +1.2, but a mirrored landing (Eq. 27) needs the body to turn back by 2θ ≈ 16.5° in 0.286 s, i.e. L ≈ −6 to −14. Inertia shaping alone only reaches +9.0..+10.8°. Solving launch and flight in one NLP (`joint=True`) fixes it (L = −6.5). The paper must have coupled them somehow (Eq. 28 text) but doesn't say how.
- Stage 1 launch time is not determined by the paper's cost: the costs are node sums with fixed N, so slow motion is cheap. With Table I's W_t = 5e5, t_sl sits on any upper bound; unbounded it goes to 3.4 s (only 4 of 40 nodes left in the 0.34 s push). Decision 2026-11-15: **W_t = 5e7** (100× the paper) gives an interior t_sl = 0.84 s; `t_max = 2.0` is only a safeguard. The push itself barely changes with t_sl (0.25–0.375 s, peak Fz 1.8–2.0 BW).
- Eq. 22's CoP fraction is 0/0 at take-off (Eq. 18 sets GRF = 0): use it multiplied by Fz, with Fz ≥ 0. Table I weights up to 1e8 need objective scaling (1e-4) for IPOPT.
- **Stage 3 would not converge until the take-off node was cleaned up.** (1) Flight base dynamics at k = 0 repeat the launch's last node (via the phase link): 3 redundant equalities. (2) Fz ≥ 0, friction and CoP at the node where the wrench is fixed to 0: five active inequalities on fixed variables. Both break constraint qualification; IPOPT then shows lg(rg) ≈ 12 and inf_du up to 1e13. Removing them: 3000 iters without convergence → 601.
- Stage 3 scaling: ground wrench as a variable in body weights, dynamics rows divided by m·g, objective × 1e-6, `expand=True` (3× faster per iteration, about 45 ms).
- **Table IV weights (1e8 on ρ̇ and ankle velocity, 2e5 on θ̈) do not converge on H1-2** (2000 iters). Used instead: ρ 0.1, ρ̇ 1e3, θ̈ 1 (joints only), jerk 1e-3, ankle 1e3. Table III (launch) is used as printed.
- **Eq. 56 contradicts Eq. 17.** Eq. 17 assumes the landing mirrors the take-off CoM offset (0.134 m); Eq. 56 mirrors the initial offset (0.048 m), and Stage 3 does not constrain the take-off CoM position at all. As written the jump came out 0.23–0.31 m for a 0.40 m target. Fix: landing foot x = l (our own target) and mirror the take-off offset.
- Stage 3 with the hard CAM constraint (Eq. 47) tracks L exactly (−6.48 vs −6.49) but not ρ (take-off 11.1 vs 7.0, flight RMS error 4.4): the arms swing up. The SRMP's ρ reference is not something the whole body follows. Without Eq. 47 the solver picks L = −21.9 and the pelvis hits the −57° bound, so keep `hard_cam=True`.
- Stage 3 uses a flat foot in launch (no heel lift) and position-level sole constraints; the sole velocity residual is up to 1.3e-2 m/s between nodes. Check this in Phase 5 playback.
- **Playback needs the torque feedforward.** PD + planned torque: take-off vz 1.404 (plan 1.401), L −6.29 (plan −6.48), apex within 3 mm. PD only (the paper's position playback, same stiff gains): vz 1.80, apex 4 cm too high, lands 70 ms late, L −4.5. No torque saturation in either run.
- Remaining playback error with feedforward: lift-off 23 ms early, vx 0.024 m/s low, lands 3 cm short. The flat-foot Stage 3 with N = 40/30 nodes is good enough for this target; no need to refine the grid yet.
- Comparing Pinocchio and MuJoCo: map joints by name, not by index; keep base velocity at zero when comparing CAM, to avoid frame-convention mismatches.

## How to work with me

- Big picture first: structure and relationships before details. Physical intuition before formalism. Keep explanations short; don't dump everything at once.
- Verify with code: run things and report numbers, not just claims. I watch for detail-level errors, and I appreciate it when you check calculations and units.
- When reading papers, be critical: point out unstated assumptions and weak evidence, and cite equation numbers.
- Prefer tools and automation over manual, repetitive checking.
- I'm not a native English speaker. When I write in English, briefly suggest a more natural phrasing before answering. Answer in English unless I ask for Korean.
