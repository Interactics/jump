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
| 3 | Stage 3 symmetric NLP, naive guess first (baseline: time, iterations, success) | Converges for one target; baseline logged | **Done** 2026-11-16, re-solved 2026-11-17 with landing offset 0.09 m (h 0.10 m, l 0.40 m, naive guess = hold the crouch): **603 iters, 26.9 s**, 2139 variables, 2809 rows; lands at 0.400 m; peak torque (L+R) hip 155/400, knee 243/600, ankle 60/120, shoulder 80/80 N·m; CoP 0.010..0.048 m. Stage 1 adds 0.54 s (90 iters). Iterations depend strongly on the landing offset: 0.134 (mirror) 982, 0.12 358, 0.11 700, 0.10 1442, 0.09 603; **0.07 and 0.048 do not converge in 3000** |
| 4 | Stage 2 QP mapping + warm start | Yes/no with numbers: does it speed up Stage 3, and does it make it converge more reliably? | **Done** 2026-11-16. **Faster: yes, modestly. More reliable: no.** 11 targets (`scripts/bench_warmstart.py`, `results/warmstart.json`): success 7/11 for both naive and Stage 2 guess (different targets fail); median Stage 3 time 23.9 s vs 32.4 s; on the 5 common successes ×1.8 (range ×0.7–×2.4). Stage 2 itself: 0.2 s. Needs warm-start IPOPT options. Current target: **12.3 s total** (0.58 + 0.21 + 11.5 s, 221 iters) vs 28 s naive. Side finding, not in the pipeline: with variable scaling (outside the paper) Stage 3 converges 11/11 in about 18 iters from either guess |
| 5 | MuJoCo execution; landing with own WBC | Full jump without falling | **Done** 2026-11-17 for one target: `python -m humanoid_jump.sim.mujoco_runner` → STANDING. Take-off v (0.517, 1.403) vs plan (0.545, 1.401) m/s, apex 0.958 vs 0.961 m, touch-down 1.122 vs 1.127 s at ankle x 0.365 vs 0.400 m, impact 5.7 BW; 2 s later pelvis pitch 0.0°, CoM speed 0.015 m/s, feet moved 12 mm. Ground-truth state, no noise, one target only |
| 6 | Evaluation: speed-up vs baseline, success rate, model-error robustness | One table + one figure | **Done** 2026-11-17 (`scripts/evaluate.py`, `results/eval/`: `targets.json`, `model_errors.json`, `evaluation.png`). 20 targets (h 0.05–0.20 × l 0.2–0.6 m, offset 0.09): Stage 1 20/20; Stage 3 naive 10/20, from Stage 2 12/20, either 16/20; MuJoCo success (stands, touch-down within 10 cm) 14/16 → **end to end 14/20** (11/20 before the roll-IACVM fix of 2026-11-17, see Lessons). Stage 3 median 34.2 s naive vs 19.1 s from Stage 2; ×1.9 on 6 common successes. Touch-down short by 43 mm median. Fails: all of l = 0.6 except h 0.2 and (0.10, 0.5) in optimization; h 0.05 with l 0.4–0.5 in MuJoCo (l = 0.2 at h ≤ 0.15 also fell before the roll fix). Model errors on (0.10, 0.40): mass ±10 %, trunk +5 kg, motors −10/−20 % all stand (touch-down within 4 cm of nominal); friction 0.5 and 0.3 fall at landing |

| D | Baseline: Dai, Valenzuela, Tedrake 2014 (centroidal dynamics + full kinematics) on 3D H1-2, same grid and criterion as Phase 6 | Times, iterations, success rates vs [T] | **Done** 2026-11-17, see "Dai 2014 baseline" below. Nominal: **1051 iters, 131–136 s**, 6611 variables / 9794 rows. Grid: optimizes 20/20 (median 122.3 s, 976 iters) but every plan exceeds H1-2 torque limits (2.3–3.3×): end to end **0/20** on real H1-2, **11/20** with the motor limits removed (the paper's sufficient-torque assumption) |

Start with modest jump targets (0.3–0.5 m forward). H1-2 is about 67 kg (handless) with the same 360 N·m legs as H1, so the paper's 1.0 m is not a first target.

## What exists now (through Phase 3)

One target runs end to end in optimization: **h = 0.10 m, l = 0.40 m**. The full jump runs in MuJoCo: launch and flight playback, then the landing controller; the robot ends standing.

1. `scripts/srmp_numbers.py` turns the H1-2 model into SRMP parameters (CoM height, pitch inertia, CoP limits).
2. Stage 1 (`models/srmp.py`, `stages/stage1_srmp.py`): SRMP launch + flight in one NLP → references for CoM velocity, CAM rate, inertia, and the phase durations (t_sl 0.841 s, t_f 0.286 s). 90 iters, 0.54 s (with the 0.09 m landing offset).
3. Stage 3 (`models/symmetric.py`, `stages/stage3_nlp.py`): 7-DOF whole-body launch + flight in one NLP, from a naive guess → joint trajectory in `results/stage3_nlp.npz`. 603 iters, 26.9 s.

4. Execution (`sim/mujoco_runner.py`): joint PD + planned torque feedforward in the full MuJoCo model for launch and flight; from touch-down `sim/landing.py`, a wrapper around the sister project's landing controller. Plot and snapshots in `results/playback.png`.

The default pipeline is now Stage 1 → Stage 2 (`stages/stage2_qp.py`, 0.2 s) → Stage 3 warm-started (221 iters, 11.5 s): 12.3 s in total. The 603 iters / 26.9 s above is the naive-guess baseline (`--naive`). MuJoCo: lands at 0.364 m (target 0.400), impact 6.4 BW, STANDING.

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
| 16 | Landing (Eq. 56) | Foot − CoM offset mirrors the **initial** posture; no distance or foot-height condition | Foot lands at x = l, sole flat on the floor, offset set by #22 | (E) Eq. 56 contradicts Eq. 17; as written a 0.40 m target gives 0.23–0.31 m |
| 17 | Flight weights (Table IV) | ρ 0.1, ρ̇ 1e8, θ̈ 2e5, jerk 1, ankle 1e8 | ρ 0.1, ρ̇ 1e3, θ̈ 1 (joints only), jerk 1e-3, ankle 1e3 | (E) no convergence in 2000 iters with the paper's values on H1-2 |
| 18 | Take-off node constraints | Not discussed | Duplicated dynamics and inequalities on the zero wrench removed | (E) otherwise the constraints are degenerate and IPOPT does not converge |
| 19 | Scaling | Not discussed | Forces in body weights, objective × 1e-4 (Stage 1) and × 1e-6 (Stage 3) | (R) IPOPT needs it with weights up to 1e8. Variable scaling (`stage3_nlp.XSCALE`) is implemented but off: it is outside the paper |
| 20 | Stage 2 model (Sec. V) | QP on the full 3D model with a hip-symmetry constraint | QP on the 7-DOF symmetric model (the same thing with the symmetry built in), OSQP, 4 QP steps per Stage 3 node | (R) it is the model Stage 3 uses; `models/full.py` stays unused |
| 24 | Stage 2 tasks | CoM, CAM (launch); inertia and foot-to-CoM position from the SRMP (flight) | Same, plus: weak pelvis-upright task, joint-range and acceleration bounds, torque/friction/CoP limits; flight foot path is a quintic to the landing offset | (E) without the pelvis task the QP leans the trunk back 30°, straightens the leg and becomes infeasible before take-off |
| 25 | Using the guess in Stage 3 | "Initial guess", solver SNOPT | IPOPT with `mu_init 1e-3`, `bound_push/frac 1e-6` (`stage3_nlp.WARM`); falls back to the naive guess if it fails | (E) with IPOPT's defaults the Stage 2 guess is not better than the naive one (7/11 each, median 37.6 s vs 32.4 s) |
| 21 | Execution of launch + flight | Open-loop joint position playback | Joint PD + planned torque as feedforward | (E) position-only playback overshoots take-off vz by 28 % in MuJoCo (1.80 vs 1.40 m/s) |
| 22 | Landing posture | Foot ahead of the CoM by the initial offset (Eq. 56) or, per Eq. 17, the take-off offset | `LAND_OFFSET = 0.09 m`, used in both Stage 1 (distance and landing angle) and Stage 3 | (E) the take-off mirror (0.134 m) lands on the heels: feet rock back 13–15° in MuJoCo. Found by a sweep, not derived |
| 23 | Landing controller | [2]'s online WBC | Sister project's re-implementation of [2], CoM PD in x softened from (60, 18) to (20, 25) | (R) [2] is a vertical jump; with forward speed the stiff pull drives the CoP to the heel |

Used as printed: Table I except W_t; Table III (launch, minus the heel term); jerk as the input in both stages; the hard CAM constraint (Eq. 47); soft tracking of CoM velocity and inertia; hard take-off velocity.

Findings about the paper so far (candidates for the report):

- Stage 1's two NLPs cannot be solved one after the other as described (#6).
- The launch duration is not determined by the published cost (#7).
- The inertia reference is not followed by the whole body: Stage 3 tracks L exactly but ρ is 11.1 vs 7.0 at take-off (arms swing up). The backward L that Stage 1 asks for comes from the SRMP's massless leg; a real robot can also swing its legs forward.
- Eqs. 17 and 56 disagree on the landing posture (#16).
- Stage 2 helps less than the paper implies: ×1.8 median on Stage 3 and no gain in success rate (Phase 4). The three-step structure alone does not make the whole-body NLP reliable. (Side check outside the paper: with variable scaling Stage 3 converges in about 18 iters from either guess, so the conditioning of the NLP matters more than the guess.)
- The paper reports no success rate. Ours, as faithful as we could make it: 14/20 targets end to end (16/20 optimize, 14/16 of those land and stand in MuJoCo; 11/16 before the roll-IACVM fix). Open-loop launch/flight (with PD + feedforward) is robust to ±10 % mass and −20 % motor strength; the landing is the weak part (friction ≤ 0.5).
- The "under 10 s" claim has no baseline. Ours from a naive guess: 26.9 s (603 iters; 17–62 s over nearby landing offsets) on a newer CPU than the paper's i5-9400F, so hardware does not explain the gap; the initial guess, the solver and the extra limits (#13) do. With the Stage 2 guess: 12.3 s for all three stages.

## Dai 2014 baseline

Paper: Dai, Valenzuela, Tedrake, "Whole-body motion planning with centroidal dynamics and full kinematics", Humanoids 2014 ([D]). [T] cites it as its ref. [17]; [T] Stage 3 is a planar, single-contact-mode version of it (Eq. 49 ≡ centroidal dynamics, Eq. 47 ≡ [D] Eq. 1b with ḣ fixed). Paper-level comparison: `docs/compare_dai2014.md` (§12 has the implementation results).

Code: `models/full.py` (Pinocchio, 27 joints, root PX PY PZ RZ RY RX so q̇ = v; `kin` → CoM, CAM, 8 sole corners; `joint_tau` → inverse dynamics; self-check `python -m humanoid_jump.models.full`), `dai2014.py` (`solve(h, l, torque_limits=False)`; `python -m humanoid_jump.dai2014 [h l] [--torque-limits]` → `results/dai2014{_tau}.npz/.png`), `sim/play_full.py` (PD + feedforward on all 27 joints; `--landing`, `--unlimited`, `--view`), `scripts/eval_dai2014.py` (grid, about 1 h; `--unlimited` re-simulates with motor limits ×100, about 15 min; `--report`). Results: `results/eval_dai2014/targets.json`, logs `results/eval_dai2014*.log`, video `results/dai2014_unlimited.mp4`.

Formulation: Eq. 6 variables (q, v, r, ṙ, r̈, c, F, h, ḣ, dt), Eqs. 7a–7k, backward Euler / trapezoid as in Eqs. 7d–7g, free dt per interval. 5 phases with a fixed schedule: stance (20 knots), toes (6), flight (12), toes touch (4), full sole (12, ends at rest).

| # | Paper [D] | Ours | Why |
|---|---|---|---|
| D1 | SNOPT | IPOPT (MUMPS), CasADi Opti, `expand=True` | (R) SNOPT is not in this CasADi build |
| D2 | Atlas, box jump | H1-2 3D, our targets (h, l) on the Phase 6 grid; take-off vz = √(2gh), toes land at x = l | (G) |
| D3 | 4 corners per foot, contact sequence by hand | Same; schedule heels off → toes off → toes touch → heels touch; friction pyramid μ 0.7; no complementarity (Eqs. 8–9) | (G) the paper's own choice for the jump |
| D4 | Collision avoidance with GJK | Corner z ≥ 0 only | (G) no obstacles in our task |
| D5 | Cost weights, dt bounds not given | W: q 1, rpy 10, v 1e-2, r̈ 1e-2, F 1; dt in 0.25–2.5 × nominal; pelvis pitch ≤ 1.2 rad | (G) |
| D6 | Initial guess not given | Hold the crouch, Fz shared equally, nominal dt | (G) same as the [T] naive baseline |
| D7 | Sufficient torque assumed (Sec. II-A), no torque limits | Default: none. `torque_limits=True` adds τ = RNEA − ΣJᵀF within H1-2 limits | (R) toggle for comparison only |
| D8 | Execution: QP controller in Drake | PD + feedforward (#21), torque from inverse dynamics after the solve; evaluated with real limits and with motor limits ×100 (the paper's assumption on the robot side) | (R) |

Findings:

- Nominal (0.10, 0.40): [D] 1051 iters, 131–136 s vs [T] 12.3 s (pipeline) or 26.9 s (naive Stage 3). Per iteration about 0.13 s vs 45 ms; iterations 1.7× the [T] naive baseline from the same kind of guess.
- Grid: [D] optimizes 20/20 ([T] 16/20; [D] also solves l 0.6). Median 122.3 s (24.5–177.9 s, 976 iters) vs [T] 22.1 s. End to end: real H1-2 0/20, limits removed 11/20 (same with the landing controller), [T] 11/20 (14/20 after the roll-IACVM fix of 2026-11-17; the [D] runs with the landing controller predate it).
- High torque comes from (1) no torque limit in the NLP, (2) the toe-only phase: ankle τ ≈ toe force × 0.174 m (≈ 145 N·m per side vs limit 60), (3) free dt rewarding a short, hard push (toe phase 31 ms at 2.3–2.6 BW). The first-order transcription adds spikes at phase transitions (knee 272 rad/s²); [T]'s jerk input avoids those, but they are not the plateau.
- With the torque toggle on: 487 iters, 153.7 s, plan within limits, but the ankle sits exactly at its limit for the whole 0.25 s toe phase; MuJoCo saturates in 44 % of the launch and falls. A finer grid (diagnostic) does not help: zero headroom in toe stance, not discretization.
- Even with unlimited torque the robot lands short (median 96 mm, up to 175 mm at h 0.20): simulated take-off velocity 0.07–0.23 m/s below the plan. Hypothesis (untested): transcription error with dt up to 104 ms and base-row residuals up to 1142 N at phase transitions.

Lessons:

- **MuJoCo H1-2 limits torque twice**: `actuator_ctrlrange` and the joint `actuatorfrcrange` (`m.jnt_actfrcrange`). Scaling only the first does nothing.
- Pinocchio vs MuJoCo bias forces: compare `pin.rnea(q, v, 0)` with `qfrc_bias`, not gravity only.
- Free dt with an unused dt[0] leaves knot-0 variables undetermined: use N−1 intervals and constrain the start (r̈[0] = 0).

Other jumps with [D] (2026-11-17, motor limits removed in MuJoCo, planned landing; [T] cannot do the diagonal one):

- **Diagonal** (`python -m humanoid_jump.dai2014 0.10 0.25 0.25`; `y_jump` = sideways toe target): 1559 iters, 201.3 s. Take-off v (0.44, 0.47, 1.40). MuJoCo lands at (0.18, 0.20) m vs (0.25, 0.25) and stands. Plan torque: shoulder pitch 3.5×, ankle pitch 3.1× the limit, 14 joints over. Videos `results/dai2014_diagonal.mp4`, `results/dai2014_diagonal_top.mp4`.
- **Onto a box** (`python -m humanoid_jump.dai2014 0.25 0.55 --box`; `BOX = (0.20 m high, front edge x 0.25)`, `build_model(box=...)`). This is the paper's own demo task (Atlas onto a box). Contacts after touch-down are at z = box height. Corners off the ground clear the edge with a smooth step, z ≥ (H + clear)·½(1 + tanh((x − edge + margin)/0.01)), in place of GJK (D4). The landing knots of the naive guess are the crouch moved onto the box. Result: 1751 iters, 242.2 s, take-off vz 2.22 m/s, peak 3.35 BW, ankle 3.2× the limit. MuJoCo lands at ankle x 0.378 (plan 0.55) and **stands on the box** (pelvis 0.97 m). Only about 4 cm of heel margin is left. Video `results/dai2014_box.mp4`.
  - Files of that run: `results/dai2014_box_0.20.*`. `BOX` is now (0.10, 0.35), the same box as the [T] box extension. **Same task as [T] (box 0.10, h 0.20, l 0.50)**: 1275 iters, 176.7 s (vs [T] 24.6 s). Ankle 3.0× the limit. MuJoCo lands at ankle x 0.349 (plan 0.50), 15 cm short, so the heel (0.26) misses the edge (0.35) and the robot **falls back**. [T] lands 7 cm short and stands. Now `results/dai2014_box_0.10_e0.35_fell.*`.
  - **`BOX` = (0.10, 0.25)**, h 0.20, l 0.50. `BOX_MARGIN` is now 0.04: with 0.08 the toe, which starts 7.6 cm from the edge, would have to be up right after take-off. Result: 1280 iters, 168.8 s. MuJoCo lands at ankle x 0.349, the same 15 cm short, and **stands on the box** (pelvis 0.86 m). The heel lands only 1.3 cm inside the edge. `results/dai2014_box.*`.
  - Two failures first, kept in `results/dai2014_box_l0.40_fell.*` and `results/dai2014_box_l0.55_clipped.*`. (1) l 0.40: lands 13.5 cm short, the heel misses the edge, and the robot falls back. (2) l 0.55 with margin 2 cm and clearance 0: the toes clip the box edge in flight. Now margin 8 cm and clearance 5 cm in flight (`BOX_MARGIN`, `BOX_CLEAR`). Both failures come from the executed jump being lower and shorter than planned (see the landing-short finding above). The plan is tight against the box only on paper.

## Extension: box landing with [T] (NOT in the paper)

The paper covers flat ground only. The user asked for a jump onto a box (2026-11-17), height 0.10 m. Off unless asked: `stage1_srmp.launch(..., box=(height, edge x))` sets it, Stage 3 reads `ref["box"]`, and the flat path is unchanged (re-checked: 221 iters, MuJoCo 0.364 m, Phase 5 gate passes).

- Changes:
  - Stage 1: t_f = (vz + √(vz² − 2gH)) / g instead of 2vz/g, because the CoM lands H higher.
  - Stage 3: the landing sole z = H.
  - Stage 3: the toe and heel clear the edge, z ≥ H·½(1 + tanh((x − edge + 0.02) / 0.01)). This is the same smooth step as `dai2014.py`.
  - Stage 2: unchanged. Its foot task is relative to the CoM, which rises by H too.
  - MuJoCo: `build_model(box=...)`, and the landing controller targets `z_land` 0.78 + H.
- Commands: `python -m humanoid_jump.stages.stage3_nlp 0.2 0.5 --box` (→ `results/stage3_box.npz/.png`), then `python scripts/render_video.py box` (→ `results/video_box.mp4`).
- Result, h 0.20, l 0.50, `BOX = (0.10, 0.35)` (the edge is 6 cm behind the planned heel):
  - Optimization: Stage 1 0.93 s (145 iters), Stage 2 0.24 s, Stage 3 from the Stage 2 guess 23.5 s (445 iters, acceptable level); **24.6 s total**.
  - MuJoCo: touch-down at ankle x 0.431 m (plan 0.500), impact 5.6 BW, **STANDING on the box** (pelvis z 0.95 m).
  - Caveat: touching down 7 cm short puts the heel on the edge (heel x ≈ 0.345 vs edge 0.35).
- Distance matters more than the edge position:
  - With h 0.2, l 0.4 (edge 0.25) is infeasible. The toe starts at x 0.174 and passes the edge after only 6 cm of travel, before the foot can rise 0.1 m.
  - l 0.6 (edge 0.45) hits 1500 iters, from either guess.

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

`collocation.py` has `trapezoid` (defect rows), `trapz_cost`, `solve`. `models/srmp.py` (Lagrangian SRMP, self-check: `python -m humanoid_jump.models.srmp`) and `stages/stage1_srmp.py` (`launch(h, l, joint=...)`, `flight(la)`; run `python -m humanoid_jump.stages.stage1_srmp [h l]`, plot in `results/stage1_srmp.png`) are done. `models/symmetric.py` (7-DOF planar model from the URDF via Pinocchio, planar root + left = right matrix `S`; self-check against MuJoCo: `python -m humanoid_jump.models.symmetric`) and `stages/stage3_nlp.py` (`solve(ref, hard_cam=..., guess=...)`; run `python -m humanoid_jump.stages.stage3_nlp [h l]`; plot `results/stage3_nlp.png`, trajectory `results/stage3_nlp.npz`) are done. `sim/mujoco_runner.py` executes `results/stage3_nlp.npz` (`python -m humanoid_jump.sim.mujoco_runner [--no-ff] [--no-landing]`, ends with the Phase 5 gate; viewer: `mjpython -m humanoid_jump.sim.mujoco_runner --view`). `stages/stage2_qp.py` (`solve(ref)` → Stage 3 guess; `python -m humanoid_jump.stages.stage2_qp` compares both guesses) is done; `python -m humanoid_jump.stages.stage3_nlp` now runs Stage 1 → 2 → 3 (`--naive` for the baseline). `scripts/bench_warmstart.py` is the Phase 4 benchmark (about 30 min; `--table` reprints). `scripts/evaluate.py` is Phase 6 (about 40 min, resumes; `--resim` reruns only MuJoCo from the saved trajectories, `--report` reprints table + figure). `models/full.py` is the 3D H1-2 model used only by the Dai 2014 baseline (see below).

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
- **Landing offset decides whether the landing is clean.** Foot ahead of the CoM at touch-down, with the soft x gains: 0.134 m (take-off mirror) stands but the feet rock back 13–15° and slide 25–50 mm; 0.12 and 0.11 rock 7°; 0.10 and 0.09 stay flat (±1°, slide about 10 mm). Reason: the CoM lands behind the heel, so the vertical impact adds backward angular momentum and brakes the forward speed; the capture point ends up at the heel. Stage 1's backward L (−6) makes it worse.
- **Landing CoM gains in x: (20, 25), not the sister project's (60, 18).** With (60, 18) the mirrored landing falls (pelvis −42°, impact 12.5 BW). kd_x = 40 also falls. A reference shifted back by 4–8 cm has a similar effect to the softer gain.
- The sister project's code targets MuJoCo 4 (`mj_fullM(m, d, M)`); this env has 3.9. `sim/landing.py` patches the call at runtime only; the sister project is untouched.
- Stage 3's convergence is fragile in the landing offset (358 to 1442 iters, failures at 0.07 and 0.048). Judge any change on reliability over several targets, not only on speed.
- **Phase 4 benchmark, 11 targets, max 1500 iters.** Success: naive/default 7, naive/warm-opts 3, stage2/default 7, stage2/warm-opts 7. Median over successes: 700 it / 32.4 s, 1222 / 54.4, 821 / 37.6, 513 / 23.9. Either Stage 2 variant succeeds on 9/11, either naive variant on 7/11. Nothing converges for (0.10, 0.40, offset 0.07). The Stage 2 solutions sit at a slightly higher cost (about 32.6 vs 31.3): a different local optimum (pelvis +12° at landing instead of −4°), which also lands and stands in MuJoCo.
- **A warm start needs warm-start solver options.** IPOPT's defaults move the start point off the bounds and start with barrier 0.1; with them the Stage 2 guess fails on the main target (Restoration_Failed, 1309 iters). The same options hurt the naive guess (3/11). One earlier run that showed "naive + mu_init 1e-3 = 359 iters" did not reproduce in the benchmark (1222): do not quote single runs.
- **Stage 3's reliability problem is not the initial guess; it is conditioning.** 4 of 11 targets fail from either guess. IPOPT log: Hessian regularization lg(rg) ≈ −2 at every iteration and slow progress with full steps; the objective curvatures span 2e-7 (ddq, jerk) to 10 (v) after × 1e-6. Variable scaling fixes it (11/11, about 18 iters, `results/warmstart.json` "scaled" columns), but it is outside the paper, so it stays off (`xscale=None`). In the paper-faithful pipeline, expect failures on some targets.
- **After changing `body_mass` at runtime, call `mujoco.mj_setConst(m, d)`.** Otherwise `body_subtreemass` stays nominal and `subtree_com` / `subtree_linvel` are scaled wrong (mass +10 % read as take-off vz 1.62 instead of 1.48 m/s).
- **"Standing" needs more than height, pitch and one speed sample.** Phase 6 found a stumble that ended at standing height with the ankle 0.6 m ahead of the CoM (h 0.05, l 0.4), and upright landings with a residual sway (single samples up to 0.09 m/s, feet still) counted as falls. Criterion now: pelvis z > 0.6, |pitch| < 0.3, mean CoM speed over the last 0.5 s < 0.05, |CoM x − ankle x| < 0.1; Phase 6 also holds 3 s and requires touch-down within 10 cm of the target.
- **The "residual sway" was an ankle-roll limit cycle in the landing controller. Fixed 2026-11-17** (found while checking the box video). After harder landings (flat h 0.2 l 0.4/0.5, h 0.15 l 0.5, box h 0.2 l 0.5) both feet keep rocking in roll, ±12° std and up to ±20°: ankle roll τ ±20 of 40 N·m, q̇ std 6.5 rad/s. Sole Fz swings 0.5–2.3 BW at about 9 Hz and never decays, while the foot positions and the CoM stay in place. After h 0.1 l 0.4 everything settles in 0.6 s.
  - Not the cause: the box contact (a 2 mm box under the h 0.1 landing stays calm), pitch IACVM and the posture gain.
  - Cause: foot roll was never stabilized. Like the sister project (`--iacvm_pitch_only 1`), the wrapper applied IACVM to pitch only.
  - Fix in `sim/landing.py`: IACVM on roll too, `iacvm_roll = -0.7`, i.e. the opposite sign of pitch. The sign and the gain come from a sweep, not a derivation:
    - Gains from -0.3 to -1 all remove the rocking.
    - -0.7 is the only one that loses no case that stood before.
    - +1 leaves the feet rolled 20°, and pitch at -1 falls everywhere.
  - Phase 6 re-simulated: 14/20 (was 11/20). The three l = 0.2 targets at h ≤ 0.15 now stand; they had been ending on rolled feet. Model errors unchanged.
  - Box: CoM speed 0.002 m/s after 3 s, feet flat.
  - Marginal landings (h 0.2 l 0.6, trunk +5 kg) flip between standing and falling with small gain changes.
- Phase 6 failure modes: short jumps (l = 0.2) fell before the roll fix; they had been blamed on the fixed 0.09 m landing offset (tuned at l 0.4). h = 0.05 with l ≥ 0.4 touches down about 0.1 s early (the swing foot scuffs, flight is only 0.20 s). In the model-error runs the landing controller sees the perturbed MuJoCo model (best case).
- Do not switch on non-paper methods by default even when they work much better (2026-11-17: variable scaling was turned on by default and had to be reverted).
- OSQP returns garbage in `x` (about 2e9) on an infeasible QP instead of None: check `res.info.status`.
- Comparing Pinocchio and MuJoCo: map joints by name, not by index; keep base velocity at zero when comparing CAM, to avoid frame-convention mismatches.

## How to work with me

- **Follow the paper; I am studying it.** Do not add techniques from outside the paper to make things faster or more reliable (e.g. variable scaling, solver tricks, extra tasks). Deviate only where the paper's formulation fails or does not fit H1-2, as the (E)/(R) rows of the differences table do, and ask before adding a new one. A non-paper idea that works may be recorded as a finding, but stays off the default pipeline.
- Big picture first: structure and relationships before details. Physical intuition before formalism. Keep explanations short; don't dump everything at once.
- Verify with code: run things and report numbers, not just claims. I watch for detail-level errors, and I appreciate it when you check calculations and units.
- When reading papers, be critical: point out unstated assumptions and weak evidence, and cite equation numbers.
- Prefer tools and automation over manual, repetitive checking.
- I'm not a native English speaker. When I write in English, briefly suggest a more natural phrasing before answering. Answer in English unless I ask for Korean.
