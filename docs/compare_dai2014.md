# Dai 2014 vs. the T-ASE three-step framework (our H1-2 re-implementation)

Compares two papers with our code:

- **[D]** H. Dai, A. Valenzuela, R. Tedrake, "Whole-body motion planning with centroidal dynamics and full kinematics", IEEE-RAS Humanoids 2014, pp. 295–302 (`../../Papers/Whole-body_motion_planning_with_centroidal_dynamics_and_full_kinematics.pdf`).
- **[T]** H. Qi et al., "A Three-Step Optimization Framework With Hybrid Models for a Humanoid Robot's Jump Motion", IEEE T-ASE 22, 2025, pp. 14120–14132 (`../../Papers/A_Three-Step_...pdf`).
- **Ours**: the [T] re-implementation on Unitree H1-2 in MuJoCo (`humanoid_jump/stages/`, results in `CLAUDE.md` and `results/`). "#n" refers to row n of the Differences table in `CLAUDE.md`.

Equation numbers are the papers' own. All our numbers come from `CLAUDE.md`, `results/eval/targets.json` or the code. The [D]-on-H1-2 problem sizes in §4 are estimates (marked as such); nobody has run them.

Purpose: groundwork for implementing [D] on H1-2 next (§11).

---

## 1. Summary

| | [D] Dai 2014 | [T] T-ASE 2025 | Ours ([T] on H1-2) |
|---|---|---|---|
| Idea | One NLP: centroidal dynamics (Eq. 1) + full kinematics (Eqs. 7h–7j) | Three models in a chain: SRMP NLP → QP mapping → 5-link NLP | Same chain; Stage 3 on a 7-DOF symmetric model built from the full URDF |
| Dynamics in the NLP | 6 Newton–Euler equations of the whole robot (Eqs. 7a–7c); joint dynamics ignored | Stage 3: the 3 unactuated rows of the planar EoM (Eq. 49) | Stage 3: all 7 rows (3 as equalities, 4 as torque limits, #13), so full dynamics |
| Kinematics | Full 3D Atlas, all joints | Planar, 1 leg + 1 arm, parameters lumped by hand (Sec. III-C) | Planar, from the URDF, left = right by a matrix (#10) |
| Contacts | Points (4 corners per foot), scheduled or found by complementarity (Eqs. 8–9) | Toe fixed, heel may lift (Eq. 50); wrench (Fx, Fz, τy) at the toe (Eq. 48) | Whole sole fixed, CoP inside the sole (#12) |
| Phases in the plan | Push-off and landing (Sec. III-A) | Launch + flight; landing is handled online by [2] | Launch + flight; landing by our WBC port of [2] (#23) |
| Time | Free dt per interval (Eq. 6) | Stage 1 free t_sl; Stage 3 durations taken from Stage 1 | Same as [T] |
| Integration | Backward Euler for q, h (Eqs. 7d, 7e); trapezoid-like for r (Eqs. 7f, 7g) | GPOPS-II (pseudospectral), jerk input | Trapezoid on (q, q̇, q̈), jerk input |
| Initial guess | Not described; landing seeded with the fixed-mode solution (Sec. III-C) | Stage 2 QP output (Sec. V) | Stage 2 QP output, IPOPT warm options (#25), naive fallback |
| Solver | SNOPT, MATLAB | GPOPS-II + SNOPT, MATLAB | CasADi + IPOPT, Python (#1) |
| Run time | "several minutes", up to "several hours" (Sec. IV) | "less than 10 seconds", i5-9400F (Sec. VII-A) | 12.3 s (0.58 + 0.21 + 11.5 s, 221 iters) at h 0.10 / l 0.40; naive 26.9 s (603 iters) |
| Problem size | Not reported | "average 1500 nonlinear constraints" | Stage 3: 2139 variables, 2809 rows |
| Validation | Drake simulation with a QP tracking controller [D15]; no hardware | CoppeliaSim + hardware: 1.0 m forward, 0.5 m onto a platform | MuJoCo, 20 targets: 11/20 end to end |
| Success rate | Not reported | Not reported | 16/20 optimize, 11/16 land and stand |

The main point: **[T]'s Stage 3 is already a [D]-type formulation**, restricted to the sagittal plane, one contact mode and a CAM reference (§2). [T] does not say this. It even cites [D] (its ref. [17]) as a full-body-model method that carries "a significant computational burden" (Sec. II-A). That is wrong: avoiding full-body dynamics is the whole point of [D].

---

## 2. Dynamics model

**[D].** The robot's dynamics are reduced to the 6 rows that relate external wrenches to momentum:

- Eq. 1a: m r̈ = Σⱼ Fⱼ + m g
- Eq. 1b: ḣ(q, v) = Σⱼ (cⱼ − r) × Fⱼ + τⱼ
- Eq. 2: h = A(q) v, where A is the centroidal momentum matrix [D19]

Full kinematics enters through r = com(q) (Eq. 7h), cⱼ = pⱼ(q) (Eq. 7i) and Eq. 7c. The joint rows of the EoM are dropped. Sec. II-A argues that this is *necessary and sufficient* as long as every joint is actuated and torque limits are ignored, and that torques can be recovered afterwards by inverse dynamics.

**[T].** Each stage uses a different model:

- Stage 1 uses the SRMP Lagrangian (Eq. 2, 4 DOF, ρ = 2 m_p ϕ², Eq. 4).
- Stage 2 uses the 20-DOF 3D model (Eq. 6) in a QP (Eqs. 32, 38).
- Stage 3 keeps "the top three rows" of the planar EoM: M̃ θ̈ + H̃ = J̃ᵀ F_ext (Eq. 49).

Those three unactuated rows *are* the planar Newton–Euler equations of the whole body (x, z and pitch), so Eq. 49 holds the same information as Eqs. 1a and 1b of [D] in the sagittal plane. Eq. 47 makes the match explicit: τ_y = L̇^ref − (F_x ℓ_rcz − F_z ℓ_rcx) is the y-row of [D] Eq. 1b with ḣ fixed to the SRMP reference. So [T] Stage 3 is "centroidal dynamics + (planar) full kinematics + fixed CAM rate".

**Ours.** `stage3_nlp.py` computes the full inverse dynamics τ = M q̈ + h for all 7 rows (`q_of`):

- rows 0–2 are equalities ("launch base dynamics", Eq. 49);
- rows 3–6 are joint torques bounded by TAU_MAX (#13).

Our Stage 3 is therefore a full-dynamics NLP, one step further than both papers. The bounds are needed on H1-2: in the nominal Stage 3 solution the shoulder hits its limit (80/80 N·m L+R) and the ankle uses half of its 120 N·m (CLAUDE.md Phase 3). Under [D]'s sufficiency argument these limits would not be visible in the plan.

**Consequence for [D] on H1-2.** "Sufficient torque" (Sec. II-A) is the assumption most likely to fail. The ankle pitch is ±60 N·m per side and the shoulder ±40 N·m per side. A faithful [D] plan must be checked by inverse dynamics afterwards. Adding torque bounds brings M(q) back into the NLP, which is exactly what [D] tries to avoid. It is the same deviation as our #13.

---

## 3. Decision variables

| [D] Eq. 6 (per knot k) | [T] Stage 3 (Eqs. 42, 43, 52) | Ours (`stage3_nlp.py`) |
|---|---|---|
| q[k] ∈ R^{n+6}, v[k] ∈ R^{n+6} | θ, θ̇, θ̈ of the 5-link model (Eq. 42) | X: q, q̇, q̈ ∈ R^{3·7} |
| r, ṙ, r̈ ∈ R³ (separate from q) | — (CoM is a function of θ) | — (`q_of`) |
| cⱼ, Fⱼ, τⱼ per contact point | F_x, F_z (Eq. 43); τ_y from Eq. 47 | Λ = (F_x, F_z, τ_y) in body weights |
| h, ḣ ∈ R³ | — (L̇ = L̇^ref through Eq. 47) | — (L̇ = L̇^ref as a hard row, Eq. 47) |
| dt[k] (free) | — (fixed per phase) | — (T, t_f from Stage 1) |
| (no input: first order, v is the derivative) | jerk input | U = jerk (launch and flight) |

[D] carries the centroidal quantities r, h and their derivatives as separate variables. They are tied to q only at the knots (Eqs. 7c, 7h) and integrated by their own rules (7e–7g). [T] and ours compute them from q, so they cannot drift apart. In [D], com(q(t)) and the integrated r(t) may differ between knots. The paper gives no bound on that error (§9.2).

---

## 4. Problem size

| | Variables | Constraint rows | Grid |
|---|---|---|---|
| [D] (Atlas examples) | not reported | not reported | N not reported |
| [T] (whole process) | not reported | "average 1500 nonlinear" (Sec. VII-A) | GPOPS-II mesh, not reported |
| Ours Stage 1 | 812 (launch 12 × 41 + 4 × 41 + 1, flight 5 × 31) | — | N_L 40, N_F 30 |
| Ours Stage 2 | ≈ 280 QPs of ≤ 10 variables | — | 4 per Stage 3 node |
| Ours Stage 3 | 2139 (41 × 31 + 31 × 28) | 2809 | 41 + 31 nodes |
| *Estimate*: [D] on the same 7-DOF planar model | ≈ 2.2k (31 per knot: q 7, v 7, r/ṙ/r̈ 6, toe + heel cⱼ, Fⱼ 8, h/ḣ 2, dt 1) | same order | 71 knots |
| *Estimate*: [D] on full 3D H1-2 (27 joints) | ≈ 9.3k (131 per knot: q 34, v 33, r/ṙ/r̈ 9, 8 corners × (c, F) 48, h/ḣ 6, dt 1) | same order | 71 knots |

On the planar model [D] is about as large as our Stage 3. [D] trades the 3 q̈ rows per node for r, h variables plus their integration rows. "Fewer variables" (Sec. II-A, n + 6 → 6 dynamic constraints) holds against a full-dynamics *transcription with torques as variables*. It does not hold against our Stage 3, which never makes torques variables.

---

## 5. Constraints

| Role | [D] | [T] Stage 3 | Ours Stage 3 |
|---|---|---|---|
| Momentum dynamics | Eqs. 7a, 7b, 7c | Eq. 49 (3 rows) | Eq. 49 (3 rows) |
| Joint torque limits | none (Sec. II-A) | "boundary constraints", no values | TAU_MAX on rows 3–6 (#13) |
| Integration | Eqs. 7d–7g (backward Euler, trapezoid on r) | GPOPS-II collocation | trapezoid defects |
| CAM | free; h integrated from contact torques | L̇ = L̇^ref (Eq. 47) | same, hard (Eq. 47) |
| Contact kinematics | cⱼ = pⱼ(q), cⱼ ∈ Sⱼ (Eqs. 7i, 7j) | toe velocity 0, heel ż ≥ 0 (Eq. 50) | sole position fixed (#12) |
| Contact wrench | Fⱼ, τⱼ ∈ wⱼ (Eq. 7k: friction cone, bounds) | not stated | Fz ≥ 0, μ 0.7, CoP in sole |
| Joint ranges, velocities | q ∈ Q, v ∈ V (Eq. 7k) | "boundary constraints" | joint ranges, pelvis ±1 rad |
| Collisions | Eqs. 3–5 (GJK, smooth hinge sum = 0) | none | foot above floor in flight |
| Take-off | none as such: emerges from the contact schedule | CoM velocity (Eq. 51) | Eq. 51 + wrench = 0 (#15) |
| Landing | contact schedule (toes then heels) | joints at rest + mirrored offset (Eq. 56) | joints at rest, foot at x = l, offset 0.09 m (#16, #22) |
| Unscheduled contacts | complementarity (Eqs. 8a–8c, 9a–9b) | — | — |

---

## 6. Contact handling

- **[D]:** contacts are points. The flat-ground jump uses the 4 corners of each foot with the fixed sequence heels off → toes off → toes touch → heels touch (Sec. III-A). Heel lift is free in this model: once the heel points are released, the wrench at the toe points is a sum of point forces, so it is physically valid by construction. Sec. II-E optionally drops the schedule and uses complementarity: φ·Fₙ = 0 (Eq. 8a), |τ|²·φ = 0 (8b), no sliding (9a, 9b). The objective then loses the wrench term (Eq. 10), because an L2 penalty spreads force over all candidate points.
- **[T]:** one contact mode in launch: toe fixed, heel may rise (Eq. 50), with a pitch torque τ_y applied at the toe (Eq. 48). We found this inconsistent (#12): a toe line contact cannot transmit that torque once the heel is up.
- **Ours:** flat sole for the whole launch, CoP inside the sole. Consistent, but no heel-lift push-off (CLAUDE.md lesson "flat foot during launch").

**What [D] offers here:** corner points with a schedule give the heel-lift push-off that [T] wanted and we dropped, without the inconsistency. Of everything in [D], this is the most concrete fix for one of our deviations (#12, #14).

**Landing:** [D] puts touch-down inside the plan. [T] stops at touch-down and hands over to an online controller. Our Phase 6 shows that the hand-over is the weak link (11 of 16 optimized jumps land and stand; friction ≤ 0.5 falls). [D] does not model the impact itself either. Momentum stays continuous across touch-down (Eqs. 7a–7g allow only finite forces), so a planned landing is a soft landing spread over one dt, and the paper never compares it with a real impact.

---

## 7. Initial guess and warm start

- **[D]:** no guess is described for the main formulation. Sec. IV: "can take as long as several hours in some poorly initialized cases". The only seeding: the complementarity landing starts from the fixed-mode solution (Sec. III-C), a continuation in all but name.
- **[T]:** the guess is the reason Stages 1 and 2 exist (Sec. V: "the quality of the guess significantly impacts the speed and effectiveness"). No ablation is shown.
- **Ours:** the ablation [T] did not run:
  - Phase 4 (11 targets): Stage 2 guess vs naive, success 7/11 vs 7/11, median Stage 3 23.9 s vs 32.4 s.
  - Phase 6 (20 targets): 12/20 vs 10/20; median Stage 3 19.1 s vs 34.2 s; ×1.9 on 6 common successes.
  - The guess only helps with warm-start IPOPT options (#25).
  - Side finding outside the paper: variable scaling converges 11/11 in about 18 iterations from *either* guess. Conditioning matters more than the guess.

Both papers leave the guess question open. [D] admits that a poor guess costs hours. [T] builds a pipeline around the guess without showing that it pays off. Our data say that it pays off in time (about ×2) but not in reliability.

---

## 8. Solver and run time

| | Solver | Reported time | What is missing |
|---|---|---|---|
| [D] | SNOPT, "un-optimized MATLAB" | "within several minutes", "as long as several hours" (Sec. IV); a kinematics-only C++ planner (no dynamics) "less than 0.2 seconds" | per-example times, N, iterations, hardware |
| [T] | GPOPS-II + SNOPT, MATLAB | "less than 10 seconds", i5-9400F 2.9 GHz (Sec. VII-A) | per-stage times, iterations, baseline without Stages 1–2 |
| Ours | IPOPT (CasADi 3.7.2, `expand=True`, about 45 ms/iter) | nominal 12.3 s = 0.58 + 0.21 + 11.5 s (221 iters); naive 26.9 s (603 iters); Phase 6 median 22.1 s over 16 optimized targets | — |

About the 22.1 s: it is the median over the 16 optimized targets of Stage 1 + Stage 2 + the Stage 3 solve that was used. If a failed Stage 3 attempt from the Stage 2 guess is also counted before the naive fallback (as the pipeline actually runs), the median is 24.1 s (recomputed from `results/eval/targets.json`).

The three numbers cannot be compared directly:

- the solvers differ (SNOPT's SQP vs IPOPT's interior point; #1);
- the models differ (3D Atlas vs planar);
- [D] reports a range of minutes to hours over very different tasks;
- [T] gives one "< 10 s" figure with no baseline.

[D]'s claim that its full-body-dynamics comparison planner failed "even after hours or days" (Sec. IV) is anecdotal. [D] itself says "more work is required".

Practical note: CasADi in `mujoco_env` has no SNOPT (`casadi.has_nlpsol('snopt')` → False). A [D] implementation here would use IPOPT, the same deviation as #1.

---

## 9. Critique of [D]

1. **The sufficiency argument (Sec. II-A) ignores torque limits.** It is stated as an assumption, but it decides feasibility on H1-2 (§2). There is no torque check and no post-hoc inverse-dynamics result in the paper.
2. **Low-order, mixed integration.** q and h use backward Euler (Eqs. 7d, 7e), first order. r uses a trapezoid-like rule (Eqs. 7f, 7g). The links r = com(q) and h = A(q) v hold only at the knots. The paper gives no integration-error analysis. Our cart-pole lesson (trapezoid, O(h²), 89° → 1.4° end error for N 20 → 160 open-loop) suggests that a first-order scheme is worse still for open-loop playback. [D] hides this by running the plan through a feedback QP controller [D15].
3. **Free dt without stated bounds.** The typesetting of Eq. 6 leaves it unclear whether dt[k] multiplies the posture, velocity and acceleration terms or only the wrench term; Eq. 10 multiplies everything. Either way, the duration is set by weights the paper does not give. We met the same pathology in [T] Stage 1 (#7: node-sum costs leave t_sl undetermined).
4. **Degenerate constraints.** Γ(q) = Σ γ(c d̄ᵢⱼ) = 0 (Eq. 5) with Γ ≥ 0 has zero gradient at every feasible point. LICQ fails wherever the constraint is satisfied. The complementarity rows (Eqs. 8, 9) are MPCC constraints, where MFCQ fails at every feasible point. No relaxation or smoothing is used, and no convergence statistics are reported. Our #18 (degenerate take-off rows stopped IPOPT: 3000 iters → 601 once removed) predicts that IPOPT will struggle with Eqs. 8–9 as written.
5. **Weak evaluation.**
   - The results are snapshots and a video, with one quantitative figure (Fig. 7, CoM height and vertical GRF of the run).
   - No N, iteration counts or per-task times.
   - No constraint violation, tracking error or success rate.
   - The complementarity result is judged "to our eyes" (Sec. III-C).
6. **Unclear contact torque model.** τⱼ appears in Eqs. 1b, 6 and 8b, but the jump example uses point contacts, where τⱼ should be 0. The paper never says which contacts carry τⱼ.
7. **The impact is not modeled** (§6), although the jump examples land.

## 10. Critique of [T] in the light of [D]

1. **Mis-citation:** [D] ([T]'s ref. [17]) is grouped with full-body-model methods "with a significant computational burden" (Sec. II-A). [D] is the opposite.
2. **No credit for the formulation it uses:** Stage 3 (Eqs. 47, 49, 50) is a planar, single-mode [D] with ḣ fixed. [T]'s new part is the reference and guess generation (Stages 1–2), and its measured value is ×1.8–1.9 in Stage 3 time, with no reliability gain (Phases 4, 6).
3. **The contact model is weaker than [D]'s** (§6, #12): [D]'s scheduled corner points would have given a consistent heel-lift push-off.
4. **Landing outside the optimization**, while [D] already showed it inside. Our failures concentrate there (Phase 6).
5. **Shared with [D]:** no success rate, no baseline time, no integration-error discussion for open-loop playback. In [T] this matters more, because the launch and flight are played back open-loop (#21: position-only playback overshoots take-off vz by 28 % in MuJoCo).

---

## 11. Implementing [D] on H1-2: what exists, what needs a decision

**Reusable from this repo (no changes to the [T] pipeline needed):**

- `models/symmetric.py`: Pinocchio-CasADi model with planar root and left = right matrix. `pinocchio.casadi` provides `computeCentroidalMap` (A(q), Eq. 2), CoM and frame placements, i.e. Eqs. 7c, 7h, 7i.
- Sole geometry in `sim/robot.py` (`FOOT_X`, `SOLE_Z`): toe and heel points (planar) or 4 corners (3D).
- `sim/mujoco_runner.py` (PD + feedforward; [D] needs torques from inverse dynamics after the solve) and `sim/landing.py`.
- `scripts/evaluate.py` target grid, so that success rates and times are comparable with Phase 6.
- Not reusable as is: `collocation.trapezoid` assumes a fixed step. [D] needs backward Euler with a free dt per interval (Eqs. 7d–7g).

**Deviations to decide before building** (each would be a new row in a Differences table):

| # | Question | [D] | Options for H1-2 |
|---|---|---|---|
| a | Solver | SNOPT | IPOPT (SNOPT is not in this CasADi build), or a SNOPT license |
| b | Model | full 3D Atlas | full 3D H1-2 (≈ 9k variables, the paper's spirit) or the 7-DOF planar model (≈ 2.2k, direct comparison with our Stage 3) |
| c | Contacts | 4 corners per foot, fixed schedule (Sec. III-A) | the same fixed schedule first; complementarity (Sec. II-E) only after that works |
| d | Torque limits | none | add them (H1-2 ankle and shoulder), like #13, or check them afterwards only |
| e | Initial guess | not specified | naive crouch (like the Phase 3 baseline); optionally seed from our [T] Stage 3 solution as a comparison |
| f | Landing | inside the plan | plan toes → heels touch-down, then hand over to the landing controller after the plan ends |
| g | Execution | QP controller [D15] in Drake | our PD + feedforward (#21) |
| h | Target | Atlas jump, box 29 cm | h 0.10 m, l 0.40 m first (same as the [T] nominal), then the Phase 6 grid |

## 12. Implementation results (2026-11-17)

Code: `humanoid_jump/models/full.py` (3D H1-2, 27 joints, rpy root so q̇ = v), `humanoid_jump/dai2014.py` (the NLP), `humanoid_jump/sim/play_full.py` (MuJoCo playback), `scripts/eval_dai2014.py` (Phase 6 grid). Decisions taken for §11: a IPOPT; b full 3D H1-2; c the fixed schedule stance → toes → flight → toes → full sole, 4 corners per foot, no complementarity; d torque limits as an off-by-default toggle (`--torque-limits`); e naive crouch guess; f the plan ends at rest (the landing controller is a second variant); g PD + feedforward; h the Phase 6 grid.

**Problem size (nominal).** 54 knots, 6611 variables (q, v 1782 each; r, ṙ, r̈ 486; h, ḣ 324; c 1296; F 888 on active corners only; dt 53), 9794 constraint rows. About 2300 of the variables (r, ṙ, h, c) are redundant with q, v through Eqs. 7c, 7h, 7i. [T] Stage 3: 2139 variables, 2809 rows.

**Nominal target (h 0.10, l 0.40).**

| | iters | time | plan peak τ / H1-2 limit | MuJoCo, H1-2 limits | MuJoCo, limits removed |
|---|---|---|---|---|---|
| [T] pipeline (S1 + S2 + S3) | 90 + QP + 221 | 12.3 s | ≤ 1 | STANDING, td 0.364 m | – |
| [D] as in the paper | 1051 | 131–136 s | ankle 2.57, hip 1.48, knee 1.34 | FELL (td 0.105 m) | STANDING, td 0.311 m |
| [D] + torque limits (ours) | 487 | 153.7 s | ≤ 1.00 | FELL | – |

**Grid (20 targets, max 1500 iters, same success criterion as Phase 6).**

| | [T] | [D] |
|---|---|---|
| Optimization success | 16/20 | 20/20 |
| Median time over successes | 22.1 s | 122.3 s (976 iters; range 24.5–177.9 s) |
| Plans over a torque limit | 0 | 20/20 (peak 2.3–3.3 × limit) |
| End to end, H1-2 limits | 11/20 | 0/20 (landing controller: 0/20) |
| End to end, limits removed ([D]'s assumption) | – | 11/20 (landing controller: 11/20) |
| Touch-down shortfall, median | 43 mm | 96 mm (limits removed) |

Successes with limits removed: all of h 0.05 (incl. l 0.6, which [T] cannot solve), h 0.10 except l 0.6, h 0.15 at l 0.2–0.3. Failures grow with h: at h 0.15–0.20 the robot lands 0.10–0.18 m short (stands off target with the landing controller, often falls with the planned landing).

**Why [D] needs so much torque on H1-2.** (1) No torque limit in the NLP (Sec. II-A); with the toggle on a feasible plan exists. (2) The toe-only phase puts the whole push on the toe corners, so ankle torque ≈ toe force × 0.174 m (130–147 N·m per side vs 140–154 planned; [T]'s flat sole, #12, keeps the CoP within 0.05 m of the ankle). (3) Free dt with a ∫(·)dt cost makes the toe phase short and hard: 31 ms at 2.3–2.6 BW. (4) The first-order transcription leaves joint accelerations Δv/dt unpenalized, so torque jumps at phase transitions (knee 272 rad/s²); [T]'s jerk input makes these continuous. (4) causes spikes only; the plateau is (1)–(3).

**Why [D] lands short even with unlimited torque (hypothesis, not tested).** The simulated take-off velocity is 0.07–0.23 m/s below the plan in both components, more for higher jumps. Likely cause: the backward-Euler transcription with dt up to the 104 ms bound and the base-row residual (up to 1142 N at phase transitions on the nominal plan): the plan is not exactly dynamically consistent, and PD + feedforward cannot reproduce a 31 ms toe phase. A finer grid with limits removed would test this.

**Torque limits and the toe phase.** With the toggle on, the solver keeps the toe phase but stretches it to the dt bound (0.25 s) with the ankle exactly at its limit; in MuJoCo the ankle saturates in 44 % of the launch and the robot falls. A finer grid (diagnostic only) does not change this: the problem is zero torque headroom in toe stance, not discretization.

**Lesson for the "evolution" [D] → [T].** [D] is more general (3D, multi-contact, all 20 targets solve) but about 5.5× slower and, on a robot with a weak ankle, only executable under its own sufficient-torque assumption. [T] gives up generality (planar, one contact mode, flat sole) and gains torque feasibility, smooth inputs and a smaller NLP.
