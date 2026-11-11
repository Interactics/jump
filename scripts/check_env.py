"""Phase 0 check: environment, H1-2 model, and Pinocchio <-> MuJoCo consistency.

Run from the workspace root, inside the conda env:
    conda activate mujoco
    python scripts/check_env.py

Checks
  1. Packages import (numpy, casadi + IPOPT, pinocchio + pinocchio.casadi, mujoco, osqp)
  2. H1-2 loads in Pinocchio (URDF) and MuJoCo (MJCF)
  3. Every joint in config/joint_map.yaml exists in both models
  4. Gate: same posture -> same total mass, CoM, and CAM in both models
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[str, bool, str]] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------- 1. imports
def check_imports() -> None:
    import casadi as ca
    report("casadi", True, ca.__version__)

    # Tiny NLP to confirm IPOPT ships with CasADi
    opti = ca.Opti()
    x = opti.variable()
    opti.minimize((x - 2) ** 2)
    opti.solver("ipopt", {"print_time": False}, {"print_level": 0})
    sol = opti.solve()
    report("ipopt solves a toy NLP", abs(sol.value(x) - 2) < 1e-6)

    import pinocchio as pin
    report("pinocchio", True, pin.__version__)
    try:
        import pinocchio.casadi  # noqa: F401
        report("pinocchio.casadi bindings", True)
    except ImportError:
        report("pinocchio.casadi bindings", False,
               "install pinocchio from conda-forge; the pip 'pin' wheel has no CasADi bindings")

    import mujoco
    report("mujoco", True, mujoco.__version__)

    try:
        import osqp  # noqa: F401
        report("osqp", True)
    except ImportError:
        report("osqp", False, "conda install -c conda-forge osqp")


# ---------------------------------------------------------------- 2-4. models
def load_models(cfg: dict):
    import mujoco
    import pinocchio as pin

    urdf = ROOT / cfg["urdf"]
    mjcf = ROOT / cfg["mjcf"]
    if not urdf.exists() or not mjcf.exists():
        report("model files found", False, "run ./setup.sh first")
        return None

    # The URDF already declares a floating base joint, so no extra root joint is added
    pmodel = pin.buildModelFromUrdf(str(urdf))
    if pmodel.joints[1].nq != 7:  # fall back if the floating joint was not parsed as free-flyer
        pmodel = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
    mmodel = mujoco.MjModel.from_xml_path(str(mjcf))
    report("H1-2 loads in Pinocchio", True, f"nq={pmodel.nq}, nv={pmodel.nv}")
    report("H1-2 loads in MuJoCo", True, f"nq={mmodel.nq}, nv={mmodel.nv}")
    return pmodel, mmodel


def check_joint_map(pmodel, mmodel, jmap: dict) -> list[str]:
    import mujoco

    names = [n for pair in jmap["solved"].values() for n in pair]
    names += list(jmap["fixed_like_paper"]) + list(jmap["fixed_h1_2_extra"])
    pin_names = set(pmodel.names)
    mj_names = {mujoco.mj_id2name(mmodel, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(mmodel.njnt)}
    missing = [n for n in names if n not in pin_names or n not in mj_names]
    report("joint_map names exist in both models", not missing,
           f"{len(names)} joints" if not missing else f"missing: {missing}")
    return names


def compare_models(pmodel, mmodel, joint_names: list[str], tol: dict) -> None:
    import mujoco
    import pinocchio as pin

    pdata, mdata = pmodel.createData(), mujoco.MjData(mmodel)

    # Total mass
    m_pin = pin.computeTotalMass(pmodel)
    m_mj = float(mmodel.body_subtreemass[0])
    report("total mass matches", abs(m_pin - m_mj) < tol["mass_kg"],
           f"pinocchio {m_pin:.3f} kg, mujoco {m_mj:.3f} kg")

    rng = np.random.default_rng(0)
    q_pin = pin.neutral(pmodel)
    v_pin = np.zeros(pmodel.nv)
    mdata.qpos[:] = 0.0
    mdata.qpos[3] = 1.0  # MuJoCo quaternion is (w, x, y, z); base at origin, identity
    mdata.qvel[:] = 0.0

    # Same random posture and joint velocities, matched by joint name.
    # Base velocity stays zero, so the two engines' base-velocity frame conventions don't matter.
    for name in joint_names:
        jp = pmodel.getJointId(name)
        jm = mmodel.joint(name)
        lo, hi = mmodel.jnt_range[jm.id]
        qi = rng.uniform(lo, hi) * 0.5 if hi > lo else rng.uniform(-0.5, 0.5)
        vi = rng.uniform(-2.0, 2.0)
        q_pin[pmodel.joints[jp].idx_q] = qi
        v_pin[pmodel.joints[jp].idx_v] = vi
        mdata.qpos[jm.qposadr[0]] = qi
        mdata.qvel[jm.dofadr[0]] = vi

    # CoM
    com_pin = pin.centerOfMass(pmodel, pdata, q_pin)
    mujoco.mj_forward(mmodel, mdata)
    com_mj = mdata.subtree_com[0].copy()
    err = np.linalg.norm(com_pin - com_mj)
    report("CoM matches (random posture)", err < tol["com_m"], f"error {err * 1e3:.3f} mm")

    # Centroidal angular momentum
    hg = pin.computeCentroidalMomentum(pmodel, pdata, q_pin, v_pin)
    L_pin = np.array(hg.angular)
    mujoco.mj_subtreeVel(mmodel, mdata)
    L_mj = mdata.subtree_angmom[0].copy()  # about the subtree CoM, world frame
    rel = np.linalg.norm(L_pin - L_mj) / max(np.linalg.norm(L_mj), 1e-9)
    report("CAM matches (random joint velocities)", rel < tol["cam_rel"], f"relative error {rel * 100:.3f} %")


def main() -> int:
    print("== 1. Packages ==")
    try:
        check_imports()
    except Exception as e:  # keep going so the user sees every problem at once
        report("package imports", False, repr(e))

    print("\n== 2-4. H1-2 model ==")
    cfg = yaml.safe_load((ROOT / "config/robot.yaml").read_text())
    jmap = yaml.safe_load((ROOT / "config/joint_map.yaml").read_text())
    try:
        models = load_models(cfg)
        if models:
            names = check_joint_map(*models, jmap)
            compare_models(*models, names, cfg["tolerance"])
    except Exception as e:
        report("model checks", False, repr(e))

    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\n{len(results) - n_fail}/{len(results)} checks passed.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
