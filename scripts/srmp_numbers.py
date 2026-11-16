"""Phase 2A: H1-2 numbers for the SRMP (leg length, pitch inertia, CoP limits) from the MuJoCo model.

    python scripts/srmp_numbers.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_jump.sim.robot import ARMS, FOOT_X, Robot, build_model, leg_pose  # noqa: E402


def pitch_inertia(r: Robot) -> float:
    """Centroidal inertia about the pitch (y) axis: sum of rotated body inertias + parallel-axis terms."""
    m, d = r.m, r.d
    c, I = r.com(), 0.0
    for b in range(1, m.nbody):
        R = d.ximat[b].reshape(3, 3)
        dx = d.xipos[b] - c
        I += (R @ np.diag(m.body_inertia[b]) @ R.T)[1, 1] + m.body_mass[b] * (dx[0] ** 2 + dx[2] ** 2)
    return I


if __name__ == "__main__":
    r = Robot(build_model())
    print(f"mass {r.mass:.2f} kg | sole x in [{FOOT_X[0]}, {FOOT_X[1]}] m (ankle frame)")
    print(" a[rad]  arms    CoM z[m]  CoM x - ankle x[m]   rho[kg m^2]   phi=sqrt(rho/m)[m]")
    for a in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        for name, sh in (("down", 0.0), ("fwd", -1.5), ("up", -3.0)):
            q = {**leg_pose(a), **ARMS, "left_shoulder_pitch_joint": sh, "right_shoulder_pitch_joint": sh}
            r.spawn(q, clearance=0.0)
            ankle_x = r.d.xpos[r.foot_body["left"]][0]
            c, rho = r.com(), pitch_inertia(r)
            print(f"  {a:.1f}   {name:5s}   {c[2]:.3f}      {c[0] - ankle_x:+.3f}            {rho:6.2f}        {np.sqrt(rho / r.mass):.3f}")
