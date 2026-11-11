"""Spawn the H1-2 standing on the floor and hold the pose with joint PD.

Viewer (macOS needs mjpython for the passive viewer):
    conda activate mujoco
    mjpython scripts/spawn_robot.py
    mjpython scripts/spawn_robot.py --a 0.3        # deeper crouch

Headless check (plain python is fine):
    python scripts/spawn_robot.py --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_jump.sim.robot import ARMS, Robot, build_model, leg_pose  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", type=float, default=0.2, help="leg flexion [rad]: hip=-a, knee=2a, ankle=-a")
    p.add_argument("--duration", type=float, default=5.0, help="headless run time [s]")
    p.add_argument("--headless", action="store_true")
    args = p.parse_args()

    robot = Robot(build_model())
    robot.spawn({**leg_pose(args.a), **ARMS})
    m, d = robot.m, robot.d
    com0 = robot.com()
    print(f"[spawn] mass {robot.mass:.2f} kg | base z {d.qpos[2]:.3f} m | CoM z {com0[2]:.3f} m | "
          f"CoM-to-foot-centre x {robot.com_offset_x() * 1e3:+.0f} mm | a = {args.a:.2f} rad")

    if args.headless:
        n = int(args.duration / m.opt.timestep)
        for _ in range(n):
            robot.step()
        com, rpy = robot.com(), np.degrees(robot.base_rpy())
        drift = np.linalg.norm(com - com0)
        ok = drift < 0.02 and np.abs(rpy[:2]).max() < 5.0
        print(f"[after {args.duration:.1f} s] CoM drift {drift * 1e3:.1f} mm | "
              f"roll {rpy[0]:.2f} deg, pitch {rpy[1]:.2f} deg | contacts {d.ncon}")
        print("[PASS] robot stands" if ok else "[FAIL] robot did not hold the pose")
        return 0 if ok else 1

    import mujoco.viewer

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            t0 = time.perf_counter()
            for _ in range(int(0.01 / m.opt.timestep)):  # 10 ms of physics per frame
                robot.step()
            viewer.sync()
            time.sleep(max(0.0, 0.01 - (time.perf_counter() - t0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
