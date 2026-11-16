"""Spawn the H1-2 standing on the floor and hold the pose with joint PD.

macOS needs mjpython for the passive viewer:
    conda activate mujoco
    mjpython scripts/spawn_robot.py
    python scripts/spawn_robot.py --headless   # 5 s stand test, PASS/FAIL
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

# Make the humanoid_jump package importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from humanoid_jump.sim.robot import ARMS, Robot, build_model, leg_pose  # noqa: E402

# Leg flexion in radians: hip = -A, knee = 2A, ankle = -A. Larger A = deeper crouch.
CROUCH = 0.2

# How much wall-clock time one rendered frame represents.
FRAME_SECONDS = 0.01


def spawn_robot() -> Robot:
    """Build the H1-2, place it in its crouch pose, and report the starting state."""
    robot = Robot(build_model())
    robot.spawn({**leg_pose(CROUCH), **ARMS})

    base_height = robot.d.qpos[2]
    com_offset_mm = robot.com_offset_x() * 1e3
    print(
        f"[spawn] mass {robot.mass:.2f} kg | base z {base_height:.3f} m | "
        f"CoM-to-foot-centre x {com_offset_mm:+.0f} mm | crouch {CROUCH:.2f} rad"
    )
    return robot


def run_viewer(robot: Robot) -> None:
    """Hold the pose in the passive viewer, stepping real-time physics each frame."""
    model = robot.m
    steps_per_frame = int(FRAME_SECONDS / model.opt.timestep)

    with mujoco.viewer.launch_passive(model, robot.d) as viewer:
        while viewer.is_running():
            frame_start = time.perf_counter()

            for _ in range(steps_per_frame):
                robot.step()
            viewer.sync()

            # Sleep off whatever is left of the frame so playback runs at real time.
            elapsed = time.perf_counter() - frame_start
            time.sleep(max(0.0, FRAME_SECONDS - elapsed))


def stand_test(robot: Robot, seconds: float = 5.0) -> bool:
    """Hold the pose without a viewer; PASS if CoM drifts < 1 cm and the pelvis tilts < 2 deg."""
    com0 = robot.com()
    for _ in range(int(seconds / robot.m.opt.timestep)):
        robot.step()
    drift = np.linalg.norm(robot.com() - com0)
    tilt = np.degrees(np.abs(robot.base_rpy()[:2])).max()
    ok = drift < 0.01 and tilt < 2.0
    print(f"[stand] {seconds:.0f} s | CoM drift {drift * 1e3:.1f} mm | max tilt {tilt:.2f} deg | "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    robot = spawn_robot()
    if "--headless" in sys.argv:
        sys.exit(0 if stand_test(robot) else 1)
    run_viewer(robot)


if __name__ == "__main__":
    main()
