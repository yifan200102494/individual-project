import os
import sys
import time
import math
import argparse
from typing import Dict, Any

import numpy as np
import pybullet as p

# -----------------------------
# Import project modules
# -----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

import environmen
import obstacle
import control


SPEED_SETTINGS = {
    "slow": 0.001,
    "medium": 0.003,
    "fast": 0.005,
    "extreme": 0.007,
    "insane": 0.017,
}


def force_setup_environment(connection_mode: int = p.GUI):
    """
    强制让 environmen.setup_environment() 以指定模式连接 PyBullet。
    """
    if p.isConnected():
        p.disconnect()

    original_connect = p.connect

    def forced_connect(*args, **kwargs):
        return original_connect(connection_mode)

    p.connect = forced_connect
    try:
        result = environmen.setup_environment()
        if not p.isConnected():
            original_connect(connection_mode)
            result = environmen.setup_environment()
    finally:
        p.connect = original_connect

    return result


def configure_debug_camera():
    """设置一个比较适合观察和手动截图的固定视角。"""
    p.resetDebugVisualizerCamera(
        cameraDistance=1.35,
        cameraYaw=42,
        cameraPitch=-30,
        cameraTargetPosition=[0.50, 0.02, 0.14],
    )


def run_visual_trial(
    speed_val: float,
    trial_num: int = 0,
    timeout: float = 45.0,
    slow_down: float = 1.0 / 120.0,
    hold_window: bool = True,
) -> Dict[str, Any]:
    """
    跑一次 GUI 可视化 trial，用来手动截图/演示。
    """
    robot_id, tray_id, cube_id = force_setup_environment(connection_mode=p.GUI)

    try:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    except Exception:
        pass

    configure_debug_camera()
    p.setRealTimeSimulation(0)

    dynamic_obs = obstacle.DynamicObstacle()
    dynamic_obs.base_speed = speed_val
    dynamic_obs.current_speed = speed_val

    controller = control.RobotController(robot_id, tray_id)

    metrics = {
        "collisions": 0,
        "min_dist": float("inf"),
        "start": time.time(),
    }

    def sim_step():
        dynamic_obs.update()

        obs_pos = dynamic_obs.get_position()
        eef_pos = controller.get_current_eef_pos()
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(obs_pos, eef_pos)))
        if dist < metrics["min_dist"]:
            metrics["min_dist"] = dist

        contact_arm = p.getContactPoints(bodyA=robot_id, bodyB=dynamic_obs.get_id())
        contact_cube = p.getContactPoints(bodyA=cube_id, bodyB=dynamic_obs.get_id())
        if len(contact_arm) > 0 or len(contact_cube) > 0:
            metrics["collisions"] += 1

        if slow_down > 0:
            time.sleep(slow_down)

    controller.sim_step_callback = sim_step

    success = False
    fail_reason = ""
    time_taken = None

    try:
        print("=" * 60)
        print("GUI benchmark demo started")
        print(f"Trial: {trial_num}")
        print(f"Obstacle speed: {speed_val}")
        print(f"Timeout: {timeout:.1f}s")
        print("You can take screenshots manually during or after the run.")
        print("=" * 60)
        time.sleep(1.0)

        controller.execute_pick_and_place(cube_id, tray_id)

        f_pos, _ = p.getBasePositionAndOrientation(cube_id)
        t_pos, _ = p.getBasePositionAndOrientation(tray_id)

        dist_xy = np.linalg.norm(np.array(f_pos[:2]) - np.array(t_pos[:2]))
        is_in_tray = (dist_xy < 0.15) and (f_pos[2] > 0.015)
        is_safe = metrics["collisions"] < 15
        time_taken = time.time() - metrics["start"]
        is_on_time = time_taken < timeout

        if is_in_tray and is_safe and is_on_time:
            success = True
        else:
            if not is_in_tray:
                fail_reason = "Drop/Miss"
            elif not is_safe:
                fail_reason = "Collision"
            elif not is_on_time:
                fail_reason = "Timeout"

    except Exception as exc:
        fail_reason = f"Crash: {exc}"
        time_taken = time.time() - metrics["start"]

    result = {
        "speed": speed_val,
        "success": success,
        "min_dist": metrics["min_dist"],
        "collisions": metrics["collisions"],
        "time_taken": time_taken,
        "reason": fail_reason,
    }

    print("\nResult summary")
    print("-" * 60)
    print(f"Success      : {result['success']}")
    print(f"Fail reason  : {result['reason'] or 'N/A'}")
    print(f"Min distance : {result['min_dist']:.4f} m")
    print(f"Collisions   : {result['collisions']}")
    print(f"Time taken   : {result['time_taken']:.2f} s")
    print("-" * 60)

    if hold_window:
        print("\nGUI will stay open. Adjust the view and take screenshots manually, then press Enter to close.")
        try:
            input()
        except KeyboardInterrupt:
            pass

    if p.isConnected():
        p.disconnect()

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-trial PyBullet GUI demo for manual paper screenshots."
    )
    parser.add_argument(
        "--preset",
        choices=list(SPEED_SETTINGS.keys()),
        default="medium",
        help="Named speed preset.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Custom speed value. If given, it overrides --preset.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=45.0,
        help="Timeout threshold in seconds.",
    )
    parser.add_argument(
        "--slow-down",
        type=float,
        default=1.0 / 120.0,
        help="Sleep inserted into each callback step so the GUI is easier to watch.",
    )
    parser.add_argument(
        "--no-hold",
        action="store_true",
        help="Close the GUI immediately after the trial finishes.",
    )
    parser.add_argument(
        "--trial-num",
        type=int,
        default=0,
        help="Trial index label for display only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    speed = args.speed if args.speed is not None else SPEED_SETTINGS[args.preset]

    run_visual_trial(
        speed_val=speed,
        trial_num=args.trial_num,
        timeout=args.timeout,
        slow_down=args.slow_down,
        hold_window=not args.no_hold,
    )