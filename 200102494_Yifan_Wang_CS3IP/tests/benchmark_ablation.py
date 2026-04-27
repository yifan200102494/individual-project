import os
import sys
import time
import numpy as np
import pybullet as p
from multiprocessing import Pool, cpu_count
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
RESULTS_DIR = os.path.join(current_dir, "result")

def run_goldilocks_trial(speed_val, mode):
    # --- Headless + fast-path patches (per worker process) ---
    import pybullet as p
    _orig_connect = p.connect
    def _patched_connect(mode_arg=None, *a, **kw):
        return _orig_connect(p.DIRECT)
    p.connect = _patched_connect
    time.sleep = lambda _x=0: None  # skip control.py's 1/240-s sleep

    import environmen
    import obstacle
    import control

    robot_id, tray_id, cube_id = environmen.setup_environment()
    dynamic_obs = obstacle.DynamicObstacle()
    dynamic_obs.base_speed = speed_val
    dynamic_obs.current_speed = speed_val
    controller = control.RobotController(robot_id, tray_id)

    controller.use_reactive_only = (mode == 'reactive')
    
    # Adjustment 1: restore the standard industrial 10Hz visual latency (0.1s = 24 steps)
    pos_history = []
    DELAY_STEPS = 24

    eef_history = []
    metrics = {
        "min_dist": float('inf'),   # Closest EEF-to-obstacle distance (m)
        "contact_frames": 0,         # Cumulative contact frames
        "jerk_sum": 0.0,             # Cumulative jerk (m/s^3)
        "jerk_count": 0,
    }
    step_counter = 0

    vision_sys = getattr(controller, 'vision_system', getattr(controller, 'camera', None))

    def patched_scan():
        if len(pos_history) > DELAY_STEPS:
            delayed_pos = pos_history[-DELAY_STEPS]
        elif len(pos_history) > 0:
            delayed_pos = pos_history[0]
        else:
            delayed_pos = np.array([10.0, 10.0, 10.0])
        return {"min": delayed_pos - 0.05, "max": delayed_pos + 0.05, "center": delayed_pos}
    
    # Fake side-view scan so we don't pay the camera-render cost each 24 steps.
    def patched_side_scan():
        info = {"max_height": 0.30, "clearance_height": 0.38, "confidence": 1.0}
        if vision_sys:
            vision_sys.obstacle_height_info = info
        return info

    if vision_sys:
        vision_sys.scan_obstacle_volume = patched_scan
        vision_sys.scan_obstacle_height_from_side = patched_side_scan

    def sim_step():
        nonlocal step_counter
        dynamic_obs.update()
        real_pos = np.array(dynamic_obs.get_position())
        pos_history.append(real_pos)

        # Per-step distance: EEF to obstacle tip
        current_eef = np.array(controller.get_current_eef_pos())
        dist = np.linalg.norm(current_eef - real_pos)
        if dist < metrics["min_dist"]:
            metrics["min_dist"] = dist

        # Jerk computation (skip the first 100 steps of vibration; sample every 12 steps / 0.05s; needs 4-point finite difference)
        if step_counter > 100 and step_counter % 12 == 0:
            eef_history.append(current_eef)
            if len(eef_history) >= 4:
                dt = 0.05
                v1 = (eef_history[-1] - eef_history[-2]) / dt
                v2 = (eef_history[-2] - eef_history[-3]) / dt
                v3 = (eef_history[-3] - eef_history[-4]) / dt
                a_curr = (v1 - v2) / dt
                a_prev = (v2 - v3) / dt
                jerk = np.linalg.norm(a_curr - a_prev) / dt
                metrics["jerk_sum"] += jerk
                metrics["jerk_count"] += 1

        if len(pos_history) > DELAY_STEPS:
            delayed_pos = pos_history[-DELAY_STEPS]
        else:
            delayed_pos = pos_history[0]

        if step_counter % 24 == 0 and vision_sys:
            vision_sys.obstacle_predictor.update(delayed_pos)

        if len(p.getContactPoints(robot_id, dynamic_obs.get_id())) > 0:
            metrics["contact_frames"] += 1

        step_counter += 1

    controller.sim_step_callback = sim_step

    try:
        controller.execute_pick_and_place(cube_id, tray_id)
    except:
        pass

    mean_jerk = metrics["jerk_sum"] / metrics["jerk_count"] if metrics["jerk_count"] > 0 else 0.0
    min_dist = metrics["min_dist"] if metrics["min_dist"] != float('inf') else 0.0

    p.disconnect()
    return min_dist, metrics["contact_frames"], mean_jerk

if __name__ == "__main__":
    test_speeds = [0.001, 0.003, 0.005, 0.007, 0.017]
    modes = ['proactive', 'reactive']
    TRIALS_PER_SET = 30 
    
    cores = max(1, cpu_count() - 2)
    print(f"🚀 Starting continuous-metric ablation test | 10Hz latency | metrics: min_dist + contact_frames + jerk")

    final_stats = []
    with Pool(processes=cores) as pool:
        for speed in test_speeds:
            for mode in modes:
                print(f"Running: Speed={speed}, Mode={mode:10}...", end=" ", flush=True)
                tasks = [(speed, mode) for _ in range(TRIALS_PER_SET)]
                results = pool.starmap(run_goldilocks_trial, tasks)

                min_dists = np.array([r[0] for r in results])
                contacts = np.array([r[1] for r in results])
                jerks = np.array([r[2] for r in results])

                mean_min_dist = float(min_dists.mean())
                std_min_dist = float(min_dists.std(ddof=1)) if len(min_dists) > 1 else 0.0
                mean_contacts = float(contacts.mean())
                mean_jerk = float(jerks.mean())

                final_stats.append({
                    "Speed": speed,
                    "Mode": mode,
                    "Mean_Min_Dist": round(mean_min_dist, 4),
                    "Std_Min_Dist": round(std_min_dist, 4),
                    "Mean_Contact_Frames": round(mean_contacts, 2),
                    "Mean_Jerk": round(mean_jerk, 2),
                })
                print(f"distance: {mean_min_dist:.3f}±{std_min_dist:.3f}m | contact frames: {mean_contacts:.1f} | Jerk: {mean_jerk:.1f} m/s³")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_file = os.path.join(RESULTS_DIR, "ablation_continuous_metrics.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["Speed", "Mode", "Mean_Min_Dist", "Std_Min_Dist", "Mean_Contact_Frames", "Mean_Jerk"])
        writer.writeheader()
        writer.writerows(final_stats)
    print(f"\n✅ Continuous-metric data written: {csv_file}")
