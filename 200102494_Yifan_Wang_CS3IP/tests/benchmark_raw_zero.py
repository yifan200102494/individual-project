"""
Parallel DIRECT benchmark with STRICT zero-collision success criterion.

Same structure as benchmark_raw.py, but a trial is labelled a success iff:
    placed AND collisions == 0 AND time_taken < 60 s

Output CSV: benchmark_raw_trials_zero.csv.
"""

import os
import sys
import time
import math
import csv
import numpy as np
import pybullet as p
from multiprocessing import Pool, cpu_count

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def run_single_trial(args):
    speed_val, trial_num, timeout = args

    import pybullet as p
    _orig_connect = p.connect
    def _patched_connect(mode=None, *a, **kw):
        return _orig_connect(p.DIRECT)
    p.connect = _patched_connect

    import environmen
    import obstacle
    import control

    robot_id, tray_id, cube_id = environmen.setup_environment()

    dynamic_obs = obstacle.DynamicObstacle()
    dynamic_obs.base_speed = speed_val
    dynamic_obs.current_speed = speed_val

    controller = control.RobotController(robot_id, tray_id)

    metrics = {
        "collisions": 0,
        "cur_consec": 0,
        "max_consec": 0,
        "min_dist": float('inf'),
        "start": time.time(),
    }

    def sim_step():
        dynamic_obs.update()
        obs_pos = dynamic_obs.get_position()
        eef_pos = controller.get_current_eef_pos()
        d = math.sqrt(sum((a - b) ** 2 for a, b in zip(obs_pos, eef_pos)))
        if d < metrics["min_dist"]:
            metrics["min_dist"] = d

        contact_arm = p.getContactPoints(bodyA=robot_id, bodyB=dynamic_obs.get_id())
        contact_cube = p.getContactPoints(bodyA=cube_id, bodyB=dynamic_obs.get_id())
        if len(contact_arm) > 0 or len(contact_cube) > 0:
            metrics["collisions"] += 1
            metrics["cur_consec"] += 1
            if metrics["cur_consec"] > metrics["max_consec"]:
                metrics["max_consec"] = metrics["cur_consec"]
        else:
            metrics["cur_consec"] = 0

    controller.sim_step_callback = sim_step

    is_placed = False
    time_taken = 0.0
    fail_reason = ""
    crashed = False
    success_at_0 = False

    try:
        controller.execute_pick_and_place(cube_id, tray_id)
        f_pos, _ = p.getBasePositionAndOrientation(cube_id)
        t_pos, _ = p.getBasePositionAndOrientation(tray_id)
        dist_xy = float(np.linalg.norm(np.array(f_pos[:2]) - np.array(t_pos[:2])))
        is_placed = (dist_xy < 0.15) and (f_pos[2] > 0.015)
        time_taken = time.time() - metrics["start"]

        # strict zero-collision criterion
        is_safe_0 = metrics["collisions"] == 0
        is_on_time = time_taken < timeout

        if is_placed and is_safe_0 and is_on_time:
            success_at_0 = True
            fail_reason = ""
        else:
            success_at_0 = False
            if not is_placed:
                fail_reason = "Drop/Miss"
            elif not is_safe_0:
                fail_reason = "Collision"
            elif not is_on_time:
                fail_reason = "Timeout"

    except Exception as e:
        crashed = True
        success_at_0 = False
        fail_reason = f"Crash:{type(e).__name__}"
        time_taken = time.time() - metrics["start"]

    try:
        p.disconnect()
    except Exception:
        pass

    return {
        "speed": speed_val,
        "trial": trial_num,
        "success_at_0": int(success_at_0),
        "placed": int(is_placed),
        "collisions": metrics["collisions"],
        "max_consecutive": metrics["max_consec"],
        "min_dist": metrics["min_dist"] if metrics["min_dist"] != float('inf') else -1.0,
        "time_taken": round(time_taken, 3),
        "fail_reason": fail_reason,
        "crashed": int(crashed),
    }


if __name__ == "__main__":
    SPEED_SETTINGS = [
        ("1-Slow",    0.001),
        ("2-Medium",  0.003),
        ("3-Fast",    0.005),
        ("4-Extreme", 0.007),
        ("5-Insane",  0.017),
    ]
    TRIALS_PER_SPEED = 30
    TIMEOUT = 60.0

    cores = max(1, cpu_count() - 2)
    total_trials = len(SPEED_SETTINGS) * TRIALS_PER_SPEED

    print(f"Zero-collision benchmark (headless DIRECT) | workers: {cores}")
    print(f"   {len(SPEED_SETTINGS)} speeds x {TRIALS_PER_SPEED} = {total_trials} trials")
    print(f"   success iff placed AND collisions == 0 AND time < {TIMEOUT}s")
    print("-" * 60)

    tasks = []
    for _, val in SPEED_SETTINGS:
        for i in range(TRIALS_PER_SPEED):
            tasks.append((val, i, TIMEOUT))

    all_results = []
    start_bench = time.time()

    with Pool(processes=cores) as pool:
        for idx, r in enumerate(pool.imap_unordered(run_single_trial, tasks), 1):
            all_results.append(r)
            print(f"  [{idx:>3}/{total_trials}] "
                  f"v={r['speed']:.3f} t={r['trial']:>2}  "
                  f"colls={r['collisions']:>3}  maxc={r['max_consecutive']:>3}  "
                  f"min={r['min_dist']:.3f}  "
                  f"{r['time_taken']:>5.1f}s  "
                  f"{r['fail_reason'] or 'OK'}")

    all_results.sort(key=lambda r: (r["speed"], r["trial"]))

    out_dir = os.path.join(current_dir, "result")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "benchmark_raw_trials_zero_2.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "speed", "trial", "success_at_0", "placed", "collisions",
            "max_consecutive", "min_dist", "time_taken", "fail_reason", "crashed",
        ])
        writer.writeheader()
        writer.writerows(all_results)

    # per-speed summary
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in all_results:
        buckets[r["speed"]].append(r)

    print("=" * 60)
    print(f"{'Speed':>8}  {'N':>4}  {'Succ':>5}  {'Rate(%)':>8}")
    total_succ = 0
    for sp in sorted(buckets):
        rs = buckets[sp]
        succ = sum(r["success_at_0"] for r in rs)
        total_succ += succ
        print(f"{sp:>8.3f}  {len(rs):>4}  {succ:>5}  {100*succ/len(rs):>8.1f}")
    print(f"{'mean':>8}  {len(all_results):>4}  {total_succ:>5}  "
          f"{100*total_succ/len(all_results):>8.1f}")

    total_time = time.time() - start_bench
    print("=" * 60)
    print(f"Done: {len(all_results)} trials in {total_time:.1f}s "
          f"({total_time/len(all_results):.1f}s avg per trial)")
    print(f"Raw per-trial CSV: {out_csv}")
    print("=" * 60)
