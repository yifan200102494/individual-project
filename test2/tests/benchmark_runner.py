import os
import sys
import time
import csv
import math
import random
import multiprocessing as mp
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pybullet as p
import pybullet_data

# 允许脚本直接复用项目主控制逻辑
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import control


@dataclass
class BenchmarkConfig:
    """
    5.2.4 Generalisation Across Workspace / Obstacle Setups

    gap_widths_m 定义为“两根柱子内侧表面之间的真实净空宽度”，
    不是柱子中心距。
    """
    gap_widths_m: tuple = (0.40, 0.35, 0.30, 0.25, 0.20)
    n_trials_per_gap: int = 5
    timeout_sec: float = 45.0

    # 柱子半尺寸 (x, y, z)
    pillar_half_extents: tuple = (0.05, 0.05, 0.20)

    # gate 中心位置
    gate_center: tuple = (0.50, 0.10, 0.20)

    # 任务成功判定：cube 最终 xy 到 tray 中心的容差
    final_xy_tolerance_m: float = 0.15

    # 与论文口径一致：连续碰撞帧超过 15 判失败
    max_consecutive_collision_frames: int = 15

    # 给每次 trial 的 cube 初始 xy 增加很小扰动，避免完全 deterministic
    # 若想完全确定性复现实验，把它改成 0.0
    cube_xy_jitter_std_m: float = 0.003

    random_seed: int = 42


CONFIG = BenchmarkConfig()


class BenchmarkFailure(RuntimeError):
    """在 sim_step_callback 中提前终止 benchmark 的内部异常。"""


def setup_benchmark_env_headless(cube_xy_jitter=(0.0, 0.0)):
    """创建 headless benchmark 环境。"""
    if p.isConnected():
        p.disconnect()

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    tray_id = p.loadURDF("tray/traybox.urdf", [0.5, 0.4, 0], globalScaling=0.8)

    cube_start_pos = [0.5 + cube_xy_jitter[0], -0.3 + cube_xy_jitter[1], 0.04]
    cube_start_orn = p.getQuaternionFromEuler([0, 0, 0])
    cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, cube_start_orn, globalScaling=1.3)
    p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

    ready_poses = [0, -math.pi / 4, 0, -math.pi / 2, 0, math.pi / 3, 0]
    for i in range(7):
        p.resetJointState(robot_id, i, ready_poses[i])

    return robot_id, tray_id, cube_id


def create_narrow_passage(clear_gap_width_m, gate_center, pillar_half_extents):
    """
    创建两根柱子形成 narrow passage。

    clear_gap_width_m = 两根柱子内侧表面之间的真实净空宽度。
    """
    hx, hy, hz = pillar_half_extents
    cx, cy, cz = gate_center

    visual_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
        rgbaColor=[0.60, 0.60, 0.60, 1.00],
    )
    collision_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
    )

    left_center_x = cx - (clear_gap_width_m / 2.0 + hx)
    right_center_x = cx + (clear_gap_width_m / 2.0 + hx)

    left_pillar_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[left_center_x, cy, cz],
    )
    right_pillar_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=[right_center_x, cy, cz],
    )
    return left_pillar_id, right_pillar_id


def min_signed_distance_between_bodies(body_a, body_b, search_radius=2.0):
    """
    返回两个 body 的最小 signed distance。

    PyBullet 中 pt[8] 含义：
      > 0  分离距离
      = 0  接触
      < 0  穿透深度
    """
    pts = p.getClosestPoints(bodyA=body_a, bodyB=body_b, distance=search_radius)
    if not pts:
        return float("inf")
    return min(pt[8] for pt in pts)


def evaluate_final_task_state(cube_id, tray_id, xy_tolerance_m):
    """评估最终是否成功放置。"""
    final_pos, _ = p.getBasePositionAndOrientation(cube_id)
    tray_pos, _ = p.getBasePositionAndOrientation(tray_id)

    final_xy_err = np.linalg.norm(np.array(final_pos[:2]) - np.array(tray_pos[:2]))
    near_tray = final_xy_err <= xy_tolerance_m
    not_dropped = final_pos[2] > 0.015

    return {
        "near_tray": near_tray,
        "not_dropped": not_dropped,
        "final_xy_error_m": final_xy_err,
        "final_cube_z_m": final_pos[2],
    }


def classify_failure(metrics, final_state):
    """根据论文口径分类失败原因。"""
    if metrics["timed_out"]:
        return "Timeout"
    if metrics["max_consecutive_collision_frames"] > metrics["collision_limit_frames"]:
        return "SustainedCollision"
    if not final_state["not_dropped"]:
        return "Drop"
    if not final_state["near_tray"]:
        return "Miss"
    return ""


def run_single_test(task):
    gap_width_m, trial_idx, seed, config_dict = task
    config = BenchmarkConfig(**config_dict)

    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)

    cube_xy_jitter = (0.0, 0.0)
    if config.cube_xy_jitter_std_m > 0.0:
        cube_xy_jitter = tuple(rng.normal(0.0, config.cube_xy_jitter_std_m, size=2))

    metrics = {
        "gap_width_m": gap_width_m,
        "trial_idx": trial_idx,
        "seed": seed,
        "success": False,
        "fail_reason": "",
        "duration_sec": 0.0,
        "min_clearance_m": float("inf"),
        "total_collision_frames": 0,
        "max_consecutive_collision_frames": 0,
        "final_xy_error_m": float("inf"),
        "final_cube_z_m": -1.0,
        "timed_out": False,
        "collision_limit_frames": config.max_consecutive_collision_frames,
        "cube_jitter_x_m": round(cube_xy_jitter[0], 5),
        "cube_jitter_y_m": round(cube_xy_jitter[1], 5),
        "error": "",
    }

    robot_id = None
    tray_id = None
    cube_id = None
    glue_constraint_id = None
    start_time = time.time()

    try:
        robot_id, tray_id, cube_id = setup_benchmark_env_headless(cube_xy_jitter=cube_xy_jitter)
        pillar_ids = create_narrow_passage(
            clear_gap_width_m=gap_width_m,
            gate_center=config.gate_center,
            pillar_half_extents=config.pillar_half_extents,
        )

        controller = control.RobotController(robot_id, tray_id)
        consecutive_collision_frames = 0

        def benchmark_monitor():
            nonlocal consecutive_collision_frames, glue_constraint_id

            elapsed = time.time() - start_time
            if elapsed > config.timeout_sec:
                metrics["timed_out"] = True
                raise BenchmarkFailure(f"Timeout > {config.timeout_sec:.1f}s")

            # 1) 只统计 robot/cube 与 benchmark 柱子的接触
            hit_gate = False
            for pillar_id in pillar_ids:
                robot_hit = p.getContactPoints(bodyA=robot_id, bodyB=pillar_id)
                cube_hit = p.getContactPoints(bodyA=cube_id, bodyB=pillar_id)
                if robot_hit or cube_hit:
                    hit_gate = True
                    break

            if hit_gate:
                metrics["total_collision_frames"] += 1
                consecutive_collision_frames += 1
                metrics["max_consecutive_collision_frames"] = max(
                    metrics["max_consecutive_collision_frames"],
                    consecutive_collision_frames,
                )
            else:
                consecutive_collision_frames = 0

            # 2) 记录 robot + cube 到柱子的最小 signed clearance
            current_clearances = []
            for pillar_id in pillar_ids:
                current_clearances.append(min_signed_distance_between_bodies(robot_id, pillar_id))
                current_clearances.append(min_signed_distance_between_bodies(cube_id, pillar_id))

            current_min = min(current_clearances) if current_clearances else float("inf")
            if current_min < metrics["min_clearance_m"]:
                metrics["min_clearance_m"] = current_min

            # 3) 保留你原 benchmark 的“粘住 cube”逻辑，避免把实验变成 grasp 稳定性测试
            eef_pos = controller.get_current_eef_pos()
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            dist_cube_eef = np.linalg.norm(np.array(cube_pos) - np.array(eef_pos))

            if dist_cube_eef < 0.06 and eef_pos[2] > 0.08 and glue_constraint_id is None:
                glue_constraint_id = p.createConstraint(
                    robot_id,
                    controller.eef_id,
                    cube_id,
                    -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0, 0, 0.02],
                    [0, 0, 0],
                )

            tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
            dist_to_tray_xy = np.linalg.norm(np.array(eef_pos[:2]) - np.array(tray_pos[:2]))
            if dist_to_tray_xy < 0.15 and glue_constraint_id is not None:
                p.removeConstraint(glue_constraint_id)
                glue_constraint_id = None

            # 4) 连续碰撞帧超过阈值直接判失败
            if metrics["max_consecutive_collision_frames"] > config.max_consecutive_collision_frames:
                raise BenchmarkFailure(
                    f"Sustained collision > {config.max_consecutive_collision_frames} frames"
                )

        controller.sim_step_callback = benchmark_monitor
        controller.execute_pick_and_place(cube_id, tray_id)

        final_state = evaluate_final_task_state(
            cube_id=cube_id,
            tray_id=tray_id,
            xy_tolerance_m=config.final_xy_tolerance_m,
        )
        metrics["final_xy_error_m"] = final_state["final_xy_error_m"]
        metrics["final_cube_z_m"] = final_state["final_cube_z_m"]

        fail_reason = classify_failure(metrics, final_state)
        metrics["fail_reason"] = fail_reason
        metrics["success"] = fail_reason == ""

    except Exception as exc:
        metrics["error"] = f"{type(exc).__name__}: {exc}"

        try:
            if p.isConnected() and cube_id is not None and tray_id is not None:
                final_state = evaluate_final_task_state(
                    cube_id=cube_id,
                    tray_id=tray_id,
                    xy_tolerance_m=config.final_xy_tolerance_m,
                )
                metrics["final_xy_error_m"] = final_state["final_xy_error_m"]
                metrics["final_cube_z_m"] = final_state["final_cube_z_m"]
                if not metrics["fail_reason"]:
                    metrics["fail_reason"] = classify_failure(metrics, final_state) or "Crash"
        except Exception:
            if not metrics["fail_reason"]:
                metrics["fail_reason"] = "Crash"

    finally:
        metrics["duration_sec"] = round(time.time() - start_time, 3)
        metrics["min_clearance_m"] = (
            "" if metrics["min_clearance_m"] == float("inf")
            else round(metrics["min_clearance_m"], 4)
        )
        metrics["final_xy_error_m"] = (
            "" if metrics["final_xy_error_m"] == float("inf")
            else round(metrics["final_xy_error_m"], 4)
        )
        metrics["final_cube_z_m"] = round(metrics["final_cube_z_m"], 4)

        if p.isConnected():
            p.disconnect()

    return metrics


def aggregate_results(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["gap_width_m"], []).append(row)

    summary = []
    for gap in sorted(grouped.keys(), reverse=True):
        trials = grouped[gap]

        success_vals = [1 if r["success"] else 0 for r in trials]
        duration_vals = [float(r["duration_sec"]) for r in trials]
        clearance_vals = [float(r["min_clearance_m"]) for r in trials if r["min_clearance_m"] != ""]
        total_collision_vals = [int(r["total_collision_frames"]) for r in trials]
        max_consec_vals = [int(r["max_consecutive_collision_frames"]) for r in trials]
        final_xy_vals = [float(r["final_xy_error_m"]) for r in trials if r["final_xy_error_m"] != ""]
        final_z_vals = [float(r["final_cube_z_m"]) for r in trials]

        fail_reasons = {}
        for r in trials:
            reason = r["fail_reason"] or "Success"
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

        summary.append({
            "gap_width_m": gap,
            "n_trials": len(trials),
            "success_rate_pct": round(100.0 * np.mean(success_vals), 2),

            "mean_duration_sec": round(float(np.mean(duration_vals)), 3),
            "std_duration_sec": round(float(np.std(duration_vals)), 3),

            "mean_min_clearance_m": round(float(np.mean(clearance_vals)), 4) if clearance_vals else "",
            "std_min_clearance_m": round(float(np.std(clearance_vals)), 4) if clearance_vals else "",

            "mean_total_collision_frames": round(float(np.mean(total_collision_vals)), 2),
            "mean_max_consecutive_collision_frames": round(float(np.mean(max_consec_vals)), 2),

            "mean_final_xy_error_m": round(float(np.mean(final_xy_vals)), 4) if final_xy_vals else "",
            "std_final_xy_error_m": round(float(np.std(final_xy_vals)), 4) if final_xy_vals else "",
            "mean_final_cube_z_m": round(float(np.mean(final_z_vals)), 4),

            "failure_breakdown": "; ".join(f"{k}:{v}" for k, v in sorted(fail_reasons.items())),
        })

    return summary


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_tasks(config):
    tasks = []
    config_dict = asdict(config)

    for gap_width_m in config.gap_widths_m:
        for trial_idx in range(config.n_trials_per_gap):
            seed = config.random_seed + int(gap_width_m * 1000) * 100 + trial_idx
            tasks.append((gap_width_m, trial_idx, seed, config_dict))

    return tasks


def main():
    mp.freeze_support()

    cfg = CONFIG
    tasks = build_tasks(cfg)

    cpu_total = os.cpu_count() or 1
    max_workers = min(len(tasks), max(1, cpu_total - 1))

    print(f"开始 5.2.4 benchmark: {len(tasks)} runs, {max_workers} workers")
    print(f"Gap widths (clear gap): {cfg.gap_widths_m}")
    print(f"Trials per gap: {cfg.n_trials_per_gap}")
    print(f"Cube XY jitter std: {cfg.cube_xy_jitter_std_m} m")

    ctx = mp.get_context("spawn")
    raw_rows = []

    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        for res in executor.map(run_single_test, tasks):
            raw_rows.append(res)
            print(
                f"gap={res['gap_width_m']:.2f}m | "
                f"trial={res['trial_idx']} | "
                f"success={res['success']} | "
                f"reason={res['fail_reason'] or 'Success'} | "
                f"clearance={res['min_clearance_m']}m | "
                f"total_coll={res['total_collision_frames']} | "
                f"max_consec={res['max_consecutive_collision_frames']} | "
                f"xy_err={res['final_xy_error_m']}m"
            )

    raw_rows.sort(key=lambda r: (-r["gap_width_m"], r["trial_idx"]))
    summary_rows = aggregate_results(raw_rows)

    out_dir = os.path.dirname(__file__)
    raw_path = os.path.join(out_dir, "benchmark_report_raw.csv")
    summary_path = os.path.join(out_dir, "benchmark_report_summary.csv")

    write_csv(raw_path, raw_rows)
    write_csv(summary_path, summary_rows)

    print("\nBenchmark completed.")
    print(f"Raw report saved to: {raw_path}")
    print(f"Summary report saved to: {summary_path}")


if __name__ == "__main__":
    main()