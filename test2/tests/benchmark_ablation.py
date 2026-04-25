import os
import sys
import time
import csv
import random
from multiprocessing import Pool, cpu_count

import numpy as np
import pybullet as p

# 强制 headless，避免 setup_environment() 里写 GUI 时弹窗
p.GUI = p.DIRECT

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import environmen
import obstacle
import control


# =========================
# Ablation config
# =========================
DELAY_STEPS = 24                  # 10 Hz visual latency in a 240 Hz loop
TRIAL_TIMEOUT = 45.0
TRIALS_PER_SET = 30

# 与主 benchmark 中的动态障碍物尺寸一致：0.8 x 0.12 x 0.12 m
OBS_HALF_EXTENTS = np.array([0.40, 0.06, 0.06], dtype=float)

# 保护约束
ACC_LIMIT = 20.0                  # m/s^2
COLLISION_FAIL_THRESHOLD = 15     # 与主 benchmark 一致：15 collision frames 判失败

# True: 只比较 prediction on/off，避免 body-aware layer 主导结果
# False: 比较 final full-stack controller 在延迟下的表现
ISOLATE_PREDICTION_ONLY = True


class TrialAbort(Exception):
    """用于在 sim_step_callback 中提前终止 trial。"""
    pass


def _disconnect_safely():
    try:
        if p.isConnected():
            p.disconnect()
    except Exception:
        pass


def run_ablation_trial(speed_val, mode, trial_idx, timeout=TRIAL_TIMEOUT):
    """
    proactive:
        使用延迟后的 10 Hz 观测 + KF prediction + motion trend + preemptive logic

    reactive:
        使用同样延迟后的 10 Hz 当前观测，但不使用 KF prediction / motion trend /
        preemptive avoid，只保留基于当前延迟观测的局部避障。

    这版只在“抓取成功后到放下前”的运输阶段统计：
        - collision frames
        - acceleration
    这样更接近论文 6.4 想比较的 latency-stressed transport behaviour。
    """
    # 确保同一 speed/trial_idx 下 proactive 与 reactive 使用相同随机种子
    seed = int(speed_val * 1_000_000) + int(trial_idx)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.perf_counter()

    robot_id = tray_id = cube_id = None
    fail_reason = ""

    metrics = {
        "max_accel": 0.0,
        "accel_violations": 0,
        "collision_frames": 0,
        "timed_out": False,
        "eval_active": False,
        "eval_started_at_step": None,
    }

    try:
        robot_id, tray_id, cube_id = environmen.setup_environment()

        try:
            p.setRealTimeSimulation(0)
        except Exception:
            pass

        dynamic_obs = obstacle.DynamicObstacle()
        dynamic_obs.base_speed = speed_val
        dynamic_obs.current_speed = speed_val

        controller = control.RobotController(robot_id, tray_id)

        # 如果只想隔离 prediction 模块贡献，就把 body-aware layer 在两边都关掉
        if ISOLATE_PREDICTION_ONLY:
            if hasattr(controller, "body_emergency_clearance"):
                controller.body_emergency_clearance = -1e9
            if hasattr(controller, "body_candidate_clearance"):
                controller.body_candidate_clearance = -1e9

        vision_sys = getattr(controller, "vision_system", getattr(controller, "camera", None))
        if vision_sys is None:
            raise RuntimeError("Cannot find vision system on RobotController.")

        pos_history = []
        eef_history = []
        step_counter = 0

        def get_delayed_center():
            if len(pos_history) == 0:
                return np.array([10.0, 10.0, 10.0], dtype=float)
            if len(pos_history) > DELAY_STEPS:
                return pos_history[-DELAY_STEPS].copy()
            return pos_history[0].copy()

        def delayed_scan_obstacle_volume():
            """
            给两种模式都提供同样的“延迟后的当前观测”：
            用障碍物中心 + 固定 half extents 构造 AABB。
            """
            delayed_center = get_delayed_center()
            return {
                "min": delayed_center - OBS_HALF_EXTENTS,
                "max": delayed_center + OBS_HALF_EXTENTS,
                "center": delayed_center,
            }

        def delayed_scan_obstacle_height_from_side():
            """
            避免 side camera 泄露实时信息。
            基于同一个 delayed AABB 构造 clearance 信息。
            """
            delayed_center = get_delayed_center()
            max_z = float(delayed_center[2] + OBS_HALF_EXTENTS[2])
            min_z = float(delayed_center[2] - OBS_HALF_EXTENTS[2])

            return {
                "max_height": max_z,
                "min_height": min_z,
                "height_95": max_z,
                "clearance_height": max_z + 0.08,
                "confidence": 1.0,
                "point_count": 100,
                "last_update": time.time(),
            }

        # 两种模式都用同样的延迟视觉输入
        vision_sys.scan_obstacle_volume = delayed_scan_obstacle_volume
        vision_sys.scan_obstacle_height_from_side = delayed_scan_obstacle_height_from_side

        if mode == "reactive":
            # reactive baseline: 不允许任何 predictor-driven 逻辑介入
            def reactive_get_predicted_obstacle_pos(robot_pos):
                return None, {"status": "reactive_only"}

            def reactive_get_motion_trend():
                return {
                    "status": "reactive_only",
                    "is_moving": False,
                    "direction": "stationary",
                    "velocity": [0.0, 0.0, 0.0],
                    "speed": 0.0,
                }

            def reactive_should_preemptive_avoid(robot_pos, robot_target):
                return False, 0.0, "proceed_normal"

            vision_sys.get_predicted_obstacle_pos = reactive_get_predicted_obstacle_pos
            vision_sys.get_motion_trend = reactive_get_motion_trend
            vision_sys.should_preemptive_avoid = reactive_should_preemptive_avoid

        elif mode != "proactive":
            raise ValueError(f"Unknown mode: {mode}")

        def sim_step():
            nonlocal step_counter, eef_history

            # 1) 超时保护（整次任务）
            if (time.perf_counter() - start_time) >= timeout:
                metrics["timed_out"] = True
                raise TrialAbort("Timeout")

            # 2) 更新动态障碍物
            dynamic_obs.update()

            # 3) 记录“真实障碍物中心”轨迹，用于构造 delayed observation
            real_center, _ = p.getBasePositionAndOrientation(dynamic_obs.get_id())
            real_center = np.asarray(real_center, dtype=float)
            pos_history.append(real_center)

            # 4) 只在 proactive 模式下，把 delayed observation 喂给 predictor
            if mode == "proactive" and step_counter % DELAY_STEPS == 0:
                delayed_center = get_delayed_center()
                vision_sys.obstacle_predictor.update(delayed_center)

            # 5) 一旦抓住 cube，正式开始统计 ablation 指标
            if (not metrics["eval_active"]) and (getattr(controller, "grabbed_object_id", None) is not None):
                metrics["eval_active"] = True
                metrics["eval_started_at_step"] = step_counter
                metrics["max_accel"] = 0.0
                metrics["accel_violations"] = 0
                metrics["collision_frames"] = 0
                eef_history = []

            # 6) 只在“抓取后运输阶段”统计指标
            if metrics["eval_active"]:
                # 宏观加速度估计（0.05 s 采样）
                if step_counter % 12 == 0:
                    current_eef = np.array(controller.get_current_eef_pos(), dtype=float)
                    eef_history.append(current_eef)

                    if len(eef_history) >= 3:
                        dt = 0.05
                        v_curr = (eef_history[-1] - eef_history[-2]) / dt
                        v_prev = (eef_history[-2] - eef_history[-3]) / dt
                        accel = float(np.linalg.norm(v_curr - v_prev) / dt)

                        if accel > metrics["max_accel"]:
                            metrics["max_accel"] = accel

                        if accel > ACC_LIMIT:
                            metrics["accel_violations"] += 1

                # collision 按“帧”统计：这一仿真步只要 arm 或 cube 任何一个接触，就记 1 帧
                contact_arm = bool(p.getContactPoints(bodyA=robot_id, bodyB=dynamic_obs.get_id()))
                contact_cube = bool(p.getContactPoints(bodyA=cube_id, bodyB=dynamic_obs.get_id()))

                if contact_arm or contact_cube:
                    metrics["collision_frames"] += 1

            step_counter += 1

        controller.sim_step_callback = sim_step

        # Run one full pick-and-place
        controller.execute_pick_and_place(cube_id, tray_id)

        # Final success check
        f_pos, _ = p.getBasePositionAndOrientation(cube_id)
        t_pos, _ = p.getBasePositionAndOrientation(tray_id)
        dist_xy = np.linalg.norm(np.array(f_pos[:2]) - np.array(t_pos[:2]))

        # 若整个 trial 都没进入 eval window，说明抓取本身失败
        if not metrics["eval_active"]:
            success = False
            fail_reason = "NoTransportWindow"
        else:
            success = (
                dist_xy < 0.15
                and f_pos[2] > 0.015
                and metrics["collision_frames"] < COLLISION_FAIL_THRESHOLD
                and metrics["accel_violations"] == 0
                and not metrics["timed_out"]
            )

            if not success:
                if metrics["timed_out"]:
                    fail_reason = "Timeout"
                elif metrics["accel_violations"] > 0:
                    fail_reason = "AccelViolation"
                elif metrics["collision_frames"] >= COLLISION_FAIL_THRESHOLD:
                    fail_reason = "Collision"
                else:
                    fail_reason = "Drop/Miss"

    except TrialAbort as exc:
        success = False
        fail_reason = str(exc)

    except Exception as exc:
        success = False
        fail_reason = f"Crash: {exc}"

    finally:
        _disconnect_safely()

    return {
        "speed": speed_val,
        "mode": mode,
        "trial_idx": trial_idx,
        "success": bool(success),
        "max_accel": float(metrics["max_accel"]),
        "collision_frames": int(metrics["collision_frames"]),
        "accel_violations": int(metrics["accel_violations"]),
        "eval_active": bool(metrics["eval_active"]),
        "reason": fail_reason,
    }


def _run_task(args):
    return run_ablation_trial(*args)


if __name__ == "__main__":
    test_speeds = [0.003, 0.007, 0.017]   # medium / extreme / insane
    modes = ["proactive", "reactive"]

    cores = max(1, cpu_count() - 2)
    print(
        f"🚀 Ablation benchmark | 10Hz delayed observation | "
        f"collision threshold = {COLLISION_FAIL_THRESHOLD} frames | "
        f"acceleration limit = {ACC_LIMIT:.1f} m/s² | "
        f"prediction_only = {ISOLATE_PREDICTION_ONLY} | "
        f"workers = {cores}"
    )

    raw_rows = []
    final_stats = []

    with Pool(processes=cores) as pool:
        for speed in test_speeds:
            for mode in modes:
                print(f"Testing: Speed={speed}, Mode={mode:10} ...", end=" ", flush=True)

                tasks = [
                    (speed, mode, trial_idx, TRIAL_TIMEOUT)
                    for trial_idx in range(TRIALS_PER_SET)
                ]

                results = pool.map(_run_task, tasks)
                raw_rows.extend(results)

                success_count = sum(1 for r in results if r["success"])
                avg_accel = sum(r["max_accel"] for r in results) / TRIALS_PER_SET
                avg_collision_frames = sum(r["collision_frames"] for r in results) / TRIALS_PER_SET
                success_rate = (success_count / TRIALS_PER_SET) * 100.0

                reason_counts = {}
                for r in results:
                    reason_counts[r["reason"]] = reason_counts.get(r["reason"], 0) + 1
                primary_failure = max(reason_counts.items(), key=lambda x: x[1])[0] if reason_counts else ""

                final_stats.append({
                    "Speed": speed,
                    "Mode": mode,
                    "Success_Rate": round(success_rate, 1),
                    "Avg_Max_Accel": round(avg_accel, 2),
                    "Avg_Collision_Frames": round(avg_collision_frames, 2),
                    "Primary_Failure": primary_failure,
                })

                print(
                    f"Success rate: {success_rate:.1f}% | "
                    f"Avg max accel: {avg_accel:.2f} m/s² | "
                    f"Avg collision frames: {avg_collision_frames:.2f} | "
                    f"Primary failure: {primary_failure or 'N/A'}"
                )

    summary_csv = "ablation_final_goldilocks_transport.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Speed",
                "Mode",
                "Success_Rate",
                "Avg_Max_Accel",
                "Avg_Collision_Frames",
                "Primary_Failure",
            ],
        )
        writer.writeheader()
        writer.writerows(final_stats)

    raw_csv = "ablation_trial_level_transport.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speed",
                "mode",
                "trial_idx",
                "success",
                "max_accel",
                "collision_frames",
                "accel_violations",
                "eval_active",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"\n✅ Summary written to: {summary_csv}")
    print(f"✅ Trial-level results written to: {raw_csv}")
