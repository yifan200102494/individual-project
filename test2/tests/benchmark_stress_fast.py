import os
# 限制底层数值库线程，避免与 multiprocessing 互相抢核
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import math
import csv
from multiprocessing import Pool, cpu_count

import numpy as np
import pybullet as p

# 无论环境内部写的是 GUI 还是 DIRECT，都强制改成 DIRECT
# 这样 setup_environment() 里如果调用 p.connect(p.GUI)，也会实际连接到 DIRECT
p.GUI = p.DIRECT

# 确保导入路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import environmen
import obstacle
import control


TRIAL_TIMEOUT = 45.0
COLLISION_FAIL_THRESHOLD = 15


class EarlyStop(Exception):
    """用于在单次试验中尽早中止失败样本。"""
    pass


def _disconnect_safely():
    try:
        if p.isConnected():
            p.disconnect()
    except Exception:
        pass


def run_single_trial(speed_val, trial_num, timeout=TRIAL_TIMEOUT):
    """
    单次测试核心逻辑：
    - 强制 DIRECT 无界面
    - 超时/碰撞过多时立刻中止，尽快回收失败样本
    - 用平方距离减少每步 sqrt 开销
    """
    start_time = time.perf_counter()
    min_dist_sq = float("inf")
    collisions = 0

    robot_id = tray_id = cube_id = None
    fail_reason = ""

    try:
        robot_id, tray_id, cube_id = environmen.setup_environment()

        # 有些环境脚本会忘记显式关闭实时模式，补一层保险
        try:
            p.setRealTimeSimulation(0)
        except Exception:
            pass

        dynamic_obs = obstacle.DynamicObstacle()
        dynamic_obs.base_speed = speed_val
        dynamic_obs.current_speed = speed_val

        controller = control.RobotController(robot_id, tray_id)

        def sim_step():
            nonlocal min_dist_sq, collisions

            # 先更新动态障碍
            dynamic_obs.update()

            # 1) 尽量轻量地记录最小距离：用平方距离，最后再开方
            obs_pos = dynamic_obs.get_position()
            eef_pos = controller.get_current_eef_pos()
            dx = obs_pos[0] - eef_pos[0]
            dy = obs_pos[1] - eef_pos[1]
            dz = obs_pos[2] - eef_pos[2]
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

            # 2) 快速失败：碰撞计数超阈值就结束，不再浪费时间
            if p.getContactPoints(bodyA=robot_id, bodyB=dynamic_obs.get_id()):
                collisions += 1
            if p.getContactPoints(bodyA=cube_id, bodyB=dynamic_obs.get_id()):
                collisions += 1
            if collisions >= COLLISION_FAIL_THRESHOLD:
                raise EarlyStop("Collision")

            # 3) 快速失败：超时立刻结束
            if (time.perf_counter() - start_time) >= timeout:
                raise EarlyStop("Timeout")

        controller.sim_step_callback = sim_step

        controller.execute_pick_and_place(cube_id, tray_id)

        # 再做最终结果判定
        f_pos, _ = p.getBasePositionAndOrientation(cube_id)
        t_pos, _ = p.getBasePositionAndOrientation(tray_id)
        dist_xy = math.hypot(f_pos[0] - t_pos[0], f_pos[1] - t_pos[1])
        is_in_tray = (dist_xy < 0.15) and (f_pos[2] > 0.015)
        is_on_time = (time.perf_counter() - start_time) < timeout
        is_safe = collisions < COLLISION_FAIL_THRESHOLD
        success = is_in_tray and is_on_time and is_safe

        if not success:
            if not is_in_tray:
                fail_reason = "Drop/Miss"
            elif not is_safe:
                fail_reason = "Collision"
            else:
                fail_reason = "Timeout"

    except EarlyStop as e:
        success = False
        fail_reason = str(e)
    except Exception:
        success = False
        fail_reason = "Crash"
    finally:
        _disconnect_safely()

    return {
        "speed": speed_val,
        "trial": trial_num,
        "success": success,
        "min_dist": math.sqrt(min_dist_sq) if min_dist_sq != float("inf") else float("inf"),
        "collisions": collisions,
        "reason": fail_reason,
    }


def _run_task(args):
    return run_single_trial(*args)


def summarize_group(setting_name, results, trials_per_speed):
    success_count = sum(1 for r in results if r["success"])
    dists = [r["min_dist"] for r in results if r["min_dist"] != float("inf")]
    colls = [r["collisions"] for r in results]

    success_rate = (success_count / trials_per_speed) * 100 if trials_per_speed else 0.0
    avg_dist = float(np.mean(dists)) if dists else float("nan")
    std_dist = float(np.std(dists)) if dists else float("nan")
    avg_coll = float(np.mean(colls)) if colls else 0.0

    return {
        "Speed Level": setting_name,
        "Success Rate": round(success_rate, 1),
        "Min Distance / m": round(avg_dist, 4) if not math.isnan(avg_dist) else "nan",
        "Dist Std Dev": round(std_dist, 6) if not math.isnan(std_dist) else "nan",
        "Collisions": round(avg_coll, 2),
    }


if __name__ == "__main__":
    SPEED_SETTINGS = [
        {"name": "1 - Slow", "val": 0.001},
        {"name": "2 - Medium", "val": 0.003},
        {"name": "3 - Fast", "val": 0.005},
        {"name": "4 - Extreme", "val": 0.007},
        {"name": "5 - Insane", "val": 0.017},
    ]

    TRIALS_PER_SPEED = 30
    summary_results = []

    # 留 1 个核给系统；并且每个进程内部已限制 BLAS 线程为 1，避免抢占
    cores = max(1, cpu_count() - 1)
    print(f"🚀 极速无界面跑分 | 并行进程数: {cores}")
    print(f"统计目标: {len(SPEED_SETTINGS)} 组速度 x {TRIALS_PER_SPEED} 次测试 = {len(SPEED_SETTINGS) * TRIALS_PER_SPEED} 次试验")

    bench_start = time.perf_counter()

    with Pool(processes=cores) as pool:
        for setting in SPEED_SETTINGS:
            print(f"\n>>> 正在进行组测试: {setting['name']} (速度: {setting['val']} m/s)")

            tasks = [(setting["val"], i, TRIAL_TIMEOUT) for i in range(TRIALS_PER_SPEED)]
            chunksize = max(1, len(tasks) // (cores * 4) or 1)
            results = list(pool.imap_unordered(_run_task, tasks, chunksize=chunksize))

            group_summary = summarize_group(setting["name"], results, TRIALS_PER_SPEED)
            summary_results.append(group_summary)

            print(
                "    ✅ 完成! 成功率: "
                f"{group_summary['Success Rate']:.1f}% | "
                f"均值安全距离: {group_summary['Min Distance / m']}m | "
                f"标准差: {group_summary['Dist Std Dev']}m | "
                f"平均碰撞: {group_summary['Collisions']}"
            )

    csv_file = os.path.join(current_dir, "benchmark_summary_stats_fast.csv")
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Speed Level", "Success Rate", "Min Distance / m", "Dist Std Dev", "Collisions"],
        )
        writer.writeheader()
        writer.writerows(summary_results)

    total_time = time.perf_counter() - bench_start
    print("\n" + "=" * 50)
    print("🎉 全部测试完成（无界面极速版）")
    print(f"⏱️ 总耗时: {total_time:.1f} 秒")
    print(f"💾 统计报告已保存至: {csv_file}")
    print("=" * 50)
