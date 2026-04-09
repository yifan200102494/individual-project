import os
import sys
import time
import math
import csv
import numpy as np
import pybullet as p
from multiprocessing import Pool, cpu_count

# 确保导入路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import environmen
import obstacle
import control

def run_single_trial(speed_val, trial_num, timeout=45.0):
    """
    单次测试核心逻辑。由于 multiprocessing 的限制，此函数必须是自包含的。
    """
    # 强制使用 DIRECT 模式以实现极致提速
    # 注意：如果 setup_environment 内部写死了 p.GUI，请确保在此处覆盖或修改原函数
    robot_id, tray_id, cube_id = environmen.setup_environment() 
    
    dynamic_obs = obstacle.DynamicObstacle()
    dynamic_obs.base_speed = speed_val
    dynamic_obs.current_speed = speed_val
    
    controller = control.RobotController(robot_id, tray_id)
    
    metrics = {
        "collisions": 0,
        "min_dist": float('inf'),
        "start": time.time(),
        "aborted": False
    }
    
    def sim_step():
        dynamic_obs.update()
        obs_pos = dynamic_obs.get_position()
        eef_pos = controller.get_current_eef_pos()
        dist = math.sqrt(sum((a-b)**2 for a,b in zip(obs_pos, eef_pos)))
        
        if dist < metrics["min_dist"]:
            metrics["min_dist"] = dist
            
        contact_arm = p.getContactPoints(bodyA=robot_id, bodyB=dynamic_obs.get_id())
        contact_cube = p.getContactPoints(bodyA=cube_id, bodyB=dynamic_obs.get_id())
        
        if len(contact_arm) > 0 or len(contact_cube) > 0:
            metrics["collisions"] += 1

    controller.sim_step_callback = sim_step
    success = False
    fail_reason = ""
    
    try:
        # 执行搬运。如果碰撞已经超过 15 帧，理论上应在此处增加逻辑提前 break 任务
        # 但为了保持逻辑简单，我们依然运行完任务，但在判定时执行“一票否决” 
        controller.execute_pick_and_place(cube_id, tray_id)
        
        f_pos, _ = p.getBasePositionAndOrientation(cube_id)
        t_pos, _ = p.getBasePositionAndOrientation(tray_id)
        
        # 判定标准：终点容差 [cite: 158]、安全否决 、超时否决 [cite: 160]
        dist_xy = np.linalg.norm(np.array(f_pos[:2]) - np.array(t_pos[:2]))
        is_in_tray = (dist_xy < 0.15) and (f_pos[2] > 0.015) 
        is_safe = metrics["collisions"] < 15 
        time_taken = time.time() - metrics["start"]
        is_on_time = time_taken < timeout 
        
        if is_in_tray and is_safe and is_on_time:
            success = True
        else:
            if not is_in_tray: fail_reason = "Drop/Miss"
            elif not is_safe: fail_reason = "Collision"
            elif not is_on_time: fail_reason = "Timeout"
            
    except Exception as e:
        fail_reason = "Crash"

    p.disconnect()
    return {
        "speed": speed_val,
        "success": success,
        "min_dist": metrics["min_dist"],
        "collisions": metrics["collisions"],
        "reason": fail_reason
    }

if __name__ == "__main__":
    # 定义 5 个速度梯度 [cite: 155]
    SPEED_SETTINGS = [
        {"name": "1 - Slow", "val": 0.001},
        {"name": "2 - Medium", "val": 0.003},
        {"name": "3 - Fast", "val": 0.005},
        {"name": "4 - Extreme", "val": 0.007},
        {"name": "5 - Insane", "val": 0.017}
    ]
    
    TRIALS_PER_SPEED = 30 # 每组 30 次，总计 150 次测试 
    summary_results = []
    
    # 获取可用核心数，预留 1-2 个核心保证系统不卡死
    cores = max(1, cpu_count() - 2)
    print(f"🚀 启动并行跑分控制台 | 并行进程数: {cores}")
    print(f"统计目标: 5 组速度 x {TRIALS_PER_SPEED} 次测试 = 150 次试验")
    
    start_bench = time.time()

    with Pool(processes=cores) as pool:
        for setting in SPEED_SETTINGS:
            print(f"\n>>> 正在进行组测试: {setting['name']} (速度: {setting['val']} m/s)")
            
            # 准备并行任务参数
            tasks = [(setting["val"], i) for i in range(TRIALS_PER_SPEED)]
            
            # 使用 starmap 并行执行
            results = pool.starmap(run_single_trial, tasks)
            
            # 提取数据进行专业统计
            success_count = sum(1 for r in results if r["success"])
            dists = [r["min_dist"] for r in results if r["min_dist"] != float('inf')]
            colls = [r["collisions"] for r in results]
            
            success_rate = (success_count / TRIALS_PER_SPEED) * 100
            avg_dist = np.mean(dists)
            std_dist = np.std(dists) # 新增：计算标准差，体现稳定性 
            avg_coll = np.mean(colls)
            
            print(f"    ✅ 完成! 成功率: {success_rate:.1f}% | 均值安全距离: {avg_dist:.3f}m | 标准差: {std_dist:.4f}m")
            
            summary_results.append({
                "Speed Level": setting['name'],
                "Success Rate": success_rate,
                "Min Distance / m": round(avg_dist, 4),
                "Dist Std Dev": round(std_dist, 6), # 关键数据：供绘图脚本使用
                "Collisions": round(avg_coll, 2)
            })

    # 保存统计报告
    csv_file = "benchmark_summary_stats.csv"
    with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["Speed Level", "Success Rate", "Min Distance / m", "Dist Std Dev", "Collisions"])
        writer.writeheader()
        writer.writerows(summary_results)
        
    total_time = time.time() - start_bench
    print(f"\n" + "="*50)
    print(f"🎉 150 次蒙特卡洛测试全部完成！")
    print(f"⏱️ 总耗时: {total_time:.1f} 秒 (并行模式)")
    print(f"💾 统计报告已保存至: {csv_file}")
    print("="*50)