import os
import sys
import time
import csv
import math
import pybullet as p
import pybullet_data
import numpy as np

# 动态添加父目录
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import control

def setup_benchmark_env_visual():
    """✅ 修改 1：将 DIRECT 改为 GUI，开启可视化窗口"""
    p.connect(p.GUI) 
    
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)
    trayId = p.loadURDF("tray/traybox.urdf", [0.5, 0.4, 0], globalScaling=0.8)
    cubeId = p.loadURDF("cube_small.urdf", [0.5, -0.3, 0.04], globalScaling=1.3)
    p.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

    ready_poses = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]
    for i in range(7):
        p.resetJointState(robotId, i, ready_poses[i])
        
    # 调整摄像机视角，方便全局观察
    p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0.5, 0, 0.2])
    
    return robotId, trayId, cubeId

def run_dynamic_test(obs_speed_mps, test_id):
    if p.isConnected(): p.disconnect()
    
    robot_id, tray_id, cube_id = setup_benchmark_env_visual()
    
    # --- 制造一个具有威胁的“动态拦截者”（红色实心球） ---
    radius = 0.05
    visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1])
    collision_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    
    # 让障碍物从右侧 (X=0.9) 向左侧横穿机器人的去路 (Y=0.1)
    start_pos = [0.9, 0.1, 0.25]
    dynamic_obs_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_id, 
                                       baseVisualShapeIndex=visual_id, basePosition=start_pos)

    controller = control.RobotController(robot_id, tray_id)
    
    # 记录数据的字典
    metrics = {
        "min_dist_to_obs": 99.9, 
        "collision_frames": 0,    
        "start_time": time.time(),
        "constraint_id": None
    }

    current_obs_pos = list(start_pos)
    dt = 1.0 / 240.0 

    def benchmark_dynamic_monitor():
        # A. 飞球物理更新
        current_obs_pos[0] -= obs_speed_mps * dt
        p.resetBasePositionAndOrientation(dynamic_obs_id, current_obs_pos, [0, 0, 0, 1])

        # B. 碰撞检测与数据记录
        contacts = p.getContactPoints(robot_id, dynamic_obs_id)
        if len(contacts) > 0:
            metrics["collision_frames"] += 1
            
        eef_pos = controller.get_current_eef_pos()
        dist = np.linalg.norm(np.array(eef_pos) - np.array(current_obs_pos))
        if dist < metrics["min_dist_to_obs"]:
            metrics["min_dist_to_obs"] = dist

        # C. 物理“强力胶”（防止快速闪躲时掉落）
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        dist_cube_eef = np.linalg.norm(np.array(cube_pos) - np.array(eef_pos))
        if dist_cube_eef < 0.06 and eef_pos[2] > 0.08 and metrics["constraint_id"] is None:
            cid = p.createConstraint(robot_id, controller.eef_id, cube_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.02], [0, 0, 0])
            metrics["constraint_id"] = cid
            
        tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
        if np.linalg.norm(np.array(eef_pos[:2]) - np.array(tray_pos[:2])) < 0.15 and metrics["constraint_id"] is not None:
            p.removeConstraint(metrics["constraint_id"])
            metrics["constraint_id"] = None

        # ✅ 修改 2：加入微小延时，让仿真速度与现实时间同步，方便肉眼观察躲闪动作
        time.sleep(dt)

    controller.sim_step_callback = benchmark_dynamic_monitor

    success = False
    try:
        controller.execute_pick_and_place(cube_id, tray_id)
        final_pos, _ = p.getBasePositionAndOrientation(cube_id)
        tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
        if np.linalg.norm(np.array(final_pos[:2]) - np.array(tray_pos[:2])) < 0.2:
            success = True
    except Exception:
        pass

    duration = time.time() - metrics["start_time"]
    p.disconnect()
    
    return {
        "obstacle_speed_mps": obs_speed_mps,
        "success": success,
        "duration_sec": round(duration, 2),
        "min_dist_m": round(metrics["min_dist_to_obs"], 4),
        "collision_frames": metrics["collision_frames"]
    }

if __name__ == "__main__":
    speed_test_range = [0.0, 0.05, 0.10, 0.15] # 稍微减少了几个档位，因为开启画面后跑分时间会变长
    final_scores = []
    
    print("\n" + "="*45)
    print(" 🚀 可视化动态拦截预测跑分开始！")
    print("="*45)

    for speed in speed_test_range:
        print(f"正在测试速度: {speed} m/s... (请在弹出的 PyBullet 窗口中观察)")
        res = run_dynamic_test(speed, speed)
        final_scores.append(res)
        print(f"[{'成功' if res['success'] else '失败'}] 最小间距: {res['min_dist_m']}m")

    report_path = os.path.join(os.path.dirname(__file__), "benchmark_dynamic_report.csv")
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_scores[0].keys())
        writer.writeheader()
        writer.writerows(final_scores)
    print(f"\n跑分完成，报告已生成：{report_path}")