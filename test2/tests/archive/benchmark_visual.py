import os
import sys
import time
import pybullet as p

# 动态添加父目录，确保能加载核心逻辑文件
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import environmen
import control

def run_visual_test(gap_width=0.3):
    # 1. 如果当前有残留的连接，先彻底清理
    if p.isConnected():
        p.disconnect()
        
    # 2. 直接调用你原来的 environmen.py，它会正常拉起 GUI 画面
    robot_id, tray_id, cube_id = environmen.setup_environment()
    
    # 3. 布置测试柱子
    pillar_half_ext = [0.05, 0.05, 0.2]
    visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=pillar_half_ext, rgbaColor=[0.8, 0.2, 0.2, 0.8])
    collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=pillar_half_ext)
    
    # 放置两个红色半透明柱子
    p1_pos = [0.5 - gap_width/2, 0.1, 0.2]
    p2_pos = [0.5 + gap_width/2, 0.1, 0.2]
    pillar1_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id, basePosition=p1_pos)
    pillar2_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id, basePosition=p2_pos)

    controller = control.RobotController(robot_id, tray_id)
    
    # 4. 视觉观察专用的“监控钩子”
    def visual_monitor():
        # A. 精准碰撞检测：只检测机械臂和“两个柱子”是否发生碰撞
        c1 = p.getContactPoints(robot_id, pillar1_id)
        c2 = p.getContactPoints(robot_id, pillar2_id)
        if len(c1) > 0 or len(c2) > 0:
            print("💥 警告：发生碰撞！机械臂撞到测试柱子了！")
            
        # B. 强力胶补丁：防止物理引擎的 Bug 导致方块掉落，影响你观察避障轨迹
        eef_pos = controller.get_current_eef_pos()
        cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
        import numpy as np
        if np.linalg.norm(np.array(cube_pos) - np.array(eef_pos)) < 0.06 and eef_pos[2] > 0.08:
            # 如果还没绑定，则绑定
            if not hasattr(controller, 'cube_cid'):
                controller.cube_cid = p.createConstraint(robot_id, controller.eef_id, cube_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0.02], [0, 0, 0])
        
        # 靠近终点时解除绑定
        tray_pos, _ = p.getBasePositionAndOrientation(tray_id)
        if np.linalg.norm(np.array(eef_pos[:2]) - np.array(tray_pos[:2])) < 0.15 and hasattr(controller, 'cube_cid'):
            p.removeConstraint(controller.cube_cid)
            delattr(controller, 'cube_cid')

        # C. 稍微放慢帧率，方便肉眼观察
        time.sleep(1./120.)

    controller.sim_step_callback = visual_monitor

    print("\n" + "="*40)
    print(f"👀 开始视觉观察测试 | 当前缝隙：{gap_width}m")
    print("="*40)
    
    try:
        controller.execute_pick_and_place(cube_id, tray_id)
    except Exception as e:
        print(f"运行中断: {e}")

    print("\n测试结束！画面已保留，你可以使用鼠标旋转/缩放视角查看最终状态。按 Ctrl+C 退出终端。")
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

if __name__ == "__main__":
    # 你可以在这里修改缝隙宽度，建议先看 0.3m 的情况
    run_visual_test(gap_width=0.3)