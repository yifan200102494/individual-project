# pybullet_starter.py (最终版，修复了重试循环)

import pybullet as p
import time
import environment
import util
import numpy as np

# --- 1. 设置环境 ---
robotId, objectId, trayId, dummyId, interferer_joints = environment.setup_environment()

# --- 【新增】定义干扰臂参数字典 ---
interferer_args = {
    "interferer_id": dummyId,
    "interferer_joints": interferer_joints,
    "interferer_update_rate": 120 
}

# --- 变量定义 ---
home_pos = [0.3, 0.0, 0.5]
home_orientation = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
pos_cube_base = [0.5, -0.3, 0.025]
pos_cube_above = [pos_cube_base[0], pos_cube_base[1], 0.25] 
pos_at_cube = [pos_cube_base[0], pos_cube_base[1], 0.13]  
obstacles_for_grasp = [dummyId, trayId] 


# =============================================================
# 【重大修改】定义一个辅助函数来处理重试逻辑
# =============================================================
def attempt_motion_until_success(step_name, motion_func, *args, **kwargs):
    """
    在一个循环中不断尝试调用一个运动/规划函数，直到它返回True。
    如果在循环中，它会调用 simulate 来让干扰臂移动。
    """
    print(f"--- {step_name} ---")
    
    # 【修复】从 kwargs 中 *只* 提取 'simulate' 函数认识的参数
    # 而不是盲目地复制所有 kwargs (这导致了 'target_joints_override' 错误)
    sim_kwargs = {
        "interferer_id": kwargs.get("interferer_id"),
        "interferer_joints": kwargs.get("interferer_joints"),
        "interferer_update_rate": kwargs.get("interferer_update_rate", 120)
    }
    
    while True:
        # 尝试执行运动规划
        # motion_func (例如 plan_and_execute_motion) 仍然会接收到
        # *所有* kwargs，包括 'target_joints_override'，这是正确的。
        success = motion_func(*args, **kwargs)
        
        if success:
            print(f"  ✅ {step_name} 成功。")
            return True # 成功，退出循环
        
        # 如果失败...
        print(f"  [!!] {step_name} 路径规划失败 (可能被阻挡)。等待 0.3 秒后重试...")
        
        # 【关键】调用 simulate 等待，并让干扰臂移动
        # 【修复】这里现在传递的是 *清理过* 的 sim_kwargs
        util.simulate(seconds=0.3, slow_down=True, **sim_kwargs)
        

# =============================================================
# --- 任务执行 (现在使用重试循环) ---
# =============================================================

# 1. 移动到Home位置
attempt_motion_until_success(
    "1. 移动到Home位置",
    util.plan_and_execute_motion,
    robotId, home_pos, home_orientation, 
    obstacles_for_grasp, 
    target_joints_override=util.ROBOT_HOME_CONFIG,
    **interferer_args # 传递 interferer_args
)

print("2. 张开夹爪")
util.gripper_open(robotId, **interferer_args) 

# 3. 移动到抓取位置上方
attempt_motion_until_success(
    "3. 移动到抓取位置上方",
    util.plan_and_execute_motion,
    robotId, pos_cube_above, home_orientation, 
    obstacles_for_grasp, 
    **interferer_args
)

# 4. 下降到抓取位置
attempt_motion_until_success(
    "4. 下降到抓取位置",
    util.plan_and_execute_motion,
    robotId, pos_at_cube, home_orientation, 
    obstacles_for_grasp, 
    **interferer_args
)

print("5. 闭合夹爪 (抓取)")
constraint_id = None 
util.gripper_close(robotId, **interferer_args)
constraint_id = p.createConstraint(robotId, util.ROBOT_END_EFFECTOR_LINK_ID, objectId, -1, p.JOINT_FIXED, 
                    jointAxis=[0, 0, 0], 
                    parentFramePosition=[0, 0, 0.05], 
                    childFramePosition=[0, 0, 0]) 

# 6. 抬起方块
attempt_motion_until_success(
    "6. 抬起方块至抓取高度",
    util.plan_and_execute_motion,
    robotId, pos_cube_above, home_orientation, 
    obstacles_for_grasp, 
    **interferer_args
)
    
print("--- 抓取阶段完成 ---")

# --- 放置阶段的障碍物和航点 ---
obstacles_for_place_all = [dummyId, trayId] 
obstacles_for_place_TARGETING_TRAY = [dummyId] 

z_safe_cruise = 0.7 
pos_grasp_safe_z = [pos_cube_above[0], pos_cube_above[1], z_safe_cruise]  
pos_above_tray = [0.5, 0.5, 0.25]
pos_tray_safe_z  = [pos_above_tray[0], pos_above_tray[1], z_safe_cruise]  
pos_at_tray = [0.5, 0.5, 0.15] 

# 步骤 8: 抬升
attempt_motion_until_success(
    "8. (航点 1) 抬升至安全高度",
    util.plan_and_execute_motion,
    robotId, pos_grasp_safe_z, home_orientation, 
    obstacles_for_place_all, 
    **interferer_args
)

# 步骤 9: 平移
attempt_motion_until_success(
    "9. (航点 2) 在安全高度平移",
    util.plan_and_execute_motion,
    robotId, pos_tray_safe_z, home_orientation, 
    obstacles_for_place_TARGETING_TRAY, 
    **interferer_args
)

# 步骤 10: 下降
attempt_motion_until_success(
    "10. (航点 3) 下降至托盘上方",
    util.plan_and_execute_motion,
    robotId, pos_above_tray, home_orientation, 
    obstacles_for_place_TARGETING_TRAY, 
    **interferer_args
)

# --- 【修改】删除了旧的 'if not success_to_tray:' 致命错误检查 ---
# 因为上面的函数会一直重试直到成功，所以我们不再需要这个检查了。

print("10. 转移路径规划成功，继续执行放置流程。")

# 步骤 11: 下降到放置位置
attempt_motion_until_success(
    "11. 下降到放置位置 (安全)",
    util.plan_and_execute_motion,
    robotId, pos_at_tray, home_orientation, 
    obstacles_for_place_TARGETING_TRAY, 
    **interferer_args
)
    
print("12. 张开夹爪 (放置)")
util.gripper_open(robotId, **interferer_args)
if constraint_id is not None:
    p.removeConstraint(constraint_id)
    print("  >> 已移除抓取约束。")
    constraint_id = None

util.simulate(seconds=0.5, **interferer_args) 

# 步骤 13: 抬起手臂
attempt_motion_until_success(
    "13. 抬起手臂 (安全)",
    util.plan_and_execute_motion,
    robotId, pos_above_tray, home_orientation, 
    obstacles_for_place_TARGETING_TRAY, 
    **interferer_args
)

# 步骤 14: 抬升至安全巡航高度
attempt_motion_until_success(
    "14. (航点 4) 抬升至安全巡航高度",
    util.plan_and_execute_motion,
    robotId, pos_tray_safe_z, home_orientation, 
    obstacles_for_place_TARGETING_TRAY, 
    **interferer_args
)

# 步骤 15: 回家
attempt_motion_until_success(
    "15. 规划并执行回到Home位置的路径",
    util.plan_and_execute_motion,
    robotId, home_pos, home_orientation, 
    obstacles_for_place_all, 
    target_joints_override=util.ROBOT_HOME_CONFIG,
    **interferer_args
)

print("任务完成！")

# --- 3. 保持仿真 (并持续激活干扰臂) ---
try:
    while True:
        util.simulate(steps=1, **interferer_args)
        
except p.error as e:
    print("用户关闭了窗口。")

p.disconnect()
print("仿真结束。")