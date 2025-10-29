# pybullet_starter.py (完整修复版 - 智能动态放置)

import pybullet as p
import time
import environment
import util
import numpy as np
import random

# --- 1. 设置环境 ---
robotId, objectId, trayId, dummyId, interferer_joints = environment.setup_environment()

# --- 【修复】interferer_args *不*应该包含 robot_id ---
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

# =============================================================
# 基于感知的重试循环
# =============================================================
def attempt_motion_until_success(step_name, motion_func, *args, **kwargs):
    """
    在一个循环中不断尝试规划和运动。
    【新】在 *每次* 规划前，它都会调用"感知"模块来动态发现障碍物。
    【新】如果连续失败多次，会尝试随机移动来改变位置。
    """
    print(f"--- {step_name} ---")
    
    # 1. 提取 sim_kwargs (用于 simulate 函数)
    sim_kwargs = {
        "interferer_id": kwargs.get("interferer_id"),
        "interferer_joints": kwargs.get("interferer_joints"),
        "interferer_update_rate": kwargs.get("interferer_update_rate", 120),
        "slow_down": kwargs.get("slow_down", True)
    }
    
    # 2. 提取 感知 和 运动规划 的特定参数
    motion_kwargs = kwargs.copy()
    motion_kwargs.pop("interferer_id", None)
    motion_kwargs.pop("interferer_joints", None)
    motion_kwargs.pop("interferer_update_rate", None)
    motion_kwargs.pop("slow_down", None)
    
    robot_id_arg = motion_kwargs.pop("robot_id") 
    
    ignore_ids_list = motion_kwargs.pop("ignore_ids", []) # 抓取后忽略方块
    sensor_debug = motion_kwargs.pop("sensor_debug", False) # 绘制射线
    
    # 失败计数器和随机移动参数
    consecutive_failures = 0
    max_failures_before_random = 3
    random_move_attempts = 0
    max_random_attempts = 5
    
    while True:
        
        # ==================================
        # ==== 1. 感知 (Perceive) ====
        # ==================================
        perceived_ids = util.perceive_obstacles_with_rays(
            robot_id_arg,
            util.ROBOT_END_EFFECTOR_LINK_ID,
            debug=sensor_debug
        )
        
        # ==================================
        # ==== 2. 建模 (Model/Filter) ====
        # ==================================
        ignore_set = set(ignore_ids_list)
        ignore_set.add(robot_id_arg) # 1. 忽略机器人自身
        ignore_set.add(0)            # 2. 忽略地面
        ignore_set.add(-1)           # 3. 忽略"未击中"
        
        filtered_obstacle_ids = [oid for oid in perceived_ids if oid not in ignore_set]
        
        if sensor_debug:
            print(f"  [感知] 原始: {perceived_ids} | 忽略: {ignore_set} | 最终障碍物: {filtered_obstacle_ids}")

        # ==================================
        # ==== 3. 规划 (Plan) ====
        # ==================================
        planning_kwargs = motion_kwargs.copy()
        planning_kwargs['obstacle_ids'] = filtered_obstacle_ids 
        planning_kwargs.update(sim_kwargs) 

        success = motion_func(robot_id_arg, *args, **planning_kwargs)
        
        if success:
            print(f"  ✅ {step_name} 成功。")
            consecutive_failures = 0  # 重置失败计数
            return True 
        
        consecutive_failures += 1
        
        # 如果连续失败多次，尝试随机移动
        if consecutive_failures >= max_failures_before_random and random_move_attempts < max_random_attempts:
            print(f"  [!!] 连续失败 {consecutive_failures} 次，尝试随机移动以寻找新路径...")
            
            # 执行随机移动
            random_move_success = util.perform_random_exploration(
                robot_id_arg, 
                filtered_obstacle_ids,
                **sim_kwargs
            )
            
            if random_move_success:
                print(f"  >> 随机探索移动成功，重新尝试原目标...")
                consecutive_failures = 0  # 重置失败计数
                random_move_attempts += 1
            else:
                print(f"  >> 随机探索移动失败，等待后重试...")
                util.simulate(seconds=0.5, **sim_kwargs)
        else:
            print(f"  [!!] {step_name} 路径规划失败 (被感知到的障碍物阻挡)。等待 0.3 秒后重试...")
            util.simulate(seconds=0.3, **sim_kwargs)
        
        # 如果尝试了太多次随机移动仍然失败，可能需要更长时间等待
        if random_move_attempts >= max_random_attempts:
            print(f"  [!!] 已尝试 {max_random_attempts} 次随机探索，等待更长时间...")
            util.simulate(seconds=1.0, **sim_kwargs)
            random_move_attempts = 0  # 重置随机移动计数
        

# =============================================================
# --- 任务执行 (现在使用基于感知的重试循环) ---
# =============================================================

# 【修复】 1. task_args 复制 *干净的* interferer_args
task_args = interferer_args.copy()
task_args["sensor_debug"] = True
# 【修复】 2. *只*给 task_args 添加 robot_id
task_args["robot_id"] = robotId

# =============================================================
# 【【【 *** 逻辑修复 (pybullet_starter.py) *** 】】】
#  为所有需要"忽略"方块的步骤 (抓取/携带) 创建参数
# =============================================================
task_args_ignore_cube = task_args.copy()
task_args_ignore_cube["ignore_ids"] = [objectId]
# =============================================================

# 1. 移动到Home位置
attempt_motion_until_success(
    "1. 移动到Home位置",
    util.plan_and_execute_motion,
    home_pos, home_orientation, # *args
    target_joints_override=util.ROBOT_HOME_CONFIG,
    **task_args # **kwargs
)

print("2. 张开夹爪")
util.gripper_open(robotId, **interferer_args) # <-- 使用干净的 args

# 3. 移动到抓取位置上方
attempt_motion_until_success(
    "3. 移动到抓取位置上方",
    util.plan_and_execute_motion,
    pos_cube_above, home_orientation, 
    **task_args_ignore_cube # <-- 修复：使用新参数
)

# 4. 下降到抓取位置
attempt_motion_until_success(
    "4. 下降到抓取位置",
    util.plan_and_execute_motion,
    pos_at_cube, home_orientation, 
    **task_args_ignore_cube # <-- 修复：使用新参数
)

print("5. 闭合夹爪 (抓取)")
constraint_id = None 
util.gripper_close(robotId, **interferer_args) # <-- 使用干净的 args
constraint_id = p.createConstraint(robotId, util.ROBOT_END_EFFECTOR_LINK_ID, objectId, -1, p.JOINT_FIXED, 
                    jointAxis=[0, 0, 0], 
                    parentFramePosition=[0, 0, 0.05], 
                    childFramePosition=[0, 0, 0])

# 6. 抬起方块
attempt_motion_until_success(
    "6. 抬起方块至抓取高度",
    util.plan_and_execute_motion,
    pos_cube_above, home_orientation, 
    **task_args_ignore_cube # <-- 修复：使用新参数
)
    
print("--- 抓取阶段完成 ---")

# --- 放置阶段的航点 ---
pos_above_tray = [0.5, 0.5, 0.25]
pos_at_tray = [0.5, 0.5, 0.15]

# 步骤 10: 动态规划并移动至托盘上方
attempt_motion_until_success(
    "10. (航点 1) 动态规划并移动至托盘上方",
    util.plan_and_execute_motion,
    pos_above_tray, home_orientation, 
    **task_args_ignore_cube
)

print("10. 转移路径规划成功，继续执行放置流程。")

# =============================================================
# 步骤 11: 智能下降到放置位置 (动态避障)
# =============================================================
print("--- 步骤 11: 智能下降到放置位置 ---")

# 【新逻辑】检查目标位置是否被占用，如果是则寻找替代位置
placement_successful = False
placement_attempts = 0
max_placement_attempts = 5

while not placement_successful and placement_attempts < max_placement_attempts:
    placement_attempts += 1
    
    # 1. 感知当前障碍物
    perceived_ids = util.perceive_obstacles_with_rays(
        robotId, util.ROBOT_END_EFFECTOR_LINK_ID, debug=True
    )
    # 注意：这里只排除机器人、地面和被抓取的物体，保留托盘以便运动规划时避开边缘
    ignore_set = {robotId, 0, -1, objectId}
    filtered_obstacle_ids = [oid for oid in perceived_ids if oid not in ignore_set]
    
    # 2. 检查目标放置位置是否安全（检查托盘内是否有其他物体）
    target_pos_is_safe = True
    if filtered_obstacle_ids:
        for obs_id in filtered_obstacle_ids:
            # 跳过托盘本身，因为我们就是要放在托盘里
            # 但托盘内的其他物体会被检测为障碍物
            if obs_id == trayId:
                continue
                
            aabb_min, aabb_max = p.getAABB(obs_id)
            # 检查目标位置是否在障碍物的AABB范围内（带安全边距）
            safety_margin = 0.12
            if (aabb_min[0] - safety_margin < pos_at_tray[0] < aabb_max[0] + safety_margin and
                aabb_min[1] - safety_margin < pos_at_tray[1] < aabb_max[1] + safety_margin and
                aabb_min[2] - safety_margin < pos_at_tray[2] < aabb_max[2] + safety_margin):
                print(f"  [!!] 警告：目标放置位置 {pos_at_tray} 被障碍物 {obs_id} 占用！")
                target_pos_is_safe = False
                break
    
    # 3. 如果目标位置不安全，生成替代位置
    if not target_pos_is_safe:
        print(f"  >> 尝试 #{placement_attempts}: 寻找替代放置位置...")
        # 在托盘范围内随机生成替代位置
        offset_x = random.uniform(-0.08, 0.08)
        offset_y = random.uniform(-0.08, 0.08)
        alternative_pos_at_tray = [
            pos_at_tray[0] + offset_x,
            pos_at_tray[1] + offset_y,
            pos_at_tray[2]
        ]
        print(f"  >> 替代位置: {alternative_pos_at_tray}")
        
        # 尝试移动到替代位置
        success = attempt_motion_until_success(
            f"11. 下降到替代放置位置 (尝试 #{placement_attempts})",
            util.plan_and_execute_motion,
            alternative_pos_at_tray, home_orientation,
            **task_args_ignore_cube
        )
        
        if success:
            placement_successful = True
            print(f"  ✅ 成功到达替代位置！")
        else:
            print(f"  ❌ 替代位置 #{placement_attempts} 失败，等待重试...")
            util.simulate(seconds=0.5, **interferer_args)
    else:
        # 4. 目标位置安全，直接移动
        print("  >> 目标放置位置安全，正常下降...")
        success = attempt_motion_until_success(
            "11. 下降到放置位置 (安全)",
            util.plan_and_execute_motion,
            pos_at_tray, home_orientation,
            **task_args_ignore_cube
        )
        
        if success:
            placement_successful = True
            print("  ✅ 成功到达目标位置！")
        else:
            print("  ❌ 移动失败，等待重试...")
            util.simulate(seconds=0.5, **interferer_args)

if not placement_successful:
    print("  ❌❌ 警告：多次尝试后仍无法找到安全的放置位置！")
    print("  >> 将在当前位置放置物体...")

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
    pos_above_tray, home_orientation, 
    **task_args_ignore_cube
)

# 步骤 14: 抬升至安全巡航高度
pos_above_tray_safe_return = [pos_above_tray[0], pos_above_tray[1], 0.7]
attempt_motion_until_success(
    "14. (航点 4) 抬升至安全巡航高度",
    util.plan_and_execute_motion,
    pos_above_tray_safe_return, home_orientation,
    **task_args
)

# 步骤 15: 回家
attempt_motion_until_success(
    "15. 规划并执行回到Home位置的路径",
    util.plan_and_execute_motion,
    home_pos, home_orientation, 
    target_joints_override=util.ROBOT_HOME_CONFIG,
    **task_args
)

print("任务完成！")

# --- 3. 保持仿真 (并持续激活干扰臂) ---
try:
    while True:
        util.perceive_obstacles_with_rays(robotId, util.ROBOT_END_EFFECTOR_LINK_ID, debug=True)
        util.simulate(steps=1, **interferer_args)
        
except p.error as e:
    print("用户关闭了窗口。")

p.disconnect()
print("仿真结束。")