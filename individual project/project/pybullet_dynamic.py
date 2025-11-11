# pybullet_dynamic.py - 实时动态感知和规划系统

import pybullet as p
import time
import environment
import util
import numpy as np
from dynamic_executor import DynamicMotionExecutor

print("="*60)
print("实时动态感知和规划系统")
print("="*60)
print("特点：")
print("  1. 边移动边感知 - 无需等待完整扫描")
print("  2. 增量式规划 - 滚动窗口式局部路径规划")
print("  3. 障碍物运动预测 - 基于历史数据预测未来位置")
print("  4. 紧急避障 - 实时响应危险情况")
print("="*60)

# --- 1. 设置环境 ---
print("正在初始化环境...")
try:
    robotId, objectId, trayId, dummyId, interferer_joints = environment.setup_environment()
    print(f"✅ 环境初始化成功")
    print(f"   机器人ID: {robotId}")
    print(f"   物体ID: {objectId}")
    print(f"   托盘ID: {trayId}")
    print(f"   干扰臂ID: {dummyId}")
    
    # 验证连接
    if not p.isConnected():
        raise Exception("PyBullet未连接！")
    print(f"✅ PyBullet连接状态: 正常")
    
except Exception as e:
    print(f"❌ 环境初始化失败: {e}")
    import sys
    sys.exit(1)

# --- 初始化动态执行器 ---
# 设置托盘信息以过滤托盘底部，只保留托盘四壁作为障碍物
tray_position = np.array([0.5, 0.5, 0.0])  # 托盘位置
tray_size = np.array([0.4, 0.3, 0.05])  # 托盘尺寸 [长, 宽, 高]
executor = DynamicMotionExecutor(
    robotId, 
    util.ROBOT_END_EFFECTOR_LINK_ID,
    tray_position=tray_position,
    tray_size=tray_size
)

# --- 辅助函数：智能移动（自动判断是否需要先抬高）---
def smart_move_to(goal_pos, goal_orn, ignore_ids, max_time=30, debug=True):
    """
    智能移动：自动判断是否需要先抬高再移动
    
    策略：
    1. 检测从当前位置到目标的直线路径上是否有障碍物
    2. 如果有，计算安全高度，先抬高再水平移动再下降
    3. 如果没有，直接移动
    """
    try:
        # 获取当前位置
        ee_state = p.getLinkState(robotId, util.ROBOT_END_EFFECTOR_LINK_ID,
                                  computeForwardKinematics=True)
        current_pos = np.array(ee_state[0])
    except p.error as e:
        print(f"  ❌ [错误] PyBullet连接已断开: {e}")
        print(f"  提示: 请确保GUI窗口没有被关闭")
        return False
    
    try:
        # 获取所有障碍物
        all_bodies = [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
        ignore_set = set(ignore_ids) if ignore_ids else set()
        ignore_set.add(robotId)
        ignore_set.add(0)  # 地面
        obstacles = [bid for bid in all_bodies if bid not in ignore_set]
    except p.error as e:
        print(f"  ❌ [错误] 无法获取障碍物信息: {e}")
        return False
    
    if not obstacles:
        # 没有障碍物，直接移动
        return executor.move_to_goal_dynamic(
            goal_pos, goal_orn, ignore_ids, **interferer_args,
            max_time=max_time, debug=debug
        )
    
    # 计算障碍物最高点
    max_obstacle_height = 0.0
    for obs_id in obstacles:
        aabb_min, aabb_max = p.getAABB(obs_id)
        max_obstacle_height = max(max_obstacle_height, aabb_max[2])
    
    # 判断是否需要绕行
    goal = np.array(goal_pos)
    needs_detour = False
    
    # 简单检查：如果路径穿过障碍物区域
    for obs_id in obstacles:
        aabb_min, aabb_max = p.getAABB(obs_id)
        obs_center = np.array([(aabb_min[i] + aabb_max[i]) / 2 for i in range(3)])
        
        # 计算障碍物到直线的距离
        line_vec = goal - current_pos
        line_len = np.linalg.norm(line_vec)
        if line_len < 0.01:
            continue
        
        line_dir = line_vec / line_len
        to_obs = obs_center - current_pos
        proj = np.dot(to_obs, line_dir)
        
        if 0 < proj < line_len:
            closest = current_pos + line_dir * proj
            dist = np.linalg.norm(obs_center - closest)
            if dist < 0.25:  # 障碍物在路径上
                needs_detour = True
                break
    
    if not needs_detour:
        # 直接路径畅通
        if debug:
            print(f"  ✅ [智能判断] 直接路径可行")
        return executor.move_to_goal_dynamic(
            goal_pos, goal_orn, ignore_ids, **interferer_args,
            max_time=max_time, debug=debug
        )
    
    # 需要绕行：计算安全高度
    safe_height = max(max_obstacle_height + 0.15, 0.40)
    safe_height = min(safe_height, 0.60)  # 不超过60cm
    
    print(f"  🚧 [智能判断] 路径被阻挡，采用安全绕行")
    print(f"     障碍物最高: {max_obstacle_height:.3f}m")
    print(f"     安全高度: {safe_height:.3f}m")
    
    # 三步走：抬高 → 水平移动 → 下降
    via_points = []
    
    # 步骤1: 如果当前低于安全高度，先抬高
    if current_pos[2] < safe_height - 0.05:
        lift_pos = current_pos.copy()
        lift_pos[2] = safe_height
        via_points.append(("抬高到安全高度", lift_pos.tolist()))
    
    # 步骤2: 水平移动到目标正上方
    if goal[2] < safe_height - 0.05:
        horizontal_pos = goal.copy()
        horizontal_pos[2] = safe_height
        via_points.append(("水平移动到目标上方", horizontal_pos.tolist()))
    
    # 执行经过点
    for i, (desc, via_pos) in enumerate(via_points):
        print(f"  📍 步骤 {i+1}: {desc}")
        success = executor.move_to_goal_dynamic(
            via_pos, goal_orn, ignore_ids, **interferer_args,
            max_time=max_time // (len(via_points) + 1), debug=debug
        )
        if not success:
            print(f"  ⚠️ 步骤 {i+1} 失败")
    
    # 最后：下降到目标
    print(f"  📍 最终步骤: 下降到目标位置")
    return executor.move_to_goal_dynamic(
        goal_pos, goal_orn, ignore_ids, **interferer_args,
        max_time=max_time // (len(via_points) + 1), debug=debug
    )

# --- 变量定义 ---
home_pos = [0.3, 0.0, 0.5]
home_orientation = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
pos_cube_base = [0.5, -0.3, 0.025]
pos_cube_above = [pos_cube_base[0], pos_cube_base[1], 0.25] 
pos_at_cube = [pos_cube_base[0], pos_cube_base[1], 0.13]

# 放置阶段的最终目标（系统会自动规划经过点）
pos_at_tray = [0.5, 0.5, 0.15]  # 最终放置位置（系统会自动判断安全高度）

# --- 干扰物体参数 ---
interferer_args = {
    "interferer_id": dummyId,
    "interferer_joints": interferer_joints,
    "interferer_update_rate": 80  # 更频繁的更新，增加挑战
}

print("\n" + "="*60)
print("开始任务执行 - 使用实时动态系统")
print("="*60)

# ============================================================
# 1. 移动到Home位置
# ============================================================
print("\n--- 步骤 1: 移动到Home位置 ---")
# 验证连接
if not p.isConnected():
    print("❌ 错误：PyBullet连接已断开！请不要关闭GUI窗口。")
    import sys
    sys.exit(1)
    
success = executor.move_to_goal_dynamic(
    home_pos, home_orientation,
    ignore_ids=[],
    **interferer_args,
    debug=True
)
if success:
    print("✅ 步骤 1 完成")
else:
    print("❌ 步骤 1 失败")

# ============================================================
# 2. 打开夹爪
# ============================================================
print("\n--- 步骤 2: 打开夹爪 ---")
util.gripper_open(robotId, **interferer_args)
print("✅ 步骤 2 完成")

# ============================================================
# 3. 移动到抓取位置上方
# ============================================================
print("\n--- 步骤 3: 移动到抓取位置上方 ---")
success = executor.move_to_goal_dynamic(
    pos_cube_above, home_orientation,
    ignore_ids=[objectId],  # 忽略方块
    **interferer_args,
    debug=True
)
if success:
    print("✅ 步骤 3 完成")
else:
    print("❌ 步骤 3 失败，重试...")

# ============================================================
# 4. 下降到抓取位置
# ============================================================
print("\n--- 步骤 4: 下降到抓取位置 ---")
success = executor.move_to_goal_dynamic(
    pos_at_cube, home_orientation,
    ignore_ids=[objectId],
    **interferer_args,
    max_time=20,
    debug=True
)
if success:
    print("✅ 步骤 4 完成")
else:
    print("❌ 步骤 4 失败")

# ============================================================
# 5. 抓取方块
# ============================================================
print("\n--- 步骤 5: 抓取方块 ---")
util.gripper_close(robotId, **interferer_args)
constraint_id = p.createConstraint(
    robotId, util.ROBOT_END_EFFECTOR_LINK_ID, 
    objectId, -1, 
    p.JOINT_FIXED,
    jointAxis=[0, 0, 0],
    parentFramePosition=[0, 0, 0.05],
    childFramePosition=[0, 0, 0]
)
print("✅ 步骤 5 完成")

# ============================================================
# 6-7. 智能移动到托盘放置位置（自动规划路径）
# ============================================================
print("\n" + "="*60)
print("抓取阶段完成 - 开始放置阶段")
print("  🤖 系统将自动规划安全路径")
print("="*60)

print("\n--- 步骤 6: 移动到托盘放置位置（智能规划） ---")
print(f"  🎯 最终目标: {pos_at_tray}")
print(f"  🧠 系统将自动判断是否需要先抬高")

success = smart_move_to(
    pos_at_tray, home_orientation,
    ignore_ids=[objectId, trayId],
    max_time=50,
    debug=True
)

if success:
    print("✅ 步骤 6 完成 - 成功到达放置位置！")
else:
    print("❌ 步骤 6 失败 - 无法到达放置位置")

# ============================================================
# 7. 放置方块
# ============================================================
print("\n--- 步骤 7: 放置方块 ---")
util.gripper_open(robotId, **interferer_args)
if constraint_id is not None:
    p.removeConstraint(constraint_id)
    print("  >> 已移除抓取约束")
util.simulate(seconds=0.5, **interferer_args)
print("✅ 步骤 7 完成")

# ============================================================
# 8. 回到Home位置（智能规划）
# ============================================================
print("\n--- 步骤 8: 回到Home位置（智能规划） ---")
success = smart_move_to(
    home_pos, home_orientation,
    ignore_ids=[objectId, trayId],  # 忽略方块和托盘，它们已经安全放置
    max_time=50,
    debug=True
)
if success:
    print("✅ 步骤 8 完成")
else:
    print("❌ 步骤 8 失败")

print("\n" + "="*60)
print("任务完成！")
print("="*60)
print("\n系统性能总结：")
print("  ✓ 实时感知 - 在运动过程中持续更新障碍物信息")
print("  ✓ 增量规划 - 短期局部路径规划，快速响应")
print("  ✓ 运动预测 - 预测障碍物未来位置，提前规划")
print("  ✓ 紧急避障 - 危险情况下快速反应")
print("="*60)

# --- 保持仿真运行 ---
print("\n仿真继续运行，按Ctrl+C或关闭窗口退出...")
try:
    while True:
        # 继续显示实时感知
        util.simulate(steps=1, **interferer_args)
        
except KeyboardInterrupt:
    print("\n用户中断")
except p.error as e:
    print("\n用户关闭了窗口")

p.disconnect()
print("仿真结束。")

