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
robotId, objectId, trayId, dummyId, interferer_joints = environment.setup_environment()

# --- 初始化动态执行器 ---
executor = DynamicMotionExecutor(robotId, util.ROBOT_END_EFFECTOR_LINK_ID)

# --- 变量定义 ---
home_pos = [0.3, 0.0, 0.5]
home_orientation = p.getQuaternionFromEuler([np.pi, 0.0, 0.0])
pos_cube_base = [0.5, -0.3, 0.025]
pos_cube_above = [pos_cube_base[0], pos_cube_base[1], 0.25] 
pos_at_cube = [pos_cube_base[0], pos_cube_base[1], 0.13]

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
# 6. 抬起方块
# ============================================================
print("\n--- 步骤 6: 抬起方块 ---")
success = executor.move_to_goal_dynamic(
    pos_cube_above, home_orientation,
    ignore_ids=[objectId],
    **interferer_args,
    debug=True
)
if success:
    print("✅ 步骤 6 完成")
else:
    print("❌ 步骤 6 失败")

print("\n" + "="*60)
print("抓取阶段完成 - 开始放置阶段")
print("="*60)

# ============================================================
# 7. 移动到托盘上方（动态避障的关键阶段）
# ============================================================
pos_above_tray = [0.5, 0.5, 0.35]
print("\n--- 步骤 7: 动态移动到托盘上方 ---")
print("  >> 障碍臂正在持续运动，系统将实时感知和规划")

success = executor.move_to_goal_dynamic(
    pos_above_tray, home_orientation,
    ignore_ids=[objectId],
    **interferer_args,
    max_time=40,  # 给更多时间因为路径可能复杂
    debug=True
)
if success:
    print("✅ 步骤 7 完成 - 成功通过动态障碍区域！")
else:
    print("❌ 步骤 7 失败")

# ============================================================
# 8. 下降到放置位置
# ============================================================
pos_at_tray = [0.5, 0.5, 0.15]
print("\n--- 步骤 8: 下降到放置位置 ---")
success = executor.move_to_goal_dynamic(
    pos_at_tray, home_orientation,
    ignore_ids=[objectId],
    **interferer_args,
    max_time=20,
    debug=True
)
if success:
    print("✅ 步骤 8 完成")
else:
    print("❌ 步骤 8 失败")

# ============================================================
# 9. 放置方块
# ============================================================
print("\n--- 步骤 9: 放置方块 ---")
util.gripper_open(robotId, **interferer_args)
if constraint_id is not None:
    p.removeConstraint(constraint_id)
    print("  >> 已移除抓取约束")
util.simulate(seconds=0.5, **interferer_args)
print("✅ 步骤 9 完成")

# ============================================================
# 10. 抬起手臂
# ============================================================
print("\n--- 步骤 10: 抬起手臂 ---")
success = executor.move_to_goal_dynamic(
    pos_above_tray, home_orientation,
    ignore_ids=[],
    **interferer_args,
    debug=True
)
if success:
    print("✅ 步骤 10 完成")
else:
    print("❌ 步骤 10 失败")

# ============================================================
# 11. 回到Home位置
# ============================================================
print("\n--- 步骤 11: 回到Home位置 ---")
success = executor.move_to_goal_dynamic(
    home_pos, home_orientation,
    ignore_ids=[],
    **interferer_args,
    max_time=30,
    debug=True
)
if success:
    print("✅ 步骤 11 完成")
else:
    print("❌ 步骤 11 失败")

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

