import pybullet as p
import time
import environment
import util # 导入我们最终版的通用功能模块

# --- 1. 设置环境 ---
# 【重要改动】接收新增的 dummyId
robotId, objectId, trayId, dummyId = environment.setup_environment()

# --- 2. 任务流程: 此部分为 test.py 主逻辑的直接复制 ---
print("开始执行最终版取放任务...")

# 步骤 1: 移动到 Home 位置并获取夹爪朝向
print("1. 移动到Home位置")
util.move_to_joints(robotId, util.ROBOT_HOME_CONFIG)
_, home_orientation, *_ = p.getLinkState(robotId, util.ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)

# 步骤 2: 张开夹爪
print("2. 张开夹爪")
util.gripper_open(robotId)

# 步骤 3: 定义所有坐标点
pos_cube_above = [0.5, -0.3, 0.25]
pos_cube_pre_grasp = [0.5, -0.3, 0.2]
pos_at_cube = [0.5, -0.3, 0.13]
pos_above_tray = [0.5, 0.5, 0.25]
pos_at_tray = [0.5, 0.5, 0.15]

# 步骤 4: 移动到预抓取位置
print("4. 移动到预抓取位置")
util.move_to_pose(robotId, pos_cube_pre_grasp, home_orientation)

# 步骤 5: 下降到抓取位置
print("5. 下降到抓取位置")
util.move_to_pose(robotId, pos_at_cube, home_orientation)

# 步骤 6: 闭合夹爪 (抓取)
print("6. 闭合夹爪 (抓取)")
util.gripper_close(robotId)

# 步骤 7: 抬起方块
print("7. 抬起方块")
util.move_to_pose(robotId, pos_cube_above, home_orientation)

# 步骤 8: 移动到Home位置作为安全中途点
print("8. 移动到Home位置")
util.move_to_joints(robotId, util.ROBOT_HOME_CONFIG)

# 步骤 9: 移动到托盘上方
print("9. 移动到托盘上方")
util.move_to_pose(robotId, pos_above_tray, home_orientation)

# 步骤 10: 下降到放置位置
print("10. 下降到放置位置")
util.move_to_pose(robotId, pos_at_tray, home_orientation)

# 步骤 11: 张开夹爪 (放置)
print("11. 张开夹爪 (放置)")
util.gripper_open(robotId)

# 步骤 12: 抬起手臂
print("12. 抬起手臂")
util.move_to_pose(robotId, pos_above_tray, home_orientation)

# 步骤 13: 回到Home位置
print("13. 回到Home位置")
util.move_to_joints(robotId, util.ROBOT_HOME_CONFIG)

print("任务完成！")

# --- 3. 保持仿真以便观察 ---
try:
    while True:
        p.stepSimulation()
        time.sleep(1./120.)
except p.error as e:
    print("用户关闭了窗口。")

p.disconnect()
print("仿真结束。")

