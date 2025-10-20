import pybullet as p
import time
import environment
import util

# --- 1. 设置环境 ---
robotId, objectId, trayId, dummyId = environment.setup_environment()

# --- 2. 任务流程 ---
print("开始执行带动态路径规划的取放任务...")

# --- 步骤 1-7: 抓取流程 (保持不变) ---
print("1. 移动到Home位置")
util.move_to_joints(robotId, util.ROBOT_HOME_CONFIG)
_, home_orientation, *_ = p.getLinkState(robotId, util.ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
print("2. 张开夹爪")
util.gripper_open(robotId)
pos_cube_above = [0.5, -0.3, 0.25]
pos_cube_pre_grasp = [0.5, -0.3, 0.2]
pos_at_cube = [0.5, -0.3, 0.13]
print("4. 移动到预抓取位置")
util.move_to_pose(robotId, pos_cube_pre_grasp, home_orientation)
print("5. 下降到抓取位置")
util.move_to_pose(robotId, pos_at_cube, home_orientation)
print("6. 闭合夹爪 (抓取)")
util.gripper_close(robotId)
print("7. 抬起方块至预备高度")
util.move_to_pose(robotId, pos_cube_above, home_orientation)

# --- 【重要改动】调用动态路径规划器来执行避障 ---
# 步骤 8: 规划并执行到托盘上方的路径
print("8. 规划并执行到托盘上方的路径...")
pos_above_tray = [0.5, 0.5, 0.25]
success = util.plan_and_execute_motion(robotId, pos_above_tray, home_orientation, dummyId)

# 检查路径规划是否成功
if not success:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! 致命错误: 无法找到通往托盘的安全路径。任务中止。")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
else:
    # --- 如果路径成功，则继续执行放置流程 ---
    print("9. 路径规划成功，继续执行放置流程。")
    pos_at_tray = [0.5, 0.5, 0.15]
    
    print("10. 下降到放置位置")
    util.move_to_pose(robotId, pos_at_tray, home_orientation)
    
    print("11. 张开夹爪 (放置)")
    util.gripper_open(robotId)
    
    print("12. 抬起手臂")
    util.move_to_pose(robotId, pos_above_tray, home_orientation)
    
    print("13. 回到Home位置")
    util.move_to_joints(robotId, util.ROBOT_HOME_CONFIG)
    
    print("任务完成！")

# --- 3. 保持仿真 ---
try:
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
except p.error as e:
    print("用户关闭了窗口。")

p.disconnect()
print("仿真结束。")

