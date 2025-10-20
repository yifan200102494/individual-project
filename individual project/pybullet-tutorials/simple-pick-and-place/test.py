import pybullet as p
import pybullet_data
import time

import util
from util import move_to_joint_pos, gripper_open, gripper_close


def move_to_ee_pose(robot_id, target_ee_pos, target_ee_orientation=None):
    """
    Moves the robot to a given end-effector pose.
    :param robot_id: pyBullet's body id of the robot
    :param target_ee_pos: (3,) list/ndarray with target end-effector position
    :param target_ee_orientation: (4,) list/ndarray with target end-effector orientation as quaternion
    """
    
    # 1. 计算逆向运动学
    if target_ee_orientation is None:
        joint_pos_all = p.calculateInverseKinematics(
            robot_id,
            util.ROBOT_END_EFFECTOR_LINK_ID,
            targetPosition=target_ee_pos,
            maxNumIterations=100,
            residualThreshold=0.001
        )
    else:
        joint_pos_all = p.calculateInverseKinematics(
            robot_id,
            util.ROBOT_END_EFFECTOR_LINK_ID,
            targetPosition=target_ee_pos,
            targetOrientation=target_ee_orientation,
            maxNumIterations=100,
            residualThreshold=0.001
        )

    # 2. 提取前 7 个关节（机械臂的关节）
    joint_pos_arm = list(joint_pos_all[0:7])

    # 3. 移动到计算出的关节位置
    move_to_joint_pos(robot_id, joint_pos_arm)


def main():
    # connect to pybullet with a graphical user interface
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.7, 60, -30, [0.2, 0.2, 0.25])

    # basic configuration
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # allows us to load plane, robots, etc.
    plane_id = p.loadURDF('plane.urdf')  # function returns an ID for the loaded body

    # load the robot
    robot_id = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)

    # load an object to grasp and a box
    # 方块的 basePosition 是 [0.5, -0.3, 0.025]
    object_id = p.loadURDF('cube_small.urdf', basePosition=[0.5, -0.3, 0.025], baseOrientation=[0, 0, 0, 1])
    p.resetVisualShapeData(object_id, -1, rgbaColor=[1, 0, 0, 1])
    # 托盘的位置是: [0.5, 0.5, 0.0]
    tray_id = p.loadURDF('tray/traybox.urdf', basePosition=[0.5, 0.5, 0.0], baseOrientation=[0, 0, 0, 1])

    print('******************************')
    input('press enter to start simulation')
    
    
    # --- START: Pick and Place Logic (Final Version) ---

    # 1. 移动到 Home 位置并获取抓手朝向
    print('going to home configuration')
    move_to_joint_pos(robot_id, util.ROBOT_HOME_CONFIG)
    pos, home_orientation, *_ = p.getLinkState(robot_id, util.ROBOT_END_EFFECTOR_LINK_ID, computeForwardKinematics=True)
    
    # 2. 打开抓手
    print('opening gripper')
    gripper_open(robot_id)

    # 3. 定义所有需要的坐标
    pos_cube_above = [0.5, -0.3, 0.25]     # 方块上方 25cm (安全高度)
    pos_cube_pre_grasp = [0.5, -0.3, 0.2]  # 方块上方 20cm (预抓取高度)
    pos_cube_at = [0.5, -0.3, 0.13]        # 抓取高度 13cm (手掌在方块上方 8cm 处)
    
    pos_tray_above = [0.5, 0.5, 0.25]      # 托盘上方 25cm (安全高度)
    pos_tray_at = [0.5, 0.5, 0.15]         # 托盘上方 15cm (放置高度)
    
    # 4. 移动到方块上方 (pre-grasp)
    print('going to pre-grasp position above cube')
    move_to_ee_pose(robot_id, pos_cube_pre_grasp, home_orientation)
    
    # 5. 移动下去抓取方块
    print('going to cube grasp position')
    move_to_ee_pose(robot_id, pos_cube_at, home_orientation)
    
    # 6. 关闭抓手 (抓!)
    # --- START: 修复抓取延迟 (问题 2) ---
    print('closing gripper (grasping)')
    # 我们不再使用 util.gripper_close()，因为它会等待超时
    # 我们直接命令手指移动到方块的宽度 (0.05m / 2 = 0.025m)
    GRASP_TARGET_POS = 0.025 
    p.setJointMotorControl2(robot_id, 9, p.POSITION_CONTROL, targetPosition=GRASP_TARGET_POS, force=100.0)
    p.setJointMotorControl2(robot_id, 10, p.POSITION_CONTROL, targetPosition=GRASP_TARGET_POS, force=100.0)
    
    # 等待抓取完成 (等待一个很短的时间)
    for _ in range(120): # 约 0.5 秒
        p.stepSimulation()
        time.sleep(1./240.)
    # --- END: 修复抓取延迟 ---
    
    # 7. 向上提起方块 (回到安全高度)
    print('lifting cube (to safe height)')
    move_to_ee_pose(robot_id, pos_cube_above, home_orientation)

    # 8. 移动到托盘上方
    # --- START: 修复运动轨迹 (问题 1) ---
    print('moving to safe home position (intermediate step)')
    move_to_joint_pos(robot_id, util.ROBOT_HOME_CONFIG) # 强制经过 Home 点
    
    print('going to position above tray')
    move_to_ee_pose(robot_id, pos_tray_above, home_orientation)
    # --- END: 修复运动轨迹 ---
    
    # 9. 移动下去准备放置
    print('going to drop-off position')
    move_to_ee_pose(robot_id, pos_tray_at, home_orientation)

    # 10. 打开抓手 (放!)
    print('opening gripper (releasing)')
    gripper_open(robot_id)

    # 11. 向上移动 (离开托盘)
    print('moving up from tray')
    move_to_ee_pose(robot_id, pos_tray_above, home_orientation)
    
    # 12. 移动回 Home 位置
    print('going to home configuration')
    move_to_joint_pos(robot_id, util.ROBOT_HOME_CONFIG)
    
    # --- END: Pick and Place Logic ---

    print('program finished. hit enter to close.')
    input()
    # clean up
    p.disconnect()


if __name__ == '__main__':
    main()