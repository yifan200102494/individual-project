# environment.py (已修改)

import pybullet as p
import pybullet_data
import numpy as np # <-- 需要 numpy

def setup_environment():
    """
    初始化PyBullet环境, 加载所有物体。
    【修改】添加了一个 *动态的* Kuka 手臂作为干扰障碍物。
    Returns:
        tuple: (robotId, objectId, trayId, dummyId, interferer_joint_indices)
    """
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.7, 60, -30, [0.2, 0.2, 0.25])
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.loadURDF('plane.urdf')
    robotId = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)
    object_id = p.loadURDF('cube_small.urdf', basePosition=[0.5, -0.3, 0.025], baseOrientation=[0, 0, 0, 1])
    tray_id = p.loadURDF('tray/traybox.urdf', basePosition=[0.5, 0.5, 0.0], baseOrientation=[0, 0, 0, 1])

    # --- 【重大修改】加载一个 *动态* 的 Kuka 手臂作为“干扰者” ---
    # 将它放置在方块和托盘之间的路径上
    interferer_pos = [0.6, 0.1, 0.0] # 放置在地面上
    # 旋转它，让它朝向工作区域
    interferer_orn = p.getQuaternionFromEuler([0, 0, np.pi/2]) 
    
    # 加载 Kuka 手臂，并将其基座固定
    dummyId = p.loadURDF("kuka_iiwa/model.urdf", 
                         basePosition=interferer_pos, 
                         baseOrientation=interferer_orn,
                         useFixedBase=True)
    print("已加载动态干扰手臂 (Kuka)。")
    
    # 获取干扰臂的关节信息，以便稍后控制
    # Kuka (lbr_iiwa_14_r820) 有 7 个可动关节
    num_interferer_joints = p.getNumJoints(dummyId)
    interferer_joint_indices = []
    for i in range(num_interferer_joints):
        info = p.getJointInfo(dummyId, i)
        if info[2] == p.JOINT_REVOLUTE: # 只控制可动关节
            interferer_joint_indices.append(i)
            # 设置一个初始姿态 (关节中间位置)
            p.resetJointState(dummyId, i, (info[8] + info[9]) / 2.0) 
    
    print(f"  >> 干扰臂有 {len(interferer_joint_indices)} 个可动关节。")
    
    # (可选) 给它上个色，让它看起来更像“危险”
    for j in range(-1, p.getNumJoints(dummyId)): # -1 是 base
         p.changeVisualShape(dummyId, j, rgbaColor=[1, 0.2, 0.2, 1])
    # --- 修改结束 ---

    # 【修正】这个列表必须包含所有可动关节 (7个手臂 + 2个夹爪)
    home_joint_config_arm = [0.0, -0.785, 0.0, -2.356, 0.0, 1.57, 0.785]
    home_joint_config_gripper = [0.04, 0.04] # 启动时就打开
    
    home_joint_positions = home_joint_config_arm + home_joint_config_gripper
    
    joint_indices_arm = [i for i in range(7)]
    joint_indices_gripper = [9, 10] # 夹爪是关节 9 和 10
    
    for i, joint_idx in enumerate(joint_indices_arm):
        p.resetJointState(robotId, joint_idx, home_joint_positions[i])
        
    for i, joint_idx in enumerate(joint_indices_gripper):
        p.resetJointState(robotId, joint_idx, home_joint_positions[i + len(joint_indices_arm)])
    
    print("环境设置完毕。")
    # 【修改】函数现在返回五个ID/列表
    return robotId, object_id, tray_id, dummyId, interferer_joint_indices