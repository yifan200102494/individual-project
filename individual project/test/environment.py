import pybullet as p
import pybullet_data

def setup_environment():
    """
    初始化PyBullet环境, 加载所有物体。
    【新增】添加了一个静态的干扰障碍物。
    Returns:
        tuple: (robotId, objectId, trayId, dummyId)
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

    # --- 【新增】加载一个静态的胶囊体作为障碍物 ---
    # 将它放置在方块和托盘之间的路径上
    dummy_pos = [0.5, 0.1, 0.35]
    # 使用 p.createCollisionShape 和 p.createVisualShape 来创建一个程序化的物体
    dummy_visual_id = p.createVisualShape(p.GEOM_CAPSULE, radius=0.06, length=0.5, rgbaColor=[1, 0, 0, 1])
    dummy_coll_id = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.06, height=0.5)
    # baseMass = 0 表示该物体是静态的, 不会因重力或碰撞而移动
    dummyId = p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=dummy_coll_id,
                                 baseVisualShapeIndex=dummy_visual_id,
                                 basePosition=dummy_pos)
    print("已加载静态干扰障碍物。")

    # 【修正】这个列表必须包含所有可动关节 (7个手臂 + 2个夹爪)
    # 关节 7 和 8 是固定(FIXED)的，不需要设置
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
    # 函数现在返回四个ID
    return robotId, object_id, tray_id, dummyId

