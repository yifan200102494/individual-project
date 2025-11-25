import pybullet as p
import pybullet_data
import time
import math

def setup_environment():
    # 1. 初始化仿真界面
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # 2. 加载地面
    p.loadURDF("plane.urdf")

    # 3. 加载机器人
    robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    # 4. 加载托盘 (位置不变，左前方)
    trayPos = [0.6, 0.5, 0]
    trayId = p.loadURDF("tray/traybox.urdf", trayPos, globalScaling=0.8)

    # ==========================================
    # 5. 加载目标物体 (位置再次调整)
    # ==========================================
    # 修改：y 从 -0.2 改为 -0.4，距离翻倍
    cubeStartPos = [0.5, -0.6, 0.04] 
    cubeStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    
    # 尺寸保持之前的 2/3 (1.3)
    cubeId = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrn, globalScaling=1.3)
    
    # 颜色保持红色
    p.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

    # 6. 初始化机器人姿态
    ready_poses = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]
    for i in range(7):
        p.resetJointState(robotId, i, ready_poses[i])

    # 7. 调整相机视角 (视角稍微拉远一点，因为现在的场景变宽了)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,  # 稍微拉远一点 (1.2 -> 1.5)
        cameraYaw=30, 
        cameraPitch=-40, 
        cameraTargetPosition=[0.3, 0, 0] # 镜头对准中心
    )
    
    print("========================================")
    print("环境更新完毕：方块已移至右侧远端 (y=-0.4)。")
    print("========================================")

    return robotId, trayId, cubeId

if __name__ == "__main__":
    setup_environment()
    while True:
        p.stepSimulation()
        time.sleep(1./240.)