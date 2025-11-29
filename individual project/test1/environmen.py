import pybullet as p
import pybullet_data
import time
import math

def setup_environment():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    
    # 加载机器人
    robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    # ==========================================
    # 【修改点】托盘位置调整
    # 原来是 [0.6, 0.5, 0] (太远了，直线距离约0.78m)
    # 改为   [0.4, 0.4, 0] (直线距离约0.56m)
    # 这样即使抬高到 Z=0.5，总距离也就 0.75m，完全够得着
    # ==========================================
    trayPos = [0.5, 0.4, 0] 
    trayId = p.loadURDF("tray/traybox.urdf", trayPos, globalScaling=0.8)

    # 加载目标物体
    cubeStartPos = [0.5, -0.3, 0.04]
    cubeStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    cubeId = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrn, globalScaling=1.3)
    p.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

    ready_poses = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]
    for i in range(7):
        p.resetJointState(robotId, i, ready_poses[i])

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5, 
        cameraYaw=30, 
        cameraPitch=-40, 
        cameraTargetPosition=[0.3, 0, 0]
    )
    
    print("环境更新完毕：托盘已拉近，确保机械臂可以触达。")
    return robotId, trayId, cubeId

if __name__ == "__main__":
    setup_environment()
    while True:
        p.stepSimulation()
        time.sleep(1./240.)