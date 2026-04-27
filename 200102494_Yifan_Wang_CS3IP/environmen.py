import pybullet as p
import pybullet_data
import time
import math

def setup_environment():
    # GUI mode shows the simulation window for interactive runs.
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")

    robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    # Tray position is inside the Panda workspace and aligned with the
    # pick-and-place target used by the controller.
    trayId = p.loadURDF("tray/traybox.urdf", [0.5, 0.4, 0], globalScaling=0.8)

    cubeStartPos = [0.5, -0.3, 0.04]
    cubeStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    cubeId = p.loadURDF("cube_small.urdf", cubeStartPos, cubeStartOrn, globalScaling=1.3)
    p.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

    # Default home pose — elbow up, wrist neutral. Matches the rp values used
    # by the IK solver as the rest configuration, so it doesn't fight us
    # during the first move.
    ready_poses = [0, -math.pi/4, 0, -math.pi/2, 0, math.pi/3, 0]
    for i in range(7):
        p.resetJointState(robotId, i, ready_poses[i])

    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=30,
        cameraPitch=-40,
        cameraTargetPosition=[0.3, 0, 0]
    )

    print("Environment ready: tray pulled closer to ensure the arm can reach it.")
    return robotId, trayId, cubeId

if __name__ == "__main__":
    setup_environment()
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
