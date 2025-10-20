import time
import numpy as np
import pybullet as p
import pybullet_data

import util


def simulate(steps=None, seconds=None, slow_down=True):
    """
    Wraps pybullet's stepSimulation function and allows some more control over duration.
    Will simulate for a number of steps, or number of seconds, whichever is reached first.
    If both are None, it will run indefinitely.

    :param steps: int, number of steps to simulate
    :param seconds: float, number of seconds to simulate
    :param slow_down: bool, if set to True will slow down the simulated time to be aligned to real time
    """
    dt = 1./240  # a single timestep is 1/240 seconds per default
    seconds_passed = 0.0
    steps_passed = 0
    start_time = time.time()

    while True:
        p.stepSimulation()
        steps_passed += 1
        seconds_passed += dt

        if slow_down:
            time_elapsed = time.time() - start_time
            wait_time = seconds_passed - time_elapsed
            time.sleep(max(wait_time, 0))
        if steps is not None and steps_passed > steps:
            break
        if seconds is not None and seconds_passed > seconds:
            break
def move_to_joint_pos(robot_id, target_joint_pos):
    """
    Move robot arm to target joint positions with synchronised PTP control.
    """
    current_pos = np.array(util.get_arm_joint_pos(robot_id))
    target_pos = np.array(target_joint_pos)
    distances = np.abs(target_pos - current_pos)

    max_velocity = 1.0
    max_dist = np.max(distances)
    if max_dist < 1e-6:
        return  # already at target

    joint_velocities = (distances / max_dist) * max_velocity

    # Set each joint target
    for joint_id in range(len(target_pos)):
        p.setJointMotorControl2(
            robot_id,
            joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_pos[joint_id],
            force=100,
            maxVelocity=joint_velocities[joint_id]
        )

    # Run simulation until reached
    while True:
        simulate(seconds=0.01)
        current_pos = np.array(util.get_arm_joint_pos(robot_id))
        if np.allclose(current_pos, target_pos, atol=0.0001):
            break

def main():
    # connect to pybullet with a graphical user interface
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # basic configuration
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # allows us to load plane, robots, etc.
    plane_id = p.loadURDF('plane.urdf')  # function returns an ID for the loaded body

    # load a robot
    robot_id = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)
    ROBOT_HOME_CONFIG = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
    num_joints = p.getNumJoints(robot_id)
    for i in range(len(ROBOT_HOME_CONFIG)):
        
        p.resetJointState(robot_id, i, ROBOT_HOME_CONFIG[i])
    
    # num_joints = p.getNumJoints(robot_id)
    
    # print("Disabling default joint motors.")
    # for i in range(num_joints):
    #     p.setJointMotorControl2(
    #         bodyUniqueId=robot_id,
    #         jointIndex=i,
    #         controlMode=p.VELOCITY_CONTROL,
    #         force=0
    #     )
    # print("Applying velocity control to the first joint.")
    # joint_id = 0
    # p.setJointMotorControl2(
    #     bodyUniqueId=robot_id,
    #     jointIndex=joint_id,
    #     controlMode=p.VELOCITY_CONTROL,
    #     targetVelocity=1,
    #     force=100
    # )
    print('******************************')
    input('press enter to start simulation')
    # joint_id = 0
    # target_position = 1.0

    # # 1. Set the motor to move towards the target
    # p.setJointMotorControl2(
    #     bodyUniqueId=robot_id,
    #     jointIndex=joint_id,
    #     controlMode=p.VELOCITY_CONTROL,
    #     targetVelocity=1,  # Positive velocity to move towards the target
    #     force=100
    # )
    # # 2. Loop until the joint reaches the target position
    # while True:
    #     # Get the current state of the joint
    #     # p.getJointState returns a tuple, the first element [0] is the position.
    #     current_position = p.getJointState(robot_id, joint_id)[0]

    #     # Break the loop if the target is reached
    #     if current_position >= target_position:
    #         break

    #     # 3. Simulate one step
    #     p.stepSimulation()
    #     time.sleep(1./240.) # Slow down simulation to real-time
    # # 4. Stop the motor
    # final_position = p.getJointState(robot_id, joint_id)[0]
    # print(f"Target reached! Final position: {final_position:.4f}")
    # p.setJointMotorControl2(
    #     bodyUniqueId=robot_id,
    #     jointIndex=joint_id,
    #     controlMode=p.VELOCITY_CONTROL,
    #     targetVelocity=0,  # Set velocity to 0 to stop
    #     force=100
    # )
    
    # # Keep the simulation running for a few seconds to observe the result
    # print("Motor stopped. Holding position for a few seconds.")
    # for _ in range(240 * 5): # Simulate for 5 more seconds
    #     p.stepSimulation()
    #     time.sleep(1./240.)
    # # ---------------------------------------------
    pose1 = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]  # home
    pose2 = [0.5, -0.5, 0.3, -2.0, 0.2, 1.0, 0.6]
    pose3 = [-0.3, -0.9, 0.2, -2.2, -0.3, 1.2, 0.5]

    # 循环移动
    start_time = time.time()
    while (time.time() - start_time) < 10:
        move_to_joint_pos(robot_id, pose1)
        # 检查是否已经超过10秒，如果超过则提前退出循环
        if (time.time() - start_time) >= 10:
            break
        move_to_joint_pos(robot_id, pose2)
        if (time.time() - start_time) >= 10:
            break
        move_to_joint_pos(robot_id, pose3)

    # clean up
    p.disconnect()
    print('program finished. bye.')


if __name__ == '__main__':
    main()
