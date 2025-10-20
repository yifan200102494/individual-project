import time

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
    joint_id = 0
    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=joint_id,
        controlMode=p.POSITION_CONTROL,
        targetPosition=1,
        force=100,
        maxVelocity=1  # This limits the speed for a smoother motion
    )
    # ------------------------------------

    # Simulate for 10 seconds to observe the movement and final position
    simulate(seconds=10)

    # clean up
    p.disconnect()
    print('program finished. bye.')


if __name__ == '__main__':
    main()
