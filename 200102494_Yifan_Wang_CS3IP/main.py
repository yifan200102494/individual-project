import pybullet as p
import time
import environmen
import obstacle
import control

if __name__ == "__main__":
    robot_id, tray_id, cube_id = environmen.setup_environment()
    dynamic_obs = obstacle.DynamicObstacle()
    controller = control.RobotController(robot_id, tray_id)

    # Drive obstacle motion from the controller's per-step hook so obstacle,
    # physics, and IK updates stay synchronized.
    controller.sim_step_callback = dynamic_obs.update

    print("Simulation started: visual perception system is online...")
    time.sleep(1)

    controller.execute_pick_and_place(cube_id, tray_id)

    # Keep stepping for a few seconds after the task ends so the final GUI
    # state remains visible.
    hold_seconds = 1
    print(f"Task complete; auto-closing the simulation window in {hold_seconds} seconds...")
    for _ in range(hold_seconds * 240):
        p.stepSimulation()
        dynamic_obs.update()
        time.sleep(1. / 240.)
    p.disconnect()
    print("Simulation window closed")
