import pybullet as p
import time
import environmen
import obstacle
import control

if __name__ == "__main__":
    # 1. 搭建环境
    robot_id, tray_id, cube_id = environmen.setup_environment()

    # 2. 初始化动态障碍物
    dynamic_obs = obstacle.DynamicObstacle()

    # 3. 初始化控制器
    controller = control.RobotController(robot_id)
    
    # 【核心修改】将障碍物的 update 方法绑定到控制器的回调中
    # 这样控制器在每次 stepSimulation 时，都会顺便让障碍物动一下
    controller.sim_step_callback = dynamic_obs.update 

    print("仿真开始：视觉感知系统已启动...")
    time.sleep(1) 

    # 4. 执行任务
    # 现在障碍物会在这个函数执行期间正常移动了
    controller.execute_pick_and_place(cube_id, tray_id)

    # 5. 任务结束后继续维持仿真
    while True:
        p.stepSimulation()
        dynamic_obs.update()
        time.sleep(1./240.)