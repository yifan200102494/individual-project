
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
    
    controller = control.RobotController(robot_id, tray_id)
    
    # 绑定回调
    controller.sim_step_callback = dynamic_obs.update 

    print("仿真开始：视觉感知系统已启动...")
    time.sleep(1) 

    # 4. 执行任务
    controller.execute_pick_and_place(cube_id, tray_id)

    # 5. 任务结束后维持仿真 3 秒以便观察，然后自动关闭窗口
    hold_seconds = 3
    print(f"任务完成，{hold_seconds} 秒后自动关闭仿真窗口...")
    for _ in range(hold_seconds * 240):
        p.stepSimulation()
        dynamic_obs.update()
        time.sleep(1./240.)
    p.disconnect()
    print("仿真窗口已关闭")