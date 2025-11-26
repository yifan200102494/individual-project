import pybullet as p
import time
import environmen  # 你的环境文件
import obstacle    # 你的障碍物文件
import control     # 你的控制文件

if __name__ == "__main__":
    # 1. 搭建基础环境
    robot_id, tray_id, cube_id = environmen.setup_environment()

    # 2. 初始化障碍物 (它会显示在场景里，但我们后面不调用它的 update)
    dynamic_obs = obstacle.DynamicObstacle()

    # 3. 初始化机器人控制器
    controller = control.RobotController(robot_id)

    print("仿真开始：障碍物保持静止，专注于测试机械臂抓取...")
    time.sleep(1) 

    # 4. 执行抓取任务
    # [关键修改] 去掉了 loop_callback=dynamic_obs.update
    # 这样机械臂在动的时候，不会触发障碍物的移动逻辑
    controller.execute_pick_and_place(
        cube_id, 
        tray_id, 
        loop_callback=None 
    )

    # 5. 任务结束后保持显示
    while True:
        # 这里也不调用 dynamic_obs.update()，让它彻底静止
        p.stepSimulation()
        time.sleep(1./240.)