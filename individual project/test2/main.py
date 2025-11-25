import pybullet as p
import time
import environmen  # 您的环境文件
import obstacle    # 刚刚新建的障碍物文件

if __name__ == "__main__":
    # 1. 搭建基础环境
    robot_id, tray_id, cube_id = environmen.setup_environment()

    # 2. 初始化动态障碍物 (一行代码搞定)
    # 这会在场景里生成那个红球
    dynamic_obs = obstacle.DynamicObstacle()

    print("仿真开始：请观察红色障碍物随机闯入...")

    # 3. 仿真主循环
    while True:
        # --- 让障碍物动起来 ---
        dynamic_obs.update()
        
        # --- (未来) 在这里写避障算法 ---
        # robot_pos = p.getLinkState(robot_id, 11)[0]
        # obs_pos = dynamic_obs.get_position()
        # if distance(robot_pos, obs_pos) < 0.2:
        #     STOP or RETREAT
        
        p.stepSimulation()
        time.sleep(1./240.)