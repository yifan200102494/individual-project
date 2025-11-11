"""
主运行文件
启动仿真环境并控制机械臂进行夹取-放置操作
"""

from environment_setup import setup_environment
from robot_controller import RobotController
from safety_monitor import SafetyMonitor
from simulation_manager import SimulationManager


def main():
    """
    主函数：设置环境并运行仿真
    """
    # 1. 设置环境
    robotId, objectId, trayId, obstacleArmId, obstacle_controller = setup_environment()
    
    # 2. 创建机械臂控制器
    robot_controller = RobotController(robotId, objectId, trayId, obstacleArmId)
    
    # 3. 创建安全监控器
    safety_monitor = SafetyMonitor(robotId, obstacleArmId)
    
    # 4. 创建仿真管理器并运行
    simulation_manager = SimulationManager(robot_controller, obstacle_controller, safety_monitor)
    simulation_manager.run()
    
    # 5. 清理资源
    simulation_manager.cleanup()


if __name__ == "__main__":
    main()