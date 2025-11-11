"""
仿真管理器模块
负责管理整个仿真的主循环和组件协调
"""

import pybullet as p
import time
from config import SIMULATION_FREQUENCY, DEBUG_PRINT_INTERVAL


class SimulationManager:
    """
    仿真管理器
    协调机械臂控制器、障碍臂控制器和安全监控器
    """
    
    def __init__(self, robot_controller, obstacle_controller, safety_monitor):
        """
        初始化仿真管理器
        
        Args:
            robot_controller: 机械臂控制器实例
            obstacle_controller: 障碍臂控制器实例
            safety_monitor: 安全监控器实例
        """
        self.robot_controller = robot_controller
        self.obstacle_controller = obstacle_controller
        self.safety_monitor = safety_monitor
        
        self.debug_counter = 0
        self.simulation_frequency = SIMULATION_FREQUENCY
        self.debug_interval = DEBUG_PRINT_INTERVAL
        
        print("仿真管理器初始化完成")
    
    def print_status(self, safety_status, distance):
        """
        打印当前状态信息
        
        Args:
            safety_status: 安全状态
            distance: 最短距离
        """
        print(f"  安全状态: {safety_status}, 最短距离: {distance:.4f} m, " +
              f"阻塞计时: {self.robot_controller.block_timer:.1f}, " +
              f"是否绕行: {self.robot_controller.is_replanning}, " +
              f"冷却: {self.robot_controller.replan_success_timer}")
    
    def run(self):
        """
        运行仿真主循环
        """
        print("\n仿真运行中，按 Ctrl+C 或关闭窗口退出...\n")
        print("机械臂将自动执行抓取-放置任务...\n")
        
        try:
            while True:
                # 1. 安全层: 获取安全状态和障碍物点
                safety_status, distance, obs_point = self.safety_monitor.get_safety_status()
                
                # 2. 灵活执行层: 将完整信息传递给机械臂控制器
                self.robot_controller.update(safety_status, obs_point)
                
                # 3. 更新障碍臂控制
                self.obstacle_controller.update()
                
                # 4. 打印调试信息
                self.debug_counter += 1
                if self.debug_counter >= self.debug_interval:
                    self.print_status(safety_status, distance)
                    self.debug_counter = 0
                
                # 5. 执行物理仿真
                p.stepSimulation()
                
                # 6. 控制仿真速度
                time.sleep(1.0 / self.simulation_frequency)
                
        except KeyboardInterrupt:
            print("\n用户中断")
    
    def cleanup(self):
        """
        清理资源
        """
        p.disconnect()
        print("仿真结束")

