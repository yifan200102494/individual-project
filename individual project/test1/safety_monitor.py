"""
安全监控模块
负责监控机械臂之间的距离，并返回安全状态
"""

import pybullet as p
from config import STOP_DISTANCE, SLOW_DISTANCE


class SafetyMonitor:
    """
    安全监控器
    实现速度与分离监控 (SSM) 功能
    """
    
    def __init__(self, robot_id, obstacle_arm_id):
        """
        初始化安全监控器
        
        Args:
            robot_id: 主机械臂的ID
            obstacle_arm_id: 障碍臂的ID
        """
        self.robot_id = robot_id
        self.obstacle_arm_id = obstacle_arm_id
        self.stop_distance = STOP_DISTANCE
        self.slow_distance = SLOW_DISTANCE
        
        print("安全监控器初始化完成")
        print(f"  停止距离: {self.stop_distance} m")
        print(f"  减速距离: {self.slow_distance} m")
    
    def get_safety_status(self):
        """
        检查两个机械臂之间的最短距离，返回安全状态
        
        Returns:
            tuple: (status, distance, obstacle_point)
                   - status: 'GO', 'SLOW', 或 'STOP'
                   - distance: 最短距离 (米)
                   - obstacle_point: 障碍臂上最接近的点，如果没有则为 None
        """
        min_dist = float('inf')
        obstacle_point = None
        
        # 使用PyBullet的内置函数高效计算最短距离
        closest_points = p.getClosestPoints(
            self.robot_id, 
            self.obstacle_arm_id, 
            distance=1.0
        ) 
        
        if closest_points:
            # 过滤掉基座，只关心可动连杆 (link index > -1) 之间的距离
            moving_links_points = []
            for point in closest_points:
                linkA = point[3]  # robot_id 的 link 索引
                linkB = point[4]  # obstacle_arm_id 的 link 索引
                
                # 只要两个连杆索引都大于-1 (即都不是基座)，就认为是有效的
                if linkA > -1 and linkB > -1:
                    moving_links_points.append(point)

            if moving_links_points:
                # 如果找到了可动连杆之间的接近点，就在这些点中找最小值
                # 字段[8]是距离, 字段[6]是障碍物上的点
                best_point = min(moving_links_points, key=lambda p: p[8])
                min_dist = best_point[8]
                obstacle_point = best_point[6]
            else:
                # 如果可动连杆离得很远 (moving_links_points 为空)
                # 我们回退到使用原始列表中的最近点（很可能就是基座距离）
                # 这可以保证 min_dist 至少有一个安全的值
                best_point = min(closest_points, key=lambda p: p[8])
                min_dist = best_point[8]
                obstacle_point = best_point[6]

        # 根据距离返回安全状态
        if min_dist < self.stop_distance:
            return "STOP", min_dist, obstacle_point
        elif min_dist < self.slow_distance:
            return "SLOW", min_dist, obstacle_point
        else:
            return "GO", min_dist, obstacle_point
    
    def set_stop_distance(self, distance):
        """设置停止距离阈值"""
        self.stop_distance = distance
    
    def set_slow_distance(self, distance):
        """设置减速距离阈值"""
        self.slow_distance = distance

