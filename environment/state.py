"""
状态定义与处理
"""
import numpy as np
from config.environment_config import T_MAX, DELTA, T_WORK_V, T_WORK_D


class State:
    """环境状态类"""
    
    def __init__(self, current_time, new_request, fleet_status, planned_routes):
        """
        初始化状态
        
        Args:
            current_time: 当前时间
            new_request: 新到达的请求（如果没有则为None）
            fleet_status: 车队状态字典 {vehicle_id: 0/1, drone_id: 0/1}
                         0表示在仓库，1表示在途中
            planned_routes: 计划路线字典
                           {vehicle_id: [(request, est_time), ...],
                            drone_id: [(request, est_time), ...]}
        """
        self.current_time = current_time
        self.new_request = new_request
        self.fleet_status = fleet_status
        self.planned_routes = planned_routes
    
    def get_upper_agent_state(self, earliest_drone_time, earliest_vehicle_time, 
                              most_urgent_remain_time, pending_requests_count=0):
        """
        获取上层智能体的状态特征（5维 - 新增RBU维度）
        
        Args:
            earliest_drone_time: 无人机队伍中最早可用的时间
            earliest_vehicle_time: 车辆队伍中最早可用的时间
            most_urgent_remain_time: 当前待调度车辆计划路线中最紧迫请求的剩余交付时间
            pending_requests_count: 当前待调度车辆路线中的请求总数（RBU维度）
        
        Returns:
            np.array: 5维状态向量 [t_k, t_k^AD, t_k^AV, t_k^remain, n_requests]
        
        归一化说明（与论文一致）：
        - t_k: 当前时间归一化
        - t_k^AD: 与当前时间做差，转化为相对等待时长
        - t_k^AV: 与当前时间做差，转化为相对等待时长
        - t_k^remain: 剩余时间归一化
        - n_requests: 待调度请求数归一化（假设最多20个请求）
        """
        # 计算相对等待时长（与当前时间做差）
        drone_wait_time = max(0.0, earliest_drone_time - self.current_time)
        vehicle_wait_time = max(0.0, earliest_vehicle_time - self.current_time)
        
        # ✅ 修复：使用T_WORK归一化，因为current_time现在可以达到600分钟
        T_WORK_MAX = max(T_WORK_V, T_WORK_D)
        
        # ✅ 改进：增强RBU特征的权重，使用更小的归一化因子
        # 让RBU特征在[0, 2.0]范围内，而不是[0, 1.0]
        # 这样当有多个请求时，特征值会更显著
        rbu_feature = min(pending_requests_count / 10.0, 2.0)  # 改为除以10，最大2.0
        
        return np.array([
            self.current_time / T_WORK_MAX,  # 归一化当前时间（使用工作时长上限）
            min(drone_wait_time / T_WORK_MAX, 1.0),  # 无人机相对等待时长（归一化）
            min(vehicle_wait_time / T_WORK_MAX, 1.0),  # 车辆相对等待时长（归一化）
            most_urgent_remain_time / DELTA if most_urgent_remain_time > 0 else 0.0,
            rbu_feature  # 待调度请求数（增强特征权重）
        ], dtype=np.float32)
    
    def get_lower_agent_state(self, request, earliest_drone_time, earliest_vehicle_time,
                             vehicle_feasible, vehicle_insert_cost, drone_feasible):
        """
        获取下层智能体的状态特征（7维）
        
        Args:
            request: 当前请求
            earliest_drone_time: 无人机最早可用时间
            earliest_vehicle_time: 车辆最早可用时间
            vehicle_feasible: 车辆服务可行性 (0/1)
            vehicle_insert_cost: 将请求插入车辆路线的时间成本
            drone_feasible: 无人机服务可行性 (0/1)
        
        Returns:
            np.array: 7维状态向量
        
        归一化说明（与论文一致）：
        - t_i^a: 到达时间归一化
        - Ed_{c_i}: 距离被速度除后变为时间成本（归一化）
        - t_k^AD, t_k^AV: 与当前时间做差，转化为相对等待时长
        - 其他特征按原方式归一化
        """
        from config.environment_config import T_MAX, DELTA, WAREHOUSE_X, WAREHOUSE_Y, V_D
        from utils.distance import euclidean_distance, travel_time_drone
        
        # 计算请求与仓库的距离
        distance = euclidean_distance(
            request.x, request.y, 
            WAREHOUSE_X, WAREHOUSE_Y
        )
        
        # 距离归一化：被速度除后变为时间成本（与论文一致）
        # 无人机飞行时间（分钟）
        drone_travel_time = travel_time_drone(distance, V_D)
        # 归一化：除以最大可能时间（假设最大距离20km，速度40km/h，约30分钟）
        max_drone_time = travel_time_drone(20.0, V_D)  # 约30分钟
        normalized_distance_time = min(drone_travel_time / max_drone_time, 1.0)
        
        # 计算相对等待时长（与当前时间做差）
        current_time = self.current_time
        drone_wait_time = max(0.0, earliest_drone_time - current_time)
        vehicle_wait_time = max(0.0, earliest_vehicle_time - current_time)
        
        # ✅ 修复：使用T_WORK归一化等待时间
        T_WORK_MAX = max(T_WORK_V, T_WORK_D)
        
        # 处理vehicle_insert_cost，避免Inf值
        if vehicle_insert_cost == float('inf') or np.isinf(vehicle_insert_cost):
            normalized_cost = 1.0  # 如果不可行，归一化为1.0
        elif vehicle_insert_cost > 0:
            normalized_cost = min(vehicle_insert_cost / DELTA, 1.0)  # 限制在[0, 1]
        else:
            normalized_cost = 0.0
        
        return np.array([
            request.arrival_time / T_MAX,  # 归一化到达时间（请求始终在0-T_MAX内到达）
            normalized_distance_time,  # 距离转换为时间成本后归一化（与论文一致）
            min(drone_wait_time / T_WORK_MAX, 1.0),  # 无人机相对等待时长（归一化）
            min(vehicle_wait_time / T_WORK_MAX, 1.0),  # 车辆相对等待时长（归一化）
            float(vehicle_feasible),
            normalized_cost,
            float(drone_feasible)
        ], dtype=np.float32)
