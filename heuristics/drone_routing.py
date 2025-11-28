"""
无人机路径规划（FIFO策略）
"""
from config.environment_config import (
    WAREHOUSE_X, WAREHOUSE_Y, T_LOAD_D, T_DROP_D, T_CHARGE, V_D, DRONE_CAPACITY
)
from utils.distance import euclidean_distance, travel_time_drone
from heuristics.feasibility_check import check_drone_feasibility


def assign_drone_fifo(request, drone_queue, current_time):
    """
    使用FIFO策略将请求分配给最早可用的无人机
    
    Args:
        request: 请求对象
        drone_queue: 无人机队列，每个元素为 (drone_id, available_time)
        current_time: 当前时间
    
    Returns:
        tuple: (是否成功分配, 分配的无人机ID, 预计送达时间, 无人机下次可用时间)
    """
    if len(drone_queue) == 0:
        return False, None, None, None
    
    # 找到最早可用的无人机
    drone_queue_sorted = sorted(drone_queue, key=lambda x: x[1])
    drone_id, drone_available_time = drone_queue_sorted[0]
    
    # 检查可行性
    feasible = check_drone_feasibility(request, current_time, drone_available_time)
    
    if not feasible:
        return False, None, None, None
    
    # 计算服务时间
    distance = euclidean_distance(
        request.x, request.y, WAREHOUSE_X, WAREHOUSE_Y
    )
    travel_time = travel_time_drone(distance, V_D)
    
    # 计算送达时间
    start_time = max(drone_available_time, current_time)
    delivery_time = start_time + T_LOAD_D + travel_time + T_DROP_D
    
    # 计算无人机下次可用时间（返回仓库 + 充电）
    next_available_time = delivery_time + travel_time + T_CHARGE
    
    return True, drone_id, delivery_time, next_available_time


def update_drone_schedule(drone_schedule, drone_id, request, delivery_time, next_available_time):
    """
    更新无人机调度计划
    
    Args:
        drone_schedule: 无人机调度字典 {drone_id: [(request, delivery_time, next_available), ...]}
        drone_id: 无人机ID
        request: 请求对象
        delivery_time: 送达时间
        next_available_time: 下次可用时间
    
    Returns:
        dict: 更新后的调度
    """
    if drone_id not in drone_schedule:
        drone_schedule[drone_id] = []
    
    drone_schedule[drone_id].append((request, delivery_time, next_available_time))
    return drone_schedule
