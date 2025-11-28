"""
可行性检查模块
"""
from config.environment_config import (
    WAREHOUSE_X, WAREHOUSE_Y, DELTA, T_LOAD_V, T_DROP_V,
    T_LOAD_D, T_DROP_D, T_CHARGE, V_E, V_D
)
from utils.distance import (
    euclidean_distance, road_distance,
    travel_time_vehicle, travel_time_drone
)


def check_vehicle_feasibility(request, vehicle_route, current_time, vehicle_available_time):
    """
    检查车辆是否可以服务该请求
    
    Args:
        request: 请求对象
        vehicle_route: 车辆当前路线 [(request, est_time), ...]
        current_time: 当前时间
        vehicle_available_time: 车辆可用时间
    
    Returns:
        tuple: (是否可行, 插入成本)
    """
    if vehicle_available_time > request.deadline:
        return False, float('inf')
    
    # 计算将请求插入到路线各个位置的成本
    min_cost = float('inf')
    best_insertion = None
    
    # 尝试插入到每个位置
    for i in range(len(vehicle_route) + 1):
        # 计算插入成本
        cost = _calculate_insertion_cost(
            request, vehicle_route, i, 
            current_time, vehicle_available_time
        )
        if cost < min_cost:
            min_cost = cost
            best_insertion = i
    
    # 检查是否能在截止时间前送达
    if best_insertion is not None:
        # 计算实际送达时间
        delivery_time = _calculate_delivery_time(
            request, vehicle_route, best_insertion,
            current_time, vehicle_available_time
        )
        feasible = delivery_time <= request.deadline
        return feasible, min_cost if feasible else float('inf')
    
    return False, float('inf')


def check_drone_feasibility(request, current_time, drone_available_time):
    """
    检查无人机是否可以服务该请求
    
    Args:
        request: 请求对象
        current_time: 当前时间
        drone_available_time: 无人机可用时间
    
    Returns:
        bool: 是否可行
    """
    if drone_available_time > request.deadline:
        return False
    
    # 计算无人机服务时间
    # 1. 从仓库到客户
    distance = euclidean_distance(
        request.x, request.y, WAREHOUSE_X, WAREHOUSE_Y
    )
    travel_time = travel_time_drone(distance, V_D)
    
    # 2. 装载 + 飞行 + 卸货 + 返回 + 充电
    total_time = T_LOAD_D + travel_time + T_DROP_D + travel_time + T_CHARGE
    
    delivery_time = max(drone_available_time, current_time) + T_LOAD_D + travel_time + T_DROP_D
    
    return delivery_time <= request.deadline


def _calculate_insertion_cost(request, route, position, current_time, vehicle_available_time):
    """计算将请求插入到指定位置的成本（时间增量）"""
    if len(route) == 0:
        # 空路线，直接计算往返时间
        distance = road_distance(
            WAREHOUSE_X, WAREHOUSE_Y, request.x, request.y
        )
        travel_time = travel_time_vehicle(distance, V_E)
        return T_LOAD_V + travel_time + T_DROP_V
    
    # 计算插入后的时间增量
    if position == 0:
        # 插入到第一个位置
        prev_pos = (WAREHOUSE_X, WAREHOUSE_Y)
        next_pos = route[0][0].x, route[0][0].y
    elif position == len(route):
        # 插入到最后一个位置
        prev_pos = route[-1][0].x, route[-1][0].y
        next_pos = (WAREHOUSE_X, WAREHOUSE_Y)
    else:
        prev_pos = route[position - 1][0].x, route[position - 1][0].y
        next_pos = route[position][0].x, route[position][0].y
    
    # 计算额外距离
    original_distance = road_distance(prev_pos[0], prev_pos[1], next_pos[0], next_pos[1])
    new_distance = (road_distance(prev_pos[0], prev_pos[1], request.x, request.y) +
                    road_distance(request.x, request.y, next_pos[0], next_pos[1]))
    extra_distance = new_distance - original_distance
    
    extra_time = travel_time_vehicle(extra_distance, V_E) + T_DROP_V
    return extra_time


def _calculate_delivery_time(request, route, position, current_time, vehicle_available_time):
    """计算插入请求后的实际送达时间"""
    start_time = max(vehicle_available_time, current_time)
    current_pos = (WAREHOUSE_X, WAREHOUSE_Y)
    current_time_est = start_time + T_LOAD_V
    
    # 遍历到插入位置
    for i in range(position):
        req, _ = route[i]
        distance = road_distance(
            current_pos[0], current_pos[1], req.x, req.y
        )
        travel_time = travel_time_vehicle(distance, V_E)
        current_time_est += travel_time + T_DROP_V
        current_pos = (req.x, req.y)
    
    # 到达插入的请求位置
    distance = road_distance(
        current_pos[0], current_pos[1], request.x, request.y
    )
    travel_time = travel_time_vehicle(distance, V_E)
    delivery_time = current_time_est + travel_time + T_DROP_V
    
    return delivery_time
