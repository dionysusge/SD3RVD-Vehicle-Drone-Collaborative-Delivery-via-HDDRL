"""
车辆路径规划（最低成本插入法）
"""
from config.environment_config import (
    WAREHOUSE_X, WAREHOUSE_Y, T_LOAD_V, T_DROP_V, V_E
)
from utils.distance import road_distance, travel_time_vehicle
from heuristics.feasibility_check import (
    check_vehicle_feasibility
)


def find_best_insertion(request, vehicle_route, current_time, vehicle_available_time):
    """
    使用最低成本插入法为车辆路线找到最佳插入位置
    
    Args:
        request: 要插入的请求
        vehicle_route: 当前车辆路线 [(request, est_time), ...]
        current_time: 当前时间
        vehicle_available_time: 车辆可用时间
    
    Returns:
        tuple: (是否可行, 最佳插入位置, 插入后的路线, 送达时间)
    """
    feasible, min_cost = check_vehicle_feasibility(
        request, vehicle_route, current_time, vehicle_available_time
    )
    
    if not feasible:
        return False, None, None, None
    
    # 找到成本最小的插入位置
    min_cost = float('inf')
    best_position = 0
    best_route = None
    delivery_time = None
    
    for i in range(len(vehicle_route) + 1):
        # 计算插入成本
        cost = _calculate_insertion_cost(
            request, vehicle_route, i, current_time, vehicle_available_time
        )
        
        if cost < min_cost:
            # 检查是否满足时间约束
            est_delivery = _calculate_delivery_time_local(
                request, vehicle_route, i, current_time, vehicle_available_time
            )
            if est_delivery <= request.deadline:
                min_cost = cost
                best_position = i
                delivery_time = est_delivery
    
    if best_position is None:
        return False, None, None, None
    
    # 使用 update_vehicle_route 来确保后续请求的时间被正确重新计算
    best_route = update_vehicle_route(
        vehicle_route, request, best_position, delivery_time,
        current_time, vehicle_available_time
    )
    
    return True, best_position, best_route, delivery_time


def update_vehicle_route(vehicle_route, request, insertion_position, delivery_time, 
                        current_time, vehicle_available_time):
    """
    更新车辆路线
    
    Args:
        vehicle_route: 当前路线
        request: 插入的请求
        insertion_position: 插入位置
        delivery_time: 预计送达时间
        current_time: 当前时间
        vehicle_available_time: 车辆可用时间
    
    Returns:
        list: 更新后的路线
    """
    new_route = vehicle_route.copy()
    new_route.insert(insertion_position, (request, delivery_time))
    
    # 重新计算后续请求的预计时间
    return _recalculate_route_times(new_route, insertion_position, current_time, vehicle_available_time)


def _calculate_insertion_cost(request, route, position, current_time, vehicle_available_time):
    """计算插入成本"""
    if len(route) == 0:
        distance = road_distance(
            WAREHOUSE_X, WAREHOUSE_Y, request.x, request.y
        )
        travel_time = travel_time_vehicle(distance, V_E)
        return T_LOAD_V + travel_time + T_DROP_V
    
    if position == 0:
        prev_pos = (WAREHOUSE_X, WAREHOUSE_Y)
        next_pos = route[0][0].x, route[0][0].y
    elif position == len(route):
        prev_pos = route[-1][0].x, route[-1][0].y
        next_pos = (WAREHOUSE_X, WAREHOUSE_Y)
    else:
        prev_pos = route[position - 1][0].x, route[position - 1][0].y
        next_pos = route[position][0].x, route[position][0].y
    
    original_distance = road_distance(prev_pos[0], prev_pos[1], next_pos[0], next_pos[1])
    new_distance = (road_distance(prev_pos[0], prev_pos[1], request.x, request.y) +
                    road_distance(request.x, request.y, next_pos[0], next_pos[1]))
    extra_distance = new_distance - original_distance
    
    extra_time = travel_time_vehicle(extra_distance, V_E) + T_DROP_V
    return extra_time


def _calculate_delivery_time_local(request, route, position, current_time, vehicle_available_time):
    """计算插入请求后的实际送达时间"""
    from config.environment_config import WAREHOUSE_X, WAREHOUSE_Y
    
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


def _recalculate_route_times(route, start_from=0, current_time=None, vehicle_available_time=None):
    """
    重新计算路线中从指定位置开始的预计时间
    
    Args:
        route: 车辆路线 [(request, est_time), ...]
        start_from: 从哪个位置开始重新计算（插入位置）
        current_time: 当前时间（如果为None，使用已插入请求的时间估算）
        vehicle_available_time: 车辆可用时间（如果为None，使用已插入请求的时间估算）
    
    Returns:
        list: 更新后的路线，包含重新计算的预计送达时间
    """
    from config.environment_config import WAREHOUSE_X, WAREHOUSE_Y
    
    if len(route) == 0 or start_from >= len(route):
        return route
    
    # 创建新路线副本
    new_route = route.copy()
    
    # 确定起始位置和时间
    if start_from == 0:
        # 从仓库出发
        start_pos = (WAREHOUSE_X, WAREHOUSE_Y)
        if vehicle_available_time is not None and current_time is not None:
            # 使用实际的车辆可用时间和当前时间
            start_time = max(vehicle_available_time, current_time) + T_LOAD_V
        else:
            # 如果没有提供，使用插入位置请求的送达时间减去行驶和卸货时间作为起始点
            first_request, first_delivery_time = new_route[0]
            # 估算出发时间：送达时间 - 行驶时间 - 卸货时间
            distance = road_distance(WAREHOUSE_X, WAREHOUSE_Y, first_request.x, first_request.y)
            travel_time = travel_time_vehicle(distance, V_E)
            start_time = first_delivery_time - travel_time - T_DROP_V - T_LOAD_V
    else:
        # 从前一个请求的位置出发
        prev_request, prev_delivery_time = new_route[start_from - 1]
        start_pos = (prev_request.x, prev_request.y)
        start_time = prev_delivery_time + T_DROP_V  # 前一个请求送达后继续
    
    # 从 start_from 位置开始，重新计算后续所有请求的预计送达时间
    current_pos = start_pos
    current_time_est = start_time
    
    for i in range(start_from, len(new_route)):
        req, _ = new_route[i]
        
        # 计算从当前位置到请求位置的行驶时间
        distance = road_distance(
            current_pos[0], current_pos[1], 
            req.x, req.y
        )
        travel_time = travel_time_vehicle(distance, V_E)
        
        # 计算新的送达时间：当前时间 + 行驶时间 + 卸货时间
        new_delivery_time = current_time_est + travel_time + T_DROP_V
        
        # 更新路线中的预计时间
        new_route[i] = (req, new_delivery_time)
        
        # 更新当前位置和时间
        current_pos = (req.x, req.y)
        current_time_est = new_delivery_time
    
    return new_route
