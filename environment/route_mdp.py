"""
Route-based MDP 核心环境
"""
import numpy as np
from config.environment_config import (
    T_MAX, DELTA, V_E, V_D, T_LOAD_V, T_DROP_V, T_LOAD_D, T_DROP_D,
    T_CHARGE, WAREHOUSE_X, WAREHOUSE_Y, VEHICLE_WAIT_INTERVAL,
    T_WORK_V, T_WORK_D
)
from environment.request_generator import RequestGenerator
from environment.state import State
from environment.action import Action
from environment.reward import RewardCalculator
from utils.rbu import RequestBuffer
from utils.distance import road_distance, euclidean_distance, travel_time_vehicle, travel_time_drone
from heuristics.vehicle_routing import find_best_insertion
from heuristics.drone_routing import assign_drone_fifo
from heuristics.feasibility_check import check_vehicle_feasibility, check_drone_feasibility


class RouteMDP:
    """Route-based MDP 环境"""
    
    def __init__(self, num_vehicles=5, num_drones=5, time_dist='homo', 
                 space_dist='normal', lambda_rate=0.5):
        """
        初始化环境
        
        Args:
            num_vehicles: 车辆数量
            num_drones: 无人机数量
            time_dist: 时间分布类型
            space_dist: 空间分布类型
            lambda_rate: 泊松过程参数
        """
        self.num_vehicles = num_vehicles
        self.num_drones = num_drones
        
        # 生成请求序列
        self.request_generator = RequestGenerator(time_dist, space_dist, lambda_rate)
        self.requests = self.request_generator.generate_requests()
        self.request_index = 0
        
        # 车队状态
        self.vehicle_status = {i: 0 for i in range(num_vehicles)}  # 0=仓库, 1=途中
        self.drone_status = {i: 0 for i in range(num_drones)}
        
        # 车辆和无人机的可用时间
        self.vehicle_available_time = {i: 0.0 for i in range(num_vehicles)}
        self.drone_available_time = {i: 0.0 for i in range(num_drones)}
        
        # 计划路线
        self.vehicle_routes = {i: [] for i in range(num_vehicles)}  # [(request, est_time), ...]
        self.drone_schedules = {i: [] for i in range(num_drones)}  # [(request, delivery_time, next_available), ...]
        
        # 车辆等待状态
        self.vehicle_waiting = {i: False for i in range(num_vehicles)}
        self.vehicle_wait_until = {i: 0.0 for i in range(num_vehicles)}
        
        # 请求缓冲区
        self.rbu = RequestBuffer()
        
        # 环境状态
        self.current_time = 0.0
        self.episode_done = False
        
        # 统计信息
        self.total_requests = len(self.requests)
        self.delivered_requests = 0
        self.rejected_requests = 0
        
        # 奖励计算器
        self.reward_calc = RewardCalculator()
    
    def reset(self):
        """重置环境"""
        # 重新生成请求
        self.request_generator.request_id_counter = 0
        self.requests = self.request_generator.generate_requests()
        self.request_index = 0
        
        # 重置车队状态
        self.vehicle_status = {i: 0 for i in range(self.num_vehicles)}
        self.drone_status = {i: 0 for i in range(self.num_drones)}
        self.vehicle_available_time = {i: 0.0 for i in range(self.num_vehicles)}
        self.drone_available_time = {i: 0.0 for i in range(self.num_drones)}
        self.vehicle_routes = {i: [] for i in range(self.num_vehicles)}
        self.drone_schedules = {i: [] for i in range(self.num_drones)}
        self.vehicle_waiting = {i: False for i in range(self.num_vehicles)}
        self.vehicle_wait_until = {i: 0.0 for i in range(self.num_vehicles)}
        
        # 重置RBU
        self.rbu.clear()
        
        # 重置环境状态
        self.current_time = 0.0
        self.episode_done = False
        self.delivered_requests = 0
        self.rejected_requests = 0
        
        # 更新总请求数（重要！每次reset后请求数可能不同）
        self.total_requests = len(self.requests)
        
        # 获取初始状态
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态"""
        # 检查是否有新请求到达（只返回第一个未处理的请求）
        new_request = None
        if (self.request_index < len(self.requests) and 
            self.requests[self.request_index].arrival_time <= self.current_time):
            new_request = self.requests[self.request_index]
            # 注意：不在这里增加request_index，让step函数处理
        
        # 构建车队状态
        fleet_status = {}
        fleet_status.update({f'vehicle_{i}': self.vehicle_status[i] for i in range(self.num_vehicles)})
        fleet_status.update({f'drone_{i}': self.drone_status[i] for i in range(self.num_drones)})
        
        # 构建计划路线
        planned_routes = {
            'vehicles': self.vehicle_routes,
            'drones': self.drone_schedules
        }
        
        return State(
            current_time=self.current_time,
            new_request=new_request,
            fleet_status=fleet_status,
            planned_routes=planned_routes
        )
    
    def step(self, action):
        """
        执行动作，推进环境
        
        Args:
            action: Action对象
        
        Returns:
            tuple: (next_state, upper_reward, lower_reward, done, info)
        """
        upper_reward = 0.0
        lower_reward = 0.0
        info = {}
        
        # ✅ 检查是否超过工作时长：如果超过600分钟且所有请求已到达，不再执行新的派遣决策
        work_time_limit = max(T_WORK_V, T_WORK_D)
        all_requests_arrived = self.request_index >= len(self.requests)
        overtime = self.current_time >= work_time_limit and all_requests_arrived
        
        # 如果超时，强制清除所有车辆的等待状态（避免死锁）
        if overtime:
            for vehicle_id in range(self.num_vehicles):
                self.vehicle_waiting[vehicle_id] = False
                self.vehicle_wait_until[vehicle_id] = 0.0
        
        # 只有在未超时的情况下才处理决策动作
        if not overtime:
            # 处理上层智能体动作（车辆调度）
            if action.upper_action == 1:  # 出发
                # 找到在仓库等待的车辆
                available_vehicles = [i for i in range(self.num_vehicles) 
                                    if self.vehicle_status[i] == 0 and 
                                    self.vehicle_available_time[i] <= self.current_time]
                
                if len(available_vehicles) > 0:
                    vehicle_id = available_vehicles[0]
                    # 检查车辆路线是否为空
                    num_vehicle_requests = len(self.vehicle_routes[vehicle_id])
                    
                    if num_vehicle_requests == 0:
                        # 空车出发：给予严厉惩罚，但不执行出发动作
                        upper_reward = self.reward_calc.calculate_upper_reward(0)  # -5.0
                        # 注意：不清空RBU，不执行出发动作
                    else:
                        # 有请求：正常执行出发流程
                        # 在车辆出发前，检测等待超时并更新RBU中的奖励
                        self._check_wait_timeout(vehicle_id)
                        # 计算上层奖励（改进版：基于车辆路线中的请求数，鼓励批量出发）
                        upper_reward = self.reward_calc.calculate_upper_reward(num_vehicle_requests)
                        # 清空RBU
                        self.rbu.clear()
                        # 车辆出发
                        self._depart_vehicle(vehicle_id)
            
            elif action.upper_action == 0:  # 等待
                # 设置车辆等待
                available_vehicles = [i for i in range(self.num_vehicles) 
                                    if self.vehicle_status[i] == 0 and 
                                    self.vehicle_available_time[i] <= self.current_time]
                if len(available_vehicles) > 0:
                    vehicle_id = available_vehicles[0]
                    self.vehicle_waiting[vehicle_id] = True
                    self.vehicle_wait_until[vehicle_id] = self.current_time + VEHICLE_WAIT_INTERVAL
                # 等待时奖励始终为0，不应该惩罚（避免奖励信号混乱）
                upper_reward = 0.0
            
            # 处理下层智能体动作（请求分配）
            # 先获取当前状态，检查是否有新请求
            current_state = self._get_state()
            if current_state.new_request is not None:
                request = current_state.new_request
                # 将请求添加到RBU
                self.rbu.add_request(request)
                
                # 处理请求后，增加索引（重要！）
                self.request_index += 1
                
                if action.lower_action == 0:  # 分配给无人机
                    success = self._assign_to_drone(request)
                    # ✅ 修复：分配成功给小奖励，实际送达需要验证
                    if success:
                        # 分配成功给小奖励+0.3（接受请求的鼓励）
                        # 实际送达的主要奖励将在验证时给予
                        lower_reward = 0.3
                    else:
                        lower_reward = -0.5  # 分配失败（可行性检查没通过）
                    self.rbu.allocate_request(request, success, lower_reward)
                
                elif action.lower_action == 1:  # 分配给车辆
                    success = self._assign_to_vehicle(request)
                    # ✅ 修复：分配成功给小奖励，实际送达由车辆出发时验证
                    if success:
                        # 分配成功给小奖励+0.3（接受请求的鼓励）
                        # 车辆出发时会检查实际送达，成功+1.2，超时-2.0
                        lower_reward = 0.3
                    else:
                        lower_reward = -0.5  # 分配失败（可行性检查没通过）
                    # 记录奖励到RBU，后续在车辆出发时根据实际送达更新
                    self.rbu.allocate_request(request, success, lower_reward)
                
                elif action.lower_action == 2:  # 拒绝
                    self.rejected_requests += 1
                    # 计算可行性信息，用于可行性感知拒绝惩罚
                    vehicle_feasible = False
                    drone_feasible = False
                    
                    # 检查车辆可行性
                    for vehicle_id in range(self.num_vehicles):
                        if self.vehicle_status[vehicle_id] == 0:
                            feasible, _ = check_vehicle_feasibility(
                                request, self.vehicle_routes[vehicle_id],
                                self.current_time, self.vehicle_available_time[vehicle_id]
                            )
                            if feasible:
                                vehicle_feasible = True
                                break
                    
                    # 检查无人机可行性
                    for drone_id in range(self.num_drones):
                        if self.drone_status[drone_id] == 0:
                            feasible = check_drone_feasibility(
                                request, self.current_time, self.drone_available_time[drone_id]
                            )
                            if feasible:
                                drone_feasible = True
                                break
                    
                    # 使用可行性感知拒绝惩罚
                    lower_reward = self.reward_calc.calculate_lower_reward(
                        2, False, vehicle_feasible=vehicle_feasible, drone_feasible=drone_feasible
                    )
        
        # 推进时间到下一个决策点
        next_decision_time = self._get_next_decision_time()
        self.current_time = next_decision_time
        
        # 更新车辆和无人机状态（处理返回事件）
        self._update_fleet_status()
        
        # 检查episode是否结束
        # ✅ 修复：允许车辆和无人机工作到T_WORK_V/T_WORK_D（600分钟）
        # Episode结束条件：
        # 1. 所有请求都已到达 AND
        # 2. 所有车辆和无人机都已返回仓库且空闲
        # 注意：即使超过600分钟，也要等待所有在路上的资源回到仓库
        all_requests_arrived = self.request_index >= len(self.requests)
        all_vehicles_idle = all(
            self.vehicle_status[i] == 0 and len(self.vehicle_routes[i]) == 0 
            for i in range(self.num_vehicles)
        )
        all_drones_idle = all(
            self.drone_status[i] == 0 
            for i in range(self.num_drones)
        )
        
        # 所有请求已到达 且 所有资源都已返回仓库
        done = all_requests_arrived and all_vehicles_idle and all_drones_idle
        self.episode_done = done
        
        # 获取下一个状态
        next_state = self._get_state()
        
        # 更新统计信息
        info['delivered'] = self.delivered_requests
        info['rejected'] = self.rejected_requests
        info['total'] = self.total_requests
        
        return next_state, upper_reward, lower_reward, done, info
    
    def _check_wait_timeout(self, vehicle_id):
        """
        检测车辆路线中因等待导致超时的请求，并更新RBU中的奖励
        
        Args:
            vehicle_id: 车辆ID
        """
        route = self.vehicle_routes[vehicle_id]
        if len(route) == 0:
            return
        
        # 检查路线中的每个请求是否因等待而超时
        for request, _ in route:
            # 如果当前时间已经超过deadline，说明因为等待导致超时
            if self.current_time > request.deadline:
                # 更新RBU中该请求的奖励（使用统一的奖励计算，惩罚增强到-2.0）
                timeout_reward = self.reward_calc.calculate_lower_reward(1, False, vehicle_wait_failed=True)
                self.rbu.update_request_reward(request, timeout_reward, False)
    
    def _depart_vehicle(self, vehicle_id):
        """车辆出发执行路线"""
        if len(self.vehicle_routes[vehicle_id]) == 0:
            # 如果路线为空，不执行任何操作
            return
        
        self.vehicle_status[vehicle_id] = 1  # 设置为途中
        
        # 计算车辆执行路线的时间
        route = self.vehicle_routes[vehicle_id]
        current_pos = (WAREHOUSE_X, WAREHOUSE_Y)
        current_time = self.current_time + T_LOAD_V
        
        # 执行路线中的所有请求
        for request, _ in route:
            distance = road_distance(
                current_pos[0], current_pos[1], request.x, request.y
            )
            travel_time = travel_time_vehicle(distance, V_E)
            delivery_time = current_time + travel_time + T_DROP_V
            
            # 检查是否按时送达（在截止时间前）
            if delivery_time <= request.deadline:
                request.delivered = True
                request.delivery_time = delivery_time
                self.delivered_requests += 1
                # ✅ 实际送达成功，更新RBU奖励（+0.3已给，再+1.2 = 总共+1.5）
                self.rbu.update_request_reward(request, 1.2, True)
            else:
                # ✅ 送达超时惩罚增强：从-1.0改为-1.5（净效果从+1.0变为-0.5）
                self.rbu.update_request_reward(request, -1.5, False)
            
            current_time = delivery_time
            current_pos = (request.x, request.y)
        
        # 返回仓库
        distance = road_distance(
            current_pos[0], current_pos[1], WAREHOUSE_X, WAREHOUSE_Y
        )
        travel_time = travel_time_vehicle(distance, V_E)
        return_time = current_time + travel_time
        
        # 更新车辆状态
        self.vehicle_available_time[vehicle_id] = return_time
        self.vehicle_routes[vehicle_id] = []  # 清空路线
        self.vehicle_waiting[vehicle_id] = False
    
    def _assign_to_drone(self, request):
        """将请求分配给无人机"""
        # 构建无人机队列
        drone_queue = [(i, self.drone_available_time[i]) 
                       for i in range(self.num_drones)]
        
        success, drone_id, delivery_time, next_available = assign_drone_fifo(
            request, drone_queue, self.current_time
        )
        
        if success:
            request.assigned = True
            request.assigned_to = f'drone_{drone_id}'
            self.drone_status[drone_id] = 1
            self.drone_available_time[drone_id] = next_available
            
            # 更新无人机调度
            if drone_id not in self.drone_schedules:
                self.drone_schedules[drone_id] = []
            self.drone_schedules[drone_id].append(
                (request, delivery_time, next_available)
            )
        
        return success
    
    def _assign_to_vehicle(self, request):
        """将请求分配给车辆"""
        # 找到最早可用的车辆
        available_vehicles = [(i, self.vehicle_available_time[i]) 
                            for i in range(self.num_vehicles)
                            if self.vehicle_status[i] == 0]
        
        if len(available_vehicles) == 0:
            return False
        
        # 按可用时间排序
        available_vehicles.sort(key=lambda x: x[1])
        
        best_vehicle_id = None
        best_insertion = None
        best_route = None
        best_delivery_time = None
        
        for vehicle_id, available_time in available_vehicles:
            feasible, position, new_route, delivery_time = find_best_insertion(
                request, self.vehicle_routes[vehicle_id],
                self.current_time, available_time
            )
            
            if feasible:
                if best_vehicle_id is None or delivery_time < best_delivery_time:
                    best_vehicle_id = vehicle_id
                    best_insertion = position
                    best_route = new_route
                    best_delivery_time = delivery_time
        
        if best_vehicle_id is not None:
            request.assigned = True
            request.assigned_to = f'vehicle_{best_vehicle_id}'
            self.vehicle_routes[best_vehicle_id] = best_route
            return True
        
        return False
    
    def _get_next_decision_time(self):
        """获取下一个决策点时间"""
        next_times = []
        
        # 下一个未处理请求的到达时间（只在600分钟前考虑新请求）
        if self.request_index < len(self.requests) and self.current_time < max(T_WORK_V, T_WORK_D):
            next_request_time = self.requests[self.request_index].arrival_time
            # 只添加未来时间的请求
            if next_request_time > self.current_time:
                next_times.append(next_request_time)
        
        # 车辆返回时间（无论何时都要等待）
        for vehicle_id in range(self.num_vehicles):
            if self.vehicle_status[vehicle_id] == 1:
                # 车辆在途中，返回时间
                if self.vehicle_available_time[vehicle_id] > self.current_time:
                    next_times.append(self.vehicle_available_time[vehicle_id])
            elif self.vehicle_waiting[vehicle_id]:
                # 车辆在等待，等待结束时间
                if self.vehicle_wait_until[vehicle_id] > self.current_time:
                    next_times.append(self.vehicle_wait_until[vehicle_id])
        
        # 无人机返回时间（无论何时都要等待）
        for drone_id in range(self.num_drones):
            if self.drone_status[drone_id] == 1:
                if len(self.drone_schedules[drone_id]) > 0:
                    last_schedule = self.drone_schedules[drone_id][-1]
                    next_available = last_schedule[2]  # next_available_time
                    if next_available > self.current_time:
                        next_times.append(next_available)
        
        if len(next_times) == 0:
            # 如果没有更多决策点，返回当前时间（episode应该结束了）
            return self.current_time
        
        next_time = min(next_times)
        # 确保时间总是向前推进（至少推进一小步）
        if next_time <= self.current_time:
            return self.current_time + 0.1
        
        # ✅ 修复：允许时间超过600分钟，以等待所有资源返回仓库
        return next_time
    
    def _update_fleet_status(self):
        """更新车队状态（处理返回事件）"""
        # 更新车辆状态
        for vehicle_id in range(self.num_vehicles):
            if (self.vehicle_status[vehicle_id] == 1 and 
                self.vehicle_available_time[vehicle_id] <= self.current_time):
                self.vehicle_status[vehicle_id] = 0
        
        # 更新无人机状态
        for drone_id in range(self.num_drones):
            if len(self.drone_schedules[drone_id]) > 0:
                # 检查所有已安排的请求是否已送达
                for request, delivery_time, next_available in self.drone_schedules[drone_id]:
                    if delivery_time <= self.current_time and not request.delivered:
                        if delivery_time <= request.deadline:
                            request.delivered = True
                            request.delivery_time = delivery_time
                            self.delivered_requests += 1
                            # ✅ 无人机实际送达成功，更新RBU奖励（+0.3已给，再+1.2 = 总共+1.5）
                            self.rbu.update_request_reward(request, 1.2, True)
                        else:
                            # ✅ 无人机送达超时，更新为负奖励
                            self.rbu.update_request_reward(request, -1.0, False)
                
                # 检查无人机是否已返回并可用
                if len(self.drone_schedules[drone_id]) > 0:
                    last_schedule = self.drone_schedules[drone_id][-1]
                    if last_schedule[2] <= self.current_time:  # next_available_time
                        self.drone_status[drone_id] = 0
        
        # 更新等待状态
        for vehicle_id in range(self.num_vehicles):
            if (self.vehicle_waiting[vehicle_id] and 
                self.vehicle_wait_until[vehicle_id] <= self.current_time):
                self.vehicle_waiting[vehicle_id] = False
    
    def get_earliest_drone_time(self):
        """获取无人机队伍中最早可用的时间"""
        if len(self.drone_available_time) == 0:
            return T_WORK_D
        return min(self.drone_available_time.values())
    
    def get_earliest_vehicle_time(self):
        """获取车辆队伍中最早可用的时间"""
        available = [self.vehicle_available_time[i] 
                    for i in range(self.num_vehicles)
                    if self.vehicle_status[i] == 0]
        if len(available) == 0:
            return T_WORK_V
        return min(available)
    
    def get_most_urgent_remain_time(self):
        """获取当前待调度车辆计划路线中最紧迫请求的剩余交付时间"""
        min_remain = float('inf')
        for vehicle_id in range(self.num_vehicles):
            if self.vehicle_status[vehicle_id] == 0:
                route = self.vehicle_routes[vehicle_id]
                for request, est_time in route:
                    remain = request.deadline - self.current_time
                    if remain < min_remain:
                        min_remain = remain
        return min_remain if min_remain != float('inf') else 0.0
    
    def get_pending_requests_count(self):
        """获取当前待调度车辆路线中的请求总数（RBU维度）"""
        total_requests = 0
        for vehicle_id in range(self.num_vehicles):
            # 只统计在仓库等待的车辆（status=0）的路线
            if self.vehicle_status[vehicle_id] == 0:
                total_requests += len(self.vehicle_routes[vehicle_id])
        return total_requests