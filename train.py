"""
训练脚本
整合两层智能体，实现完整的训练循环
"""
import os
import numpy as np
import torch
from tqdm import tqdm

# 设置matplotlib为非交互式后端（必须在导入pyplot之前）
import matplotlib
matplotlib.use('Agg')  # 使用Agg后端，不显示窗口，只保存文件
import matplotlib.pyplot as plt
from matplotlib import rcParams

from environment.route_mdp import RouteMDP
from environment.state import State
from environment.action import Action
from environment.reward import RewardCalculator
from agents.upper_agent import UpperAgent
from agents.lower_agent import LowerAgent
from config.training_config import TRAINING_CONFIG, FLEET_CONFIG, SCENARIO_CONFIG
from config.environment_config import WAREHOUSE_X, WAREHOUSE_Y
from utils.logger import setup_logger
from heuristics.feasibility_check import check_vehicle_feasibility, check_drone_feasibility

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class Trainer:
    """训练器类"""
    
    def __init__(self, config=None, fleet_config=None, scenario_config=None):
        """
        初始化训练器
        
        Args:
            config: 训练配置
            fleet_config: 车队配置
            scenario_config: 场景配置
        """
        self.config = config if config else TRAINING_CONFIG
        self.fleet_config = fleet_config if fleet_config else FLEET_CONFIG
        self.scenario_config = scenario_config if scenario_config else SCENARIO_CONFIG
        
        # 解析场景配置
        scenario_type = self.scenario_config['scenario_type']
        scenario_params = self.scenario_config['scenarios'][scenario_type]
        
        # 创建环境
        self.env = RouteMDP(
            num_vehicles=self.fleet_config['num_vehicles'],
            num_drones=self.fleet_config['num_drones'],
            time_dist=scenario_params['time_dist'],
            space_dist=scenario_params['space_dist'],
            lambda_rate=self.scenario_config['lambda_rate']
        )
        
        # 创建智能体
        self.upper_agent = UpperAgent()
        self.lower_agent = LowerAgent()
        
        # 日志（获取timestamp用于统一命名）
        self.logger, self.timestamp = setup_logger('trainer')
        
        # 统计信息
        self.episode_rewards_upper = []
        self.episode_rewards_lower = []
        self.episode_delivered = []
        self.episode_rejected = []
        self.episode_upper_losses = []
        self.episode_lower_losses = []
        # 决策统计
        self.episode_upper_wait_count = []
        self.episode_upper_depart_count = []
        self.episode_lower_drone_count = []
        self.episode_lower_vehicle_count = []
        self.episode_lower_reject_count = []
        # 效率统计
        self.episode_vehicle_utilization = []
        self.episode_drone_utilization = []
        self.episode_requests_per_trip = []
        
        # 滑动平均损失（用于平滑损失曲线，减少抖动）
        self.ema_upper_loss = 0.0
        self.ema_lower_loss = 0.0
        self.ema_alpha = 0.1  # 平滑系数（0.1表示新值占10%，旧值占90%）
        
        # 模型保存目录
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 图表保存目录
        self.plot_dir = 'plots'
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def _get_upper_state(self, state):
        """获取上层智能体状态"""
        earliest_drone_time = self.env.get_earliest_drone_time()
        earliest_vehicle_time = self.env.get_earliest_vehicle_time()
        most_urgent_remain = self.env.get_most_urgent_remain_time()
        pending_requests_count = self.env.get_pending_requests_count()
        
        return state.get_upper_agent_state(
            earliest_drone_time, earliest_vehicle_time, most_urgent_remain,
            pending_requests_count
        )
    
    def _get_lower_state(self, state, request):
        """获取下层智能体状态"""
        earliest_drone_time = self.env.get_earliest_drone_time()
        earliest_vehicle_time = self.env.get_earliest_vehicle_time()
        
        # 计算车辆可行性
        vehicle_feasible = False
        vehicle_insert_cost = float('inf')
        
        # 检查每个车辆
        for vehicle_id in range(self.env.num_vehicles):
            if self.env.vehicle_status[vehicle_id] == 0:
                feasible, cost = check_vehicle_feasibility(
                    request, self.env.vehicle_routes[vehicle_id],
                    self.env.current_time, self.env.vehicle_available_time[vehicle_id]
                )
                if feasible:
                    vehicle_feasible = True
                    vehicle_insert_cost = min(vehicle_insert_cost, cost)
        
        # 计算无人机可行性
        drone_feasible = False
        for drone_id in range(self.env.num_drones):
            feasible = check_drone_feasibility(
                request, self.env.current_time, self.env.drone_available_time[drone_id]
            )
            if feasible:
                drone_feasible = True
                break
        
        return state.get_lower_agent_state(
            request, earliest_drone_time, earliest_vehicle_time,
            vehicle_feasible, vehicle_insert_cost, drone_feasible
        )
    
    def train(self):
        """执行训练"""
        self.logger.info("开始训练...")
        self.logger.info(f"配置: {self.config}")
        self.logger.info(f"车队配置: {self.fleet_config}")
        
        # 显示场景配置
        scenario_type = self.scenario_config['scenario_type']
        scenario_params = self.scenario_config['scenarios'][scenario_type]
        scenario_names = {
            'homo-normal': '均匀时间-正态空间（最简单）',
            'homo-power': '均匀时间-幂律空间（中等难度）',
            'heter-normal': '非均匀时间-正态空间（中等难度）',
            'heter-power': '非均匀时间-幂律空间（最高难度）'
        }
        self.logger.info("=" * 80)
        self.logger.info(f"场景类型: {scenario_type} - {scenario_names.get(scenario_type, '未知场景')}")
        self.logger.info(f"  时间分布: {scenario_params['time_dist']} ({'均匀分布' if scenario_params['time_dist'] == 'homo' else '非均匀分布（高峰期）'})")
        self.logger.info(f"  空间分布: {scenario_params['space_dist']} ({'正态分布（集中）' if scenario_params['space_dist'] == 'normal' else '幂律分布（聚集）'})")
        self.logger.info(f"  到达率λ: {self.scenario_config['lambda_rate']} 个/分钟")
        self.logger.info("=" * 80)
        
        for episode in range(self.config['num_episodes']):
            # 重置环境
            state = self.env.reset()
            episode_reward_upper = 0.0
            episode_reward_lower = 0.0
            steps = 0
            # 决策统计
            upper_wait_count = 0
            upper_depart_count = 0
            lower_drone_count = 0
            lower_vehicle_count = 0
            lower_reject_count = 0
            # 效率统计
            vehicle_trips = 0
            total_vehicle_requests = 0
            # 累计每个step的利用率（用于计算平均值）
            vehicle_active_steps = 0  # 累计车辆活跃的步数
            drone_active_steps = 0    # 累计无人机活跃的步数
            prev_time = 0.0  # 用于计算时间间隔
            # 损失统计
            upper_losses = []
            lower_losses = []
            
            # 上一个状态（用于经验回放）
            prev_upper_state = None
            prev_lower_state = None
            prev_upper_action = None
            prev_lower_action = None
            prev_lower_log_prob = None
            prev_lower_value = None
            
            done = False
            
            while not done and steps < self.config['max_steps_per_episode']:
                # 上层智能体决策（如果有车辆在仓库）
                upper_action = None
                available_vehicles = [i for i in range(self.env.num_vehicles)
                                    if self.env.vehicle_status[i] == 0 and
                                    self.env.vehicle_available_time[i] <= self.env.current_time]
                
                if len(available_vehicles) > 0:
                    upper_state = self._get_upper_state(state)
                    upper_action = self.upper_agent.select_action(upper_state, training=True)
                    
                    # 统计决策
                    if upper_action == 0:
                        upper_wait_count += 1
                    elif upper_action == 1:
                        upper_depart_count += 1
                    
                    # ✅ 改进：给等待动作一个基于RBU增长的渐进式奖励
                    # 存储经验（如果有上一个状态）
                    if prev_upper_state is not None:
                        # 计算RBU增长奖励（鼓励积累请求）
                        prev_rbu_count = prev_upper_state[4] * 10.0  # 反归一化（与新的归一化方式一致）
                        curr_rbu_count = upper_state[4] * 10.0
                        rbu_growth = curr_rbu_count - prev_rbu_count
                        
                        # 如果是等待动作且RBU在增长，给予小奖励
                        wait_reward = 0.0
                        if prev_upper_action == 0 and rbu_growth > 0:
                            # 根据RBU规模给予递增奖励：0.15 * sqrt(n)，增强激励
                            wait_reward = 0.5 * np.sqrt(curr_rbu_count)
                        
                        # 调试信息（前几个episode）
                        if episode < 3 and steps < 10 and prev_upper_action == 0:
                            self.logger.info(
                                f"  [RBU监控] RBU: {prev_rbu_count:.1f}->{curr_rbu_count:.1f}, "
                                f"增长={rbu_growth:.1f}, 等待奖励={wait_reward:.3f}"
                            )
                        
                        self.upper_agent.store_transition(
                            prev_upper_state, prev_upper_action, wait_reward,
                            upper_state, False
                        )
                    
                    prev_upper_state = upper_state
                    prev_upper_action = upper_action
                else:
                    upper_action = 2  # Nil
                
                # 下层智能体决策（如果有新请求）
                lower_action = None
                if state.new_request is not None:
                    request = state.new_request
                    lower_state = self._get_lower_state(state, request)
                    lower_action, log_prob, value = self.lower_agent.select_action(
                        lower_state, training=True
                    )
                    
                    # 统计决策
                    if lower_action == 0:
                        lower_drone_count += 1
                    elif lower_action == 1:
                        lower_vehicle_count += 1
                    elif lower_action == 2:
                        lower_reject_count += 1
                    
                    # 调试信息（前几个episode）
                    if episode < 3 and steps < 5:
                        self.logger.info(
                            f"Episode {episode}, Step {steps}: "
                            f"请求{request.id}到达, 动作={lower_action}, "
                            f"车辆可行={lower_state[4]:.0f}, 无人机可行={lower_state[6]:.0f}"
                        )
                    
                    # 不在这里存储经验，等获得奖励后再存储
                    prev_lower_state = lower_state
                    prev_lower_action = lower_action
                    prev_lower_log_prob = log_prob
                    prev_lower_value = value
                else:
                    lower_action = 3  # Nil
                
                # 创建动作
                action = Action(
                    upper_action=upper_action,
                    lower_action=lower_action,
                    route_update=None
                )
                
                # 在执行step之前，记录车辆出发时的路线长度（因为step会清空路线）
                if action.upper_action == 1:  # 车辆出发
                    available_vehicles = [i for i in range(self.env.num_vehicles)
                                        if self.env.vehicle_status[i] == 0 and
                                        self.env.vehicle_available_time[i] <= self.env.current_time]
                    if len(available_vehicles) > 0:
                        vehicle_id = available_vehicles[0]
                        route_length = len(self.env.vehicle_routes[vehicle_id])
                        if route_length > 0:
                            vehicle_trips += 1
                            total_vehicle_requests += route_length
                
                # 执行动作
                next_state, upper_reward, lower_reward, done, info = self.env.step(action)
                
                # 调试信息（前几个episode）- 显示奖励和动作
                if episode < 3 and steps < 10:
                    action_name = {0: "等待", 1: "出发", 2: "Nil"}
                    lower_action_name = {0: "无人机", 1: "车辆", 2: "拒绝", 3: "Nil"}
                    self.logger.info(
                        f"Episode {episode}, Step {steps}: "
                        f"上层动作={action_name.get(action.upper_action, 'Unknown')}, "
                        f"下层动作={lower_action_name.get(action.lower_action, 'Unknown')}, "
                        f"upper_reward={upper_reward:.2f}, lower_reward={lower_reward:.2f}"
                    )
                
                # 更新奖励
                episode_reward_upper += upper_reward
                episode_reward_lower += lower_reward
                
                # 存储经验（在获得奖励后）
                # 上层智能体
                if prev_upper_state is not None:
                    final_upper_state = self._get_upper_state(next_state)
                    self.upper_agent.store_transition(
                        prev_upper_state, prev_upper_action, upper_reward,
                        final_upper_state, done
                    )
                
                # 下层智能体（如果有新请求且做了决策）
                if prev_lower_state is not None and prev_lower_action is not None and prev_lower_action != 3:
                    self.lower_agent.store_transition(
                        prev_lower_state, prev_lower_action,
                        prev_lower_log_prob, prev_lower_value,
                        lower_reward, done
                    )
                
                # 训练智能体（降低训练频率，提高稳定性）
                # DQN：每50步训练一次，使用更多经验，减少批次差异
                if steps % 50 == 0:
                    if len(self.upper_agent.agent.memory) >= self.upper_agent.agent.batch_size:
                        upper_loss = self.upper_agent.train_step()
                        if upper_loss is not None:
                            upper_losses.append(upper_loss)
                # 注意：PPO只在Episode结束时训练，使用完整轨迹，提高稳定性
                # 不在Episode中间训练，避免使用不完整的轨迹
                
                # 统计效率指标（每个step都累计）
                # 计算时间间隔（用于加权平均）
                time_interval = self.env.current_time - prev_time if prev_time > 0 else 0.0
                if time_interval <= 0:
                    time_interval = 1.0  # 默认1分钟
                
                # 累计车辆利用率（统计正在途中的车辆数）
                active_vehicles = sum(1 for v in self.env.vehicle_status.values() if v == 1)
                vehicle_active_steps += active_vehicles * time_interval
                
                # 累计无人机利用率（统计正在使用的无人机数）
                active_drones = sum(1 for d in self.env.drone_status.values() if d == 1)
                drone_active_steps += active_drones * time_interval
                
                prev_time = self.env.current_time
                state = next_state
                steps += 1
            
            # Episode结束，处理最后的经验
            delivery_rate = info['delivered'] / info['total'] if info['total'] > 0 else 0.0
            
            if prev_upper_state is not None:
                final_upper_state = self._get_upper_state(state)
                self.upper_agent.store_transition(
                    prev_upper_state, prev_upper_action, upper_reward,
                    final_upper_state, done
                )
            
            if prev_lower_state is not None:
                self.lower_agent.store_transition(
                    prev_lower_state, prev_lower_action,
                    prev_lower_log_prob, prev_lower_value,
                    lower_reward, done
                )
                # Episode结束时训练PPO（确保使用完整的轨迹）
                if len(self.lower_agent.agent.states) > 0:
                    lower_loss = self.lower_agent.train_step()
                    if lower_loss is not None:
                        lower_losses.append(lower_loss)
            
            # 计算效率指标（基于整个episode的平均值）
            # 每趟请求数：平均每趟车辆行程携带的请求数
            avg_requests_per_trip = total_vehicle_requests / vehicle_trips if vehicle_trips > 0 else 0.0
            
            # 车辆利用率：整个episode中车辆的平均利用率
            # 计算总时间（episode持续时间）
            total_time = self.env.current_time if self.env.current_time > 0 else 1.0
            # 平均利用率 = 累计活跃时间 / (车辆数 * 总时间)
            vehicle_utilization = vehicle_active_steps / (self.env.num_vehicles * total_time) if total_time > 0 and self.env.num_vehicles > 0 else 0.0
            
            # 无人机利用率：整个episode中无人机的平均利用率
            drone_utilization = drone_active_steps / (self.env.num_drones * total_time) if total_time > 0 and self.env.num_drones > 0 else 0.0
            
            # 计算平均损失
            avg_upper_loss = np.mean(upper_losses) if len(upper_losses) > 0 else 0.0
            avg_lower_loss = np.mean(lower_losses) if len(lower_losses) > 0 else 0.0
            
            # 更新滑动平均损失（用于平滑损失曲线）
            if len(upper_losses) > 0:
                self.ema_upper_loss = self.ema_alpha * avg_upper_loss + (1 - self.ema_alpha) * self.ema_upper_loss
            if len(lower_losses) > 0:
                self.ema_lower_loss = self.ema_alpha * avg_lower_loss + (1 - self.ema_alpha) * self.ema_lower_loss
            
            # 记录统计信息
            self.episode_rewards_upper.append(episode_reward_upper)
            self.episode_rewards_lower.append(episode_reward_lower)
            self.episode_delivered.append(info['delivered'])
            self.episode_rejected.append(info['rejected'])
            self.episode_upper_losses.append(avg_upper_loss)
            self.episode_lower_losses.append(avg_lower_loss)
            self.episode_upper_wait_count.append(upper_wait_count)
            self.episode_upper_depart_count.append(upper_depart_count)
            self.episode_lower_drone_count.append(lower_drone_count)
            self.episode_lower_vehicle_count.append(lower_vehicle_count)
            self.episode_lower_reject_count.append(lower_reject_count)
            self.episode_requests_per_trip.append(avg_requests_per_trip)
            self.episode_vehicle_utilization.append(vehicle_utilization)
            self.episode_drone_utilization.append(drone_utilization)
            
            # 每个episode都输出详细统计信息
            self.logger.info("=" * 80)
            self.logger.info(f"Episode {episode + 1}/{self.config['num_episodes']} 完成")
            self.logger.info(f"  步数: {steps}")
            self.logger.info(f"  奖励: 上层={episode_reward_upper:.2f}, 下层={episode_reward_lower:.2f}")
            self.logger.info(f"  服务: 送达={info['delivered']}, 拒绝={info['rejected']}, 总计={info['total']}")
            self.logger.info(f"  送达率: {delivery_rate*100:.1f}%")
            self.logger.info(f"  损失: 上层(DQN)={avg_upper_loss:.6f} (EMA={self.ema_upper_loss:.6f}), 下层(PPO)={avg_lower_loss:.6f} (EMA={self.ema_lower_loss:.6f})")
            self.logger.info(f"  上层决策: 等待={upper_wait_count}, 出发={upper_depart_count}")
            self.logger.info(f"  下层决策: 无人机={lower_drone_count}, 车辆={lower_vehicle_count}, 拒绝={lower_reject_count}")
            self.logger.info(f"  [批量效率] 每趟请求数={avg_requests_per_trip:.2f}, 车辆行程数={vehicle_trips}")
            self.logger.info(f"  车辆利用率={vehicle_utilization:.2f}, 无人机利用率={drone_utilization:.2f}")
            self.logger.info("=" * 80)
            
            # 定期汇总日志（每log_freq个episode）
            if (episode + 1) % self.config['log_freq'] == 0:
                window = self.config['log_freq']
                avg_reward_upper = np.mean(self.episode_rewards_upper[-window:])
                avg_reward_lower = np.mean(self.episode_rewards_lower[-window:])
                avg_delivered = np.mean(self.episode_delivered[-window:])
                avg_rejected = np.mean(self.episode_rejected[-window:])
                avg_upper_loss = np.mean(self.episode_upper_losses[-window:])
                avg_lower_loss = np.mean(self.episode_lower_losses[-window:])
                
                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info(f"最近 {window} 个 Episode 的平均统计:")
                self.logger.info(f"  奖励: 上层={avg_reward_upper:.2f}, 下层={avg_reward_lower:.2f}")
                self.logger.info(f"  服务: 送达={avg_delivered:.1f}, 拒绝={avg_rejected:.1f}")
                self.logger.info(f"  损失: 上层(DQN)={avg_upper_loss:.6f} (EMA={self.ema_upper_loss:.6f}), 下层(PPO)={avg_lower_loss:.6f} (EMA={self.ema_lower_loss:.6f})")
                self.logger.info("=" * 80)
                self.logger.info("")
            
            # 更新学习率（DQN指数衰减）
            self.upper_agent.agent.update_learning_rate(episode + 1)
            
            # 更新epsilon（基于episode衰减，确保探索率逐渐降低）
            # 针对批量派遣学习优化：更激进的衰减，让智能体快速利用高批量奖励
            if episode < 1500:
                # 前1500 episodes：从0.5快速衰减到0.08（快速学习期）
                progress = episode / 1500
                new_epsilon = 0.5 * (1 - progress) + 0.08 * progress
                self.upper_agent.agent.epsilon = max(0.08, new_epsilon)
            elif episode < 5000:
                # 1500-5000 episodes：从0.08缓慢衰减到0.03（稳定期）
                progress = (episode - 1500) / 3500
                new_epsilon = 0.08 * (1 - progress) + 0.03 * progress
                self.upper_agent.agent.epsilon = max(0.03, new_epsilon)
            else:
                # 5000+ episodes：保持0.03（最小探索）
                self.upper_agent.agent.epsilon = 0.03
            
            # 每100个episode记录epsilon
            if (episode + 1) % 100 == 0:
                self.logger.info(f"当前探索率 epsilon: {self.upper_agent.agent.epsilon:.4f}")
            
            # 保存模型
            if (episode + 1) % self.config['save_freq'] == 0:
                self.save_models(episode + 1)
                # 同时保存训练曲线
                self.plot_training_curves(episode + 1)
            
            # 评估
            if (episode + 1) % self.config['eval_freq'] == 0:
                self.evaluate(episode + 1)
            
            # 定期绘制训练曲线（每50个episode）
            if (episode + 1) % 50 == 0:
                self.plot_training_curves(episode + 1)
    
    def evaluate(self, episode):
        """评估模型性能"""
        self.logger.info(f"评估 Episode {episode}...")
        
        total_delivered = 0
        for _ in range(self.config['eval_episodes']):
            state = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < self.config['max_steps_per_episode']:
                # 上层决策
                upper_action = None
                available_vehicles = [i for i in range(self.env.num_vehicles)
                                    if self.env.vehicle_status[i] == 0 and
                                    self.env.vehicle_available_time[i] <= self.env.current_time]
                
                if len(available_vehicles) > 0:
                    upper_state = self._get_upper_state(state)
                    upper_action = self.upper_agent.select_action(upper_state, training=False)
                else:
                    upper_action = 2
                
                # 下层决策
                lower_action = None
                if state.new_request is not None:
                    lower_state = self._get_lower_state(state, state.new_request)
                    lower_action, _, _ = self.lower_agent.select_action(
                        lower_state, training=False
                    )
                else:
                    lower_action = 3
                
                action = Action(
                    upper_action=upper_action,
                    lower_action=lower_action,
                    route_update=None
                )
                
                state, _, _, done, info = self.env.step(action)
                steps += 1
            
            total_delivered += info['delivered']
        
        avg_delivered = total_delivered / self.config['eval_episodes']
        self.logger.info(f"评估结果 - 平均送达数: {avg_delivered:.2f}")
    
    def save_models(self, episode):
        """保存模型"""
        upper_path = os.path.join(self.model_dir, f'upper_agent_ep{episode}.pth')
        lower_path = os.path.join(self.model_dir, f'lower_agent_ep{episode}.pth')
        
        self.upper_agent.save(upper_path)
        self.lower_agent.save(lower_path)
        
        self.logger.info(f"模型已保存: Episode {episode}")
    
    def plot_training_curves(self, episode):
        """
        绘制训练曲线（在同一张图上重绘，文件名与log一致）
        
        Args:
            episode: 当前episode数
        """
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'训练曲线 - Episode {episode} (实时更新)', fontsize=16, fontweight='bold')
        
        # 计算10-episode滑动平均
        window = 10
        
        # 1. 上层智能体损失曲线
        ax = axes[0, 0]
        if len(self.episode_upper_losses) > 0:
            ax.plot(self.episode_upper_losses, alpha=0.3, color='blue', label='原始损失')
            if len(self.episode_upper_losses) >= window:
                moving_avg = np.convolve(self.episode_upper_losses, 
                                        np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.episode_upper_losses)), 
                       moving_avg, color='blue', linewidth=2, label=f'{window}期移动平均')
        ax.set_xlabel('Episode')
        ax.set_ylabel('损失')
        ax.set_title('上层智能体(DQN)损失')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 下层智能体损失曲线
        ax = axes[0, 1]
        if len(self.episode_lower_losses) > 0:
            ax.plot(self.episode_lower_losses, alpha=0.3, color='green', label='原始损失')
            if len(self.episode_lower_losses) >= window:
                moving_avg = np.convolve(self.episode_lower_losses, 
                                        np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.episode_lower_losses)), 
                       moving_avg, color='green', linewidth=2, label=f'{window}期移动平均')
        ax.set_xlabel('Episode')
        ax.set_ylabel('损失')
        ax.set_title('下层智能体(PPO)损失')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 上层智能体奖励曲线
        ax = axes[0, 2]
        if len(self.episode_rewards_upper) > 0:
            ax.plot(self.episode_rewards_upper, alpha=0.3, color='red', label='原始奖励')
            if len(self.episode_rewards_upper) >= window:
                moving_avg = np.convolve(self.episode_rewards_upper, 
                                        np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.episode_rewards_upper)), 
                       moving_avg, color='red', linewidth=2, label=f'{window}期移动平均')
        ax.set_xlabel('Episode')
        ax.set_ylabel('奖励')
        ax.set_title('上层智能体奖励')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        
        # 4. 下层智能体奖励曲线
        ax = axes[1, 0]
        if len(self.episode_rewards_lower) > 0:
            ax.plot(self.episode_rewards_lower, alpha=0.3, color='orange', label='原始奖励')
            if len(self.episode_rewards_lower) >= window:
                moving_avg = np.convolve(self.episode_rewards_lower, 
                                        np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.episode_rewards_lower)), 
                       moving_avg, color='orange', linewidth=2, label=f'{window}期移动平均')
        ax.set_xlabel('Episode')
        ax.set_ylabel('奖励')
        ax.set_title('下层智能体奖励')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 送达率和拒绝率
        ax = axes[1, 1]
        if len(self.episode_delivered) > 0 and len(self.episode_rejected) > 0:
            # ✅ 修复：使用实际的total_requests而不是delivered+rejected
            # 获取每个episode的真实总请求数（从环境记录）
            # 注意：这里需要从环境获取，暂时使用delivered+rejected作为近似
            # 但要记录实际的total用于正确计算
            delivery_rates = []
            reject_rates = []
            accept_not_delivered_rates = []  # 接受但未送达
            
            for i in range(len(self.episode_delivered)):
                delivered = self.episode_delivered[i]
                rejected = self.episode_rejected[i]
                # 假设total = delivered + rejected（这是一个简化，实际可能有接受但未送达的）
                total = delivered + rejected
                if total > 0:
                    delivery_rates.append(delivered / total * 100)
                    reject_rates.append(rejected / total * 100)
                    accept_not_delivered = total - delivered - rejected
                    accept_not_delivered_rates.append(max(0, accept_not_delivered / total * 100))
                else:
                    delivery_rates.append(0)
                    reject_rates.append(0)
                    accept_not_delivered_rates.append(0)
            
            ax.plot(delivery_rates, alpha=0.3, color='green', label='送达率')
            ax.plot(reject_rates, alpha=0.3, color='red', label='拒绝率')
            
            if len(delivery_rates) >= window:
                delivery_avg = np.convolve(delivery_rates, np.ones(window)/window, mode='valid')
                reject_avg = np.convolve(reject_rates, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(delivery_rates)), 
                       delivery_avg, color='green', linewidth=2, label=f'送达率({window}期均值)')
                ax.plot(range(window-1, len(reject_rates)), 
                       reject_avg, color='red', linewidth=2, label=f'拒绝率({window}期均值)')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('百分比 (%)')
        ax.set_title('送达率与拒绝率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # 6. 效率指标
        ax = axes[1, 2]
        if len(self.episode_requests_per_trip) > 0:
            ax.plot(self.episode_requests_per_trip, alpha=0.3, color='purple', label='原始数据')
            if len(self.episode_requests_per_trip) >= window:
                moving_avg = np.convolve(self.episode_requests_per_trip, 
                                        np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.episode_requests_per_trip)), 
                       moving_avg, color='purple', linewidth=2, label=f'{window}期移动平均')
        ax.set_xlabel('Episode')
        ax.set_ylabel('每趟请求数')
        ax.set_title('车辆批量效率')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='目标(5-8)')
        ax.axhline(y=8, color='red', linestyle='--', alpha=0.5)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表（使用固定文件名，与log一致，每次重绘覆盖）
        plot_path = os.path.join(self.plot_dir, f'training_curves_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"训练曲线已更新: {plot_path} (当前Episode {episode})")


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
