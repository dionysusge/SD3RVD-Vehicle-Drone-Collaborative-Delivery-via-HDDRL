"""
Episode级别的详细可视化分析脚本
包含时间序列图、决策分布、资源利用率、请求处理流程等可视化
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
import seaborn as sns
from collections import defaultdict
from datetime import datetime

from environment.route_mdp import RouteMDP
from environment.state import State
from environment.action import Action
from agents.upper_agent import UpperAgent
from agents.lower_agent import LowerAgent
from config.training_config import FLEET_CONFIG
from heuristics.feasibility_check import check_vehicle_feasibility, check_drone_feasibility


# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class EpisodeVisualizer:
    """Episode级别可视化分析器"""
    
    def __init__(self, fleet_config=None):
        """
        初始化可视化器
        
        Args:
            fleet_config: 车队配置
        """
        self.fleet_config = fleet_config if fleet_config else FLEET_CONFIG
        
        # 创建环境
        self.env = RouteMDP(
            num_vehicles=self.fleet_config['num_vehicles'],
            num_drones=self.fleet_config['num_drones'],
            time_dist='homo',
            space_dist='normal',
            lambda_rate=0.5
        )
        
        # 创建智能体
        self.upper_agent = UpperAgent()
        self.lower_agent = LowerAgent()
    
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
        
        vehicle_feasible = False
        vehicle_insert_cost = float('inf')
        
        for vehicle_id in range(self.env.num_vehicles):
            if self.env.vehicle_status[vehicle_id] == 0:
                feasible, cost = check_vehicle_feasibility(
                    request, self.env.vehicle_routes[vehicle_id],
                    self.env.current_time, self.env.vehicle_available_time[vehicle_id]
                )
                if feasible:
                    vehicle_feasible = True
                    vehicle_insert_cost = min(vehicle_insert_cost, cost)
        
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
    
    def run_and_visualize_episode(self, episode_num=1, model_path_upper=None, 
                                  model_path_lower=None, output_dir='visualizations'):
        """
        运行单个episode并进行详细可视化
        
        Args:
            episode_num: Episode编号
            model_path_upper: 上层模型路径
            model_path_lower: 下层模型路径
            output_dir: 输出目录
        """
        # 加载模型
        if model_path_upper and os.path.exists(model_path_upper):
            self.upper_agent.load(model_path_upper)
        if model_path_lower and os.path.exists(model_path_lower):
            self.lower_agent.load(model_path_lower)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 运行episode并收集数据
        print(f"运行Episode {episode_num}...")
        episode_data = self._run_episode()
        
        # 生成所有可视化
        print("生成可视化...")
        self._plot_time_series(episode_data, episode_num, output_dir)
        self._plot_decision_distribution(episode_data, episode_num, output_dir)
        self._plot_resource_utilization(episode_data, episode_num, output_dir)
        self._plot_request_timeline(episode_data, episode_num, output_dir)
        self._plot_reward_cumulative(episode_data, episode_num, output_dir)
        self._plot_action_heatmap(episode_data, episode_num, output_dir)
        
        # 生成综合报告图
        self._plot_comprehensive_report(episode_data, episode_num, output_dir)
        
        # 保存数据
        json_path = os.path.join(output_dir, f'episode_{episode_num}_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(episode_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\nEpisode {episode_num} 可视化完成！")
        print(f"结果保存在: {output_dir}")
        print(f"数据文件: {json_path}")
    
    def _run_episode(self):
        """运行单个episode并收集详细数据"""
        state = self.env.reset()
        done = False
        steps = 0
        
        episode_data = {
            'steps': [],
            'times': [],
            'rewards_upper': [],
            'rewards_lower': [],
            'cumulative_reward_upper': [],
            'cumulative_reward_lower': [],
            'delivered': [],
            'rejected': [],
            'total': [],
            'delivery_rate': [],
            'upper_actions': [],
            'lower_actions': [],
            'vehicle_active': [],
            'drone_active': [],
            'requests': []
        }
        
        cumulative_reward_upper = 0.0
        cumulative_reward_lower = 0.0
        
        while not done and steps < 10000:
            # 记录当前状态
            vehicle_active = sum(1 for v in self.env.vehicle_status.values() if v == 1)
            drone_active = sum(1 for d in self.env.drone_status.values() if d == 1)
            
            request_info = None
            if state.new_request is not None:
                request_info = {
                    'id': state.new_request.id,
                    'arrival_time': state.new_request.arrival_time,
                    'deadline': state.new_request.deadline,
                    'x': state.new_request.x,
                    'y': state.new_request.y
                }
            
            # 上层决策
            upper_action = None
            available_vehicles = [i for i in range(self.env.num_vehicles)
                                if self.env.vehicle_status[i] == 0 and
                                self.env.vehicle_available_time[i] <= self.env.current_time]
            
            if len(available_vehicles) > 0:
                upper_state = self._get_upper_state(state)
                upper_action = self.upper_agent.select_action(upper_state, training=False)
            else:
                upper_action = 2  # Nil
            
            # 下层决策
            lower_action = None
            if state.new_request is not None:
                lower_state = self._get_lower_state(state, state.new_request)
                lower_action, _, _ = self.lower_agent.select_action(lower_state, training=False)
            else:
                lower_action = 3  # Nil
            
            # 执行动作
            action = Action(upper_action=upper_action, lower_action=lower_action, route_update=None)
            next_state, upper_reward, lower_reward, done, info = self.env.step(action)
            
            cumulative_reward_upper += upper_reward
            cumulative_reward_lower += lower_reward
            
            # 记录数据
            episode_data['steps'].append(steps)
            episode_data['times'].append(self.env.current_time)
            episode_data['rewards_upper'].append(upper_reward)
            episode_data['rewards_lower'].append(lower_reward)
            episode_data['cumulative_reward_upper'].append(cumulative_reward_upper)
            episode_data['cumulative_reward_lower'].append(cumulative_reward_lower)
            episode_data['delivered'].append(info['delivered'])
            episode_data['rejected'].append(info['rejected'])
            episode_data['total'].append(info['total'])
            episode_data['delivery_rate'].append(info['delivered'] / info['total'] if info['total'] > 0 else 0)
            episode_data['upper_actions'].append(upper_action)
            episode_data['lower_actions'].append(lower_action)
            episode_data['vehicle_active'].append(vehicle_active)
            episode_data['drone_active'].append(drone_active)
            episode_data['requests'].append(request_info)
            
            state = next_state
            steps += 1
        
        return episode_data
    
    def _plot_time_series(self, data, episode_num, output_dir):
        """绘制时间序列图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Episode {episode_num} - Time Series Analysis', fontsize=16, fontweight='bold')
        
        times = np.array(data['times'])
        
        # 奖励时间序列
        axes[0, 0].plot(times, data['cumulative_reward_upper'], label='Upper Agent', linewidth=2, alpha=0.8)
        axes[0, 0].plot(times, data['cumulative_reward_lower'], label='Lower Agent', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Time (minutes)')
        axes[0, 0].set_ylabel('Cumulative Reward')
        axes[0, 0].set_title('Cumulative Reward Time Series')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 送达/拒绝数量
        axes[0, 1].plot(times, data['delivered'], label='Delivered', linewidth=2, color='green', alpha=0.8)
        axes[0, 1].plot(times, data['rejected'], label='Rejected', linewidth=2, color='red', alpha=0.8)
        axes[0, 1].set_xlabel('Time (minutes)')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Request Processing Statistics')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 送达率
        axes[1, 0].plot(times, np.array(data['delivery_rate']) * 100, linewidth=2, color='blue', alpha=0.8)
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Delivery Rate (%)')
        axes[1, 0].set_title('Delivery Rate Over Time')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 105])
        
        # 资源利用率
        vehicle_util = np.array(data['vehicle_active']) / self.env.num_vehicles * 100
        drone_util = np.array(data['drone_active']) / self.env.num_drones * 100
        
        axes[1, 1].plot(times, vehicle_util, label=f'Vehicles ({self.env.num_vehicles})', 
                       linewidth=2, alpha=0.8)
        axes[1, 1].plot(times, drone_util, label=f'Drones ({self.env.num_drones})', 
                       linewidth=2, alpha=0.8)
        axes[1, 1].set_xlabel('Time (minutes)')
        axes[1, 1].set_ylabel('Active Rate (%)')
        axes[1, 1].set_title('Resource Utilization')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 105])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_time_series.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_decision_distribution(self, data, episode_num, output_dir):
        """绘制决策分布图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Episode {episode_num} - Decision Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 上层决策分布
        upper_action_names = ['Wait', 'Depart', 'Nil']
        upper_counts = [data['upper_actions'].count(i) for i in range(3)]
        
        axes[0].bar(upper_action_names, upper_counts, color=['#3498db', '#2ecc71', '#95a5a6'], alpha=0.8)
        axes[0].set_ylabel('Count')
        axes[0].set_title('Upper Agent Decision Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, count in enumerate(upper_counts):
            axes[0].text(i, count + max(upper_counts) * 0.01, str(count), 
                        ha='center', va='bottom', fontweight='bold')
        
        # 下层决策分布
        lower_action_names = ['Drone', 'Vehicle', 'Reject', 'Nil']
        lower_counts = [data['lower_actions'].count(i) for i in range(4)]
        
        axes[1].bar(lower_action_names, lower_counts, color=['#e74c3c', '#f39c12', '#c0392b', '#95a5a6'], alpha=0.8)
        axes[1].set_ylabel('Count')
        axes[1].set_title('Lower Agent Decision Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, count in enumerate(lower_counts):
            axes[1].text(i, count + max(lower_counts) * 0.01, str(count), 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_decision_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_resource_utilization(self, data, episode_num, output_dir):
        """绘制资源利用率图"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'Episode {episode_num} - Resource Utilization Analysis', fontsize=16, fontweight='bold')
        
        times = np.array(data['times'])
        vehicle_active = np.array(data['vehicle_active'])
        drone_active = np.array(data['drone_active'])
        
        # 车辆利用率
        axes[0].fill_between(times, 0, vehicle_active, alpha=0.6, color='#3498db', label='Active Vehicles')
        axes[0].axhline(y=self.env.num_vehicles, color='r', linestyle='--', linewidth=2, label='Total Vehicles')
        axes[0].set_xlabel('Time (minutes)')
        axes[0].set_ylabel('Vehicle Count')
        axes[0].set_title(f'Vehicle Utilization (Total: {self.env.num_vehicles})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, self.env.num_vehicles + 1])
        
        # 无人机利用率
        axes[1].fill_between(times, 0, drone_active, alpha=0.6, color='#e74c3c', label='Active Drones')
        axes[1].axhline(y=self.env.num_drones, color='r', linestyle='--', linewidth=2, label='Total Drones')
        axes[1].set_xlabel('Time (minutes)')
        axes[1].set_ylabel('Drone Count')
        axes[1].set_title(f'Drone Utilization (Total: {self.env.num_drones})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, self.env.num_drones + 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_resource_utilization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_request_timeline(self, data, episode_num, output_dir):
        """绘制请求处理时间线"""
        # 提取请求信息
        requests = [r for r in data['requests'] if r is not None]
        
        if len(requests) == 0:
            return
        
        fig, ax = plt.subplots(figsize=(16, 8))
        fig.suptitle(f'Episode {episode_num} - Request Processing Timeline', fontsize=16, fontweight='bold')
        
        times = np.array(data['times'])
        lower_actions = np.array(data['lower_actions'])
        
        # 为每个请求绘制时间线
        request_indices = []
        request_arrival_times = []
        request_deadlines = []
        request_actions = []
        
        request_idx = 0
        for i, req in enumerate(requests):
            if req is not None:
                request_indices.append(request_idx)
                request_arrival_times.append(req['arrival_time'])
                request_deadlines.append(req['deadline'])
                request_actions.append(lower_actions[i])
                request_idx += 1
        
        # 绘制请求到达时间和截止时间
        colors_map = {0: 'green', 1: 'blue', 2: 'red', 3: 'gray'}
        action_names_map = {0: 'Drone', 1: 'Vehicle', 2: 'Reject', 3: 'Nil'}
        
        for i, (arr_time, deadline, action) in enumerate(zip(request_arrival_times, request_deadlines, request_actions)):
            color = colors_map.get(action, 'gray')
            # 绘制时间窗口（从到达时间到截止时间）
            ax.barh(i, deadline - arr_time, left=arr_time, height=0.6, 
                   color=color, alpha=0.5, edgecolor='black', linewidth=1)
            # 标记到达时间
            ax.plot(arr_time, i, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Request Index')
        ax.set_title('Request Processing Timeline (Color indicates assignment decision)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加图例
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', 
                      markersize=10, label='Drone', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=10, label='Vehicle', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Reject', markeredgecolor='black'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_request_timeline.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_reward_cumulative(self, data, episode_num, output_dir):
        """绘制累积奖励对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Episode {episode_num} - Reward Analysis', fontsize=16, fontweight='bold')
        
        times = np.array(data['times'])
        
        # 累积奖励
        axes[0].plot(times, data['cumulative_reward_upper'], label='Upper Agent', 
                    linewidth=2.5, color='#3498db', alpha=0.8)
        axes[0].plot(times, data['cumulative_reward_lower'], label='Lower Agent', 
                    linewidth=2.5, color='#e74c3c', alpha=0.8)
        axes[0].fill_between(times, data['cumulative_reward_upper'], alpha=0.2, color='#3498db')
        axes[0].fill_between(times, data['cumulative_reward_lower'], alpha=0.2, color='#e74c3c')
        axes[0].set_xlabel('Time (minutes)')
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].set_title('Cumulative Reward Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 瞬时奖励
        axes[1].plot(times, data['rewards_upper'], label='Upper Agent', 
                    linewidth=1.5, color='#3498db', alpha=0.6, marker='o', markersize=2)
        axes[1].plot(times, data['rewards_lower'], label='Lower Agent', 
                    linewidth=1.5, color='#e74c3c', alpha=0.6, marker='s', markersize=2)
        axes[1].set_xlabel('Time (minutes)')
        axes[1].set_ylabel('Instant Reward')
        axes[1].set_title('Instant Reward')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_rewards.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_action_heatmap(self, data, episode_num, output_dir):
        """绘制动作热力图（时间窗口内动作分布）"""
        # 创建时间窗口
        times = np.array(data['times'])
        if len(times) == 0:
            return
        
        total_time = times[-1] if len(times) > 0 else 100
        num_windows = 20
        window_size = total_time / num_windows
        
        # 统计每个时间窗口内的动作
        upper_action_matrix = np.zeros((3, num_windows))  # Wait, Depart, Nil
        lower_action_matrix = np.zeros((4, num_windows))  # Drone, Vehicle, Reject, Nil
        
        for i, (t, upper_action, lower_action) in enumerate(zip(times, data['upper_actions'], data['lower_actions'])):
            window_idx = min(int(t / window_size), num_windows - 1)
            if 0 <= upper_action < 3:
                upper_action_matrix[upper_action, window_idx] += 1
            if 0 <= lower_action < 4:
                lower_action_matrix[lower_action, window_idx] += 1
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle(f'Episode {episode_num} - Action Heatmap (Time Windows)', fontsize=16, fontweight='bold')
        
        # 上层动作热力图
        im1 = axes[0].imshow(upper_action_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        axes[0].set_yticks([0, 1, 2])
        axes[0].set_yticklabels(['Wait', 'Depart', 'Nil'])
        axes[0].set_xlabel('Time Window')
        axes[0].set_title('Upper Agent Action Distribution')
        axes[0].set_xticks(range(0, num_windows, max(1, num_windows // 10)))
        axes[0].set_xticklabels([f'{i*window_size:.0f}' for i in range(0, num_windows, max(1, num_windows // 10))])
        plt.colorbar(im1, ax=axes[0])
        
        # 下层动作热力图
        im2 = axes[1].imshow(lower_action_matrix, aspect='auto', cmap='YlGnBu', interpolation='nearest')
        axes[1].set_yticks([0, 1, 2, 3])
        axes[1].set_yticklabels(['Drone', 'Vehicle', 'Reject', 'Nil'])
        axes[1].set_xlabel('Time Window')
        axes[1].set_title('Lower Agent Action Distribution')
        axes[1].set_xticks(range(0, num_windows, max(1, num_windows // 10)))
        axes[1].set_xticklabels([f'{i*window_size:.0f}' for i in range(0, num_windows, max(1, num_windows // 10))])
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_action_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_comprehensive_report(self, data, episode_num, output_dir):
        """生成综合报告图"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'Episode {episode_num} - Comprehensive Analysis Report', fontsize=18, fontweight='bold', y=0.995)
        
        times = np.array(data['times'])
        
        # 1. 累积奖励
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(times, data['cumulative_reward_upper'], label='Upper', linewidth=2, alpha=0.8)
        ax1.plot(times, data['cumulative_reward_lower'], label='Lower', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Cumulative Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 送达率
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(times, np.array(data['delivery_rate']) * 100, linewidth=2, color='blue', alpha=0.8)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Delivery Rate (%)')
        ax2.set_title('Delivery Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        # 3. 请求统计
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(times, data['delivered'], label='Delivered', linewidth=2, color='green', alpha=0.8)
        ax3.plot(times, data['rejected'], label='Rejected', linewidth=2, color='red', alpha=0.8)
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Count')
        ax3.set_title('Request Processing')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 上层决策分布
        ax4 = fig.add_subplot(gs[1, 0])
        upper_counts = [data['upper_actions'].count(i) for i in range(3)]
        ax4.bar(['Wait', 'Depart', 'Nil'], upper_counts, color=['#3498db', '#2ecc71', '#95a5a6'], alpha=0.8)
        ax4.set_ylabel('Count')
        ax4.set_title('Upper Agent Decisions')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. 下层决策分布
        ax5 = fig.add_subplot(gs[1, 1])
        lower_counts = [data['lower_actions'].count(i) for i in range(4)]
        ax5.bar(['Drone', 'Vehicle', 'Reject', 'Nil'], lower_counts, 
               color=['#e74c3c', '#f39c12', '#c0392b', '#95a5a6'], alpha=0.8)
        ax5.set_ylabel('Count')
        ax5.set_title('Lower Agent Decisions')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 资源利用率
        ax6 = fig.add_subplot(gs[1, 2])
        vehicle_util = np.array(data['vehicle_active']) / self.env.num_vehicles * 100
        drone_util = np.array(data['drone_active']) / self.env.num_drones * 100
        ax6.plot(times, vehicle_util, label='Vehicles', linewidth=2, alpha=0.8)
        ax6.plot(times, drone_util, label='Drones', linewidth=2, alpha=0.8)
        ax6.set_xlabel('Time')
        ax6.set_ylabel('Active Rate (%)')
        ax6.set_title('Resource Utilization')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim([0, 105])
        
        # 7-9. 关键指标统计
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # 计算最终统计
        final_delivered = data['delivered'][-1] if data['delivered'] else 0
        final_rejected = data['rejected'][-1] if data['rejected'] else 0
        final_total = data['total'][-1] if data['total'] else 0
        final_delivery_rate = data['delivery_rate'][-1] if data['delivery_rate'] else 0
        final_reward_upper = data['cumulative_reward_upper'][-1] if data['cumulative_reward_upper'] else 0
        final_reward_lower = data['cumulative_reward_lower'][-1] if data['cumulative_reward_lower'] else 0
        
        stats_text = f"""
Episode {episode_num} Final Statistics
{'='*80}
Request Processing:  Delivered={final_delivered}, Rejected={final_rejected}, Total={final_total}, Delivery Rate={final_delivery_rate*100:.2f}%
Reward Statistics:  Upper Cumulative Reward={final_reward_upper:.2f}, Lower Cumulative Reward={final_reward_lower:.2f}
Step Statistics:  Total Steps={len(data['steps'])}, Total Duration={times[-1] if len(times) > 0 else 0:.2f} minutes
Upper Decisions:  Wait={upper_counts[0]}, Depart={upper_counts[1]}, Nil={upper_counts[2]}
Lower Decisions:  Drone={lower_counts[0]}, Vehicle={lower_counts[1]}, Reject={lower_counts[2]}, Nil={lower_counts[3]}
Resource Utilization:  Avg Vehicle Utilization={np.mean(vehicle_util):.1f}%, Avg Drone Utilization={np.mean(drone_util):.1f}%
        """
        
        ax7.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.savefig(os.path.join(output_dir, f'episode_{episode_num}_comprehensive_report.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Episode级别的详细可视化分析')
    parser.add_argument('--episode', type=int, default=1, help='要可视化的episode编号')
    parser.add_argument('--upper-model', type=str, default=None, help='上层模型路径')
    parser.add_argument('--lower-model', type=str, default=None, help='下层模型路径')
    parser.add_argument('--model-dir', type=str, default='models', help='模型目录')
    parser.add_argument('--output-dir', type=str, default='visualizations', help='输出目录')
    
    args = parser.parse_args()
    
    # 如果没有指定模型路径，尝试从模型目录查找最新的模型
    upper_path = args.upper_model
    lower_path = args.lower_model
    
    if not upper_path or not lower_path:
        if os.path.exists(args.model_dir):
            upper_models = [f for f in os.listdir(args.model_dir) if f.startswith('upper_agent')]
            lower_models = [f for f in os.listdir(args.model_dir) if f.startswith('lower_agent')]
            
            if upper_models and lower_models:
                upper_models.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]))
                lower_models.sort(key=lambda x: int(x.split('ep')[1].split('.')[0]))
                
                if not upper_path:
                    upper_path = os.path.join(args.model_dir, upper_models[-1])
                if not lower_path:
                    lower_path = os.path.join(args.model_dir, lower_models[-1])
    
    if not upper_path or not lower_path:
        print("警告: 未找到模型文件，将使用未训练的模型进行可视化")
    
    visualizer = EpisodeVisualizer()
    visualizer.run_and_visualize_episode(
        episode_num=args.episode,
        model_path_upper=upper_path,
        model_path_lower=lower_path,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

