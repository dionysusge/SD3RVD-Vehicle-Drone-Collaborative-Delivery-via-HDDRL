"""
详细的测试/评估脚本
包含全面的性能指标统计、决策分析和结果可视化
"""
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

from environment.route_mdp import RouteMDP
from environment.state import State
from environment.action import Action
from agents.upper_agent import UpperAgent
from agents.lower_agent import LowerAgent
from config.training_config import FLEET_CONFIG
from config.environment_config import WAREHOUSE_X, WAREHOUSE_Y
from heuristics.feasibility_check import check_vehicle_feasibility, check_drone_feasibility


class DetailedEvaluator:
    """详细评估器类"""
    
    def __init__(self, fleet_config=None):
        """
        初始化评估器
        
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
        
        # 计算车辆可行性
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
    
    def evaluate_detailed(self, num_episodes=10, model_path_upper=None, model_path_lower=None, 
                         save_detailed=True, output_dir='test_results'):
        """
        详细评估HDDRL方法
        
        Args:
            num_episodes: 评估episode数量
            model_path_upper: 上层智能体模型路径
            model_path_lower: 下层智能体模型路径
            save_detailed: 是否保存详细数据
            output_dir: 输出目录
        
        Returns:
            dict: 详细评估结果
        """
        # 加载模型
        if model_path_upper and os.path.exists(model_path_upper):
            print(f"加载上层模型: {model_path_upper}")
            self.upper_agent.load(model_path_upper)
        if model_path_lower and os.path.exists(model_path_lower):
            print(f"加载下层模型: {model_path_lower}")
            self.lower_agent.load(model_path_lower)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 汇总统计
        summary = {
            'episodes': [],
            'delivered': [],
            'rejected': [],
            'total': [],
            'delivery_rate': [],
            'episode_reward_upper': [],
            'episode_reward_lower': [],
            'steps': [],
            'upper_wait_count': [],
            'upper_depart_count': [],
            'lower_drone_count': [],
            'lower_vehicle_count': [],
            'lower_reject_count': []
        }
        
        # Episode级别详细信息
        episode_details = []
        
        print(f"\n开始评估 {num_episodes} 个episode...")
        print("=" * 80)
        
        for episode in tqdm(range(num_episodes), desc="评估进度"):
            # 重置环境
            state = self.env.reset()
            done = False
            steps = 0
            
            # Episode级别的统计
            episode_stats = {
                'episode': episode + 1,
                'steps': 0,
                'reward_upper': 0.0,
                'reward_lower': 0.0,
                'delivered': 0,
                'rejected': 0,
                'total': 0,
                'delivery_rate': 0.0,
                'upper_actions': {'wait': 0, 'depart': 0, 'nil': 0},
                'lower_actions': {'drone': 0, 'vehicle': 0, 'reject': 0, 'nil': 0},
                'step_details': []  # 保存每步的详细信息
            }
            
            # 效率统计
            vehicle_trips = 0
            total_vehicle_requests = 0
            vehicle_active_time = 0.0
            drone_active_time = 0.0
            prev_time = 0.0
            
            while not done and steps < 10000:
                step_info = {
                    'step': steps,
                    'time': self.env.current_time,
                    'upper_action': None,
                    'lower_action': None,
                    'upper_reward': 0.0,
                    'lower_reward': 0.0,
                    'new_request': None,
                    'delivered_increment': 0,
                    'rejected_increment': 0
                }
                
                prev_delivered = info['delivered'] if steps > 0 else 0
                prev_rejected = info['rejected'] if steps > 0 else 0
                
                # 上层智能体决策
                upper_action = None
                available_vehicles = [i for i in range(self.env.num_vehicles)
                                    if self.env.vehicle_status[i] == 0 and
                                    self.env.vehicle_available_time[i] <= self.env.current_time]
                
                if len(available_vehicles) > 0:
                    upper_state = self._get_upper_state(state)
                    upper_action = self.upper_agent.select_action(upper_state, training=False)
                    
                    if upper_action == 0:
                        episode_stats['upper_actions']['wait'] += 1
                    elif upper_action == 1:
                        episode_stats['upper_actions']['depart'] += 1
                else:
                    upper_action = 2  # Nil
                    episode_stats['upper_actions']['nil'] += 1
                
                step_info['upper_action'] = upper_action
                
                # 下层智能体决策
                lower_action = None
                if state.new_request is not None:
                    request = state.new_request
                    lower_state = self._get_lower_state(state, request)
                    lower_action, _, _ = self.lower_agent.select_action(
                        lower_state, training=False
                    )
                    
                    if lower_action == 0:
                        episode_stats['lower_actions']['drone'] += 1
                    elif lower_action == 1:
                        episode_stats['lower_actions']['vehicle'] += 1
                    elif lower_action == 2:
                        episode_stats['lower_actions']['reject'] += 1
                    
                    step_info['new_request'] = {
                        'id': request.id,
                        'arrival_time': request.arrival_time,
                        'deadline': request.deadline,
                        'x': request.x,
                        'y': request.y
                    }
                else:
                    lower_action = 3  # Nil
                    episode_stats['lower_actions']['nil'] += 1
                
                step_info['lower_action'] = lower_action
                
                # 创建动作
                action = Action(
                    upper_action=upper_action,
                    lower_action=lower_action,
                    route_update=None
                )
                
                # 统计车辆行程（在执行step之前）
                if action.upper_action == 1:
                    if len(available_vehicles) > 0:
                        vehicle_id = available_vehicles[0]
                        route_length = len(self.env.vehicle_routes[vehicle_id])
                        if route_length > 0:
                            vehicle_trips += 1
                            total_vehicle_requests += route_length
                
                # 执行动作
                next_state, upper_reward, lower_reward, done, info = self.env.step(action)
                
                step_info['upper_reward'] = upper_reward
                step_info['lower_reward'] = lower_reward
                step_info['delivered_increment'] = info['delivered'] - prev_delivered
                step_info['rejected_increment'] = info['rejected'] - prev_rejected
                
                # 更新奖励
                episode_stats['reward_upper'] += upper_reward
                episode_stats['reward_lower'] += lower_reward
                
                # 统计效率指标
                time_interval = self.env.current_time - prev_time if prev_time > 0 else 0.0
                if time_interval <= 0:
                    time_interval = 1.0
                
                active_vehicles = sum(1 for v in self.env.vehicle_status.values() if v == 1)
                vehicle_active_time += active_vehicles * time_interval
                
                active_drones = sum(1 for d in self.env.drone_status.values() if d == 1)
                drone_active_time += active_drones * time_interval
                
                prev_time = self.env.current_time
                
                # 保存步骤信息（可选，只保存关键步骤）
                if save_detailed and (steps < 10 or steps % 100 == 0 or done):
                    episode_stats['step_details'].append(step_info)
                
                state = next_state
                steps += 1
            
            # Episode结束时的统计
            episode_stats['steps'] = steps
            episode_stats['delivered'] = info['delivered']
            episode_stats['rejected'] = info['rejected']
            episode_stats['total'] = info['total']
            episode_stats['delivery_rate'] = info['delivered'] / info['total'] if info['total'] > 0 else 0.0
            
            # 计算效率指标
            total_time = self.env.current_time if self.env.current_time > 0 else 1.0
            episode_stats['avg_requests_per_trip'] = total_vehicle_requests / vehicle_trips if vehicle_trips > 0 else 0.0
            episode_stats['vehicle_utilization'] = vehicle_active_time / (self.env.num_vehicles * total_time) if total_time > 0 else 0.0
            episode_stats['drone_utilization'] = drone_active_time / (self.env.num_drones * total_time) if total_time > 0 else 0.0
            episode_stats['total_time'] = total_time
            
            # 更新汇总
            summary['episodes'].append(episode + 1)
            summary['delivered'].append(episode_stats['delivered'])
            summary['rejected'].append(episode_stats['rejected'])
            summary['total'].append(episode_stats['total'])
            summary['delivery_rate'].append(episode_stats['delivery_rate'])
            summary['episode_reward_upper'].append(episode_stats['reward_upper'])
            summary['episode_reward_lower'].append(episode_stats['reward_lower'])
            summary['steps'].append(steps)
            summary['upper_wait_count'].append(episode_stats['upper_actions']['wait'])
            summary['upper_depart_count'].append(episode_stats['upper_actions']['depart'])
            summary['lower_drone_count'].append(episode_stats['lower_actions']['drone'])
            summary['lower_vehicle_count'].append(episode_stats['lower_actions']['vehicle'])
            summary['lower_reject_count'].append(episode_stats['lower_actions']['reject'])
            
            episode_details.append(episode_stats)
        
        # 计算总体统计
        results = {
            'summary': {
                'num_episodes': num_episodes,
                'avg_delivered': np.mean(summary['delivered']),
                'std_delivered': np.std(summary['delivered']),
                'avg_rejected': np.mean(summary['rejected']),
                'std_rejected': np.std(summary['rejected']),
                'avg_total': np.mean(summary['total']),
                'avg_delivery_rate': np.mean(summary['delivery_rate']),
                'std_delivery_rate': np.std(summary['delivery_rate']),
                'avg_reward_upper': np.mean(summary['episode_reward_upper']),
                'avg_reward_lower': np.mean(summary['episode_reward_lower']),
                'avg_steps': np.mean(summary['steps']),
                'avg_upper_wait': np.mean(summary['upper_wait_count']),
                'avg_upper_depart': np.mean(summary['upper_depart_count']),
                'avg_lower_drone': np.mean(summary['lower_drone_count']),
                'avg_lower_vehicle': np.mean(summary['lower_vehicle_count']),
                'avg_lower_reject': np.mean(summary['lower_reject_count']),
            },
            'episode_details': episode_details if save_detailed else None
        }
        
        # 打印结果
        self._print_summary(results['summary'])
        
        # 保存结果
        if save_detailed:
            json_path = os.path.join(output_dir, 'evaluation_results.json')
            # 转换numpy类型为Python原生类型
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                return obj
            
            serializable_results = convert_to_serializable(results)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"\n详细结果已保存到: {json_path}")
        
        return results
    
    def _print_summary(self, summary):
        """打印汇总统计"""
        print("\n" + "=" * 80)
        print("评估结果汇总")
        print("=" * 80)
        print(f"\nEpisode数量: {summary['num_episodes']}")
        print(f"\n【送达统计】")
        print(f"  平均送达数: {summary['avg_delivered']:.2f} ± {summary['std_delivered']:.2f}")
        print(f"  平均拒绝数: {summary['avg_rejected']:.2f} ± {summary['std_rejected']:.2f}")
        print(f"  平均总请求数: {summary['avg_total']:.2f}")
        print(f"  平均送达率: {summary['avg_delivery_rate']:.2%} ± {summary['std_delivery_rate']:.2%}")
        
        print(f"\n【奖励统计】")
        print(f"  平均上层奖励: {summary['avg_reward_upper']:.2f}")
        print(f"  平均下层奖励: {summary['avg_reward_lower']:.2f}")
        
        print(f"\n【决策统计】")
        print(f"  上层决策 - 等待: {summary['avg_upper_wait']:.1f}, 出发: {summary['avg_upper_depart']:.1f}")
        print(f"  下层决策 - 无人机: {summary['avg_lower_drone']:.1f}, 车辆: {summary['avg_lower_vehicle']:.1f}, 拒绝: {summary['avg_lower_reject']:.1f}")
        
        print(f"\n【效率统计】")
        print(f"  平均步数: {summary['avg_steps']:.1f}")
        
        print("=" * 80)
    
    def compare_with_baseline(self, num_episodes=10, model_path_upper=None, 
                             model_path_lower=None, output_dir='test_results'):
        """
        与基准方法对比
        
        Args:
            num_episodes: 评估episode数量
            model_path_upper: 上层智能体模型路径
            model_path_lower: 下层智能体模型路径
            output_dir: 输出目录
        """
        print("\n" + "=" * 80)
        print("方法对比评估")
        print("=" * 80)
        
        # 评估HDDRL
        print("\n【评估HDDRL方法】")
        hddrl_results = self.evaluate_detailed(
            num_episodes=num_episodes,
            model_path_upper=model_path_upper,
            model_path_lower=model_path_lower,
            save_detailed=False,
            output_dir=output_dir
        )
        
        # 评估贪心基准（简化版，不保存详细信息）
        print("\n【评估贪心基准方法】")
        greedy_results = self._evaluate_greedy_baseline(num_episodes)
        
        # 对比分析
        print("\n" + "=" * 80)
        print("对比分析")
        print("=" * 80)
        
        hddrl_summary = hddrl_results['summary']
        greedy_summary = greedy_results['summary']
        
        print(f"\n送达数对比:")
        print(f"  HDDRL: {hddrl_summary['avg_delivered']:.2f} ± {hddrl_summary['std_delivered']:.2f}")
        print(f"  贪心:  {greedy_summary['avg_delivered']:.2f} ± {greedy_summary['std_delivered']:.2f}")
        if greedy_summary['avg_delivered'] > 0:
            improvement = ((hddrl_summary['avg_delivered'] - greedy_summary['avg_delivered']) /
                          greedy_summary['avg_delivered'] * 100)
            print(f"  提升:  {improvement:+.2f}%")
        
        print(f"\n送达率对比:")
        print(f"  HDDRL: {hddrl_summary['avg_delivery_rate']:.2%} ± {hddrl_summary['std_delivery_rate']:.2%}")
        print(f"  贪心:  {greedy_summary['avg_delivery_rate']:.2%} ± {greedy_summary['std_delivery_rate']:.2%}")
        if greedy_summary['avg_delivery_rate'] > 0:
            improvement = ((hddrl_summary['avg_delivery_rate'] - greedy_summary['avg_delivery_rate']) /
                          greedy_summary['avg_delivery_rate'] * 100)
            print(f"  提升:  {improvement:+.2f}%")
        
        # 绘制对比图
        self._plot_comparison(hddrl_results, greedy_results, output_dir)
    
    def _evaluate_greedy_baseline(self, num_episodes):
        """评估贪心基准方法"""
        results = {
            'delivered': [],
            'rejected': [],
            'total': [],
            'delivery_rate': []
        }
        
        for episode in tqdm(range(num_episodes), desc="贪心方法"):
            state = self.env.reset()
            done = False
            steps = 0
            
            while not done and steps < 10000:
                # 贪心策略：总是出发，选择最早可用的资源
                upper_action = 1 if len([i for i in range(self.env.num_vehicles)
                                         if self.env.vehicle_status[i] == 0 and
                                         self.env.vehicle_available_time[i] <= self.env.current_time]) > 0 else 2
                
                lower_action = 3
                if state.new_request is not None:
                    request = state.new_request
                    
                    # 检查可行性
                    vehicle_feasible = False
                    for vehicle_id in range(self.env.num_vehicles):
                        if self.env.vehicle_status[vehicle_id] == 0:
                            feasible, _ = check_vehicle_feasibility(
                                request, self.env.vehicle_routes[vehicle_id],
                                self.env.current_time, self.env.vehicle_available_time[vehicle_id]
                            )
                            if feasible:
                                vehicle_feasible = True
                                break
                    
                    drone_feasible = False
                    for drone_id in range(self.env.num_drones):
                        feasible = check_drone_feasibility(
                            request, self.env.current_time, self.env.drone_available_time[drone_id]
                        )
                        if feasible:
                            drone_feasible = True
                            break
                    
                    # 贪心选择
                    if vehicle_feasible and drone_feasible:
                        earliest_vehicle = self.env.get_earliest_vehicle_time()
                        earliest_drone = self.env.get_earliest_drone_time()
                        lower_action = 1 if earliest_vehicle <= earliest_drone else 0
                    elif vehicle_feasible:
                        lower_action = 1
                    elif drone_feasible:
                        lower_action = 0
                    else:
                        lower_action = 2
                
                action = Action(upper_action=upper_action, lower_action=lower_action, route_update=None)
                state, _, _, done, info = self.env.step(action)
                steps += 1
            
            results['delivered'].append(info['delivered'])
            results['rejected'].append(info['rejected'])
            results['total'].append(info['total'])
            results['delivery_rate'].append(info['delivered'] / info['total'] if info['total'] > 0 else 0)
        
        return {
            'summary': {
                'num_episodes': num_episodes,
                'avg_delivered': np.mean(results['delivered']),
                'std_delivered': np.std(results['delivered']),
                'avg_rejected': np.mean(results['rejected']),
                'avg_total': np.mean(results['total']),
                'avg_delivery_rate': np.mean(results['delivery_rate']),
                'std_delivery_rate': np.std(results['delivery_rate'])
            }
        }
    
    def _plot_comparison(self, hddrl_results, greedy_results, output_dir):
        """绘制对比图"""
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        hddrl_summary = hddrl_results['summary']
        greedy_summary = greedy_results['summary']
        
        # 送达数对比
        methods = ['HDDRL', 'Greedy']
        delivered = [hddrl_summary['avg_delivered'], greedy_summary['avg_delivered']]
        delivered_std = [hddrl_summary['std_delivered'], greedy_summary['std_delivered']]
        
        axes[0].bar(methods, delivered, yerr=delivered_std, capsize=5, alpha=0.7, 
                   color=['#2ecc71', '#3498db'])
        axes[0].set_ylabel('Delivered Count', fontsize=12)
        axes[0].set_title('Average Delivered Count Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 送达率对比
        delivery_rates = [hddrl_summary['avg_delivery_rate'], greedy_summary['avg_delivery_rate']]
        delivery_rates_std = [hddrl_summary['std_delivery_rate'], greedy_summary['std_delivery_rate']]
        
        axes[1].bar(methods, delivery_rates, yerr=delivery_rates_std, capsize=5, alpha=0.7,
                   color=['#2ecc71', '#3498db'])
        axes[1].set_ylabel('Delivery Rate', fontsize=12)
        axes[1].set_title('Average Delivery Rate Comparison', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, 'method_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"\n对比图已保存到: {comparison_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='详细的模型测试和评估脚本')
    parser.add_argument('--episodes', type=int, default=10, help='评估episode数量')
    parser.add_argument('--upper-model', type=str, default=None, help='上层模型路径')
    parser.add_argument('--lower-model', type=str, default=None, help='下层模型路径')
    parser.add_argument('--model-dir', type=str, default='models', help='模型目录')
    parser.add_argument('--compare', action='store_true', help='与基准方法对比')
    parser.add_argument('--output-dir', type=str, default='test_results', help='输出目录')
    parser.add_argument('--no-details', action='store_true', help='不保存详细步骤信息')
    
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
    
    evaluator = DetailedEvaluator()
    
    if args.compare:
        evaluator.compare_with_baseline(
            num_episodes=args.episodes,
            model_path_upper=upper_path,
            model_path_lower=lower_path,
            output_dir=args.output_dir
        )
    else:
        evaluator.evaluate_detailed(
            num_episodes=args.episodes,
            model_path_upper=upper_path,
            model_path_lower=lower_path,
            save_detailed=not args.no_details,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()