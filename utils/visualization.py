"""
可视化工具
"""
import matplotlib.pyplot as plt
import numpy as np
from config.environment_config import WAREHOUSE_X, WAREHOUSE_Y


def plot_requests(requests, title="客户请求分布"):
    """
    绘制客户请求分布图
    
    Args:
        requests: 请求列表
        title: 图表标题
    """
    if len(requests) == 0:
        return
    
    x_coords = [r.x for r in requests]
    y_coords = [r.y for r in requests]
    
    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords, alpha=0.6, s=50, label='客户位置')
    plt.scatter(WAREHOUSE_X, WAREHOUSE_Y, color='red', s=200, marker='s', label='仓库')
    plt.xlabel('X坐标 (km)')
    plt.ylabel('Y坐标 (km)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    return plt


def plot_training_curves(rewards_upper, rewards_lower, delivered, save_path='results/training_curves.png'):
    """
    绘制训练曲线
    
    Args:
        rewards_upper: 上层智能体奖励列表
        rewards_lower: 下层智能体奖励列表
        delivered: 送达数量列表
        save_path: 保存路径
    """
    import os
    os.makedirs('results', exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    episodes = range(1, len(rewards_upper) + 1)
    
    # 上层奖励
    axes[0].plot(episodes, rewards_upper, alpha=0.6, label='原始奖励')
    if len(rewards_upper) > 10:
        # 移动平均
        window = min(50, len(rewards_upper) // 10)
        moving_avg = np.convolve(rewards_upper, np.ones(window)/window, mode='valid')
        axes[0].plot(episodes[window-1:], moving_avg, label=f'{window}期移动平均', linewidth=2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('奖励')
    axes[0].set_title('上层智能体奖励')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 下层奖励
    axes[1].plot(episodes, rewards_lower, alpha=0.6, label='原始奖励')
    if len(rewards_lower) > 10:
        window = min(50, len(rewards_lower) // 10)
        moving_avg = np.convolve(rewards_lower, np.ones(window)/window, mode='valid')
        axes[1].plot(episodes[window-1:], moving_avg, label=f'{window}期移动平均', linewidth=2)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('奖励')
    axes[1].set_title('下层智能体奖励')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 送达数量
    axes[2].plot(episodes, delivered, alpha=0.6, label='原始值')
    if len(delivered) > 10:
        window = min(50, len(delivered) // 10)
        moving_avg = np.convolve(delivered, np.ones(window)/window, mode='valid')
        axes[2].plot(episodes[window-1:], moving_avg, label=f'{window}期移动平均', linewidth=2)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('送达数量')
    axes[2].set_title('成功送达数量')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线已保存到: {save_path}")
    plt.close()
