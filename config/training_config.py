"""
训练超参数配置
"""

# DQN 参数（上层智能体）- 稳定性优化，防止策略震荡
DQN_CONFIG = {
    'learning_rate': 5e-4,  # 从1e-4增加到5e-4，提高学习速度
    'lr_decay_base': 0.99,  # 从0.96改为0.98，衰减更慢
    'lr_decay_interval': 200,  # 学习率衰减间隔（每6000个epoch）
    'gamma': 0.9,  # 修正为0.9（论文Table 4）
    'epsilon_start': 0.5,  # 降低初始探索率，避免频繁空车出发（从1.0改为0.5）
    'epsilon_end': 0.1,  # 修正为0.1（论文Table 4）
    'epsilon_decay': 0.995,  # 从0.999改为0.995，加快衰减速度
    'batch_size': 256,  # 增大batch size以利用24G GPU（原64）
    'memory_size': 50000,  # 增大经验回放缓冲区（原10000）
    'target_update_freq': 100,
    'hidden_layers': [128, 256, 128],  # 改成3层， 原文2层，每层128个单元（论文Table 4）
    'state_dim': 5,  # t_k, t_k^AD, t_k^AV, t_k^remain, n_requests（新增RBU维度）
    'action_dim': 2  # 等待, 出发

}

# PPO 参数（下层智能体）- 按论文Table 4修正，优化GPU使用
PPO_CONFIG = {
    'actor_lr': 0.0003,  # Actor学习率（论文Table 4）
    'critic_lr': 0.0003,  # Critic学习率（从0.01降低到0.0003，与actor相同，提高稳定性）
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'clip_epsilon': 0.2,
    'value_coef': 0.5,
    'entropy_coef': 0.2,  # 修正为0.2（论文Table 4）
    'update_epochs': 10,
    'batch_size': 512,  # 增大batch size以利用24G GPU（原128）
    'hidden_layers': [128, 128, 128],  # 2层，每层64个单元（论文Table 4）
    'state_dim': 7,  # 7维状态特征
    'action_dim': 3  # 无人机, 车辆, 拒绝
}

# 训练参数
TRAINING_CONFIG = {
    'num_episodes': 10000,
    'max_steps_per_episode': 10000,
    'save_freq': 100,
    'eval_freq': 50,
    'eval_episodes': 10,
    'log_freq': 10
}

# 车队规模（可配置）
FLEET_CONFIG = {
    'num_vehicles': 3,  
    'num_drones': 5     
}

# 场景配置（请求生成）
# 论文定义的四种场景类型：
# 1. 'homo-normal': 均匀时间 + 正态空间（最简单，适合初期训练）
# 2. 'homo-power': 均匀时间 + 幂律空间（中等难度，测试空间聚集）
# 3. 'heter-normal': 非均匀时间 + 正态空间（中等难度，测试高峰期）
# 4. 'heter-power': 非均匀时间 + 幂律空间（最难，测试双重挑战）
SCENARIO_CONFIG = {
    'scenario_type': 'homo-normal',  # 默认场景：均匀-正态
    'lambda_rate': 0.5,  # 泊松过程参数（每分钟平均到达数）
    
    # 场景映射（自动解析，无需手动修改）
    'scenarios': {
        'homo-normal': {'time_dist': 'homo', 'space_dist': 'normal'},
        'homo-power': {'time_dist': 'homo', 'space_dist': 'power'},
        'heter-normal': {'time_dist': 'heter', 'space_dist': 'normal'},
        'heter-power': {'time_dist': 'heter', 'space_dist': 'power'}
    }
}
