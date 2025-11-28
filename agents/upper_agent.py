"""
上层智能体（车辆调度决策）
使用DQN实现
"""
from agents.dqn import DQNAgent
from config.training_config import DQN_CONFIG


class UpperAgent:
    """上层智能体：负责车辆调度决策"""
    
    def __init__(self, config=None):
        """
        初始化上层智能体
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = DQN_CONFIG
        
        self.agent = DQNAgent(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            lr=config['learning_rate'],
            lr_decay_base=config.get('lr_decay_base', 0.96),
            lr_decay_interval=config.get('lr_decay_interval', 6000),
            gamma=config['gamma'],
            epsilon_start=config['epsilon_start'],
            epsilon_end=config['epsilon_end'],
            epsilon_decay=config['epsilon_decay'],
            memory_size=config['memory_size'],
            batch_size=config['batch_size'],
            target_update_freq=config['target_update_freq'],
            hidden_layers=config.get('hidden_layers', [128, 128])
        )
    
    def select_action(self, state, training=True):
        """
        选择动作（等待或出发）
        
        Args:
            state: 4维状态向量 [t_k, t_k^AD, t_k^AV, t_k^remain]
            training: 是否在训练模式
        
        Returns:
            int: 动作 (0=等待, 1=出发)
        """
        return self.agent.select_action(state, training)
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.agent.store_transition(state, action, reward, next_state, done)
    
    def train_step(self):
        """执行一步训练"""
        return self.agent.train_step()
    
    def save(self, filepath):
        """保存模型"""
        self.agent.save(filepath)
    
    def load(self, filepath):
        """加载模型"""
        self.agent.load(filepath)
