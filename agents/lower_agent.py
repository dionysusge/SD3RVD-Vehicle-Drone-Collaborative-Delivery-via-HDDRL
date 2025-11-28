"""
下层智能体（请求分配决策）
使用PPO实现
"""
from agents.ppo import PPOAgent
from config.training_config import PPO_CONFIG


class LowerAgent:
    """下层智能体：负责请求分配决策"""
    
    def __init__(self, config=None):
        """
        初始化下层智能体
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        if config is None:
            config = PPO_CONFIG
        
        self.agent = PPOAgent(
            state_dim=config['state_dim'],
            action_dim=config['action_dim'],
            actor_lr=config['actor_lr'],
            critic_lr=config['critic_lr'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_epsilon=config['clip_epsilon'],
            value_coef=config['value_coef'],
            entropy_coef=config['entropy_coef'],
            update_epochs=config['update_epochs'],
            batch_size=config['batch_size'],
            hidden_layers=config.get('hidden_layers', [64, 64])
        )
    
    def select_action(self, state, training=True):
        """
        选择动作（分配给无人机、车辆或拒绝）
        
        Args:
            state: 7维状态向量
            training: 是否在训练模式
        
        Returns:
            tuple: (动作, 对数概率, 价值)
                     动作: 0=无人机, 1=车辆, 2=拒绝
        """
        return self.agent.select_action(state, training)
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        """存储经验"""
        self.agent.store_transition(state, action, log_prob, value, reward, done)
    
    def train_step(self):
        """执行一步训练"""
        return self.agent.train_step()
    
    def save(self, filepath):
        """保存模型"""
        self.agent.save(filepath)
    
    def load(self, filepath):
        """加载模型"""
        self.agent.load(filepath)
