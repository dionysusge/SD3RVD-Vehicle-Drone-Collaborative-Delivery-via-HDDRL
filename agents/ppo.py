"""
PPO 网络实现（用于下层智能体）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Actor(nn.Module):
    """Actor网络（策略网络）"""
    
    def __init__(self, state_dim, action_dim, hidden_layers=[64, 64]):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_layers: 隐藏层维度列表，默认[64, 64]（论文Table 4）
        """
        super(Actor, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        # 获取除了最后一层的所有层
        hidden = x
        for i in range(len(self.network) - 1):
            hidden = self.network[i](hidden)
        # 最后一层输出后应用softmax
        logits = self.network[-1](hidden)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    """Critic网络（价值网络）"""
    
    def __init__(self, state_dim, hidden_layers=[64, 64]):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            hidden_layers: 隐藏层维度列表，默认[64, 64]（论文Table 4）
        """
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """前向传播"""
        return self.network(x)


class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim, action_dim, actor_lr=0.0003, critic_lr=0.01,
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, value_coef=0.5,
                 entropy_coef=0.2, update_epochs=10, batch_size=128,
                 hidden_layers=[64, 64]):
        """
        初始化PPO智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            actor_lr: Actor学习率（论文：0.0003）
            critic_lr: Critic学习率（论文：0.01）
            gamma: 折扣因子
            gae_lambda: GAE参数
            clip_epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵系数（论文：0.2）
            update_epochs: 更新轮数
            batch_size: 批次大小（论文：128）
            hidden_layers: 隐藏层维度列表（论文：[64, 64]）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.actor = Actor(state_dim, action_dim, hidden_layers).to(self.device)
        self.critic = Critic(state_dim, hidden_layers).to(self.device)
        
        # 初始化网络权重（避免NaN）
        self._initialize_weights()
        
        # 优化器（分离的学习率）
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 经验缓冲区
        self.reset_buffer()
    
    def _initialize_weights(self):
        """使用Xavier（Glorot）初始化网络权重，初始策略偏向拒绝"""
        # 初始化Actor网络的所有层
        for m in self.actor.modules():
            if isinstance(m, nn.Linear):
                # Xavier均匀初始化（标准实现，gain=1.0）
                nn.init.xavier_uniform_(m.weight)
                # 偏置初始化为0（标准做法）
                nn.init.constant_(m.bias, 0.0)
        
        # 调整Actor最后一层的偏置，使初始策略偏向拒绝
        # 动作: 0=无人机, 1=车辆, 2=拒绝
        # 策略: 提高拒绝动作的初始logit，降低接受动作的初始logit
        # 这样在训练初期，智能体更倾向于拒绝，从保守策略开始学习
        last_layer = self.actor.network[-1]
        if isinstance(last_layer, nn.Linear) and last_layer.out_features == self.action_dim:
            with torch.no_grad():
                # 拒绝动作的偏置设为正，接受动作的偏置设为负
                last_layer.bias[0] = -0.5  # 无人机（降低初始概率）
                last_layer.bias[1] = -0.5  # 车辆（降低初始概率）
                last_layer.bias[2] = 0.5   # 拒绝（提高初始概率）
        
        # 初始化Critic网络
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                # Xavier均匀初始化（标准实现，gain=1.0）
                nn.init.xavier_uniform_(m.weight)
                # 偏置初始化为0（标准做法）
                nn.init.constant_(m.bias, 0.0)
    
    def reset_buffer(self):
        """重置经验缓冲区"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state, training=True):
        """
        选择动作（带动作掩码，与论文一致）
        
        Args:
            state: 当前状态（7维向量，包含可行性信息）
            training: 是否在训练模式
        
        Returns:
            tuple: (动作, 对数概率, 价值)
        """
        # 检查状态中是否有NaN或Inf
        state_array = np.array(state, dtype=np.float32)
        if np.isnan(state_array).any() or np.isinf(state_array).any():
            # 如果状态包含NaN或Inf，使用默认策略（均匀分布）
            print(f"Warning: State contains NaN or Inf: {state_array}")
            # 替换NaN和Inf为0
            state_array = np.nan_to_num(state_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        state_tensor = torch.FloatTensor(state_array).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            
            # 检查输出是否有NaN
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print(f"Warning: Actor output contains NaN or Inf, using uniform distribution")
                # 使用均匀分布作为后备
                action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
            
            # 应用动作掩码（根据可行性动态屏蔽无效动作，与论文一致）
            # 状态向量中：state[4] = vehicle_feasible, state[6] = drone_feasible
            vehicle_feasible = state_array[4] > 0.5
            drone_feasible = state_array[6] > 0.5
            
            # 创建动作掩码（1.0表示允许，0.0表示屏蔽）
            action_mask = torch.ones_like(action_probs)
            
            if not vehicle_feasible and not drone_feasible:
                # 两者都不可行，只能拒绝
                action_mask[0, 0] = 0.0  # 屏蔽无人机
                action_mask[0, 1] = 0.0  # 屏蔽车辆
                action_mask[0, 2] = 1.0  # 只允许拒绝
            elif not vehicle_feasible:
                # 车辆不可行，屏蔽车辆动作（但允许无人机和拒绝）
                action_mask[0, 1] = 0.0
            elif not drone_feasible:
                # 无人机不可行，屏蔽无人机动作（但允许车辆和拒绝）
                action_mask[0, 0] = 0.0
            
            # 应用掩码：将无效动作的概率设为0
            action_probs = action_probs * action_mask
            # 重新归一化有效动作的概率
            prob_sum = action_probs.sum(dim=-1, keepdim=True)
            if prob_sum.item() > 1e-8:
                action_probs = action_probs / prob_sum
            else:
                # 如果所有动作都被屏蔽（理论上不应该发生），使用掩码本身作为均匀分布
                action_probs = action_mask / action_mask.sum(dim=-1, keepdim=True)
            
            # 确保概率有效
            action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            
            value = self.critic(state_tensor)
            
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, next_value=0):
        """
        计算GAE（Generalized Advantage Estimation）
        
        Args:
            next_value: 下一个状态的价值
        
        Returns:
            tuple: (优势, 回报)
        """
        rewards = np.array(self.rewards)
        values = np.array(self.values + [next_value])
        dones = np.array(self.dones)
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def train_step(self):
        """执行一步训练"""
        if len(self.states) == 0:
            return None
        
        # 计算GAE
        next_value = 0  # 假设episode结束
        advantages, returns = self.compute_gae(next_value)
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # 归一化优势（避免除零和NaN）
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std
        else:
            advantages = advantages - adv_mean  # 如果std为0，只减去均值
        
        # 用于记录总损失
        total_losses = []
        
        # 多轮更新
        for _ in range(self.update_epochs):
            # 随机打乱
            indices = torch.randperm(len(states))
            
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor更新
                action_probs = self.actor(batch_states)
                # 检查NaN和Inf
                if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                    print("Warning: NaN or Inf in action_probs, skipping this batch")
                    continue
                # 确保概率和为1（数值稳定性）
                action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
                action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                dist = torch.distributions.Categorical(action_probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                # 梯度裁剪，防止梯度爆炸，提高训练稳定性
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                
                # Critic更新（使用Huber损失代替MSE，对异常值更鲁棒）
                values = self.critic(batch_states)
                if values.dim() > 1:
                    values = values.squeeze()
                # 确保维度匹配
                if values.dim() == 0:
                    values = values.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                # delta=1.0表示误差超过1时使用线性损失，而不是平方损失
                # 由于PPO的奖励范围较小（-2到+1.5），delta设置较小
                critic_loss = F.huber_loss(values, batch_returns, delta=1.0)
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                # 梯度裁剪，防止梯度爆炸，提高训练稳定性
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.critic_optimizer.step()
                
                # 记录总损失（用于返回）
                total_losses.append(actor_loss.item() + critic_loss.item())
        
        # 清空缓冲区
        self.reset_buffer()
        
        # 返回平均损失
        return np.mean(total_losses) if len(total_losses) > 0 else None
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
