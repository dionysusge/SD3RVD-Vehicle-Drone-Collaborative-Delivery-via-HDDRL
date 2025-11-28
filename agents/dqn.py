"""
DQN 网络实现（用于上层智能体）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DQN(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim, action_dim, hidden_layers=[128, 128]):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_layers: 隐藏层维度列表，默认[128, 128]（论文Table 4）
        """
        super(DQN, self).__init__()
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
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一个状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        从缓冲区采样一批经验
        
        Args:
            batch_size: 批次大小
        
        Returns:
            tuple: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """返回缓冲区大小"""
        return len(self.buffer)


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim, action_dim, lr=1e-4, lr_decay_base=0.96, 
                 lr_decay_interval=6000, gamma=0.9,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.999,
                 memory_size=10000, batch_size=64, target_update_freq=100,
                 hidden_layers=[128, 128]):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 初始学习率
            lr_decay_base: 学习率衰减基数（论文：0.96）
            lr_decay_interval: 学习率衰减间隔（论文：每6000个epoch）
            gamma: 折扣因子（论文：0.9）
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率（论文：0.1）
            epsilon_decay: 探索率衰减
            memory_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
            hidden_layers: 隐藏层维度列表（论文：[128, 128]）
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initial_lr = lr
        self.lr_decay_base = lr_decay_base
        self.lr_decay_interval = lr_decay_interval
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.epoch = 0  # 用于学习率衰减
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 网络
        self.q_network = DQN(state_dim, action_dim, hidden_layers).to(self.device)
        self.target_network = DQN(state_dim, action_dim, hidden_layers).to(self.device)
        
        # 初始化网络权重，使初始策略偏向出发（当有请求时）
        self._initialize_weights()
        
        # 将初始化后的权重复制到目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放
        self.memory = ReplayBuffer(memory_size)
    
    def _initialize_weights(self):
        """
        使用Xavier（Glorot）初始化网络权重
        改进版：更平衡的初始化，让智能体通过学习发现最优策略
        """
        # 对所有Linear层使用标准的Xavier均匀初始化
        layer_count = 0
        total_layers = sum(1 for m in self.q_network.modules() if isinstance(m, nn.Linear))
        
        for m in self.q_network.modules():
            if isinstance(m, nn.Linear):
                layer_count += 1
                # Xavier均匀初始化（标准实现，gain=1.0）
                nn.init.xavier_uniform_(m.weight)
                
                # 偏置初始化
                if layer_count == total_layers:
                    # ✅ 改进：更平衡的初始化
                    # 最后一层（输出层）：轻度引导等待，但不过分
                    # 动作0=等待（初始Q值略高），动作1=出发（初始Q值略低）
                    m.bias.data[0] = 0.5   # 等待动作，轻度提高初始Q值
                    m.bias.data[1] = -0.5  # 出发动作，轻度降低初始Q值
                else:
                    # 其他层：标准初始化
                    nn.init.constant_(m.bias, 0.0)
        
        # 注意：目标网络的权重会在后面通过load_state_dict从Q网络复制
    
    def update_learning_rate(self, epoch):
        """
        更新学习率（指数衰减：1.0e-4 * 0.96^(epoch/6000)）
        
        Args:
            epoch: 当前epoch数
        """
        self.epoch = epoch
        lr = self.initial_lr * (self.lr_decay_base ** (epoch / self.lr_decay_interval))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def select_action(self, state, training=True):
        """
        选择动作（ε-贪婪策略）
        
        Args:
            state: 当前状态
            training: 是否在训练模式
        
        Returns:
            int: 选择的动作
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """执行一步训练"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算当前Q值
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # 计算损失（使用Huber损失代替MSE，对大误差更鲁棒，减少损失值）
        # delta=10.0表示误差超过10时使用线性损失，而不是平方损失
        loss = nn.HuberLoss(delta=10.0)(q_value, target_q_value)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸，提高训练稳定性
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # 衰减探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
