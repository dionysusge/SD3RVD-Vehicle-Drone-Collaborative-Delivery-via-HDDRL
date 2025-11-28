# SD3RVD HDDRL 项目

基于分层深度强化学习（HDDRL）的车辆-无人机协同同日达配送系统

## 项目概述

本项目实现了论文《Optimizing same-day delivery with vehicles and drones: A hierarchical deep reinforcement learning approach》中提出的HDDRL算法，用于解决同日达配送调度与路径问题（SD3RVD）。

### 核心特性

- **三层架构**：上层智能体（DQN）负责车辆调度，下层智能体（PPO）负责请求分配，启发式算法负责路径规划
- **完整环境模拟**：实现Route-based MDP环境，支持动态请求生成和实时决策
- **多种分布支持**：支持均匀/非均匀时间分布和正态/幂律空间分布
- **性能评估**：包含完整的训练和测试脚本，支持与基准方法对比

## 项目结构

```
RL_truck_drone/
├── config/                      # 配置文件
│   ├── environment_config.py   # 环境参数配置
│   └── training_config.py      # 训练超参数配置
├── environment/                 # 环境模块
│   ├── route_mdp.py            # Route-based MDP 核心环境
│   ├── state.py                # 状态定义
│   ├── action.py               # 动作空间
│   ├── reward.py               # 奖励函数
│   └── request_generator.py    # 动态请求生成器
├── agents/                      # 智能体模块
│   ├── upper_agent.py          # 上层智能体（DQN）
│   ├── lower_agent.py          # 下层智能体（PPO）
│   ├── dqn.py                  # DQN 网络实现
│   └── ppo.py                  # PPO 网络实现
├── heuristics/                  # 启发式算法
│   ├── vehicle_routing.py      # 车辆路径规划
│   ├── drone_routing.py        # 无人机路径规划
│   └── feasibility_check.py    # 可行性检查
├── utils/                       # 工具模块
│   ├── rbu.py                  # 请求缓冲区
│   ├── distance.py             # 距离计算
│   ├── visualization.py       # 可视化工具
│   └── logger.py               # 日志记录
├── train.py                     # 训练脚本
├── test.py                      # 测试/评估脚本
├── requirements.txt             # 依赖包
└── README.md                    # 项目说明
```

## 环境参数

根据论文Table 2和Table 4配置：

- **时间参数**：
  - 运营时间：T_max = 420分钟（7小时）
  - 服务承诺：δ = 240分钟（4小时内送达）
  - 车辆/无人机工作时长：600分钟（10小时）

- **车辆参数**：
  - 速度：30 km/h
  - 装载/卸货时间：3分钟
  - 道路距离系数：1.5（欧氏距离 × 1.5）

- **无人机参数**：
  - 速度：40 km/h
  - 装载/卸货时间：3分钟
  - 充电时间：20分钟
  - 容量：1个包裹

## 安装

1. 克隆项目：
```bash
git clone <repository_url>
cd RL_truck_drone
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

运行训练脚本：
```bash
python train.py
```

训练配置可在 `config/training_config.py` 中修改：
- 训练episode数量
- DQN和PPO的超参数
- 车队规模（车辆和无人机数量）

训练过程中会：
- 自动保存模型到 `models/` 目录
- 记录日志到 `logs/` 目录
- 定期评估模型性能

### 测试/评估

运行测试脚本：
```bash
python test.py
```

测试脚本会：
- 加载训练好的模型（如果存在）
- 评估HDDRL方法性能
- 与贪心基准方法对比
- 生成性能对比图（保存到 `results/` 目录）

### 自定义配置

#### 修改环境参数

编辑 `config/environment_config.py`：
```python
T_MAX = 420  # 运营时间
DELTA = 240  # 服务承诺时间
V_E = 30.0   # 车辆速度
V_D = 40.0   # 无人机速度
```

#### 修改训练参数

编辑 `config/training_config.py`：
```python
TRAINING_CONFIG = {
    'num_episodes': 1000,  # 训练episode数
    'max_steps_per_episode': 10000,
    'save_freq': 100,  # 模型保存频率
    'eval_freq': 50,   # 评估频率
}
```

#### 修改车队规模

编辑 `config/training_config.py`：
```python
FLEET_CONFIG = {
    'num_vehicles': 5,  # 车辆数量
    'num_drones': 5     # 无人机数量
}
```

## 算法架构

### 上层智能体（Upper Agent）

- **算法**：DQN（Deep Q-Network）
- **职责**：车辆调度决策（等待/出发）
- **状态**：4维向量 [t_k, t_k^AD, t_k^AV, t_k^remain]
- **动作**：等待(0) / 出发(1)

### 下层智能体（Lower Agent）

- **算法**：PPO（Proximal Policy Optimization）
- **职责**：请求分配决策
- **状态**：7维向量（请求属性 + 系统状态 + 启发式结果）
- **动作**：分配给无人机(0) / 分配给车辆(1) / 拒绝(2)

### 启发式算法层

- **车辆路径规划**：最低成本插入法
- **无人机路径规划**：FIFO（先入先出）策略
- **可行性检查**：实时计算时间约束和容量约束

### 请求缓冲区（RBU）

- 存储两次车辆出发之间的请求
- 实现层次间协作和奖励聚合

## 性能指标

- **送达数量**：成功送达的请求总数
- **送达率**：送达数量 / 总请求数量
- **拒绝率**：拒绝的请求数量 / 总请求数量

## 文件说明

### 核心模块

- `environment/route_mdp.py`：核心环境实现，包含状态转移、奖励计算等
- `agents/upper_agent.py`：上层智能体封装
- `agents/lower_agent.py`：下层智能体封装
- `heuristics/`：启发式算法实现

### 工具模块

- `utils/rbu.py`：请求缓冲区实现
- `utils/distance.py`：距离和时间计算
- `utils/visualization.py`：可视化工具
- `utils/logger.py`：日志记录工具

## 注意事项

1. **GPU支持**：代码自动检测并使用GPU（如果可用），否则使用CPU
2. **随机种子**：建议在训练前设置随机种子以保证可复现性
3. **内存使用**：经验回放缓冲区大小可根据内存情况调整
4. **训练时间**：完整训练可能需要数小时，建议使用GPU加速

## 扩展功能

### 支持新的请求分布

在 `environment/request_generator.py` 中添加新的分布类型：
```python
def _generate_location(self):
    if self.space_dist == 'custom':
        # 实现自定义分布
        pass
```

### 添加新的基准方法

在 `test.py` 中添加新的评估方法：
```python
def evaluate_custom_baseline(self, num_episodes=10):
    # 实现自定义基准方法
    pass
```

