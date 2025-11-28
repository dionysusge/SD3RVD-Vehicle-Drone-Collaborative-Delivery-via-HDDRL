"""
请求缓冲区（RBU - Request Buffer Unit）
用于上层和下层智能体之间的协作
"""
from environment.request_generator import Request


class RequestBuffer:
    """请求缓冲区类"""
    
    def __init__(self):
        """初始化请求缓冲区"""
        self.requests = []  # 存储的请求列表
        # 已分配的请求，存储格式：(request, success, reward)
        # success: 是否成功分配（用于上层奖励计算）
        # reward: 下层智能体的奖励值（可以后续更新）
        self.allocated_requests = []  # [(request, success, reward), ...]
    
    def add_request(self, request):
        """
        添加请求到缓冲区
        
        Args:
            request: 请求对象
        """
        self.requests.append(request)
    
    def allocate_request(self, request, success, reward=None):
        """
        标记请求已分配
        
        Args:
            request: 请求对象
            success: 是否成功分配（用于上层奖励计算）
            reward: 下层智能体的奖励值（可选，如果为None则根据success推断）
        """
        if reward is None:
            # 根据success推断奖励
            reward = 1.0 if success else 0.0
        self.allocated_requests.append((request, success, reward))
    
    def update_request_reward(self, request, new_reward, new_success=None):
        """
        更新请求的奖励值（用于延迟奖励更新）
        
        Args:
            request: 请求对象
            new_reward: 新的奖励值
            new_success: 新的成功状态（可选，如果为None则根据reward推断）
        """
        for i, (req, success, reward) in enumerate(self.allocated_requests):
            if req.id == request.id:
                # 更新奖励和成功状态
                if new_success is None:
                    # 根据奖励推断成功状态
                    new_success = (new_reward > 0)
                self.allocated_requests[i] = (req, new_success, new_reward)
                break
    
    def get_successful_count(self):
        """
        获取成功分配的请求数量（用于上层智能体奖励）
        基于更新后的success状态计算
        
        Returns:
            int: 成功分配的请求数量
        """
        return sum(1 for _, success, _ in self.allocated_requests if success)
    
    def get_vehicle_requests(self):
        """
        获取分配给车辆的请求列表（用于等待超时检测）
        
        Returns:
            list: 分配给车辆的请求列表
        """
        vehicle_requests = []
        for req, success, reward in self.allocated_requests:
            if success and req.assigned_to and req.assigned_to.startswith('vehicle_'):
                vehicle_requests.append(req)
        return vehicle_requests
    
    def clear(self):
        """清空缓冲区（车辆出发时调用）"""
        self.requests = []
        self.allocated_requests = []
    
    def get_pending_requests(self):
        """
        获取待分配的请求列表
        
        Returns:
            list: 待分配的请求列表
        """
        return [req for req in self.requests if not req.assigned]
    
    def __len__(self):
        """返回缓冲区中请求的数量"""
        return len(self.requests)
