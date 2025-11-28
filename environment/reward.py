"""
奖励函数定义
优化版本：提高送达率的奖励设计
"""

class RewardCalculator:
    """奖励计算器"""
    
    @staticmethod
    def calculate_upper_reward(num_vehicle_requests, num_successful_requests=None):
        """
        计算上层智能体的奖励（优化版：强化批量送达的对比度）
        
        Args:
            num_vehicle_requests: 当前车辆路线中的请求数量（主要奖励）
            num_successful_requests: 从上次车辆出发到本次出发之间成功分配的请求数量（向后兼容，已弃用）
        
        Returns:
            float: 奖励值
        
        奖励设计：
        - 0个请求（空车）：-5.0（严厉惩罚，避免空车出发）
        - 1个请求：+1.0（基础奖励，但不鼓励）
        - 2-4个请求：线性增长 +2.0, +3.5, +5.5（鼓励积累）
        - 5+个请求：额外奖励 +7.0 + (n-5)*2.0（强烈鼓励批量）
        """
        n = num_vehicle_requests
        
        if n == 0:
            return -10.0  # 空车出发，严厉惩罚
        elif n == 1:
            return -2.0   # 单请求，给予负奖励（强烈不鼓励）
        elif n == 2:
            return 3.0    # 2个请求，小正奖励
        elif n == 3:
            return 8.0    # 3个请求，明显提升
        elif n == 4:
            return 15.0   # 4个请求，大幅提升
        elif n == 5:
            return 25.0   # 5个请求，进一步加大
        else:
            # 6+个请求：使用幂函数加速增长
            # 例如：6个→35.0, 7个→46.0, 8个→58.2, 9个→71.5
            # 形成明显的"越多越好"的激励信号
            extra = pow(n - 5, 1.5) * 5.0  # 增强幂函数增长系数
            return 35.0 + extra
    
    @staticmethod
    def calculate_lower_reward(action, request_delivered, vehicle_wait_failed=False, 
                              vehicle_feasible=False, drone_feasible=False):
        """
        计算下层智能体的奖励（优化版：提高送达率）
        
        Args:
            action: 动作 (0=无人机, 1=车辆, 2=拒绝)
            request_delivered: 请求是否成功送达
            vehicle_wait_failed: 分配给车辆但因等待而最终未能按时交付
            vehicle_feasible: 车辆是否可行（用于拒绝惩罚）
            drone_feasible: 无人机是否可行（用于拒绝惩罚）
        
        Returns:
            float: 奖励值
        """
        if action == 2:  # 拒绝
            # 可行性感知拒绝惩罚：可行但拒绝给予严厉惩罚
            if vehicle_feasible or drone_feasible:
                return -2.0  # 可行但拒绝，严厉惩罚
            else:
                return 0.0   # 不可行而拒绝，中性（合理拒绝）
        elif vehicle_wait_failed:  # 分配给车辆但等待超时
            return -2.0  # 从-1.0增加到-2.0，提高惩罚
        elif request_delivered:  # 成功送达
            return 1.5  # ✅ 修复：从0.5改为1.5（真正增强奖励）
        else:  # 其他情况（接受但未送达等）
            return 0.0
    
