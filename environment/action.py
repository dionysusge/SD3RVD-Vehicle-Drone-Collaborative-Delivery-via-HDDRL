"""
动作空间定义
"""

# 上层智能体动作
UPPER_ACTION_WAIT = 0  # 等待
UPPER_ACTION_DEPART = 1  # 出发
UPPER_ACTION_NIL = 2  # 无可用车辆

# 下层智能体动作
LOWER_ACTION_DRONE = 0  # 分配给无人机
LOWER_ACTION_VEHICLE = 1  # 分配给车辆
LOWER_ACTION_REJECT = 2  # 拒绝
LOWER_ACTION_NIL = 3  # 无请求


class Action:
    """动作类"""
    
    def __init__(self, upper_action=None, lower_action=None, route_update=None):
        """
        初始化动作
        
        Args:
            upper_action: 上层智能体动作 (0=等待, 1=出发, 2=Nil)
            lower_action: 下层智能体动作 (0=无人机, 1=车辆, 2=拒绝, 3=Nil)
            route_update: 路径规划更新（由启发式算法生成）
        """
        self.upper_action = upper_action
        self.lower_action = lower_action
        self.route_update = route_update  # 更新后的路线计划
    
    def __repr__(self):
        return f"Action(upper={self.upper_action}, lower={self.lower_action})"
