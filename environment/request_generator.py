"""
动态请求生成器
支持均匀/非均匀时间分布和正态/幂律空间分布
"""
import numpy as np
from config.environment_config import (
    T_MAX, DELTA, WAREHOUSE_X, WAREHOUSE_Y, 
    NORMAL_STD, LAMBDA_DEFAULT
)


class Request:
    """客户请求类"""
    def __init__(self, request_id, arrival_time, x, y, deadline):
        self.id = request_id
        self.arrival_time = arrival_time  # 到达时间
        self.x = x  # 客户位置x坐标
        self.y = y  # 客户位置y坐标
        self.deadline = deadline  # 截止时间
        self.assigned = False  # 是否已分配
        self.assigned_to = None  # 分配给车辆或无人机
        self.delivered = False  # 是否已送达
        self.delivery_time = None  # 实际送达时间
    
    def __repr__(self):
        return f"Request(id={self.id}, t={self.arrival_time:.1f}, pos=({self.x:.2f}, {self.y:.2f}), deadline={self.deadline:.1f})"


class RequestGenerator:
    """请求生成器"""
    
    def __init__(self, time_dist='homo', space_dist='normal', lambda_rate=LAMBDA_DEFAULT):
        """
        初始化请求生成器
        
        Args:
            time_dist: 时间分布类型 ('homo' 均匀, 'heter' 非均匀)
            space_dist: 空间分布类型 ('normal' 正态, 'power' 幂律)
            lambda_rate: 泊松过程参数（每分钟平均到达数）
        """
        self.time_dist = time_dist
        self.space_dist = space_dist
        self.lambda_rate = lambda_rate
        self.request_id_counter = 0
    
    def generate_requests(self):
        """
        生成一天的请求序列
        
        Returns:
            list: 按到达时间排序的请求列表
        """
        requests = []
        current_time = 0.0
        
        while current_time < T_MAX:
            # 生成下一个请求的到达时间间隔
            if self.time_dist == 'homo':
                # 均匀分布：泊松过程
                interval = np.random.exponential(1.0 / self.lambda_rate)
            else:  # heter
                # 非均匀分布：早晚高峰
                # 假设高峰时段为 [0, 120] 和 [300, 420] 分钟
                if (0 <= current_time < 120) or (300 <= current_time < 420):
                    peak_lambda = 2 * self.lambda_rate
                else:
                    peak_lambda = self.lambda_rate
                interval = np.random.exponential(1.0 / peak_lambda)
            
            current_time += interval
            
            if current_time >= T_MAX:
                break
            
            # 生成客户位置
            x, y = self._generate_location()
            
            # 生成请求
            deadline = current_time + DELTA
            request = Request(
                request_id=self.request_id_counter,
                arrival_time=current_time,
                x=x,
                y=y,
                deadline=deadline
            )
            requests.append(request)
            self.request_id_counter += 1
        
        # 按到达时间排序
        requests.sort(key=lambda r: r.arrival_time)
        return requests
    
    def _generate_location(self):
        """生成客户位置"""
        if self.space_dist == 'normal':
            # 正态分布：围绕仓库呈正态分布
            x = np.random.normal(WAREHOUSE_X, NORMAL_STD)
            y = np.random.normal(WAREHOUSE_Y, NORMAL_STD)
        else:  # power
            # 幂律分布：客户聚集在几个高密度区域
            # 简化实现：生成几个中心点，然后围绕中心点生成位置
            centers = [
                (5.0, 5.0), (-5.0, 5.0), (5.0, -5.0), (-5.0, -5.0),
                (0.0, 8.0), (8.0, 0.0), (-8.0, 0.0), (0.0, -8.0)
            ]
            center = centers[np.random.randint(len(centers))]
            # 使用指数分布模拟幂律
            radius = np.random.exponential(2.0)
            angle = np.random.uniform(0, 2 * np.pi)
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
        
        return x, y
