"""
环境参数配置（基于论文 Table 2 和 Table 4）
"""

# 时间参数
T_MAX = 420  # 运营时间（分钟），7小时
DELTA = 240  # 服务承诺时间（分钟），4小时内必须送达
T_WORK_V = 600  # 车辆总工作时长（分钟），10小时
T_WORK_D = 600  # 无人机总工作时长（分钟），10小时

# 车辆参数
V_E = 30.0  # 车辆速度（km/h）
T_LOAD_V = 5  # 车辆装载时间（分钟）- 按论文Table 2修正
T_DROP_V = 5  # 车辆卸货时间（分钟）- 按论文Table 2修正
ROAD_DISTANCE_FACTOR = 1.5  # 道路网络距离系数（欧氏距离 × 1.5）

# 无人机参数
V_D = 40.0  # 无人机速度（km/h）
T_LOAD_D = 5  # 无人机装载时间（分钟）- 按论文Table 2修正
T_DROP_D = 5  # 无人机卸货时间（分钟）- 按论文Table 2修正
T_CHARGE = 20  # 无人机充电时间（分钟）
DRONE_CAPACITY = 1  # 无人机容量（一次只能携带一个包裹）

# 车辆等待间隔
VEHICLE_WAIT_INTERVAL = 1  # 车辆等待时间间隔（分钟）

# 仓库位置（假设在原点）
WAREHOUSE_X = 0.0
WAREHOUSE_Y = 0.0

# 空间分布参数
NORMAL_STD = 3.0  # 正态分布标准差（公里）

# 请求生成参数
LAMBDA_DEFAULT = 0.5  # 默认泊松过程参数（每分钟平均到达数）
