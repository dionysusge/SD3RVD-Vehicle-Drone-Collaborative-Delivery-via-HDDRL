"""
距离计算工具
"""
import numpy as np


def euclidean_distance(x1, y1, x2, y2):
    """计算欧氏距离"""
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def road_distance(x1, y1, x2, y2, factor=1.5):
    """
    计算道路网络距离（基于Boscoe et al., 2012的近似）
    道路距离 ≈ 欧氏距离 × factor
    """
    euclidean = euclidean_distance(x1, y1, x2, y2)
    return euclidean * factor


def travel_time_vehicle(distance_km, speed_kmh=30.0):
    """计算车辆行驶时间（分钟）"""
    return (distance_km / speed_kmh) * 60


def travel_time_drone(distance_km, speed_kmh=40.0):
    """计算无人机飞行时间（分钟）"""
    return (distance_km / speed_kmh) * 60
