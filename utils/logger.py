"""
日志记录工具
"""
import logging
import os
from datetime import datetime


def setup_logger(name, log_dir='logs', level=logging.INFO, timestamp=None):
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志目录
        level: 日志级别
        timestamp: 时间戳字符串（可选，如果不提供则自动生成）
    
    Returns:
        tuple: (logging.Logger, timestamp) - 日志记录器和时间戳
    """
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建或使用提供的时间戳
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建日志文件路径
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    # 配置日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, timestamp
