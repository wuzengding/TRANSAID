# utils/logger.py
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

def setup_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """设置日志记录器
    
    Args:
        name: 记录器名称
        level: 日志级别
        log_file: 日志文件路径
        log_format: 日志格式
    
    Returns:
        logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    
    # 如果没有提供格式，使用默认格式
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加时间戳到日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = log_file.parent / f"{log_file.stem}_{timestamp}{log_file.suffix}"
        
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

class LoggerWriter:
    """将logger作为文件对象使用的包装器"""
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = []

    def write(self, message):
        if message and message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass

def redirect_stdout_to_logger(logger, level=logging.INFO):
    """重定向标准输出到logger"""
    sys.stdout = LoggerWriter(logger, level)

def redirect_stderr_to_logger(logger, level=logging.ERROR):
    """重定向标准错误到logger"""
    sys.stderr = LoggerWriter(logger, level)