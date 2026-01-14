#!/usr/bin/env python3
import logging


def setup_client_logger(
    enable_log: bool = True, log_level: int = logging.INFO
) -> logging.Logger:
    """
    配置客户端专属日志器，支持开关控制
    :param enable_log: 是否启用日志输出
    :param log_level: 日志级别（默认logging.INFO）
    :return: 配置完成的Logger实例
    """
    # 创建日志器
    logger = logging.getLogger("TrackingDataClient")
    logger.setLevel(log_level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 日志格式配置
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)

    # 添加处理器（根据开关控制是否启用）
    if enable_log:
        logger.addHandler(console_handler)
    else:
        # 禁用所有日志输出
        logger.disabled = True

    return logger
