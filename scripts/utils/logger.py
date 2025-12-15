"""
Logging configuration for the project
"""
import logging
import sys
from pathlib import Path
from config import LOGS_DIR, LOG_LEVEL

def setup_logger(name, log_file=None):
    """
    Setup logger for a module
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional log file path
    
    Returns:
        logger: Configured logger instance
    """
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger