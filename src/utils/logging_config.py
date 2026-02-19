import logging
import sys
import warnings
from typing import Optional

# Track if logging has been configured
_logging_configured = False

def setup_logging(
    log_file: str = 'debug.log',
    level: int = logging.INFO,
    force_reset: bool = False
) -> None:
    """
    Sets up logging with UTF-8 encoding and redirects warnings to log.
    Prevents duplicate configuration by checking if already configured.
    
    Args:
        log_file: Path to log file
        level: Logging level (default: INFO)
        force_reset: Force reconfiguration even if already set up
    """
    global _logging_configured
    
    # Prevent duplicate configuration
    if _logging_configured and not force_reset:
        return
    
    # Reset any existing handlers if forcing reset
    if force_reset:
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler with UTF-8
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # File handler with UTF-8
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.error(f"Failed to setup file handler: {e}")
    
    # Add console handler
    root_logger.addHandler(console_handler)
    
    # Capture python warnings into the log
    logging.captureWarnings(True)
    
    # Suppress specific spammy warnings
    warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    _logging_configured = True
    logging.info(f"Logging initialized. Output to console and {log_file} (UTF-8).")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    Ensures logging is configured before returning.
    
    Args:
        name: Name of the logger
        
    Returns:
        Logger instance
    """
    if not _logging_configured:
        setup_logging()
    
    return logging.getLogger(name)

def set_log_level(level: int) -> None:
    """
    Change the log level after initial setup.
    
    Args:
        level: New logging level (e.g., logging.DEBUG, logging.INFO)
    """
    logging.getLogger().setLevel(level)
    for handler in logging.getLogger().handlers:
        handler.setLevel(level)
