"""
Logging utilities for the Idiomapp application.
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Try to load dotenv if available, otherwise continue without it
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Environment variables loaded from .env file")
except ImportError:
    print("python-dotenv not installed, using default environment variables")

# Define log levels mapping
LOG_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# Get log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Get log level from environment variable
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
log_level = LOG_LEVELS.get(LOG_LEVEL, logging.INFO)

# Keep a list of recent log messages for display in the UI
recent_log_messages = []
MAX_RECENT_LOGS = 100

class RecentLogsHandler(logging.Handler):
    """Custom handler to store recent log messages for UI display"""
    def emit(self, record):
        global recent_log_messages
        try:
            msg = self.format(record)
            recent_log_messages.append(msg)
            # Keep only the most recent messages
            if len(recent_log_messages) > MAX_RECENT_LOGS:
                recent_log_messages = recent_log_messages[-MAX_RECENT_LOGS:]
        except Exception:
            self.handleError(record)

def setup_logging(module_name="idiomapp"):
    """
    Set up logging for the application.
    
    Args:
        module_name (str): The name of the module to log for.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create the logger
    logger = logging.getLogger(module_name)
    
    # Set log level from environment variable
    logger.setLevel(log_level)
    
    # Remove existing handlers to prevent duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s [%(name)s:%(lineno)d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler - use module name for the log file
    log_file = LOG_DIR / f"{module_name}.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=3
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add UI log handler
    ui_handler = RecentLogsHandler()
    ui_handler.setFormatter(formatter)
    logger.addHandler(ui_handler)
    
    # Make sure logger propagates to parent
    logger.propagate = True
    
    # Log startup message to confirm logger is working
    logger.info(f"Logger initialized for module: {module_name}")
    
    return logger

def get_recent_logs(max_logs=50, filter_text=None):
    """
    Get the most recent log messages.
    
    Args:
        max_logs (int): Maximum number of logs to return.
        filter_text (str, optional): Text to filter logs by.
        
    Returns:
        list: Recent log messages.
    """
    if filter_text:
        filtered_logs = [log for log in recent_log_messages if filter_text.lower() in log.lower()]
        return filtered_logs[-max_logs:] if filtered_logs else []
    
    return recent_log_messages[-max_logs:] if recent_log_messages else []

def clear_logs():
    """Clear the recent logs buffer."""
    global recent_log_messages
    recent_log_messages = [] 