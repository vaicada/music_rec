"""
StepLogger - Logging utility for tracking project implementation steps.

This module provides timestamped logging functionality that writes to a text file
for use in graduation thesis documentation.

Author: Graduation Project
Created: 2026-01-06
"""

import os
import threading
from datetime import datetime
from typing import Optional
from functools import wraps


class StepLogger:
    """
    A thread-safe logger that writes timestamped entries to a log file.
    
    This logger is designed to document all implementation steps for
    academic thesis writing purposes.
    
    Attributes:
        log_file (str): Path to the log file.
        _lock (threading.Lock): Thread lock for safe concurrent writes.
    
    Example:
        >>> logger = StepLogger("project_implementation_log.txt")
        >>> logger.log("Loading BERT model", "MODEL")
        >>> logger.log("Training Epoch 1", "TRAINING", details={"loss": 0.5})
    """
    
    _instance: Optional['StepLogger'] = None
    _lock = threading.Lock()
    
    def __new__(cls, log_file: str = "project_implementation_log.txt") -> 'StepLogger':
        """Singleton pattern to ensure single log file instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_file: str = "project_implementation_log.txt") -> None:
        """
        Initialize the StepLogger.
        
        Args:
            log_file: Path to the log file. Default is 'project_implementation_log.txt'.
        """
        if self._initialized:
            return
            
        self.log_file = log_file
        self._file_lock = threading.Lock()
        self._initialized = True
        
        # Create log file with header if it doesn't exist
        if not os.path.exists(self.log_file):
            self._write_header()
    
    def _write_header(self) -> None:
        """Write the log file header."""
        header = """
================================================================================
                    MUSIC RECOMMENDER SYSTEM - IMPLEMENTATION LOG
================================================================================
Project: Hybrid Multi-modal Music Recommendation System
Purpose: Graduation Thesis Documentation
Created: {timestamp}
================================================================================

LEGEND:
  [MODEL]      - Model loading/initialization
  [DATA]       - Data processing operations
  [TRAINING]   - Training loop activities
  [INDEX]      - FAISS indexing operations
  [INFERENCE]  - Prediction/recommendation operations
  [SYSTEM]     - System configuration/setup
  [ERROR]      - Error messages
  [SUCCESS]    - Successful completion markers

================================================================================
                              IMPLEMENTATION LOG
================================================================================

""".format(timestamp=self._get_timestamp())
        
        with self._file_lock:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                f.write(header)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in standardized format."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    def log(
        self, 
        message: str, 
        category: str = "SYSTEM",
        details: Optional[dict] = None,
        level: str = "INFO"
    ) -> None:
        """
        Log a timestamped message to the log file.
        
        Args:
            message: The main log message.
            category: Category tag (e.g., MODEL, DATA, TRAINING, INDEX, INFERENCE).
            details: Optional dictionary with additional details.
            level: Log level (INFO, DEBUG, WARNING, ERROR, SUCCESS).
        
        Example:
            >>> logger.log("Loading BERT model", "MODEL")
            >>> logger.log("Epoch completed", "TRAINING", {"loss": 0.25, "accuracy": 0.85})
        """
        timestamp = self._get_timestamp()
        
        # Format the log entry
        entry_parts = [
            f"[{timestamp}]",
            f"[{level:^7}]",
            f"[{category:^10}]",
            message
        ]
        entry = " ".join(entry_parts)
        
        # Add details if provided
        if details:
            details_str = " | ".join([f"{k}={v}" for k, v in details.items()])
            entry += f"\n{'':>45}Details: {details_str}"
        
        entry += "\n"
        
        # Thread-safe write
        with self._file_lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(entry)
    
    def log_section(self, section_name: str) -> None:
        """
        Log a section separator for better organization.
        
        Args:
            section_name: Name of the new section.
        """
        separator = f"""
--------------------------------------------------------------------------------
                              {section_name.upper()}
--------------------------------------------------------------------------------
"""
        with self._file_lock:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(separator)
    
    def log_start(self, operation: str, category: str = "SYSTEM") -> None:
        """Log the start of an operation."""
        self.log(f"STARTED: {operation}", category, level="INFO")
    
    def log_end(self, operation: str, category: str = "SYSTEM", duration: Optional[float] = None) -> None:
        """Log the end of an operation with optional duration."""
        details = {"duration_seconds": f"{duration:.2f}"} if duration else None
        self.log(f"COMPLETED: {operation}", category, details=details, level="SUCCESS")
    
    def log_error(self, message: str, category: str = "SYSTEM", exception: Optional[Exception] = None) -> None:
        """Log an error message."""
        details = {"exception": str(exception)} if exception else None
        self.log(f"ERROR: {message}", category, details=details, level="ERROR")
    
    def log_metric(self, metric_name: str, value: float, epoch: Optional[int] = None) -> None:
        """Log a training metric."""
        details = {"value": f"{value:.6f}"}
        if epoch is not None:
            details["epoch"] = epoch
        self.log(f"METRIC: {metric_name}", "TRAINING", details=details)
    
    def log_config(self, config: dict) -> None:
        """Log configuration settings."""
        self.log_section("Configuration")
        for key, value in config.items():
            self.log(f"CONFIG: {key} = {value}", "SYSTEM")


def log_step(category: str = "SYSTEM", operation: str = ""):
    """
    Decorator to automatically log function execution.
    
    Args:
        category: Log category.
        operation: Operation name (defaults to function name).
    
    Example:
        >>> @log_step("MODEL", "Loading BERT")
        ... def load_bert():
        ...     # function body
        ...     pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            logger = StepLogger()
            op_name = operation or func.__name__
            
            logger.log_start(op_name, category)
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_end(op_name, category, duration)
                return result
            except Exception as e:
                logger.log_error(op_name, category, e)
                raise
        
        return wrapper
    return decorator


# Global logger instance
_global_logger: Optional[StepLogger] = None


def get_logger(log_file: str = "project_implementation_log.txt") -> StepLogger:
    """
    Get or create the global logger instance.
    
    Args:
        log_file: Path to the log file.
    
    Returns:
        StepLogger instance.
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = StepLogger(log_file)
    return _global_logger


if __name__ == "__main__":
    # Test the logger
    logger = StepLogger("test_log.txt")
    
    logger.log_section("Testing Logger")
    logger.log("This is a test message", "SYSTEM")
    logger.log("Loading BERT model", "MODEL")
    logger.log("Processing 1000 samples", "DATA", {"samples": 1000, "batch_size": 32})
    logger.log_metric("loss", 0.2534, epoch=1)
    logger.log_start("FAISS Index Building", "INDEX")
    logger.log_end("FAISS Index Building", "INDEX", duration=12.5)
    logger.log_error("Failed to load file", "DATA", Exception("FileNotFoundError"))
    
    print("Logger test completed. Check test_log.txt")
