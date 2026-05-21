import logging

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the inference pipeline.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("FishSpeciesInference")
    logger.setLevel(getattr(logging, log_level))
    
    # Console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    logger.addHandler(console_handler)
    
    return logger