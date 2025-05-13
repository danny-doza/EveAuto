import logging
from rich.logging import RichHandler

def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Returns a logger configured with rich formatting, color highlighting, and
    enhanced tracebacks. Subsequent calls with the same name will reuse handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        # Create a RichHandler for pretty-printing logs with colors
        handler = RichHandler(rich_tracebacks=True)
        # Format: timestamp, level, logger name, message
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
