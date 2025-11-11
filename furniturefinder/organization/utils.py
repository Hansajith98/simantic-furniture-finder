import time
import logging

logger = logging.getLogger('chat_with_website')


def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(
            f"function_[{func.__name__}]_executed_in_[{execution_time:.4f}]_seconds")
        return result
    return wrapper
