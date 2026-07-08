from functools import wraps
import time

def timed():
    """Timing decorator that returns elapsed time"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            timing = {
                f"{func.__name__}": end - start
            }
            return result, timing
        return wrapper
    return decorator