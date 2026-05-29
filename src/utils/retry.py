from typing import Type, Tuple
import tenacity
from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log, retry_if_exception_type
from src.utils.logging import logger

def robust_retry(
    attempts: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    A robust retry decorator using tenacity.
    Retries up to `attempts` times, waiting `delay` seconds between each attempt.
    """
    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_fixed(delay),
        retry=retry_if_exception_type(exceptions),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True
    )
