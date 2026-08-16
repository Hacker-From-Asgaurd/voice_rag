import time
from functools import wraps


def _is_rate_limit_error(error):
    """
    Detect API rate-limit / quota errors.

    Gemini commonly raises errors containing:
    - 429
    - RESOURCE_EXHAUSTED
    """

    error_text = str(error).upper()

    return (
        "429" in error_text
        or "RESOURCE_EXHAUSTED" in error_text
        or "RATE_LIMIT" in error_text
    )


def retry(max_attempts=3, delay=1.0):

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):

            last_error = None

            for attempt in range(1, max_attempts + 1):

                try:
                    return func(*args, **kwargs)

                except Exception as e:

                    last_error = e

                    # -------------------------------------------------
                    # HANDLE RATE-LIMIT / QUOTA ERRORS WITH BACKOFF
                    # -------------------------------------------------
                    if _is_rate_limit_error(e):
                        backoff = max(delay * (2 ** (attempt - 1)), 5.0)
                        print(
                            f"[Retry] Rate limit/quota error detected on attempt "
                            f"{attempt}/{max_attempts}. Backing off for {backoff:.1f}s..."
                        )
                        if attempt < max_attempts:
                            time.sleep(backoff)
                            continue
                        raise

                    # -------------------------------------------------
                    # NORMAL TRANSIENT ERROR
                    # -------------------------------------------------
                    print(
                        f"[Retry] Attempt "
                        f"{attempt}/{max_attempts} failed: {e}"
                    )

                    if attempt < max_attempts:
                        time.sleep(delay)

            raise last_error

        return wrapper

    return decorator