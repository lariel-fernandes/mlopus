import contextlib
import os
import threading
from typing import Dict, ContextManager

_env_lock = threading.Lock()


@contextlib.contextmanager
def using_env_vars(_vars: Dict[str, str]) -> ContextManager[None]:
    """Thread-safely set env vars inside context and reset them on exit."""
    with _env_lock:
        original_env = dict(os.environ)
        try:
            os.environ.update(_vars)
            yield None
        finally:
            os.environ = original_env
