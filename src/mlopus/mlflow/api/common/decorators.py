import functools


def online(func):
    """Decorator for API methods that are unavailable in offline mode."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.offline_mode:
            raise RuntimeError(f"Cannot access `{self.__class__.__name__}.{func.__name__}` in offline mode.")
        return func(self, *args, **kwargs)

    return wrapper


def require_update(func):
    """Decorator for API methods that require updating the entity data."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        retval = func(self, *args, **kwargs)
        self.update()
        return retval

    return wrapper
