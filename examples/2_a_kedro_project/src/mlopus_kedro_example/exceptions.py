from dataclasses import dataclass, replace


@dataclass
class ParamOutOfBounds(Exception):
    """Raised when a param is out of bounds."""

    param_name: str
    lower: int
    upper: int
    actual: int = None

    def __post_init__(self):
        msg = f"Value {self.actual} for param '{self.param_name}' is out of bounds (>={self.lower},<{self.upper})"
        Exception.__init__(self, msg)

    def __repr__(self):
        return Exception.__repr__(self)

    def maybe_raise(self, value: int):
        """Raise with value if out of bounds."""
        if not self.lower <= value < self.upper:
            raise replace(self, actual=value)
