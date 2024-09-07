import base64
import gzip


def to_gz_b64(x: str, encoding: str = "UTF-8") -> str:
    """Encode in gzip compressed base64."""
    return base64.b64encode(gzip.compress(x.encode(encoding))).decode("ascii")


def from_gz_b64(x: str, encoding: str = "UTF-8") -> str:
    """Decode from gzip compressed base64."""
    return gzip.decompress(base64.b64decode(x)).decode(encoding)
