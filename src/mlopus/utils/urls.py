from pathlib import Path

import urllib3.util

Url = urllib3.util.Url

UrlLike = Url | str | Path


def parse_url(url: UrlLike, default_scheme: str = "file", resolve_if_local: bool = False) -> Url:
    """Parse URL with default scheme."""
    if not isinstance(url, Url):
        url = urllib3.util.parse_url(str(url))
    if not url.scheme:
        url = urllib3.util.parse_url(f"{default_scheme}://{url}")
    if resolve_if_local and is_local(url):
        url = parse_url(Path((url.host or "") + (url.path or "")).expanduser().resolve())
    return url


def is_local(url: UrlLike) -> bool:
    """Tell if URL points to local file system."""
    return parse_url(url).scheme == "file"


def urljoin(*parts: UrlLike) -> Url:
    """Join URL parts while trimming slashes."""
    return parse_url("/".join(trimmed for x in parts if (trimmed := str(x).strip("/ "))))
