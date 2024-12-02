import os
from pathlib import Path

import pytest

from mlopus.utils import urls


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://example.com", None),
        ("file:///example.com", None),
        ("~/mlruns", f"file://{Path.home()}/mlruns"),
        ("~/mlruns/test", f"file://{Path.home()}/mlruns/test"),
        ("mlruns", f"file://{Path.cwd()}/mlruns"),
        ("./mlruns", f"file://{Path.cwd()}/mlruns"),
        ("./mlruns/test", f"file://{Path.cwd()}/mlruns/test"),
    ],
)
def test_parse_url(url: str, expected: str):
    assert str(urls.parse_url(url, resolve_if_local=True)) == (expected or url)
