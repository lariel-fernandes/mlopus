import pytest

from mlopus.mlflow.api.common.transfer import FileTransfer
from mlopus.utils import urls


@pytest.mark.parametrize(
    "transfer,url,expected",
    [
        (FileTransfer(use_scheme="foo"), "file:///path/to/file", "file:///path/to/file"),
        (FileTransfer(use_scheme="foo"), "s3://path/to/file", "foo://path/to/file"),
        (FileTransfer(map_scheme={r"^s3://": "foo"}), "s3://path/to/file", "foo://path/to/file"),
        (FileTransfer(map_scheme={r"^gs://": "bar", r"^s3://": "foo"}), "s3://path/to/file", "foo://path/to/file"),
        (FileTransfer(map_scheme={r"^s3://": "foo", r"^gs://": "bar"}), "s3://path/to/file", "foo://path/to/file"),
        (FileTransfer(map_scheme={r"^s3:/": "foo", r"^s3://": "bar"}), "s3://path/to/file", "foo://path/to/file"),
        (FileTransfer(map_scheme={}), "s3://path/to/file", "s3://path/to/file"),
        (FileTransfer(map_scheme={r"^gs://": "bar", r"^s3://": "foo"}), "other://path/to/file", "other://path/to/file"),
    ],
)
def test_translate_scheme(transfer: FileTransfer, url: str, expected: str):
    """Test URL scheme translation."""
    actual = transfer._translate_scheme(url)
    assert str(actual) == expected
    assert urls.parse_url(actual) == urls.parse_url(expected)
