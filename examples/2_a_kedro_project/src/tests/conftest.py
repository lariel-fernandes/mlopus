from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from kedro.framework.startup import bootstrap_project

pytest_plugins = ["mlopus_kedro_example.testing"]


@pytest.fixture
def tmp_overrides() -> dict:
    bootstrap_project(".")

    with TemporaryDirectory() as tmp:
        yield {
            "globals": {
                "data": (tmp := Path(tmp)) / "data",
                "model": {
                    "version": "1",
                },
                "mlflow": {
                    "api": {
                        "conf": {
                            "cache_dir": tmp / "cache",
                            "tracking_uri": tmp / "server",
                        },
                    },
                },
            },
        }
