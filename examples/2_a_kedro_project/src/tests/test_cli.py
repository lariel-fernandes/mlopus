from click.testing import CliRunner
from mlopus.utils import json_utils  # Handles JSON-encoding of `Path`

from mlopus_kedro_example.cli import cli


def test_cli(tmp_overrides):
    for pipeline in ["build", "eval"]:
        assert CliRunner().invoke(
            cli,
            args=[
                "run",
                "--env", "empty",
                "--pipeline", pipeline,
                "--chain-id", "foobar",
                "--json-params", json_utils.dumps(tmp_overrides),
            ]
        ).exit_code == 0
