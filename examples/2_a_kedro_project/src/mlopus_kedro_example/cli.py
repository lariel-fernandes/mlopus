"""Custom CLI."""
import json
import logging
from typing import Mapping

import click
import mlopus.kedro

logger = logging.getLogger(__name__)

cli = click.Group(
    name=__name__.split(".")[0],
    help="""MLOpus Kedro Example.""",
)

# Copy the `run` command from the Kedro CLI, then patch it with extra options and callbacks
run_command = mlopus.kedro.RunCommand(
    decorators=[
        mlopus.kedro.cli_option(
            "--pipeline",
            help="Pipeline name.",
            target_key="mlflow.run.name",  # Forward value to `mlflow.run.name`
        ),
        mlopus.kedro.cli_option(
            "--json-params",
            type=json.loads,
            params_root=True,
            help="JSON object to be used as root of runtime params (overrides).",
        ),
        mlopus.kedro.cli_option(
            "--chain-id",
            target_key="globals.chain_id",  # Forward value to `globals.chain_id`
            help="Use a common ID to link upstream/downstream pipelines.",
        ),
    ],
).register(cli)


@run_command.dynamic_override(
    pipelines=["eval"],
    target_key="globals.model.version",
)
def choose_model_version(kedro_config: Mapping) -> str | None:
    """Before the pipeline `eval` is executed, find the last model version
    that was produced by an experiment run with the same `chain_id` tag."""
    results = mlopus.mlflow \
        .get_api(**kedro_config["mlflow"]["api"]) \
        .get_model(name := kedro_config["globals"]["model"]["name"]) \
        .find_versions(
            sorting=[("version", -1)],
            query={"run.tags.chain_id": kedro_config["globals"]["chain_id"]},
        )

    logger.info("Using model '%s v%s'", name, version := next(results).version)
    return version
