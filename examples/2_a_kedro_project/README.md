# Example 2: A Kedro Project

This simple project showcases the Kedro tools offered by MLOpus.

For an overview of these tools and how they work, have a look at
[this section of the architecture guide](../../docs/architecture.md#mlopus-kedro-flavor).

If you're yet not familiar with the basics of MLOpus,
have a look first at the [Example 1: Introduction to MLOpus](../1_introduction).

**Table of contents**:
- [Project overview](#project-overview)
- [Setting up the environment](#setting-up-the-environment)
- [Running pipelines](#running-pipelines)
- [Custom CLI and Chain ID](#custom-cli-and-chain-id)

## Project overview

### Pipelines
- `build`: Generates random vectors and builds a vector index, which is then published as an ANN model.
- `eval`: Evaluates the vector index model in terms of vector distances at K and publishes the metrics.

### Modules
- [pipeline_registry](src/mlopus_kedro_example/pipeline_registry.py):
  Each pipeline is defined as a factory that configures the nodes and the pipeline itself after receiving the Kedro config.

- [nodes](src/mlopus_kedro_example/nodes.py):
  Each node is defined as a callable Pydantic class, so parameters are validated against the schema eagerly, at pipeline load-time.

- [model](src/mlopus_kedro_example/model.py):
  Declares the `AnnModel` class.

- [artschema](src/mlopus_kedro_example/artschema.py):
  Declares `AnnModelSchema`, an artifact schema that knows how to save/load `AnnModel`.

- [cli](src/mlopus_kedro_example/cli.py):
  Adds custom commands and callbacks to the Kedro CLI (see `kedro --help`).

### Configuration

- [catalog](conf/base/catalog.yml):
  Uses `AnnModelSchema` to save/load instances of `AnnModel`.

- [parameters](conf/base/parameters)/[nodes](conf/base/parameters/nodes.yml):
  Parameters for configuring each node's Pydantic class.

- [parameters](conf/base/parameters)/[hooks](conf/base/parameters/hooks)/[mlflow_tracker](conf/base/parameters/hooks/mlflow_tracker.yml):
  Configures how to record session information on MLflow.

- [parameters](conf/base/parameters)/[hooks](conf/base/parameters/hooks)/[mlflow_artifacts](conf/base/parameters/hooks/mlflow_artifacts.yml)
  Configures which artifacts are to be fetched/published before/after each pipeline.

- [globals](conf/base/globals)/[mlflow](conf/base/globals/mlflow.yml):
  Configures the MLflow API and experiment run instance that will be shared across all nodes/hooks/datasets that need access to MLflow.

## Setting up the environment

### Recommended Software
1. [UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
2. [Rclone CLI](https://rclone.org/install/#script-installation) (required for artifact transfer from/to cloud storage)

### Python environment setup

This will set up the virtualenv using UV and install the example code and dependencies:
```bash
make install
```

### MLflow Sandbox

This example is better experienced with a connection to a sandbox MLflow environment,
which can be obtained by running the following command:
```bash
make mlflow-sandbox-restart
```
- Further instructions will be printed to the terminal
- All sandbox data is lost on stop or restart, caches should be manually invalidated
  (by default at `~/.cache/mlopus/mlflow-providers/mlflow`)

## Running pipelines
1. Activate the virtual environment with `source .venv/bin/activate`


2. Run the `build` pipeline:
   ```bash
   kedro run --pipeline build
   ```
   - The logs will show the URL to the MLflow run.
   - Have a look at the artifacts, params and model version that have been produced.


3. Run the `eval` pipeline using the model version `1` as input:
   ```bash
   kedro run --pipeline eval --params "globals.model.version=1"
   ```
   - Have a look at the metrics in the MLflow run.


4. The model produced in step `#2` can be loaded with the following snippet:
   ```python
   import mlopus

   # Get the model version metadata
   version = mlopus.mlflow \
       .get_api() \
       .get_model("mlopus_kedro_example") \
       .get_version("1")

   # Load the model using its `default` schema (dynamic)
   model = mlopus.artschema.load_artifact(version, schema="default")

   # Load the model using an explicit schema (type-safe)
   model = mlopus.artschema.load_artifact(version, schema=mlopus_kedro_example.artschema.Schema)
   ```

5. Because of the lineage tags that are set automatically by the hook `MlflowArtifacts`,
   it is also possible to find the experiment run from step `#3` by searching for runs
   that used the model `mlopus_kedro_example v1` as input:
   ```python
   import mlopus

   # Define a lineage query
   query = mlopus.lineage.Query() \
       .with_input_model("mlopus_kedro_example", version="1") \
       .render()

   # Find runs in experiment
   results = mlopus.mlflow \
       .get_api() \
       .get_or_create_experiment("mlopus_kedro_example") \
       .find_runs(query)

   # Check results
   print(next(results).url)
   ```

## Custom CLI and Chain ID

Observe that when running the pipeline `eval` in the step `#3` of the [previous section](#running-pipelines),
the value for the param `globals.model.version` should match the model version that was produced
by the pipeline `build` in step `#2`.
Because that version is only known after the pipeline runs, some manual intervention
or automation needs to take place in order to determine and inject the argument value.

In this scenario,
it's useful to have a **chain ID**, which is responsible for linking upstream and downstream pipelines.
That is especially convenient when triggering pipelines from a scheduler like Airflow, where each DAG execution
already has a unique ID that can be forwarded to every task, thus isolating task inputs/outputs between DAG executions.

The [custom CLI](src/mlopus_kedro_example/cli.py)
in this project patches the `kedro run` command with the option `--chain-id`,
whose value will be set on the run tags.
Then, when running the `eval` pipeline, a callback will choose the model version to
be evaluated by searching for the latest version whose parent run had that tag.

Let's see this feature in action:

1. Generate a random chain ID:
   ```bash
   export CHAIN_ID=$(head /dev/urandom | tr -dc a-z0-9 | head -c 12)
   ```


2. Run the `build` pipeline using this project's custom CLI and passing the chain ID option:
   ```bash
   kedro run --pipeline build --chain-id $CHAIN_ID
   ```


3. Run the `eval` pipeline with the **same** chain ID:
   ```bash
   kedro run --pipeline eval --chain-id $CHAIN_ID
   ```
   - Observe that we don't specify the model version to be evaluated


4. Check the lineage tags to confirm that the output of `build` was used as the input of `eval`:
   ```python
   import os
   import mlopus

   exp = mlopus.mlflow.utils.get_api().get_or_create_exp("mlopus_kedro_example")

   build = next(exp.find_runs({"name": "build", "tags.chain_id": os.environ["CHAIN_ID"]}))

   eval_ = next(exp.find_runs({"name": "eval", "tags.chain_id": os.environ["CHAIN_ID"]}))

   print(mlopus.lineage.of(build))

   print(mlopus.lineage.of(eval_))
   ```
