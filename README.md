# MLOpus
[![Test Coverage](https://lariel-fernandes.github.io/mlopus/coverage/coverage.svg)](https://lariel-fernandes.github.io/mlopus/coverage)

A collection of MLOps tools for AI/ML/DS research and development.

### Main features:
- **Agnostic experiment tracking and model registry:**
  - Compatible with any "MLflow-like" provider through plugins.
  - Search entities in MongoDB Query Language with predicate push-down to the MLflow provider.
  - Local cache for artifacts and entity metadata.
  - Offline mode to work with local cache only.
  - Support for nested tags/params/metrics and JSON-encoded tags/params for non-scalar types.
  - Not dependent on env vars, global vars or a single global active run.


- **Artifact Schemas**:
  - Packaging framework for models and datasets.
  - Can be used with or without MLflow and/or Kedro.
  - Schemas can be registered by alias at the experiment, run, model or model version.
  - Artifacts catalog for type-safe, configuration-based artifact loading/downloading in serving applications.


- **Extended Kedro support:**
  - Dynamic pipeline and hook evaluation with direct access to the Kedro config loader.
  - Artifact Schemas can be used in the Kedro datasets catalog.
  - Extend the Kedro CLI with project-specific options, callbacks and param modifiers.
  - Artifacts hook to set up pipeline inputs and/or collect outputs (optionally schema-aware).
  - Highly customizable MLflow tracker hook for storing any pipeline information in experiment runs.

Check the [tutorials](https://github.com/lariel-fernandes/mlopus/tree/main/examples)
for a friendly walkthrough of (almost) everything you can do with MLOpus.

Have a look at the [architecture guide](https://github.com/lariel-fernandes/mlopus/blob/main/docs/architecture.md)
for an overview of how these and other features work.

A minimal API reference is also available [here](https://lariel-fernandes.github.io/mlopus/docs/api/latest).

### Installation

**Recommended software:**
- [Rclone CLI](https://rclone.org/install/#script-installation) (required for artifact transfer from/to cloud storage)

**Optional extras:**
- `mlflow`: Enables support for the default MLflow plugin, which handles communication with open-source MLflow servers.
- `search`: Enables searching entities with MongoDB query syntax
- `kedro`: Enables Kedro tools (e.g.: hooks, datasets, CLI extensions, etc)

**Using pip:**
```bash
pip install mlopus[mlflow,kedro,search]
```

**Using Poetry:**
```bash
poetry add mlopus --extras "mlflow,kedro,search"
```

**Using UV:**
```bash
uv add mlopus --extra mlflow --extra kedro --extra search
```
