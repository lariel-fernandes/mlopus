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


Check the [tutorials](https://github.com/lariel-fernandes/mlopus/tree/main/examples)
for a friendly walkthrough of (almost) everything you can do with MLOpus.

A minimal API reference is also available [here](https://lariel-fernandes.github.io/mlopus/docs/api/stable/latest).

### Installation

**Recommended software:**
- [Rclone CLI](https://rclone.org/install/#script-installation) (required for artifact transfer from/to cloud storage)

**Optional extras:**
- `mlflow`: Enables support for the default MLflow plugin, which handles communication with open-source MLflow servers.
- `search`: Enables searching entities with MongoDB query syntax

**Using pip:**
```bash
pip install mlopus[mlflow,search]
```

**Using Poetry:**
```bash
poetry add mlopus --extras "mlflow,search"
```

**Using UV:**
```bash
uv add mlopus --extra mlflow --extra search
```
