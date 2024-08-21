# Example 1: Introduction to MLOpus

The notebooks in this example serve as a walkthrough for the basic features of MLOpus.

For an overview of how those features work, have a look at the [architecture guide](../../docs/architecture.md).

### Recommended Software
1. [UV](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)
2. [Rclone CLI](https://rclone.org/install/#script-installation) (required for artifact transfer from/to cloud storage)

### Python environment setup

Run the following command from the **root** of this repository.
This will set up the virtualenv using UV and install the project code and dependencies.
```bash
make install-project
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

### Start Jupyter Lab

From the **root** of this repository, activate the virtualenv with `source .venv/bin/activate`
and start the Jupyter Lab server with `jupyter lab`.
