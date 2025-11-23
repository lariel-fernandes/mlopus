"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

from kedro.framework.session import KedroSession

from mlopus.kedro import JinjaYamlConfigLoader, MlopusKedroSession, hooks

# Patch the Kedro session
KedroSession.create = MlopusKedroSession.create

# Patch the Kedro config loader
CONFIG_LOADER_CLASS = JinjaYamlConfigLoader
CONFIG_LOADER_ARGS = {
    "namespaces": ["globals"],  # load this namespace first
    "load_mode": "all",  # then load all other namespaces
    "namespace_mappings": {
        "catalog": "io",  # remap so kedro finds the data catalog configs
        "parameters": "nodes",  # remap so the node factories find the parameters
    }
}

# Hooks in reverse order of execution
HOOKS = (
    hooks.MlflowTracker.factory("hooks.mlflow_tracker"),
    hooks.MlflowArtifacts.factory("hooks.mlflow_artifacts"),
)
