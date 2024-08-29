"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://kedro.readthedocs.io/en/stable/kedro_project_setup/settings.html."""

from kedro.framework.session import KedroSession

from mlopus.kedro import MlopusConfigLoader, MlopusKedroSession, hooks

# Patch the Kedro session
KedroSession.create = MlopusKedroSession.create

# Patch the Kedro config loader
CONFIG_LOADER_CLASS = MlopusConfigLoader

# Hooks in reverse order of execution
HOOKS = (
    hooks.MlflowTracker.factory(),
    hooks.MlflowArtifacts.factory(),
)
