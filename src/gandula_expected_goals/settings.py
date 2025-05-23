"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/kedro_project_setup/settings.html."""

# Instantiated project hooks.
# For example, after creating a hooks.py and defining a ProjectHooks class there, do
# from gandula_expected_goals.hooks import ProjectHooks
# Hooks are executed in a Last-In-First-Out (LIFO) order.
# HOOKS = (ProjectHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

import os

# Class that manages how configuration is loaded.
from kedro.config import OmegaConfigLoader, MissingConfigException
from kedro.framework.project import settings
from pathlib import Path, PosixPath
from gandula_expected_goals.utils import get_parameters



# Directory that holds configuration.
CODE_SOURCE = PosixPath(os.path.dirname(os.path.abspath(__file__)))
PROJECT_SOURCE =  (CODE_SOURCE / "../../")
CONF_SOURCE = "conf"

# CONFIG_LOADER_CLASS = OmegaConfigLoader

conf_path = str(PROJECT_SOURCE / settings.CONF_SOURCE)
conf_loader = OmegaConfigLoader(conf_source=conf_path)


try:
    parameters = get_parameters(conf_loader)
except MissingConfigException:
    parameters = {}

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    # "config_patterns": {
    #     "spark" : ["spark*/"],
    #     "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
    # }
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
