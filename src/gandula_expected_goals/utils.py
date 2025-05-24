from kedro.config import OmegaConfigLoader, MissingConfigException

def get_parameters(conf_loader: OmegaConfigLoader) -> dict:
    """
    Get parameters from the configuration loader.

    Args:
        conf_loader (OmegaConfigLoader): The configuration loader instance.

    Returns:
        dict: The parameters loaded from the configuration.
    """
    try:
        parameters = conf_loader["parameters"]
        return parameters
    except MissingConfigException:
        return {}