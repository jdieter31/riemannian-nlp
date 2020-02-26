"""
Tools for saving a global config for the project
"""
from .config import ConfigDict
from typing import List, Dict
import json
import collections.abc
import pkgutil
import os


class GlobalConfigDict(ConfigDict):
    """
    Stores all of the specific configs registered
    """

    local_configs: Dict[str, ConfigDict] = {}


global_config: GlobalConfigDict = None
config_specs: Dict = {}


def register_config_spec(name, config_spec_class):
    """
    Registers a submodule configuration
    Args:
        name (str): Name of submodule
        config_spec_class (class extending ConfigDict)
    """
    global config_specs
    config_specs[name] = config_spec_class


for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.dirname(__file__)]):
    # Run register_config_spec for each detected config_spec
    pass


def initialize_config(path, load_config=True, config_updates:Dict=None, save_config=True, save_directory=None):
    """
    Initializes global config - run before referencing global_config
    Args:
        path (str): Path to config file - should be json
        load_config (Bool): Should config be loaded or generated from default values
        config_updates (Dict): Updates configuration dictionary with these updates
            useful for command line specification
        save_config (str): Whether or not the config should be saved somewhere
        save_directory (str): Different directory to save config file (same as path if None)

    """
    for name, config_spec in config_specs:
        GlobalConfigDict.local_configs[name] = config_spec()

    global global_config

    if load_config:
        with open(path) as json_file:
            global_config = GlobalConfigDict(json.load(json_file))
    else:
        global_config = GlobalConfigDict()

    if config_updates is not None:
        global_config.update(**kwargs)

    if save_config:
        if save_directory is None:
            save_directory = path
        with open(save_directory, "w+") as outfile:
            json.dumps(global_config.as_json(), outfile)

import ipdb; ipdb.set_trace()
