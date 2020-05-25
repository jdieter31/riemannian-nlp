"""
Tools for saving a global config for the project that automatically reads
modularized configs
"""
import importlib
import inspect
import json
import os
import pkgutil
import sys
from typing import Dict

from .config import ConfigDict, ConfigDictParser


class GlobalConfigDictMeta(type):
    """
    Meta-class for GlobalConfigDict - programmatically injects each ConfigDic
    sublass in config_specs into GlobalConfigDict as well as type hints
    """

    @staticmethod
    def _get_config_specs():
        config_specs: Dict = {}

        for (module_loader, name, ispkg) in pkgutil.iter_modules(
                [os.path.dirname(__file__) + "/config_specs"]):

            # Run register_config_spec for each detected config_spec
            importlib.import_module(".config_specs." + name, __package__)
            pkg_name = __package__ + '.config_specs.' + name
            obj = sys.modules[pkg_name]
            for dir_name in dir(obj):
                if dir_name.startswith('_') or dir_name == "ConfigDict":
                    # Continue if private member or the ConfigDict class itself
                    continue

                dir_obj = getattr(obj, dir_name)
                if not inspect.isclass(dir_obj):
                    # Continue if not a class
                    continue

                if issubclass(dir_obj, ConfigDict):
                    if dir_obj.__module__ != pkg_name:
                        # Continue if this comes from another file
                        continue

                    if not hasattr(obj, "CONFIG_NAME"):
                        raise Exception(f"{dir_name} object not contained in a file \
                                        with variable CONFIG_NAME")
                    if obj.CONFIG_NAME not in config_specs.keys():
                        # If not already added add this config_spec
                        config_specs[obj.CONFIG_NAME] = dir_obj
        return config_specs

    def __new__(mcs, name, bases, namespace, **kwds):
        for name, config_spec in GlobalConfigDictMeta._get_config_specs().items():
            if '__annotations__' not in namespace:
                namespace['__annotations__'] = {}

            # Inject type annotation
            namespace['__annotations__'][name] = config_spec
            # Inject actual variable
            namespace[name] = config_spec()
        return super().__new__(mcs, name, bases, namespace, **kwds)


class GlobalConfigDict(ConfigDict, metaclass=GlobalConfigDictMeta):
    """
    Stores all of the specific configs registered

    For each subclass of ConfigDict in the config_specs folder,
    GlobalConfigDict will have a variable given by the value of 
    CONFIG_NAME in the python file where the class is defined. In
    this way modular configs can be dynamically loaded.
    """


global_config: GlobalConfigDict


def get_config():
    """
    Get the loaded config - should be called after initialize_config
    """
    return global_config


def initialize_config(path: str = None, load_config: bool = False, config_updates: str = "",
                      save_config: bool = False, save_directory: str = None):
    """
    Initializes global config - run before referencing global_config
    Args:
        path (str): Path to config file - should be json
        load_config (Bool): Should config be loaded or generated from default values
        config_updates (str): Updates configuration dictionary with these
            updates according to grammar of riemann.config.ConfigDictParser -
            useful for command line specification
        save_config (str): Whether or not the config should be saved somewhere
        save_directory (str): Different directory to save config file (same as path if None)
    """

    global global_config

    global_config = GlobalConfigDict()
    if load_config:
        with open(path) as json_file:
            global_config.update(**json.load(json_file))

    if config_updates is not None:
        global_config.update(**ConfigDictParser.parse(config_updates))

    if save_config:
        if save_directory is None:
            save_directory = path
        with open(save_directory, "w+") as outfile:
            json.dump(global_config.as_json(), outfile)
