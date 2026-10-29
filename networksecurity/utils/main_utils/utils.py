import os
import sys
import dill
import pickle
import yaml
import numpy as np

from networksecurity.exception.custom_exception import NetworkSecurityException
from networksecurity.logging.logger import logger

def read_yaml_file(file_path: str) -> dict:
    """
    Reads a YAML file and return its parsed contents
    """
    try:
        logger.info("Reading YAML file: %s", file_path)
        with open(file_path, "rb") as yaml_file:
            content = yaml.safe_load(yaml_file)
        logger.info("Successfully read YAML file: %s", file_path)
        return content
    
    except Exception as e:
        logger.error("Failed to read YAML file %s: %s", file_path, e)
        raise NetworkSecurityException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Writes `content` to a YAML file, creating parent directories as needed.
    If `replace` is True, an existing file at `file_path` is removed first.
    """
    try:
        if replace and os.path.exists(file_path):
            logger.info("Replacing existing YAML file: %s", file_path)
            os.remove(file_path)

        dir_name = os.path.dirname(file_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        logger.info("Writing YAML file: %s", file_path)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
        logger.info("Successfully wrote YAML file: %s", file_path)

    except Exception as e:
        logger.error("Failed to write YAML file %s: %s", file_path, e)
        raise NetworkSecurityException(e, sys)