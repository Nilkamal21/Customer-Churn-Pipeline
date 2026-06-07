import os
import sys
import logging
import yaml
import joblib

def load_yaml(file_path: str) -> dict:
    """Loads a YAML configuration file."""
    try:
        with open(file_path, "r") as yaml_file:
            config = yaml.safe_load(yaml_file)
            return config
    except Exception as e:
        raise Exception(f"Error occurred while reading yaml file: {e}")

def setup_logging(log_dir: str = "logs", log_file_name: str = "running_logs.log") -> logging.Logger:
    """Sets up standard logging to console and file."""
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_file_name)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger("MLPipeline")

def save_object(file_path: str, obj: any):
    """Saves a Python object (e.g. model or preprocessor) as a serialized file."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise Exception(f"Failed to save object to {file_path}: {e}")

def load_object(file_path: str) -> any:
    """Loads a serialized Python object."""
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise Exception(f"Failed to load object from {file_path}: {e}")
