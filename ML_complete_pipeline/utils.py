import yaml
import os
import logging


def get_logger(name):
# Logging 
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)

    logger.setLevel('DEBUG')

    filepath = os.path.join(log_dir,f'{name}.log')
    file_handler = logging.FileHandler(filepath)
    file_handler.setLevel('DEBUG')

    console_handler = logging.StreamHandler()
    console_handler.setLevel('DEBUG')

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def load_params(params_path: str)-> dict:
    try:
        logger = get_logger('params_yaml')
        with open(params_path, 'r') as file:
            data = yaml.safe_load(file)
        logger.debug("Paramers retrieved from %s", params_path)
        return data
    except Exception as e:
        logger.error("Unexpected Error while loading a params yaml file %s", e)

