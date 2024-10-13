import logging.config
import yaml

with open('Configs/logger.yaml', 'r') as file:
    config = yaml.safe_load(file.read())
    logging.config.dictConfig(config)
    logging.captureWarnings(True)


def getLogger(name: str):
    """
    Return a logger that interact with
    config file.
    :param name: name of logger
    :return: logger that report msg to yaml file
    """
    logger = logging.getLogger(name)
    return logger
