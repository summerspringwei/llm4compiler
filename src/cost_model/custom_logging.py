
import logging
def get_custom_logger():
    # Create a logger
    logger = logging.getLogger('ll2compiler')
    logger.setLevel(logging.DEBUG)

    # Create a custom formatter
    formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')

    # Create a handler and set the formatter
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    return logger