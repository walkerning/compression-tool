import logging
def init_logging(quiet=False):
    level = logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level)
