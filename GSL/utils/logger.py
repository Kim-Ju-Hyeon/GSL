import logging


def setup_logging(log_level, log_file, logger_name="exp_logger"):
    """ Setup logging """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(levelname)-5s | File %(filename)-20s | Line %(lineno)-5d | %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=numeric_level)

    logger = logging.getLogger(logger_name)

    formatter = logging.Formatter(
        "%(levelname)-5s | %(filename)-25s | line %(lineno)-5d: %(message)s"
    )

    # define a Handler which writes messages to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    # add the handler to the root logger

    file_console = logging.FileHandler(log_file)
    file_console.setLevel(numeric_level)
    file_console.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_console)

    return logger


def get_logger(logger_name="exp_logger"):
    return logging.getLogger(logger_name)
