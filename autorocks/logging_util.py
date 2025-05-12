import logging

LOGGING_FORMAT = "%(asctime)s | %(levelname)s | %(name)s: %(message)s"

ENV_RUNNER_LOGGER = "env_runner"
BOGRAPH_LOGGER = "bograph"
SYSTEM_LOGGER = "system"


class ColoredFormater(logging.Formatter):

    green = "\x1b[32;2m"
    white = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    detailed_format = LOGGING_FORMAT + " (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: white + LOGGING_FORMAT + reset,
        logging.INFO: green + LOGGING_FORMAT + reset,
        logging.WARNING: yellow + detailed_format + reset,
        logging.ERROR: red + detailed_format + reset,
        logging.CRITICAL: bold_red + detailed_format + reset,
    }

    def format(self, record: logging.LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class FileFilter:
    """Allow only LogRecords whose severity levels are below ERROR."""

    def __call__(self, log):
        if log.levelno < logging.ERROR:
            return 1
        else:
            return 0


def log_config_dict(logs_dir: str, debug_mode: bool) -> dict:
    logging_level = logging.DEBUG if debug_mode else logging.ERROR

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colorful": {"()": ColoredFormater},
            "standard": {"format": LOGGING_FORMAT},
        },
        "handlers": {
            "default": {
                # Default is stderr (for pycharm)
                "level": logging.DEBUG,
                "formatter": "colorful",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "errors_to_file": {
                "level": logging.ERROR,
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": f"{logs_dir}/errors.log",
            },
            "info_to_file": {
                "level": logging.INFO,
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": f"{logs_dir}/info.log",
                "filters": ["file_filter"],
            },
        },
        "filters": {
            "file_filter": {
                "()": FileFilter,
            },
        },
        "loggers": {
            SYSTEM_LOGGER: {  # system general logger
                "handlers": ["default", "errors_to_file", "info_to_file"],
                "level": logging_level,
                "propagate": True,
            },
            BOGRAPH_LOGGER: {  # optimizer specific logger
                "handlers": ["default", "errors_to_file", "info_to_file"],
                "level": logging_level,
                "propagate": True,
            },
            ENV_RUNNER_LOGGER: {
                "handlers": ["default", "errors_to_file", "info_to_file"],
                "level": logging_level,
                "propagate": True,
            },
        },
    }
