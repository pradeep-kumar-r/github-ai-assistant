import os
import sys

from loguru import logger

from .constants import ROOT_DIR

LOGS_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOGS_DIR, "app.log")
LOG_FILE_DEBUG = os.path.join(LOGS_DIR, "app_debug.log")

logger.remove()

LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=LOG_FORMAT,
    enqueue=True,
    backtrace=True,
    diagnose=False,
)

logger.add(
    LOG_FILE_DEBUG,
    rotation="10 MB",
    retention="14 days",
    level="DEBUG",
    format=LOG_FORMAT,
    enqueue=True,
    backtrace=True,
    diagnose=False,
)

logger.add(
    LOG_FILE,
    rotation="10 MB",
    retention="14 days",
    level="INFO",
    format=LOG_FORMAT,
    enqueue=True,
    backtrace=True,
    diagnose=False,
)

if __name__ == "__main__":
    logger.info("TEST -> Logger Info")
    logger.debug("TEST -> Logger Debug")
    logger.error("TEST -> Logger Error")
