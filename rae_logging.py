"""Centralized logging configuration for RAE pipeline runs."""

import logging
import os


def setup_logging(run_id, log_dir="logs", console_level=logging.INFO):
    """Configure root logger with console + file handlers.

    Console: INFO level, human-readable format.
    File: DEBUG level, written to logs/run_{run_id}.log.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"run_{run_id}.log")

    fmt = "%(asctime)s %(levelname)-5s %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear any existing handlers (e.g. from basicConfig)
    root.handlers.clear()

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

    return log_path
