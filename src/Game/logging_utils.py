import logging
import sys
from pathlib import Path


def build_gameplay_logger(
    name: str,
    *,
    to_stdout: bool = False,
    file_path: Path | None = None,
) -> logging.Logger:
    """Build a per-instance gameplay logger with isolated handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Ensure handler configuration is deterministic even if a logger name is reused.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    formatter = logging.Formatter("%(message)s")
    has_real_handler = False

    if to_stdout:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        has_real_handler = True

    if file_path is not None:
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        has_real_handler = True

    if not has_real_handler:
        logger.addHandler(logging.NullHandler())

    return logger
