import logging
import os
import sys

from loguru import logger as pipecat_logger


def configure_pipecat_logging() -> None:
    """Pipecat logs the full LLM context (names, dates of birth) at DEBUG. Debtor data is
    treated as real PII, so Pipecat stays at WARNING unless PIPECAT_LOG_LEVEL raises it.
    Every entry point (web app, eval CLI) must call this before running a pipeline."""
    pipecat_logger.remove()
    pipecat_logger.add(sys.stderr, level=os.environ.get("PIPECAT_LOG_LEVEL", "WARNING"))
    # Our own `app.*` loggers write ids, counts and timings only (never what was said), so
    # INFO is safe. Without a handler, Python dropped them (voice call 5: no probe lines).
    app_logger = logging.getLogger("app")
    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        app_logger.addHandler(handler)
    app_logger.setLevel(os.environ.get("APP_LOG_LEVEL", "INFO"))
