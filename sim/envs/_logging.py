"""Drop-in replacement for ``rlinf.utils.logging.get_logger``."""

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a standard-library logger.

    RLinf used ``Worker.logger`` (a process-scoped logger).  For the
    extracted env layer we simply return a module-level logger.
    """
    return logging.getLogger(name or __name__)
