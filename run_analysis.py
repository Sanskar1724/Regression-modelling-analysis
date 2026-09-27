"""Command-line entry point: ``python run_analysis.py``.

A logging handler tees every record to both the console and
``outputs/console_output.txt`` so the run can be inspected after the fact.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

# Allow running directly from a clone without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from regression_analysis.config import OUTPUTS_DIR  # noqa: E402
from regression_analysis.logging_utils import get_logger  # noqa: E402
from regression_analysis.pipeline import run  # noqa: E402

LOG_PATH = OUTPUTS_DIR / "console_output.txt"


def _attach_file_handler() -> logging.Handler:
    """Mirror every log record (including those from child loggers) to the log file."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    # Child loggers propagate to the root logger, so the handler goes there.
    logging.getLogger().addHandler(handler)
    return handler


def main() -> int:
    """Run the full analysis and write the artefacts."""
    logger = get_logger("regression_analysis")
    handler = _attach_file_handler()
    started = time.time()
    try:
        results = run()
        figures = len(list((OUTPUTS_DIR / "figures").glob("*.png")))
        tables = len(list((OUTPUTS_DIR / "tables").glob("*")))
        logger.info("")
        logger.info("=" * 78)
        logger.info("ANALYSIS COMPLETE  (%.1f seconds)", time.time() - started)
        logger.info("=" * 78)
        logger.info("Figures written : %d  -> outputs/figures/", figures)
        logger.info("Tables written  : %d  -> outputs/tables/", tables)
        logger.info("Console log     : %s", LOG_PATH)
        logger.info("")
        logger.info("Headline result : %s", results["comparison"].index[0])
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

