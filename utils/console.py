"""Console printing helpers with optional ANSI color formatting.

Provides `print_metrics_report` for unified, colorful reports without adding new
dependencies. Colors can be disabled by setting environment variable NO_COLOR=1.
"""
from typing import Dict, Optional
import os

# Basic ANSI styles
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"

# Colors
_CYAN = "\x1b[96m"
_GREEN = "\x1b[92m"
_MAGENTA = "\x1b[95m"
_YELLOW = "\x1b[93m"
_RED = "\x1b[91m"
_WHITE = "\x1b[97m"


def _use_color() -> bool:
    return os.getenv("NO_COLOR", "0") == "0"


def _color(text: str, code: str) -> str:
    if not _use_color():
        return text
    return f"{code}{text}{_RESET}"


def print_metrics_report(title: str, metrics: Dict[str, float], units: Optional[Dict[str, str]] = None) -> None:
    """Nicely format a metrics dictionary as a colorful report.

    Args:
        title: Header text
        metrics: Ordered dict-like mapping of metric name -> numeric value
        units: optional mapping name -> unit string
    """
    units = units or {}

    # header
    width = 60
    print("\n" + _color("┏" + "━" * width + "┓", _CYAN))
    print(_color("┃ ", _CYAN) + _color(_BOLD + f"{title}".center(width) + _RESET, _WHITE) + _color(" ┃", _CYAN))
    print(_color("┣" + "━" * width + "┫", _CYAN))

    # body
    for name, val in metrics.items():
        unit = units.get(name, "")
        name_s = _color(f" {name}", _WHITE)
        val_s = _color(f"{val:.6f}", _MAGENTA)
        if unit:
            unit_s = _color(f" {unit}", _DIM)
        else:
            unit_s = ""
        print(_color("┃", _CYAN) + name_s.ljust(width - 20) + val_s.rjust(12) + unit_s + _color(" ┃", _CYAN))

    print(_color("┗" + "━" * width + "┛", _CYAN) + "\n")


def print_simple(title: str, msg: str) -> None:
    print(_color(f"== {title} ==", _YELLOW) + " " + _color(msg, _WHITE))
