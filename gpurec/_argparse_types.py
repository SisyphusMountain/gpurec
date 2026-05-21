from __future__ import annotations

import argparse
import math
from collections.abc import Callable


def _integer_arg(name: str, text: str) -> int:
    try:
        return int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc


def positive_int_arg(name: str) -> Callable[[str], int]:
    def parse(text: str) -> int:
        number = _integer_arg(name, text)
        if number <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be positive")
        return number

    return parse


def positive_even_int_arg(name: str) -> Callable[[str], int]:
    def parse(text: str) -> int:
        number = _integer_arg(name, text)
        if number <= 0 or number % 2 != 0:
            raise argparse.ArgumentTypeError(
                f"{name} must be a positive even integer"
            )
        return number

    return parse


def nonnegative_int_arg(name: str) -> Callable[[str], int]:
    def parse(text: str) -> int:
        number = _integer_arg(name, text)
        if number < 0:
            raise argparse.ArgumentTypeError(f"{name} must be non-negative")
        return number

    return parse


def nonnegative_float_arg(name: str) -> Callable[[str], float]:
    def parse(text: str) -> float:
        try:
            number = float(text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be a number") from exc
        if not math.isfinite(number):
            raise argparse.ArgumentTypeError(f"{name} must be finite")
        if number < 0.0:
            raise argparse.ArgumentTypeError(f"{name} must be non-negative")
        return number

    return parse
