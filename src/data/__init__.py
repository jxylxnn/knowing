"""Data acquisition and scraping module."""

from importlib import import_module
from typing import Any

__all__ = ['InjuryScraper', 'ScheduleScraper']


def __getattr__(name: str) -> Any:
    if name == 'InjuryScraper':
        return import_module('.injury_scraper', __name__).InjuryScraper
    if name == 'ScheduleScraper':
        return import_module('.schedule_scraper', __name__).ScheduleScraper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
