"""Base scraper + registry for pluggable data sources.

Historically every data source was wired directly into ``update_data.py`` with
no shared interface, so adding a new source meant editing the orchestrator.
This module provides the contract that lets a new data source be added by
dropping a single self-contained module into ``src/data/extensions/``.

Contract
--------
A scraper is a concrete subclass of :class:`BaseScraper` that implements:

* ``name``           — stable identifier used by config and logging.
* ``output_files()`` — the relative paths (under ``data/``) this scraper
  writes, so ``clear_cache`` / contracts / the cache key can see them.
* ``fetch(config)``  — return a mapping of ``{relative_path: DataFrame}`` for
  the files it produces. Returning ``{}`` means "nothing to write this run".

Discovery
---------
:class:`ScraperRegistry` scans ``src/data/extensions/`` for modules and
collects concrete ``BaseScraper`` subclasses. Import failures are isolated so
one broken extension never breaks the core data fetch. ``update_data.py``
iterates the *enabled* scrapers (gated by the ``data_sources`` config block)
after the core NBA fetch and writes each DataFrame, isolating per-scraper
errors.

Safety
------
* Scrapers are opt-in: absent the ``data_sources`` config block, zero
  extension scrapers run, so existing behaviour is unchanged.
* A scraper's ``output_files()`` never overwrite the canonical
  ``nba_players.csv`` / ``nba_games.csv`` produced by the core fetch — the
  orchestrator refuses to write any path that collides with the core outputs.
"""

from __future__ import annotations

import abc
import importlib
import logging
import os
import pkgutil
from typing import Any, Dict, List, Optional, Tuple, Type

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")
EXTENSIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extensions")

# Canonical core outputs that extension scrapers must never overwrite.
CORE_PROTECTED_FILES = {"nba_players.csv", "nba_games.csv"}


class BaseScraper(abc.ABC):
    """Contract for a pluggable data source."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable identifier for this scraper (used by config + logs)."""

    @abc.abstractmethod
    def output_files(self) -> List[str]:
        """Relative paths (under ``data/``) this scraper writes.

        Used for cache-key awareness and to guard against overwriting core
        outputs. Returning an empty list is valid (a scraper that enriches
        in-memory only).
        """

    @abc.abstractmethod
    def fetch(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
        """Fetch data and return ``{relative_path_under_data: DataFrame}``.

        Returning an empty dict means nothing should be written this run
        (e.g. no new data available). Implementations should be resilient:
        raise only on truly fatal errors; transient issues should be logged
        and an empty dict returned.
        """

    def requires(self) -> List[str]:
        """Declare external files this scraper needs to already exist.

        The orchestrator logs (but does not hard-fail) if a declared
        dependency is missing, so a scraper that augments the core data can
        degrade gracefully when run before the first core fetch.
        """
        return []


def _is_concrete_scraper(obj: object) -> bool:
    return (
        isinstance(obj, type)
        and issubclass(obj, BaseScraper)
        and obj is not BaseScraper
    )


def discover_scrapers(extensions_dir: Optional[str] = None) -> List[Type[BaseScraper]]:
    """Import every scraper extension module and collect concrete scrapers."""
    scan_dir = extensions_dir or EXTENSIONS_DIR
    discovered: List[Type[BaseScraper]] = []
    if not os.path.isdir(scan_dir):
        return discovered

    package = "src.data.extensions"
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not import data extensions package: %s", exc)
        return discovered

    for mod_info in pkgutil.iter_modules(pkg.__path__):
        mod_name = f"{package}.{mod_info.name}"
        try:
            module = importlib.import_module(mod_name)
        except Exception as exc:
            logger.warning("Skipping data extension %s (import failed): %s", mod_name, exc)
            continue
        for attr_name in dir(module):
            obj = getattr(module, attr_name, None)
            if _is_concrete_scraper(obj):
                discovered.append(obj)  # type: ignore[arg-type]
    return discovered


class ScraperRegistry:
    """Discovers and instantiates pluggable scrapers."""

    def __init__(self, extensions_dir: Optional[str] = None) -> None:
        self._extensions_dir = extensions_dir
        self._scrapers: Optional[List[Type[BaseScraper]]] = None

    def _load(self) -> List[Type[BaseScraper]]:
        if self._scrapers is None:
            self._scrapers = discover_scrapers(self._extensions_dir)
        return self._scrapers

    def all_names(self) -> List[str]:
        names: List[str] = []
        for cls in self._load():
            try:
                names.append(cls().name)
            except Exception as exc:
                logger.warning("Could not read name from scraper %s: %s", cls.__name__, exc)
        return names

    def build_enabled(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, BaseScraper]]:
        """Instantiate scrapers that are enabled in the ``data_sources`` config.

        ``config`` is the parsed ``data_sources`` block: a mapping of
        ``{scraper_name: {"enabled": bool, ...}}``. Scrapers not listed in
        config default to disabled. Returns ``[(name, instance), ...]`` in
        config order, with un-configured scrapers appended at the end only if
        ``enable_unconfigured`` is True (default False — opt-in safety).
        """
        config = config or {}
        enabled_names = [
            name for name, cfg in config.items() if isinstance(cfg, dict) and cfg.get("enabled")
        ]

        instances: Dict[str, BaseScraper] = {}
        for cls in self._load():
            try:
                inst = cls()
            except Exception as exc:
                logger.warning("Could not instantiate scraper %s: %s", cls.__name__, exc)
                continue
            instances[inst.name] = inst

        ordered: List[Tuple[str, BaseScraper]] = []
        for name in enabled_names:
            if name in instances:
                ordered.append((name, instances[name]))
        return ordered

    def validate_outputs(self, scraper: BaseScraper) -> Tuple[bool, List[str]]:
        """Ensure a scraper's declared outputs do not collide with core files."""
        collisions = [f for f in scraper.output_files() if f in CORE_PROTECTED_FILES]
        return (not collisions, collisions)


_default_registry: Optional[ScraperRegistry] = None


def get_scraper_registry() -> ScraperRegistry:
    """Return the process-wide default scraper registry (built lazily)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ScraperRegistry()
    return _default_registry