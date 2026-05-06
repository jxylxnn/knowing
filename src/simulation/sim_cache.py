"""Disk-cache utilities for the GameSimulator."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SimCacheMixin:
    """Mixin providing JSON-based disk-cache operations for the game simulator.

    Requires the host class to define ``self.cache_dir`` (a Path) and
    ``self.cache_dir.mkdir(parents=True, exist_ok=True)`` during init.
    """

    cache_dir: Path

    def _get_cache_key(self, *args) -> str:
        """Generate a cache key from arguments."""
        key_str = str(args)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _serialize_for_cache(self, data: Any) -> Any:
        """Convert data to JSON-serializable format."""
        if isinstance(data, np.ndarray) or (
            hasattr(data, 'detach') and hasattr(data, 'cpu') and hasattr(data, 'numpy')
        ):
            return {'__type__': 'array', 'data': data.tolist()}
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, dict):
            return {k: self._serialize_for_cache(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._serialize_for_cache(item) for item in data]
        elif isinstance(data, pd.DataFrame):
            return {'__type__': 'dataframe', 'data': data.to_dict(orient='records')}
        return data

    def _deserialize_from_cache(self, data: Any) -> Any:
        """Convert JSON data back to original types."""
        if isinstance(data, dict):
            if data.get('__type__') == 'array':
                return np.array(data['data'])
            elif data.get('__type__') == 'dataframe':
                return pd.DataFrame(data['data'])
            return {k: self._deserialize_from_cache(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._deserialize_from_cache(item) for item in data]
        return data

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from disk cache if available (JSON format for security)."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return self._deserialize_from_cache(data)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.debug(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to disk cache (JSON format for security)."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            serializable_data = self._serialize_for_cache(data)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
        except (TypeError, ValueError) as e:
            logger.debug(f"Failed to save cache {cache_key}: {e}")
