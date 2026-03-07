"""Smart caching system for expensive feature engineering operations."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureCache:
    """Cache for expensive feature engineering and data processing.
    
    Automatically manages cache invalidation based on data hashes
    and provides persistent storage for processed features.
    """
    
    def __init__(
        self,
        cache_dir: Union[str, Path] = 'cache/training',
        max_age_hours: float = 24.0,
        compression: str = 'gzip',
    ):
        """Initialize the feature cache.
        
        Args:
            cache_dir: Directory to store cached files
            max_age_hours: Maximum age of cache before recompute
            compression: Compression method for stored files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_hours = max_age_hours
        self.compression = compression
        self._access_log: Dict[str, float] = {}
        
        logger.info(f"FeatureCache initialized at {self.cache_dir}")
    
    def _compute_hash(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> str:
        """Compute a hash for the given data.
        
        Args:
            df: DataFrame to hash
            columns: Specific columns to include in hash (default: all)
            
        Returns:
            MD5 hash string (truncated to 16 chars)
        """
        if columns:
            df = df[columns].copy()
        
        # Use shape, dtypes, and sample data for hash
        content = [
            str(df.shape),
            str(df.dtypes.to_dict()),
        ]
        
        # Add sample of data (first and last 1000 rows)
        sample_size = min(1000, len(df))
        if len(df) > 0:
            content.append(str(df.head(sample_size).values.tobytes()))
            if len(df) > sample_size:
                content.append(str(df.tail(sample_size).values.tobytes()))
        
        content_str = '|'.join(content)
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, key: str, extension: str = 'pkl') -> Path:
        """Get the full path for a cache entry.
        
        Args:
            key: Cache key
            extension: File extension
            
        Returns:
            Path to cache file
        """
        return self.cache_dir / f"{key}.{extension}"
    
    def get(
        self,
        df: pd.DataFrame,
        operation_name: str,
        columns: Optional[List[str]] = None,
    ) -> Optional[Any]:
        """Try to load cached result for an operation.
        
        Args:
            df: Input DataFrame
            operation_name: Name of the operation
            columns: Columns to include in hash computation
            
        Returns:
            Cached result or None if not found/stale
        """
        data_hash = self._compute_hash(df, columns)
        cache_key = f"{operation_name}_{data_hash}"
        cache_path = self._get_cache_path(cache_key)
        
        if not cache_path.exists():
            return None
        
        # Check age
        import time
        age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if age_hours > self.max_age_hours:
            logger.info(f"Cache entry {cache_key} is stale (age={age_hours:.1f}h)")
            cache_path.unlink(missing_ok=True)
            return None
        
        try:
            result = joblib.load(cache_path)
            logger.info(f"Loaded cached {operation_name} (key={cache_key})")
            self._access_log[cache_key] = time.time()
            return result
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None
    
    def set(
        self,
        df: pd.DataFrame,
        operation_name: str,
        result: Any,
        columns: Optional[List[str]] = None,
    ) -> None:
        """Store result in cache.
        
        Args:
            df: Input DataFrame
            operation_name: Name of the operation
            result: Result to cache
            columns: Columns to include in hash computation
        """
        data_hash = self._compute_hash(df, columns)
        cache_key = f"{operation_name}_{data_hash}"
        cache_path = self._get_cache_path(cache_key)
        
        try:
            joblib.dump(result, cache_path, compress=self.compression)
            logger.info(f"Cached {operation_name} (key={cache_key})")
        except Exception as e:
            logger.warning(f"Failed to cache {cache_key}: {e}")
    
    def cache_or_compute(
        self,
        df: pd.DataFrame,
        operation_name: str,
        compute_fn: callable,
        columns: Optional[List[str]] = None,
    ) -> Any:
        """Get from cache or compute and store.
        
        Args:
            df: Input DataFrame
            operation_name: Name of the operation
            compute_fn: Function to compute result if not cached
            columns: Columns to include in hash computation
            
        Returns:
            Result (from cache or computed)
        """
        # Try cache first
        cached = self.get(df, operation_name, columns)
        if cached is not None:
            return cached
        
        # Compute and store
        result = compute_fn()
        self.set(df, operation_name, result, columns)
        return result
    
    def clear(self, older_than_hours: Optional[float] = None) -> int:
        """Clear cache entries.
        
        Args:
            older_than_hours: Only clear entries older than this (None = clear all)
            
        Returns:
            Number of entries cleared
        """
        import time
        
        cleared = 0
        for cache_file in self.cache_dir.glob('*.pkl'):
            if older_than_hours is not None:
                age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
                if age_hours < older_than_hours:
                    continue
            
            cache_file.unlink(missing_ok=True)
            cleared += 1
        
        logger.info(f"Cleared {cleared} cache entries")
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        import time
        
        total_size = 0
        num_entries = 0
        oldest_age = 0
        
        for cache_file in self.cache_dir.glob('*.pkl'):
            stat = cache_file.stat()
            total_size += stat.st_size
            num_entries += 1
            age = (time.time() - stat.st_mtime) / 3600
            oldest_age = max(oldest_age, age)
        
        return {
            'num_entries': num_entries,
            'total_size_mb': total_size / (1024 * 1024),
            'oldest_age_hours': oldest_age,
            'cache_dir': str(self.cache_dir),
        }


class DataSplitCache:
    """Cache for data splits and preprocessing results."""
    
    def __init__(self, cache_dir: Union[str, Path] = 'cache/splits'):
        """Initialize the split cache.
        
        Args:
            cache_dir: Directory to store cached splits
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._splits: Dict[str, Dict[str, Any]] = {}
    
    def get_split_key(
        self,
        df: pd.DataFrame,
        split_type: str,
        split_params: Dict[str, Any],
    ) -> str:
        """Generate a unique key for a data split.
        
        Args:
            df: DataFrame to split
            split_type: Type of split (e.g., 'temporal', 'random')
            split_params: Parameters for the split
            
        Returns:
            Unique hash key for this split
        """
        data_hash = hashlib.md5(
            str(df.shape).encode() + 
            str(df.dtypes.to_dict()).encode() +
            str(df.head(100).values.tobytes())
        ).hexdigest()[:16]
        
        params_hash = hashlib.md5(
            json.dumps(split_params, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"{split_type}_{data_hash}_{params_hash}"
    
    def cache_split(
        self,
        key: str,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Cache a data split.
        
        Args:
            key: Split key
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame (optional)
        """
        cache_path = self.cache_dir / f"{key}.joblib"
        
        data = {
            'train': train_df,
            'val': val_df,
            'test': test_df,
        }
        
        joblib.dump(data, cache_path, compress='gzip')
        self._splits[key] = data
        
        logger.info(f"Cached data split: {key}")
    
    def load_split(
        self,
        key: str,
    ) -> Optional[Dict[str, Any]]:
        """Load a cached data split.
        
        Args:
            key: Split key
            
        Returns:
            Dictionary with train/val/test DataFrames, or None if not found
        """
        # Check in-memory first
        if key in self._splits:
            return self._splits[key]
        
        # Check disk
        cache_path = self.cache_dir / f"{key}.joblib"
        if not cache_path.exists():
            return None
        
        try:
            data = joblib.load(cache_path)
            self._splits[key] = data
            logger.info(f"Loaded cached split: {key}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load split {key}: {e}")
            return None