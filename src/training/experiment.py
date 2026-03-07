"""Experiment tracking system for training runs."""

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExperimentRun:
    """Data class representing a single training experiment."""
    id: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_files: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    status: str = "running"
    error_message: Optional[str] = None
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRun':
        """Create from dictionary."""
        return cls(**data)


class ExperimentTracker:
    """Track training experiments with metrics and artifacts."""
    
    def __init__(
        self,
        experiments_dir: Union[str, Path] = 'experiments',
        experiment_name: Optional[str] = None,
    ):
        """Initialize experiment tracker.
        
        Args:
            experiments_dir: Directory to store experiment data
            experiment_name: Name for this experiment (auto-generated if None)
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.experiments_dir / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize run
        self.run_id = f"{experiment_name}_{int(time.time())}"
        self.current_run: Optional[ExperimentRun] = None
        self._start_time: Optional[float] = None
        
        logger.info(f"Experiment tracker initialized: {self.experiment_dir}")
    
    def start_run(
        self,
        config: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> ExperimentRun:
        """Start a new training run.
        
        Args:
            config: Training configuration
            notes: Optional notes about this run
            
        Returns:
            ExperimentRun object
        """
        self._start_time = time.time()
        
        self.current_run = ExperimentRun(
            id=self.run_id,
            config=config or {},
            notes=notes,
        )
        
        logger.info(f"Started experiment run: {self.run_id}")
        
        return self.current_run
    
    def log_metric(
        self,
        metric_name: str,
        value: float,
        step: Optional[int] = None,
        target: Optional[str] = None,
    ) -> None:
        """Log a metric value.
        
        Args:
            metric_name: Name of the metric (e.g., 'mae', 'rmse')
            value: Metric value
            step: Training step/iteration
            target: Target stat (e.g., 'PTS', 'REB')
        """
        if self.current_run is None:
            logger.warning("No active run, cannot log metric")
            return
        
        key = f"{target}_{metric_name}" if target else metric_name
        
        if "history" not in self.current_run.config:
            self.current_run.config["history"] = {}
        
        if key not in self.current_run.config["history"]:
            self.current_run.config["history"][key] = []
        
        entry = {"value": value, "timestamp": time.time()}
        if step is not None:
            entry["step"] = step
        
        self.current_run.config["history"][key].append(entry)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        target: Optional[str] = None,
    ) -> None:
        """Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
            target: Target stat (optional)
        """
        if self.current_run is None:
            return
        
        for name, value in metrics.items():
            self.log_metric(name, value, target=target)
    
    def log_model_metrics(
        self,
        model_name: str,
        metrics: Dict[str, float],
        target: Optional[str] = None,
    ) -> None:
        """Log metrics for a specific model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metrics
            target: Target stat (optional)
        """
        if self.current_run is None:
            return
        
        key = target if target else "overall"
        if key not in self.current_run.metrics:
            self.current_run.metrics[key] = {}
        
        self.current_run.metrics[key][model_name] = metrics
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str = "model",
    ) -> None:
        """Log an artifact (model file, plot, etc.).
        
        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact
        """
        if self.current_run is None:
            return
        
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return
        
        # Copy to experiment directory
        dest_dir = self.experiment_dir / "artifacts" / artifact_type
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_path = dest_dir / artifact_path.name
        joblib.dump(joblib.load(artifact_path), dest_path)
        
        relative_path = str(dest_path.relative_to(self.experiment_dir))
        self.current_run.model_files.append(relative_path)
        
        logger.info(f"Logged artifact: {relative_path}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            params: Dictionary of parameters
        """
        if self.current_run is None:
            return
        
        if "params" not in self.current_run.config:
            self.current_run.config["params"] = {}
        
        self.current_run.config["params"].update(params)
    
    def end_run(
        self,
        status: str = "completed",
        error_message: Optional[str] = None,
    ) -> ExperimentRun:
        """End the current run.
        
        Args:
            status: Final status ('completed', 'failed', 'aborted')
            error_message: Error message if failed
            
        Returns:
            Completed ExperimentRun
        """
        if self.current_run is None:
            raise RuntimeError("No active run to end")
        
        if self._start_time is not None:
            self.current_run.duration_seconds = time.time() - self._start_time
        
        self.current_run.status = status
        if error_message:
            self.current_run.error_message = error_message
        
        # Save run data
        self._save_run()
        
        logger.info(f"Ended run {self.run_id}: {status} ({self.current_run.duration_seconds:.1f}s)")
        
        return self.current_run
    
    def _save_run(self) -> None:
        """Save current run to disk."""
        if self.current_run is None:
            return
        
        run_file = self.experiment_dir / f"run_{self.run_id}.json"
        
        with open(run_file, 'w') as f:
            json.dump(self.current_run.to_dict(), f, indent=2, default=str)
    
    def get_best_run(
        self,
        metric_name: str = "mae",
        target: str = "PTS",
        select: str = "min",
    ) -> Optional[ExperimentRun]:
        """Get the best run based on a metric.
        
        Args:
            metric_name: Metric to compare
            target: Target stat
            select: 'min' or 'max'
            
        Returns:
            Best ExperimentRun or None
        """
        runs = self.list_runs()
        if not runs:
            return None
        
        def get_metric(run: ExperimentRun) -> float:
            metrics = run.metrics.get(target, {})
            # Look for metric in any model
            for model_metrics in metrics.values():
                if metric_name in model_metrics:
                    return model_metrics[metric_name]
            return float('inf') if select == "min" else float('-inf')
        
        sorted_runs = sorted(runs, key=get_metric, reverse=(select == "max"))
        return sorted_runs[0] if sorted_runs else None
    
    def list_runs(self) -> List[ExperimentRun]:
        """List all runs for this experiment.
        
        Returns:
            List of ExperimentRun objects
        """
        runs = []
        
        for run_file in self.experiment_dir.glob("run_*.json"):
            try:
                with open(run_file, 'r') as f:
                    data = json.load(f)
                    runs.append(ExperimentRun.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load run {run_file}: {e}")
        
        return sorted(runs, key=lambda r: r.timestamp, reverse=True)
    
    def compare_runs(
        self,
        metric_name: str = "mae",
        target: str = "PTS",
    ) -> Dict[str, Any]:
        """Compare metrics across runs.
        
        Args:
            metric_name: Metric to compare
            target: Target stat
            
        Returns:
            Dictionary with comparison data
        """
        runs = self.list_runs()
        
        comparison = {
            "metric": metric_name,
            "target": target,
            "runs": [],
        }
        
        for run in runs:
            metrics = run.metrics.get(target, {})
            for model_name, model_metrics in metrics.items():
                if metric_name in model_metrics:
                    comparison["runs"].append({
                        "run_id": run.id,
                        "timestamp": run.timestamp,
                        "model": model_name,
                        "value": model_metrics[metric_name],
                        "status": run.status,
                    })
        
        # Sort by metric value
        comparison["runs"] = sorted(
            comparison["runs"],
            key=lambda x: x["value"]
        )
        
        return comparison
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of this experiment.
        
        Returns:
            Dictionary with experiment summary
        """
        runs = self.list_runs()
        
        total_runs = len(runs)
        completed = sum(1 for r in runs if r.status == "completed")
        failed = sum(1 for r in runs if r.status == "failed")
        
        # Get average duration
        durations = [r.duration_seconds for r in runs if r.duration_seconds > 0]
        avg_duration = np.mean(durations) if durations else 0
        
        return {
            "experiment_name": self.experiment_name,
            "experiment_dir": str(self.experiment_dir),
            "total_runs": total_runs,
            "completed_runs": completed,
            "failed_runs": failed,
            "average_duration_seconds": avg_duration,
            "best_run": self.get_best_run().id if self.get_best_run() else None,
        }


class ExperimentComparison:
    """Compare multiple experiments."""
    
    def __init__(self, experiments_dir: Union[str, Path] = 'experiments'):
        """Initialize comparison.
        
        Args:
            experiments_dir: Directory containing experiments
        """
        self.experiments_dir = Path(experiments_dir)
    
    def compare_experiments(
        self,
        experiment_names: List[str],
        metric: str = "mae",
        target: str = "PTS",
    ) -> Dict[str, Any]:
        """Compare specific experiments.
        
        Args:
            experiment_names: Names of experiments to compare
            metric: Metric to compare
            target: Target stat
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name in experiment_names:
            tracker = ExperimentTracker(self.experiments_dir, name)
            best_run = tracker.get_best_run(metric, target)
            
            if best_run:
                results[name] = {
                    "best_run_id": best_run.id,
                    "metrics": best_run.metrics.get(target, {}),
                    "duration": best_run.duration_seconds,
                    "timestamp": best_run.timestamp,
                }
        
        return results
    
    def list_all_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments with summaries.
        
        Returns:
            List of experiment summaries
        """
        summaries = []
        
        for exp_dir in self.experiments_dir.iterdir():
            if exp_dir.is_dir():
                tracker = ExperimentTracker(self.experiments_dir, exp_dir.name)
                summaries.append(tracker.get_summary())
        
        return sorted(summaries, key=lambda x: x["experiment_name"])