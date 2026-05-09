"""Evaluation module for backtesting, optimization, and drift detection."""

from src.evaluation.metrics import BacktestResult, TargetMetrics
from src.evaluation.backtest_runner import BacktestRunner

__all__ = ["BacktestResult", "TargetMetrics", "BacktestRunner"]
