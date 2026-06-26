"""KineBench public API."""

from kinebench.config import load_config
from kinebench.eval.evaluator import KineBenchEvaluator

__all__ = ["KineBenchEvaluator", "load_config"]

