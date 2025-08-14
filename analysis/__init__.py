# analysis/__init__.py
"""
Analysis module for railway locomotive energy consumption norms
Optimized with Python 3.12 features and vectorized operations
"""

from .analyzer import (
    InteractiveNormsAnalyzer,
    NormDefinition,
    NormsManager,
    VectorizedAnalysisStrategy
)

__version__ = "2.0.0"
__author__ = "Railway Energy Analysis Team"

__all__ = [
    "InteractiveNormsAnalyzer",
    "NormDefinition", 
    "NormsManager",
    "VectorizedAnalysisStrategy"
]