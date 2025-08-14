# core/__init__.py
"""
Core module for locomotive energy analysis
Contains optimized data processing components with Python 3.12 features
"""

from .filter import LocomotiveFilter
from .coefficients import LocomotiveCoefficientsManager, CoefficientManager, LocomotiveCoefficient

__version__ = "2.0.0"
__author__ = "Railway Energy Analysis Team"

__all__ = [
    "LocomotiveFilter",
    "LocomotiveCoefficientsManager",
    "CoefficientManager", 
    "LocomotiveCoefficient"
]