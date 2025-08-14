# core/__init__.py
"""
Core module for locomotive energy analysis
Contains optimized data processing components
"""

from .filter import LocomotiveFilter
from .coefficients import CoefficientManager, LocomotiveCoefficient

__version__ = "2.0.0"
__author__ = "Railway Energy Analysis Team"

__all__ = [
    "LocomotiveFilter",
    "CoefficientManager", 
    "LocomotiveCoefficient"
]