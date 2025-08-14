# dialogs/__init__.py
"""
Dialog components for locomotive energy analysis GUI
Modern UI patterns with Python 3.12 optimizations
"""

from .selector import LocomotiveSelectorDialog, DialogResult
from .editor import NormEditorDialog, NormComparator, NormData

__version__ = "2.0.0"
__author__ = "Railway Energy Analysis Team"

__all__ = [
    "LocomotiveSelectorDialog",
    "DialogResult",
    "NormEditorDialog", 
    "NormComparator",
    "NormData"
]