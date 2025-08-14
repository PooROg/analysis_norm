# gui/__init__.py
"""
GUI module for locomotive energy analysis application
Modern tkinter interface with responsive threading
"""

from .interface import NormsAnalyzerGUI, ApplicationState

__version__ = "2.0.0"
__author__ = "Railway Energy Analysis Team"

__all__ = [
    "NormsAnalyzerGUI",
    "ApplicationState"
]