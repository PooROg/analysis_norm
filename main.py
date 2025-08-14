# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Точка входа в приложение анализатора норм локомотивов РЖД
Python 3.12 optimized version
"""

import tkinter as tk
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gui.interface import NormsAnalyzerGUI

def setup_logging():
    """Configure application logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('locomotive_analyzer.log', encoding='utf-8')
        ]
    )

def main():
    """Application entry point."""
    try:
        setup_logging()
        logger = logging.getLogger(__name__)
        logger.info("Starting Locomotive Energy Analyzer v2.0")
        
        root = tk.Tk()
        app = NormsAnalyzerGUI(root)
        root.mainloop()
        
    except Exception as e:
        logging.error(f"Critical application error: {e}")
        if 'root' in locals():
            tk.messagebox.showerror("Критическая ошибка", 
                                  f"Не удалось запустить приложение:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()