# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Основной файл запуска анализатора норм локомотивов РЖД
Python 3.12 optimized version (Fixed)
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
        logger.info("Starting Locomotive Energy Analyzer v2.0 (Fixed)")
        
        # Check Python version
        if sys.version_info < (3, 8):
            logger.error(f"Python 3.8+ required, got {sys.version}")
            tk.messagebox.showerror("Ошибка версии", 
                                  f"Требуется Python 3.8+\nТекущая версия: {sys.version}")
            sys.exit(1)
        
        # Create main window
        root = tk.Tk()
        
        # Set window properties
        root.state('zoomed') if sys.platform == 'win32' else root.attributes('-zoomed', True)
        
        # Initialize application
        app = NormsAnalyzerGUI(root)
        
        # Start main loop
        root.mainloop()
        
    except ImportError as e:
        error_msg = f"Отсутствует необходимая библиотека: {e}"
        logger.error(error_msg)
        if 'root' in locals():
            tk.messagebox.showerror("Ошибка импорта", error_msg)
        else:
            print(error_msg)
        sys.exit(1)
        
    except Exception as e:
        error_msg = f"Критическая ошибка приложения: {e}"
        logging.error(error_msg)
        if 'root' in locals():
            tk.messagebox.showerror("Критическая ошибка", error_msg)
        else:
            print(error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main()