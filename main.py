# main.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализатор норм расхода электроэнергии РЖД
Обновленная версия для работы с HTML файлами

Автор: AI Assistant
Версия: 2.0 (HTML Support)
Дата: 2024
"""

import tkinter as tk
import sys
import os
import logging
from pathlib import Path

# Добавляем текущую директорию в путь для импортов
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from gui.interface import NormsAnalyzerGUI

def setup_logging():
    """Настройка системы логирования"""
    log_dir = current_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'analyzer.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Настройка уровня логирования для различных модулей
    logging.getLogger('analysis.html_parser').setLevel(logging.DEBUG)
    logging.getLogger('analysis.analyzer').setLevel(logging.INFO)
    logging.getLogger('core.filter').setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info("Система логирования настроена")
    logger.info(f"Рабочая директория: {current_dir}")
    
    return logger

def check_dependencies():
    """Проверка необходимых зависимостей"""
    required_modules = [
        'pandas', 'numpy', 'plotly', 'scipy', 'openpyxl', 'bs4'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("❌ Отсутствуют необходимые модули:")
        for module in missing_modules:
            print(f"  - {module}")
        print("\nУстановите их командой:")
        print(f"pip install {' '.join(missing_modules)}")
        return False
    
    return True

def main():
    """Главная функция приложения"""
    print("=" * 60)
    print("АНАЛИЗАТОР НОРМ РАСХОДА ЭЛЕКТРОЭНЕРГИИ РЖД")
    print("Версия 2.0 - Поддержка HTML файлов")
    print("=" * 60)
    
    # Настройка логирования
    logger = setup_logging()
    
    # Проверка зависимостей
    if not check_dependencies():
        input("Нажмите Enter для выхода...")
        return
    
    logger.info("Все зависимости найдены")
    
    try:
        # Создание главного окна
        logger.info("Запуск GUI приложения")
        root = tk.Tk()
        
        # Настройка иконки если есть
        try:
            icon_path = current_dir / "assets" / "icon.ico"
            if icon_path.exists():
                root.iconbitmap(str(icon_path))
        except:
            pass
        
        # Создание интерфейса
        app = NormsAnalyzerGUI(root)
        
        # Центрирование окна
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"{width}x{height}+{x}+{y}")
        
        logger.info("GUI приложение запущено успешно")
        
        # Запуск главного цикла
        root.mainloop()
        
    except Exception as e:
        logger.exception("Критическая ошибка при запуске приложения")
        print(f"❌ Критическая ошибка: {e}")
        input("Нажмите Enter для выхода...")
    
    finally:
        logger.info("Приложение завершено")

if __name__ == "__main__":
    main()