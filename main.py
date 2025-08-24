# main.py (обновленный)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import tkinter as tk
import logging
from datetime import datetime

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Настраивает систему логирования"""
    
    # Создаем директорию для логов если её нет
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Формируем имя файла лога с текущей датой
    log_filename = os.path.join(log_dir, f'analyzer_{datetime.now().strftime("%Y%m%d")}.log')
    
    # Настройка корневого логгера
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Настройка логгеров для внешних библиотек
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('plotly').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("=== ЗАПУСК АНАЛИЗАТОРА НОРМ РАСХОДА ЭЛЕКТРОЭНЕРГИИ (HTML версия) ===")
    logger.info(f"Версия Python: {sys.version}")
    logger.info(f"Рабочая директория: {os.getcwd()}")
    logger.info(f"Файл логов: {log_filename}")
    
    return logger

def check_dependencies():
    """Проверяет наличие необходимых зависимостей"""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('plotly', 'plotly'),
        ('scipy', 'scipy'),
        ('beautifulsoup4', 'bs4'),
        ('openpyxl', 'openpyxl')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            logger.debug(f"✓ Пакет {package_name} найден")
        except ImportError:
            missing_packages.append(package_name)
            logger.error(f"✗ Пакет {package_name} не найден")
    
    if missing_packages:
        logger.error("Отсутствуют необходимые пакеты:")
        for package in missing_packages:
            logger.error(f"  - {package}")
        logger.error("Установите отсутствующие пакеты с помощью: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("Все необходимые зависимости найдены")
    return True

def check_file_structure():
    """Проверяет структуру файлов проекта"""
    logger = logging.getLogger(__name__)
    
    required_dirs = [
        'analysis',
        'core',
        'dialogs',
        'gui'
    ]
    
    required_files = [
        'analysis/__init__.py',
        'analysis/analyzer.py',
        'analysis/html_route_processor.py',
        'analysis/html_norm_processor.py',
        'core/__init__.py',
        'core/filter.py',
        'core/coefficients.py',
        'core/norm_storage.py',
        'dialogs/__init__.py',
        'dialogs/selector.py',
        'gui/__init__.py',
        'gui/interface.py'
    ]
    
    # Проверяем директории
    for directory in required_dirs:
        if not os.path.exists(directory):
            logger.error(f"Отсутствует директория: {directory}")
            return False
        logger.debug(f"✓ Директория {directory} найдена")
    
    # Проверяем файлы
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            logger.error(f"✗ Отсутствует файл: {file_path}")
        else:
            logger.debug(f"✓ Файл {file_path} найден")
    
    if missing_files:
        logger.error(f"Отсутствует {len(missing_files)} файлов проекта")
        return False
    
    logger.info("Структура файлов проекта корректна")
    return True

def create_initial_directories():
    """Создает необходимые директории"""
    logger = logging.getLogger(__name__)
    
    directories = ['logs', 'temp', 'exports']
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Создана директория: {directory}")
            except Exception as e:
                logger.warning(f"Не удалось создать директорию {directory}: {e}")
        else:
            logger.debug(f"Директория {directory} уже существует")

def main():
    """Главная функция приложения"""
    
    # Настройка логирования
    logger = setup_logging()
    
    try:
        # Проверка зависимостей
        if not check_dependencies():
            logger.error("Проверка зависимостей не пройдена")
            input("Нажмите Enter для выхода...")
            return False
        
        # Проверка структуры файлов
        if not check_file_structure():
            logger.error("Проверка структуры файлов не пройдена")
            input("Нажмите Enter для выхода...")
            return False
        
        # Создание необходимых директорий
        create_initial_directories()
        
        # Импорт GUI модуля
        logger.info("Загружаем интерфейс приложения...")
        from gui.interface import NormsAnalyzerGUI
        
        # Создание главного окна
        logger.info("Создаем главное окно приложения...")
        root = tk.Tk()
        
        # Настройка окна
        root.state('zoomed') if os.name == 'nt' else root.attributes('-zoomed', True)
        
        # Создание приложения
        app = NormsAnalyzerGUI(root)
        
        logger.info("Интерфейс загружен успешно")
        logger.info("=== ПРИЛОЖЕНИЕ ГОТОВО К РАБОТЕ ===")
        
        # Запуск главного цикла
        root.mainloop()
        
        logger.info("Приложение завершено")
        return True
        
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}")
        logger.error("Убедитесь, что все модули проекта находятся в правильных директориях")
        input("Нажмите Enter для выхода...")
        return False
        
    except Exception as e:
        logger.error(f"Критическая ошибка запуска: {e}")
        logger.exception("Полная информация об ошибке:")
        input("Нажмите Enter для выхода...")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nПриложение прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"Неожиданная ошибка: {e}")
        sys.exit(1)
