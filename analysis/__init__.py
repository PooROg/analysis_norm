#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль анализа норм расхода электроэнергии.
Оптимизирован для Python 3.12 с современными практиками разработки.
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Section Normalization System"
__description__ = "Анализ норм расхода электроэнергии на железнодорожном транспорте"

# Основные классы для импорта
from .analyzer import InteractiveNormsAnalyzer
from .status_config import (
    DEFAULT_STATUS_CONFIG, 
    DEFAULT_PLOT_CONFIG,
    StatusConfig,
    PlotConfig,
    create_custom_status_config
)
from .data_models import (
    AnalysisResult,
    ProcessingStats,
    DefaultDataProcessor,
    NormData,
    NormType,
    RouteMetadata,
    LocoData,
    Yu7Data,
    RouteSection
)

# Процессоры данных
from .html_route_processor import HTMLRouteProcessor
from .html_norm_processor import HTMLNormProcessor
from .norm_storage import NormStorage

# Парсеры
from .html_parser import FastHTMLParser
from .norm_parser import FastNormParser

# Экспорт основных компонентов
__all__ = [
    # Основной анализатор
    "InteractiveNormsAnalyzer",
    
    # Конфигурация
    "DEFAULT_STATUS_CONFIG",
    "DEFAULT_PLOT_CONFIG", 
    "StatusConfig",
    "PlotConfig",
    "create_custom_status_config",
    
    # Модели данных
    "AnalysisResult",
    "ProcessingStats",
    "DefaultDataProcessor",
    "NormData",
    "NormType",
    "RouteMetadata", 
    "LocoData",
    "Yu7Data",
    "RouteSection",
    
    # Процессоры
    "HTMLRouteProcessor",
    "HTMLNormProcessor", 
    "NormStorage",
    
    # Парсеры
    "FastHTMLParser",
    "FastNormParser",
]

# Проверка версии Python
import sys

if sys.version_info < (3, 12):
    import warnings
    warnings.warn(
        f"Модуль analysis оптимизирован для Python 3.12+. "
        f"Текущая версия: {sys.version_info.major}.{sys.version_info.minor}. "
        f"Некоторые оптимизации могут быть недоступны.",
        UserWarning,
        stacklevel=2
    )

# Настройка логирования по умолчанию
import logging

# Создаем logger для модуля если он еще не настроен
_logger = logging.getLogger(__name__)
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    _handler.setFormatter(_formatter)
    _logger.addHandler(_handler)
    _logger.setLevel(logging.INFO)

# Информация о доступности дополнительных библиотек
try:
    import selectolax
    _SELECTOLAX_AVAILABLE = True
except ImportError:
    _SELECTOLAX_AVAILABLE = False

try:
    import plotly
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

try:
    import openpyxl
    _EXCEL_SUPPORT = True
except ImportError:
    _EXCEL_SUPPORT = False

# Функции для проверки возможностей
def get_module_info() -> dict[str, any]:
    """Возвращает информацию о модуле и доступных возможностях."""
    return {
        "version": __version__,
        "description": __description__, 
        "author": __author__,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "capabilities": {
            "fast_html_parsing": _SELECTOLAX_AVAILABLE,
            "interactive_plots": _PLOTLY_AVAILABLE,
            "excel_export": _EXCEL_SUPPORT,
            "python_312_optimizations": sys.version_info >= (3, 12)
        }
    }

def check_dependencies() -> dict[str, bool]:
    """Проверяет наличие всех зависимостей."""
    dependencies = {}
    
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"), 
        ("scipy", "scipy"),
        ("structlog", "structlog")
    ]
    
    optional_packages = [
        ("selectolax", "selectolax"),
        ("plotly", "plotly"),
        ("openpyxl", "openpyxl"),
        ("beautifulsoup4", "bs4")
    ]
    
    # Проверяем обязательные пакеты
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            dependencies[package_name] = True
        except ImportError:
            dependencies[package_name] = False
            _logger.warning(f"Отсутствует обязательный пакет: {package_name}")
    
    # Проверяем опциональные пакеты
    for package_name, import_name in optional_packages:
        try:
            __import__(import_name)
            dependencies[package_name] = True
        except ImportError:
            dependencies[package_name] = False
    
    return dependencies

def verify_installation() -> bool:
    """Проверяет корректность установки модуля."""
    try:
        # Проверяем основные компоненты
        from .analyzer import InteractiveNormsAnalyzer
        from .status_config import DEFAULT_STATUS_CONFIG
        from .data_models import DefaultDataProcessor
        
        # Пробуем создать основные объекты
        analyzer = InteractiveNormsAnalyzer()
        processor = DefaultDataProcessor()
        
        _logger.info("Модуль analysis установлен корректно")
        return True
        
    except Exception as e:
        _logger.error(f"Ошибка при проверке установки модуля: {e}")
        return False

# Автоматическая проверка при импорте (только в debug режиме)
if __debug__ and sys.flags.optimize == 0:
    _deps = check_dependencies()
    _missing_required = [pkg for pkg, available in _deps.items() 
                        if not available and pkg in ["pandas", "numpy", "scipy", "structlog"]]
    
    if _missing_required:
        _logger.error(f"Отсутствуют критически важные зависимости: {_missing_required}")
        _logger.error("Установите их командой: pip install " + " ".join(_missing_required))
    
    if not _SELECTOLAX_AVAILABLE:
        _logger.info("selectolax не установлен. Будет использоваться BeautifulSoup (медленнее)")
        _logger.info("Для улучшения производительности установите: pip install selectolax")

# Lazy import для тяжелых модулей
def _get_advanced_analyzer():
    """Ленивый импорт расширенного анализатора."""
    try:
        from .advanced_analyzer import AdvancedNormsAnalyzer
        return AdvancedNormsAnalyzer
    except ImportError:
        _logger.warning("Расширенный анализатор недоступен")
        return None

# Добавляем в __all__ если доступен
try:
    _AdvancedAnalyzer = _get_advanced_analyzer()
    if _AdvancedAnalyzer:
        __all__.append("AdvancedNormsAnalyzer")
        globals()["AdvancedNormsAnalyzer"] = _AdvancedAnalyzer
except Exception:
    pass

# Настройка предупреждений
import warnings

# Фильтруем некоторые предупреждения pandas/numpy для чистого вывода
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="plotly")

_logger.info(f"Модуль analysis v{__version__} загружен успешно")