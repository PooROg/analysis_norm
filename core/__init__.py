#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Базовый модуль core для системы нормирования участков.
Содержит фильтры, коэффициенты и другие вспомогательные компоненты.
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Section Normalization System"
__description__ = "Базовые компоненты для системы нормирования участков"

# Основные классы
from .filter import (
    LocomotiveFilter,
    FilterConfig,
    BasicFilterStrategy,
    AdvancedFilterStrategy,
    create_locomotive_filter,
    get_preset_config,
    PRESET_CONFIGS
)

from .coefficients import (
    LocomotiveCoefficientsManager,
    CoefficientEntry,
    CoefficientStats
)

# Экспорт основных компонентов
__all__ = [
    # Фильтрация
    "LocomotiveFilter",
    "FilterConfig", 
    "BasicFilterStrategy",
    "AdvancedFilterStrategy",
    "create_locomotive_filter",
    "get_preset_config",
    "PRESET_CONFIGS",
    
    # Коэффициенты
    "LocomotiveCoefficientsManager",
    "CoefficientEntry",
    "CoefficientStats",
]

# Настройка логирования
import logging
import sys

_logger = logging.getLogger(__name__)

# Проверка версии Python
if sys.version_info < (3, 12):
    import warnings
    warnings.warn(
        f"Модуль core оптимизирован для Python 3.12+. "
        f"Текущая версия: {sys.version_info.major}.{sys.version_info.minor}",
        UserWarning,
        stacklevel=2
    )

def get_core_info() -> dict[str, any]:
    """Возвращает информацию о базовом модуле."""
    return {
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "components": {
            "locomotive_filter": True,
            "coefficients_manager": True,
            "advanced_strategies": True
        }
    }

def create_default_filter() -> LocomotiveFilter:
    """Создает фильтр локомотивов с настройками по умолчанию."""
    return create_locomotive_filter('basic')

def create_advanced_filter() -> LocomotiveFilter:
    """Создает продвинутый фильтр локомотивов."""
    return create_locomotive_filter('advanced')

def create_coefficients_manager(data_file: str = None) -> LocomotiveCoefficientsManager:
    """Создает менеджер коэффициентов."""
    return LocomotiveCoefficientsManager(data_file)

# Быстрые функции для создания предустановленных конфигураций
def create_electric_filter() -> LocomotiveFilter:
    """Создает фильтр для электровозов."""
    filter_obj = create_locomotive_filter('basic')
    config = get_preset_config('electric_locomotives')
    filter_obj.config = config
    return filter_obj

def create_freight_filter() -> LocomotiveFilter:
    """Создает фильтр для грузовых локомотивов."""
    filter_obj = create_locomotive_filter('basic')
    config = get_preset_config('freight_locomotives')
    filter_obj.config = config
    return filter_obj

def create_passenger_filter() -> LocomotiveFilter:
    """Создает фильтр для пассажирских локомотивов."""
    filter_obj = create_locomotive_filter('basic')
    config = get_preset_config('passenger_locomotives')
    filter_obj.config = config
    return filter_obj

# Добавляем вспомогательные функции в экспорт
__all__.extend([
    "get_core_info",
    "create_default_filter",
    "create_advanced_filter", 
    "create_coefficients_manager",
    "create_electric_filter",
    "create_freight_filter",
    "create_passenger_filter"
])

_logger.info(f"Модуль core v{__version__} загружен успешно")