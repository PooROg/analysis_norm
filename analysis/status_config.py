#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленная конфигурация статусов и цветовых схем анализатора.
Оптимизирована для Python 3.12 с использованием dataclasses и slots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Final, Dict, List, Tuple
from enum import Enum, auto

# Типы для Python 3.12
type ColorHex = str
type StatusName = str
type ThresholdValue = float

class StatusCategory(Enum):
    """Категории статусов для группировки"""
    ECONOMY = auto()
    NORMAL = auto() 
    OVERRUN = auto()

@dataclass(slots=True, frozen=True)
class StatusThresholds:
    """Пороговые значения для определения статусов"""
    STRONG_ECONOMY: Final[ThresholdValue] = -30.0
    MEDIUM_ECONOMY: Final[ThresholdValue] = -20.0 
    WEAK_ECONOMY: Final[ThresholdValue] = -5.0
    NORMAL_POSITIVE: Final[ThresholdValue] = 5.0
    WEAK_OVERRUN: Final[ThresholdValue] = 20.0
    MEDIUM_OVERRUN: Final[ThresholdValue] = 30.0

@dataclass(slots=True, frozen=True)
class ColorScheme:
    """Цветовая схема для графиков"""
    # Цвета экономии (зеленые оттенки)
    ECONOMY_STRONG: Final[ColorHex] = '#006400'    # Темно-зеленый
    ECONOMY_MEDIUM: Final[ColorHex] = '#228B22'    # Зеленый  
    ECONOMY_WEAK: Final[ColorHex] = '#32CD32'      # Светло-зеленый
    
    # Цвет нормы
    NORMAL: Final[ColorHex] = '#FFD700'            # Золотой
    
    # Цвета перерасхода (красные оттенки)
    OVERRUN_WEAK: Final[ColorHex] = '#FF8C00'      # Оранжевый
    OVERRUN_MEDIUM: Final[ColorHex] = '#FF4500'    # Красно-оранжевый
    OVERRUN_STRONG: Final[ColorHex] = '#DC143C'    # Малиновый
    
    # Цвета границ на графиках
    BOUNDARY_NORMAL: Final[ColorHex] = '#FFD700'   # Границы нормы
    BOUNDARY_WARNING: Final[ColorHex] = '#FF4500'  # Границы предупреждения
    BOUNDARY_CRITICAL: Final[ColorHex] = '#DC143C' # Критические границы
    BOUNDARY_ZERO: Final[ColorHex] = 'black'       # Нулевая линия

@dataclass(slots=True, frozen=True)
class StatusConfig:
    """Главная конфигурация статусов анализа"""
    
    # Статические конфигурации
    thresholds: StatusThresholds = field(default_factory=StatusThresholds)
    colors: ColorScheme = field(default_factory=ColorScheme)
    
    # Динамические маппинги
    _status_ranges: Dict[StatusName, Tuple[ThresholdValue, ThresholdValue]] = field(init=False)
    _status_colors: Dict[StatusName, ColorHex] = field(init=False)
    _status_categories: Dict[StatusName, StatusCategory] = field(init=False)
    
    def __post_init__(self):
        """Инициализация динамических маппингов после создания объекта"""
        # Создаем маппинги статусов и их диапазонов
        status_ranges = {
            'Экономия сильная': (-float('inf'), self.thresholds.STRONG_ECONOMY),
            'Экономия средняя': (self.thresholds.STRONG_ECONOMY, self.thresholds.MEDIUM_ECONOMY),
            'Экономия слабая': (self.thresholds.MEDIUM_ECONOMY, self.thresholds.WEAK_ECONOMY),
            'Норма': (self.thresholds.WEAK_ECONOMY, self.thresholds.NORMAL_POSITIVE),
            'Перерасход слабый': (self.thresholds.NORMAL_POSITIVE, self.thresholds.WEAK_OVERRUN),
            'Перерасход средний': (self.thresholds.WEAK_OVERRUN, self.thresholds.MEDIUM_OVERRUN),
            'Перерасход сильный': (self.thresholds.MEDIUM_OVERRUN, float('inf'))
        }
        
        # Создаем маппинги статусов и цветов
        status_colors = {
            'Экономия сильная': self.colors.ECONOMY_STRONG,
            'Экономия средняя': self.colors.ECONOMY_MEDIUM,
            'Экономия слабая': self.colors.ECONOMY_WEAK,
            'Норма': self.colors.NORMAL,
            'Перерасход слабый': self.colors.OVERRUN_WEAK,
            'Перерасход средний': self.colors.OVERRUN_MEDIUM,
            'Перерасход сильный': self.colors.OVERRUN_STRONG
        }
        
        # Создаем маппинги статусов и категорий
        status_categories = {
            'Экономия сильная': StatusCategory.ECONOMY,
            'Экономия средняя': StatusCategory.ECONOMY,
            'Экономия слабая': StatusCategory.ECONOMY,
            'Норма': StatusCategory.NORMAL,
            'Перерасход слабый': StatusCategory.OVERRUN,
            'Перерасход средний': StatusCategory.OVERRUN,
            'Перерасход сильный': StatusCategory.OVERRUN
        }
        
        # Используем object.__setattr__ так как класс frozen
        object.__setattr__(self, '_status_ranges', status_ranges)
        object.__setattr__(self, '_status_colors', status_colors)
        object.__setattr__(self, '_status_categories', status_categories)
    
    @property
    def status_ranges(self) -> Dict[StatusName, Tuple[ThresholdValue, ThresholdValue]]:
        """Диапазоны отклонений для каждого статуса"""
        return self._status_ranges
    
    @property 
    def status_colors(self) -> Dict[StatusName, ColorHex]:
        """Цвета для каждого статуса"""
        return self._status_colors
    
    @property
    def status_categories(self) -> Dict[StatusName, StatusCategory]:
        """Категории для каждого статуса"""
        return self._status_categories
    
    def get_status_by_deviation(self, deviation: ThresholdValue) -> StatusName:
        """Определяет статус по значению отклонения"""
        for status, (min_val, max_val) in self.status_ranges.items():
            if min_val < deviation <= max_val:
                return status
        return 'Не определен'
    
    def get_color_by_status(self, status: StatusName) -> ColorHex:
        """Возвращает цвет по статусу"""
        return self.status_colors.get(status, '#808080')  # Серый для неопределенных
    
    def get_category_by_status(self, status: StatusName) -> StatusCategory | None:
        """Возвращает категорию по статусу"""
        return self.status_categories.get(status)
    
    def get_statuses_by_category(self, category: StatusCategory) -> List[StatusName]:
        """Возвращает список статусов для категории"""
        return [status for status, cat in self.status_categories.items() if cat == category]
    
    def get_boundary_lines_config(self) -> List[Tuple[ThresholdValue, ColorHex, str]]:
        """Возвращает конфигурацию граничных линий для графиков"""
        return [
            (self.thresholds.NORMAL_POSITIVE, self.colors.BOUNDARY_NORMAL, 'dash'),
            (self.thresholds.WEAK_ECONOMY, self.colors.BOUNDARY_NORMAL, 'dash'),
            (self.thresholds.WEAK_OVERRUN, self.colors.BOUNDARY_WARNING, 'dot'),
            (self.thresholds.MEDIUM_ECONOMY, self.colors.BOUNDARY_WARNING, 'dot'),
            (self.thresholds.MEDIUM_OVERRUN, self.colors.BOUNDARY_CRITICAL, 'dashdot'),
            (self.thresholds.STRONG_ECONOMY, self.colors.BOUNDARY_CRITICAL, 'dashdot'),
            (0, self.colors.BOUNDARY_ZERO, 'solid')
        ]
    
    def get_normal_zone_fill(self) -> Tuple[List[ThresholdValue], str]:
        """Возвращает конфигурацию заливки зоны нормы"""
        y_values = [self.thresholds.WEAK_ECONOMY, self.thresholds.WEAK_ECONOMY, 
                   self.thresholds.NORMAL_POSITIVE, self.thresholds.NORMAL_POSITIVE]
        fill_color = 'rgba(255, 215, 0, 0.1)'  # Прозрачный золотой
        return y_values, fill_color

@dataclass(slots=True, frozen=True)
class PlotConfig:
    """Конфигурация для настройки графиков"""
    
    # Размеры и пропорции
    PLOT_HEIGHT: Final[int] = 1000
    MAIN_PLOT_RATIO: Final[float] = 0.6
    DEVIATION_PLOT_RATIO: Final[float] = 0.4
    VERTICAL_SPACING: Final[float] = 0.05
    
    # Стили маркеров
    MARKER_SIZE: Final[int] = 8
    MARKER_OPACITY: Final[float] = 0.8
    MARKER_LINE_WIDTH: Final[float] = 0.5
    MARKER_LINE_COLOR: Final[str] = 'black'
    
    # Стили линий
    NORM_LINE_WIDTH: Final[int] = 2
    BOUNDARY_LINE_WIDTH: Final[int] = 2
    
    # Настройки hover
    HOVER_MODE: Final[str] = 'closest'
    
    # Настройки легенды
    LEGEND_ORIENTATION: Final[str] = 'v'
    LEGEND_Y_ANCHOR: Final[str] = 'middle'
    LEGEND_Y: Final[float] = 0.5
    LEGEND_X_ANCHOR: Final[str] = 'left'
    LEGEND_X: Final[float] = 1.02
    
    # Шаблон для графиков
    TEMPLATE: Final[str] = 'plotly_white'
    
    # Интерполяция норм
    INTERPOLATION_POINTS: Final[int] = 100

# Создаем глобальные экземпляры для использования в других модулях
DEFAULT_STATUS_CONFIG: Final[StatusConfig] = StatusConfig()
DEFAULT_PLOT_CONFIG: Final[PlotConfig] = PlotConfig()

# Утилитарные функции для работы с конфигурацией
def create_custom_status_config(
    economy_thresholds: Tuple[ThresholdValue, ThresholdValue, ThresholdValue] = (-30, -20, -5),
    normal_threshold: ThresholdValue = 5,
    overrun_thresholds: Tuple[ThresholdValue, ThresholdValue] = (20, 30)
) -> StatusConfig:
    """Создает кастомную конфигурацию статусов"""
    
    @dataclass(slots=True, frozen=True) 
    class CustomThresholds:
        STRONG_ECONOMY: ThresholdValue = economy_thresholds[0]
        MEDIUM_ECONOMY: ThresholdValue = economy_thresholds[1]
        WEAK_ECONOMY: ThresholdValue = economy_thresholds[2]
        NORMAL_POSITIVE: ThresholdValue = normal_threshold
        WEAK_OVERRUN: ThresholdValue = overrun_thresholds[0]
        MEDIUM_OVERRUN: ThresholdValue = overrun_thresholds[1]
    
    return StatusConfig(thresholds=CustomThresholds())

def get_status_summary(config: StatusConfig = DEFAULT_STATUS_CONFIG) -> Dict[str, List[StatusName]]:
    """Возвращает сводку по статусам и их категориям"""
    summary = {}
    for category in StatusCategory:
        summary[category.name] = config.get_statuses_by_category(category)
    return summary

def validate_status_config(config: StatusConfig) -> Tuple[bool, List[str]]:
    """Валидирует конфигурацию статусов"""
    errors = []
    
    # Проверяем логическую последовательность порогов
    thresholds = [
        config.thresholds.STRONG_ECONOMY,
        config.thresholds.MEDIUM_ECONOMY,
        config.thresholds.WEAK_ECONOMY,
        0,  # Нулевая линия
        config.thresholds.NORMAL_POSITIVE,
        config.thresholds.WEAK_OVERRUN,
        config.thresholds.MEDIUM_OVERRUN
    ]
    
    for i in range(len(thresholds) - 1):
        if thresholds[i] >= thresholds[i + 1]:
            errors.append(f"Нарушен порядок порогов: {thresholds[i]} >= {thresholds[i + 1]}")
    
    # Проверяем наличие всех цветов
    required_colors = ['ECONOMY_STRONG', 'ECONOMY_MEDIUM', 'ECONOMY_WEAK', 'NORMAL', 
                      'OVERRUN_WEAK', 'OVERRUN_MEDIUM', 'OVERRUN_STRONG']
    
    for color_name in required_colors:
        if not hasattr(config.colors, color_name):
            errors.append(f"Отсутствует цвет: {color_name}")
    
    return len(errors) == 0, errors

# Пример использования модуля для тестирования
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Тестирование конфигурации
    config = DEFAULT_STATUS_CONFIG
    
    # Тест определения статуса
    test_deviations = [-35, -25, -10, 0, 10, 25, 35]
    print("=== ТЕСТ ОПРЕДЕЛЕНИЯ СТАТУСОВ ===")
    for deviation in test_deviations:
        status = config.get_status_by_deviation(deviation)
        color = config.get_color_by_status(status)
        category = config.get_category_by_status(status)
        print(f"Отклонение {deviation:+3}%: {status:20} ({category.name if category else 'Unknown':7}) - {color}")
    
    # Тест граничных линий
    print("\n=== ГРАНИЧНЫЕ ЛИНИИ ===")
    for y_val, color, style in config.get_boundary_lines_config():
        print(f"Y={y_val:+5.0f}: {color} ({style})")
    
    # Тест зоны нормы
    y_vals, fill_color = config.get_normal_zone_fill()
    print(f"\n=== ЗОНА НОРМЫ ===")
    print(f"Y={y_vals}, Цвет={fill_color}")
    
    # Тест валидации
    is_valid, errors = validate_status_config(config)
    print(f"\n=== ВАЛИДАЦИЯ КОНФИГУРАЦИИ ===")
    print(f"Конфигурация валидна: {is_valid}")
    if errors:
        for error in errors:
            print(f"Ошибка: {error}")
    
    # Тест сводки
    summary = get_status_summary(config)
    print(f"\n=== СВОДКА ПО КАТЕГОРИЯМ ===")
    for category, statuses in summary.items():
        print(f"{category}: {', '.join(statuses)}")