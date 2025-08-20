#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль фильтрации локомотивов для системы нормирования участков.
Использует современные возможности Python 3.12 для оптимальной производительности.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Set, Optional, Protocol
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)

# Типы для Python 3.12
type SeriesPattern = str
type LocomotiveNumber = str | int
type FilterCriteria = dict[str, any]

@dataclass(slots=True, frozen=True)
class LocomotiveInfo:
    """Информация о локомотиве с оптимизацией памяти."""
    series: str
    number: str
    depot: Optional[str] = None
    
    def __post_init__(self):
        """Валидация после создания."""
        if not self.series or not self.number:
            raise ValueError("Серия и номер локомотива не могут быть пустыми")

@dataclass(slots=True)
class FilterConfig:
    """Конфигурация фильтрации с валидацией."""
    included_series: Set[str] = field(default_factory=set)
    excluded_series: Set[str] = field(default_factory=set)
    included_numbers: Set[str] = field(default_factory=set)
    excluded_numbers: Set[str] = field(default_factory=set)
    depot_filter: Optional[str] = None
    number_pattern: Optional[str] = None
    case_sensitive: bool = False
    
    def __post_init__(self):
        """Валидация конфигурации после создания."""
        # Проверяем конфликты в настройках
        if self.included_series and self.excluded_series:
            overlap = self.included_series & self.excluded_series
            if overlap:
                raise ValueError(f"Конфликт в сериях: {overlap} указаны и для включения, и для исключения")
        
        if self.included_numbers and self.excluded_numbers:
            overlap = self.included_numbers & self.excluded_numbers
            if overlap:
                raise ValueError(f"Конфликт в номерах: {overlap} указаны и для включения, и для исключения")
        
        # Компилируем regex паттерн если указан
        if self.number_pattern:
            try:
                flags = 0 if self.case_sensitive else re.IGNORECASE
                re.compile(self.number_pattern, flags)
            except re.error as e:
                raise ValueError(f"Неверный regex паттерн '{self.number_pattern}': {e}")

class FilterStrategy(Protocol):
    """Протокол для стратегий фильтрации."""
    
    def apply_filter(self, routes_df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        """Применяет фильтр к DataFrame маршрутов."""
        ...

class BasicFilterStrategy:
    """Базовая стратегия фильтрации локомотивов."""
    
    def apply_filter(self, routes_df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        """Применяет базовую фильтрацию по сериям и номерам."""
        if routes_df.empty:
            return routes_df
        
        filtered_df = routes_df.copy()
        
        # Фильтрация по сериям
        filtered_df = self._filter_by_series(filtered_df, config)
        
        # Фильтрация по номерам
        filtered_df = self._filter_by_numbers(filtered_df, config)
        
        # Фильтрация по депо
        if config.depot_filter:
            filtered_df = self._filter_by_depot(filtered_df, config)
        
        # Фильтрация по паттерну номера
        if config.number_pattern:
            filtered_df = self._filter_by_pattern(filtered_df, config)
        
        return filtered_df
    
    def _filter_by_series(self, df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        """Фильтрует по сериям локомотивов."""
        series_col = 'Серия локомотива'
        if series_col not in df.columns:
            return df
        
        # Применяем регистронезависимость если нужно
        if not config.case_sensitive:
            df_series = df[series_col].str.upper().fillna('')
            included = {s.upper() for s in config.included_series}
            excluded = {s.upper() for s in config.excluded_series}
        else:
            df_series = df[series_col].fillna('')
            included = config.included_series
            excluded = config.excluded_series
        
        # Включаем только указанные серии
        if included:
            mask = df_series.isin(included)
            df = df[mask]
        
        # Исключаем указанные серии
        if excluded:
            mask = ~df_series.isin(excluded)
            df = df[mask]
        
        return df
    
    def _filter_by_numbers(self, df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        """Фильтрует по номерам локомотивов."""
        number_col = 'Номер локомотива'
        if number_col not in df.columns:
            return df
        
        # Преобразуем номера в строки для единообразия
        df_numbers = df[number_col].astype(str).fillna('')
        
        if not config.case_sensitive:
            df_numbers = df_numbers.str.upper()
            included = {str(n).upper() for n in config.included_numbers}
            excluded = {str(n).upper() for n in config.excluded_numbers}
        else:
            included = {str(n) for n in config.included_numbers}
            excluded = {str(n) for n in config.excluded_numbers}
        
        # Включаем только указанные номера
        if included:
            mask = df_numbers.isin(included)
            df = df[mask]
        
        # Исключаем указанные номера
        if excluded:
            mask = ~df_numbers.isin(excluded)
            df = df[mask]
        
        return df
    
    def _filter_by_depot(self, df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        """Фильтрует по депо."""
        depot_col = 'Депо'
        if depot_col not in df.columns:
            return df
        
        depot_filter = config.depot_filter
        if not config.case_sensitive:
            mask = df[depot_col].str.contains(depot_filter, case=False, na=False)
        else:
            mask = df[depot_col].str.contains(depot_filter, case=True, na=False)
        
        return df[mask]
    
    def _filter_by_pattern(self, df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        """Фильтрует по regex паттерну номера."""
        number_col = 'Номер локомотива'
        if number_col not in df.columns:
            return df
        
        flags = 0 if config.case_sensitive else re.IGNORECASE
        pattern = re.compile(config.number_pattern, flags)
        
        df_numbers = df[number_col].astype(str).fillna('')
        mask = df_numbers.str.match(pattern, na=False)
        
        return df[mask]

class AdvancedFilterStrategy:
    """Продвинутая стратегия фильтрации с дополнительными возможностями."""
    
    def __init__(self):
        self.basic_strategy = BasicFilterStrategy()
    
    def apply_filter(self, routes_df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
        """Применяет продвинутую фильтрацию."""
        if routes_df.empty:
            return routes_df
        
        # Сначала применяем базовую фильтрацию
        filtered_df = self.basic_strategy.apply_filter(routes_df, config)
        
        # Дополнительные фильтры
        filtered_df = self._filter_by_locomotive_type(filtered_df)
        filtered_df = self._filter_by_data_quality(filtered_df)
        
        return filtered_df
    
    def _filter_by_locomotive_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрует по типу локомотива (электровозы, тепловозы и т.д.)."""
        series_col = 'Серия локомотива'
        if series_col not in df.columns:
            return df
        
        # Известные серии электровозов
        electric_series = {
            'ЭП1М', 'ЭП20', 'ЭС4К', 'ЭС5К', '2ЭС4К', '2ЭС5К',
            'ВЛ10', 'ВЛ11', 'ВЛ80', 'ВЛ82', 'ВЛ85', 'ЭП1', 'ЭП2К'
        }
        
        # Фильтруем только электровозы (можно расширить логику)
        series_values = df[series_col].fillna('')
        mask = series_values.isin(electric_series)
        
        # Если найдены серии электровозов, фильтруем по ним
        if mask.any():
            return df[mask]
        
        # Иначе возвращаем все данные
        return df
    
    def _filter_by_data_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрует записи по качеству данных."""
        # Исключаем записи с критически важными пропущенными данными
        required_columns = ['ТКМ брутто', 'КМ', 'Расход фактический']
        
        for col in required_columns:
            if col in df.columns:
                # Исключаем записи с нулевыми или отрицательными значениями
                mask = (df[col].notna()) & (df[col] > 0)
                df = df[mask]
        
        return df

class LocomotiveFilter:
    """
    Основной класс фильтрации локомотивов.
    
    Поддерживает различные стратегии фильтрации и гибкую конфигурацию.
    """
    
    def __init__(self, strategy: Optional[FilterStrategy] = None):
        """Инициализирует фильтр с выбранной стратегией."""
        self.strategy = strategy or BasicFilterStrategy()
        self.config = FilterConfig()
        self._filter_stats = {
            'total_processed': 0,
            'total_filtered': 0,
            'last_filter_ratio': 0.0
        }
        
        logger.info("Инициализирован фильтр локомотивов")
    
    def set_strategy(self, strategy: FilterStrategy) -> None:
        """Устанавливает стратегию фильтрации."""
        self.strategy = strategy
        logger.info(f"Установлена стратегия фильтрации: {strategy.__class__.__name__}")
    
    def configure(self, **kwargs) -> None:
        """
        Настраивает параметры фильтрации.
        
        Args:
            included_series: Список серий для включения
            excluded_series: Список серий для исключения
            included_numbers: Список номеров для включения
            excluded_numbers: Список номеров для исключения
            depot_filter: Фильтр по депо
            number_pattern: Regex паттерн для номеров
            case_sensitive: Учет регистра
        """
        # Преобразуем списки в множества для оптимизации
        if 'included_series' in kwargs:
            self.config.included_series = set(kwargs['included_series'])
        
        if 'excluded_series' in kwargs:
            self.config.excluded_series = set(kwargs['excluded_series'])
        
        if 'included_numbers' in kwargs:
            self.config.included_numbers = set(str(n) for n in kwargs['included_numbers'])
        
        if 'excluded_numbers' in kwargs:
            self.config.excluded_numbers = set(str(n) for n in kwargs['excluded_numbers'])
        
        for key in ['depot_filter', 'number_pattern', 'case_sensitive']:
            if key in kwargs:
                setattr(self.config, key, kwargs[key])
        
        logger.info("Конфигурация фильтра обновлена")
    
    def filter_routes(self, routes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет фильтрацию к DataFrame маршрутов.
        
        Args:
            routes_df: DataFrame с данными маршрутов
            
        Returns:
            Отфильтрованный DataFrame
        """
        if routes_df.empty:
            logger.warning("Передан пустой DataFrame для фильтрации")
            return routes_df
        
        original_count = len(routes_df)
        logger.debug(f"Начало фильтрации: {original_count} записей")
        
        try:
            # Применяем стратегию фильтрации
            filtered_df = self.strategy.apply_filter(routes_df, self.config)
            
            # Обновляем статистику
            filtered_count = len(filtered_df)
            self._update_filter_stats(original_count, filtered_count)
            
            filter_ratio = (original_count - filtered_count) / original_count * 100
            logger.info(f"Фильтрация завершена: {filtered_count}/{original_count} записей "
                       f"осталось ({filter_ratio:.1f}% отфильтровано)")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Ошибка фильтрации: {e}")
            return routes_df  # Возвращаем исходные данные в случае ошибки
    
    def _update_filter_stats(self, original_count: int, filtered_count: int) -> None:
        """Обновляет статистику фильтрации."""
        self._filter_stats['total_processed'] += original_count
        self._filter_stats['total_filtered'] += (original_count - filtered_count)
        
        if original_count > 0:
            self._filter_stats['last_filter_ratio'] = (original_count - filtered_count) / original_count
    
    def get_filter_stats(self) -> dict[str, any]:
        """Возвращает статистику фильтрации."""
        return self._filter_stats.copy()
    
    def validate_locomotive_data(self, routes_df: pd.DataFrame) -> dict[str, any]:
        """
        Валидирует данные локомотивов в DataFrame.
        
        Returns:
            Словарь с результатами валидации
        """
        validation_results = {
            'total_routes': len(routes_df),
            'valid_locomotive_data': 0,
            'missing_series': 0,
            'missing_numbers': 0,
            'invalid_series': 0,
            'invalid_numbers': 0,
            'series_distribution': {},
            'recommendations': []
        }
        
        if routes_df.empty:
            return validation_results
        
        # Проверяем наличие колонок
        series_col = 'Серия локомотива'
        number_col = 'Номер локомотива'
        
        if series_col not in routes_df.columns:
            validation_results['recommendations'].append(f"Отсутствует колонка '{series_col}'")
            return validation_results
        
        if number_col not in routes_df.columns:
            validation_results['recommendations'].append(f"Отсутствует колонка '{number_col}'")
            return validation_results
        
        # Анализируем данные
        for _, row in routes_df.iterrows():
            series = row.get(series_col)
            number = row.get(number_col)
            
            # Проверяем серию
            if pd.isna(series) or series == '':
                validation_results['missing_series'] += 1
            elif not isinstance(series, str) or len(series.strip()) == 0:
                validation_results['invalid_series'] += 1
            else:
                # Обновляем распределение серий
                series_clean = series.strip()
                validation_results['series_distribution'][series_clean] = \
                    validation_results['series_distribution'].get(series_clean, 0) + 1
            
            # Проверяем номер
            if pd.isna(number) or number == '':
                validation_results['missing_numbers'] += 1
            elif not str(number).strip():
                validation_results['invalid_numbers'] += 1
            
            # Считаем валидные записи
            if (not pd.isna(series) and series != '' and 
                not pd.isna(number) and number != ''):
                validation_results['valid_locomotive_data'] += 1
        
        # Генерируем рекомендации
        if validation_results['missing_series'] > 0:
            validation_results['recommendations'].append(
                f"Найдено {validation_results['missing_series']} записей с пропущенной серией локомотива"
            )
        
        if validation_results['missing_numbers'] > 0:
            validation_results['recommendations'].append(
                f"Найдено {validation_results['missing_numbers']} записей с пропущенным номером локомотива"
            )
        
        # Анализируем распределение серий
        total_series = len(validation_results['series_distribution'])
        if total_series > 20:
            validation_results['recommendations'].append(
                f"Обнаружено {total_series} различных серий локомотивов. "
                "Рассмотрите возможность фильтрации по основным сериям."
            )
        
        return validation_results
    
    def suggest_filter_config(self, routes_df: pd.DataFrame, top_series_count: int = 10) -> FilterConfig:
        """
        Предлагает конфигурацию фильтра на основе анализа данных.
        
        Args:
            routes_df: DataFrame для анализа
            top_series_count: Количество топ серий для включения
            
        Returns:
            Рекомендуемая конфигурация фильтра
        """
        validation = self.validate_locomotive_data(routes_df)
        
        # Находим топ серии по частоте использования
        series_dist = validation['series_distribution']
        top_series = sorted(series_dist.items(), key=lambda x: x[1], reverse=True)[:top_series_count]
        
        suggested_config = FilterConfig(
            included_series={series for series, _ in top_series},
            case_sensitive=False
        )
        
        logger.info(f"Предложена конфигурация фильтра с топ {len(top_series)} сериями")
        
        return suggested_config
    
    def clear_config(self) -> None:
        """Очищает конфигурацию фильтра."""
        self.config = FilterConfig()
        logger.info("Конфигурация фильтра очищена")

# Фабрика для создания фильтров
def create_locomotive_filter(filter_type: str = 'basic') -> LocomotiveFilter:
    """
    Создает фильтр локомотивов указанного типа.
    
    Args:
        filter_type: Тип фильтра ('basic' или 'advanced')
        
    Returns:
        Настроенный фильтр локомотивов
    """
    if filter_type == 'advanced':
        strategy = AdvancedFilterStrategy()
    else:
        strategy = BasicFilterStrategy()
    
    return LocomotiveFilter(strategy)

# Предустановленные конфигурации
PRESET_CONFIGS = {
    'electric_locomotives': FilterConfig(
        included_series={'ЭП1М', 'ЭП20', 'ЭС4К', 'ЭС5К', '2ЭС4К', '2ЭС5К'},
        case_sensitive=False
    ),
    'freight_locomotives': FilterConfig(
        included_series={'ВЛ10', 'ВЛ11', 'ВЛ80', 'ВЛ82', 'ВЛ85'},
        case_sensitive=False
    ),
    'passenger_locomotives': FilterConfig(
        included_series={'ЭП1', 'ЭП2К', 'ЭП20'},
        case_sensitive=False
    )
}

def get_preset_config(preset_name: str) -> Optional[FilterConfig]:
    """Возвращает предустановленную конфигурацию фильтра."""
    return PRESET_CONFIGS.get(preset_name)