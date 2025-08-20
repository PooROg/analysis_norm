#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Менеджер коэффициентов локомотивов для корректировки расхода электроэнергии.
Использует современные возможности Python 3.12 для оптимальной производительности.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path
from functools import lru_cache
import pandas as pd

logger = logging.getLogger(__name__)

# Типы для Python 3.12
type CoefficientValue = float
type LocomotiveSeries = str
type LocomotiveNumber = str | int
type CoefficientKey = tuple[LocomotiveSeries, LocomotiveNumber]

@dataclass(slots=True, frozen=True)
class CoefficientEntry:
    """Запись коэффициента для конкретного локомотива."""
    series: LocomotiveSeries
    number: LocomotiveNumber
    coefficient: CoefficientValue
    reason: str = ""
    date_created: str = ""
    date_modified: str = ""
    is_active: bool = True
    
    def __post_init__(self):
        """Валидация после создания."""
        if self.coefficient <= 0:
            raise ValueError(f"Коэффициент должен быть положительным, получен: {self.coefficient}")
        
        if not self.series or not str(self.number):
            raise ValueError("Серия и номер локомотива не могут быть пустыми")

@dataclass(slots=True)
class CoefficientStats:
    """Статистика применения коэффициентов."""
    total_coefficients: int = 0
    active_coefficients: int = 0
    series_count: int = 0
    avg_coefficient: float = 1.0
    min_coefficient: float = 1.0
    max_coefficient: float = 1.0
    applied_count: int = 0
    
    def update_from_data(self, coefficients: Dict[CoefficientKey, CoefficientEntry]) -> None:
        """Обновляет статистику на основе данных коэффициентов."""
        if not coefficients:
            return
        
        active_coeffs = [entry for entry in coefficients.values() if entry.is_active]
        
        self.total_coefficients = len(coefficients)
        self.active_coefficients = len(active_coeffs)
        self.series_count = len(set(entry.series for entry in coefficients.values()))
        
        if active_coeffs:
            coeff_values = [entry.coefficient for entry in active_coeffs]
            self.avg_coefficient = sum(coeff_values) / len(coeff_values)
            self.min_coefficient = min(coeff_values)
            self.max_coefficient = max(coeff_values)

class LocomotiveCoefficientsManager:
    """
    Менеджер коэффициентов локомотивов для корректировки расхода.
    
    Основные функции:
    - Хранение и управление коэффициентами
    - Применение коэффициентов к данным
    - Импорт/экспорт коэффициентов
    - Валидация и статистика
    """
    
    def __init__(self, data_file: Optional[str | Path] = None):
        """Инициализирует менеджер коэффициентов."""
        self._coefficients: Dict[CoefficientKey, CoefficientEntry] = {}
        self._data_file = Path(data_file) if data_file else None
        self._stats = CoefficientStats()
        self._default_coefficient = 1.0
        
        # Загружаем данные если файл указан
        if self._data_file and self._data_file.exists():
            self.load_from_file(self._data_file)
        
        logger.info("Инициализирован менеджер коэффициентов локомотивов")
    
    def add_coefficient(self, series: LocomotiveSeries, number: LocomotiveNumber, 
                       coefficient: CoefficientValue, reason: str = "", 
                       overwrite: bool = False) -> bool:
        """
        Добавляет коэффициент для локомотива.
        
        Args:
            series: Серия локомотива
            number: Номер локомотива
            coefficient: Значение коэффициента
            reason: Причина применения коэффициента
            overwrite: Перезаписать существующий коэффициент
            
        Returns:
            True если коэффициент добавлен успешно
        """
        try:
            key = (str(series).strip(), str(number).strip())
            
            if key in self._coefficients and not overwrite:
                logger.warning(f"Коэффициент для {series} №{number} уже существует")
                return False
            
            from datetime import datetime
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            entry = CoefficientEntry(
                series=key[0],
                number=key[1],
                coefficient=coefficient,
                reason=reason,
                date_created=current_time if key not in self._coefficients else self._coefficients[key].date_created,
                date_modified=current_time,
                is_active=True
            )
            
            self._coefficients[key] = entry
            self._update_stats()
            
            logger.info(f"Добавлен коэффициент {coefficient:.3f} для {series} №{number}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка добавления коэффициента: {e}")
            return False
    
    def get_coefficient(self, series: LocomotiveSeries, number: LocomotiveNumber) -> float:
        """
        Получает коэффициент для локомотива.
        
        Args:
            series: Серия локомотива
            number: Номер локомотива
            
        Returns:
            Коэффициент (1.0 если не найден)
        """
        key = (str(series).strip(), str(number).strip())
        
        if key in self._coefficients:
            entry = self._coefficients[key]
            if entry.is_active:
                return entry.coefficient
        
        return self._default_coefficient
    
    @lru_cache(maxsize=1000)
    def _cached_get_coefficient(self, series: str, number: str) -> float:
        """Кэшированная версия получения коэффициента для оптимизации."""
        return self.get_coefficient(series, number)
    
    def remove_coefficient(self, series: LocomotiveSeries, number: LocomotiveNumber) -> bool:
        """
        Удаляет коэффициент для локомотива.
        
        Args:
            series: Серия локомотива
            number: Номер локомотива
            
        Returns:
            True если коэффициент удален
        """
        key = (str(series).strip(), str(number).strip())
        
        if key in self._coefficients:
            del self._coefficients[key]
            self._update_stats()
            self._clear_cache()
            
            logger.info(f"Удален коэффициент для {series} №{number}")
            return True
        
        return False
    
    def deactivate_coefficient(self, series: LocomotiveSeries, number: LocomotiveNumber) -> bool:
        """
        Деактивирует коэффициент без удаления.
        
        Args:
            series: Серия локомотива
            number: Номер локомотива
            
        Returns:
            True если коэффициент деактивирован
        """
        key = (str(series).strip(), str(number).strip())
        
        if key in self._coefficients:
            # Создаем новую запись с обновленным статусом
            old_entry = self._coefficients[key]
            new_entry = CoefficientEntry(
                series=old_entry.series,
                number=old_entry.number,
                coefficient=old_entry.coefficient,
                reason=old_entry.reason,
                date_created=old_entry.date_created,
                date_modified=old_entry.date_modified,
                is_active=False
            )
            
            self._coefficients[key] = new_entry
            self._update_stats()
            self._clear_cache()
            
            logger.info(f"Деактивирован коэффициент для {series} №{number}")
            return True
        
        return False
    
    def apply_coefficients_to_dataframe(self, df: pd.DataFrame, 
                                       series_col: str = 'Серия локомотива',
                                       number_col: str = 'Номер локомотива',
                                       consumption_col: str = 'Факт уд') -> pd.DataFrame:
        """
        Применяет коэффициенты к DataFrame с данными маршрутов.
        
        Args:
            df: DataFrame с данными маршрутов
            series_col: Название колонки с серией локомотива
            number_col: Название колонки с номером локомотива
            consumption_col: Название колонки с расходом
            
        Returns:
            DataFrame с примененными коэффициентами
        """
        if df.empty:
            return df
        
        # Проверяем наличие необходимых колонок
        required_cols = [series_col, number_col, consumption_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Отсутствуют колонки: {missing_cols}")
            return df
        
        result_df = df.copy()
        applied_count = 0
        
        # Добавляем колонки для результатов
        if 'Коэффициент' not in result_df.columns:
            result_df['Коэффициент'] = 1.0
        
        if f'{consumption_col} исходный' not in result_df.columns:
            result_df[f'{consumption_col} исходный'] = result_df[consumption_col]
        
        # Применяем коэффициенты
        for idx, row in result_df.iterrows():
            series = row.get(series_col)
            number = row.get(number_col)
            original_consumption = row.get(consumption_col)
            
            if pd.notna(series) and pd.notna(number) and pd.notna(original_consumption):
                coefficient = self.get_coefficient(series, number)
                
                if coefficient != 1.0:
                    result_df.loc[idx, 'Коэффициент'] = coefficient
                    result_df.loc[idx, consumption_col] = original_consumption * coefficient
                    applied_count += 1
        
        # Обновляем статистику
        self._stats.applied_count += applied_count
        
        logger.info(f"Применены коэффициенты к {applied_count} записям из {len(df)}")
        return result_df
    
    def load_from_file(self, file_path: str | Path) -> bool:
        """
        Загружает коэффициенты из JSON файла.
        
        Args:
            file_path: Путь к файлу с коэффициентами
            
        Returns:
            True если загрузка успешна
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                logger.warning(f"Файл коэффициентов не найден: {file_path}")
                return False
            
            with file_path.open('r', encoding='utf-8') as f:
                data = json.load(f)
            
            self._coefficients.clear()
            loaded_count = 0
            
            for entry_data in data.get('coefficients', []):
                try:
                    entry = CoefficientEntry(
                        series=entry_data['series'],
                        number=entry_data['number'],
                        coefficient=entry_data['coefficient'],
                        reason=entry_data.get('reason', ''),
                        date_created=entry_data.get('date_created', ''),
                        date_modified=entry_data.get('date_modified', ''),
                        is_active=entry_data.get('is_active', True)
                    )
                    
                    key = (entry.series, entry.number)
                    self._coefficients[key] = entry
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"Ошибка загрузки записи коэффициента: {e}")
                    continue
            
            self._update_stats()
            self._clear_cache()
            
            logger.info(f"Загружено {loaded_count} коэффициентов из {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки коэффициентов: {e}")
            return False
    
    def save_to_file(self, file_path: Optional[str | Path] = None) -> bool:
        """
        Сохраняет коэффициенты в JSON файл.
        
        Args:
            file_path: Путь к файлу (если не указан, используется файл из конструктора)
            
        Returns:
            True если сохранение успешно
        """
        try:
            save_path = Path(file_path) if file_path else self._data_file
            
            if not save_path:
                logger.error("Не указан путь для сохранения коэффициентов")
                return False
            
            # Создаем директорию если не существует
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Подготавливаем данные для сохранения
            data = {
                'version': '1.0',
                'total_coefficients': len(self._coefficients),
                'coefficients': []
            }
            
            for entry in self._coefficients.values():
                entry_data = {
                    'series': entry.series,
                    'number': entry.number,
                    'coefficient': entry.coefficient,
                    'reason': entry.reason,
                    'date_created': entry.date_created,
                    'date_modified': entry.date_modified,
                    'is_active': entry.is_active
                }
                data['coefficients'].append(entry_data)
            
            # Сохраняем в файл
            with save_path.open('w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Сохранено {len(self._coefficients)} коэффициентов в {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка сохранения коэффициентов: {e}")
            return False
    
    def import_from_excel(self, excel_path: str | Path, 
                         series_col: str = 'Серия',
                         number_col: str = 'Номер',
                         coefficient_col: str = 'Коэффициент',
                         reason_col: str = 'Причина') -> bool:
        """
        Импортирует коэффициенты из Excel файла.
        
        Args:
            excel_path: Путь к Excel файлу
            series_col: Название колонки с серией
            number_col: Название колонки с номером
            coefficient_col: Название колонки с коэффициентом
            reason_col: Название колонки с причиной
            
        Returns:
            True если импорт успешен
        """
        try:
            df = pd.read_excel(excel_path)
            
            required_cols = [series_col, number_col, coefficient_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"Отсутствуют колонки в Excel: {missing_cols}")
                return False
            
            imported_count = 0
            
            for _, row in df.iterrows():
                series = row[series_col]
                number = row[number_col]
                coefficient = row[coefficient_col]
                reason = row.get(reason_col, '') if reason_col in df.columns else ''
                
                if pd.notna(series) and pd.notna(number) and pd.notna(coefficient):
                    if self.add_coefficient(series, number, coefficient, reason, overwrite=True):
                        imported_count += 1
            
            logger.info(f"Импортировано {imported_count} коэффициентов из Excel")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка импорта из Excel: {e}")
            return False
    
    def export_to_excel(self, excel_path: str | Path) -> bool:
        """
        Экспортирует коэффициенты в Excel файл.
        
        Args:
            excel_path: Путь к Excel файлу
            
        Returns:
            True если экспорт успешен
        """
        try:
            data = []
            
            for entry in self._coefficients.values():
                data.append({
                    'Серия': entry.series,
                    'Номер': entry.number,
                    'Коэффициент': entry.coefficient,
                    'Причина': entry.reason,
                    'Дата создания': entry.date_created,
                    'Дата изменения': entry.date_modified,
                    'Активен': entry.is_active
                })
            
            df = pd.DataFrame(data)
            df.to_excel(excel_path, index=False)
            
            logger.info(f"Экспортировано {len(data)} коэффициентов в Excel")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта в Excel: {e}")
            return False
    
    def get_coefficients_by_series(self, series: LocomotiveSeries) -> List[CoefficientEntry]:
        """Возвращает все коэффициенты для указанной серии."""
        return [entry for entry in self._coefficients.values() if entry.series == series]
    
    def get_active_coefficients(self) -> Dict[CoefficientKey, CoefficientEntry]:
        """Возвращает только активные коэффициенты."""
        return {key: entry for key, entry in self._coefficients.items() if entry.is_active}
    
    def get_statistics(self) -> CoefficientStats:
        """Возвращает статистику коэффициентов."""
        return self._stats
    
    def validate_coefficients(self) -> Dict[str, any]:
        """
        Валидирует коэффициенты и возвращает отчет.
        
        Returns:
            Словарь с результатами валидации
        """
        validation_report = {
            'total_coefficients': len(self._coefficients),
            'valid_coefficients': 0,
            'invalid_coefficients': 0,
            'warnings': [],
            'errors': []
        }
        
        for key, entry in self._coefficients.items():
            try:
                # Проверяем базовые требования
                if entry.coefficient <= 0:
                    validation_report['errors'].append(
                        f"Неположительный коэффициент для {entry.series} №{entry.number}: {entry.coefficient}"
                    )
                    validation_report['invalid_coefficients'] += 1
                    continue
                
                # Проверяем разумные пределы
                if entry.coefficient < 0.5 or entry.coefficient > 2.0:
                    validation_report['warnings'].append(
                        f"Необычный коэффициент для {entry.series} №{entry.number}: {entry.coefficient}"
                    )
                
                if not entry.series or not str(entry.number):
                    validation_report['errors'].append(
                        f"Пустая серия или номер: {entry.series} №{entry.number}"
                    )
                    validation_report['invalid_coefficients'] += 1
                    continue
                
                validation_report['valid_coefficients'] += 1
                
            except Exception as e:
                validation_report['errors'].append(f"Ошибка валидации записи {key}: {e}")
                validation_report['invalid_coefficients'] += 1
        
        return validation_report
    
    def _update_stats(self) -> None:
        """Обновляет статистику коэффициентов."""
        self._stats.update_from_data(self._coefficients)
    
    def _clear_cache(self) -> None:
        """Очищает кэш коэффициентов."""
        self._cached_get_coefficient.cache_clear()
    
    def set_default_coefficient(self, value: float) -> None:
        """Устанавливает значение коэффициента по умолчанию."""
        if value <= 0:
            raise ValueError("Коэффициент по умолчанию должен быть положительным")
        
        self._default_coefficient = value
        self._clear_cache()
        
        logger.info(f"Установлен коэффициент по умолчанию: {value}")
    
    def bulk_add_coefficients(self, coefficients_data: List[Dict[str, any]]) -> Tuple[int, int]:
        """
        Массово добавляет коэффициенты.
        
        Args:
            coefficients_data: Список словарей с данными коэффициентов
            
        Returns:
            Кортеж (успешно добавлено, ошибки)
        """
        success_count = 0
        error_count = 0
        
        for coeff_data in coefficients_data:
            try:
                series = coeff_data['series']
                number = coeff_data['number']
                coefficient = coeff_data['coefficient']
                reason = coeff_data.get('reason', '')
                
                if self.add_coefficient(series, number, coefficient, reason, overwrite=True):
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.warning(f"Ошибка добавления коэффициента: {e}")
                error_count += 1
        
        logger.info(f"Массовое добавление: {success_count} успешно, {error_count} ошибок")
        return success_count, error_count
    
    def clear_all_coefficients(self) -> None:
        """Очищает все коэффициенты."""
        count = len(self._coefficients)
        self._coefficients.clear()
        self._update_stats()
        self._clear_cache()
        
        logger.info(f"Очищено {count} коэффициентов")
    
    def __len__(self) -> int:
        """Возвращает количество коэффициентов."""
        return len(self._coefficients)
    
    def __contains__(self, key: Tuple[str, str]) -> bool:
        """Проверяет наличие коэффициента."""
        return key in self._coefficients