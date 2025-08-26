# visualization/plot_modes.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Исправленное управление режимами отображения точек на графике."""

import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from core.utils import safe_float

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Режимы отображения точек."""
    WORK = "work"       # Уд. на работу (текущий)
    NF_RATIO = "nf"     # Н/Ф (по соотношению норма/факт)


class PlotModeManager:
    """
    ИСПРАВЛЕННЫЙ менеджер режимов отображения с robust error handling.
    Убраны сложные вычисления, добавлена защита от ошибок данных.
    """
    
    def __init__(self):
        self.current_mode = DisplayMode.WORK
        
        # Основные данные - строгая типизация для предотвращения ошибок
        self._original_data: Dict[str, Dict] = {}  
        self._nf_cache: Dict[str, np.ndarray] = {}  # кэш Y координат для Н/Ф режима
        self._cache_valid = False
        
        logger.debug("PlotModeManager инициализирован")
        
    def set_original_data(self, traces_data: Dict[str, Dict]) -> None:
        """
        ИСПРАВЛЕННОЕ сохранение исходных данных с валидацией.
        
        Args:
            traces_data: {trace_name: {'x': array, 'y': array, 'metadata': list, 'routes_df': DataFrame}}
        """
        logger.info("Установка исходных данных: %d трасс", len(traces_data))
        
        try:
            self._original_data.clear()
            self._nf_cache.clear()
            
            for trace_name, data in traces_data.items():
                try:
                    # Валидация и безопасное копирование данных
                    x_array = np.array(data.get('x', []), dtype=float)
                    y_array = np.array(data.get('y', []), dtype=float)
                    metadata = data.get('metadata', [])
                    routes_df = data.get('routes_df')
                    
                    # Проверяем консистентность данных
                    if len(x_array) != len(y_array):
                        logger.warning("Несоответствие размеров для трассы %s: x=%d, y=%d", 
                                     trace_name, len(x_array), len(y_array))
                        continue
                        
                    if len(metadata) != len(x_array):
                        logger.warning("Несоответствие метаданных для трассы %s: metadata=%d, points=%d",
                                     trace_name, len(metadata), len(x_array))
                        # Дополняем метаданные пустыми записями при необходимости
                        while len(metadata) < len(x_array):
                            metadata.append({'route_number': 'N/A', 'error': 'missing_metadata'})
                    
                    # Валидация массивов
                    if np.any(~np.isfinite(x_array)) or np.any(~np.isfinite(y_array)):
                        logger.warning("Трасса %s содержит некорректные значения", trace_name)
                        # Фильтруем некорректные значения
                        mask = np.isfinite(x_array) & np.isfinite(y_array) & (x_array > 0) & (y_array > 0)
                        x_array = x_array[mask]
                        y_array = y_array[mask]
                        metadata = [metadata[i] for i in range(len(mask)) if mask[i]]
                        
                    if len(x_array) == 0:
                        logger.warning("Трасса %s не содержит валидных точек", trace_name)
                        continue
                    
                    # Сохраняем валидированные данные
                    self._original_data[trace_name] = {
                        'x': x_array,
                        'y': y_array,  # Исходные Y для режима WORK
                        'metadata': metadata,
                        'routes_df': routes_df.copy() if routes_df is not None and not routes_df.empty else None
                    }
                    
                    logger.debug("✓ Трасса %s: %d валидных точек", trace_name, len(x_array))
                    
                except Exception as trace_error:
                    logger.error("Ошибка обработки трассы %s: %s", trace_name, trace_error)
                    continue
                    
            self._cache_valid = False  # Помечаем кэш как невалидный
            
            logger.info("✓ Исходные данные установлены: %d валидных трасс", len(self._original_data))
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка установки исходных данных: %s", e, exc_info=True)
            self._original_data.clear()
            self._nf_cache.clear()
            
    def switch_mode(self, mode: DisplayMode) -> Dict[str, np.ndarray]:
        """
        ИСПРАВЛЕННОЕ переключение режима с полной обработкой ошибок.
        
        Args:
            mode: Новый режим отображения
            
        Returns:
            Словарь {trace_name: new_y_array} или пустой словарь при ошибке
        """
        logger.info("Переключение режима: %s -> %s", self.current_mode.value, mode.value)
        
        try:
            # Проверяем наличие данных
            if not self._original_data:
                logger.error("Нет исходных данных для переключения режима")
                return {}
                
            # Если режим не изменился и кэш валиден, возвращаем кэшированные данные
            if mode == self.current_mode and self._cache_valid:
                return self._get_current_y_data()
                
            self.current_mode = mode
            
            # Выбираем метод получения данных
            if mode == DisplayMode.WORK:
                result = self._get_work_mode_data_safe()
            elif mode == DisplayMode.NF_RATIO:
                result = self._get_nf_mode_data_safe()
            else:
                logger.error("Неизвестный режим отображения: %s", mode)
                return {}
                
            self._cache_valid = True
            logger.info("✓ Режим переключен успешно: %d трасс обновлено", len(result))
            return result
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка переключения режима: %s", e, exc_info=True)
            return {}
            
    def _get_work_mode_data_safe(self) -> Dict[str, np.ndarray]:
        """Безопасное получение данных для режима 'Уд. на работу'."""
        try:
            result = {}
            for trace_name, data in self._original_data.items():
                # Возвращаем исходные Y координаты
                result[trace_name] = np.array(data['y'], dtype=float)
                
            logger.debug("Режим WORK: возвращены исходные Y координаты для %d трасс", len(result))
            return result
            
        except Exception as e:
            logger.error("Ошибка получения данных режима WORK: %s", e)
            return {}
            
    def _get_nf_mode_data_safe(self) -> Dict[str, np.ndarray]:
        """
        ИСПРАВЛЕННОЕ получение данных для режима 'Н/Ф' с кэшированием и защитой.
        """
        try:
            # Проверяем кэш
            if self._cache_valid and self._nf_cache:
                logger.debug("Используем кэшированные данные Н/Ф")
                return self._nf_cache.copy()
                
            logger.info("Вычисляем данные режима Н/Ф...")
            result = {}
            calculation_stats = {'success': 0, 'fallback': 0, 'error': 0}
            
            for trace_name, data in self._original_data.items():
                try:
                    routes_df = data.get('routes_df')
                    original_y = np.array(data['y'], dtype=float)
                    metadata_list = data.get('metadata', [])
                    
                    # Если нет данных маршрутов или метаданных, оставляем исходные значения
                    if routes_df is None or routes_df.empty or len(metadata_list) != len(original_y):
                        result[trace_name] = original_y
                        calculation_stats['fallback'] += len(original_y)
                        logger.debug("Трасса %s: используем исходные значения (нет данных)", trace_name)
                        continue
                        
                    nf_y = np.zeros_like(original_y)
                    
                    # Вычисляем Н/Ф для каждой точки
                    for i in range(len(original_y)):
                        try:
                            if i < len(metadata_list):
                                # Используем предрасчитанное значение если есть
                                nf_value = metadata_list[i].get('nf_y_value')
                                if nf_value and nf_value > 0:
                                    nf_y[i] = float(nf_value)
                                    calculation_stats['success'] += 1
                                else:
                                    # Вычисляем по формуле из метаданных
                                    calc_nf = self._calculate_nf_from_metadata(metadata_list[i], original_y[i])
                                    nf_y[i] = calc_nf
                                    calculation_stats['success'] += 1
                            else:
                                # Fallback на исходное значение
                                nf_y[i] = original_y[i]
                                calculation_stats['fallback'] += 1
                                
                        except Exception as point_error:
                            logger.debug("Ошибка вычисления Н/Ф для точки %d: %s", i, point_error)
                            nf_y[i] = original_y[i]  # fallback
                            calculation_stats['error'] += 1
                            
                    result[trace_name] = nf_y
                    logger.debug("✓ Трасса %s: Н/Ф рассчитан для %d точек", trace_name, len(nf_y))
                    
                except Exception as trace_error:
                    logger.error("Ошибка вычисления Н/Ф для трассы %s: %s", trace_name, trace_error)
                    # Fallback на исходные данные
                    result[trace_name] = np.array(data['y'], dtype=float)
                    calculation_stats['error'] += len(data['y'])
                    
            # Кэшируем результат
            self._nf_cache = result.copy()
            
            logger.info("✓ Режим Н/Ф вычислен: успешно=%d, fallback=%d, ошибок=%d", 
                       calculation_stats['success'], calculation_stats['fallback'], calculation_stats['error'])
            
            return result
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка вычисления режима Н/Ф: %s", e, exc_info=True)
            # В случае критической ошибки возвращаем исходные данные
            return self._get_work_mode_data_safe()
            
    def _calculate_nf_from_metadata(self, metadata: Dict, fallback_y: float) -> float:
        """
        ИСПРАВЛЕННЫЙ расчет Н/Ф значения из метаданных с множественными fallback.
        
        Формула: (Расход.факт / Расход.норма) * Уд.норма.исходная
        """
        try:
            # Метод 1: Используем предрасчитанное значение
            nf_value = metadata.get('nf_y_value')
            if nf_value and nf_value > 0:
                return float(nf_value)
                
            # Метод 2: Расчет по основной формуле
            rashod_fact = safe_float(metadata.get('rashod_fact'))
            rashod_norm = safe_float(metadata.get('rashod_norm'))  
            ud_norma = safe_float(metadata.get('ud_norma_original'))
            
            if rashod_fact > 0 and rashod_norm > 0 and ud_norma > 0:
                coefficient = rashod_fact / rashod_norm
                result = coefficient * ud_norma
                
                # Проверка разумности результата (не должен отличаться от исходного в 10+ раз)
                if 0.1 <= result / fallback_y <= 10.0:
                    return float(result)
                else:
                    logger.debug("Н/Ф результат выходит за разумные границы: %.2f vs %.2f", result, fallback_y)
                    
            # Метод 3: Альтернативная формула через отклонения
            deviation = safe_float(metadata.get('deviation_percent'))
            if deviation is not None and ud_norma > 0:
                # deviation% = (fact - norm) / norm * 100
                # fact = norm * (1 + deviation/100)
                adjusted_fact = ud_norma * (1 + deviation / 100.0)
                if adjusted_fact > 0:
                    return float(adjusted_fact)
                    
            # Fallback на исходное значение
            return float(fallback_y)
            
        except Exception as e:
            logger.debug("Ошибка расчета Н/Ф из метаданных: %s", e)
            return float(fallback_y)
            
    def _get_current_y_data(self) -> Dict[str, np.ndarray]:
        """Безопасно возвращает Y координаты текущего режима."""
        try:
            if self.current_mode == DisplayMode.WORK:
                return self._get_work_mode_data_safe()
            else:
                return self._get_nf_mode_data_safe()
                
        except Exception as e:
            logger.error("Ошибка получения текущих Y данных: %s", e)
            return {}
            
    def get_current_mode(self) -> DisplayMode:
        """Возвращает текущий режим отображения."""
        return self.current_mode
        
    def clear_cache(self) -> None:
        """Очищает кэш и помечает данные как измененные."""
        try:
            self._nf_cache.clear()
            self._cache_valid = False
            logger.debug("Кэш режимов очищен")
            
        except Exception as e:
            logger.error("Ошибка очистки кэша: %s", e)
            
    def get_mode_label(self, mode: DisplayMode) -> str:
        """Возвращает человекочитаемое название режима."""
        labels = {
            DisplayMode.WORK: "Уд. на работу",
            DisplayMode.NF_RATIO: "Н/Ф (норма/факт)"
        }
        return labels.get(mode, f"Неизвестный режим: {mode.value}")
        
    def get_mode_description(self, mode: DisplayMode) -> str:
        """Возвращает описание режима."""
        descriptions = {
            DisplayMode.WORK: "Показывает фактический удельный расход на тягу (исходные данные участков)",
            DisplayMode.NF_RATIO: "Показывает скорректированный расход по соотношению норма/факт всего маршрута"
        }
        return descriptions.get(mode, "Нет описания")
        
    def validate_data_consistency(self) -> Dict[str, Dict]:
        """
        Валидирует консистентность данных между режимами.
        
        Returns:
            Словарь с результатами валидации по каждой трассе
        """
        validation_results = {}
        
        try:
            work_data = self._get_work_mode_data_safe()
            nf_data = self._get_nf_mode_data_safe()
            
            for trace_name in self._original_data:
                result = {
                    'total_points': 0,
                    'valid_points': 0,
                    'work_y_range': (0.0, 0.0),
                    'nf_y_range': (0.0, 0.0),
                    'avg_ratio': 0.0,
                    'issues': []
                }
                
                try:
                    work_y = work_data.get(trace_name, np.array([]))
                    nf_y = nf_data.get(trace_name, np.array([]))
                    
                    if len(work_y) != len(nf_y):
                        result['issues'].append(f"Размеры не совпадают: work={len(work_y)}, nf={len(nf_y)}")
                        
                    if len(work_y) > 0:
                        result['total_points'] = len(work_y)
                        
                        # Проверяем валидность значений
                        valid_mask = (work_y > 0) & (nf_y > 0) & np.isfinite(work_y) & np.isfinite(nf_y)
                        result['valid_points'] = int(np.sum(valid_mask))
                        
                        if result['valid_points'] > 0:
                            valid_work = work_y[valid_mask]
                            valid_nf = nf_y[valid_mask]
                            
                            result['work_y_range'] = (float(np.min(valid_work)), float(np.max(valid_work)))
                            result['nf_y_range'] = (float(np.min(valid_nf)), float(np.max(valid_nf)))
                            
                            # Средний коэффициент трансформации
                            ratios = valid_nf / valid_work
                            result['avg_ratio'] = float(np.mean(ratios))
                            
                        if result['valid_points'] < result['total_points']:
                            invalid_count = result['total_points'] - result['valid_points']
                            result['issues'].append(f"{invalid_count} невалидных значений")
                            
                except Exception as trace_validation_error:
                    result['issues'].append(f"Ошибка валидации: {trace_validation_error}")
                    
                validation_results[trace_name] = result
                
        except Exception as e:
            logger.error("Ошибка валидации данных: %s", e)
            
        return validation_results
        
    def get_statistics(self) -> Dict:
        """Возвращает статистику работы менеджера режимов."""
        try:
            stats = {
                'current_mode': self.current_mode.value,
                'traces_count': len(self._original_data),
                'cache_valid': self._cache_valid,
                'cache_size': len(self._nf_cache),
                'total_points': 0,
                'traces_details': {}
            }
            
            for trace_name, data in self._original_data.items():
                points_count = len(data.get('y', []))
                stats['total_points'] += points_count
                stats['traces_details'][trace_name] = {
                    'points': points_count,
                    'has_routes_df': data.get('routes_df') is not None
                }
                
            return stats
            
        except Exception as e:
            logger.error("Ошибка получения статистики: %s", e)
            return {'error': str(e)}