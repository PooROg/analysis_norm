# visualization/plot_modes.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Управление режимами отображения точек на графике."""

import logging
from enum import Enum
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from core.utils import safe_float

logger = logging.getLogger(__name__)


class DisplayMode(Enum):
    """Режимы отображения точек."""
    WORK = "work"  # Уд. на работу (текущий)
    NF_RATIO = "nf"  # Н/Ф (по соотношению норма/факт)


class PlotModeManager:
    """Менеджер режимов отображения точек с эффективным кэшированием."""
    
    def __init__(self):
        self.current_mode = DisplayMode.WORK
        self._original_data: Dict[str, Dict] = {}  # trace_name -> {x, y, metadata}
        self._cached_nf_data: Dict[str, Dict] = {}  # кэш для режима Н/Ф
        self._data_dirty = True
        
    def set_original_data(self, traces_data: Dict[str, Dict]) -> None:
        """
        Сохраняет исходные данные трасс для переключения режимов.
        
        Args:
            traces_data: {trace_name: {'x': array, 'y': array, 'routes_df': DataFrame}}
        """
        self._original_data = {}
        self._cached_nf_data = {}
        
        for trace_name, data in traces_data.items():
            # Создаем глубокую копию для безопасности
            self._original_data[trace_name] = {
                'x': np.array(data['x']),
                'y': np.array(data['y']),  # WORK mode Y values
                'routes_df': data['routes_df'].copy() if 'routes_df' in data else None,
                'metadata': data.get('metadata', {})
            }
            
        self._data_dirty = True
        logger.info("Сохранено %d трасс для переключения режимов", len(self._original_data))
        
    def switch_mode(self, mode: DisplayMode) -> Dict[str, np.ndarray]:
        """
        Переключает режим отображения и возвращает новые Y координаты.
        
        Args:
            mode: Новый режим отображения
            
        Returns:
            Словарь {trace_name: new_y_array}
        """
        if mode == self.current_mode and not self._data_dirty:
            logger.debug("Режим не изменился, возвращаем текущие данные")
            return self._get_current_y_data()
            
        logger.info("Переключение с %s на %s", self.current_mode.value, mode.value)
        self.current_mode = mode
        
        if mode == DisplayMode.WORK:
            return self._get_work_mode_data()
        elif mode == DisplayMode.NF_RATIO:
            return self._get_nf_mode_data()
        else:
            logger.error("Неизвестный режим: %s", mode)
            return {}
            
    def _get_work_mode_data(self) -> Dict[str, np.ndarray]:
        """Возвращает Y координаты для режима 'Уд. на работу'."""
        result = {}
        for trace_name, data in self._original_data.items():
            result[trace_name] = np.array(data['y'])  # Исходные значения
            
        logger.debug("Возвращены исходные Y координаты для %d трасс", len(result))
        return result
        
    def _get_nf_mode_data(self) -> Dict[str, np.ndarray]:
        """Возвращает Y координаты для режима 'Н/Ф' с кэшированием."""
        if not self._data_dirty and self._cached_nf_data:
            logger.debug("Используем кэшированные данные Н/Ф")
            return self._cached_nf_data.copy()
            
        logger.info("Вычисляем данные режима Н/Ф для %d трасс", len(self._original_data))
        result = {}
        
        for trace_name, data in self._original_data.items():
            routes_df = data.get('routes_df')
            if routes_df is None or routes_df.empty:
                # Если нет данных маршрутов, оставляем исходные значения
                result[trace_name] = np.array(data['y'])
                continue
                
            original_y = np.array(data['y'])
            nf_y = np.zeros_like(original_y)
            
            # Вычисляем Y для режима Н/Ф по каждой точке
            for i, (_, row) in enumerate(routes_df.iterrows()):
                if i >= len(original_y):
                    break
                    
                nf_value = self._calculate_nf_value(row, original_y[i])
                nf_y[i] = nf_value
                
            result[trace_name] = nf_y
            
        # Кэшируем результат
        self._cached_nf_data = result.copy()
        self._data_dirty = False
        
        logger.info("Вычислены данные режима Н/Ф для %d трасс", len(result))
        return result
        
    def _calculate_nf_value(self, row: pd.Series, original_y: float) -> float:
        """
        Вычисляет Y координату для режима Н/Ф из данных маршрута.
        
        Логика: (Расх.факт.всего / Расх.норма.всего) * Уд.норма.исходная
        """
        try:
            # Пытаемся получить предрасчитанное значение
            expected_nf = safe_float(row.get('expected_nf_y'))
            if expected_nf > 0:
                return expected_nf
                
            # Расчет по формуле
            rashod_fact_total = safe_float(row.get('Расход фактический'))
            rashod_norm_total = safe_float(row.get('Расход по норме'))
            ud_norma_original = safe_float(row.get('Уд. норма, норма на 1 час ман. раб.'))
            
            if rashod_fact_total > 0 and rashod_norm_total > 0 and ud_norma_original > 0:
                coefficient = rashod_fact_total / rashod_norm_total
                adjusted_y = coefficient * ud_norma_original
                
                logger.debug("Н/Ф расчет: %.2f -> %.2f (коэф=%.3f)", 
                           original_y, adjusted_y, coefficient)
                return adjusted_y
                
            # Если данных недостаточно, возвращаем исходное значение
            return original_y
            
        except Exception as e:
            logger.debug("Ошибка расчета Н/Ф для точки: %s", e)
            return original_y
            
    def _get_current_y_data(self) -> Dict[str, np.ndarray]:
        """Возвращает Y координаты текущего режима."""
        if self.current_mode == DisplayMode.WORK:
            return self._get_work_mode_data()
        else:
            return self._get_nf_mode_data()
            
    def get_current_mode(self) -> DisplayMode:
        """Возвращает текущий режим отображения."""
        return self.current_mode
        
    def clear_cache(self) -> None:
        """Очищает кэш вычисленных данных."""
        self._cached_nf_data.clear()
        self._data_dirty = True
        logger.debug("Кэш данных очищен")
        
    def get_mode_label(self, mode: DisplayMode) -> str:
        """Возвращает человекочитаемое название режима."""
        labels = {
            DisplayMode.WORK: "Уд. на работу",
            DisplayMode.NF_RATIO: "Н/Ф (норма/факт)"
        }
        return labels.get(mode, str(mode.value))