#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленное хранилище норм с оптимизацией производительности.
Использует современные возможности Python 3.12 для управления данными норм.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Callable
from functools import lru_cache
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

from .data_models import NormData, NormType, NormPoint

logger = logging.getLogger(__name__)

# Типы для Python 3.12
type NormFunction = Callable[[float], float]
type NormDict = Dict[str, Any]
type ValidationResult = tuple[bool, str]

class NormStorage:
    """
    Оптимизированное хранилище норм с кэшированием и быстрым доступом.
    
    Ключевые особенности:
    - Кэширование функций интерполяции
    - Валидация данных норм
    - Эффективное управление памятью
    - Быстрый поиск и фильтрация
    """
    
    def __init__(self):
        self._norms: Dict[str, NormDict] = {}
        self._norm_functions: Dict[str, NormFunction] = {}
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        logger.info("Инициализировано хранилище норм")
    
    def add_or_update_norms(self, norms: Dict[str, NormDict]) -> None:
        """Добавляет или обновляет нормы в хранилище."""
        logger.info(f"Добавление/обновление {len(norms)} норм в хранилище")
        
        updated_count = 0
        new_count = 0
        
        for norm_id, norm_data in norms.items():
            if self._validate_norm_data(norm_data):
                if norm_id in self._norms:
                    updated_count += 1
                else:
                    new_count += 1
                
                self._norms[norm_id] = norm_data
                
                # Инвалидируем кэш функции для этой нормы
                if norm_id in self._norm_functions:
                    del self._norm_functions[norm_id]
            else:
                logger.warning(f"Норма {norm_id} не прошла валидацию, пропускаем")
        
        logger.info(f"Добавлено новых норм: {new_count}, обновлено: {updated_count}")
    
    def get_norm(self, norm_id: str) -> Optional[NormDict]:
        """Получает норму по ID."""
        return self._norms.get(norm_id)
    
    def get_norm_function(self, norm_id: str) -> Optional[NormFunction]:
        """
        Получает функцию интерполяции для нормы с кэшированием.
        
        Args:
            norm_id: Идентификатор нормы
            
        Returns:
            Функция интерполяции или None если норма не найдена
        """
        if norm_id in self._norm_functions:
            self._cache_stats['hits'] += 1
            return self._norm_functions[norm_id]
        
        self._cache_stats['misses'] += 1
        
        norm_data = self.get_norm(norm_id)
        if not norm_data:
            return None
        
        points = norm_data.get('points', [])
        if len(points) < 2:
            logger.warning(f"Недостаточно точек для интерполяции нормы {norm_id}")
            return None
        
        try:
            # Создаем функцию интерполяции
            func = self._create_interpolation_function(points)
            
            # Кэшируем функцию
            self._norm_functions[norm_id] = func
            
            logger.debug(f"Создана функция интерполяции для нормы {norm_id}")
            return func
            
        except Exception as e:
            logger.error(f"Ошибка создания функции интерполяции для нормы {norm_id}: {e}")
            return None
    
    def _create_interpolation_function(self, points: List[tuple[float, float]]) -> NormFunction:
        """Создает функцию интерполяции из точек нормы."""
        # Сортируем точки по X (нагрузке)
        sorted_points = sorted(points, key=lambda p: p[0])
        
        x_vals = [p[0] for p in sorted_points]
        y_vals = [p[1] for p in sorted_points]
        
        # Проверяем на дублирующиеся значения X
        if len(set(x_vals)) != len(x_vals):
            logger.warning("Обнаружены дублирующиеся значения X, удаляем дубликаты")
            unique_points = {}
            for x, y in sorted_points:
                if x not in unique_points:
                    unique_points[x] = y
            
            x_vals = list(unique_points.keys())
            y_vals = list(unique_points.values())
        
        # Выбираем метод интерполяции в зависимости от количества точек
        if len(x_vals) == 2:
            # Линейная интерполяция для 2 точек
            interpolator = interp1d(x_vals, y_vals, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
        elif len(x_vals) == 3:
            # Квадратичная интерполяция для 3 точек
            interpolator = interp1d(x_vals, y_vals, kind='quadratic', 
                                  bounds_error=False, fill_value='extrapolate')
        else:
            # Кубический сплайн для 4+ точек
            try:
                interpolator = CubicSpline(x_vals, y_vals, extrapolate=True)
            except Exception:
                # Fallback на линейную интерполяцию
                interpolator = interp1d(x_vals, y_vals, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
        
        def norm_function(load: float) -> float:
            """Функция интерполяции нормы."""
            try:
                result = interpolator(load)
                # Возвращаем положительное значение
                return max(0.0, float(result))
            except Exception:
                # В случае ошибки возвращаем среднее значение Y
                return max(0.0, float(np.mean(y_vals)))
        
        return norm_function
    
    def _validate_norm_data(self, norm_data: NormDict) -> bool:
        """Валидирует данные нормы."""
        try:
            # Проверяем обязательные поля
            if 'norm_id' not in norm_data:
                return False
            
            points = norm_data.get('points', [])
            if not isinstance(points, list) or len(points) < 2:
                return False
            
            # Проверяем каждую точку
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    return False
                
                x, y = point
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    return False
                
                if x <= 0 or y <= 0:  # Проверяем на положительные значения
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации нормы: {e}")
            return False
    
    def get_norms_by_type(self, norm_type: NormType) -> Dict[str, NormDict]:
        """Возвращает нормы определенного типа."""
        result = {}
        
        for norm_id, norm_data in self._norms.items():
            if norm_data.get('norm_type') == norm_type.value:
                result[norm_id] = norm_data
        
        return result
    
    def get_all_norm_ids(self) -> List[str]:
        """Возвращает список всех ID норм."""
        return list(self._norms.keys())
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Возвращает информацию о хранилище."""
        norm_types = {}
        for norm_data in self._norms.values():
            norm_type = norm_data.get('norm_type', 'Unknown')
            norm_types[norm_type] = norm_types.get(norm_type, 0) + 1
        
        return {
            'total_norms': len(self._norms),
            'norm_types': norm_types,
            'cached_functions': len(self._norm_functions),
            'cache_stats': self._cache_stats.copy()
        }
    
    def validate_norms(self) -> Dict[str, Any]:
        """Валидирует все нормы в хранилище."""
        validation_results = {
            'total_norms': len(self._norms),
            'valid_norms': 0,
            'invalid_norms': 0,
            'validation_errors': []
        }
        
        for norm_id, norm_data in self._norms.items():
            if self._validate_norm_data(norm_data):
                validation_results['valid_norms'] += 1
            else:
                validation_results['invalid_norms'] += 1
                validation_results['validation_errors'].append(norm_id)
        
        return validation_results
    
    def get_norm_statistics(self) -> Dict[str, Any]:
        """Возвращает детальную статистику по нормам."""
        if not self._norms:
            return {'total': 0}
        
        stats = {
            'total': len(self._norms),
            'by_type': {},
            'points_distribution': {},
            'load_ranges': {},
            'consumption_ranges': {}
        }
        
        # Статистика по типам
        for norm_data in self._norms.values():
            norm_type = norm_data.get('norm_type', 'Unknown')
            stats['by_type'][norm_type] = stats['by_type'].get(norm_type, 0) + 1
        
        # Статистика по количеству точек
        for norm_data in self._norms.values():
            points_count = len(norm_data.get('points', []))
            range_key = self._get_points_range_key(points_count)
            stats['points_distribution'][range_key] = stats['points_distribution'].get(range_key, 0) + 1
        
        # Диапазоны нагрузок и расходов
        all_loads = []
        all_consumptions = []
        
        for norm_data in self._norms.values():
            points = norm_data.get('points', [])
            for point in points:
                if len(point) == 2:
                    all_loads.append(point[0])
                    all_consumptions.append(point[1])
        
        if all_loads:
            stats['load_ranges'] = {
                'min': min(all_loads),
                'max': max(all_loads),
                'mean': np.mean(all_loads)
            }
        
        if all_consumptions:
            stats['consumption_ranges'] = {
                'min': min(all_consumptions),
                'max': max(all_consumptions),
                'mean': np.mean(all_consumptions)
            }
        
        return stats
    
    def _get_points_range_key(self, points_count: int) -> str:
        """Возвращает ключ диапазона для количества точек."""
        if points_count <= 3:
            return '1-3'
        elif points_count <= 5:
            return '4-5'
        elif points_count <= 10:
            return '6-10'
        else:
            return '11+'
    
    def clear_cache(self) -> None:
        """Очищает кэш функций интерполяции."""
        cleared_count = len(self._norm_functions)
        self._norm_functions.clear()
        self._cache_stats = {'hits': 0, 'misses': 0}
        
        logger.info(f"Очищен кэш: удалено {cleared_count} функций")
    
    def remove_norm(self, norm_id: str) -> bool:
        """Удаляет норму из хранилища."""
        if norm_id in self._norms:
            del self._norms[norm_id]
            
            # Удаляем из кэша если есть
            if norm_id in self._norm_functions:
                del self._norm_functions[norm_id]
            
            logger.info(f"Норма {norm_id} удалена из хранилища")
            return True
        
        return False
    
    def export_norms_to_dataframe(self) -> pd.DataFrame:
        """Экспортирует нормы в DataFrame для анализа."""
        data = []
        
        for norm_id, norm_data in self._norms.items():
            points = norm_data.get('points', [])
            base_data = norm_data.get('base_data', {})
            
            for point in points:
                if len(point) == 2:
                    record = {
                        'norm_id': norm_id,
                        'norm_type': norm_data.get('norm_type', ''),
                        'load': point[0],
                        'consumption': point[1],
                        **base_data
                    }
                    data.append(record)
        
        return pd.DataFrame(data)
    
    def optimize_storage(self) -> Dict[str, int]:
        """Оптимизирует хранилище, удаляя неиспользуемые данные."""
        initial_cache_size = len(self._norm_functions)
        
        # Очищаем кэш неиспользуемых функций
        # (в реальном приложении можно добавить логику отслеживания использования)
        
        # Удаляем дублированные точки в нормах
        cleaned_count = 0
        for norm_id, norm_data in self._norms.items():
            points = norm_data.get('points', [])
            unique_points = []
            seen_loads = set()
            
            for point in points:
                if len(point) == 2 and point[0] not in seen_loads:
                    unique_points.append(point)
                    seen_loads.add(point[0])
                elif len(point) == 2:
                    cleaned_count += 1
            
            if len(unique_points) != len(points):
                norm_data['points'] = unique_points
        
        return {
            'initial_cache_size': initial_cache_size,
            'final_cache_size': len(self._norm_functions),
            'duplicate_points_removed': cleaned_count
        }
    
    def __len__(self) -> int:
        """Возвращает количество норм в хранилище."""
        return len(self._norms)
    
    def __contains__(self, norm_id: str) -> bool:
        """Проверяет наличие нормы в хранилище."""
        return norm_id in self._norms
    
    def __iter__(self):
        """Итератор по ID норм."""
        return iter(self._norms.keys())