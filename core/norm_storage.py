# core/norm_storage.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy.interpolate import interp1d, CubicSpline
import numpy as np

# Настройка логирования
logger = logging.getLogger(__name__)

class NormStorage:
    """Высокопроизводительное хранилище норм"""
    
    def __init__(self, storage_file: str = "norms_storage.pkl"):
        self.storage_file = storage_file
        self.norms_data = {}
        self.norm_functions = {}  # Кэш интерполяционных функций
        self.metadata = {
            'version': '1.0',
            'total_norms': 0,
            'last_updated': None,
            'norm_types': {}
        }
        
        # Загружаем существующие данные
        self.load_storage()
    
    def load_storage(self):
        """Загружает данные из файла хранилища"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'rb') as f:
                    data = pickle.load(f)
                    self.norms_data = data.get('norms_data', {})
                    self.metadata = data.get('metadata', self.metadata)
                
                logger.info(f"Загружено {len(self.norms_data)} норм из {self.storage_file}")
                
                # Пересоздаем функции интерполяции
                self._rebuild_interpolation_functions()
                
            except Exception as e:
                logger.error(f"Ошибка загрузки хранилища норм: {e}")
                self.norms_data = {}
                self.norm_functions = {}
        else:
            logger.info(f"Файл хранилища {self.storage_file} не найден, создаем новое хранилище")
    
    def save_storage(self):
        """Сохраняет данные в файл хранилища"""
        try:
            # Обновляем метаданные
            self.metadata['total_norms'] = len(self.norms_data)
            self.metadata['last_updated'] = pd.Timestamp.now().isoformat()
            
            # Подсчитываем типы норм
            norm_types = {}
            for norm_data in self.norms_data.values():
                norm_type = norm_data.get('norm_type', 'Unknown')
                norm_types[norm_type] = norm_types.get(norm_type, 0) + 1
            self.metadata['norm_types'] = norm_types
            
            data = {
                'norms_data': self.norms_data,
                'metadata': self.metadata
            }
            
            with open(self.storage_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Хранилище норм сохранено в {self.storage_file}")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения хранилища норм: {e}")
    
    def add_or_update_norms(self, new_norms: Dict[str, Dict]) -> Dict[str, str]:
        """Добавляет или обновляет нормы"""
        logger.info(f"Добавление/обновление {len(new_norms)} норм")
        
        update_results = {}
        
        for norm_id, norm_data in new_norms.items():
            if norm_id in self.norms_data:
                if self._norms_are_different(norm_data, self.norms_data[norm_id]):
                    self.norms_data[norm_id] = norm_data
                    update_results[norm_id] = 'updated'
                    logger.debug(f"Норма {norm_id} обновлена")
                else:
                    update_results[norm_id] = 'unchanged'
                    logger.debug(f"Норма {norm_id} не изменилась")
            else:
                self.norms_data[norm_id] = norm_data
                update_results[norm_id] = 'new'
                logger.debug(f"Норма {norm_id} добавлена")
        
        # Пересоздаем функции интерполяции для обновленных норм
        self._rebuild_interpolation_functions(
            [norm_id for norm_id, status in update_results.items() 
             if status in ['new', 'updated']]
        )
        
        # Сохраняем изменения
        self.save_storage()
        
        logger.info(f"Результат обновления: {dict(pd.Series(list(update_results.values())).value_counts())}")
        return update_results
    
    def get_norm(self, norm_id: str) -> Optional[Dict]:
        """Получает норму по ID"""
        return self.norms_data.get(norm_id)
    
    def get_all_norms(self) -> Dict[str, Dict]:
        """Получает все нормы"""
        return self.norms_data.copy()
    
    def get_norm_function(self, norm_id: str):
        """Получает интерполяционную функцию для нормы"""
        if norm_id in self.norm_functions:
            return self.norm_functions[norm_id]
        
        # Создаем функцию если её нет
        norm_data = self.get_norm(norm_id)
        if norm_data and norm_data.get('points'):
            func = self._create_interpolation_function(norm_data['points'])
            self.norm_functions[norm_id] = func
            return func
        
        return None
    
    def interpolate_norm_value(self, norm_id: str, load_value: float) -> Optional[float]:
        """Интерполирует значение нормы для заданной нагрузки"""
        func = self.get_norm_function(norm_id)
        if func:
            try:
                return float(func(load_value))
            except Exception as e:
                logger.error(f"Ошибка интерполяции для нормы {norm_id}: {e}")
        return None
    
    def search_norms(self, **criteria) -> Dict[str, Dict]:
        """Поиск норм по критериям"""
        results = {}
        
        for norm_id, norm_data in self.norms_data.items():
            match = True
            
            for key, value in criteria.items():
                if key == 'norm_type':
                    if norm_data.get('norm_type') != value:
                        match = False
                        break
                elif key == 'norm_id_pattern':
                    if value not in norm_id:
                        match = False
                        break
                elif key in norm_data.get('base_data', {}):
                    if norm_data['base_data'].get(key) != value:
                        match = False
                        break
            
            if match:
                results[norm_id] = norm_data
        
        return results
    
    def get_norms_by_type(self, norm_type: str) -> Dict[str, Dict]:
        """Получает нормы по типу"""
        return self.search_norms(norm_type=norm_type)
    
    def get_storage_info(self) -> Dict:
        """Получает информацию о хранилище"""
        info = self.metadata.copy()
        info['storage_file'] = self.storage_file
        info['file_size_mb'] = os.path.getsize(self.storage_file) / (1024*1024) if os.path.exists(self.storage_file) else 0
        info['cached_functions'] = len(self.norm_functions)
        return info
    
    def export_to_json(self, output_file: str) -> bool:
        """Экспортирует нормы в JSON файл"""
        try:
            export_data = {
                'metadata': self.metadata,
                'norms': self.norms_data
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"Нормы экспортированы в JSON: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта в JSON: {e}")
            return False
    
    def import_from_json(self, input_file: str) -> bool:
        """Импортирует нормы из JSON файла"""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_norms = data.get('norms', {})
            if imported_norms:
                update_results = self.add_or_update_norms(imported_norms)
                logger.info(f"Импортировано норм из JSON: {len(imported_norms)}")
                return True
            else:
                logger.warning("В JSON файле не найдено норм для импорта")
                return False
                
        except Exception as e:
            logger.error(f"Ошибка импорта из JSON: {e}")
            return False
    
    def validate_norms(self) -> Dict[str, List[str]]:
        """Валидирует все нормы в хранилище"""
        validation_results = {
            'valid': [],
            'invalid': [],
            'warnings': []
        }
        
        for norm_id, norm_data in self.norms_data.items():
            try:
                # Проверяем обязательные поля
                if not norm_data.get('points'):
                    validation_results['invalid'].append(f"Норма {norm_id}: отсутствуют точки")
                    continue
                
                points = norm_data['points']
                if len(points) < 2:
                    validation_results['invalid'].append(f"Норма {norm_id}: менее 2 точек")
                    continue
                
                # Проверяем точки
                x_values = [p[0] for p in points]
                y_values = [p[1] for p in points]
                
                if len(set(x_values)) != len(x_values):
                    validation_results['invalid'].append(f"Норма {norm_id}: дублирующиеся значения нагрузки")
                    continue
                
                if any(x <= 0 or y <= 0 for x, y in points):
                    validation_results['invalid'].append(f"Норма {norm_id}: отрицательные или нулевые значения")
                    continue
                
                # Проверяем возможность создания функции интерполяции
                try:
                    self._create_interpolation_function(points)
                    validation_results['valid'].append(norm_id)
                except Exception as e:
                    validation_results['invalid'].append(f"Норма {norm_id}: ошибка интерполяции - {str(e)}")
                    continue
                
                # Предупреждения
                if len(points) > 20:
                    validation_results['warnings'].append(f"Норма {norm_id}: много точек ({len(points)})")
                
            except Exception as e:
                validation_results['invalid'].append(f"Норма {norm_id}: ошибка валидации - {str(e)}")
        
        logger.info(f"Валидация завершена: валидных {len(validation_results['valid'])}, "
                   f"невалидных {len(validation_results['invalid'])}, "
                   f"предупреждений {len(validation_results['warnings'])}")
        
        return validation_results
    
    def _rebuild_interpolation_functions(self, norm_ids: Optional[List[str]] = None):
        """Пересоздает функции интерполяции"""
        if norm_ids is None:
            # Пересоздаем все функции
            norm_ids = list(self.norms_data.keys())
            self.norm_functions = {}
        
        logger.debug(f"Пересоздание функций интерполяции для {len(norm_ids)} норм")
        
        for norm_id in norm_ids:
            norm_data = self.norms_data.get(norm_id)
            if norm_data and norm_data.get('points'):
                try:
                    func = self._create_interpolation_function(norm_data['points'])
                    self.norm_functions[norm_id] = func
                except Exception as e:
                    logger.error(f"Ошибка создания функции для нормы {norm_id}: {e}")
    
    def _create_interpolation_function(self, points: List[Tuple[float, float]]):
        """Создает функцию интерполяции для точек нормы"""
        if len(points) < 2:
            raise ValueError("Недостаточно точек для интерполяции")
        
        # Сортируем точки по X
        sorted_points = sorted(points, key=lambda x: x[0])
        x_values = [p[0] for p in sorted_points]
        y_values = [p[1] for p in sorted_points]
        
        # Проверяем на дубли X
        if len(set(x_values)) != len(x_values):
            raise ValueError("Дублирующиеся значения X")
        
        if len(points) == 2:
            # Линейная интерполяция
            return interp1d(x_values, y_values, kind='linear', 
                          fill_value='extrapolate', bounds_error=False)
        else:
            # Кубическая сплайн-интерполяция
            try:
                return CubicSpline(x_values, y_values, bc_type='natural')
            except:
                # Fallback к квадратичной интерполяции
                try:
                    return interp1d(x_values, y_values, kind='quadratic', 
                                  fill_value='extrapolate', bounds_error=False)
                except:
                    # Fallback к линейной интерполяции
                    return interp1d(x_values, y_values, kind='linear', 
                                  fill_value='extrapolate', bounds_error=False)
    
    def _norms_are_different(self, norm1: Dict, norm2: Dict) -> bool:
        """Сравнивает две нормы на предмет различий"""
        try:
            # Сравниваем точки
            points1 = set(tuple(p) for p in norm1.get('points', []))
            points2 = set(tuple(p) for p in norm2.get('points', []))
            
            if points1 != points2:
                return True
            
            # Сравниваем базовые данные
            base1 = norm1.get('base_data', {})
            base2 = norm2.get('base_data', {})
            
            for key in set(base1.keys()) | set(base2.keys()):
                if base1.get(key) != base2.get(key):
                    return True
            
            # Сравниваем другие поля
            for key in ['norm_type', 'description']:
                if norm1.get(key) != norm2.get(key):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка сравнения норм: {e}")
            return True
    
    def cleanup_storage(self):
        """Очищает неиспользуемые функции интерполяции"""
        # Удаляем функции для норм, которых больше нет
        existing_norm_ids = set(self.norms_data.keys())
        cached_norm_ids = set(self.norm_functions.keys())
        
        for norm_id in cached_norm_ids - existing_norm_ids:
            del self.norm_functions[norm_id]
        
        logger.debug(f"Очистка кэша: удалено {len(cached_norm_ids - existing_norm_ids)} функций")
    
    def get_norm_statistics(self) -> Dict:
        """Получает статистику по нормам"""
        stats = {
            'total_norms': len(self.norms_data),
            'by_type': {},
            'points_distribution': {},
            'avg_points_per_norm': 0,
            'load_range': {'min': float('inf'), 'max': float('-inf')},
            'consumption_range': {'min': float('inf'), 'max': float('-inf')}
        }
        
        total_points = 0
        
        for norm_data in self.norms_data.values():
            # Статистика по типам
            norm_type = norm_data.get('norm_type', 'Unknown')
            stats['by_type'][norm_type] = stats['by_type'].get(norm_type, 0) + 1
            
            # Статистика по точкам
            points = norm_data.get('points', [])
            points_count = len(points)
            total_points += points_count
            
            stats['points_distribution'][points_count] = stats['points_distribution'].get(points_count, 0) + 1
            
            # Диапазоны значений
            for load, consumption in points:
                stats['load_range']['min'] = min(stats['load_range']['min'], load)
                stats['load_range']['max'] = max(stats['load_range']['max'], load)
                stats['consumption_range']['min'] = min(stats['consumption_range']['min'], consumption)
                stats['consumption_range']['max'] = max(stats['consumption_range']['max'], consumption)
        
        if stats['total_norms'] > 0:
            stats['avg_points_per_norm'] = total_points / stats['total_norms']
        
        # Обработка случая, когда нет норм
        if stats['load_range']['min'] == float('inf'):
            stats['load_range'] = {'min': 0, 'max': 0}
        if stats['consumption_range']['min'] == float('inf'):
            stats['consumption_range'] = {'min': 0, 'max': 0}
        
        return stats

# Импортируем pandas только здесь, чтобы избежать циклических импортов
import pandas as pd
