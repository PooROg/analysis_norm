#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленные модели данных для нормирования участков.
Устранены проблемы с типами Python 3.12 и добавлена совместимость.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Protocol, Optional, Any, List, Tuple, Dict, Union
from enum import Enum
from pathlib import Path

# Импорты pandas с проверкой
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Заглушки для работы без pandas
    class pd:
        class Series:
            pass
        DataFrame = None
        def notna(x): return x is not None
        def isna(x): return x is None
    
    class np:
        def isscalar(x): return isinstance(x, (int, float))
        def isnan(x): return False

logger = logging.getLogger(__name__)

# Условные типы для Python 3.12
if sys.version_info >= (3, 12):
    type NormPoints = List[Tuple[float, float]]
    type RouteData = Dict[str, Any]
    type ValidationResult = Tuple[bool, str]
    type ProcessingStats = Dict[str, int]
else:
    # Для старых версий Python используем обычные типы
    NormPoints = List[Tuple[float, float]]
    RouteData = Dict[str, Any]
    ValidationResult = Tuple[bool, str]
    ProcessingStats = Dict[str, int]

# =============== ENUMS ===============

class NormType(Enum):
    """Типы норм."""
    AXLE_LOAD = "Нажатие"
    TRAIN_WEIGHT = "Вес"

class StatusCategory(Enum):
    """Категории статусов."""
    ECONOMY = "economy"
    NORMAL = "normal" 
    OVERRUN = "overrun"

# =============== БАЗОВЫЕ МОДЕЛИ ===============

@dataclass(slots=True, frozen=True)
class RouteMetadata:
    """Метаданные маршрута с оптимизацией памяти."""
    number: Optional[str] = None
    date: Optional[str] = None
    depot: Optional[str] = ""
    identifier: Optional[str] = None

@dataclass(slots=True, frozen=True)
class LocoData:
    """Данные локомотива."""
    series: Optional[str] = None
    number: Optional[str] = None

@dataclass(slots=True, frozen=True)
class Yu7Data:
    """Данные Ю7 (НЕТТО, БРУТТО, ОСИ)."""
    netto: int
    brutto: int
    osi: int

@dataclass(slots=True)
class RouteSection:
    """Секция маршрута с возможностью изменения."""
    name: Optional[str] = None
    norm_number: Optional[str] = None
    ud_norma_url: Optional[str] = None
    tkm_brutto: Optional[float] = None
    km: Optional[float] = None
    pr: Optional[float] = None
    rashod_fact: Optional[float] = None
    rashod_norm: Optional[float] = None
    ud_norma: Optional[float] = None
    norma_rabotu: Optional[float] = None
    norma_odinochnoe: Optional[float] = None
    
    # Данные станций
    prostoy_vsego: Optional[float] = None
    prostoy_norma: Optional[float] = None
    manevry_vsego: Optional[float] = None
    manevry_norma: Optional[float] = None
    troganie_vsego: Optional[float] = None
    troganie_norma: Optional[float] = None
    nagon_vsego: Optional[float] = None
    nagon_norma: Optional[float] = None
    ogranich_vsego: Optional[float] = None
    ogranich_norma: Optional[float] = None
    peresyl_vsego: Optional[float] = None
    peresyl_norma: Optional[float] = None
    
    # Результаты обработки
    netto: Optional[Any] = None
    brutto: Optional[Any] = None
    osi: Optional[Any] = None
    use_red_color: bool = False
    double_traction: Optional[str] = None
    is_merged: bool = False

# =============== МОДЕЛИ НОРМ ===============

@dataclass(slots=True, frozen=True)
class NormPoint:
    """Точка нормы (нагрузка, расход)."""
    load: float
    consumption: float
    
    def __post_init__(self):
        """Валидация после создания."""
        if self.load <= 0 or self.consumption <= 0:
            raise ValueError(f"Значения должны быть положительными: load={self.load}, consumption={self.consumption}")

@dataclass(slots=True, frozen=True)
class BaseNormData:
    """Базовые данные нормы."""
    priznok_sost_tyag: Optional[float] = None
    priznok_rek: Optional[float] = None
    vid_dvizheniya: str = ""
    simvol_rod_raboty: Optional[float] = None
    rps: Optional[float] = None
    identif_gruppy: Optional[float] = None
    priznok_sost: Optional[float] = None
    priznok_alg: Optional[float] = None
    date_start: str = ""
    date_end: str = ""

@dataclass(slots=True, frozen=True)
class NormData:
    """Полные данные нормы."""
    norm_id: str
    norm_type: NormType
    description: str
    points: List[NormPoint] = field(default_factory=list)
    base_data: BaseNormData = field(default_factory=BaseNormData)
    
    def __post_init__(self):
        """Валидация после создания."""
        if not self.norm_id:
            raise ValueError("ID нормы не может быть пустым")
        
        if len(self.points) < 2:
            logger.warning(f"Норма {self.norm_id} имеет менее 2 точек")
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для совместимости."""
        try:
            return {
                'norm_id': self.norm_id,
                'norm_type': self.norm_type.value,
                'description': self.description,
                'points': [(point.load, point.consumption) for point in self.points],
                'base_data': {
                    'priznok_sost_tyag': self.base_data.priznok_sost_tyag,
                    'priznok_rek': self.base_data.priznok_rek,
                    'vid_dvizheniya': self.base_data.vid_dvizheniya,
                    'simvol_rod_raboty': self.base_data.simvol_rod_raboty,
                    'rps': self.base_data.rps,
                    'identif_gruppy': self.base_data.identif_gruppy,
                    'priznok_sost': self.base_data.priznok_sost,
                    'priznok_alg': self.base_data.priznok_alg,
                    'date_start': self.base_data.date_start,
                    'date_end': self.base_data.date_end
                }
            }
        except Exception as e:
            logger.error(f"Ошибка конвертации нормы {self.norm_id} в словарь: {e}")
            return {}

@dataclass(slots=True, frozen=True)
class TableSection:
    """Секция таблицы с заголовком и содержимым."""
    title: str
    content: str
    norm_type: NormType

# =============== РЕЗУЛЬТАТЫ АНАЛИЗА ===============

@dataclass(slots=True)
class AnalysisResult:
    """Результат анализа участка."""
    routes: Optional[pd.DataFrame] if PANDAS_AVAILABLE else Optional[Any]
    norms: Dict[str, Any]
    statistics: Dict[str, Any]
    section_name: str
    norm_id: Optional[str] = None
    single_section_only: bool = False

@dataclass(slots=True)
class ProcessingStats:
    """Универсальная статистика обработки."""
    total_files: int = 0
    total_items_found: int = 0  # маршруты или нормы
    new_items: int = 0
    updated_items: int = 0
    processed_items: int = 0
    skipped_items: int = 0
    processing_errors: int = 0
    duplicate_details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для совместимости."""
        return {
            'total_files': self.total_files,
            'total_items_found': self.total_items_found,
            'new_items': self.new_items,
            'updated_items': self.updated_items,
            'processed_items': self.processed_items,
            'skipped_items': self.skipped_items,
            'processing_errors': self.processing_errors,
            'duplicate_details': self.duplicate_details
        }
    
    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Обновляет статистику из словаря."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

# =============== ПРОТОКОЛЫ ===============

class DataProcessor(Protocol):
    """Протокол для обработчиков данных."""
    
    def calculate_axle_load(self, route_data: Union[pd.Series, Dict[str, Any]]) -> Optional[float]:
        """Вычисляет нагрузку на ось."""
        ...
    
    def determine_status(self, deviation: float) -> str:
        """Определяет статус по отклонению."""
        ...
    
    def apply_coefficients(self, routes: Union[pd.DataFrame, Any], manager: Any) -> Union[pd.DataFrame, Any]:
        """Применяет коэффициенты к маршрутам."""
        ...

class DataParser(Protocol):
    """Протокол для парсеров данных."""
    
    def parse_routes(self, html_content: str) -> List[RouteData]:
        """Парсит маршруты из HTML."""
        ...
    
    def parse_norms(self, html_content: str) -> Dict[str, NormData]:
        """Парсит нормы из HTML."""
        ...

# =============== КЛАССЫ-РЕАЛИЗАЦИИ ===============

class DefaultDataProcessor:
    """Базовая реализация обработчика данных с исправленными ошибками."""
    
    def __init__(self, status_config=None):
        self.status_config = status_config
    
    def calculate_axle_load(self, route_data: Union[pd.Series, Dict[str, Any]]) -> Optional[float]:
        """Вычисляет нагрузку на ось из данных маршрута."""
        try:
            # Обрабатываем разные типы входных данных
            if hasattr(route_data, 'get'):  # pd.Series или Dict
                get_func = route_data.get
            else:
                logger.warning(f"Неподдерживаемый тип данных маршрута: {type(route_data)}")
                return None
            
            # Попробуем разные варианты полей
            tkm_brutto = None
            for field in ['ТКМ брутто', 'tkm_brutto', 'Tkm_brutto']:
                value = get_func(field)
                if value is not None and not (PANDAS_AVAILABLE and pd.isna(value)):
                    try:
                        tkm_brutto = float(value)
                        break
                    except (ValueError, TypeError):
                        continue
            
            if tkm_brutto is None:
                return None
            
            km = None
            for field in ['КМ', 'km', 'Km']:
                value = get_func(field)
                if value is not None and not (PANDAS_AVAILABLE and pd.isna(value)):
                    try:
                        km = float(value)
                        break
                    except (ValueError, TypeError):
                        continue
            
            if km is None or km <= 0:
                return None
            
            axle_load = tkm_brutto / km / 1000  # Приведение к т/ось
            return abs(axle_load) if axle_load is not None else None
            
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Ошибка вычисления нагрузки на ось: {e}")
            return None
        except Exception as e:
            logger.error(f"Неожиданная ошибка в calculate_axle_load: {e}")
            return None
    
    def determine_status(self, deviation: float) -> str:
        """Определяет статус по отклонению."""
        try:
            if not isinstance(deviation, (int, float)) or (PANDAS_AVAILABLE and pd.isna(deviation)):
                return 'Не определен'
            
            if deviation <= -30:
                return 'Экономия сильная'
            elif deviation <= -20:
                return 'Экономия средняя'
            elif deviation <= -5:
                return 'Экономия слабая'
            elif deviation <= 5:
                return 'Норма'
            elif deviation <= 20:
                return 'Перерасход слабый'
            elif deviation <= 30:
                return 'Перерасход средний'
            else:
                return 'Перерасход сильный'
        except Exception as e:
            logger.error(f"Ошибка определения статуса для отклонения {deviation}: {e}")
            return 'Не определен'
    
    def apply_coefficients(self, routes: Union[pd.DataFrame, Any], manager: Any) -> Union[pd.DataFrame, Any]:
        """Применяет коэффициенты к маршрутам."""
        if manager is None or routes is None:
            return routes
        
        try:
            if PANDAS_AVAILABLE and isinstance(routes, pd.DataFrame):
                return self._apply_coefficients_dataframe(routes, manager)
            else:
                logger.warning("Применение коэффициентов к не-DataFrame данным не поддерживается")
                return routes
        except Exception as e:
            logger.error(f"Ошибка применения коэффициентов: {e}")
            return routes
    
    def _apply_coefficients_dataframe(self, routes_df: pd.DataFrame, manager: Any) -> pd.DataFrame:
        """Применяет коэффициенты к DataFrame."""
        result_df = routes_df.copy()
        
        for idx, route in routes_df.iterrows():
            try:
                series = route.get('Серия локомотива', '')
                number = route.get('Номер локомотива', '')
                
                if series and number:
                    coeff = manager.get_coefficient(series, number)
                    if coeff and coeff != 1.0:
                        # Сохраняем исходное значение
                        original_value = route.get('Факт уд')
                        if pd.notna(original_value):
                            result_df.loc[idx, 'Факт. удельный исходный'] = original_value
                            result_df.loc[idx, 'Факт уд'] = original_value * coeff
                            result_df.loc[idx, 'Коэффициент'] = coeff
                            
            except Exception as e:
                logger.debug(f"Ошибка применения коэффициента для строки {idx}: {e}")
                continue
        
        return result_df

# =============== УТИЛИТЫ ===============

def safe_float_conversion(value: Any) -> Optional[float]:
    """Безопасное преобразование значения в float."""
    if value is None:
        return None
    
    if PANDAS_AVAILABLE and pd.isna(value):
        return None
    
    try:
        result = float(value)
        return result if not (PANDAS_AVAILABLE and pd.isna(result)) else None
    except (ValueError, TypeError):
        return None

def safe_int_conversion(value: Any) -> Optional[int]:
    """Безопасное преобразование значения в int."""
    if value is None:
        return None
    
    if PANDAS_AVAILABLE and pd.isna(value):
        return None
    
    try:
        return int(float(value))  # Через float для обработки "1.0"
    except (ValueError, TypeError):
        return None

def validate_norm_points(points: List[Tuple[float, float]]) -> ValidationResult:
    """Валидирует точки нормы."""
    try:
        if not isinstance(points, list):
            return False, "Точки должны быть списком"
        
        if len(points) < 2:
            return False, "Минимум 2 точки для нормы"
        
        seen_x = set()
        for i, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                return False, f"Точка {i} имеет неверный формат"
            
            x, y = point
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False, f"Точка {i} содержит не числовые значения"
            
            if x <= 0 or y <= 0:
                return False, f"Точка {i} содержит отрицательные или нулевые значения"
            
            if x in seen_x:
                return False, f"Дублирующееся значение X: {x}"
            
            seen_x.add(x)
        
        return True, "OK"
        
    except Exception as e:
        return False, f"Ошибка валидации: {str(e)}"

# =============== СОВМЕСТИМОСТЬ ===============

# Для обратной совместимости с существующим кодом
MatchResult = Tuple[Optional[Any], bool, bool]

def create_match_result(data: Any = None, is_double: bool = False, 
                       is_approximate: bool = False) -> MatchResult:
    """Создает результат поиска соответствия."""
    return (data, is_double, is_approximate)

def create_default_processing_stats() -> ProcessingStats:
    """Создает статистику обработки по умолчанию."""
    return ProcessingStats()

def ensure_pandas_compatibility():
    """Проверяет совместимость с pandas."""
    if not PANDAS_AVAILABLE:
        logger.warning("Pandas не доступен. Некоторые функции могут работать некорректно.")
        return False
    return True

# Проверяем совместимость при импорте
if __name__ != "__main__":
    ensure_pandas_compatibility()