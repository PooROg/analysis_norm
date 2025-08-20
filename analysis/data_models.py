#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленные и оптимизированные модели данных для нормирования участков.
Использует современные возможности Python 3.12 для максимальной производительности.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, Optional, Any, List, Tuple, Dict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Новый синтаксис типов Python 3.12
type NormPoints = List[Tuple[float, float]]
type RouteData = Dict[str, Any]
type ValidationResult = Tuple[bool, str]
type ProcessingStats = Dict[str, int]

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
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для совместимости."""
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
    routes: pd.DataFrame
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

# =============== ПРОТОКОЛЫ ===============

class DataProcessor(Protocol):
    """Протокол для обработчиков данных."""
    
    def calculate_axle_load(self, route_data: pd.Series) -> Optional[float]:
        """Вычисляет нагрузку на ось."""
        ...
    
    def determine_status(self, deviation: float) -> str:
        """Определяет статус по отклонению."""
        ...
    
    def apply_coefficients(self, routes: pd.DataFrame, manager: Any) -> pd.DataFrame:
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
    """Базовая реализация обработчика данных."""
    
    def __init__(self, status_config=None):
        self.status_config = status_config
    
    def calculate_axle_load(self, route_data: pd.Series) -> Optional[float]:
        """Вычисляет нагрузку на ось из данных маршрута."""
        try:
            # Попробуем разные варианты полей
            for field in ['ТКМ брутто', 'tkm_brutto', 'Tkm_brutto']:
                if field in route_data and pd.notna(route_data[field]):
                    tkm_brutto = float(route_data[field])
                    break
            else:
                return None
            
            for field in ['КМ', 'km', 'Km']:
                if field in route_data and pd.notna(route_data[field]):
                    km = float(route_data[field])
                    break
            else:
                return None
            
            if km > 0:
                axle_load = tkm_brutto / km / 1000  # Приведение к т/ось
                return abs(axle_load)  # Возвращаем положительное значение
            
            return None
            
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    
    def determine_status(self, deviation: float) -> str:
        """Определяет статус по отклонению."""
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
    
    def apply_coefficients(self, routes: pd.DataFrame, manager: Any) -> pd.DataFrame:
        """Применяет коэффициенты к маршрутам."""
        if manager is None:
            return routes
        
        result_df = routes.copy()
        
        for idx, route in routes.iterrows():
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

# =============== СОВМЕСТИМОСТЬ ===============

# Для обратной совместимости с существующим кодом
MatchResult = Tuple[Optional[Any], bool, bool]

def create_match_result(data: Any = None, is_double: bool = False, 
                       is_approximate: bool = False) -> MatchResult:
    """Создает результат поиска соответствия."""
    return (data, is_double, is_approximate)