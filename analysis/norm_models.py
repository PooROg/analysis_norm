#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модели данных для обработки HTML норм с использованием современных практик Python 3.12.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum


class NormType(Enum):
    """Типы норм."""
    AXLE_LOAD = "Нажатие"
    TRAIN_WEIGHT = "Вес"


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


@dataclass(slots=True)
class NormProcessingStats:
    """Статистика обработки норм."""
    total_files: int = 0
    total_norms_found: int = 0
    new_norms: int = 0
    updated_norms: int = 0
    skipped_norms: int = 0
    processing_errors: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Преобразует в словарь для совместимости."""
        return {
            'total_files': self.total_files,
            'total_norms_found': self.total_norms_found,
            'new_norms': self.new_norms,
            'updated_norms': self.updated_norms,
            'skipped_norms': self.skipped_norms,
            'processing_errors': self.processing_errors
        }


@dataclass(slots=True, frozen=True)
class TableSection:
    """Секция таблицы с заголовком и содержимым."""
    title: str
    content: str
    norm_type: NormType


@dataclass(slots=True, frozen=True)
class ParsedRow:
    """Распарсенная строка таблицы."""
    norm_id: str
    raw_data: List[str]
    numeric_data: Dict[float, float] = field(default_factory=dict)