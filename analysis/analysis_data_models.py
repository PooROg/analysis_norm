# analysis/data_models.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированные модели данных с использованием современных возможностей Python 3.12.
Использует dataclasses с slots для экономии памяти и улучшения производительности.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Новый синтаксис типов Python 3.12 (PEP 695)
type NormPoints[T: float] = list[tuple[T, T]]
type RouteData = dict[str, Any]
type ValidationResult = tuple[bool, str]
type ProcessingStats = dict[str, int | float]

@dataclass(slots=True, frozen=True)
class RouteMetadata:
    """Метаданные маршрута с оптимизацией памяти."""
    number: int
    date: str
    depot: str
    identifier: int
    
    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if self.number <= 0:
            raise ValueError(f"Route number must be positive, got {self.number}")
        if not self.date:
            raise ValueError("Date cannot be empty")

@dataclass(slots=True, frozen=True)
class LocomotiveInfo:
    """Информация о локомотиве с оптимизацией памяти."""
    series: str
    number: int
    
    def __post_init__(self) -> None:
        """Валидация после инициализации."""
        if not self.series:
            raise ValueError("Locomotive series cannot be empty")
        if self.number <= 0:
            raise ValueError(f"Locomotive number must be positive, got {self.number}")

@dataclass(slots=True, frozen=True)
class SectionData:
    """Данные участка с оптимизацией производительности."""
    name: str
    norm_number: Optional[str]
    tkm_brutto: float
    km: float
    actual_consumption: float
    norm_consumption: Optional[float] = None
    axle_load: Optional[float] = None
    
    def __post_init__(self) -> None:
        """Валидация и вычисления после инициализации."""
        if not self.name:
            raise ValueError("Section name cannot be empty")
        if self.tkm_brutto < 0 or self.km < 0 or self.actual_consumption < 0:
            raise ValueError("Numeric values must be non-negative")

@dataclass(slots=True)
class ProcessedRoute:
    """Обработанный маршрут с мутабельными полями для процессинга."""
    metadata: RouteMetadata
    locomotive: Optional[LocomotiveInfo]
    sections: list[SectionData] = field(default_factory=list)
    processing_flags: dict[str, bool] = field(default_factory=dict)
    
    def add_section(self, section: SectionData) -> None:
        """Добавляет участок к маршруту."""
        self.sections.append(section)
        logger.debug(f"Added section '{section.name}' to route {self.metadata.number}")
    
    def set_flag(self, flag_name: str, value: bool) -> None:
        """Устанавливает флаг обработки."""
        self.processing_flags[flag_name] = value

@dataclass(slots=True, frozen=True)
class NormDefinition:
    """Определение нормы с оптимизацией памяти."""
    norm_id: str
    points: NormPoints[float]
    description: str = ""
    norm_type: str = "Unknown"
    
    def validate(self) -> ValidationResult:
        """Валидирует данные нормы."""
        if len(self.points) < 2:
            return False, "Minimum 2 points required for norm"
        
        # Проверка на дублирующиеся X значения
        x_values = [p[0] for p in self.points]
        if len(x_values) != len(set(x_values)):
            return False, "Duplicate load values found"
        
        # Проверка положительности значений
        for load, consumption in self.points:
            if load <= 0 or consumption <= 0:
                return False, "All values must be positive"
        
        return True, "OK"
    
    @property
    def load_range(self) -> tuple[float, float]:
        """Диапазон нагрузок."""
        if not self.points:
            return 0.0, 0.0
        loads = [p[0] for p in self.points]
        return min(loads), max(loads)
    
    @property
    def consumption_range(self) -> tuple[float, float]:
        """Диапазон расходов."""
        if not self.points:
            return 0.0, 0.0
        consumptions = [p[1] for p in self.points]
        return min(consumptions), max(consumptions)

@dataclass(slots=True)
class AnalysisResult:
    """Результат анализа с мутабельными полями."""
    section_name: str
    routes_data: list[ProcessedRoute] = field(default_factory=list)
    statistics: dict[str, float] = field(default_factory=dict)
    norm_functions: dict[str, Any] = field(default_factory=dict)
    
    def add_route(self, route: ProcessedRoute) -> None:
        """Добавляет маршрут к результатам."""
        self.routes_data.append(route)
    
    def update_statistics(self, stats: dict[str, float]) -> None:
        """Обновляет статистику."""
        self.statistics.update(stats)

# Протоколы для типизации (современная замена ABC)
class DataParser(Protocol):
    """Протокол для парсеров данных."""
    
    def parse(self, content: str) -> list[ProcessedRoute]:
        """Парсит контент и возвращает список маршрутов."""
        ...
    
    def validate_content(self, content: str) -> ValidationResult:
        """Валидирует контент перед парсингом."""
        ...

class DataProcessor(Protocol):
    """Протокол для процессоров данных."""
    
    def process(self, routes: list[ProcessedRoute]) -> AnalysisResult:
        """Обрабатывает маршруты и возвращает результат анализа."""
        ...
    
    def get_statistics(self) -> ProcessingStats:
        """Возвращает статистику обработки."""
        ...

class NormStorage(Protocol):
    """Протокол для хранилища норм."""
    
    def get_norm(self, norm_id: str) -> Optional[NormDefinition]:
        """Получает норму по ID."""
        ...
    
    def store_norm(self, norm: NormDefinition) -> bool:
        """Сохраняет норму."""
        ...
    
    def validate_norms(self) -> dict[str, ValidationResult]:
        """Валидирует все нормы."""
        ...

# Утилитарные функции с современным синтаксисом
def create_route_from_dict(data: RouteData) -> ProcessedRoute:
    """Создает маршрут из словаря данных."""
    metadata = RouteMetadata(
        number=data['number'],
        date=data['date'], 
        depot=data.get('depot', ''),
        identifier=data['identifier']
    )
    
    locomotive = None
    if data.get('locomotive_series') and data.get('locomotive_number'):
        locomotive = LocomotiveInfo(
            series=data['locomotive_series'],
            number=data['locomotive_number']
        )
    
    route = ProcessedRoute(metadata=metadata, locomotive=locomotive)
    
    # Добавляем участки если есть
    for section_data in data.get('sections', []):
        section = SectionData(
            name=section_data['name'],
            norm_number=section_data.get('norm_number'),
            tkm_brutto=section_data['tkm_brutto'],
            km=section_data['km'],
            actual_consumption=section_data['actual_consumption'],
            norm_consumption=section_data.get('norm_consumption'),
            axle_load=section_data.get('axle_load')
        )
        route.add_section(section)
    
    return route

def validate_norm_points(points: NormPoints[float]) -> ValidationResult:
    """Валидирует точки нормы с подробной диагностикой."""
    if not points:
        return False, "No points provided"
    
    if len(points) < 2:
        return False, f"Need at least 2 points, got {len(points)}"
    
    # Проверка сортировки по X
    x_values = [p[0] for p in points]
    if x_values != sorted(x_values):
        return False, "Points must be sorted by load value"
    
    # Проверка на дубликаты
    if len(set(x_values)) != len(x_values):
        duplicates = [x for x in set(x_values) if x_values.count(x) > 1]
        return False, f"Duplicate load values: {duplicates}"
    
    # Проверка диапазонов
    for i, (load, consumption) in enumerate(points):
        if load <= 0:
            return False, f"Load at point {i+1} must be positive, got {load}"
        if consumption <= 0:
            return False, f"Consumption at point {i+1} must be positive, got {consumption}"
    
    return True, "Valid norm points"
