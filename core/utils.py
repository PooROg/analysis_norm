# core/utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Общие утилиты для работы с данными."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd


def normalize_text(text: str) -> str:
    """Единая очистка текста от nbsp/мультипробелов по всему проекту."""
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    """Безопасное преобразование к float с обработкой различных входных типов."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    
    if isinstance(value, (int, float)):
        return float(value)
    
    # Обработка строк
    if isinstance(value, str):
        cleaned = value.strip().replace(' ', '').replace('\xa0', '')
        if cleaned.endswith('.'):
            cleaned = cleaned[:-1]
        cleaned = cleaned.replace(',', '.')
        
        if not cleaned or cleaned.lower() in ('nan', 'none', '-', 'n/a'):
            return default
        
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return default
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Безопасное преобразование к int."""
    float_val = safe_float(value, float(default))
    return int(float_val) if float_val == int(float_val) else default


def safe_divide(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """Безопасное деление с проверкой на None/NaN и деление на ноль."""
    num = safe_float(numerator)
    den = safe_float(denominator)
    
    if den == 0:
        return default
    
    return abs(num / den)


def format_number(value: Any, decimals: int = 1, fallback: str = "N/A") -> str:
    """Безопасное форматирование числа."""
    try:
        num = safe_float(value)
        if num == 0 and value in (None, "", "N/A"):
            return fallback
        return f"{num:.{decimals}f}"
    except Exception:
        return fallback


def read_html_file(file_path: Union[str, Path], encodings: tuple[str, ...] = ('cp1251', 'utf-8', 'utf-8-sig')) -> Optional[str]:
    """Читает HTML файл с несколькими fallback кодировками."""
    path = Path(file_path)
    
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except Exception:
            return None
    
    return None


class StatusClassifier:
    """Классификатор статусов по отклонениям."""
    
    # Пороги отклонений в процентах
    THRESHOLDS = {
        'strong_economy': -30,
        'medium_economy': -20,
        'weak_economy': -5,
        'normal_upper': 5,
        'weak_overrun': 20,
        'medium_overrun': 30,
    }
    
    @classmethod
    def get_status(cls, deviation: float) -> str:
        """Определяет статус по отклонению в процентах."""
        match deviation:
            case d if d < cls.THRESHOLDS['strong_economy']:
                return "Экономия сильная"
            case d if d < cls.THRESHOLDS['medium_economy']:
                return "Экономия средняя"
            case d if d < cls.THRESHOLDS['weak_economy']:
                return "Экономия слабая"
            case d if d <= cls.THRESHOLDS['normal_upper']:
                return "Норма"
            case d if d <= cls.THRESHOLDS['weak_overrun']:
                return "Перерасход слабый"
            case d if d <= cls.THRESHOLDS['medium_overrun']:
                return "Перерасход средний"
            case _:
                return "Перерасход сильный"
    
    @classmethod
    def get_status_color(cls, status: str) -> str:
        """Возвращает цвет для статуса."""
        color_map = {
            "Экономия сильная": "darkgreen",
            "Экономия средняя": "green", 
            "Экономия слабая": "lightgreen",
            "Норма": "blue",
            "Перерасход слабый": "orange",
            "Перерасход средний": "darkorange",
            "Перерасход сильный": "red",
        }
        return color_map.get(status, "gray")


def extract_route_key(route_data: dict) -> Optional[str]:
    """Извлекает уникальный ключ маршрута для группировки дубликатов."""
    number = route_data.get('number')
    trip_date = route_data.get('trip_date') 
    driver_tab = route_data.get('driver_tab')
    
    if all(x is not None for x in [number, trip_date, driver_tab]):
        return f"{number}_{trip_date}_{driver_tab}"
    
    return None