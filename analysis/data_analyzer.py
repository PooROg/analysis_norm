# analysis/data_analyzer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Модуль анализа данных маршрутов и норм."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

from core.utils import StatusClassifier, safe_float, safe_int
from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager

logger = logging.getLogger(__name__)


class RouteDataAnalyzer:
    """Анализатор данных маршрутов с интерполяцией норм."""
    
    def __init__(self, norm_storage):
        self.norm_storage = norm_storage
    
    def analyze_section_data(
        self,
        section_name: str,
        routes_df: pd.DataFrame,
        specific_norm_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Анализирует данные участка с интерполяцией норм."""
        logger.debug("Анализ участка %s, строк: %s", section_name, len(routes_df))
        
        # Определяем нормы для анализа
        norm_numbers = [specific_norm_id] if specific_norm_id else routes_df["Номер нормы"].dropna().unique()
        
        # Создаем функции норм
        norm_functions = self._create_norm_functions(norm_numbers, routes_df)
        if not norm_functions:
            logger.warning("Не найдено функций норм для участка %s", section_name)
            return routes_df, {}
        
        # Применяем интерполяцию и расчеты
        analyzed_df = self._interpolate_and_calculate(routes_df, norm_functions)
        
        logger.info("Проанализировано %s строк для участка %s", len(analyzed_df), section_name)
        return analyzed_df, norm_functions
    
    def _create_norm_functions(self, norm_numbers, routes_df: pd.DataFrame) -> Dict:
        """Создает функции интерполяции для норм."""
        norm_functions = {}
        
        for norm_number in norm_numbers:
            norm_str = str(safe_int(norm_number)) if pd.notna(norm_number) else None
            if not norm_str:
                continue
            
            norm_data = self.norm_storage.get_norm(norm_str)
            if not norm_data or not norm_data.get("points"):
                logger.warning("Норма %s не найдена или не содержит точек", norm_str)
                continue
            
            try:
                base_points = list(norm_data["points"])
                norm_type = norm_data.get("norm_type", "Нажатие")
                
                # Дополнительные точки из маршрутов
                additional_points = self._extract_additional_points(routes_df, norm_str, norm_type)
                all_points = self._merge_points(base_points, additional_points)
                
                # Создаем функцию интерполяции
                interpolation_func = self.norm_storage._create_interpolation_function(all_points)
                
                norm_functions[norm_str] = {
                    "function": interpolation_func,
                    "points": all_points,
                    "base_points": base_points,
                    "additional_points": additional_points,
                    "x_range": (min(p[0] for p in base_points), max(p[0] for p in base_points)),
                    "data": norm_data,
                    "norm_type": norm_type,
                }
                
                logger.debug("Создана функция для нормы %s (тип: %s)", norm_str, norm_type)
                
            except Exception as e:
                logger.error("Ошибка создания функции для нормы %s: %s", norm_str, e, exc_info=True)
        
        return norm_functions
    
    def _extract_additional_points(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> List[Tuple[float, float]]:
        """Извлекает дополнительные точки норм из данных маршрутов."""
        # Ищем колонку с удельной нормой
        ud_norm_col = self._find_ud_norm_column(routes_df)
        if not ud_norm_col:
            logger.debug("Колонка удельной нормы не найдена")
            return []
        
        points = []
        
        # ИСПРАВЛЕНО: правильная фильтрация по норме
        for _, row in routes_df.iterrows():
            row_norm = row.get("Номер нормы")
            if pd.isna(row_norm):
                continue
                
            row_norm_str = str(safe_int(row_norm))
            if row_norm_str != norm_id:
                continue
            
            # ИСПРАВЛЕНО: правильное получение значения удельной нормы
            ud_norm_value = row.get(ud_norm_col)
            if pd.isna(ud_norm_value) or str(ud_norm_value).strip() in ("", "-", "N/A"):
                continue
            
            try:
                ud_norm_float = float(str(ud_norm_value).replace(',', '.'))
                if ud_norm_float <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            
            # Вычисляем параметр нормирования
            x_param = self._calculate_normalization_parameter(row, norm_type)
            if x_param and x_param > 0:
                points.append((x_param, ud_norm_float))
                logger.debug("Добавлена доп. точка нормы %s: x=%.2f, y=%.2f", norm_id, x_param, ud_norm_float)
        
        logger.debug("Извлечено %d дополнительных точек для нормы %s", len(points), norm_id)
        return points
    
    def _find_ud_norm_column(self, routes_df: pd.DataFrame) -> Optional[str]:
        """Находит колонку с удельной нормой."""
        candidates = [
            "Уд. норма, норма на 1 час ман. раб.",
            "Удельная норма",
            "Уд норма",
            "Норма на 1 час",
            "УД. НОРМА",
        ]
        
        for candidate in candidates:
            if candidate in routes_df.columns:
                return candidate
        
        return None
    
    def _calculate_normalization_parameter(self, row: pd.Series, norm_type: str) -> Optional[float]:
        """Вычисляет параметр нормирования в зависимости от типа нормы."""
        match norm_type:
            case "Вес":
                return self._calculate_weight_parameter(row)
            case _:  # "Нажатие" и остальные
                return self._calculate_axle_load_parameter(row)
    
    def _calculate_axle_load_parameter(self, row: pd.Series) -> Optional[float]:
        """Вычисляет нажатие на ось."""
        # Готовое значение
        axle_load = safe_float(row.get("Нажатие на ось"))
        if axle_load > 0:
            return axle_load
        
        # Расчет по БРУТТО/ОСИ
        brutto = safe_float(row.get("БРУТТО"))
        osi = safe_float(row.get("ОСИ"))
        
        if brutto > 0 and osi > 0:
            return brutto / osi
        
        # Приблизительный расчет из ткм и км
        tkm_brutto = safe_float(row.get("Ткм брутто"))
        km = safe_float(row.get("Км"))
        
        if tkm_brutto > 0 and km > 0:
            # Эмпирическая формула
            return (tkm_brutto / km) / 50.0  # Приблизительное преобразование
        
        return None
    
    def _calculate_weight_parameter(self, row: pd.Series) -> Optional[float]:
        """Вычисляет вес поезда БРУТТО."""
        # Прямые колонки веса
        for col in ("БРУТТО", "Вес БРУТТО", "Вес поезда БРУТТО", "Брутто"):
            weight = safe_float(row.get(col))
            if weight > 0:
                return weight
        
        # Расчет из ткм брутто / км
        tkm_brutto = safe_float(row.get("Ткм брутто"))
        km = safe_float(row.get("Км"))
        
        if tkm_brutto > 0 and km > 0:
            return tkm_brutto / km
        
        return None
    
    def _merge_points(self, base_points: List[Tuple[float, float]], 
                     additional_points: List[Tuple[float, float]], 
                     tolerance: float = 0.1) -> List[Tuple[float, float]]:
        """Объединяет базовые и дополнительные точки, удаляя дубликаты."""
        if not additional_points:
            return base_points
        
        all_points = list(base_points)
        
        for x_new, y_new in additional_points:
            # Проверяем, нет ли близкой точки
            is_duplicate = False
            for i, (x_existing, y_existing) in enumerate(all_points):
                if abs(x_new - x_existing) <= tolerance:
                    # Усредняем Y-значения
                    all_points[i] = (x_existing, (y_existing + y_new) / 2.0)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                all_points.append((x_new, y_new))
        
        return sorted(all_points, key=lambda p: p[0])
    
    def _interpolate_and_calculate(self, routes_df: pd.DataFrame, norm_functions: Dict) -> pd.DataFrame:
        """Применяет интерполяцию норм и вычисляет отклонения."""
        df = routes_df.copy()
        
        for idx, row in df.iterrows():
            norm_number = row.get("Номер нормы")
            if pd.isna(norm_number):
                continue
            
            norm_str = str(safe_int(norm_number))
            norm_func_data = norm_functions.get(norm_str)
            if not norm_func_data:
                continue
            
            # Вычисляем параметр нормирования
            norm_type = norm_func_data.get("norm_type", "Нажатие")
            x_param = self._calculate_normalization_parameter(row, norm_type)
            
            if not x_param or x_param <= 0:
                continue
            
            try:
                # Интерполируем норму
                interpolated_norm = float(norm_func_data["function"](x_param))
                df.at[idx, "Норма интерполированная"] = interpolated_norm
                df.at[idx, "Параметр нормирования"] = "вес поезда (БРУТТО)" if norm_type == "Вес" else "нажатие на ось"
                df.at[idx, "Значение параметра"] = x_param
                
                # Вычисляем отклонение
                actual_value = safe_float(row.get("Факт уд")) or safe_float(row.get("Расход фактический"))
                if actual_value and interpolated_norm > 0:
                    deviation = ((actual_value - interpolated_norm) / interpolated_norm) * 100.0
                    df.at[idx, "Отклонение, %"] = deviation
                    df.at[idx, "Статус"] = StatusClassifier.get_status(deviation)
                
            except Exception as e:
                logger.debug("Ошибка интерполяции для строки %s: %s", idx, e)
                continue
        
        return df
    
    def calculate_statistics(self, routes_df: pd.DataFrame) -> Dict:
        """Вычисляет статистику по анализированным данным."""
        total = len(routes_df)
        valid_routes = routes_df[routes_df["Статус"].notna() & (routes_df["Статус"] != "Не определен")]
        processed = len(valid_routes)
        
        if processed == 0:
            return {
                "total": total, "processed": 0, "economy": 0, "normal": 0, "overrun": 0,
                "mean_deviation": 0, "median_deviation": 0, "detailed_stats": {},
            }
        
        # Подсчет по статусам
        status_counts = valid_routes["Статус"].value_counts()
        
        detailed_stats = {
            "economy_strong": int(status_counts.get("Экономия сильная", 0)),
            "economy_medium": int(status_counts.get("Экономия средняя", 0)),
            "economy_weak": int(status_counts.get("Экономия слабая", 0)),
            "normal": int(status_counts.get("Норма", 0)),
            "overrun_weak": int(status_counts.get("Перерасход слабый", 0)),
            "overrun_medium": int(status_counts.get("Перерасход средний", 0)),
            "overrun_strong": int(status_counts.get("Перерасход сильный", 0)),
        }
        
        # Общие показатели
        economy_total = detailed_stats["economy_strong"] + detailed_stats["economy_medium"] + detailed_stats["economy_weak"]
        overrun_total = detailed_stats["overrun_weak"] + detailed_stats["overrun_medium"] + detailed_stats["overrun_strong"]
        
        # Статистика отклонений
        deviations = valid_routes["Отклонение, %"].dropna()
        mean_deviation = float(deviations.mean()) if not deviations.empty else 0.0
        median_deviation = float(deviations.median()) if not deviations.empty else 0.0
        
        return {
            "total": total,
            "processed": processed,
            "economy": economy_total,
            "normal": detailed_stats["normal"],
            "overrun": overrun_total,
            "mean_deviation": mean_deviation,
            "median_deviation": median_deviation,
            "detailed_stats": detailed_stats,
        }


class CoefficientsApplier:
    """Применение коэффициентов к данным маршрутов."""
    
    @staticmethod
    def apply_coefficients(routes_df: pd.DataFrame, manager: LocomotiveCoefficientsManager) -> pd.DataFrame:
        """Применяет коэффициенты к 'Расход фактический'."""
        df = routes_df.copy()
        
        # Векторизованное вычисление коэффициентов
        coefficients = df.apply(
            lambda row: CoefficientsApplier._get_coefficient_for_row(row, manager), 
            axis=1
        )
        
        df["Коэффициент"] = coefficients
        
        # Применяем коэффициенты только там, где они отличаются от 1.0
        mask = coefficients != 1.0
        
        if mask.any():
            df.loc[mask, "Факт. удельный исходный"] = df.loc[mask, "Расход фактический"]
            df.loc[mask, "Расход фактический"] = df.loc[mask, "Расход фактический"] / coefficients[mask]
            
            applied_count = int(mask.sum())
            logger.info("Применено коэффициентов: %s", applied_count)
        
        return df
    
    @staticmethod
    def _get_coefficient_for_row(row: pd.Series, manager: LocomotiveCoefficientsManager) -> float:
        """Получает коэффициент для одной строки."""
        series = str(row.get("Серия локомотива", "") or "").strip()
        number_raw = row.get("Номер локомотива")
        
        if not series or pd.isna(number_raw):
            return 1.0
        
        try:
            # Обработка номера локомотива
            if isinstance(number_raw, str):
                number_str = number_raw.strip().lstrip("0")
                number = int(number_str) if number_str else 0
            else:
                number = int(number_raw)
            
            return manager.get_coefficient(series, number)
            
        except (ValueError, TypeError):
            return 1.0