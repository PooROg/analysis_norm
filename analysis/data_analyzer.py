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
        """ИСПРАВЛЕННЫЙ анализ данных участка с детальной валидацией."""
        logger.info("=== АНАЛИЗ ДАННЫХ УЧАСТКА ===")
        logger.debug("Участок: %s | Записей: %d | Норма: %s", 
                    section_name, len(routes_df), specific_norm_id or "Все")

        try:
            # ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
            if routes_df is None or routes_df.empty:
                logger.error("DataFrame пуст для анализа")
                return pd.DataFrame(), {}
                
            # ОПРЕДЕЛЕНИЕ НОРМ ДЛЯ АНАЛИЗА
            if specific_norm_id:
                norm_numbers = [str(specific_norm_id)]
                logger.info("Анализ для конкретной нормы: %s", specific_norm_id)
            else:
                # Получаем все уникальные нормы из данных
                unique_norms = routes_df["Номер нормы"].dropna().unique()
                norm_numbers = [str(safe_int(n)) for n in unique_norms if safe_int(n) != 0]
                norm_numbers = list(set(norm_numbers))  # Удаляем дубликаты
                logger.info("Анализ для всех норм участка: %s", norm_numbers)
            
            if not norm_numbers:
                logger.error("Не найдено норм для анализа")
                return routes_df.copy(), {}

            # СОЗДАНИЕ ФУНКЦИЙ НОРМ С ЗАЩИТОЙ ОТ ОШИБОК
            logger.info("Создание функций интерполяции для %d норм", len(norm_numbers))
            norm_functions = self._create_norm_functions_safe(norm_numbers, routes_df)
            
            if not norm_functions:
                logger.warning("Не создано ни одной функции нормы")
                # Возвращаем исходные данные без анализа, но не пустой результат
                return routes_df.copy(), {}

            logger.info("✓ Создано функций норм: %d", len(norm_functions))

            # ПРИМЕНЕНИЕ ИНТЕРПОЛЯЦИИ И РАСЧЕТОВ
            logger.info("Применение интерполяции и расчет отклонений")
            analyzed_df = self._interpolate_and_calculate_safe(routes_df, norm_functions)
            
            # ВАЛИДАЦИЯ РЕЗУЛЬТАТА
            processed_count = len(analyzed_df[analyzed_df["Статус"].notna() & (analyzed_df["Статус"] != "Не определен")])
            logger.info("✓ АНАЛИЗ ЗАВЕРШЕН: обработано %d из %d записей", processed_count, len(analyzed_df))
            
            return analyzed_df, norm_functions
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка анализа данных участка %s: %s", section_name, e, exc_info=True)
            # Возвращаем исходные данные в случае критической ошибки
            return routes_df.copy() if routes_df is not None else pd.DataFrame(), {}

    def _interpolate_and_calculate_safe(self, routes_df: pd.DataFrame, norm_functions: Dict) -> pd.DataFrame:
        """ИСПРАВЛЕННОЕ применение интерполяции с обработкой каждой строки."""
        try:
            df = routes_df.copy()
            
            # Добавляем новые колонки если их нет
            new_columns = ["Норма интерполированная", "Параметр нормирования", "Значение параметра", "Отклонение, %", "Статус"]
            for col in new_columns:
                if col not in df.columns:
                    df[col] = None
            
            processed_count = 0
            error_count = 0
            
            for idx, (df_idx, row) in enumerate(df.iterrows()):
                try:
                    # Получаем номер нормы
                    norm_number = row.get("Номер нормы")
                    if pd.isna(norm_number):
                        continue
                    
                    norm_str = str(safe_int(norm_number))
                    norm_func_data = norm_functions.get(norm_str)
                    if not norm_func_data:
                        continue
                    
                    # Вычисляем параметр нормирования
                    norm_type = norm_func_data.get("norm_type", "Нажатие")
                    x_param = self._calculate_normalization_parameter_safe(row, norm_type)
                    
                    if not x_param or x_param <= 0:
                        continue
                    
                    # Применяем интерполяцию
                    try:
                        interpolation_func = norm_func_data["function"]
                        interpolated_norm = float(interpolation_func(x_param))
                        
                        # Проверяем разумность результата
                        if interpolated_norm <= 0 or interpolated_norm > 10000:  # Разумные границы
                            logger.debug("Некорректный результат интерполяции для строки %d: %.2f", idx, interpolated_norm)
                            continue
                            
                    except Exception as func_error:
                        logger.debug("Ошибка вызова функции интерполяции для строки %d: %s", idx, func_error)
                        continue
                    
                    # Записываем результаты интерполяции
                    df.at[df_idx, "Норма интерполированная"] = interpolated_norm
                    df.at[df_idx, "Параметр нормирования"] = "вес поезда (БРУТТО)" if norm_type == "Вес" else "нажатие на ось"
                    df.at[df_idx, "Значение параметра"] = x_param
                    
                    # Вычисляем отклонение
                    actual_value = safe_float(row.get("Факт уд"))
                    if actual_value and actual_value > 0 and interpolated_norm > 0:
                        try:
                            deviation = ((actual_value - interpolated_norm) / interpolated_norm) * 100.0
                            
                            # Проверяем разумность отклонения (не более ±500%)
                            if -500 <= deviation <= 500:
                                df.at[df_idx, "Отклонение, %"] = deviation
                                df.at[df_idx, "Статус"] = StatusClassifier.get_status(deviation)
                                processed_count += 1
                            else:
                                logger.debug("Некорректное отклонение для строки %d: %.1f%%", idx, deviation)
                                
                        except Exception as deviation_error:
                            logger.debug("Ошибка расчета отклонения для строки %d: %s", idx, deviation_error)
                            
                except Exception as row_error:
                    logger.debug("Ошибка обработки строки %d: %s", idx, row_error)
                    error_count += 1
                    continue
            
            logger.info("✓ Интерполяция завершена: обработано=%d, ошибок=%d из %d строк", 
                    processed_count, error_count, len(df))
            
            return df
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка интерполяции: %s", e, exc_info=True)
            # Возвращаем исходный DataFrame в случае критической ошибки
            return routes_df.copy()

    def _create_norm_functions_safe(self, norm_numbers, routes_df: pd.DataFrame) -> Dict:
        """ИСПРАВЛЕННОЕ создание функций интерполяции с robust error handling."""
        norm_functions = {}
        
        for norm_number in norm_numbers:
            try:
                norm_str = str(safe_int(norm_number)) if pd.notna(norm_number) else None
                if not norm_str or norm_str == "0":
                    continue
                    
                # Получаем данные нормы из хранилища
                norm_data = self.norm_storage.get_norm(norm_str)
                if not norm_data or not norm_data.get("points"):
                    logger.debug("Норма %s не найдена в хранилище или не содержит точек", norm_str)
                    continue
                    
                try:
                    base_points = list(norm_data["points"])
                    norm_type = norm_data.get("norm_type", "Нажатие")
                    
                    # Дополнительные точки из маршрутов - с защитой от ошибок
                    additional_points = []
                    try:
                        additional_points = self._extract_additional_points_safe(routes_df, norm_str, norm_type)
                    except Exception as extract_error:
                        logger.warning("Ошибка извлечения доп. точек для нормы %s: %s", norm_str, extract_error)
                        additional_points = []
                    
                    # Объединение точек
                    all_points = self._merge_points_safe(base_points, additional_points)
                    
                    # Создание функции интерполяции с fallback
                    try:
                        interpolation_func = self.norm_storage._create_interpolation_function(all_points)
                    except Exception as interp_error:
                        logger.warning("Ошибка создания функции интерполяции для нормы %s: %s", norm_str, interp_error)
                        # Создаем простую константную функцию как fallback
                        if base_points:
                            const_y = base_points[0][1]
                            interpolation_func = lambda x: float(const_y)
                        else:
                            continue
                    
                    # Безопасное вычисление диапазона X
                    try:
                        x_range = (min(p[0] for p in base_points), max(p[0] for p in base_points))
                    except Exception:
                        x_range = (1.0, 100.0)  # fallback диапазон
                    
                    norm_functions[norm_str] = {
                        "function": interpolation_func,
                        "points": all_points,
                        "base_points": base_points,
                        "additional_points": additional_points,
                        "x_range": x_range,
                        "data": norm_data,
                        "norm_type": norm_type,
                    }
                    
                    logger.debug("✓ Создана функция для нормы %s (тип: %s, точек: %d)", 
                            norm_str, norm_type, len(all_points))
                    
                except Exception as norm_creation_error:
                    logger.error("Ошибка создания функции для нормы %s: %s", norm_str, norm_creation_error, exc_info=True)
                    continue
                    
            except Exception as norm_error:
                logger.error("Ошибка обработки нормы %s: %s", norm_number, norm_error)
                continue
            
        logger.info("✓ Создано функций норм: %d из %d запрошенных", len(norm_functions), len(norm_numbers))
        return norm_functions

    def _merge_points_safe(self, base_points: List[Tuple[float, float]], 
                        additional_points: List[Tuple[float, float]], 
                        tolerance: float = 0.1) -> List[Tuple[float, float]]:
        """ИСПРАВЛЕННОЕ объединение точек с защитой от ошибок."""
        try:
            if not base_points:
                logger.warning("Нет базовых точек для объединения")
                return additional_points[:] if additional_points else []
                
            if not additional_points:
                return base_points[:]
            
            all_points = list(base_points)  # Копируем базовые точки
            merged_count = 0
            added_count = 0
            
            for x_new, y_new in additional_points:
                try:
                    x_new_f, y_new_f = float(x_new), float(y_new)
                    
                    # Проверяем валидность
                    if x_new_f <= 0 or y_new_f <= 0:
                        continue
                    
                    # Ищем близкую точку для объединения
                    is_merged = False
                    for i, (x_existing, y_existing) in enumerate(all_points):
                        if abs(x_new_f - x_existing) <= tolerance:
                            # Усредняем Y-значения
                            avg_y = (y_existing + y_new_f) / 2.0
                            all_points[i] = (x_existing, avg_y)
                            is_merged = True
                            merged_count += 1
                            break
                    
                    if not is_merged:
                        all_points.append((x_new_f, y_new_f))
                        added_count += 1
                        
                except Exception as point_error:
                    logger.debug("Ошибка обработки дополнительной точки (%s, %s): %s", x_new, y_new, point_error)
                    continue
            
            # Сортируем по X для корректной интерполяции
            all_points.sort(key=lambda p: p[0])
            
            logger.debug("✓ Объединение точек: базовых=%d, объединено=%d, добавлено=%d, итого=%d", 
                        len(base_points), merged_count, added_count, len(all_points))
            
            return all_points
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка объединения точек: %s", e, exc_info=True)
            return base_points[:] if base_points else []

    def _extract_additional_points_safe(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> List[Tuple[float, float]]:
        """ИСПРАВЛЕННОЕ извлечение дополнительных точек с защитой от ошибок."""
        logger.debug("Извлечение дополнительных точек для нормы %s (тип: %s)", norm_id, norm_type)
        
        try:
            # Находим колонку с удельной нормой
            ud_norm_col = self._find_ud_norm_column_safe(routes_df)
            if not ud_norm_col:
                logger.debug("Колонка удельной нормы не найдена")
                return []
            
            points = []
            processed_rows = 0
            
            # Безопасная обработка каждой строки
            for idx, (_, row) in enumerate(routes_df.iterrows()):
                try:
                    processed_rows += 1
                    
                    # Проверка нормы
                    row_norm = row.get("Номер нормы")
                    if pd.isna(row_norm):
                        continue
                        
                    row_norm_str = str(safe_int(row_norm))
                    if row_norm_str != norm_id:
                        continue
                    
                    # Получение значения удельной нормы
                    ud_norm_value = row.get(ud_norm_col)
                    if pd.isna(ud_norm_value) or str(ud_norm_value).strip() in ("", "-", "N/A"):
                        continue
                    
                    try:
                        ud_norm_float = float(str(ud_norm_value).replace(',', '.'))
                        if ud_norm_float <= 0:
                            continue
                    except (ValueError, TypeError):
                        continue
                    
                    # Вычисление параметра нормирования
                    x_param = self._calculate_normalization_parameter_safe(row, norm_type)
                    if x_param and x_param > 0:
                        points.append((x_param, ud_norm_float))
                        logger.debug("Добавлена точка нормы %s: x=%.2f, y=%.2f", norm_id, x_param, ud_norm_float)
                    
                except Exception as row_error:
                    logger.debug("Ошибка обработки строки %d для нормы %s: %s", idx, norm_id, row_error)
                    continue
            
            logger.info("✓ Извлечено дополнительных точек для нормы %s: %d из %d строк", 
                    norm_id, len(points), processed_rows)
            
            return points
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка извлечения точек для нормы %s: %s", norm_id, e, exc_info=True)
            return []

    def _calculate_normalization_parameter_safe(self, row: pd.Series, norm_type: str) -> Optional[float]:
        """ИСПРАВЛЕННОЕ вычисление параметра нормирования с множественными fallback."""
        try:
            if norm_type == "Вес":
                return self._calculate_weight_parameter_safe(row)
            else:  # "Нажатие" и все остальные
                return self._calculate_axle_load_parameter_safe(row)
        except Exception as e:
            logger.debug("Ошибка расчета параметра нормирования: %s", e)
            return None

    def _calculate_weight_parameter_safe(self, row: pd.Series) -> Optional[float]:
        """Безопасно вычисляет вес поезда БРУТТО."""
        try:
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
            
        except Exception as e:
            logger.debug("Ошибка расчета веса: %s", e)
            return None

    def _find_ud_norm_column_safe(self, routes_df: pd.DataFrame) -> Optional[str]:
        """Безопасно находит колонку с удельной нормой."""
        try:
            candidates = [
                "Уд. норма, норма на 1 час ман. раб.",
                "Удельная норма",
                "Уд норма",
                "Норма на 1 час",
                "УД. НОРМА",
            ]
            
            available_columns = list(routes_df.columns)
            
            for candidate in candidates:
                if candidate in available_columns:
                    logger.debug("Найдена колонка удельной нормы: %s", candidate)
                    return candidate
            
            logger.debug("Колонка удельной нормы не найдена среди: %s", available_columns[:10])
            return None
            
        except Exception as e:
            logger.error("Ошибка поиска колонки удельной нормы: %s", e)
            return None

    def _calculate_axle_load_parameter_safe(self, row: pd.Series) -> Optional[float]:
        """Безопасно вычисляет нажатие на ось с fallback методами."""
        try:
            # Метод 1: Готовое значение
            axle_load = safe_float(row.get("Нажатие на ось"))
            if axle_load > 0:
                return axle_load
            
            # Метод 2: Расчет по БРУТТО/ОСИ
            brutto = safe_float(row.get("БРУТТО"))
            osi = safe_float(row.get("ОСИ"))
            
            if brutto > 0 and osi > 0:
                return brutto / osi
            
            # Метод 3: Приблизительный расчет из ткм и км
            tkm_brutto = safe_float(row.get("Ткм брутто"))
            km = safe_float(row.get("Км"))
            
            if tkm_brutto > 0 and km > 0:
                weight = tkm_brutto / km
                # Эмпирическая формула: примерное нажатие на ось
                return weight / 80.0  # ~80 тонн на ось в среднем
            
            return None
            
        except Exception as e:
            logger.debug("Ошибка расчета нажатия на ось: %s", e)

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