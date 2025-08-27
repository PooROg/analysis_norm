# analysis/visualization.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Модуль создания интерактивных графиков для анализа норм."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.utils import StatusClassifier, format_number, safe_float, safe_int

logger = logging.getLogger(__name__)


class PlotBuilder:
    """Построитель интерактивных графиков норм расхода."""
    
    def __init__(self):
        self.js_file_path = Path(__file__).parent.parent / "static" / "interactive_plot.js"
    
    def create_interactive_plot(
        self,
        section_name: str,
        routes_df: pd.DataFrame,
        norm_functions: Dict,
        specific_norm_id: Optional[str] = None,
        single_section_only: bool = False,
    ) -> go.Figure:
        """Создает интерактивный график анализа участка."""
        title_suffix = f" (норма {specific_norm_id})" if specific_norm_id else ""
        filter_suffix = " [только один участок]" if single_section_only else ""
        
        fig = self._create_base_structure(section_name, title_suffix, filter_suffix)
        norm_types_used = self._get_norm_types_used(norm_functions)
        
        # Добавляем элементы графика
        self._add_norm_curves(fig, norm_functions, routes_df, specific_norm_id, norm_types_used)
        self._add_route_points(fig, routes_df, norm_functions, norm_types_used)
        self._add_deviation_analysis(fig, routes_df)
        
        self._configure_layout(fig, norm_types_used)
        return fig
    
    def _create_base_structure(self, section_name: str, title_suffix: str, filter_suffix: str) -> go.Figure:
        """Создает базовую структуру с двумя подграфиками."""
        return make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f"Нормы расхода для участка: {section_name}{title_suffix}{filter_suffix}",
                "Отклонение фактического расхода от нормы",
            ),
        )
    
    def _get_norm_types_used(self, norm_functions: Dict) -> Set[str]:
        """Определяет типы норм, используемые в анализе."""
        return {nf.get("norm_type", "Нажатие") for nf in norm_functions.values()}
    
    def _add_norm_curves(
        self,
        fig: go.Figure,
        norm_functions: Dict,
        routes_df: pd.DataFrame,
        specific_norm_id: Optional[str],
        norm_types_used: Set[str],
    ) -> None:
        """Добавляет кривые норм на график."""
        for norm_id, norm_data in norm_functions.items():
            if specific_norm_id and norm_id != specific_norm_id:
                continue
            
            norm_type = norm_data.get("norm_type", "Нажатие")
            x_axis_name = "Вес поезда БРУТТО, т" if norm_type == "Вес" else "Нажатие на ось, т/ось"
            
            # 1. Добавляем базовые точки нормы (из файла норм) - СИНИЕ КВАДРАТЫ
            base_points = norm_data.get("base_points", [])
            if base_points:
                self._add_base_norm_points(fig, norm_id, base_points, x_axis_name)
            
            # 2. Добавляем дополнительные точки (из маршрутов) - ЖЕЛТЫЕ КВАДРАТЫ
            additional_points = norm_data.get("additional_points", [])
            if additional_points:
                self._add_additional_norm_points_from_data(fig, norm_id, additional_points, norm_type, routes_df, x_axis_name)
            
            # 3. Добавляем интерполированную кривую (используем все точки)
            all_points = norm_data.get("points", [])
            if len(all_points) > 1:
                self._add_interpolated_norm_curve(fig, norm_id, all_points, norm_type, routes_df, x_axis_name)
            elif len(all_points) == 1:
                self._add_constant_norm_curve(fig, norm_id, all_points[0], routes_df, norm_type, x_axis_name)

    def _add_constant_norm_curve(self, fig: go.Figure, norm_id: str, point: Tuple[float, float], 
                            routes_df: pd.DataFrame, norm_type: str, x_axis_name: str) -> None:
        """Добавляет константную норму (одна точка)."""
        x_single, y_single = point
        
        # Определяем диапазон для константной линии
        x_vals_from_data = self._get_route_x_values(routes_df, norm_id, norm_type)
        if x_vals_from_data:
            x_min, x_max = min(x_vals_from_data), max(x_vals_from_data)
            x_range_size = max(x_max - x_min, 1.0)
            x_start = max(x_min - x_range_size * 0.2, x_min * 0.8)
            x_end = x_max + x_range_size * 0.2
        else:
            x_start = max(x_single * 0.5, x_single - 100)
            x_end = x_single * 1.5 + 100

        x_const = np.linspace(x_start, x_end, 100)
        y_const = np.full_like(x_const, y_single)
        
        fig.add_trace(
            go.Scatter(
                x=x_const, y=y_const,
                mode="lines",
                name=f"Норма {norm_id} (константа)",
                line=dict(width=3, color="blue"),
                hovertemplate=f"<b>Норма {norm_id}</b><br>{x_axis_name}: %{{x:.1f}}<br>"
                            f"Расход: %{{y:.1f}} кВт·ч/10⁴ ткм<extra></extra>",
            ),
            row=1, col=1,
        )   

    def _add_interpolated_norm_curve(self, fig: go.Figure, norm_id: str, 
                                all_points: List[Tuple[float, float]], norm_type: str, 
                                routes_df: pd.DataFrame, x_axis_name: str) -> None:
        """Добавляет интерполированную кривую нормы."""
        if len(all_points) < 2:
            return
        
        x_vals = [p[0] for p in all_points]
        y_vals = [p[1] for p in all_points]

        # Определяем диапазон интерполяции
        x_range = self._calculate_interpolation_range(x_vals, routes_df, norm_id, norm_type)
        x_interp = np.linspace(x_range[0], x_range[1], 500)
        y_interp = np.interp(x_interp, x_vals, y_vals)

        fig.add_trace(
            go.Scatter(
                x=x_interp, y=y_interp,
                mode="lines",
                name=f"Норма {norm_id} ({len(all_points)} точек)",
                line=dict(width=3, color="blue"),
                hovertemplate=f"<b>Норма {norm_id}</b><br>{x_axis_name}: %{{x:.1f}}<br>"
                            f"Расход: %{{y:.1f}} кВт·ч/10⁴ ткм<extra></extra>",
            ),
            row=1, col=1,
        )

    def _add_base_norm_points(self, fig: go.Figure, norm_id: str, base_points: List[Tuple[float, float]], x_axis_name: str) -> None:
        """Добавляет базовые точки норм из файла (синие квадраты)."""
        if not base_points:
            return
        
        x_vals = [p[0] for p in base_points]
        y_vals = [p[1] for p in base_points]
        
        hover_texts = [
            f"<b>Базовая точка нормы {norm_id}</b><br>{x_axis_name}: {x:.1f}<br>"
            f"Расход: {y:.1f} кВт·ч/10⁴ ткм<br><i>Источник: файл нормы</i>"
            for x, y in base_points
        ]
        
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                name=f"Базовые точки нормы {norm_id} ({len(base_points)})",
                marker=dict(size=8, symbol="square", color="blue", opacity=0.9,
                        line=dict(color="darkblue", width=1)),
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts
            ),
            row=1, col=1,
        )

    def _add_additional_norm_points_from_data(self, fig: go.Figure, norm_id: str, 
                                            additional_points: List[Tuple[float, float]], 
                                            norm_type: str, routes_df: pd.DataFrame, x_axis_name: str) -> None:
        """Добавляет дополнительные точки норм из маршрутов (желтые квадраты)."""
        if not additional_points:
            return
        
        # Получаем подробную информацию о точках из маршрутов
        points_with_routes = self._get_additional_points_with_route_info(routes_df, norm_id, norm_type, additional_points)
        
        if not points_with_routes:
            return
        
        x_vals = [p[0] for p in points_with_routes]
        y_vals = [p[1] for p in points_with_routes]
        
        hover_texts = []
        for x, y, route_info in points_with_routes:
            hover_text = (
                f"<b>Точка нормы {norm_id} из маршрута</b><br>"
                f"{x_axis_name}: {x:.1f}<br>"
                f"Расход: {y:.1f} кВт·ч/10⁴ ткм<br>"
                f"<i>Источник: {route_info}</i>"
            )
            hover_texts.append(hover_text)
        
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                name=f"Из маршрутов {norm_id} ({len(points_with_routes)})",
                marker=dict(size=8, symbol="square", color="gold", opacity=0.9,
                        line=dict(color="orange", width=1)),
                hovertemplate="%{text}<extra></extra>",
                text=hover_texts
            ),
            row=1, col=1,
        )

    def _get_additional_points_with_route_info(self, routes_df: pd.DataFrame, norm_id: str, 
                                            norm_type: str, additional_points: List[Tuple[float, float]]) -> List[Tuple[float, float, str]]:
        """Получает дополнительные точки с информацией о маршрутах."""
        # Находим колонку с удельной нормой
        ud_norm_col = None
        for col in ["Уд. норма, норма на 1 час ман. раб.", "Удельная норма", "Уд норма"]:
            if col in routes_df.columns:
                ud_norm_col = col
                break
        
        if not ud_norm_col:
            return []
        
        # Создаем словарь для группировки точек по координатам
        points_dict = {}
        
        for _, row in routes_df.iterrows():
            # Проверяем, что это нужная норма
            row_norm = row.get("Номер нормы")
            if pd.isna(row_norm) or str(safe_int(row_norm)) != norm_id:
                continue
            
            # Получаем удельную норму
            ud_norm_value = row.get(ud_norm_col)
            if pd.isna(ud_norm_value) or str(ud_norm_value).strip() in ("", "-", "N/A"):
                continue
            
            try:
                y_val = float(str(ud_norm_value).replace(',', '.'))
                if y_val <= 0:
                    continue
            except (ValueError, TypeError):
                continue
            
            # Вычисляем X координату
            if norm_type == "Вес":
                x_val = self._calculate_weight_parameter(row)
            else:
                x_val = self._calculate_axle_load_parameter(row)
            
            if not x_val or x_val <= 0:
                continue
            
            # Проверяем, что эта точка есть в additional_points
            point_found = False
            for add_x, add_y in additional_points:
                if abs(add_x - x_val) < 0.1 and abs(add_y - y_val) < 0.1:
                    point_found = True
                    break
            
            if not point_found:
                continue
            
            # Создаем ключ для группировки близких точек
            key = (round(x_val, 1), round(y_val, 1))
            
            # Получаем информацию о маршруте
            route_number = row.get("Номер маршрута", "N/A")
            route_date = row.get("Дата маршрута", "N/A")
            route_info = f"Маршрут №{route_number}, {route_date}"
            
            if key not in points_dict:
                points_dict[key] = {
                    'x': x_val,
                    'y': y_val, 
                    'routes': [route_info]
                }
            else:
                # Добавляем информацию о маршруте, если её еще нет
                if route_info not in points_dict[key]['routes']:
                    points_dict[key]['routes'].append(route_info)
        
        # Преобразуем в список с объединенной информацией о маршрутах
        result = []
        for point_data in points_dict.values():
            x, y = point_data['x'], point_data['y']
            routes_info = "; ".join(point_data['routes'][:3])  # Ограничиваем до 3 маршрутов
            if len(point_data['routes']) > 3:
                routes_info += f" и еще {len(point_data['routes']) - 3}"
            
            result.append((x, y, routes_info))
        
        # Сортируем по X координате
        result.sort(key=lambda item: item[0])
        
        return result

    def _calculate_interpolation_range(
        self, 
        x_vals: List[float], 
        routes_df: pd.DataFrame, 
        norm_id: str, 
        norm_type: str
    ) -> Tuple[float, float]:
        """Вычисляет диапазон для интерполяции кривой нормы."""
        x_min, x_max = min(x_vals), max(x_vals)
        x_range = max(x_max - x_min, 1.0)
        
        # Базовый диапазон
        x_start = max(x_min - x_range * 0.3, x_min * 0.5)
        x_end = x_max + x_range * 0.3
        
        # Расширяем под данные маршрутов
        route_x_vals = self._get_route_x_values(routes_df, norm_id, norm_type)
        if route_x_vals:
            x_start = min(x_start, min(route_x_vals) * 0.8)
            x_end = max(x_end, max(route_x_vals) * 1.2)
        
        return x_start, x_end
    
    def _get_route_x_values(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> List[float]:
        """Получает X-координаты из данных маршрутов для конкретной нормы."""
        norm_routes = routes_df[routes_df["Номер нормы"].astype(str) == str(norm_id)]
        x_values = []
        
        for _, row in norm_routes.iterrows():
            if norm_type == "Вес":
                x_val = self._calculate_weight_parameter(row)
            else:
                x_val = self._calculate_axle_load_parameter(row)
            
            if x_val and x_val > 0:
                x_values.append(x_val)
        
        return x_values
    
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
        
        return None
    
    def _calculate_weight_parameter(self, row: pd.Series) -> Optional[float]:
        """Вычисляет вес поезда БРУТТО."""
        # Прямые колонки
        for col in ("БРУТТО", "Вес БРУТТО", "Вес поезда БРУТТО"):
            weight = safe_float(row.get(col))
            if weight > 0:
                return weight
        
        # Расчет по ткм/км
        tkm_brutto = safe_float(row.get("Ткм брутто"))
        km = safe_float(row.get("Км"))
        
        if tkm_brutto > 0 and km > 0:
            return tkm_brutto / km
        
        return None
    
    def _add_route_points(self, fig: go.Figure, routes_df: pd.DataFrame, 
                        norm_functions: Dict, norm_types_used: Set[str]) -> None:
        """Добавляет точки маршрутов - ИСПРАВЛЕННАЯ ВЕРСИЯ."""
        status_colors = {
            "Экономия сильная": "darkgreen",
            "Экономия средняя": "green",
            "Экономия слабая": "lightgreen", 
            "Норма": "blue",
            "Перерасход слабый": "orange",
            "Перерасход средний": "darkorange",
            "Перерасход сильный": "red",
        }
        
        for status, color in status_colors.items():
            status_routes = routes_df[routes_df["Статус"] == status]
            if status_routes.empty:
                continue
            
            for norm_type in norm_types_used:
                x_vals, y_vals, texts, custom = [], [], [], []
                
                for _, row in status_routes.iterrows():
                    # ИСПРАВЛЕНО: Используем исходные значения участка
                    point_data = self._process_route_point(row, norm_functions, norm_type)
                    if not point_data:
                        continue
                    
                    x_vals.append(point_data["x"])
                    y_vals.append(point_data["y"])  # ← Теперь это исходный "Факт уд"
                    texts.append(point_data["hover"])
                    custom.append(point_data["custom_data"])
                
                if x_vals:
                    norm_type_suffix = f" ({norm_type})" if len(norm_types_used) > 1 else ""
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals, y=y_vals,
                            mode="markers", 
                            name=f"{status}{norm_type_suffix}",
                            marker=dict(color=color, size=6, opacity=0.7),
                            customdata=custom,
                            hovertemplate="%{text}<extra></extra>",
                            text=texts,
                        ),
                        row=1, col=1,
                    )
    
    def _add_status_points(self, fig: go.Figure, status_data: pd.DataFrame,
                          norm_functions: Dict, status_name: str, color: str) -> None:
        """Добавляет точки одного статуса."""
        x_vals, y_vals, texts, custom = [], [], [], []
        
        for _, row in status_data.iterrows():
            point_data = self._process_route_point(row, norm_functions)
            if not point_data:
                continue
            
            x_vals.append(point_data["x"])
            y_vals.append(point_data["y"])
            texts.append(point_data["hover"])
            custom.append(point_data["custom_data"])
        
        if x_vals:
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="markers",
                    name=f"{status_name} ({len(x_vals)})",
                    marker=dict(color=color, size=6, opacity=0.7),
                    customdata=custom,
                    hovertemplate="%{text}<extra></extra>",
                    text=texts,
                ),
                row=1, col=1,
            )
    
    def _process_route_point(self, row: pd.Series, norm_functions: Dict, 
                            target_norm_type: str = None) -> Optional[Dict]:
        """Обрабатывает одну точку маршрута для построения графика - ИСПРАВЛЕНО ДЛЯ СЕРИАЛИЗАЦИИ."""
        try:
            # 1. Проверяем наличие номера нормы
            norm_number = row.get("Номер нормы")
            if pd.isna(norm_number):
                logger.debug("Пропущена точка: отсутствует номер нормы")
                return None
            
            # 2. Получаем функцию интерполяции для данной нормы
            norm_str = str(safe_int(norm_number))
            norm_func_data = norm_functions.get(norm_str)
            if not norm_func_data:
                logger.debug("Пропущена точка: норма %s не найдена в функциях", norm_str)
                return None
            
            # 3. Проверяем соответствие типа нормы (если задан фильтр)
            norm_type = norm_func_data.get("norm_type", "Нажатие")
            if target_norm_type and norm_type != target_norm_type:
                logger.debug("Пропущена точка: тип нормы %s не соответствует фильтру %s", norm_type, target_norm_type)
                return None
            
            # 4. Вычисляем X координату (параметр нормирования)
            if norm_type == "Вес":
                x_val = self._calculate_weight_parameter(row)
                param_label = "Вес поезда БРУТТО"
            else:
                x_val = self._calculate_axle_load_parameter(row)
                param_label = "Нажатие на ось"
            
            if not x_val or x_val <= 0:
                logger.debug("Пропущена точка: невалидный параметр нормирования x=%.2f", x_val or 0)
                return None
            
            # 5. ИСПРАВЛЕНО: Получаем Y координату - ИСХОДНЫЙ "Факт уд" из участка
            y_val_fact_ud = safe_float(row.get("Факт уд"))
            if not y_val_fact_ud or y_val_fact_ud <= 0:
                logger.debug("Пропущена точка: невалидный 'Факт уд' = %.2f", y_val_fact_ud or 0)
                return None
            
            # 6. ИСПРАВЛЕНО: Получаем исходную норму участка для корректного hover
            y_val_norma = safe_float(row.get("Уд. норма, норма на 1 час ман. раб."))
            if not y_val_norma or y_val_norma <= 0:
                logger.debug("Предупреждение: невалидная 'Уд. норма' = %.2f для маршрута", y_val_norma or 0)
                y_val_norma = 0.0  # Не блокируем точку, но помечаем проблему
            
            # 7. Создаем полные данные для модального окна и переключателя режимов
            try:
                full_custom_data = self._build_full_custom_data_safe(row)
            except Exception as e:
                logger.error("Ошибка создания custom_data для маршрута %s: %s", row.get("Номер маршрута"), e)
                # Создаем минимальные данные для предотвращения краха
                full_custom_data = {
                    "route_number": str(row.get("Номер маршрута", "ERROR")),
                    "route_date": str(row.get("Дата маршрута", "ERROR")),
                    "error": "data_creation_failed",
                    "all_sections": [],
                    "rashod_fact": 0.0,
                    "rashod_norm": 0.0,
                    "rashod_fact_total": 0.0,
                    "rashod_norm_total": 0.0,
                    "ud_norma_original": float(y_val_norma),
                    "coefficient_route": 1.0,
                    "expected_nf_y": 0.0,
                    "coefficient_section": 1.0,
                    "fact_ud_original_section": float(y_val_fact_ud),
                    "norm_interpolated": 0.0,
                    "deviation_percent": 0.0,
                    "status": "N/A",
                    "n_equals_f": "N/A",
                    "debug_info": {"error": "failed"},
                    "use_red_rashod": False,
                    "totals": {}
                }
            
            # 8. Создаем hover-текст с правильными значениями
            hover_text = self._build_route_hover_corrected(
                row, norm_str, x_val, y_val_fact_ud, y_val_norma, param_label
            )
            
            # 9. ИСПРАВЛЕНО: Приводим координаты к нативным Python типам
            final_x = float(x_val)
            final_y = float(y_val_fact_ud)
            
            # 10. Финальная валидация координат
            if not isinstance(final_x, float) or not isinstance(final_y, float):
                logger.error("Некорректные типы координат: x=%s, y=%s", type(final_x), type(final_y))
                return None
            
            if not (0 < final_x < float('inf')) or not (0 < final_y < float('inf')):
                logger.error("Некорректные значения координат: x=%s, y=%s", final_x, final_y)
                return None
            
            # 11. Возвращаем данные точки
            return {
                "x": final_x,                    # X координата (параметр нормирования) 
                "y": final_y,                    # Y координата (исходный "Факт уд")
                "hover": hover_text,             # Текст при наведении мыши
                "custom_data": full_custom_data  # Полные данные для модального окна
            }
            
        except Exception as e:
            logger.error("Критическая ошибка в _process_route_point для маршрута %s: %s", 
                        row.get("Номер маршрута", "UNKNOWN"), e, exc_info=True)
            return None

    def _build_full_custom_data_safe(self, row: pd.Series) -> Dict:
        """Безопасная версия создания полных данных маршрута - ИСПРАВЛЕНО ДЛЯ СЕРИАЛИЗАЦИИ."""
        try:
            # Базовые данные маршрута
            route_number = str(row.get("Номер маршрута", "N/A"))
            route_date = str(row.get("Дата маршрута", "N/A"))
            
            # Данные текущего участка - ИСПРАВЛЕНО: приведение к Python типам
            fact_ud = float(safe_float(row.get("Факт уд"), 0.0))
            ud_norma = float(safe_float(row.get("Уд. норма, норма на 1 час ман. раб."), 0.0))
            rashod_fact = float(safe_float(row.get("Расход фактический"), 0.0))
            rashod_norm = float(safe_float(row.get("Расход по норме"), 0.0))
            
            # Попытка получить данные всего маршрута
            rashod_fact_total = rashod_fact  # По умолчанию = текущий участок
            rashod_norm_total = rashod_norm   # По умолчанию = текущий участок
            
            try:
                # Пытаемся получить суммарные данные по всему маршруту
                analyzer = getattr(self, '_analyzer', None)
                if analyzer and hasattr(analyzer, 'routes_df') and analyzer.routes_df is not None:
                    same_route = analyzer.routes_df[
                        (analyzer.routes_df['Номер маршрута'] == row.get("Номер маршрута")) &
                        (analyzer.routes_df['Дата маршрута'] == row.get("Дата маршрута"))
                    ]
                    if not same_route.empty:
                        total_fact = float(same_route['Расход фактический'].sum())
                        total_norm = float(same_route['Расход по норме'].sum())
                        if total_fact > 0 and total_norm > 0:
                            rashod_fact_total = total_fact
                            rashod_norm_total = total_norm
            except Exception:
                pass  # Используем данные текущего участка
            
            # Расчет коэффициентов - ИСПРАВЛЕНО: приведение к float
            coef_section = float(fact_ud / ud_norma if (fact_ud > 0 and ud_norma > 0) else 1.0)
            coef_route = float(rashod_fact_total / rashod_norm_total if (rashod_fact_total > 0 and rashod_norm_total > 0) else 1.0)
            expected_nf = float(coef_route * ud_norma if (coef_route > 0 and ud_norma > 0) else 0.0)
            
            # Простая структура участков - ИСПРАВЛЕНО: все значения приведены к сериализуемым типам
            sections = [
                {
                    "section_name": str(row.get("Наименование участка", "N/A")),
                    "netto": str(row.get("НЕТТО", "N/A")),
                    "brutto": str(row.get("БРУТТО", "N/A")),
                    "osi": str(row.get("ОСИ", "N/A")),
                    "norm_number": str(row.get("Номер нормы", "N/A")),
                    "movement_type": str(row.get("Дв. тяга", "N/A")),
                    "tkm_brutto": str(row.get("Ткм брутто", "N/A")),
                    "km": str(row.get("Км", "N/A")),
                    "pr": str(row.get("Пр.", "N/A")),
                    "rashod_fact": str(rashod_fact) if rashod_fact > 0 else "N/A",
                    "rashod_norm": str(rashod_norm) if rashod_norm > 0 else "N/A",
                    "fact_ud": str(fact_ud) if fact_ud > 0 else "N/A",
                    "ud_norma": str(ud_norma) if ud_norma > 0 else "N/A",
                    "axle_load": str(row.get("Нажатие на ось", "N/A")),
                    "norma_work": str(row.get("Норма на работу", "N/A")),
                    "fact_work": str(row.get("Факт на работу", "N/A")),
                    "norma_single": str(row.get("Норма на одиночное", "N/A")),
                    "idle_brigada_total": str(row.get("Простой с бригадой, мин., всего", "N/A")),
                    "idle_brigada_norm": str(row.get("Простой с бригадой, мин., норма", "N/A")),
                    "manevr_total": str(row.get("Маневры, мин., всего", "N/A")),
                    "manevr_norm": str(row.get("Маневры, мин., норма", "N/A")),
                    "start_total": str(row.get("Трогание с места, случ., всего", "N/A")),
                    "start_norm": str(row.get("Трогание с места, случ., норма", "N/A")),
                    "delay_total": str(row.get("Нагон опозданий, мин., всего", "N/A")),
                    "delay_norm": str(row.get("Нагон опозданий, мин., норма", "N/A")),
                    "speed_limit_total": str(row.get("Ограничения скорости, случ., всего", "N/A")),
                    "speed_limit_norm": str(row.get("Ограничения скорости, случ., норма", "N/A")),
                    "transfer_loco_total": str(row.get("На пересылаемые л-вы, всего", "N/A")),
                    "transfer_loco_norm": str(row.get("На пересылаемые л-вы, норма", "N/A")),
                    "duplicates_count": str(row.get("Количество дубликатов маршрута", "N/A")),
                    "use_red_color": bool(row.get('USE_RED_COLOR', False)),
                    "use_red_rashod": bool(row.get('USE_RED_RASHOD', False))
                }
            ]
            
            # ИСПРАВЛЕНО: все значения приведены к нативным Python типам для JSON сериализации
            return {
                # Основная информация
                "route_number": str(route_number),
                "route_date": str(route_date),
                "trip_date": str(row.get("Дата поездки", "N/A")),
                "driver_tab": str(row.get("Табельный машиниста", "N/A")),
                "locomotive_series": str(row.get("Серия локомотива", "N/A")),
                "locomotive_number": str(row.get("Номер локомотива", "N/A")),
                
                # Суммарные расходы для режима Н/Ф
                "rashod_fact_total": rashod_fact_total,
                "rashod_norm_total": rashod_norm_total,
                "rashod_fact": rashod_fact_total,  # Дублируем для совместимости
                "rashod_norm": rashod_norm_total,   # Дублируем для совместимости
                
                # Данные участка для режима Н/Ф
                "ud_norma_original": ud_norma,
                "fact_ud_original_section": fact_ud,
                
                # Коэффициенты
                "coefficient_section": coef_section,
                "coefficient_route": coef_route, 
                "expected_nf_y": expected_nf,
                
                # Анализ
                "norm_interpolated": float(safe_float(row.get("Норма интерполированная"), 0.0)),
                "deviation_percent": float(safe_float(row.get("Отклонение, %"), 0.0)),
                "status": str(row.get("Статус", "N/A")),
                "n_equals_f": str(row.get("Н=Ф", "N/A")),
                
                # Участки
                "all_sections": sections,
                
                # Отладка
                "debug_info": {
                    "fact_ud_current": fact_ud,
                    "ud_norma_current": ud_norma,
                    "rashod_fact_total": rashod_fact_total,
                    "rashod_norm_total": rashod_norm_total,
                },
                
                "use_red_rashod": bool(row.get("USE_RED_RASHOD", False)),
                "totals": {}
            }
            
        except Exception as e:
            logger.error("Ошибка в _build_full_custom_data_safe: %s", e)
            # Минимальные данные при ошибке - ИСПРАВЛЕНО: все значения сериализуемы
            return {
                "route_number": str(row.get("Номер маршрута", "ERROR")),
                "route_date": str(row.get("Дата маршрута", "ERROR")),
                "trip_date": "N/A",
                "driver_tab": "N/A", 
                "locomotive_series": "N/A",
                "locomotive_number": "N/A",
                "error": str(e),
                "all_sections": [],
                "rashod_fact": 0.0,
                "rashod_norm": 0.0,
                "rashod_fact_total": 0.0,
                "rashod_norm_total": 0.0,
                "ud_norma_original": 0.0,
                "coefficient_route": 1.0,
                "expected_nf_y": 0.0,
                "coefficient_section": 1.0,
                "fact_ud_original_section": 0.0,
                "norm_interpolated": 0.0,
                "deviation_percent": 0.0,
                "status": "N/A",
                "n_equals_f": "N/A",
                "debug_info": {"error": "failed"},
                "use_red_rashod": False,
                "totals": {}
            }

    def _build_route_hover_corrected(self, row: pd.Series, norm_str: str, 
                                    x_val: float, fact_ud: float, norma_ud: float, 
                                    param_label: str) -> str:
        """Создает ПРАВИЛЬНЫЙ hover-текст с исходными значениями участка."""
        try:
            route_number = row.get("Номер маршрута", "N/A")
            route_date = row.get("Дата маршрута", "N/A")
            section_name = row.get("Наименование участка", "N/A")
            series = row.get("Серия локомотива", "N/A")
            number = row.get("Номер локомотива", "N/A")
            
            # ИСПРАВЛЕНО: Показываем ИСХОДНЫЕ значения участка
            deviation = format_number(row.get("Отклонение, %"))
            rashod_fact = format_number(row.get("Расход фактический"))
            rashod_norm = format_number(row.get("Расход по норме"))
            
            return (
                f"<b>Маршрут №{route_number} | {route_date}</b><br>"
                f"Участок: {section_name}<br>"
                f"Локомотив: {series} №{number}<br>"
                f"{param_label}: {x_val:.1f}<br>"
                f"Факт: {format_number(fact_ud)}<br>"           # ← ИСХОДНЫЙ "Факт уд"
                f"Норма: {format_number(norma_ud)}<br>"         # ← ИСХОДНАЯ "Уд. норма"
                f"Расход фактический: {rashod_fact}<br>"
                f"Расход по норме: {rashod_norm}<br>"
                f"Отклонение: {deviation}%<br>"
                f"Номер нормы: {norm_str}"
            )
        except Exception as e:
            logger.error("Ошибка создания hover-текста: %s", e)
            return f"Ошибка отображения данных для маршрута {row.get('Номер маршрута', 'N/A')}"

    def _calculate_actual_specific_consumption_for_work(self, row: pd.Series) -> Optional[float]:
        """Вычисляет фактический удельный расход на работу."""
        # Базовый расход фактический
        rashod_fact = safe_float(row.get("Расход фактический"))
        if not rashod_fact:
            return None
        
        # Вычитаемые нормируемые составляющие
        deductions = [
            safe_float(row.get("Норма на одиночное")),
            safe_float(row.get("Простой с бригадой, мин., всего")),
            safe_float(row.get("Простой с бригадой, мин., норма")),
            safe_float(row.get("Маневры, мин., всего")),
            safe_float(row.get("Маневры, мин., норма")),
            safe_float(row.get("Трогание с места, случ., всего")),
            safe_float(row.get("Трогание с места, случ., норма")),
            safe_float(row.get("Нагон опозданий, мин., всего")),
            safe_float(row.get("Нагон опозданий, мин., норма")),
            safe_float(row.get("Ограничения скорости, случ., всего")),
            safe_float(row.get("Ограничения скорости, случ., норма")),
            safe_float(row.get("На пересылаемые л-вы, всего")),
            safe_float(row.get("На пересылаемые л-вы, норма")),
        ]
        
        # Вычисляем факт на работу
        fact_na_rabotu = rashod_fact - sum(deductions)
        
        # Переводим в удельные единицы (делим на ткм брутто / 10000)
        tkm_brutto = safe_float(row.get("Ткм брутто"))
        if tkm_brutto <= 0:
            return None
        
        tkm_10000 = tkm_brutto / 10000.0
        fact_udelnyj_na_rabotu = fact_na_rabotu / tkm_10000
        
        logger.debug("Расчет удельного на работу: rashod_fact=%.1f, deductions_sum=%.1f, fact_na_rabotu=%.1f, tkm_10000=%.1f, result=%.3f",
                    rashod_fact, sum(deductions), fact_na_rabotu, tkm_10000, fact_udelnyj_na_rabotu)
        
        return fact_udelnyj_na_rabotu if fact_udelnyj_na_rabotu > 0 else None   

    def _build_route_hover(self, row: pd.Series, norm_str: str, x_val: float, 
                        y_val: float, param_label: str) -> str:
        """Создает hover-текст для точки маршрута."""
        route_number = row.get("Номер маршрута", "N/A")
        route_date = row.get("Дата маршрута", "N/A")
        section_name = row.get("Наименование участка", "N/A")
        series = row.get("Серия локомотива", "N/A")
        number = row.get("Номер локомотива", "N/A")
        
        norm_interpolated = format_number(row.get("Норма интерполированная"))
        deviation = format_number(row.get("Отклонение, %"))
        rashod_fact = format_number(row.get("Расход фактический"))
        rashod_norm = format_number(row.get("Расход по норме"))
        
        return (
            f"<b>Маршрут №{route_number} | {route_date}</b><br>"
            f"Участок: {section_name}<br>"
            f"Локомотив: {series} №{number}<br>"
            f"{param_label}: {x_val:.1f}<br>"
            f"Факт: {format_number(y_val)}<br>"
            f"Норма: {norm_interpolated}<br>"
            f"Расход фактический: {rashod_fact}<br>"
            f"Расход по норме: {rashod_norm}<br>"
            f"Отклонение: {deviation}%<br>"
            f"Номер нормы: {norm_str}"
        )
    
    def _build_custom_data(self, row: pd.Series) -> Dict:
        """Создает данные для модального окна (упрощенная версия)."""
        return {
            "route_number": row.get("Номер маршрута", "N/A"),
            "route_date": row.get("Дата маршрута", "N/A"),
            "section_name": row.get("Наименование участка", "N/A"),
            "norm_interpolated": safe_float(row.get("Норма интерполированная")),
            "deviation_percent": safe_float(row.get("Отклонение, %")),
            "status": row.get("Статус", "N/A"),
        }
    
    def _add_deviation_analysis(self, fig: go.Figure, routes_df: pd.DataFrame) -> None:
        """Добавляет анализ отклонений на нижний график."""
        if routes_df.empty:
            return
        
        # Точки отклонений
        self._add_deviation_points(fig, routes_df)
        # Границы и зоны
        self._add_boundary_lines(fig, routes_df)
    
    def _add_deviation_points(self, fig: go.Figure, routes_df: pd.DataFrame) -> None:
        """Добавляет точки отклонений."""
        for status_name in ["Экономия сильная", "Экономия средняя", "Экономия слабая", 
                           "Норма", "Перерасход слабый", "Перерасход средний", "Перерасход сильный"]:
            status_data = routes_df[routes_df["Статус"] == status_name]
            if status_data.empty:
                continue
            
            color = StatusClassifier.get_status_color(status_name)
            
            x_vals = []
            y_vals = []
            
            for _, row in status_data.iterrows():
                axle_load = self._calculate_axle_load_parameter(row)
                deviation = safe_float(row.get("Отклонение, %"))
                
                if axle_load and axle_load > 0:
                    x_vals.append(axle_load)
                    y_vals.append(deviation)
            
            if x_vals:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_vals,
                        mode="markers",
                        name=f"{status_name} ({len(x_vals)})",
                        marker=dict(color=color, size=10, opacity=0.8,
                                   line=dict(color="black", width=0.5)),
                        showlegend=False,  # Не показываем в легенде дубли
                    ),
                    row=2, col=1,
                )
    
    def _add_boundary_lines(self, fig: go.Figure, routes_df: pd.DataFrame) -> None:
        """Добавляет граничные линии и зоны на нижний график."""
        # Определяем диапазон X
        x_vals = []
        for _, row in routes_df.iterrows():
            axle_load = self._calculate_axle_load_parameter(row)
            if axle_load and axle_load > 0:
                x_vals.append(axle_load)
        
        if not x_vals:
            return
        
        x_range = [min(x_vals) - 1, max(x_vals) + 1]
        
        # Граничные линии
        boundaries = [
            (5, "#FFD700", "dash", "Верхняя граница нормы (+5%)"),
            (-5, "#FFD700", "dash", "Нижняя граница нормы (-5%)"),
            (20, "#FF4500", "dot", "Граница значительного перерасхода (+20%)"),
            (-20, "#FF4500", "dot", "Граница значительной экономии (-20%)"),
            (30, "#DC143C", "dashdot", "Граница критического перерасхода (+30%)"),
            (-30, "#DC143C", "dashdot", "Граница критической экономии (-30%)"),
            (0, "black", "solid", "Идеальная норма (0%)"),
        ]
        
        for y_val, color, dash_style, name in boundaries:
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=[y_val, y_val],
                    mode="lines",
                    line=dict(color=color, dash=dash_style, width=2),
                    showlegend=False,
                    hovertemplate=f"{name}<extra></extra>",
                ),
                row=2, col=1,
            )
        
        # Зона нормы (-5% до +5%)
        fig.add_trace(
            go.Scatter(
                x=x_range + x_range[::-1],
                y=[-5, -5, 5, 5],
                fill="toself",
                fillcolor="rgba(255,215,0,0.1)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2, col=1,
        )
    
    def _configure_layout(self, fig: go.Figure, norm_types_used: Set[str]) -> None:
        """Настраивает оси и общий вид графика."""
        # Определяем название оси X
        if len(norm_types_used) > 1:
            x_title = "Параметр нормирования (т/ось или т БРУТТО)"
        elif "Вес" in norm_types_used:
            x_title = "Вес поезда БРУТТО, т"
        else:
            x_title = "Нажатие на ось, т/ось"
        
        # Настройка осей
        fig.update_xaxes(title_text=x_title, row=1, col=1)
        fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм", row=1, col=1)
        fig.update_xaxes(title_text=x_title, row=2, col=1)
        fig.update_yaxes(title_text="Отклонение, %", row=2, col=1)
        
        # Горизонтальная линия на нижнем графике
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Общий layout
        fig.update_layout(
            height=800,
            hovermode="closest",
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
        )
    
    def add_browser_controls(self, html_content: str) -> str:
        """Добавляет контролы и подключает JS к HTML - ИСПРАВЛЕННАЯ ВЕРСИЯ."""
        # Проверяем путь к JS файлу
        if not self.js_file_path.exists():
            logger.warning("JS файл не найден: %s", self.js_file_path)
            
        # ИСПРАВЛЕНО: правильное подключение JS
        js_content = ""
        try:
            js_content = self.js_file_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error("Ошибка чтения JS файла: %s", e)
            js_content = "console.error('Не удалось загрузить JS код');"
        
        controls_html = '''
        <div id="mode-switcher" style="
            position: fixed; top: 10px; right: 10px; z-index: 1000; background: white; padding: 15px;
            border: 2px solid #4a90e2; border-radius: 10px; box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            font-family: Arial, sans-serif; font-size: 14px;">
            <h4 style="margin: 0 0 12px 0; color: #333;">Режим отображения точек:</h4>
            <label style="display: block; margin-bottom: 8px; cursor: pointer;">
                <input type="radio" name="display_mode" value="work" checked style="margin-right: 8px;">
                <strong>Уд. на работу</strong> (текущий)
            </label>
            <label style="display: block; cursor: pointer;">
                <input type="radio" name="display_mode" value="nf" style="margin-right: 8px;">
                <strong>Н/Ф</strong> (по соотношению норма/факт)
            </label>
        </div>
        
        <div id="route-modal" style="display:none; position:fixed; z-index:2000; left:0; top:0; width:100%; height:100%;
            background-color:rgba(0,0,0,0.5);">
            <div id="modal-content" style="background-color:white; margin:2% auto; padding:20px; border-radius:10px;
                width:95%; max-width:1400px; max-height:90%; overflow-y:auto; position:relative;">
                <span id="close-modal" style="position:absolute; right:20px; top:15px; font-size:28px; font-weight:bold;
                    cursor:pointer; color:#aaa; user-select:none;">&times;</span>
                <div id="route-details"></div>
            </div>
        </div>
        '''
        
        # Встраиваем JS прямо в HTML
        js_embed = f"<script type='text/javascript'>\n{js_content}\n</script>"
        
        insertion_point = "</body>"
        if insertion_point in html_content:
            return html_content.replace(insertion_point, f"{controls_html}\n{js_embed}\n{insertion_point}")
        
        return html_content + f"\n{controls_html}\n{js_embed}"
    
    def _build_full_custom_data(self, row: pd.Series) -> Dict:
        """Упрощенная версия для отладки проблемы сериализации."""
        try:
            route_number = row.get("Номер маршрута", "N/A")
            route_date = row.get("Дата маршрута", "N/A")
            
            # Простые базовые данные без сложной обработки
            route_data = {
                "route_number": str(route_number),
                "route_date": str(route_date),
                "trip_date": str(row.get("Дата поездки", "N/A")),
                "driver_tab": str(row.get("Табельный машиниста", "N/A")),
                "locomotive_series": str(row.get("Серия локомотива", "N/A")),
                "locomotive_number": str(row.get("Номер локомотива", "N/A")),
                
                # Простые числовые значения
                "rashod_fact_total": float(row.get("Расход фактический", 0) or 0),
                "rashod_norm_total": float(row.get("Расход по норме", 0) or 0),
                "fact_ud_original_section": float(row.get("Факт уд", 0) or 0),
                "ud_norma_original": float(row.get("Уд. норма, норма на 1 час ман. раб.", 0) or 0),
                
                # Простые коэффициенты
                "coefficient_section": 1.0,
                "coefficient_route": 1.0,
                "expected_nf_y": 0.0,
                
                # Статус и отклонение
                "status": str(row.get("Статус", "N/A")),
                "deviation_percent": float(row.get("Отклонение, %", 0) or 0),
                "norm_interpolated": float(row.get("Норма интерполированная", 0) or 0),
                "n_equals_f": str(row.get("Н=Ф", "N/A")),
                
                # Минимальные данные участков
                "all_sections": [
                    {
                        "section_name": str(row.get("Наименование участка", "N/A")),
                        "rashod_fact": str(row.get("Расход фактический", "N/A")),
                        "rashod_norm": str(row.get("Расход по норме", "N/A")),
                        "fact_ud": str(row.get("Факт уд", "N/A")),
                        "ud_norma": str(row.get("Уд. норма, норма на 1 час ман. раб.", "N/A")),
                        "use_red_color": False,
                        "use_red_rashod": False
                    }
                ],
                
                # Простая отладочная информация
                "debug_info": {
                    "test": "simple_data_works"
                },
                
                "use_red_rashod": False,
                "totals": {}
            }
            
            logger.info("Созданы простые данные для маршрута %s", route_number)
            return route_data
            
        except Exception as e:
            logger.error("Ошибка в упрощенной версии _build_full_custom_data: %s", e, exc_info=True)
            return {
                "route_number": "ERROR",
                "error": str(e),
                "all_sections": [],
                "debug_info": {"error": "function_failed"}
            }