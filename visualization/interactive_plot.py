# visualization/interactive_plot.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Интерактивный график на основе matplotlib для анализа норм."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

from core.utils import StatusClassifier, safe_float, format_number
from .plot_modes import PlotModeManager, DisplayMode

logger = logging.getLogger(__name__)


class InteractivePlot:
    """
    Высокопроизводительный интерактивный график с возможностью переключения режимов.
    Использует matplotlib для встроенного отображения в tkinter приложении.
    """
    
    def __init__(self, figure: Figure):
        self.figure = figure
        self.mode_manager = PlotModeManager()
        
        # Состояние графика
        self._traces_data: Dict[str, Dict] = {}  # Данные трасс
        self._scatter_objects: Dict[str, plt.Artist] = {}  # matplotlib объекты для обновления
        self._norm_lines: List[plt.Artist] = []  # Линии норм
        self._norm_points: List[plt.Artist] = []  # Точки норм
        
        # Настройки внешнего вида
        self._status_colors = {
            "Экономия сильная": "#006400",    # darkgreen
            "Экономия средняя": "#008000",     # green
            "Экономия слабая": "#90EE90",      # lightgreen
            "Норма": "#0000FF",                # blue
            "Перерасход слабый": "#FFA500",    # orange
            "Перерасход средний": "#FF8C00",   # darkorange
            "Перерасход сильный": "#FF0000",   # red
        }
        
        # Создаем подграфики
        self.ax1 = None  # Верхний график (нормы и точки)
        self.ax2 = None  # Нижний график (отклонения)
        self._setup_subplots()
        
        # Подключаем обработчики событий
        self._setup_event_handlers()
        
    def _setup_subplots(self) -> None:
        """Создает структуру подграфиков."""
        self.figure.clear()
        
        # Создаем два подграфика с общей осью X
        self.ax1 = self.figure.add_subplot(2, 1, 1)
        self.ax2 = self.figure.add_subplot(2, 1, 2, sharex=self.ax1)
        
        # Настройка верхнего графика
        self.ax1.set_ylabel('Удельный расход, кВт·ч/10⁴ ткм', fontsize=10)
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_title('Нормы расхода и фактические данные', fontsize=12, fontweight='bold')
        
        # Настройка нижнего графика  
        self.ax2.set_ylabel('Отклонение, %', fontsize=10)
        self.ax2.set_xlabel('Параметр нормирования', fontsize=10)
        self.ax2.grid(True, alpha=0.3)
        self.ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)  # Линия нормы
        
        # Добавляем зоны отклонений на нижний график
        self._add_deviation_zones()
        
        plt.subplots_adjust(hspace=0.3)  # Расстояние между графиками
        
    def _add_deviation_zones(self) -> None:
        """Добавляет цветные зоны отклонений на нижний график."""
        # Зона нормы (-5% до +5%)
        self.ax2.axhspan(-5, 5, alpha=0.1, color='gold', label='Норма (±5%)')
        
        # Границы значительных отклонений
        self.ax2.axhline(y=20, color='orange', linestyle=':', alpha=0.7, label='Значительный перерасход (+20%)')
        self.ax2.axhline(y=-20, color='orange', linestyle=':', alpha=0.7, label='Значительная экономия (-20%)')
        
        # Границы критических отклонений  
        self.ax2.axhline(y=30, color='red', linestyle='-.', alpha=0.7, label='Критический перерасход (+30%)')
        self.ax2.axhline(y=-30, color='red', linestyle='-.', alpha=0.7, label='Критическая экономия (-30%)')
        
    def _setup_event_handlers(self) -> None:
        """Настраивает обработчики событий для интерактивности."""
        # Подключаем обработчик кликов для показа деталей маршрута
        self.figure.canvas.mpl_connect('button_press_event', self._on_click)
        
        # Подключаем обработчик для hover эффектов
        self.figure.canvas.mpl_connect('motion_notify_event', self._on_hover)
        
    def create_plot(
        self, 
        section_name: str,
        routes_df: pd.DataFrame,
        norm_functions: Dict,
        specific_norm_id: Optional[str] = None,
        single_section_only: bool = False
    ) -> None:
        """
        Создает интерактивный график анализа участка.
        
        Args:
            section_name: Название участка
            routes_df: DataFrame с данными маршрутов  
            norm_functions: Функции интерполяции норм
            specific_norm_id: ID конкретной нормы (если выбрана)
            single_section_only: Только маршруты с одним участком
        """
        logger.info("Создание графика для участка: %s", section_name)
        
        # Очищаем предыдущие данные
        self._clear_plot_data()
        
        # Определяем заголовок
        title_suffix = f" (норма {specific_norm_id})" if specific_norm_id else ""
        filter_suffix = " [только один участок]" if single_section_only else ""
        title = f"Анализ участка: {section_name}{title_suffix}{filter_suffix}"
        
        self.ax1.set_title(title, fontsize=12, fontweight='bold')
        
        # Определяем типы норм для правильной подписи осей
        norm_types_used = self._get_norm_types_used(norm_functions)
        x_label = self._get_x_axis_label(norm_types_used)
        self.ax2.set_xlabel(x_label, fontsize=10)
        
        # Добавляем кривые норм
        self._add_norm_curves(norm_functions, routes_df, specific_norm_id)
        
        # Добавляем точки маршрутов
        self._add_route_points(routes_df, norm_functions)
        
        # Добавляем анализ отклонений на нижний график
        self._add_deviation_points(routes_df)
        
        # Настраиваем легенду
        self._setup_legend()
        
        # Автоматическое масштабирование
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.ax2.relim() 
        self.ax2.autoscale_view()
        
        # Обновляем отображение
        self.figure.canvas.draw_idle()
        
        logger.info("График создан успешно")
        
    def _clear_plot_data(self) -> None:
        """Очищает данные предыдущего графика."""
        self.ax1.clear()
        self.ax2.clear()
        self._setup_subplots()
        
        self._traces_data.clear()
        self._scatter_objects.clear()
        self._norm_lines.clear()
        self._norm_points.clear()
        
    def _get_norm_types_used(self, norm_functions: Dict) -> Set[str]:
        """Определяет типы норм, используемые в анализе."""
        return {nf.get("norm_type", "Нажатие") for nf in norm_functions.values()}
        
    def _get_x_axis_label(self, norm_types_used: Set[str]) -> str:
        """Возвращает подпись для оси X в зависимости от типов норм."""
        if len(norm_types_used) > 1:
            return "Параметр нормирования (т/ось или т БРУТТО)"
        elif "Вес" in norm_types_used:
            return "Вес поезда БРУТТО, т"
        else:
            return "Нажатие на ось, т/ось"
            
    def _add_norm_curves(self, norm_functions: Dict, routes_df: pd.DataFrame, 
                        specific_norm_id: Optional[str]) -> None:
        """Добавляет кривые норм на верхний график."""
        for norm_id, norm_data in norm_functions.items():
            if specific_norm_id and norm_id != specific_norm_id:
                continue
                
            all_points = norm_data.get("points", [])
            base_points = norm_data.get("base_points", [])
            additional_points = norm_data.get("additional_points", [])
            
            if not all_points:
                continue
                
            # Добавляем базовые точки нормы (синие квадраты)
            if base_points:
                self._add_base_norm_points(norm_id, base_points)
                
            # Добавляем дополнительные точки из маршрутов (желтые квадраты)
            if additional_points:
                self._add_additional_norm_points(norm_id, additional_points)
                
            # Добавляем интерполированную кривую
            if len(all_points) > 1:
                self._add_norm_curve_line(norm_id, all_points, routes_df)
            elif len(all_points) == 1:
                self._add_constant_norm_line(norm_id, all_points[0], routes_df)
                
    def _add_base_norm_points(self, norm_id: str, base_points: List[Tuple[float, float]]) -> None:
        """Добавляет базовые точки норм (синие квадраты)."""
        if not base_points:
            return
            
        x_vals = [p[0] for p in base_points]
        y_vals = [p[1] for p in base_points]
        
        scatter = self.ax1.scatter(
            x_vals, y_vals,
            marker='s',  # квадраты
            s=64,        # размер
            c='blue',
            alpha=0.9,
            edgecolor='darkblue',
            linewidth=1,
            label=f'Базовые точки нормы {norm_id} ({len(base_points)})',
            zorder=5
        )
        
        self._norm_points.append(scatter)
        
    def _add_additional_norm_points(self, norm_id: str, additional_points: List[Tuple[float, float]]) -> None:
        """Добавляет дополнительные точки норм из маршрутов (желтые квадраты)."""
        if not additional_points:
            return
            
        x_vals = [p[0] for p in additional_points]
        y_vals = [p[1] for p in additional_points]
        
        scatter = self.ax1.scatter(
            x_vals, y_vals,
            marker='s',  # квадраты
            s=64,        # размер
            c='gold',
            alpha=0.9,
            edgecolor='orange',
            linewidth=1,
            label=f'Из маршрутов {norm_id} ({len(additional_points)})',
            zorder=5
        )
        
        self._norm_points.append(scatter)
        
    def _add_norm_curve_line(self, norm_id: str, all_points: List[Tuple[float, float]], 
                            routes_df: pd.DataFrame) -> None:
        """Добавляет интерполированную кривую нормы."""
        if len(all_points) < 2:
            return
            
        # Сортируем точки по X
        sorted_points = sorted(all_points, key=lambda p: p[0])
        x_vals = [p[0] for p in sorted_points]
        y_vals = [p[1] for p in sorted_points]
        
        # Определяем диапазон для интерполяции
        x_range = self._calculate_interpolation_range(x_vals, routes_df, norm_id)
        x_interp = np.linspace(x_range[0], x_range[1], 500)
        y_interp = np.interp(x_interp, x_vals, y_vals)
        
        line, = self.ax1.plot(
            x_interp, y_interp,
            color='blue',
            linewidth=3,
            label=f'Норма {norm_id} ({len(all_points)} точек)',
            zorder=3
        )
        
        self._norm_lines.append(line)
        
    def _add_constant_norm_line(self, norm_id: str, point: Tuple[float, float], 
                               routes_df: pd.DataFrame) -> None:
        """Добавляет константную линию нормы (одна точка)."""
        x_single, y_single = point
        
        # Определяем диапазон для константной линии
        x_vals_from_data = self._get_route_x_values(routes_df, norm_id)
        if x_vals_from_data:
            x_min, x_max = min(x_vals_from_data), max(x_vals_from_data)
            x_range_size = max(x_max - x_min, 1.0)
            x_start = max(x_min - x_range_size * 0.2, x_min * 0.8)
            x_end = x_max + x_range_size * 0.2
        else:
            x_start = max(x_single * 0.5, x_single - 100)
            x_end = x_single * 1.5 + 100
            
        line, = self.ax1.plot(
            [x_start, x_end], [y_single, y_single],
            color='blue',
            linewidth=3,
            linestyle='-',
            label=f'Норма {norm_id} (константа)',
            zorder=3
        )
        
        self._norm_lines.append(line)
        
    def _calculate_interpolation_range(self, x_vals: List[float], routes_df: pd.DataFrame, 
                                     norm_id: str) -> Tuple[float, float]:
        """Вычисляет диапазон для интерполяции кривой нормы."""
        x_min, x_max = min(x_vals), max(x_vals)
        x_range = max(x_max - x_min, 1.0)
        
        # Базовый диапазон
        x_start = max(x_min - x_range * 0.3, x_min * 0.5)
        x_end = x_max + x_range * 0.3
        
        # Расширяем под данные маршрутов
        route_x_vals = self._get_route_x_values(routes_df, norm_id)
        if route_x_vals:
            x_start = min(x_start, min(route_x_vals) * 0.8)
            x_end = max(x_end, max(route_x_vals) * 1.2)
            
        return x_start, x_end
        
    def _get_route_x_values(self, routes_df: pd.DataFrame, norm_id: str) -> List[float]:
        """Получает X-координаты из данных маршрутов для конкретной нормы."""
        # Простая реализация - можно расширить при необходимости
        x_values = []
        norm_routes = routes_df[routes_df["Номер нормы"].astype(str) == str(norm_id)]
        
        for _, row in norm_routes.iterrows():
            x_val = self._calculate_x_parameter(row)
            if x_val and x_val > 0:
                x_values.append(x_val)
                
        return x_values
        
    def _add_route_points(self, routes_df: pd.DataFrame, norm_functions: Dict) -> None:
        """Добавляет точки маршрутов на верхний график."""
        # Группируем данные по статусам для эффективного отображения
        for status_name, color in self._status_colors.items():
            status_routes = routes_df[routes_df["Статус"] == status_name]
            if status_routes.empty:
                continue
                
            x_vals, y_vals, metadata = [], [], []
            
            for _, row in status_routes.iterrows():
                point_data = self._process_route_point(row, norm_functions)
                if not point_data:
                    continue
                    
                x_vals.append(point_data["x"])
                y_vals.append(point_data["y"])
                metadata.append(point_data["metadata"])
                
            if x_vals:
                # Создаем scatter plot для группы
                scatter = self.ax1.scatter(
                    x_vals, y_vals,
                    c=color,
                    s=36,  # размер точек
                    alpha=0.7,
                    edgecolor='black',
                    linewidth=0.5,
                    label=f'{status_name} ({len(x_vals)})',
                    zorder=4
                )
                
                # Сохраняем данные для переключения режимов и интерактивности
                trace_name = status_name
                self._traces_data[trace_name] = {
                    'x': np.array(x_vals),
                    'y': np.array(y_vals),
                    'routes_df': status_routes,
                    'metadata': metadata
                }
                self._scatter_objects[trace_name] = scatter
                
        # Передаем данные в менеджер режимов
        self.mode_manager.set_original_data(self._traces_data)
        
    def _process_route_point(self, row: pd.Series, norm_functions: Dict) -> Optional[Dict]:
        """Обрабатывает одну точку маршрута для отображения."""
        try:
            # Вычисляем X координату (параметр нормирования)
            x_val = self._calculate_x_parameter(row)
            if not x_val or x_val <= 0:
                return None
                
            # Получаем Y координату (факт удельный)
            y_val = safe_float(row.get("Факт уд"))
            if not y_val or y_val <= 0:
                return None
                
            # Создаем метаданные для интерактивности
            metadata = {
                'route_number': row.get("Номер маршрута", "N/A"),
                'route_date': row.get("Дата маршрута", "N/A"),
                'section_name': row.get("Наименование участка", "N/A"),
                'locomotive_series': row.get("Серия локомотива", "N/A"),
                'locomotive_number': row.get("Номер локомотива", "N/A"),
                'deviation_percent': safe_float(row.get("Отклонение, %")),
                'status': row.get("Статус", "N/A"),
                'norm_interpolated': safe_float(row.get("Норма интерполированная")),
                'rashod_fact': safe_float(row.get("Расход фактический")),
                'rashod_norm': safe_float(row.get("Расход по норме")),
                # Данные для режима Н/Ф
                'rashod_fact_total': safe_float(row.get("Расход фактический")),
                'rashod_norm_total': safe_float(row.get("Расход по норме")),
                'ud_norma_original': safe_float(row.get("Уд. норма, норма на 1 час ман. раб.")),
                'expected_nf_y': self._calculate_nf_y_value(row, y_val)
            }
            
            return {
                "x": float(x_val),
                "y": float(y_val), 
                "metadata": metadata
            }
            
        except Exception as e:
            logger.debug("Ошибка обработки точки маршрута: %s", e)
            return None
            
    def _calculate_x_parameter(self, row: pd.Series) -> Optional[float]:
        """Вычисляет параметр нормирования (X координату)."""
        # Сначала пытаемся получить готовое значение нажатия на ось
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
            return tkm_brutto / km  # Вес поезда
            
        return None
        
    def _calculate_nf_y_value(self, row: pd.Series, original_y: float) -> float:
        """Вычисляет Y координату для режима Н/Ф."""
        try:
            # Пытаемся получить предрасчитанное значение
            expected_nf = safe_float(row.get('expected_nf_y'))
            if expected_nf > 0:
                return expected_nf
                
            # Расчет по формуле: (Расх.факт / Расх.норма) * Уд.норма
            rashod_fact = safe_float(row.get('Расход фактический'))
            rashod_norm = safe_float(row.get('Расход по норме'))  
            ud_norma = safe_float(row.get('Уд. норма, норма на 1 час ман. раб.'))
            
            if rashod_fact > 0 and rashod_norm > 0 and ud_norma > 0:
                coefficient = rashod_fact / rashod_norm
                return coefficient * ud_norma
                
            return original_y
            
        except Exception:
            return original_y
            
    def _add_deviation_points(self, routes_df: pd.DataFrame) -> None:
        """Добавляет точки отклонений на нижний график."""
        for status_name, color in self._status_colors.items():
            status_data = routes_df[routes_df["Статус"] == status_name]
            if status_data.empty:
                continue
                
            x_vals, y_vals = [], []
            
            for _, row in status_data.iterrows():
                x_val = self._calculate_x_parameter(row)
                y_val = safe_float(row.get("Отклонение, %"))
                
                if x_val and x_val > 0 and y_val is not None:
                    x_vals.append(x_val)
                    y_vals.append(y_val)
                    
            if x_vals:
                self.ax2.scatter(
                    x_vals, y_vals,
                    c=color,
                    s=64,  # больше размер для нижнего графика
                    alpha=0.8,
                    edgecolor='black',
                    linewidth=0.5,
                    zorder=4
                )
                
    def _setup_legend(self) -> None:
        """Настраивает легенду для обоих графиков."""
        # Легенда для верхнего графика
        handles1, labels1 = self.ax1.get_legend_handles_labels()
        if handles1:
            self.ax1.legend(
                handles1, labels1,
                loc='upper left',
                bbox_to_anchor=(1.02, 1),
                fontsize=9,
                frameon=True,
                fancybox=True,
                shadow=True
            )
            
    def switch_display_mode(self, mode: DisplayMode) -> None:
        """
        Переключает режим отображения точек маршрутов.
        
        Args:
            mode: Новый режим отображения
        """
        logger.info("Переключение режима отображения на: %s", mode.value)
        
        # Получаем новые Y координаты от менеджера режимов
        new_y_data = self.mode_manager.switch_mode(mode)
        
        if not new_y_data:
            logger.warning("Нет данных для переключения режима")
            return
            
        # Обновляем Y координаты scatter объектов
        for trace_name, new_y in new_y_data.items():
            scatter_obj = self._scatter_objects.get(trace_name)
            if scatter_obj is None:
                logger.debug("Scatter объект для %s не найден", trace_name)
                continue
                
            # Получаем текущие координаты
            offsets = scatter_obj.get_offsets()
            if len(offsets) != len(new_y):
                logger.warning("Несоответствие количества точек для %s: %d vs %d", 
                             trace_name, len(offsets), len(new_y))
                continue
                
            # Обновляем Y координаты, сохраняя X
            new_offsets = np.column_stack([offsets[:, 0], new_y])
            scatter_obj.set_offsets(new_offsets)
            
        # Обновляем заголовок с указанием текущего режима
        current_title = self.ax1.get_title()
        mode_label = self.mode_manager.get_mode_label(mode)
        
        if " | Режим:" in current_title:
            title_base = current_title.split(" | Режим:")[0]
        else:
            title_base = current_title
            
        new_title = f"{title_base} | Режим: {mode_label}"
        self.ax1.set_title(new_title, fontsize=12, fontweight='bold')
        
        # Автоматически подгоняем масштаб оси Y верхнего графика
        self.ax1.relim()
        self.ax1.autoscale_view()
        
        # Перерисовываем график
        self.figure.canvas.draw_idle()
        
        logger.info("Режим отображения переключен успешно")
        
    def _on_click(self, event) -> None:
        """Обработчик клика по графику для показа деталей точки."""
        if event.inaxes != self.ax1:  # Реагируем только на верхний график
            return
            
        # Находим ближайшую точку к клику
        closest_point = self._find_closest_point(event.xdata, event.ydata)
        if closest_point:
            self._show_route_details(closest_point)
            
    def _find_closest_point(self, click_x: float, click_y: float) -> Optional[Dict]:
        """Находит ближайшую точку к координатам клика."""
        if not click_x or not click_y:
            return None
            
        min_distance = float('inf')
        closest_point = None
        
        for trace_name, data in self._traces_data.items():
            x_vals = data['x']
            y_vals = self._get_current_y_values(trace_name)  # Учитываем текущий режим
            metadata_list = data['metadata']
            
            for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                # Нормализованное расстояние (учитываем масштаб осей)
                x_range = self.ax1.get_xlim()[1] - self.ax1.get_xlim()[0]
                y_range = self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0]
                
                norm_dist = ((click_x - x) / x_range) ** 2 + ((click_y - y) / y_range) ** 2
                
                if norm_dist < min_distance:
                    min_distance = norm_dist
                    if i < len(metadata_list):
                        closest_point = metadata_list[i].copy()
                        closest_point['current_y'] = y  # Добавляем текущую Y координату
                        
        # Считаем точку близкой, если расстояние меньше порога
        if min_distance < 0.001:  # Пороговое значение
            return closest_point
            
        return None
        
    def _get_current_y_values(self, trace_name: str) -> np.ndarray:
        """Получает текущие Y координаты для трассы с учетом режима отображения."""
        current_y_data = self.mode_manager._get_current_y_data()
        return current_y_data.get(trace_name, self._traces_data[trace_name]['y'])
        
    def _show_route_details(self, point_metadata: Dict) -> None:
        """Показывает детальную информацию о маршруте в отдельном окне."""
        from tkinter import messagebox
        
        # Формируем детальную информацию
        details = [
            f"Маршрут №{point_metadata['route_number']} | {point_metadata['route_date']}",
            f"Участок: {point_metadata['section_name']}",
            f"Локомотив: {point_metadata['locomotive_series']} №{point_metadata['locomotive_number']}",
            "",
            f"Текущий Y: {format_number(point_metadata.get('current_y', 0))}",
            f"Норма интерпол.: {format_number(point_metadata['norm_interpolated'])}",
            f"Отклонение: {format_number(point_metadata['deviation_percent'])}%",
            f"Статус: {point_metadata['status']}",
            "",
            f"Расход фактический: {format_number(point_metadata['rashod_fact'])}",
            f"Расход по норме: {format_number(point_metadata['rashod_norm'])}"
        ]
        
        messagebox.showinfo(
            f"Детали маршрута №{point_metadata['route_number']}", 
            "\n".join(details)
        )
        
    def _on_hover(self, event) -> None:
        """Обработчик движения мыши для hover эффектов (опционально)."""
        # Можно добавить подсветку точек при наведении
        pass
        
    def export_plot(self, filename: str, dpi: int = 300) -> bool:
        """
        Экспортирует график в файл.
        
        Args:
            filename: Путь к файлу для сохранения
            dpi: Разрешение изображения
            
        Returns:
            True если успешно сохранено
        """
        try:
            self.figure.savefig(
                filename, 
                dpi=dpi, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )
            logger.info("График экспортирован в %s", filename)
            return True
            
        except Exception as e:
            logger.error("Ошибка экспорта графика: %s", e)
            return False