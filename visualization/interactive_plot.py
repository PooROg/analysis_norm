# visualization/interactive_plot.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Исправленный интерактивный график на основе matplotlib для анализа норм."""

import logging
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np
import pandas as pd
import traceback

import matplotlib
matplotlib.use('TkAgg')  # Обязательно устанавливаем backend до импортов
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from core.utils import StatusClassifier, safe_float, format_number
from .plot_modes import PlotModeManager, DisplayMode

logger = logging.getLogger(__name__)


class InteractivePlot:
    """
    Исправленный интерактивный график с синхронным созданием.
    Убраны threading операции для предотвращения зависаний.
    """
    
    def __init__(self, figure: Figure):
        self.figure = figure
        self.mode_manager = PlotModeManager()
        
        # Состояние графика
        self._traces_data: Dict[str, Dict] = {}
        self._scatter_objects: Dict[str, Any] = {}  # matplotlib scatter objects
        self._norm_lines: List[Any] = []  # Line2D objects
        self._norm_points: List[Any] = []  # PathCollection objects
        
        # Цвета статусов - оптимизированные для видимости
        self._status_colors = {
            "Экономия сильная": "#006400",    # darkgreen
            "Экономия средняя": "#32CD32",     # limegreen  
            "Экономия слабая": "#90EE90",      # lightgreen
            "Норма": "#0000FF",                # blue
            "Перерасход слабый": "#FFA500",    # orange
            "Перерасход средний": "#FF8C00",   # darkorange
            "Перерасход сильный": "#DC143C",   # crimson
        }
        
        # Subplot'ы - инициализируем сразу
        self.ax1 = None  # Верхний график (нормы и точки)
        self.ax2 = None  # Нижний график (отклонения)
        
        # Инициализируем структуру немедленно
        self._initialize_plots()
        
        logger.info("InteractivePlot инициализирован успешно")
        
    def _initialize_plots(self) -> None:
        """Безопасная инициализация subplot'ов с error handling."""
        try:
            self.figure.clear()  # Очищаем figure полностью
            
            # Создаем subplot'ы с правильными пропорциями
            gs = self.figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)
            self.ax1 = self.figure.add_subplot(gs[0, 0])
            self.ax2 = self.figure.add_subplot(gs[1, 0], sharex=self.ax1)
            
            # Базовая настройка верхнего графика
            self.ax1.set_ylabel('Удельный расход, кВт·ч/10⁴ ткм', fontsize=11)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.set_title('Готов к построению графика', fontsize=12, pad=20)
            
            # Базовая настройка нижнего графика  
            self.ax2.set_ylabel('Отклонение, %', fontsize=11)
            self.ax2.set_xlabel('Параметр нормирования', fontsize=11)
            self.ax2.grid(True, alpha=0.3)
            
            # Настройки для лучшей читаемости
            for ax in [self.ax1, self.ax2]:
                ax.tick_params(labelsize=10)
                ax.spines['top'].set_linewidth(0.5)
                ax.spines['right'].set_linewidth(0.5)
                ax.spines['bottom'].set_linewidth(0.5)
                ax.spines['left'].set_linewidth(0.5)
            
            logger.debug("Subplot'ы инициализированы успешно")
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка инициализации subplot'ов: %s", e, exc_info=True)
            raise RuntimeError(f"Не удалось инициализировать графики: {e}")
        
    def create_plot(
        self, 
        section_name: str,
        routes_df: pd.DataFrame,
        norm_functions: Dict,
        specific_norm_id: Optional[str] = None,
        single_section_only: bool = False
    ) -> None:
        """
        ИСПРАВЛЕННЫЙ метод создания графика - БЕЗ THREADING.
        Все операции выполняются синхронно в основном потоке для стабильности.
        """
        logger.info("=== НАЧАЛО СОЗДАНИЯ ГРАФИКА ===")
        logger.info("Участок: %s | Норма: %s | Один участок: %s | Записей: %d", 
                   section_name, specific_norm_id or "Все", single_section_only, len(routes_df))
        
        try:
            # ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
            if routes_df is None or routes_df.empty:
                raise ValueError("DataFrame с маршрутами пуст или None")
            if not norm_functions:
                raise ValueError("Функции норм не переданы")
                
            # ПОЛНАЯ ОЧИСТКА ПРЕДЫДУЩИХ ДАННЫХ
            self._clear_all_plot_data()
            
            # ЗАГОЛОВОК ГРАФИКА
            title_parts = [f"Участок: {section_name}"]
            if specific_norm_id:
                title_parts.append(f"норма {specific_norm_id}")
            if single_section_only:
                title_parts.append("[только один участок]")
            
            full_title = " | ".join(title_parts)
            self.ax1.set_title(full_title, fontsize=12, fontweight='bold', pad=20)
            
            # ОПРЕДЕЛЕНИЕ ТИПОВ НОРМ ДЛЯ ПОДПИСЕЙ ОСЕЙ
            norm_types_used = self._get_norm_types_used(norm_functions)
            x_label = self._get_x_axis_label(norm_types_used)
            self.ax2.set_xlabel(x_label, fontsize=11)
            
            logger.info("Найдено типов норм: %s", list(norm_types_used))
            
            # ПОСЛЕДОВАТЕЛЬНОЕ СОЗДАНИЕ ЭЛЕМЕНТОВ ГРАФИКА
            # 1. Кривые и точки норм
            self._add_norm_curves_safe(norm_functions, routes_df, specific_norm_id)
            
            # 2. Точки маршрутов 
            self._add_route_points_safe(routes_df, norm_functions)
            
            # 3. Анализ отклонений на нижнем графике
            self._add_deviation_analysis_safe(routes_df)
            
            # 4. Зоны отклонений
            self._add_deviation_zones()
            
            # НАСТРОЙКА ИНТЕРАКТИВНОСТИ
            self._setup_event_handlers_safe()
            
            # НАСТРОЙКА ЛЕГЕНДЫ И ОСЕЙ
            self._setup_final_layout()
            
            # ПЕРЕДАЧА ДАННЫХ В МЕНЕДЖЕР РЕЖИМОВ
            self.mode_manager.set_original_data(self._traces_data)
            
            logger.info("=== ГРАФИК СОЗДАН УСПЕШНО ===")
            logger.info("Элементы: линий норм=%d, точек норм=%d, трасс данных=%d", 
                       len(self._norm_lines), len(self._norm_points), len(self._traces_data))
            
        except Exception as e:
            logger.error("=== КРИТИЧЕСКАЯ ОШИБКА СОЗДАНИЯ ГРАФИКА ===")
            logger.error("Ошибка: %s", e, exc_info=True)
            self._show_error_plot(f"Ошибка создания графика: {str(e)}")
            raise
        
    def _clear_all_plot_data(self) -> None:
        """Полная очистка всех данных графика."""
        try:
            # Очищаем axes
            if self.ax1:
                self.ax1.clear()
            if self.ax2:
                self.ax2.clear()
                
            # Очищаем коллекции данных
            self._traces_data.clear()
            self._scatter_objects.clear()
            self._norm_lines.clear()
            self._norm_points.clear()
            
            # Переинициализируем subplot'ы
            self._initialize_plots()
            
            logger.debug("Данные графика полностью очищены")
            
        except Exception as e:
            logger.error("Ошибка очистки графика: %s", e)
            # Пытаемся принудительно переинициализировать
            self._initialize_plots()
        
    def _get_norm_types_used(self, norm_functions: Dict) -> Set[str]:
        """Безопасно определяет типы норм."""
        types = set()
        for norm_data in norm_functions.values():
            if isinstance(norm_data, dict):
                norm_type = norm_data.get("norm_type", "Нажатие")
                types.add(str(norm_type))
        return types if types else {"Нажатие"}  # fallback
        
    def _get_x_axis_label(self, norm_types_used: Set[str]) -> str:
        """Возвращает подпись для оси X."""
        if len(norm_types_used) > 1:
            return "Параметр нормирования (т/ось или т БРУТТО)"
        elif "Вес" in norm_types_used:
            return "Вес поезда БРУТТО, т"
        else:
            return "Нажатие на ось, т/ось"
            
    def _add_norm_curves_safe(self, norm_functions: Dict, routes_df: pd.DataFrame, 
                             specific_norm_id: Optional[str]) -> None:
        """Безопасно добавляет кривые норм с полной обработкой ошибок."""
        logger.info("Добавление кривых норм: %d функций", len(norm_functions))
        
        added_curves = 0
        for norm_id, norm_data in norm_functions.items():
            try:
                # Фильтр по конкретной норме
                if specific_norm_id and str(norm_id) != str(specific_norm_id):
                    continue
                    
                if not isinstance(norm_data, dict):
                    logger.warning("Данные нормы %s не являются словарем", norm_id)
                    continue
                    
                all_points = norm_data.get("points", [])
                base_points = norm_data.get("base_points", [])
                additional_points = norm_data.get("additional_points", [])
                
                if not all_points:
                    logger.warning("Норма %s не содержит точек", norm_id)
                    continue
                    
                # Добавляем базовые точки (синие квадраты)
                if base_points:
                    self._add_base_norm_points_safe(norm_id, base_points)
                    
                # Добавляем дополнительные точки (золотые квадраты)  
                if additional_points:
                    self._add_additional_norm_points_safe(norm_id, additional_points)
                    
                # Добавляем кривую интерполяции
                if len(all_points) > 1:
                    self._add_norm_curve_line_safe(norm_id, all_points, routes_df)
                elif len(all_points) == 1:
                    self._add_constant_norm_line_safe(norm_id, all_points[0])
                    
                added_curves += 1
                logger.debug("✓ Добавлена норма %s (%d точек)", norm_id, len(all_points))
                
            except Exception as e:
                logger.error("Ошибка добавления нормы %s: %s", norm_id, e)
                continue
                
        logger.info("Добавлено кривых норм: %d из %d", added_curves, len(norm_functions))
        
    def _add_base_norm_points_safe(self, norm_id: str, base_points: List[Tuple[float, float]]) -> None:
        """Безопасно добавляет базовые точки норм."""
        try:
            if not base_points:
                return
                
            x_vals = [float(p[0]) for p in base_points]
            y_vals = [float(p[1]) for p in base_points]
            
            scatter = self.ax1.scatter(
                x_vals, y_vals,
                marker='s',  # квадраты
                s=80,        # увеличенный размер для видимости
                c='blue',
                alpha=0.9,
                edgecolor='darkblue',
                linewidth=2,
                label=f'Базовые точки {norm_id} ({len(base_points)})',
                zorder=6  # над кривыми
            )
            
            self._norm_points.append(scatter)
            logger.debug("✓ Добавлено базовых точек для нормы %s: %d", norm_id, len(base_points))
            
        except Exception as e:
            logger.error("Ошибка добавления базовых точек нормы %s: %s", norm_id, e)
            
    def _add_additional_norm_points_safe(self, norm_id: str, additional_points: List[Tuple[float, float]]) -> None:
        """Безопасно добавляет дополнительные точки норм из маршрутов."""
        try:
            if not additional_points:
                return
                
            x_vals = [float(p[0]) for p in additional_points]
            y_vals = [float(p[1]) for p in additional_points]
            
            scatter = self.ax1.scatter(
                x_vals, y_vals,
                marker='s',  # квадраты
                s=80,        # размер
                c='gold',
                alpha=0.9,
                edgecolor='orange',
                linewidth=2,
                label=f'Из маршрутов {norm_id} ({len(additional_points)})',
                zorder=6
            )
            
            self._norm_points.append(scatter)
            logger.debug("✓ Добавлено дополнительных точек для нормы %s: %d", norm_id, len(additional_points))
            
        except Exception as e:
            logger.error("Ошибка добавления дополнительных точек нормы %s: %s", norm_id, e)
            
    def _add_norm_curve_line_safe(self, norm_id: str, all_points: List[Tuple[float, float]], 
                                 routes_df: pd.DataFrame) -> None:
        """Безопасно добавляет интерполированную кривую нормы."""
        try:
            if len(all_points) < 2:
                logger.warning("Недостаточно точек для кривой нормы %s: %d", norm_id, len(all_points))
                return
                
            # Сортируем точки по X для корректной интерполяции
            sorted_points = sorted(all_points, key=lambda p: float(p[0]))
            x_vals = np.array([float(p[0]) for p in sorted_points])
            y_vals = np.array([float(p[1]) for p in sorted_points])
            
            # Проверяем валидность данных
            if np.any(x_vals <= 0) or np.any(y_vals <= 0):
                logger.warning("Норма %s содержит нулевые или отрицательные значения", norm_id)
                return
                
            # Определяем диапазон интерполяции
            x_range = self._calculate_safe_interpolation_range(x_vals, routes_df, norm_id)
            
            # Создаем плотную сетку для гладкой кривой
            x_interp = np.linspace(x_range[0], x_range[1], 300)
            
            # Используем безопасную интерполяцию
            try:
                y_interp = np.interp(x_interp, x_vals, y_vals)
            except Exception as interp_error:
                logger.error("Ошибка интерполяции нормы %s: %s", norm_id, interp_error)
                return
            
            # Создаем линию
            line, = self.ax1.plot(
                x_interp, y_interp,
                color='blue',
                linewidth=3,
                alpha=0.8,
                label=f'Норма {norm_id} ({len(all_points)} точек)',
                zorder=4
            )
            
            self._norm_lines.append(line)
            logger.debug("✓ Добавлена кривая нормы %s с %d точками", norm_id, len(all_points))
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка добавления кривой нормы %s: %s", norm_id, e, exc_info=True)
            
    def _add_constant_norm_line_safe(self, norm_id: str, point: Tuple[float, float]) -> None:
        """Безопасно добавляет константную линию нормы."""
        try:
            x_single, y_single = float(point[0]), float(point[1])
            
            if x_single <= 0 or y_single <= 0:
                logger.warning("Константная норма %s имеет некорректные значения: x=%.2f, y=%.2f", 
                             norm_id, x_single, y_single)
                return
            
            # Определяем разумный диапазон для константной линии
            x_start = max(x_single * 0.3, x_single - 50)
            x_end = x_single * 2.0 + 50
            
            line, = self.ax1.plot(
                [x_start, x_end], [y_single, y_single],
                color='blue',
                linewidth=3,
                linestyle='-',
                alpha=0.8,
                label=f'Норма {norm_id} (константа)',
                zorder=4
            )
            
            self._norm_lines.append(line)
            logger.debug("✓ Добавлена константная норма %s: y=%.2f", norm_id, y_single)
            
        except Exception as e:
            logger.error("Ошибка добавления константной нормы %s: %s", norm_id, e)
            
    def _calculate_safe_interpolation_range(self, x_vals: np.ndarray, routes_df: pd.DataFrame, 
                                          norm_id: str) -> Tuple[float, float]:
        """Безопасно вычисляет диапазон интерполяции."""
        try:
            x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
            x_range = max(x_max - x_min, 1.0)
            
            # Базовый диапазон с запасом
            x_start = max(x_min - x_range * 0.2, x_min * 0.7, 0.1)
            x_end = x_max + x_range * 0.2
            
            # Расширяем под данные маршрутов если есть
            try:
                route_x_vals = self._get_route_x_values_safe(routes_df, norm_id)
                if route_x_vals:
                    route_min, route_max = min(route_x_vals), max(route_x_vals)
                    x_start = min(x_start, route_min * 0.9)
                    x_end = max(x_end, route_max * 1.1)
            except Exception:
                pass  # Используем базовый диапазон
                
            return x_start, x_end
            
        except Exception as e:
            logger.error("Ошибка расчета диапазона интерполяции: %s", e)
            return 1.0, 100.0  # fallback диапазон
            
    def _get_route_x_values_safe(self, routes_df: pd.DataFrame, norm_id: str) -> List[float]:
        """Безопасно получает X-координаты из данных маршрутов."""
        x_values = []
        try:
            # Фильтруем по норме
            norm_routes = routes_df[routes_df["Номер нормы"].astype(str) == str(norm_id)]
            
            for _, row in norm_routes.iterrows():
                x_val = self._calculate_x_parameter_safe(row)
                if x_val and x_val > 0:
                    x_values.append(x_val)
                    
        except Exception as e:
            logger.debug("Ошибка получения X-координат маршрутов: %s", e)
            
        return x_values
        
    def _add_route_points_safe(self, routes_df: pd.DataFrame, norm_functions: Dict) -> None:
        """ИСПРАВЛЕННЫЙ метод добавления точек маршрутов с детальной обработкой ошибок."""
        logger.info("Добавление точек маршрутов: %d записей", len(routes_df))
        
        total_points_added = 0
        
        # Группируем маршруты по статусам для эффективного отображения
        for status_name, color in self._status_colors.items():
            try:
                status_routes = routes_df[routes_df["Статус"] == status_name]
                if status_routes.empty:
                    continue
                    
                x_vals, y_vals, metadata_list = [], [], []
                
                # Обрабатываем каждую строку статуса
                for idx, (_, row) in enumerate(status_routes.iterrows()):
                    try:
                        point_data = self._process_route_point_safe(row, norm_functions)
                        if point_data:
                            x_vals.append(point_data["x"])
                            y_vals.append(point_data["y"]) 
                            metadata_list.append(point_data["metadata"])
                        else:
                            logger.debug("Точка %d статуса %s пропущена (невалидные данные)", idx, status_name)
                            
                    except Exception as point_error:
                        logger.warning("Ошибка обработки точки %d статуса %s: %s", idx, status_name, point_error)
                        continue
                        
                # Создаем scatter plot если есть валидные точки
                if x_vals:
                    try:
                        scatter = self.ax1.scatter(
                            x_vals, y_vals,
                            c=color,
                            s=50,  # размер точек
                            alpha=0.7,
                            edgecolor='black',
                            linewidth=0.8,
                            label=f'{status_name} ({len(x_vals)})',
                            zorder=5,  # над кривыми норм
                            picker=True  # для интерактивности
                        )
                        
                        # Сохраняем данные для режимов отображения
                        self._scatter_objects[status_name] = scatter
                        self._traces_data[status_name] = {
                            'x': np.array(x_vals, dtype=float),
                            'y': np.array(y_vals, dtype=float),
                            'metadata': metadata_list,
                            'routes_df': status_routes.copy()  # копия для безопасности
                        }
                        
                        total_points_added += len(x_vals)
                        logger.debug("✓ Статус %s: добавлено %d точек", status_name, len(x_vals))
                        
                    except Exception as scatter_error:
                        logger.error("Ошибка создания scatter для статуса %s: %s", status_name, scatter_error)
                        continue
                        
            except Exception as status_error:
                logger.error("Ошибка обработки статуса %s: %s", status_name, status_error, exc_info=True)
                continue
                
        logger.info("✓ Всего добавлено точек маршрутов: %d", total_points_added)
        
        if total_points_added == 0:
            logger.warning("НЕ ДОБАВЛЕНО НИ ОДНОЙ ТОЧКИ МАРШРУТА!")
            self.ax1.text(0.5, 0.5, 'Нет валидных точек маршрутов\nдля отображения', 
                         ha='center', va='center', transform=self.ax1.transAxes, 
                         fontsize=14, color='red')
        
    def _process_route_point_safe(self, row: pd.Series, norm_functions: Dict) -> Optional[Dict]:
        """Безопасная обработка одной точки маршрута с детальной валидацией."""
        try:
            # 1. Проверяем номер нормы
            norm_number_raw = row.get("Номер нормы")
            if pd.isna(norm_number_raw):
                return None
                
            # 2. Получаем функцию нормы
            norm_str = str(safe_float(norm_number_raw, 0))
            if norm_str == "0" or not norm_str:
                return None
                
            norm_func_data = norm_functions.get(norm_str)
            if not norm_func_data:
                return None
                
            # 3. Вычисляем X координату (параметр нормирования)
            x_val = self._calculate_x_parameter_safe(row)
            if not x_val or x_val <= 0:
                return None
                
            # 4. Получаем Y координату (фактический удельный расход)
            y_val = safe_float(row.get("Факт уд"))
            if not y_val or y_val <= 0:
                return None
                
            # 5. Создаем метаданные для интерактивности
            metadata = self._build_route_metadata_safe(row, x_val, y_val)
            
            return {
                "x": float(x_val),
                "y": float(y_val),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.debug("Ошибка обработки точки маршрута: %s", e)
            return None
            
    def _calculate_x_parameter_safe(self, row: pd.Series) -> Optional[float]:
        """Безопасно вычисляет параметр нормирования (X координату)."""
        try:
            # Метод 1: готовое значение нажатия на ось
            axle_load = safe_float(row.get("Нажатие на ось"))
            if axle_load > 0:
                return axle_load
                
            # Метод 2: расчет по БРУТТО/ОСИ
            brutto = safe_float(row.get("БРУТТО"))
            osi = safe_float(row.get("ОСИ"))
            
            if brutto > 0 and osi > 0:
                return brutto / osi
                
            # Метод 3: приблизительный расчет из ткм и км (вес поезда)
            tkm_brutto = safe_float(row.get("Ткм брутто"))
            km = safe_float(row.get("Км"))
            
            if tkm_brutto > 0 and km > 0:
                weight = tkm_brutto / km
                # Конвертируем вес в нажатие на ось (эмпирическая формула)
                return weight / 80.0  # примерное количество осей
                
            return None
            
        except Exception as e:
            logger.debug("Ошибка расчета X параметра: %s", e)
            return None
            
    def _build_route_metadata_safe(self, row: pd.Series, x_val: float, y_val: float) -> Dict:
        """Безопасно создает метаданные маршрута для интерактивности."""
        try:
            return {
                'route_number': str(row.get("Номер маршрута", "N/A")),
                'route_date': str(row.get("Дата маршрута", "N/A")),
                'section_name': str(row.get("Наименование участка", "N/A")),
                'locomotive_series': str(row.get("Серия локомотива", "N/A")),
                'locomotive_number': str(row.get("Номер локомотива", "N/A")),
                'x_param': float(x_val),
                'y_fact': float(y_val),
                'deviation_percent': safe_float(row.get("Отклонение, %"), 0.0),
                'status': str(row.get("Статус", "N/A")),
                'norm_interpolated': safe_float(row.get("Норма интерполированная"), 0.0),
                'rashod_fact': safe_float(row.get("Расход фактический"), 0.0),
                'rashod_norm': safe_float(row.get("Расход по норме"), 0.0),
                'ud_norma_original': safe_float(row.get("Уд. норма, норма на 1 час ман. раб."), 0.0),
                # Данные для режима Н/Ф
                'nf_y_value': self._calculate_nf_y_safe(row, y_val)
            }
            
        except Exception as e:
            logger.error("Ошибка создания метаданных: %s", e)
            return {'route_number': 'ERROR', 'error': str(e)}
            
    def _calculate_nf_y_safe(self, row: pd.Series, original_y: float) -> float:
        """Безопасно вычисляет Y координату для режима Н/Ф."""
        try:
            rashod_fact = safe_float(row.get('Расход фактический'))
            rashod_norm = safe_float(row.get('Расход по норме'))
            ud_norma = safe_float(row.get('Уд. норма, норма на 1 час ман. раб.'))
            
            if rashod_fact > 0 and rashod_norm > 0 and ud_norma > 0:
                coefficient = rashod_fact / rashod_norm
                return coefficient * ud_norma
                
            return original_y  # fallback на исходное значение
            
        except Exception:
            return original_y
            
    def _add_deviation_analysis_safe(self, routes_df: pd.DataFrame) -> None:
        """Безопасно добавляет анализ отклонений на нижний график."""
        logger.info("Добавление анализа отклонений")
        
        try:
            total_deviation_points = 0
            
            # Добавляем точки отклонений по статусам
            for status_name, color in self._status_colors.items():
                status_data = routes_df[routes_df["Статус"] == status_name]
                if status_data.empty:
                    continue
                    
                x_vals, y_vals = [], []
                
                for _, row in status_data.iterrows():
                    try:
                        x_val = self._calculate_x_parameter_safe(row)
                        y_val = safe_float(row.get("Отклонение, %"))
                        
                        if x_val and x_val > 0 and y_val is not None:
                            x_vals.append(x_val)
                            y_vals.append(y_val)
                            
                    except Exception:
                        continue
                        
                if x_vals:
                    self.ax2.scatter(
                        x_vals, y_vals,
                        c=color,
                        s=50,  # размер точек отклонений
                        alpha=0.8,
                        edgecolor='black',
                        linewidth=0.5,
                        zorder=4,
                        label=f'{status_name} ({len(x_vals)})'  # добавляем в легенду нижнего графика
                    )
                    
                    total_deviation_points += len(x_vals)
                    
            logger.info("✓ Добавлено точек отклонений: %d", total_deviation_points)
            
        except Exception as e:
            logger.error("Ошибка добавления анализа отклонений: %s", e, exc_info=True)
            
    def _add_deviation_zones(self) -> None:
        """Добавляет цветные зоны отклонений на нижний график."""
        try:
            # Нулевая линия (идеальная норма)
            self.ax2.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
            
            # Зона нормы (-5% до +5%)
            self.ax2.axhspan(-5, 5, alpha=0.15, color='gold', label='Зона нормы (±5%)')
            
            # Границы отклонений
            boundaries = [
                (20, 'orange', ':', 'Значительный перерасход (+20%)'),
                (-20, 'orange', ':', 'Значительная экономия (-20%)'),
                (30, 'red', '-.', 'Критический перерасход (+30%)'),
                (-30, 'red', '-.', 'Критическая экономия (-30%)')
            ]
            
            for y_val, color, linestyle, label in boundaries:
                self.ax2.axhline(
                    y=y_val, 
                    color=color, 
                    linestyle=linestyle, 
                    alpha=0.7, 
                    linewidth=1.5,
                    label=label
                )
                
            logger.debug("✓ Добавлены зоны отклонений")
            
        except Exception as e:
            logger.error("Ошибка добавления зон отклонений: %s", e)
            
    def _setup_event_handlers_safe(self) -> None:
        """Безопасно настраивает обработчики событий."""
        try:
            # Подключаем обработчик кликов с защитой от ошибок
            self.figure.canvas.mpl_connect('button_press_event', self._on_click_safe)
            logger.debug("✓ Обработчики событий подключены")
            
        except Exception as e:
            logger.error("Ошибка настройки обработчиков событий: %s", e)
            
    def _setup_final_layout(self) -> None:
        """Финальная настройка layout'а и легенды."""
        try:
            # Настройка легенды верхнего графика
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
                
            # Автомасштабирование с проверкой данных
            for ax in [self.ax1, self.ax2]:
                try:
                    ax.relim()
                    ax.autoscale_view(tight=False)
                except Exception as autoscale_error:
                    logger.warning("Проблема автомасштабирования: %s", autoscale_error)
                    
            # Tight layout для лучшего размещения
            try:
                self.figure.tight_layout()
            except Exception:
                pass  # Не критично если tight_layout не работает
                
            logger.debug("✓ Layout настроен")
            
        except Exception as e:
            logger.error("Ошибка настройки layout: %s", e)
            
    def _on_click_safe(self, event) -> None:
        """Безопасный обработчик клика с детальной диагностикой."""
        try:
            if event.inaxes != self.ax1 or not event.xdata or not event.ydata:
                return
                
            # Находим ближайшую точку
            closest_point = self._find_closest_point_safe(event.xdata, event.ydata)
            if closest_point:
                self._show_route_details_safe(closest_point)
                
        except Exception as e:
            logger.error("Ошибка обработки клика: %s", e)
            
    def _find_closest_point_safe(self, click_x: float, click_y: float) -> Optional[Dict]:
        """Безопасно находит ближайшую точку к клику."""
        try:
            min_distance = float('inf')
            closest_metadata = None
            
            # Получаем границы графика для нормализации расстояния
            x_range = abs(self.ax1.get_xlim()[1] - self.ax1.get_xlim()[0])
            y_range = abs(self.ax1.get_ylim()[1] - self.ax1.get_ylim()[0])
            
            if x_range <= 0 or y_range <= 0:
                return None
                
            # Ищем по всем трассам
            for trace_name, data in self._traces_data.items():
                x_vals = data.get('x', np.array([]))
                y_vals = self._get_current_y_values_safe(trace_name)
                metadata_list = data.get('metadata', [])
                
                for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                    try:
                        # Нормализованное расстояние
                        norm_dist = ((click_x - x) / x_range) ** 2 + ((click_y - y) / y_range) ** 2
                        
                        if norm_dist < min_distance:
                            min_distance = norm_dist
                            if i < len(metadata_list):
                                closest_metadata = metadata_list[i].copy()
                                closest_metadata['current_y'] = float(y)
                                closest_metadata['trace_name'] = trace_name
                                
                    except Exception:
                        continue
                        
            # Порог близости - достаточно щедрый для удобства
            if min_distance < 0.005:  # 0.5% от диапазона графика
                return closest_metadata
                
            return None
            
        except Exception as e:
            logger.error("Ошибка поиска ближайшей точки: %s", e)
            return None
            
    def _get_current_y_values_safe(self, trace_name: str) -> np.ndarray:
        """Безопасно получает текущие Y координаты с учетом режима отображения."""
        try:
            # Получаем данные от менеджера режимов
            current_y_data = self.mode_manager._get_current_y_data()
            
            if trace_name in current_y_data:
                return current_y_data[trace_name]
            
            # Fallback на исходные данные
            original_data = self._traces_data.get(trace_name, {})
            return original_data.get('y', np.array([]))
            
        except Exception as e:
            logger.debug("Ошибка получения Y координат для %s: %s", trace_name, e)
            return np.array([])
            
    def _show_route_details_safe(self, metadata: Dict) -> None:
        """Безопасно показывает детали маршрута в messagebox."""
        try:
            current_mode = self.mode_manager.get_current_mode()
            mode_text = "Уд. на работу" if current_mode == DisplayMode.WORK else "Н/Ф"
            
            details = [
                f"Маршрут №{metadata.get('route_number', 'N/A')} | {metadata.get('route_date', 'N/A')}",
                f"Участок: {metadata.get('section_name', 'N/A')}",
                f"Локомотив: {metadata.get('locomotive_series', 'N/A')} №{metadata.get('locomotive_number', 'N/A')}",
                "",
                f"Режим отображения: {mode_text}",
                f"X (параметр): {metadata.get('x_param', 0):.1f}",
                f"Y (текущий): {metadata.get('current_y', 0):.2f}",
                "",
                f"Норма интерпол.: {format_number(metadata.get('norm_interpolated', 0))}",
                f"Отклонение: {format_number(metadata.get('deviation_percent', 0))}%",
                f"Статус: {metadata.get('status', 'N/A')}",
                "",
                f"Расход фактический: {format_number(metadata.get('rashod_fact', 0))}",
                f"Расход по норме: {format_number(metadata.get('rashod_norm', 0))}"
            ]
            
            # Используем messagebox для простоты и надежности
            from tkinter import messagebox
            messagebox.showinfo(
                f"Детали маршрута №{metadata.get('route_number', 'N/A')}", 
                "\n".join(details)
            )
            
        except Exception as e:
            logger.error("Ошибка отображения деталей маршрута: %s", e)
            
    def switch_display_mode(self, mode: DisplayMode) -> None:
        """
        ИСПРАВЛЕННОЕ переключение режима отображения с защитой от ошибок.
        """
        logger.info("Переключение режима отображения на: %s", mode.value)
        
        try:
            # Получаем новые Y координаты
            new_y_data = self.mode_manager.switch_mode(mode)
            
            if not new_y_data:
                logger.warning("Нет данных для переключения режима")
                return
                
            # Обновляем каждый scatter объект
            updated_traces = 0
            for trace_name, new_y in new_y_data.items():
                scatter_obj = self._scatter_objects.get(trace_name)
                if scatter_obj is None:
                    continue
                    
                try:
                    # Получаем текущие координаты
                    offsets = scatter_obj.get_offsets()
                    if len(offsets) != len(new_y):
                        logger.warning("Несоответствие точек для %s: %d vs %d", 
                                     trace_name, len(offsets), len(new_y))
                        continue
                        
                    # Обновляем координаты
                    new_offsets = np.column_stack([offsets[:, 0], new_y])
                    scatter_obj.set_offsets(new_offsets)
                    updated_traces += 1
                    
                except Exception as trace_error:
                    logger.error("Ошибка обновления трассы %s: %s", trace_name, trace_error)
                    continue
                    
            # Обновляем заголовок
            self._update_title_with_mode(mode)
            
            # Автомасштабирование Y оси верхнего графика
            try:
                self.ax1.relim()
                self.ax1.autoscale_view()
            except Exception as rescale_error:
                logger.warning("Проблема автомасштабирования: %s", rescale_error)
            
            # Перерисовываем
            self.figure.canvas.draw_idle()
            
            logger.info("✓ Переключено %d трасс, режим: %s", updated_traces, mode.value)
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка переключения режима: %s", e, exc_info=True)
            
    def _update_title_with_mode(self, mode: DisplayMode) -> None:
        """Обновляет заголовок с указанием текущего режима."""
        try:
            current_title = self.ax1.get_title()
            mode_label = "Уд. на работу" if mode == DisplayMode.WORK else "Н/Ф"
            
            # Убираем старую информацию о режиме
            if " | Режим:" in current_title:
                title_base = current_title.split(" | Режим:")[0]
            else:
                title_base = current_title
                
            new_title = f"{title_base} | Режим: {mode_label}"
            self.ax1.set_title(new_title, fontsize=12, fontweight='bold', pad=20)
            
        except Exception as e:
            logger.error("Ошибка обновления заголовка: %s", e)
            
    def _show_error_plot(self, error_message: str) -> None:
        """Показывает график с сообщением об ошибке."""
        try:
            self._clear_all_plot_data()
            
            self.ax1.text(0.5, 0.5, f'ОШИБКА:\n{error_message}', 
                         ha='center', va='center', transform=self.ax1.transAxes,
                         fontsize=14, color='red', weight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
            self.ax1.set_title('Ошибка создания графика', color='red', fontweight='bold')
            
            # Убираем оси для чистоты
            self.ax1.set_xticks([])
            self.ax1.set_yticks([])
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])
            
        except Exception as nested_error:
            logger.error("Ошибка отображения ошибки: %s", nested_error)
    
    def export_plot(self, filename: str, dpi: int = 300) -> bool:
        """
        Экспортирует график в файл с обработкой ошибок.
        """
        try:
            # Временно улучшаем качество для экспорта
            original_dpi = self.figure.dpi
            self.figure.dpi = dpi
            
            self.figure.savefig(
                filename, 
                dpi=dpi, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                format=None  # автоматически по расширению
            )
            
            # Восстанавливаем исходное dpi
            self.figure.dpi = original_dpi
            
            logger.info("График экспортирован: %s (DPI: %d)", filename, dpi)
            return True
            
        except Exception as e:
            logger.error("Ошибка экспорта графика: %s", e, exc_info=True)
            return False