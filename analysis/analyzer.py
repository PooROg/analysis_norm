#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный анализатор норм расхода электроэнергии.
Устранены ошибки с Plotly scatter plots и другие потенциальные проблемы.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from functools import lru_cache, cached_property
from typing import Protocol, Optional, Any, Dict, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, CubicSpline
import structlog

from .data_models import (
    AnalysisResult, DefaultDataProcessor, ProcessingStats,
    NormData, NormType, RouteSection
)
from .html_route_processor import HTMLRouteProcessor
from .html_norm_processor import HTMLNormProcessor
from .norm_storage import NormStorage
from .status_config import DEFAULT_STATUS_CONFIG, StatusConfig

warnings.filterwarnings('ignore')

# Настройка структурированного логирования
logger = structlog.get_logger(__name__)

# Типы для Python 3.12
type PlotlyFigure = go.Figure
type StatisticsDict = Dict[str, int | float]
type NormFunctions = Dict[str, Dict[str, Any]]

class RouteCalculator(Protocol):
    """Протокол для расчетов маршрутов"""
    def calculate_axle_load(self, route_data: pd.Series) -> Optional[float]: ...
    def apply_coefficients(self, routes: pd.DataFrame, manager: Any) -> pd.DataFrame: ...

@dataclass(slots=True)
class PlotConfig:
    """Конфигурация графиков с оптимизацией памяти."""
    height: int = 1000
    main_ratio: float = 0.6
    deviation_ratio: float = 0.4
    vertical_spacing: float = 0.05
    interpolation_points: int = 100
    marker_size: int = 8
    marker_opacity: float = 0.8
    line_width: int = 2

class PlotBuilder:
    """Строитель интерактивных графиков с исправленными ошибками."""
    
    def __init__(self, status_config: StatusConfig, data_processor: DefaultDataProcessor):
        self.status_config = status_config
        self.data_processor = data_processor
        self.plot_config = PlotConfig()
    
    def create_interactive_plot(self, section_name: str, routes_df: pd.DataFrame, 
                               norm_functions: NormFunctions, specific_norm_id: Optional[str] = None,
                               single_section_only: bool = False) -> PlotlyFigure:
        """Создает интерактивный график для участка с исправленными ошибками."""
        title_parts = [f"Нормы расхода для участка: {section_name}"]
        if specific_norm_id:
            title_parts.append(f"(норма {specific_norm_id})")
        if single_section_only:
            title_parts.append("[только один участок]")
        
        main_title = " ".join(title_parts)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=self.plot_config.vertical_spacing,
            row_heights=[self.plot_config.main_ratio, self.plot_config.deviation_ratio],
            subplot_titles=(main_title, "Отклонение фактического расхода от нормы")
        )
        
        # Добавляем компоненты графика с проверками
        try:
            self._add_norm_curves(fig, norm_functions, specific_norm_id)
            self._add_route_points(fig, routes_df)
            self._add_deviation_analysis(fig, routes_df)
            self._configure_layout(fig)
        except Exception as e:
            logger.error("Ошибка создания графика", error=str(e))
            # Возвращаем пустой график в случае ошибки
            return self._create_empty_plot(section_name)
        
        return fig
    
    def _create_empty_plot(self, section_name: str) -> PlotlyFigure:
        """Создает пустой график в случае ошибки."""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Ошибка создания графика для участка {section_name}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(
            title=f"Ошибка анализа участка {section_name}",
            height=600
        )
        return fig
    
    def _add_norm_curves(self, fig: PlotlyFigure, norm_functions: NormFunctions, specific_norm_id: Optional[str]):
        """Добавляет кривые норм на верхний график с исправленными ошибками."""
        for norm_id, norm_data in norm_functions.items():
            if specific_norm_id and norm_id != specific_norm_id:
                continue
                
            try:
                points = norm_data.get('points', [])
                if len(points) < 2:
                    logger.warning(f"Недостаточно точек для нормы {norm_id}")
                    continue
                
                x_vals, y_vals = zip(*points)
                
                # Создаем интерполированную кривую с проверками
                x_min, x_max = min(x_vals), max(x_vals)
                if x_max <= x_min:
                    logger.warning(f"Некорректный диапазон X для нормы {norm_id}")
                    continue
                
                x_interp = np.linspace(x_min, x_max, self.plot_config.interpolation_points)
                
                # ИСПРАВЛЕНИЕ: Проверяем тип результата интерполяции
                norm_func = norm_data.get('function')
                if norm_func is None:
                    logger.warning(f"Отсутствует функция интерполяции для нормы {norm_id}")
                    continue
                
                try:
                    y_interp_result = norm_func(x_interp)
                    
                    # Убеждаемся, что результат - массив
                    if np.isscalar(y_interp_result):
                        y_interp = np.full_like(x_interp, y_interp_result)
                    else:
                        y_interp = np.asarray(y_interp_result)
                    
                    # Проверяем размеры массивов
                    if len(x_interp) != len(y_interp):
                        logger.warning(f"Несоответствие размеров массивов для нормы {norm_id}")
                        continue
                        
                except Exception as e:
                    logger.error(f"Ошибка интерполяции для нормы {norm_id}: {e}")
                    continue
                
                # Добавляем интерполированную кривую
                fig.add_trace(go.Scatter(
                    x=list(x_interp), y=list(y_interp), mode='lines',
                    name=f'Норма №{norm_id}', 
                    line=dict(width=self.plot_config.line_width),
                    hovertemplate='Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм<extra></extra>'
                ), row=1, col=1)
                
                # Добавляем опорные точки с проверками
                if len(x_vals) > 0 and len(y_vals) > 0:
                    fig.add_trace(go.Scatter(
                        x=list(x_vals), y=list(y_vals), mode='markers',
                        marker=dict(symbol='square', size=self.plot_config.marker_size, color='black'),
                        name=f'Опорные точки нормы №{norm_id}',
                        hovertemplate='Опорная точка<br>Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм<extra></extra>'
                    ), row=1, col=1)
                    
            except Exception as e:
                logger.error(f"Ошибка добавления кривой нормы {norm_id}: {e}")
                continue
    
    def _add_route_points(self, fig: PlotlyFigure, routes_df: pd.DataFrame):
        """Добавляет фактические точки маршрутов с исправленными ошибками."""
        if routes_df.empty:
            logger.warning("Пустой DataFrame маршрутов")
            return
            
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен']
        if valid_routes.empty:
            logger.warning("Нет валидных маршрутов для отображения")
            return
        
        for status, color in self.status_config.status_colors.items():
            status_routes = valid_routes[valid_routes['Статус'] == status]
            if status_routes.empty:
                continue
            
            x_values, y_values, hover_texts = [], [], []
            
            for _, route in status_routes.iterrows():
                try:
                    axle_load = self.data_processor.calculate_axle_load(route)
                    consumption = route.get('Факт уд') or route.get('Расход фактический')
                    
                    # ИСПРАВЛЕНИЕ: Строгие проверки типов и значений
                    if (axle_load is not None and consumption is not None and 
                        not pd.isna(axle_load) and not pd.isna(consumption) and
                        isinstance(axle_load, (int, float)) and isinstance(consumption, (int, float)) and
                        axle_load > 0 and consumption > 0):
                        
                        x_values.append(float(axle_load))
                        y_values.append(float(consumption))
                        hover_texts.append(self._create_hover_text(route, axle_load, consumption))
                        
                except Exception as e:
                    logger.debug(f"Ошибка обработки маршрута: {e}")
                    continue
            
            # ИСПРАВЛЕНИЕ: Добавляем trace только если есть валидные данные
            if len(x_values) > 0 and len(y_values) > 0 and len(x_values) == len(y_values):
                try:
                    fig.add_trace(go.Scatter(
                        x=x_values, y=y_values, mode='markers',
                        name=f'{status} ({len(x_values)})',
                        marker=dict(
                            color=color, 
                            size=self.plot_config.marker_size, 
                            opacity=self.plot_config.marker_opacity, 
                            line=dict(color='black', width=0.5)
                        ),
                        hovertemplate='%{text}<extra></extra>', 
                        text=hover_texts
                    ), row=1, col=1)
                except Exception as e:
                    logger.error(f"Ошибка добавления точек для статуса {status}: {e}")
                    continue
    
    def _add_deviation_analysis(self, fig: PlotlyFigure, routes_df: pd.DataFrame):
        """Добавляет анализ отклонений на нижний график с исправленными ошибками."""
        if routes_df.empty:
            return
            
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен']
        if valid_routes.empty:
            return
        
        # Точки отклонений
        for status, color in self.status_config.status_colors.items():
            status_data = valid_routes[valid_routes['Статус'] == status]
            if status_data.empty:
                continue
            
            x_values, y_values, hover_texts = [], [], []
            
            for _, route in status_data.iterrows():
                try:
                    axle_load = self.data_processor.calculate_axle_load(route)
                    deviation = route.get('Отклонение, %')
                    
                    # ИСПРАВЛЕНИЕ: Строгие проверки
                    if (axle_load is not None and deviation is not None and
                        not pd.isna(axle_load) and not pd.isna(deviation) and
                        isinstance(axle_load, (int, float)) and isinstance(deviation, (int, float))):
                        
                        x_values.append(float(axle_load))
                        y_values.append(float(deviation))
                        
                        consumption = route.get('Факт уд') or route.get('Расход фактический')
                        hover_texts.append(self._create_hover_text(route, axle_load, consumption))
                        
                except Exception as e:
                    logger.debug(f"Ошибка обработки отклонения: {e}")
                    continue
            
            # ИСПРАВЛЕНИЕ: Добавляем trace только с валидными данными
            if len(x_values) > 0 and len(y_values) > 0 and len(x_values) == len(y_values):
                try:
                    fig.add_trace(go.Scatter(
                        x=x_values, y=y_values, mode='markers',
                        name=f'{status} ({len(x_values)})',
                        marker=dict(
                            color=color, 
                            size=10, 
                            opacity=self.plot_config.marker_opacity,
                            line=dict(color='black', width=0.5)
                        ),
                        hovertemplate='%{text}<extra></extra>', 
                        text=hover_texts
                    ), row=2, col=1)
                except Exception as e:
                    logger.error(f"Ошибка добавления отклонений для статуса {status}: {e}")
                    continue
        
        # Граничные линии
        try:
            self._add_boundary_lines(fig, valid_routes)
        except Exception as e:
            logger.error(f"Ошибка добавления граничных линий: {e}")
    
    def _add_boundary_lines(self, fig: PlotlyFigure, routes_df: pd.DataFrame):
        """Добавляет граничные линии с исправленными ошибками."""
        if routes_df.empty:
            return
            
        # Собираем все нагрузки на ось
        axle_loads = []
        for _, route in routes_df.iterrows():
            try:
                load = self.data_processor.calculate_axle_load(route)
                if load is not None and not pd.isna(load) and isinstance(load, (int, float)) and load > 0:
                    axle_loads.append(float(load))
            except Exception:
                continue
        
        if len(axle_loads) < 2:
            logger.warning("Недостаточно данных для граничных линий")
            return
        
        # ИСПРАВЛЕНИЕ: Создаем корректный диапазон X
        x_min, x_max = min(axle_loads), max(axle_loads)
        margin = (x_max - x_min) * 0.05  # 5% маржин
        x_range = [x_min - margin, x_max + margin]
        
        # Границы с разными стилями
        boundaries = [
            (5, '#FFD700', 'dash', 'Норма +5%'),
            (-5, '#FFD700', 'dash', 'Норма -5%'),
            (20, '#FF4500', 'dot', 'Перерасход 20%'),
            (-20, '#FF4500', 'dot', 'Экономия 20%'),
            (30, '#DC143C', 'dashdot', 'Перерасход 30%'),
            (-30, '#DC143C', 'dashdot', 'Экономия 30%'),
            (0, 'black', 'solid', 'Нулевая линия')
        ]
        
        for y_val, color, dash, name in boundaries:
            try:
                # ИСПРАВЛЕНИЕ: Убеждаемся, что передаем списки
                fig.add_trace(go.Scatter(
                    x=list(x_range), 
                    y=[float(y_val), float(y_val)],  # Явно преобразуем в список floats
                    mode='lines',
                    line=dict(color=color, dash=dash, width=2),
                    showlegend=False, 
                    hoverinfo='skip',
                    name=name
                ), row=2, col=1)
            except Exception as e:
                logger.error(f"Ошибка добавления граничной линии {y_val}: {e}")
                continue
        
        # Зеленая зона нормы
        try:
            fig.add_trace(go.Scatter(
                x=list(x_range) + list(x_range[::-1]),
                y=[-5.0, -5.0, 5.0, 5.0],
                fill='toself', 
                fillcolor='rgba(255, 215, 0, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False, 
                hoverinfo='skip',
                name='Зона нормы'
            ), row=2, col=1)
        except Exception as e:
            logger.error(f"Ошибка добавления зоны нормы: {e}")
    
    def _create_hover_text(self, route: pd.Series, axle_load: float, consumption: Optional[float]) -> str:
        """Создает текст для hover эффекта."""
        try:
            base_text = (
                f"Маршрут №{route.get('Номер маршрута', 'N/A')}<br>"
                f"Дата: {route.get('Дата маршрута', 'N/A')}<br>"
                f"Локомотив: {route.get('Серия локомотива', '')} №{route.get('Номер локомотива', '')}<br>"
            )
            
            # Коэффициент если применялся
            coeff = route.get('Коэффициент')
            if pd.notna(coeff) and coeff != 1.0:
                base_text += f"Коэффициент: {coeff:.3f}<br>"
                original = route.get('Факт. удельный исходный')
                if pd.notna(original):
                    base_text += f"Факт исходный: {original:.1f}<br>"
            
            consumption_text = f"{consumption:.1f}" if consumption is not None else "N/A"
            norm_text = route.get('Норма интерполированная', 'N/A')
            if isinstance(norm_text, (int, float)) and not pd.isna(norm_text):
                norm_text = f"{norm_text:.1f}"
            
            deviation_text = route.get('Отклонение, %', 'N/A')
            if isinstance(deviation_text, (int, float)) and not pd.isna(deviation_text):
                deviation_text = f"{deviation_text:.1f}%"
            
            return (base_text + 
                    f"Нажатие: {axle_load:.2f} т/ось<br>"
                    f"Факт: {consumption_text}<br>"
                    f"Норма: {norm_text}<br>"
                    f"Отклонение: {deviation_text}")
                    
        except Exception as e:
            logger.error(f"Ошибка создания hover текста: {e}")
            return f"Маршрут (ошибка отображения)"
    
    def _configure_layout(self, fig: PlotlyFigure):
        """Настраивает layout графика."""
        try:
            fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=1, col=1)
            fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм брутто", row=1, col=1)
            fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=2, col=1)
            fig.update_yaxes(title_text="Отклонение от нормы, %", row=2, col=1)
            
            fig.update_layout(
                height=self.plot_config.height, 
                showlegend=True, 
                hovermode='closest',
                template='plotly_white',
                legend=dict(
                    orientation="v", 
                    yanchor="middle", 
                    y=0.5, 
                    xanchor="left", 
                    x=1.02
                )
            )
        except Exception as e:
            logger.error(f"Ошибка настройки layout: {e}")

class InteractiveNormsAnalyzer:
    """Исправленный анализатор норм расхода электроэнергии."""
    
    def __init__(self, status_config: Optional[StatusConfig] = None):
        self.route_processor = HTMLRouteProcessor()
        self.norm_processor = HTMLNormProcessor()
        self.norm_storage = NormStorage()
        
        # Использование переданной конфигурации или создание по умолчанию
        config = status_config or DEFAULT_STATUS_CONFIG
        self.data_processor = DefaultDataProcessor(config)
        self.plot_builder = PlotBuilder(config, self.data_processor)
        
        self._routes_df: Optional[pd.DataFrame] = None
        self._analyzed_results: Dict[str, AnalysisResult] = {}
        self._sections_norms_map: Dict[str, List[str]] = {}
        
        logger.info("Инициализирован оптимизированный анализатор норм")
    
    @property
    def routes_df(self) -> Optional[pd.DataFrame]:
        """Доступ к данным маршрутов"""
        return self._routes_df
    
    @routes_df.setter 
    def routes_df(self, value: pd.DataFrame):
        """Установка данных маршрутов с автоматическим построением карты"""
        self._routes_df = value
        self._build_sections_norms_map()
    
    def load_routes_from_html(self, html_files: List[str]) -> bool:
        """Загружает маршруты из HTML файлов"""
        logger.info("Загрузка маршрутов", files_count=len(html_files))
        
        try:
            self.routes_df = self.route_processor.process_html_files(html_files)
            
            if self.routes_df.empty:
                logger.error("Не удалось загрузить маршруты из HTML файлов")
                return False
            
            logger.info("Загружено записей маршрутов", count=len(self.routes_df))
            self._log_routes_statistics()
            return True
            
        except Exception as e:
            logger.error("Ошибка загрузки маршрутов", error=str(e))
            return False
    
    def load_norms_from_html(self, html_files: List[str]) -> bool:
        """Загружает нормы из HTML файлов"""
        logger.info("Загрузка норм", files_count=len(html_files))
        
        try:
            new_norms = self.norm_processor.process_html_files(html_files)
            
            if not new_norms:
                logger.warning("Не найдено норм в HTML файлах")
                return False
            
            self.norm_storage.add_or_update_norms(new_norms)
            
            stats = self.norm_processor.get_processing_stats()
            logger.info("Обработано норм", 
                       total=stats['total_norms_found'],
                       new=stats['new_norms'], 
                       updated=stats['updated_norms'])
            return True
            
        except Exception as e:
            logger.error("Ошибка загрузки норм", error=str(e))
            return False
    
    @lru_cache(maxsize=128)
    def get_sections_list(self) -> Tuple[str, ...]:
        """Возвращает список доступных участков (кэшированный)"""
        if self.routes_df is None or self.routes_df.empty:
            return ()
        
        sections = tuple(sorted(self.routes_df['Наименование участка'].dropna().unique()))
        logger.debug("Найдено участков", count=len(sections))
        return sections
    
    def get_norms_for_section(self, section_name: str) -> List[str]:
        """Возвращает список норм для участка"""
        return self._sections_norms_map.get(section_name, [])
    
    def analyze_section(self, section_name: str, norm_id: Optional[str] = None,
                       single_section_only: bool = False,
                       locomotive_filter: Optional[Any] = None,
                       coefficients_manager: Optional[Any] = None,
                       use_coefficients: bool = False) -> Tuple[Optional[PlotlyFigure], Optional[Dict], Optional[str]]:
        """Анализирует участок с исправленной обработкой ошибок"""
        logger.info("Анализ участка", 
                   section=section_name, 
                   norm=norm_id, 
                   single_section=single_section_only)
        
        if self.routes_df is None or self.routes_df.empty:
            return None, None, "Данные маршрутов не загружены"
        
        try:
            # Фильтрация данных
            section_routes = self._filter_section_routes(
                section_name, norm_id, single_section_only, locomotive_filter
            )
            
            if section_routes.empty:
                return None, None, self._get_empty_data_message(section_name, norm_id, single_section_only)
            
            # Применение коэффициентов
            if use_coefficients and coefficients_manager:
                section_routes = self.data_processor.apply_coefficients(section_routes, coefficients_manager)
            
            # ИСПРАВЛЕНИЕ: Добавлена проверка корректности данных
            section_routes = self._validate_and_clean_data(section_routes)
            
            if section_routes.empty:
                return None, None, f"Нет корректных данных для анализа участка {section_name}"
            
            # Анализ данных участка
            analyzed_data, norm_functions = self._analyze_section_data(section_name, section_routes, norm_id)
            
            if analyzed_data.empty:
                return None, None, f"Не удалось проанализировать участок {section_name}"
            
            # Создание графика и статистики
            fig = self.plot_builder.create_interactive_plot(
                section_name, analyzed_data, norm_functions, norm_id, single_section_only
            )
            statistics = self._calculate_section_statistics(analyzed_data)
            
            # Сохранение результатов
            analysis_key = self._get_analysis_key(section_name, norm_id, single_section_only)
            self._analyzed_results[analysis_key] = AnalysisResult(
                routes=analyzed_data,
                norms=norm_functions,
                statistics=statistics,
                section_name=section_name,
                norm_id=norm_id,
                single_section_only=single_section_only
            )
            
            logger.info("Анализ участка завершен успешно", section=section_name)
            return fig, statistics, None
            
        except Exception as e:
            logger.error("Ошибка анализа участка", section=section_name, error=str(e))
            return None, None, f"Ошибка анализа: {str(e)}"
    
    def _validate_and_clean_data(self, routes_df: pd.DataFrame) -> pd.DataFrame:
        """Валидирует и очищает данные маршрутов."""
        if routes_df.empty:
            return routes_df
        
        try:
            # Проверяем наличие обязательных колонок
            required_columns = ['Наименование участка']
            missing_columns = [col for col in required_columns if col not in routes_df.columns]
            
            if missing_columns:
                logger.warning(f"Отсутствуют обязательные колонки: {missing_columns}")
                return pd.DataFrame()
            
            # Удаляем строки с пустыми названиями участков
            cleaned_df = routes_df[routes_df['Наименование участка'].notna()].copy()
            
            # Проверяем и исправляем числовые поля
            numeric_fields = ['ТКМ брутто', 'КМ', 'Факт уд', 'Расход фактический']
            
            for field in numeric_fields:
                if field in cleaned_df.columns:
                    # Заменяем некорректные значения на NaN
                    cleaned_df[field] = pd.to_numeric(cleaned_df[field], errors='coerce')
                    # Заменяем отрицательные значения на NaN
                    cleaned_df.loc[cleaned_df[field] < 0, field] = np.nan
            
            logger.debug(f"Данные очищены: {len(cleaned_df)} из {len(routes_df)} строк")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Ошибка валидации данных: {e}")
            return routes_df
    
    def _filter_section_routes(self, section_name: str, norm_id: Optional[str], 
                              single_section_only: bool, 
                              locomotive_filter: Optional[Any]) -> pd.DataFrame:
        """Фильтрует маршруты по заданным критериям"""
        try:
            # Базовая фильтрация по участку (векторизованная операция)
            section_routes = self.routes_df[
                self.routes_df['Наименование участка'] == section_name
            ].copy()
            
            # Фильтрация по одному участку (оптимизированная)
            if single_section_only:
                # Кэшируем подсчет секций для избежания повторных вычислений
                route_section_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
                single_section_routes = route_section_counts[route_section_counts == 1].index
                
                section_routes = section_routes.set_index(['Номер маршрута', 'Дата маршрута'])
                section_routes = section_routes.loc[section_routes.index.intersection(single_section_routes)]
                section_routes = section_routes.reset_index()
            
            # Фильтрация по норме (векторизованная)
            if norm_id:
                section_routes = section_routes[
                    section_routes['Номер нормы'].astype(str) == str(norm_id)
                ]
            
            # Применение фильтра локомотивов
            if locomotive_filter:
                section_routes = locomotive_filter.filter_routes(section_routes)
            
            return section_routes
            
        except Exception as e:
            logger.error(f"Ошибка фильтрации маршрутов: {e}")
            return pd.DataFrame()
    
    def _analyze_section_data(self, section_name: str, routes_df: pd.DataFrame, 
                             specific_norm_id: Optional[str] = None) -> Tuple[pd.DataFrame, NormFunctions]:
        """Анализирует данные участка с использованием норм из хранилища"""
        logger.debug("Анализ данных участка", section=section_name)
        
        # Подготовка данных
        routes_df = routes_df.copy()
        routes_df['Норма интерполированная'] = 0.0
        routes_df['Отклонение, %'] = 0.0
        routes_df['Статус'] = 'Не определен'
        
        # Получение норм для анализа
        norm_numbers = [specific_norm_id] if specific_norm_id else routes_df['Номер нормы'].dropna().unique()
        norm_functions = self._build_norm_functions(norm_numbers)
        
        if not norm_functions:
            logger.warning("Не найдено функций норм", section=section_name)
            return routes_df, {}
        
        # Векторизованный анализ маршрутов
        self._process_routes_vectorized(routes_df, norm_functions)
        
        logger.info("Проанализировано записей", count=len(routes_df), section=section_name)
        return routes_df, norm_functions
    
    def _build_norm_functions(self, norm_numbers) -> NormFunctions:
        """Строит функции интерполяции для норм с исправленными ошибками"""
        norm_functions = {}
        
        for norm_number in norm_numbers:
            try:
                norm_number_str = str(int(norm_number)) if pd.notna(norm_number) else None
                if not norm_number_str:
                    continue
                    
                norm_data = self.norm_storage.get_norm(norm_number_str)
                if not (norm_data and norm_data.get('points')):
                    continue
                
                func = self.norm_storage.get_norm_function(norm_number_str)
                if func:
                    # ИСПРАВЛЕНИЕ: Тестируем функцию перед добавлением
                    test_points = norm_data['points']
                    if len(test_points) >= 2:
                        test_x = test_points[0][0]  # Первая точка по X
                        try:
                            test_result = func(test_x)
                            if np.isscalar(test_result) and not np.isnan(test_result):
                                norm_functions[norm_number_str] = {
                                    'function': func,
                                    'points': norm_data['points'],
                                    'x_range': (
                                        min(p[0] for p in norm_data['points']),
                                        max(p[0] for p in norm_data['points'])
                                    ),
                                    'data': norm_data
                                }
                                logger.debug("Создана функция для нормы", norm=norm_number_str)
                            else:
                                logger.warning(f"Функция нормы {norm_number_str} возвращает некорректные значения")
                        except Exception as e:
                            logger.warning(f"Функция нормы {norm_number_str} не работает: {e}")
                            
            except Exception as e:
                logger.error("Ошибка создания функции для нормы", norm=str(norm_number), error=str(e))
        
        return norm_functions
    
    def _process_routes_vectorized(self, routes_df: pd.DataFrame, norm_functions: NormFunctions):
        """Векторизованная обработка маршрутов с исправленными ошибками"""
        for i, row in routes_df.iterrows():
            try:
                norm_number = row.get('Номер нормы')
                if pd.notna(norm_number):
                    norm_number_str = str(int(norm_number))
                    
                    if norm_number_str in norm_functions:
                        axle_load = self.data_processor.calculate_axle_load(row)
                        
                        if (axle_load is not None and 
                            isinstance(axle_load, (int, float)) and 
                            not pd.isna(axle_load) and 
                            axle_load > 0):
                            
                            try:
                                norm_func = norm_functions[norm_number_str]['function']
                                
                                # ИСПРАВЛЕНИЕ: Проверяем результат функции
                                norm_result = norm_func(axle_load)
                                
                                if (np.isscalar(norm_result) and 
                                    not np.isnan(norm_result) and 
                                    norm_result > 0):
                                    
                                    norm_value = float(norm_result)
                                    routes_df.loc[i, 'Норма интерполированная'] = norm_value
                                    
                                    # Определение фактического расхода
                                    actual_value = row.get('Факт уд') or row.get('Расход фактический')
                                    
                                    if (actual_value is not None and 
                                        isinstance(actual_value, (int, float)) and
                                        not pd.isna(actual_value) and 
                                        actual_value > 0 and 
                                        norm_value > 0):
                                        
                                        deviation = ((actual_value - norm_value) / norm_value) * 100
                                        routes_df.loc[i, 'Отклонение, %'] = deviation
                                        routes_df.loc[i, 'Статус'] = self.data_processor.determine_status(deviation)
                                else:
                                    logger.debug(f"Некорректный результат интерполяции для строки {i}: {norm_result}")
                                    
                            except Exception as e:
                                logger.debug("Ошибка интерполяции для строки", row_index=i, error=str(e))
                                continue
                                
            except Exception as e:
                logger.debug("Ошибка обработки строки", row_index=i, error=str(e))
                continue
    
    def _calculate_section_statistics(self, routes_df: pd.DataFrame) -> StatisticsDict:
        """Вычисляет статистику для участка"""
        total = len(routes_df)
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен']
        processed = len(valid_routes)
        
        if processed == 0:
            return {
                'total': total, 'processed': processed,
                'economy': 0, 'normal': 0, 'overrun': 0,
                'mean_deviation': 0, 'median_deviation': 0,
                'detailed_stats': {}
            }
        
        # Группировка по категориям
        status_counts = valid_routes['Статус'].value_counts().to_dict()
        
        detailed_stats = {
            'economy_strong': status_counts.get('Экономия сильная', 0),
            'economy_medium': status_counts.get('Экономия средняя', 0),
            'economy_weak': status_counts.get('Экономия слабая', 0),
            'normal': status_counts.get('Норма', 0),
            'overrun_weak': status_counts.get('Перерасход слабый', 0),
            'overrun_medium': status_counts.get('Перерасход средний', 0),
            'overrun_strong': status_counts.get('Перерасход сильный', 0)
        }
        
        # ИСПРАВЛЕНИЕ: Безопасное вычисление статистики
        try:
            deviations = valid_routes['Отклонение, %'].dropna()
            mean_deviation = float(deviations.mean()) if len(deviations) > 0 else 0.0
            median_deviation = float(deviations.median()) if len(deviations) > 0 else 0.0
        except Exception as e:
            logger.warning(f"Ошибка вычисления статистики отклонений: {e}")
            mean_deviation = 0.0
            median_deviation = 0.0
        
        return {
            'total': total,
            'processed': processed,
            'economy': detailed_stats['economy_strong'] + detailed_stats['economy_medium'] + detailed_stats['economy_weak'],
            'normal': detailed_stats['normal'],
            'overrun': detailed_stats['overrun_weak'] + detailed_stats['overrun_medium'] + detailed_stats['overrun_strong'],
            'mean_deviation': mean_deviation,
            'median_deviation': median_deviation,
            'detailed_stats': detailed_stats
        }
    
    def _build_sections_norms_map(self):
        """Строит карту участков и их норм"""
        if self.routes_df is None or self.routes_df.empty:
            return
        
        try:
            sections_groups = self.routes_df.groupby('Наименование участка')['Номер нормы'].apply(
                lambda x: list(x.dropna().unique())
            ).to_dict()
            
            self._sections_norms_map = {
                section: [str(norm) for norm in norms if str(norm) != 'nan']
                for section, norms in sections_groups.items()
            }
            
            logger.info("Построена карта участков и норм", sections_count=len(self._sections_norms_map))
        except Exception as e:
            logger.error(f"Ошибка построения карты участков: {e}")
            self._sections_norms_map = {}
    
    def _log_routes_statistics(self):
        """Логирует статистику загруженных маршрутов"""
        if self.routes_df is None or self.routes_df.empty:
            return
        
        try:
            stats = self.route_processor.get_processing_stats()
            
            logger.info("=== СТАТИСТИКА ЗАГРУЖЕННЫХ МАРШРУТОВ ===")
            for key, value in stats.items():
                logger.info(f"{key}: {value}")
            
            sections_count = self.routes_df['Наименование участка'].nunique()
            norms_count = self.routes_df['Номер нормы'].nunique()
            logger.info(f"Уникальных участков: {sections_count}")
            logger.info(f"Уникальных норм: {norms_count}")
        except Exception as e:
            logger.error(f"Ошибка логирования статистики: {e}")
    
    def _get_empty_data_message(self, section_name: str, norm_id: Optional[str], single_section_only: bool) -> str:
        """Формирует сообщение об отсутствии данных"""
        filter_parts = []
        if single_section_only:
            filter_parts.append("с одним участком")
        if norm_id:
            filter_parts.append(f"с нормой {norm_id}")
        
        filter_text = " " + " ".join(filter_parts) if filter_parts else ""
        return f"Нет маршрутов{filter_text} для участка {section_name}"
    
    def _get_analysis_key(self, section_name: str, norm_id: Optional[str], single_section_only: bool) -> str:
        """Формирует ключ для кэширования анализа"""
        parts = [section_name]
        if norm_id:
            parts.append(norm_id)
        if single_section_only:
            parts.append("single")
        return "_".join(parts)
    
    # Методы для совместимости с существующим интерфейсом
    def get_norms_with_counts_for_section(self, section_name: str, single_section_only: bool = False) -> List[Tuple[str, int]]:
        """Возвращает список норм для участка с количеством маршрутов"""
        if self.routes_df is None or self.routes_df.empty:
            return []
        
        try:
            section_routes = self._filter_section_routes(section_name, None, single_section_only, None)
            if section_routes.empty:
                return []
            
            norm_counts = section_routes['Номер нормы'].value_counts()
            norms_with_counts = [
                (norm, norm_counts.get(int(norm) if norm.isdigit() else norm, 0))
                for norm in self.get_norms_for_section(section_name)
            ]
            
            return sorted(norms_with_counts, key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        except Exception as e:
            logger.error(f"Ошибка получения норм с подсчетом: {e}")
            return []
    
    def get_norm_info(self, norm_id: str) -> Optional[Dict]:
        """Возвращает информацию о норме"""
        try:
            norm_data = self.norm_storage.get_norm(norm_id)
            if not norm_data:
                return None
            
            info = {
                'norm_id': norm_id,
                'description': norm_data.get('description', f'Норма №{norm_id}'),
                'norm_type': norm_data.get('norm_type', 'Неизвестно'),
                'points_count': len(norm_data.get('points', [])),
                'points': norm_data.get('points', []),
                'base_data': norm_data.get('base_data', {})
            }
            
            # Добавляем диапазоны
            if info['points']:
                x_vals = [p[0] for p in info['points']]
                y_vals = [p[1] for p in info['points']]
                info['load_range'] = f"{min(x_vals):.1f} - {max(x_vals):.1f} т/ось"
                info['consumption_range'] = f"{min(y_vals):.1f} - {max(y_vals):.1f} кВт·ч/10⁴ ткм"
            else:
                info['load_range'] = "Нет данных"
                info['consumption_range'] = "Нет данных"
            
            return info
        except Exception as e:
            logger.error(f"Ошибка получения информации о норме {norm_id}: {e}")
            return None
    
    # Дополнительные методы для API совместимости
    def get_routes_count_for_section(self, section_name: str, single_section_only: bool = False) -> int:
        """Возвращает общее количество маршрутов для участка"""
        try:
            section_routes = self._filter_section_routes(section_name, None, single_section_only, None)
            return len(section_routes)
        except Exception as e:
            logger.error(f"Ошибка подсчета маршрутов: {e}")
            return 0
    
    def get_norm_routes_count_for_section(self, section_name: str, norm_id: str, single_section_only: bool = False) -> int:
        """Возвращает количество маршрутов для конкретной нормы в участке"""
        try:
            section_routes = self._filter_section_routes(section_name, norm_id, single_section_only, None)
            return len(section_routes)
        except Exception as e:
            logger.error(f"Ошибка подсчета маршрутов для нормы: {e}")
            return 0
    
    def get_norm_storage_info(self) -> Dict:
        """Возвращает информацию о хранилище норм"""
        try:
            return self.norm_storage.get_storage_info()
        except Exception as e:
            logger.error(f"Ошибка получения информации о хранилище: {e}")
            return {}
    
    def export_routes_to_excel(self, output_file: str) -> bool:
        """Экспортирует маршруты в Excel"""
        if self.routes_df is None or self.routes_df.empty:
            logger.warning("Нет данных маршрутов для экспорта")
            return False
        
        try:
            return self.route_processor.export_to_excel(self.routes_df, output_file)
        except Exception as e:
            logger.error(f"Ошибка экспорта в Excel: {e}")
            return False
    
    def validate_norms_storage(self) -> Dict:
        """Валидирует хранилище норм"""
        try:
            return self.norm_storage.validate_norms()
        except Exception as e:
            logger.error(f"Ошибка валидации норм: {e}")
            return {'valid_norms': 0, 'invalid_norms': 0, 'validation_errors': []}
    
    def get_norm_storage_statistics(self) -> Dict:
        """Получает статистику хранилища норм"""
        try:
            return self.norm_storage.get_norm_statistics()
        except Exception as e:
            logger.error(f"Ошибка получения статистики норм: {e}")
            return {'total': 0}
    
    def get_routes_data(self) -> pd.DataFrame:
        """Возвращает полные данные маршрутов"""
        return self.routes_df.copy() if self.routes_df is not None else pd.DataFrame()