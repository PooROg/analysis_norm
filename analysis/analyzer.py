#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный и оптимизированный анализатор норм расхода электроэнергии.
Использует современные возможности Python 3.12 для максимальной производительности.
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
    """Строитель интерактивных графиков с оптимизацией."""
    
    def __init__(self, status_config: StatusConfig, data_processor: DefaultDataProcessor):
        self.status_config = status_config
        self.data_processor = data_processor
        self.plot_config = PlotConfig()
    
    def create_interactive_plot(self, section_name: str, routes_df: pd.DataFrame, 
                               norm_functions: NormFunctions, specific_norm_id: Optional[str] = None,
                               single_section_only: bool = False) -> PlotlyFigure:
        """Создает интерактивный график для участка с оптимизацией."""
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
        
        self._add_norm_curves(fig, norm_functions, specific_norm_id)
        self._add_route_points(fig, routes_df)
        self._add_deviation_analysis(fig, routes_df)
        self._configure_layout(fig)
        
        return fig
    
    def _add_norm_curves(self, fig: PlotlyFigure, norm_functions: NormFunctions, specific_norm_id: Optional[str]):
        """Добавляет кривые норм на верхний график."""
        for norm_id, norm_data in norm_functions.items():
            if specific_norm_id and norm_id != specific_norm_id:
                continue
                
            points = norm_data['points']
            x_vals, y_vals = zip(*points)
            
            # Интерполированная кривая
            x_interp = np.linspace(min(x_vals), max(x_vals), self.plot_config.interpolation_points)
            y_interp = norm_data['function'](x_interp)
            
            fig.add_trace(go.Scatter(
                x=x_interp, y=y_interp, mode='lines',
                name=f'Норма №{norm_id}', line=dict(width=self.plot_config.line_width),
                hovertemplate='Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм'
            ), row=1, col=1)
            
            # Опорные точки
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals, mode='markers',
                marker=dict(symbol='square', size=self.plot_config.marker_size, color='black'),
                name=f'Опорные точки нормы №{norm_id}',
                hovertemplate='Опорная точка<br>Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм'
            ), row=1, col=1)
    
    def _add_route_points(self, fig: PlotlyFigure, routes_df: pd.DataFrame):
        """Добавляет фактические точки маршрутов."""
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен']
        
        for status, color in self.status_config.status_colors.items():
            status_routes = valid_routes[valid_routes['Статус'] == status]
            if status_routes.empty:
                continue
            
            x_values, y_values, hover_texts = [], [], []
            
            for _, route in status_routes.iterrows():
                axle_load = self.data_processor.calculate_axle_load(route)
                consumption = route.get('Факт уд') or route.get('Расход фактический')
                
                if axle_load and consumption:
                    x_values.append(axle_load)
                    y_values.append(consumption)
                    hover_texts.append(self._create_hover_text(route, axle_load, consumption))
            
            if x_values:
                fig.add_trace(go.Scatter(
                    x=x_values, y=y_values, mode='markers',
                    name=f'{status} ({len(status_routes)})',
                    marker=dict(color=color, size=self.plot_config.marker_size, 
                               opacity=self.plot_config.marker_opacity, 
                               line=dict(color='black', width=0.5)),
                    hovertemplate='%{text}', text=hover_texts
                ), row=1, col=1)
    
    def _add_deviation_analysis(self, fig: PlotlyFigure, routes_df: pd.DataFrame):
        """Добавляет анализ отклонений на нижний график."""
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен']
        
        # Точки отклонений
        for status, color in self.status_config.status_colors.items():
            status_data = valid_routes[valid_routes['Статус'] == status]
            if status_data.empty:
                continue
            
            x_values, y_values, hover_texts = [], [], []
            for _, route in status_data.iterrows():
                axle_load = self.data_processor.calculate_axle_load(route)
                if axle_load:
                    x_values.append(axle_load)
                    y_values.append(route['Отклонение, %'])
                    hover_texts.append(self._create_hover_text(route, axle_load, 
                                     route.get('Факт уд') or route.get('Расход фактический')))
            
            if x_values:
                fig.add_trace(go.Scatter(
                    x=x_values, y=y_values, mode='markers',
                    name=f'{status} ({len(status_data)})',
                    marker=dict(color=color, size=10, opacity=self.plot_config.marker_opacity,
                               line=dict(color='black', width=0.5)),
                    hovertemplate='%{text}', text=hover_texts
                ), row=2, col=1)
        
        # Граничные линии
        if not valid_routes.empty:
            self._add_boundary_lines(fig, valid_routes)
    
    def _add_boundary_lines(self, fig: PlotlyFigure, routes_df: pd.DataFrame):
        """Добавляет граничные линии."""
        axle_loads = [self.data_processor.calculate_axle_load(route) 
                     for _, route in routes_df.iterrows()]
        axle_loads = [load for load in axle_loads if load]
        
        if not axle_loads:
            return
        
        x_range = [min(axle_loads) - 1, max(axle_loads) + 1]
        
        # Границы с разными стилями
        boundaries = [
            (5, '#FFD700', 'dash'),
            (-5, '#FFD700', 'dash'),
            (20, '#FF4500', 'dot'),
            (-20, '#FF4500', 'dot'),
            (30, '#DC143C', 'dashdot'),
            (-30, '#DC143C', 'dashdot'),
            (0, 'black', 'solid')
        ]
        
        for y_val, color, dash in boundaries:
            fig.add_trace(go.Scatter(
                x=x_range, y=[y_val, y_val], mode='lines',
                line=dict(color=color, dash=dash, width=2),
                showlegend=False, hoverinfo='skip'
            ), row=2, col=1)
        
        # Зеленая зона нормы
        fig.add_trace(go.Scatter(
            x=x_range + x_range[::-1], y=[-5, -5, 5, 5],
            fill='toself', fillcolor='rgba(255, 215, 0, 0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False, hoverinfo='skip'
        ), row=2, col=1)
    
    def _create_hover_text(self, route: pd.Series, axle_load: float, consumption: float) -> str:
        """Создает текст для hover эффекта."""
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
        
        return (base_text + 
                f"Нажатие: {axle_load:.2f} т/ось<br>"
                f"Факт: {consumption:.1f}<br>"
                f"Норма: {route.get('Норма интерполированная', 'N/A'):.1f}<br>"
                f"Отклонение: {route.get('Отклонение, %', 'N/A'):.1f}%")
    
    def _configure_layout(self, fig: PlotlyFigure):
        """Настраивает layout графика."""
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=1, col=1)
        fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм брутто", row=1, col=1)
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=2, col=1)
        fig.update_yaxes(title_text="Отклонение от нормы, %", row=2, col=1)
        
        fig.update_layout(
            height=self.plot_config.height, showlegend=True, hovermode='closest',
            template='plotly_white',
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02)
        )

class InteractiveNormsAnalyzer:
    """Исправленный и оптимизированный анализатор норм расхода электроэнергии."""
    
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
        """Анализирует участок с возможностью выбора конкретной нормы"""
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
    
    def _filter_section_routes(self, section_name: str, norm_id: Optional[str], 
                              single_section_only: bool, 
                              locomotive_filter: Optional[Any]) -> pd.DataFrame:
        """Фильтрует маршруты по заданным критериям"""
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
        """Строит функции интерполяции для норм"""
        norm_functions = {}
        
        for norm_number in norm_numbers:
            norm_number_str = str(int(norm_number)) if pd.notna(norm_number) else None
            if not norm_number_str:
                continue
                
            norm_data = self.norm_storage.get_norm(norm_number_str)
            if not (norm_data and norm_data.get('points')):
                continue
            
            try:
                func = self.norm_storage.get_norm_function(norm_number_str)
                if func:
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
            except Exception as e:
                logger.error("Ошибка создания функции для нормы", norm=norm_number_str, error=str(e))
        
        return norm_functions
    
    def _process_routes_vectorized(self, routes_df: pd.DataFrame, norm_functions: NormFunctions):
        """Векторизованная обработка маршрутов"""
        for i, row in routes_df.iterrows():
            norm_number = row.get('Номер нормы')
            if pd.notna(norm_number):
                norm_number_str = str(int(norm_number))
                
                if norm_number_str in norm_functions:
                    try:
                        axle_load = self.data_processor.calculate_axle_load(row)
                        
                        if axle_load and axle_load > 0:
                            norm_func = norm_functions[norm_number_str]['function']
                            norm_value = float(norm_func(axle_load))
                            routes_df.loc[i, 'Норма интерполированная'] = norm_value
                            
                            # Определение фактического расхода
                            actual_value = row.get('Факт уд') or row.get('Расход фактический')
                            
                            if actual_value and norm_value > 0:
                                deviation = ((actual_value - norm_value) / norm_value) * 100
                                routes_df.loc[i, 'Отклонение, %'] = deviation
                                routes_df.loc[i, 'Статус'] = self.data_processor.determine_status(deviation)
                    
                    except Exception as e:
                        logger.debug("Ошибка интерполяции для строки", row_index=i, error=str(e))
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
        
        return {
            'total': total,
            'processed': processed,
            'economy': detailed_stats['economy_strong'] + detailed_stats['economy_medium'] + detailed_stats['economy_weak'],
            'normal': detailed_stats['normal'],
            'overrun': detailed_stats['overrun_weak'] + detailed_stats['overrun_medium'] + detailed_stats['overrun_strong'],
            'mean_deviation': valid_routes['Отклонение, %'].mean(),
            'median_deviation': valid_routes['Отклонение, %'].median(),
            'detailed_stats': detailed_stats
        }
    
    def _build_sections_norms_map(self):
        """Строит карту участков и их норм"""
        if self.routes_df is None or self.routes_df.empty:
            return
        
        sections_groups = self.routes_df.groupby('Наименование участка')['Номер нормы'].apply(
            lambda x: list(x.dropna().unique())
        ).to_dict()
        
        self._sections_norms_map = {
            section: [str(norm) for norm in norms if str(norm) != 'nan']
            for section, norms in sections_groups.items()
        }
        
        logger.info("Построена карта участков и норм", sections_count=len(self._sections_norms_map))
    
    def _log_routes_statistics(self):
        """Логирует статистику загруженных маршрутов"""
        if self.routes_df is None or self.routes_df.empty:
            return
        
        stats = self.route_processor.get_processing_stats()
        
        logger.info("=== СТАТИСТИКА ЗАГРУЖЕННЫХ МАРШРУТОВ ===")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        
        sections_count = self.routes_df['Наименование участка'].nunique()
        norms_count = self.routes_df['Номер нормы'].nunique()
        logger.info(f"Уникальных участков: {sections_count}")
        logger.info(f"Уникальных норм: {norms_count}")
    
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
        
        section_routes = self._filter_section_routes(section_name, None, single_section_only, None)
        if section_routes.empty:
            return []
        
        norm_counts = section_routes['Номер нормы'].value_counts()
        norms_with_counts = [
            (norm, norm_counts.get(int(norm) if norm.isdigit() else norm, 0))
            for norm in self.get_norms_for_section(section_name)
        ]
        
        return sorted(norms_with_counts, key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
    
    def get_norm_info(self, norm_id: str) -> Optional[Dict]:
        """Возвращает информацию о норме"""
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
    
    # Дополнительные методы для API совместимости
    def get_routes_count_for_section(self, section_name: str, single_section_only: bool = False) -> int:
        """Возвращает общее количество маршрутов для участка"""
        section_routes = self._filter_section_routes(section_name, None, single_section_only, None)
        return len(section_routes)
    
    def get_norm_routes_count_for_section(self, section_name: str, norm_id: str, single_section_only: bool = False) -> int:
        """Возвращает количество маршрутов для конкретной нормы в участке"""
        section_routes = self._filter_section_routes(section_name, norm_id, single_section_only, None)
        return len(section_routes)
    
    def get_norm_storage_info(self) -> Dict:
        """Возвращает информацию о хранилище норм"""
        return self.norm_storage.get_storage_info()
    
    def export_routes_to_excel(self, output_file: str) -> bool:
        """Экспортирует маршруты в Excel"""
        if self.routes_df is None or self.routes_df.empty:
            logger.warning("Нет данных маршрутов для экспорта")
            return False
        
        return self.route_processor.export_to_excel(self.routes_df, output_file)
    
    def validate_norms_storage(self) -> Dict:
        """Валидирует хранилище норм"""
        return self.norm_storage.validate_norms()
    
    def get_norm_storage_statistics(self) -> Dict:
        """Получает статистику хранилища норм"""
        return self.norm_storage.get_norm_statistics()
    
    def get_routes_data(self) -> pd.DataFrame:
        """Возвращает полные данные маршрутов"""
        return self.routes_df.copy() if self.routes_df is not None else pd.DataFrame()