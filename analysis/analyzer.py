# analysis/analyzer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированный анализатор норм расхода электроэнергии с использованием 
современных возможностей Python 3.12 и эффективных алгоритмов.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, CubicSpline
import warnings
warnings.filterwarnings('ignore')
import logging
from typing import Optional
from pathlib import Path
from dataclasses import dataclass, field

from .data_models import (
    ProcessedRoute, AnalysisResult, NormDefinition, 
    ValidationResult, ProcessingStats, DataParser, DataProcessor
)
from .utils import MathUtils, ConfigManager
from .html_route_processor import OptimizedHTMLRouteProcessor
from .html_norm_processor import OptimizedHTMLNormProcessor

logger = logging.getLogger(__name__)

type PlotlyFigure = go.Figure
type StatisticsDict = dict[str, int | float]
type FilterFunction = callable[[pd.DataFrame], pd.DataFrame]

@dataclass(slots=True)
class AnalysisConfig:
    """Конфигурация анализа с оптимизацией памяти."""
    single_section_only: bool = False
    use_coefficients: bool = False
    exclude_low_work: bool = False
    min_routes_threshold: int = 1
    interpolation_points: int = 100
    
    def validate(self) -> ValidationResult:
        """Валидирует конфигурацию."""
        if self.min_routes_threshold < 0:
            return False, "Minimum routes threshold must be non-negative"
        if self.interpolation_points < 10:
            return False, "Interpolation points must be at least 10"
        return True, "Valid configuration"

@dataclass(slots=True)
class AnalysisCache:
    """Кэш для оптимизации повторных вычислений."""
    norm_functions: dict[str, callable] = field(default_factory=dict)
    section_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    statistics: dict[str, StatisticsDict] = field(default_factory=dict)
    
    def clear(self) -> None:
        """Очищает кэш."""
        self.norm_functions.clear()
        self.section_data.clear()
        self.statistics.clear()
        logger.debug("Analysis cache cleared")

class OptimizedNormsAnalyzer:
    """
    Оптимизированный анализатор норм расхода электроэнергии.
    
    Использует современные практики Python 3.12:
    - Композицию вместо наследования
    - Типизацию с новым синтаксисом
    - Оптимизированные структуры данных
    - Кэширование для улучшения производительности
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Инициализирует анализатор с конфигурацией."""
        self.config = config or AnalysisConfig()
        self.cache = AnalysisCache()
        self.config_manager = ConfigManager()
        
        # Композиция: внедряем процессоры как зависимости
        self.route_processor = OptimizedHTMLRouteProcessor()
        self.norm_processor = OptimizedHTMLNormProcessor()
        
        # Основные данные
        self.routes_df: Optional[pd.DataFrame] = None
        self.sections_norms_map: dict[str, list[str]] = {}
        
        logger.info("Optimized norms analyzer initialized")
    
    def load_routes_from_html(self, html_files: list[Path | str]) -> bool:
        """
        Загружает маршруты из HTML файлов с оптимизацией.
        
        Args:
            html_files: Список путей к HTML файлам
            
        Returns:
            True если загрузка успешна
        """
        logger.info(f"Loading routes from {len(html_files)} HTML files")
        
        try:
            # Конвертируем в Path объекты для современного API
            paths = [Path(f) for f in html_files]
            
            # Обрабатываем файлы с оптимизированным процессором
            self.routes_df = self.route_processor.process_html_files(paths)
            
            if self.routes_df is None or self.routes_df.empty:
                logger.error("No routes loaded from HTML files")
                return False
            
            # Оптимизация DataFrame
            self.routes_df = self._optimize_dataframe(self.routes_df)
            
            logger.info(f"Loaded {len(self.routes_df)} route records")
            self._log_routes_statistics()
            self._build_sections_norms_map()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading routes: {e}")
            return False
    
    def load_norms_from_html(self, html_files: list[Path | str]) -> bool:
        """
        Загружает нормы из HTML файлов с оптимизацией.
        
        Args:
            html_files: Список путей к HTML файлам норм
            
        Returns:
            True если загрузка успешна
        """
        logger.info(f"Loading norms from {len(html_files)} HTML files")
        
        try:
            paths = [Path(f) for f in html_files]
            success = self.norm_processor.process_html_files(paths)
            
            if success:
                # Очищаем кэш функций норм при обновлении
                self.cache.norm_functions.clear()
                logger.info("Norms loaded successfully")
                return True
            else:
                logger.error("Failed to load norms")
                return False
                
        except Exception as e:
            logger.error(f"Error loading norms: {e}")
            return False
    
    def get_sections_list(self) -> list[str]:
        """Возвращает отсортированный список участков."""
        if self.routes_df is None or self.routes_df.empty:
            return []
        
        sections = self.routes_df['Наименование участка'].dropna().unique()
        return sorted(sections.tolist())
    
    def get_norms_for_section(self, section_name: str) -> list[str]:
        """Возвращает список норм для участка."""
        return self.sections_norms_map.get(section_name, [])
    
    def get_norms_with_counts_for_section(
        self, 
        section_name: str, 
        single_section_only: bool = False
    ) -> list[tuple[str, int]]:
        """
        Возвращает список норм для участка с количеством маршрутов.
        
        Args:
            section_name: Название участка
            single_section_only: Фильтр только маршрутов с одним участком
            
        Returns:
            Список кортежей (норма, количество_маршрутов)
        """
        if self.routes_df is None or self.routes_df.empty:
            return []
        
        # Кэшируем результат для оптимизации
        cache_key = f"{section_name}_{single_section_only}"
        if cache_key in self.cache.statistics:
            cached_stats = self.cache.statistics[cache_key]
            return [(str(norm), count) for norm, count in cached_stats.items()]
        
        # Получаем данные участка
        section_data = self._get_section_data_optimized(section_name, single_section_only)
        
        if section_data.empty:
            return []
        
        # Векторизованный подсчет норм
        norm_counts = section_data['Номер нормы'].value_counts()
        
        # Формируем результат
        result = []
        for norm in self.sections_norms_map.get(section_name, []):
            count = norm_counts.get(norm, 0) if norm.isdigit() else 0
            result.append((norm, count))
        
        # Сортируем по номеру нормы
        result.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        
        # Кэшируем результат
        self.cache.statistics[cache_key] = {norm: count for norm, count in result}
        
        return result
    
    def analyze_section(
        self,
        section_name: str,
        norm_id: Optional[str] = None,
        single_section_only: bool = False,
        locomotive_filter = None,
        coefficients_manager = None,
        use_coefficients: bool = False
    ) -> tuple[Optional[PlotlyFigure], Optional[StatisticsDict], Optional[str]]:
        """
        Анализирует участок с оптимизированными алгоритмами.
        
        Args:
            section_name: Название участка
            norm_id: Конкретная норма (опционально)
            single_section_only: Фильтр по одному участку
            locomotive_filter: Фильтр локомотивов
            coefficients_manager: Менеджер коэффициентов
            use_coefficients: Применять коэффициенты
            
        Returns:
            Кортеж (график, статистика, ошибка)
        """
        logger.info(f"Analyzing section: {section_name}, norm: {norm_id}, single_only: {single_section_only}")
        
        if self.routes_df is None or self.routes_df.empty:
            return None, None, "Route data not loaded"
        
        try:
            # Получаем данные участка с оптимизацией
            section_data = self._get_section_data_optimized(section_name, single_section_only)
            
            if section_data.empty:
                return None, None, f"No data for section {section_name}"
            
            # Применяем фильтр нормы
            if norm_id:
                section_data = section_data[
                    section_data['Номер нормы'].astype(str) == str(norm_id)
                ]
                if section_data.empty:
                    filter_text = " with single section" if single_section_only else ""
                    return None, None, f"No routes{filter_text} for section {section_name} with norm {norm_id}"
            
            # Применяем фильтр локомотивов
            if locomotive_filter:
                section_data = locomotive_filter.filter_routes(section_data)
                if section_data.empty:
                    return None, None, "No data after locomotive filter"
            
            # Применяем коэффициенты
            if use_coefficients and coefficients_manager:
                section_data = self._apply_coefficients_vectorized(section_data, coefficients_manager)
            
            # Анализируем данные
            analyzed_data, norm_functions = self._analyze_section_data_optimized(
                section_name, section_data, norm_id
            )
            
            if analyzed_data.empty:
                return None, None, f"Failed to analyze section {section_name}"
            
            # Создаем график
            fig = self._create_optimized_plot(section_name, analyzed_data, norm_functions, norm_id, single_section_only)
            
            # Вычисляем статистику
            statistics = self._calculate_statistics_vectorized(analyzed_data)
            
            logger.info(f"Section {section_name} analysis completed successfully")
            return fig, statistics, None
            
        except Exception as e:
            logger.error(f"Error analyzing section {section_name}: {e}")
            return None, None, f"Analysis error: {str(e)}"
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Оптимизирует DataFrame для экономии памяти."""
        optimized_df = df.copy()
        
        # Оптимизация типов данных
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Категоризация часто повторяющихся строк
        categorical_candidates = ['Наименование участка', 'Серия локомотива', 'Депо']
        for col in categorical_candidates:
            if col in optimized_df.columns:
                unique_ratio = optimized_df[col].nunique() / len(optimized_df)
                if unique_ratio < 0.5:  # Если уникальных значений меньше 50%
                    optimized_df[col] = optimized_df[col].astype('category')
        
        memory_reduction = (df.memory_usage(deep=True).sum() - 
                          optimized_df.memory_usage(deep=True).sum()) / 1024**2
        
        logger.info(f"DataFrame optimized: {memory_reduction:.2f} MB memory saved")
        return optimized_df
    
    def _get_section_data_optimized(self, section_name: str, single_section_only: bool) -> pd.DataFrame:
        """Получает данные участка с оптимизацией и кэшированием."""
        cache_key = f"{section_name}_{single_section_only}"
        
        if cache_key in self.cache.section_data:
            return self.cache.section_data[cache_key]
        
        # Фильтруем по участку
        section_data = self.routes_df[
            self.routes_df['Наименование участка'] == section_name
        ].copy()
        
        if single_section_only and not section_data.empty:
            # Векторизованная фильтрация маршрутов с одним участком
            route_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
            single_routes = route_counts[route_counts == 1].index
            
            section_data = section_data.set_index(['Номер маршрута', 'Дата маршрута'])
            section_data = section_data.loc[section_data.index.intersection(single_routes)]
            section_data = section_data.reset_index()
        
        # Кэшируем результат
        self.cache.section_data[cache_key] = section_data
        
        return section_data
    
    def _apply_coefficients_vectorized(self, df: pd.DataFrame, coefficients_manager) -> pd.DataFrame:
        """Применяет коэффициенты векторизованно для улучшения производительности."""
        df_with_coeff = df.copy()
        
        # Создаем векторизованную функцию получения коэффициентов
        def get_coefficient_vectorized(series, number):
            try:
                return coefficients_manager.get_coefficient(str(series), int(number))
            except:
                return 1.0
        
        # Применяем векторизованно
        mask = df_with_coeff['Серия локомотива'].notna() & df_with_coeff['Номер локомотива'].notna()
        
        if mask.any():
            coefficients = df_with_coeff.loc[mask].apply(
                lambda row: get_coefficient_vectorized(
                    row['Серия локомотива'], 
                    row['Номер локомотива']
                ), axis=1
            )
            
            df_with_coeff.loc[mask, 'Коэффициент'] = coefficients
            df_with_coeff.loc[mask, 'Факт. удельный исходный'] = df_with_coeff.loc[mask, 'Расход фактический']
            
            # Применяем коэффициенты векторизованно
            non_unity_mask = mask & (coefficients != 1.0)
            if non_unity_mask.any():
                df_with_coeff.loc[non_unity_mask, 'Расход фактический'] = (
                    df_with_coeff.loc[non_unity_mask, 'Расход фактический'] / 
                    df_with_coeff.loc[non_unity_mask, 'Коэффициент']
                )
        
        applied_count = (df_with_coeff.get('Коэффициент', pd.Series([1.0])) != 1.0).sum()
        logger.info(f"Applied {applied_count} coefficients vectorized")
        
        return df_with_coeff
    
    def _analyze_section_data_optimized(
        self, 
        section_name: str, 
        routes_df: pd.DataFrame, 
        specific_norm_id: Optional[str] = None
    ) -> tuple[pd.DataFrame, dict]:
        """Оптимизированный анализ данных участка."""
        analyzed_df = routes_df.copy()
        
        # Инициализируем колонки для анализа
        analyzed_df['Норма интерполированная'] = 0.0
        analyzed_df['Отклонение, %'] = 0.0
        analyzed_df['Статус'] = 'Не определен'
        
        # Получаем нормы для анализа
        norm_numbers = [specific_norm_id] if specific_norm_id else analyzed_df['Номер нормы'].dropna().unique()
        
        # Создаем функции интерполяции с кэшированием
        norm_functions = {}
        for norm_number in norm_numbers:
            norm_id = str(int(norm_number)) if pd.notna(norm_number) else None
            if norm_id and norm_id not in self.cache.norm_functions:
                norm_data = self.norm_processor.get_norm(norm_id)
                if norm_data and norm_data.get('points'):
                    try:
                        func = self._create_interpolation_function(norm_data['points'])
                        self.cache.norm_functions[norm_id] = func
                        norm_functions[norm_id] = {
                            'function': func,
                            'points': norm_data['points'],
                            'data': norm_data
                        }
                    except Exception as e:
                        logger.error(f"Error creating function for norm {norm_id}: {e}")
            elif norm_id in self.cache.norm_functions:
                norm_functions[norm_id] = {
                    'function': self.cache.norm_functions[norm_id],
                    'points': [],  # Загрузим при необходимости
                    'data': {}
                }
        
        if not norm_functions:
            logger.warning(f"No norm functions found for section {section_name}")
            return analyzed_df, {}
        
        # Векторизованная интерполяция и расчет отклонений
        self._interpolate_and_calculate_vectorized(analyzed_df, norm_functions)
        
        logger.info(f"Analyzed {len(analyzed_df)} records for section {section_name}")
        return analyzed_df, norm_functions
    
    def _interpolate_and_calculate_vectorized(self, df: pd.DataFrame, norm_functions: dict) -> None:
        """Векторизованная интерполяция и расчет отклонений."""
        for norm_id, norm_data in norm_functions.items():
            norm_mask = df['Номер нормы'].astype(str) == norm_id
            
            if not norm_mask.any():
                continue
            
            norm_subset = df[norm_mask]
            func = norm_data['function']
            
            # Векторизованный расчет нажатия на ось
            axle_loads = self._calculate_axle_loads_vectorized(norm_subset)
            valid_loads_mask = axle_loads.notna() & (axle_loads > 0)
            
            if not valid_loads_mask.any():
                continue
            
            # Векторизованная интерполяция
            valid_loads = axle_loads[valid_loads_mask]
            try:
                interpolated_values = pd.Series(func(valid_loads), index=valid_loads.index)
                df.loc[interpolated_values.index, 'Норма интерполированная'] = interpolated_values
                
                # Векторизованный расчет фактических значений и отклонений
                actual_values = norm_subset.loc[valid_loads.index, 'Факт уд'].fillna(
                    norm_subset.loc[valid_loads.index, 'Расход фактический']
                )
                
                valid_actual_mask = actual_values.notna() & (interpolated_values > 0)
                if valid_actual_mask.any():
                    deviations = (
                        (actual_values[valid_actual_mask] - interpolated_values[valid_actual_mask]) /
                        interpolated_values[valid_actual_mask] * 100
                    )
                    
                    df.loc[deviations.index, 'Отклонение, %'] = deviations
                    
                    # Векторизованное определение статуса
                    df.loc[deviations.index, 'Статус'] = self._categorize_deviations_vectorized(deviations)
                    
            except Exception as e:
                logger.error(f"Error in vectorized interpolation for norm {norm_id}: {e}")
    
    def _calculate_axle_loads_vectorized(self, df: pd.DataFrame) -> pd.Series:
        """Векторизованный расчет нажатия на ось."""
        # Сначала пробуем готовые значения
        axle_loads = df['Нажатие на ось'].copy()
        
        # Для строк без готового значения вычисляем из БРУТТО/ОСИ
        missing_mask = axle_loads.isna() | (axle_loads == "-")
        
        if missing_mask.any():
            brutto_values = pd.to_numeric(df.loc[missing_mask, 'БРУТТО'], errors='coerce')
            osi_values = pd.to_numeric(df.loc[missing_mask, 'ОСИ'], errors='coerce')
            
            valid_calc_mask = (brutto_values.notna() & osi_values.notna() & 
                              (brutto_values != 0) & (osi_values != 0))
            
            if valid_calc_mask.any():
                calculated_loads = brutto_values[valid_calc_mask] / osi_values[valid_calc_mask]
                axle_loads.loc[calculated_loads.index] = calculated_loads
        
        return pd.to_numeric(axle_loads, errors='coerce')
    
    def _categorize_deviations_vectorized(self, deviations: pd.Series) -> pd.Series:
        """Векторизованная категоризация отклонений."""
        conditions = [
            deviations < -30,
            (deviations >= -30) & (deviations < -20),
            (deviations >= -20) & (deviations < -5),
            (deviations >= -5) & (deviations <= 5),
            (deviations > 5) & (deviations <= 20),
            (deviations > 20) & (deviations <= 30),
            deviations > 30
        ]
        
        choices = [
            'Экономия сильная',
            'Экономия средняя', 
            'Экономия слабая',
            'Норма',
            'Перерасход слабый',
            'Перерасход средний',
            'Перерасход сильный'
        ]
        
        return pd.Series(np.select(conditions, choices, default='Не определен'), index=deviations.index)
    
    def _create_interpolation_function(self, points: list[tuple[float, float]]):
        """Создает оптимизированную функцию интерполяции."""
        if len(points) < 2:
            raise ValueError("Need at least 2 points for interpolation")
        
        sorted_points = sorted(points, key=lambda x: x[0])
        x_values = np.array([p[0] for p in sorted_points])
        y_values = np.array([p[1] for p in sorted_points])
        
        if len(np.unique(x_values)) != len(x_values):
            raise ValueError("Duplicate X values found")
        
        if len(points) == 2:
            return interp1d(x_values, y_values, kind='linear', 
                          fill_value='extrapolate', bounds_error=False)
        else:
            try:
                return CubicSpline(x_values, y_values, bc_type='natural')
            except:
                try:
                    return interp1d(x_values, y_values, kind='quadratic',
                                  fill_value='extrapolate', bounds_error=False)
                except:
                    return interp1d(x_values, y_values, kind='linear',
                                  fill_value='extrapolate', bounds_error=False)
    
    def _calculate_statistics_vectorized(self, df: pd.DataFrame) -> StatisticsDict:
        """Векторизованное вычисление статистики."""
        total = len(df)
        valid_routes = df[df['Статус'] != 'Не определен']
        processed = len(valid_routes)
        
        if processed == 0:
            return {'total': total, 'processed': processed}
        
        # Векторизованный подсчет статусов
        status_counts = valid_routes['Статус'].value_counts()
        
        detailed_stats = {
            'economy_strong': status_counts.get('Экономия сильная', 0),
            'economy_medium': status_counts.get('Экономия средняя', 0),
            'economy_weak': status_counts.get('Экономия слабая', 0),
            'normal': status_counts.get('Норма', 0),
            'overrun_weak': status_counts.get('Перерасход слабый', 0),
            'overrun_medium': status_counts.get('Перерасход средний', 0),
            'overrun_strong': status_counts.get('Перерасход сильный', 0)
        }
        
        # Векторизованные статистические расчеты
        deviations = valid_routes['Отклонение, %']
        
        return {
            'total': total,
            'processed': processed,
            'economy': detailed_stats['economy_strong'] + detailed_stats['economy_medium'] + detailed_stats['economy_weak'],
            'normal': detailed_stats['normal'],
            'overrun': detailed_stats['overrun_weak'] + detailed_stats['overrun_medium'] + detailed_stats['overrun_strong'],
            'mean_deviation': float(deviations.mean()),
            'median_deviation': float(deviations.median()),
            'detailed_stats': detailed_stats
        }
    
    def _create_optimized_plot(
        self, 
        section_name: str, 
        routes_df: pd.DataFrame,
        norm_functions: dict,
        specific_norm_id: Optional[str] = None,
        single_section_only: bool = False
    ) -> PlotlyFigure:
        """Создает оптимизированный интерактивный график."""
        title_suffix = f" (норма {specific_norm_id})" if specific_norm_id else ""
        filter_suffix = " [только один участок]" if single_section_only else ""
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f"Нормы расхода для участка: {section_name}{title_suffix}{filter_suffix}",
                "Отклонение фактического расхода от нормы"
            )
        )
        
        # Оптимизированная отрисовка норм
        self._add_norm_curves_optimized(fig, norm_functions, specific_norm_id)
        
        # Оптимизированная отрисовка точек маршрутов
        self._add_route_points_optimized(fig, routes_df)
        
        # Настройка осей и layout
        self._configure_plot_layout(fig)
        
        return fig
    
    def _add_norm_curves_optimized(self, fig: PlotlyFigure, norm_functions: dict, specific_norm_id: Optional[str]) -> None:
        """Оптимизированная отрисовка кривых норм."""
        for norm_id, norm_data in norm_functions.items():
            if specific_norm_id and norm_id != specific_norm_id:
                continue
            
            points = norm_data.get('points', [])
            if not points:
                continue
            
            x_vals = np.array([p[0] for p in points])
            y_vals = np.array([p[1] for p in points])
            
            # Оптимизированная интерполяция
            x_interp = np.linspace(x_vals.min(), x_vals.max(), self.config.interpolation_points)
            norm_func = norm_data['function']
            y_interp = norm_func(x_interp)
            
            # Добавляем кривую
            fig.add_trace(
                go.Scatter(
                    x=x_interp, y=y_interp,
                    mode='lines', name=f'Норма №{norm_id}',
                    line=dict(width=2),
                    hovertemplate='Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм'
                ), row=1, col=1
            )
            
            # Добавляем опорные точки
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='markers',
                    marker=dict(symbol='square', size=8, color='black'),
                    name=f'Опорные точки нормы №{norm_id}',
                    hovertemplate='Опорная точка<br>Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм'
                ), row=1, col=1
            )
    
    def _add_route_points_optimized(self, fig: PlotlyFigure, routes_df: pd.DataFrame) -> None:
        """Оптимизированная отрисовка точек маршрутов."""
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен']
        
        if valid_routes.empty:
            return
        
        # Группируем по статусам для оптимизации
        status_groups = valid_routes.groupby('Статус')
        
        status_colors = {
            'Экономия сильная': '#006400',
            'Экономия средняя': '#228B22', 
            'Экономия слабая': '#32CD32',
            'Норма': '#FFD700',
            'Перерасход слабый': '#FF8C00',
            'Перерасход средний': '#FF4500',
            'Перерасход сильный': '#DC143C'
        }
        
        for status, group in status_groups:
            if status not in status_colors:
                continue
            
            # Векторизованная подготовка данных
            axle_loads = self._calculate_axle_loads_vectorized(group)
            consumption_values = group['Факт уд'].fillna(group['Расход фактический'])
            
            valid_mask = axle_loads.notna() & consumption_values.notna()
            
            if not valid_mask.any():
                continue
            
            valid_data = group[valid_mask]
            x_values = axle_loads[valid_mask].values
            y_values = consumption_values[valid_mask].values
            
            # Создаем hover текст векторизованно
            hover_texts = self._create_hover_texts_vectorized(valid_data, x_values, y_values)
            
            # Добавляем точки на верхний график
            fig.add_trace(
                go.Scatter(
                    x=x_values, y=y_values,
                    mode='markers',
                    name=f'{status} ({len(valid_data)})',
                    marker=dict(
                        color=status_colors[status],
                        size=8, opacity=0.8,
                        line=dict(color='black', width=0.5)
                    ),
                    hovertemplate='%{text}',
                    text=hover_texts
                ), row=1, col=1
            )
            
            # Добавляем точки на нижний график (отклонения)
            deviations = valid_data['Отклонение, %'].values
            
            fig.add_trace(
                go.Scatter(
                    x=x_values, y=deviations,
                    mode='markers',
                    name=f'{status} ({len(valid_data)})',
                    marker=dict(
                        color=status_colors[status],
                        size=10, opacity=0.8,
                        line=dict(color='black', width=0.5)
                    ),
                    hovertemplate='%{text}',
                    text=hover_texts,
                    showlegend=False
                ), row=2, col=1
            )
        
        # Добавляем граничные линии на нижний график
        self._add_boundary_lines_optimized(fig, valid_routes)
    
    def _create_hover_texts_vectorized(self, df: pd.DataFrame, x_values: np.ndarray, y_values: np.ndarray) -> list[str]:
        """Векторизованное создание hover текстов."""
        hover_texts = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            text = (
                f"Маршрут №{row.get('Номер маршрута', 'N/A')}<br>"
                f"Дата: {row.get('Дата маршрута', 'N/A')}<br>"
                f"Локомотив: {row.get('Серия локомотива', '')} №{row.get('Номер локомотива', '')}<br>"
            )
            
            # Добавляем информацию о коэффициенте если есть
            coeff = row.get('Коэффициент')
            if coeff is not None and coeff != 1.0:
                text += f"Коэффициент: {coeff:.3f}<br>"
                original = row.get('Факт. удельный исходный')
                if original is not None:
                    text += f"Факт исходный: {original:.1f}<br>"
            
            text += (
                f"Нажатие: {x_values[idx]:.2f} т/ось<br>"
                f"Факт: {y_values[idx]:.1f}<br>"
                f"Норма: {row.get('Норма интерполированная', 'N/A'):.1f}<br>"
                f"Отклонение: {row.get('Отклонение, %', 'N/A'):.1f}%"
            )
            
            hover_texts.append(text)
        
        return hover_texts
    
    def _add_boundary_lines_optimized(self, fig: PlotlyFigure, routes_df: pd.DataFrame) -> None:
        """Оптимизированное добавление граничных линий."""
        if routes_df.empty:
            return
        
        axle_loads = self._calculate_axle_loads_vectorized(routes_df)
        valid_loads = axle_loads.dropna()
        
        if valid_loads.empty:
            return
        
        x_range = [valid_loads.min() - 1, valid_loads.max() + 1]
        
        # Конфигурация линий
        lines_config = [
            (5, '#FFD700', 'dash'),
            (-5, '#FFD700', 'dash'),
            (20, '#FF4500', 'dot'),
            (-20, '#FF4500', 'dot'),
            (30, '#DC143C', 'dashdot'),
            (-30, '#DC143C', 'dashdot'),
            (0, 'black', 'solid')
        ]
        
        for y_val, color, dash in lines_config:
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=[y_val, y_val],
                    mode='lines',
                    line=dict(color=color, dash=dash, width=2 if y_val == 0 else 1),
                    showlegend=False, hoverinfo='skip'
                ), row=2, col=1
            )
        
        # Добавляем зону нормы
        fig.add_trace(
            go.Scatter(
                x=x_range + x_range[::-1], y=[-5, -5, 5, 5],
                fill='toself', fillcolor='rgba(255, 215, 0, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False, hoverinfo='skip'
            ), row=2, col=1
        )
    
    def _configure_plot_layout(self, fig: PlotlyFigure) -> None:
        """Настраивает layout графика."""
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=1, col=1)
        fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм брутто", row=1, col=1)
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=2, col=1)
        fig.update_yaxes(title_text="Отклонение от нормы, %", row=2, col=1)
        
        fig.update_layout(
            height=1000,
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
    
    def _build_sections_norms_map(self) -> None:
        """Строит карту участков и их норм с оптимизацией."""
        if self.routes_df is None or self.routes_df.empty:
            return
        
        # Векторизованная группировка
        sections_groups = (
            self.routes_df
            .dropna(subset=['Наименование участка', 'Номер нормы'])
            .groupby('Наименование участка')['Номер нормы']
            .apply(lambda x: sorted(list(x.astype(str).unique())))
            .to_dict()
        )
        
        self.sections_norms_map = {
            section: [norm for norm in norms if norm != 'nan']
            for section, norms in sections_groups.items()
        }
        
        logger.info(f"Built sections-norms map: {len(self.sections_norms_map)} sections")
    
    def _log_routes_statistics(self) -> None:
        """Логирует статистику маршрутов."""
        if self.routes_df is None:
            return
        
        stats = self.route_processor.get_processing_stats()
        
        logger.info("=== ROUTES STATISTICS ===")
        logger.info(f"Files processed: {stats['total_files']}")
        logger.info(f"Routes found: {stats['total_routes_found']}")
        logger.info(f"Unique routes: {stats['unique_routes']}")
        logger.info(f"Final records: {len(self.routes_df)}")
        
        sections_count = self.routes_df['Наименование участка'].nunique()
        norms_count = self.routes_df['Номер нормы'].nunique()
        
        logger.info(f"Unique sections: {sections_count}")
        logger.info(f"Unique norms: {norms_count}")
    
    # Методы для совместимости с существующим интерфейсом
    def get_routes_data(self) -> pd.DataFrame:
        """Возвращает данные маршрутов."""
        return self.routes_df.copy() if self.routes_df is not None else pd.DataFrame()
    
    def export_routes_to_excel(self, output_file: Path | str) -> bool:
        """Экспортирует маршруты в Excel."""
        if self.routes_df is None or self.routes_df.empty:
            return False
        return self.route_processor.export_to_excel(self.routes_df, output_file)
    
    def get_norm_storage_info(self) -> dict:
        """Возвращает информацию о хранилище норм."""
        return self.norm_processor.get_storage_info()
    
    def validate_norms_storage(self) -> dict:
        """Валидирует хранилище норм."""
        return self.norm_processor.validate_norms()
    
    def get_norm_storage_statistics(self) -> dict:
        """Возвращает статистику хранилища норм."""
        return self.norm_processor.get_storage_statistics()
