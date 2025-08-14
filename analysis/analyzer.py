# analysis/analyzer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированный анализатор норм с Python 3.12 features
Vectorized operations и modern plotting
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, CubicSpline
from dataclasses import dataclass, field
from functools import lru_cache, cached_property
from pathlib import Path
from typing import Any, Protocol
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Python 3.12 enhanced type system
type NormPoints[T: float] = list[tuple[T, T]]
type AnalysisResults = dict[str, float | int | str]
type RouteData = dict[str, Any]

@dataclass(slots=True, frozen=True)
class NormDefinition:
    """Immutable norm definition with Python 3.12 slots optimization."""
    norm_id: int
    points: NormPoints[float]
    description: str = ""
    
    def __post_init__(self):
        if len(self.points) < 2:
            raise ValueError(f"Norm {self.norm_id}: minimum 2 points required")
        
        # Validate points are sorted and unique
        sorted_points = sorted(self.points, key=lambda p: p[0])
        x_values = [p[0] for p in sorted_points]
        if len(x_values) != len(set(x_values)):
            raise ValueError(f"Norm {self.norm_id}: duplicate X values not allowed")
        
        # Replace with sorted points
        object.__setattr__(self, 'points', sorted_points)
    
    @cached_property
    def interpolation_function(self):
        """Cached interpolation function for optimal performance."""
        x_vals, y_vals = zip(*self.points)
        
        if len(self.points) == 2:
            return interp1d(x_vals, y_vals, kind='linear', 
                          fill_value='extrapolate', bounds_error=False)
        else:
            try:
                return CubicSpline(x_vals, y_vals, bc_type='natural')
            except Exception:
                return interp1d(x_vals, y_vals, kind='quadratic', 
                              fill_value='extrapolate', bounds_error=False)
    
    @cached_property
    def x_range(self) -> tuple[float, float]:
        """Get X value range."""
        x_vals = [p[0] for p in self.points]
        return (min(x_vals), max(x_vals))
    
    def evaluate(self, x: float) -> float:
        """Evaluate norm at given point with caching."""
        try:
            return float(self.interpolation_function(x))
        except Exception as e:
            logger.warning(f"Norm {self.norm_id} evaluation failed at x={x}: {e}")
            return 0.0

class AnalysisStrategy(Protocol):
    """Protocol for analysis strategies."""
    def analyze(self, data: pd.DataFrame, norms: dict[int, NormDefinition]) -> pd.DataFrame: ...

class VectorizedAnalysisStrategy:
    """Vectorized analysis strategy for optimal performance."""
    
    def analyze(self, data: pd.DataFrame, norms: dict[int, NormDefinition]) -> pd.DataFrame:
        """Perform vectorized analysis on route data."""
        result_df = data.copy()
        
        # Initialize result columns
        result_df['Норма интерполированная'] = 0.0
        result_df['Отклонение, %'] = 0.0
        result_df['Статус'] = 'Не определен'
        
        # Group by norm for vectorized processing
        for norm_id, norm_def in norms.items():
            norm_mask = result_df['Номер нормы'] == norm_id
            norm_data = result_df[norm_mask]
            
            if norm_data.empty:
                continue
            
            try:
                # Vectorized norm evaluation
                load_values = norm_data['Нажатие на ось'].values
                norm_values = np.array([norm_def.evaluate(x) for x in load_values])
                actual_values = norm_data['Фактический удельный'].values
                
                # Calculate deviations
                valid_mask = norm_values > 0
                deviations = np.zeros_like(norm_values)
                deviations[valid_mask] = ((actual_values[valid_mask] - norm_values[valid_mask]) 
                                        / norm_values[valid_mask]) * 100
                
                # Vectorized status assignment
                statuses = self._classify_deviations_vectorized(deviations)
                
                # Update result DataFrame
                result_df.loc[norm_mask, 'Норма интерполированная'] = norm_values
                result_df.loc[norm_mask, 'Отклонение, %'] = deviations
                result_df.loc[norm_mask, 'Статус'] = statuses
                
            except Exception as e:
                logger.error(f"Vectorized analysis failed for norm {norm_id}: {e}")
                continue
        
        return result_df
    
    @staticmethod
    def _classify_deviations_vectorized(deviations: np.ndarray) -> np.ndarray:
        """Classify deviations using vectorized operations."""
        statuses = np.full_like(deviations, 'Не определен', dtype=object)
        
        # Use numpy where for efficient classification
        statuses = np.where(deviations >= 30, 'Экономия сильная', statuses)
        statuses = np.where((deviations >= 20) & (deviations < 30), 'Экономия средняя', statuses)
        statuses = np.where((deviations >= 5) & (deviations < 20), 'Экономия слабая', statuses)
        statuses = np.where((deviations >= -5) & (deviations < 5), 'Норма', statuses)
        statuses = np.where((deviations >= -20) & (deviations < -5), 'Перерасход слабый', statuses)
        statuses = np.where((deviations >= -30) & (deviations < -20), 'Перерасход средний', statuses)
        statuses = np.where(deviations < -30, 'Перерасход сильный', statuses)
        
        return statuses

class NormsManager:
    """Modern norms manager with optimized Excel processing."""
    
    def __init__(self):
        self.section_norms: dict[str, dict[int, NormDefinition]] = {}
        self.file_path: Path | None = None
    
    def load_norms(self, file_path: Path) -> bool:
        """Load norms from Excel file with optimized processing."""
        try:
            self.file_path = file_path
            self.section_norms.clear()
            
            excel_file = pd.ExcelFile(file_path)
            loaded_sections = 0
            
            for sheet_name in excel_file.sheet_names:
                logger.info(f"Processing sheet: {sheet_name}")
                
                # Read sheet with optimized settings
                df = pd.read_excel(file_path, sheet_name=sheet_name, 
                                 header=None, dtype=str)
                
                section_norms = self._parse_norms_from_sheet(df, sheet_name)
                
                if section_norms:
                    self.section_norms[sheet_name] = section_norms
                    loaded_sections += 1
                    logger.info(f"Loaded {len(section_norms)} norms for {sheet_name}")
            
            logger.info(f"Successfully loaded {loaded_sections} sections")
            return loaded_sections > 0
            
        except Exception as e:
            logger.error(f"Failed to load norms: {e}")
            return False
    
    def _parse_norms_from_sheet(self, df: pd.DataFrame, sheet_name: str) -> dict[int, NormDefinition]:
        """Parse norms from Excel sheet with error handling."""
        norms = {}
        
        for row_idx in range(len(df)):
            try:
                cell_value = str(df.iloc[row_idx, 0]) if pd.notna(df.iloc[row_idx, 0]) else ""
                
                if "Норма №" in cell_value:
                    norm_id = self._extract_norm_number(cell_value)
                    if norm_id is None:
                        continue
                    
                    # Parse description
                    description = ""
                    if row_idx + 1 < len(df):
                        desc_cell = df.iloc[row_idx + 1, 0]
                        if pd.notna(desc_cell):
                            description = str(desc_cell)
                    
                    # Parse data points
                    points = self._parse_norm_points(df, row_idx + 2)
                    
                    if len(points) >= 2:
                        try:
                            norm_def = NormDefinition(
                                norm_id=norm_id,
                                points=points,
                                description=description
                            )
                            norms[norm_id] = norm_def
                            logger.debug(f"Parsed norm {norm_id} with {len(points)} points")
                        except ValueError as e:
                            logger.warning(f"Invalid norm {norm_id} in {sheet_name}: {e}")
                    
            except Exception as e:
                logger.debug(f"Error parsing row {row_idx} in {sheet_name}: {e}")
                continue
        
        return norms
    
    @staticmethod
    def _extract_norm_number(text: str) -> int | None:
        """Extract norm number from text."""
        try:
            # Try different patterns
            import re
            patterns = [
                r'Норма\s*№\s*(\d+)',
                r'норма\s*(\d+)',
                r'№\s*(\d+)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return int(match.group(1))
            
            return None
        except (ValueError, AttributeError):
            return None
    
    def _parse_norm_points(self, df: pd.DataFrame, start_row: int) -> NormPoints[float]:
        """Parse norm points from DataFrame."""
        points = []
        
        try:
            # Check if we have enough rows
            if start_row + 1 >= len(df):
                return points
            
            # Get load and consumption rows
            load_row = df.iloc[start_row, 1:].dropna()
            consumption_row = df.iloc[start_row + 1, 1:].dropna()
            
            # Process point pairs
            min_length = min(len(load_row), len(consumption_row))
            
            for i in range(min_length):
                try:
                    load_val = float(load_row.iloc[i])
                    cons_val = float(consumption_row.iloc[i])
                    
                    # Validate values
                    if load_val > 0 and cons_val > 0:
                        points.append((load_val, cons_val))
                        
                except (ValueError, TypeError):
                    continue
            
        except Exception as e:
            logger.debug(f"Error parsing points from row {start_row}: {e}")
        
        return points
    
    def get_section_norms(self, section_name: str) -> dict[int, NormDefinition]:
        """Get norms for a specific section."""
        return self.section_norms.get(section_name, {})
    
    def get_all_sections(self) -> list[str]:
        """Get list of all sections."""
        return list(self.section_norms.keys())

class InteractiveNormsAnalyzer:
    """Main analyzer class with vectorized operations and modern features."""
    
    def __init__(self):
        self.routes_df: pd.DataFrame | None = None
        self.norms_manager = NormsManager()
        self.analysis_strategy = VectorizedAnalysisStrategy()
        self.analysis_results: dict[str, Any] = {}
    
    def load_data(self, routes_file: Path) -> bool:
        """Load route data with optimized processing."""
        try:
            logger.info(f"Loading routes from {routes_file}")
            
            # Optimized Excel reading
            self.routes_df = pd.read_excel(routes_file, engine='openpyxl')
            
            # Filter single-section routes using vectorized operations
            logger.info("Filtering single-section routes...")
            route_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
            single_routes = route_counts[route_counts == 1].index
            
            self.routes_df = (self.routes_df
                            .set_index(['Номер маршрута', 'Дата маршрута'])
                            .loc[single_routes]
                            .reset_index())
            
            # Add calculated fields
            if 'Нажатие на ось' not in self.routes_df.columns:
                self.routes_df['Нажатие на ось'] = (
                    self.routes_df['БРУТТО'] / self.routes_df['ОСИ']
                )
            
            # Filter valid norms
            initial_count = len(self.routes_df)
            self.routes_df = self.routes_df[self.routes_df['Номер нормы'].notna()]
            self.routes_df['Номер нормы'] = self.routes_df['Номер нормы'].astype(int)
            
            final_count = len(self.routes_df)
            unique_sections = self.routes_df['Наименование участка'].nunique()
            
            logger.info(f"Loaded {final_count} routes from {initial_count} total")
            logger.info(f"Found {unique_sections} unique sections")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load routes: {e}")
            return False
    
    def load_norms(self, norms_file: Path) -> bool:
        """Load norms data."""
        return self.norms_manager.load_norms(norms_file)
    
    def get_sections_list(self) -> list[str]:
        """Get list of available sections."""
        if self.routes_df is None:
            return []
        return sorted(self.routes_df['Наименование участка'].unique())
    
    def analyze_section(self, section_name: str, 
                       locomotive_filter=None, 
                       coefficient_manager=None,
                       use_coefficients: bool = False) -> tuple[pd.DataFrame, AnalysisResults]:
        """Analyze section with optional filtering and coefficients."""
        if self.routes_df is None:
            raise ValueError("No route data loaded")
        
        # Get section data
        section_data = self.routes_df[
            self.routes_df['Наименование участка'] == section_name
        ].copy()
        
        if section_data.empty:
            raise ValueError(f"No data found for section: {section_name}")
        
        # Apply locomotive filter if provided
        if locomotive_filter:
            section_data = locomotive_filter.filter_routes(section_data)
            if section_data.empty:
                raise ValueError("No data after applying locomotive filter")
        
        # Apply coefficients if requested
        if use_coefficients and coefficient_manager:
            section_data = self._apply_coefficients(section_data, coefficient_manager)
        
        # Get section norms
        section_norms = self.norms_manager.get_section_norms(section_name)
        if not section_norms:
            raise ValueError(f"No norms found for section: {section_name}")
        
        # Perform analysis
        analyzed_data = self.analysis_strategy.analyze(section_data, section_norms)
        
        # Calculate statistics
        stats = self._calculate_statistics(analyzed_data)
        
        # Cache results
        self.analysis_results[section_name] = {
            'data': analyzed_data,
            'stats': stats,
            'norms': section_norms
        }
        
        logger.info(f"Analyzed {len(analyzed_data)} routes for section {section_name}")
        return analyzed_data, stats
    
    def _apply_coefficients(self, df: pd.DataFrame, coefficient_manager) -> pd.DataFrame:
        """Apply locomotive coefficients using vectorized operations."""
        result_df = df.copy()
        result_df['Коэффициент'] = 1.0
        result_df['Фактический удельный исходный'] = result_df['Фактический удельный']
        
        # Vectorized coefficient application
        coefficients = []
        for _, row in result_df.iterrows():
            try:
                series = str(row.get('Серия локомотива', ''))
                number_str = str(row.get('Номер локомотива', '0'))
                number = int(number_str.lstrip('0') or '0')
                
                coeff = coefficient_manager.get_coefficient(series, number)
                coefficients.append(coeff)
                
            except (ValueError, TypeError):
                coefficients.append(1.0)
        
        # Apply coefficients vectorized
        result_df['Коэффициент'] = coefficients
        coeff_mask = np.array(coefficients) != 1.0
        
        result_df.loc[coeff_mask, 'Фактический удельный'] = (
            result_df.loc[coeff_mask, 'Фактический удельный'] / 
            result_df.loc[coeff_mask, 'Коэффициент']
        )
        
        applied_count = np.sum(coeff_mask)
        logger.info(f"Applied coefficients to {applied_count} routes")
        
        return result_df
    
    @staticmethod
    def _calculate_statistics(df: pd.DataFrame) -> AnalysisResults:
        """Calculate comprehensive analysis statistics."""
        total_routes = len(df)
        valid_data = df[df['Статус'] != 'Не определен']
        processed_routes = len(valid_data)
        
        if processed_routes == 0:
            return {
                'total_routes': total_routes,
                'processed_routes': 0,
                'mean_deviation': 0.0,
                'median_deviation': 0.0
            }
        
        # Count by status using vectorized operations
        status_counts = valid_data['Статус'].value_counts()
        
        # Statistical measures
        deviations = valid_data['Отклонение, %']
        
        return {
            'total_routes': total_routes,
            'processed_routes': processed_routes,
            'economy_strong': status_counts.get('Экономия сильная', 0),
            'economy_medium': status_counts.get('Экономия средняя', 0),
            'economy_weak': status_counts.get('Экономия слабая', 0),
            'normal': status_counts.get('Норма', 0),
            'overrun_weak': status_counts.get('Перерасход слабый', 0),
            'overrun_medium': status_counts.get('Перерасход средний', 0),
            'overrun_strong': status_counts.get('Перерасход сильный', 0),
            'mean_deviation': float(deviations.mean()),
            'median_deviation': float(deviations.median()),
            'std_deviation': float(deviations.std()),
            'processing_efficiency': (processed_routes / total_routes) * 100
        }
    
    def create_interactive_plot(self, section_name: str) -> go.Figure:
        """Create interactive Plotly visualization."""
        if section_name not in self.analysis_results:
            raise ValueError(f"No analysis results for section: {section_name}")
        
        result_data = self.analysis_results[section_name]
        analyzed_data = result_data['data']
        section_norms = result_data['norms']
        
        return self._create_plotly_figure(section_name, analyzed_data, section_norms)
    
    def _create_plotly_figure(self, section_name: str, 
                            analyzed_data: pd.DataFrame,
                            norms: dict[int, NormDefinition]) -> go.Figure:
        """Create optimized Plotly figure."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=[
                f"Нормы расхода: {section_name}",
                "Отклонения от нормы"
            ]
        )
        
        # Add norm curves with optimized rendering
        for norm_id, norm_def in norms.items():
            x_vals, y_vals = zip(*norm_def.points)
            x_range = np.linspace(min(x_vals), max(x_vals), 100)
            y_interp = [norm_def.evaluate(x) for x in x_range]
            
            # Norm curve
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=y_interp,
                    mode='lines',
                    name=f'Норма №{norm_id}',
                    line=dict(width=2),
                    hovertemplate='Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм<extra></extra>'
                ), row=1, col=1
            )
            
            # Control points
            fig.add_trace(
                go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='markers',
                    marker=dict(symbol='square', size=8, color='black'),
                    name=f'Точки нормы №{norm_id}',
                    showlegend=False,
                    hovertemplate='Опорная точка<br>Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм<extra></extra>'
                ), row=1, col=1
            )
        
        # Add route data with optimized grouping
        self._add_route_data_to_plot(fig, analyzed_data)
        
        # Add reference lines to deviation plot
        self._add_reference_lines(fig, analyzed_data)
        
        # Optimize layout
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
        
        # Update axes
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=2, col=1)
        fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм", row=1, col=1)
        fig.update_yaxes(title_text="Отклонение от нормы, %", row=2, col=1)
        
        return fig
    
    def _add_route_data_to_plot(self, fig: go.Figure, data: pd.DataFrame) -> None:
        """Add route data points to plot with optimized rendering."""
        valid_data = data[data['Статус'] != 'Не определен']
        
        # Color mapping for statuses
        color_map = {
            'Экономия сильная': '#7C3AED',
            'Экономия средняя': '#9333EA',
            'Экономия слабая': '#06B6D4',
            'Норма': '#22C55E',
            'Перерасход слабый': '#EAB308',
            'Перерасход средний': '#F97316',
            'Перерасход сильный': '#DC2626'
        }
        
        # Group by status for efficient rendering
        for status, color in color_map.items():
            status_data = valid_data[valid_data['Статус'] == status]
            if status_data.empty:
                continue
            
            # Create hover text efficiently
            hover_texts = self._create_hover_texts(status_data)
            
            # Upper plot - actual consumption
            fig.add_trace(
                go.Scattergl(  # Use WebGL for better performance
                    x=status_data['Нажатие на ось'],
                    y=status_data['Фактический удельный'],
                    mode='markers',
                    marker=dict(color=color, size=8, opacity=0.8, 
                               line=dict(color='black', width=0.5)),
                    name=f'{status} ({len(status_data)})',
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ), row=1, col=1
            )
            
            # Lower plot - deviations
            fig.add_trace(
                go.Scattergl(
                    x=status_data['Нажатие на ось'],
                    y=status_data['Отклонение, %'],
                    mode='markers',
                    marker=dict(color=color, size=10, opacity=0.8,
                               line=dict(color='black', width=0.5)),
                    name=f'{status} отклонения',
                    showlegend=False,
                    text=hover_texts,
                    hovertemplate='%{text}<extra></extra>'
                ), row=2, col=1
            )
    
    @staticmethod
    def _create_hover_texts(data: pd.DataFrame) -> list[str]:
        """Create hover text for route data points."""
        hover_texts = []
        for _, row in data.iterrows():
            text = (
                f"Маршрут №{row['Номер маршрута']}<br>"
                f"Дата: {row['Дата маршрута']}<br>"
            )
            
            # Add locomotive info if available
            if pd.notna(row.get('Серия локомотива')):
                text += f"Локомотив: {row['Серия локомотива']}"
                if pd.notna(row.get('Номер локомотива')):
                    text += f" №{row['Номер локомотива']}"
                text += "<br>"
            
            # Add coefficient info if available
            if 'Коэффициент' in row.index and row['Коэффициент'] != 1.0:
                text += f"Коэффициент: {row['Коэффициент']:.3f}<br>"
            
            text += (
                f"Нажатие: {row['Нажатие на ось']:.2f} т/ось<br>"
                f"Факт: {row['Фактический удельный']:.1f}<br>"
                f"Норма: {row['Норма интерполированная']:.1f}<br>"
                f"Отклонение: {row['Отклонение, %']:.1f}%"
            )
            
            hover_texts.append(text)
        
        return hover_texts
    
    @staticmethod
    def _add_reference_lines(fig: go.Figure, data: pd.DataFrame) -> None:
        """Add reference lines to deviation plot."""
        valid_data = data[data['Статус'] != 'Не определен']
        if valid_data.empty:
            return
        
        x_range = [
            valid_data['Нажатие на ось'].min() - 1,
            valid_data['Нажатие на ось'].max() + 1
        ]
        
        # Reference lines
        reference_lines = [
            (5, '#22C55E', 'dash'),
            (-5, '#22C55E', 'dash'),
            (20, '#F97316', 'dot'),
            (-20, '#F97316', 'dot'),
            (0, 'black', 'solid')
        ]
        
        for y_val, color, dash in reference_lines:
            fig.add_trace(
                go.Scatter(
                    x=x_range, y=[y_val, y_val],
                    mode='lines',
                    line=dict(color=color, dash=dash, width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ), row=2, col=1
            )
        
        # Add shaded normal range
        fig.add_trace(
            go.Scatter(
                x=x_range + x_range[::-1],
                y=[-5, -5, 5, 5],
                fill='toself',
                fillcolor='rgba(34, 197, 94, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip'
            ), row=2, col=1
        )