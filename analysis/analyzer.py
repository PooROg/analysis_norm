# Файл: analysis/analyzer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, CubicSpline
import warnings
warnings.filterwarnings('ignore')
from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager

class InteractiveNormsAnalyzer:
    """Класс для интерактивного анализа норм расхода электроэнергии"""
    
    def __init__(self, routes_file='Processed_Routes.xlsx', norms_file='Нормы участков.xlsx'):
        self.routes_file = routes_file
        self.norms_file = norms_file
        self.routes_df = None
        self.norms_data = {}
        self.analysis_results = {}
        
    def load_data(self):
        try:
            print("\n[1] Загрузка данных маршрутов...")
            self.routes_df = pd.read_excel(self.routes_file)
            print("    Фильтрация маршрутов с одним участком...")
            route_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
            single_routes = route_counts[route_counts == 1].index
            self.routes_df = self.routes_df.set_index(['Номер маршрута', 'Дата маршрута']).loc[single_routes].reset_index()
            if 'Нажатие на ось' not in self.routes_df.columns:
                self.routes_df['Нажатие на ось'] = self.routes_df['БРУТТО'] / self.routes_df['ОСИ']
            initial_count = len(self.routes_df)
            self.routes_df = self.routes_df[self.routes_df['Номер нормы'].notna()]
            self.routes_df['Номер нормы'] = self.routes_df['Номер нормы'].astype(int)
            print(f"    Загружено маршрутов: {initial_count}")
            print(f"    С указанной нормой: {len(self.routes_df)}")
            print(f"    Уникальных участков: {self.routes_df['Наименование участка'].nunique()}")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return False
    
    def load_norms(self):
        try:
            print("\n[2] Загрузка норм...")
            excel_file = pd.ExcelFile(self.norms_file)
            for sheet_name in excel_file.sheet_names:
                print(f"\n    Обработка листа: {sheet_name}")
                df = pd.read_excel(self.norms_file, sheet_name=sheet_name, header=None)
                sheet_norms = self.parse_norms_from_sheet(df, sheet_name)
                if sheet_norms:
                    self.norms_data[sheet_name] = sheet_norms
                    print(f"      Найдено норм: {len(sheet_norms)}")
                    for norm_id, norm_data in sheet_norms.items():
                        print(f"        Норма №{norm_id}: {len(norm_data['points'])} точек")
            if not self.norms_data:
                print("❌ Не найдено ни одной нормы в файле")
                return False
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки норм: {e}")
            return False
    
    def parse_norms_from_sheet(self, df, sheet_name):
        norms = {}
        for row_idx in range(len(df)):
            first_cell = str(df.iloc[row_idx, 0]) if pd.notna(df.iloc[row_idx, 0]) else ""
            if "Норма №" in first_cell:
                try:
                    norm_number = int(first_cell.split("№")[1].strip())
                    load_row = df.iloc[row_idx + 2, 1:]
                    consumption_row = df.iloc[row_idx + 3, 1:]
                    points = []
                    for i in range(len(load_row)):
                        if pd.notna(load_row.iloc[i]) and pd.notna(consumption_row.iloc[i]):
                            try:
                                load_val = float(load_row.iloc[i])
                                consumption_val = float(consumption_row.iloc[i])
                                points.append((load_val, consumption_val))
                            except:
                                continue
                    if len(points) >= 2:
                        points.sort(key=lambda x: x[0])
                        norms[norm_number] = {
                            'points': points,
                            'description': str(df.iloc[row_idx + 1, 0]) if pd.notna(df.iloc[row_idx + 1, 0]) else ""
                        }
                except:
                    continue
        return norms
    
    def analyze_section(self, section_name, routes_df, norms):
        norm_functions = {}
        for norm_id, norm_data in norms.items():
            points = norm_data['points']
            x_values = [p[0] for p in points]
            y_values = [p[1] for p in points]
            if len(points) == 2:
                interp_func = interp1d(x_values, y_values, kind='linear', fill_value='extrapolate', bounds_error=False)
            else:
                try:
                    interp_func = CubicSpline(x_values, y_values, bc_type='natural')
                except:
                    interp_func = interp1d(x_values, y_values, kind='quadratic' if len(points) > 2 else 'linear', fill_value='extrapolate', bounds_error=False)
            norm_functions[norm_id] = {
                'function': interp_func,
                'points': points,
                'x_range': (min(x_values), max(x_values))
            }
        routes_df['Норма интерполированная'] = 0.0
        routes_df['Отклонение, %'] = 0.0
        routes_df['Статус'] = 'Не определен'
        for idx, row in routes_df.iterrows():
            norm_id = row['Номер нормы']
            if norm_id in norm_functions:
                try:
                    x_value = row['Нажатие на ось']
                    norm_func = norm_functions[norm_id]['function']
                    norm_value = float(norm_func(x_value))
                    routes_df.loc[idx, 'Норма интерполированная'] = norm_value
                    fact_value = row['Фактический удельный']
                    if norm_value > 0:
                        deviation = ((fact_value - norm_value) / norm_value) * 100
                        routes_df.loc[idx, 'Отклонение, %'] = deviation
                        if deviation < -5:
                            routes_df.loc[idx, 'Статус'] = 'Экономия'
                        elif deviation > 5:
                            routes_df.loc[idx, 'Статус'] = 'Перерасход'
                        else:
                            routes_df.loc[idx, 'Статус'] = 'Норма'
                except:
                    continue
        self.analysis_results[section_name] = {
            'routes': routes_df,
            'norms': norms,
            'norm_functions': norm_functions
        }
        return routes_df, norm_functions
    
    def analyze_section_with_filters(self, section_name, routes_df, norms, locomotive_filter=None, coefficients_manager=None, use_coefficients=False):
        if locomotive_filter:
            routes_df = locomotive_filter.filter_routes(routes_df)
            if routes_df.empty:
                return routes_df, None
        routes_df = routes_df.copy()
        if use_coefficients and coefficients_manager:
            routes_df['Коэффициент'] = 1.0
            routes_df['Факт. удельный исходный'] = routes_df['Фактический удельный']
            for idx, row in routes_df.iterrows():
                if 'Серия локомотива' in row and 'Номер локомотива' in row:
                    series = row['Серия локомотива']
                    number = row['Номер локомотива']
                    if pd.notna(series) and pd.notna(number):
                        try:
                            if isinstance(number, str):
                                number = int(number.lstrip('0')) if number.strip().lstrip('0') else 0
                            else:
                                number = int(number)
                            coef = coefficients_manager.get_coefficient(str(series), number)
                            routes_df.at[idx, 'Коэффициент'] = coef
                            routes_df.at[idx, 'Фактический удельный'] = routes_df.at[idx, 'Фактический удельный'] / coef
                        except:
                            continue
        return self.analyze_section(section_name, routes_df, norms)
    
    def create_interactive_plot(self, section_name, routes_df, norm_functions):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=('Удельный расход', 'Отклонения от нормы'))
        for norm_id, norm_info in norm_functions.items():
            points = norm_info['points']
            x_points = [p[0] for p in points]
            y_points = [p[1] for p in points]
            interp_func = norm_info['function']
            x_range = np.linspace(min(x_points), max(x_points), 100)
            y_interp = interp_func(x_range)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_interp,
                    mode='lines',
                    name=f'Норма №{norm_id}',
                    line=dict(color='blue', width=2),
                    legendgroup=f'norm{norm_id}'
                ),
                row=1, col=1
            )
            point_colors = ['black'] * len(points)
            hover_text = [f"Точка нормы №{norm_id}<br>Нажатие: {x:.2f} т/ось<br>Расход: {y:.2f} кВт·ч" for x, y in points]
            fig.add_trace(
                go.Scatter(
                    x=x_points,
                    y=y_points,
                    mode='markers',
                    name=f'Точки нормы №{norm_id}',
                    marker=dict(color=point_colors, size=8, opacity=0.8, line=dict(color='black', width=0.5)),
                    legendgroup=f'norm{norm_id}',
                    showlegend=False,
                    hovertemplate='%{text}',
                    text=hover_text
                ),
                row=1, col=1
            )
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен']
        if len(valid_routes) > 0:
            deviation_groups = {
                'Экономия +30% и более': valid_routes[valid_routes['Отклонение, %'] >= 30],
                'Экономия +20% до +30%': valid_routes[(valid_routes['Отклонение, %'] >= 20) & (valid_routes['Отклонение, %'] < 30)],
                'Экономия +5% до +20%': valid_routes[(valid_routes['Отклонение, %'] >= 5) & (valid_routes['Отклонение, %'] < 20)],
                'Норма -5% до +5%': valid_routes[(valid_routes['Отклонение, %'] >= -5) & (valid_routes['Отклонение, %'] < 5)],
                'Перерасход -5% до -20%': valid_routes[(valid_routes['Отклонение, %'] >= -20) & (valid_routes['Отклонение, %'] < -5)],
                'Перерасход -20% до -30%': valid_routes[(valid_routes['Отклонение, %'] >= -30) & (valid_routes['Отклонение, %'] < -20)],
                'Перерасход -30% и менее': valid_routes[valid_routes['Отклонение, %'] < -30]
            }
            group_colors = {
                'Экономия +30% и более': '#7C3AED',
                'Экономия +20% до +30%': '#9333EA',
                'Экономия +5% до +20%': '#06B6D4',
                'Норма -5% до +5%': '#22C55E',
                'Перерасход -5% до -20%': '#EAB308',
                'Перерасход -20% до -30%': '#F97316',
                'Перерасход -30% и менее': '#DC2626'
            }
            for group_name, group_data in deviation_groups.items():
                if len(group_data) > 0:
                    hover_text = []
                    for _, row in group_data.iterrows():
                        text = f"Маршрут №{row['Номер маршрута']}<br>Дата: {row['Дата маршрута']}<br>Норма №{row['Номер нормы']}<br>"
                        if 'Серия локомотива' in row and pd.notna(row['Серия локомотива']):
                            text += f"Локомотив: {row['Серия локомотива']} "
                        if 'Номер локомотива' in row and pd.notna(row['Номер локомотива']):
                            text += f"№{row['Номер локомотива']}<br>"
                        else:
                            text += "<br>"
                        text += f"Нажатие: {row['Нажатие на ось']:.2f} т/ось<br>Отклонение: {row['Отклонение, %']:.1f}%"
                        hover_text.append(text)
                    fig.add_trace(
                        go.Scatter(
                            x=group_data['Нажатие на ось'],
                            y=group_data['Отклонение, %'],
                            mode='markers',
                            name=f'{group_name} ({len(group_data)})',
                            marker=dict(color=group_colors[group_name], size=10, opacity=0.8, line=dict(color='black', width=0.5)),
                            hovertemplate='%{text}',
                            text=hover_text
                        ),
                        row=2, col=1
                    )
            x_range = [valid_routes['Нажатие на ось'].min() - 1, valid_routes['Нажатие на ось'].max() + 1]
            fig.add_trace(go.Scatter(x=x_range, y=[5, 5], mode='lines', line=dict(color='#22C55E', dash='dash', width=2), showlegend=False, hoverinfo='skip'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_range, y=[-5, -5], mode='lines', line=dict(color='#22C55E', dash='dash', width=2), showlegend=False, hoverinfo='skip'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_range, y=[20, 20], mode='lines', line=dict(color='#F97316', dash='dot', width=2), showlegend=False, hoverinfo='skip'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_range, y=[-20, -20], mode='lines', line=dict(color='#F97316', dash='dot', width=2), showlegend=False, hoverinfo='skip'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_range, y=[0, 0], mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_range + x_range[::-1], y=[-5, -5, 5, 5], fill='toself', fillcolor='rgba(34, 197, 94, 0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=1, col=1)
        fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм брутто", row=1, col=1)
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=2, col=1)
        fig.update_yaxes(title_text="Отклонение от нормы, %", row=2, col=1)
        fig.update_layout(height=1000, showlegend=True, hovermode='closest', template='plotly_white', legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))
        return fig
    
    def get_sections_list(self):
        if self.routes_df is None:
            return []
        return self.routes_df['Наименование участка'].unique().tolist()
    
    def analyze_single_section(self, section_name):
        if section_name not in self.norms_data:
            return None, None, "Нормы для участка не найдены"
        section_routes = self.routes_df[self.routes_df['Наименование участка'] == section_name].copy()
        if section_routes.empty:
            return None, None, "Нет маршрутов для участка"
        routes_analyzed, norm_functions = self.analyze_section(section_name, section_routes, self.norms_data[section_name])
        fig = self.create_interactive_plot(section_name, routes_analyzed, norm_functions)
        valid_routes = routes_analyzed[routes_analyzed['Статус'] != 'Не определен']
        detailed_stats = {
            'economy_strong': len(valid_routes[valid_routes['Отклонение, %'] >= 30]),
            'economy_medium': len(valid_routes[(valid_routes['Отклонение, %'] >= 20) & (valid_routes['Отклонение, %'] < 30)]),
            'economy_weak': len(valid_routes[(valid_routes['Отклонение, %'] >= 5) & (valid_routes['Отклонение, %'] < 20)]),
            'normal': len(valid_routes[(valid_routes['Отклонение, %'] >= -5) & (valid_routes['Отклонение, %'] < 5)]),
            'overrun_weak': len(valid_routes[(valid_routes['Отклонение, %'] >= -20) & (valid_routes['Отклонение, %'] < -5)]),
            'overrun_medium': len(valid_routes[(valid_routes['Отклонение, %'] >= -30) & (valid_routes['Отклонение, %'] < -20)]),
            'overrun_strong': len(valid_routes[valid_routes['Отклонение, %'] < -30])
        }
        stats = {
            'total': len(routes_analyzed),
            'processed': len(valid_routes),
            'economy': detailed_stats['economy_strong'] + detailed_stats['economy_medium'] + detailed_stats['economy_weak'],
            'normal': detailed_stats['normal'],
            'overrun': detailed_stats['overrun_weak'] + detailed_stats['overrun_medium'] + detailed_stats['overrun_strong'],
            'mean_deviation': valid_routes['Отклонение, %'].mean() if len(valid_routes) > 0 else 0,
            'detailed_stats': detailed_stats
        }
        return fig, stats, None