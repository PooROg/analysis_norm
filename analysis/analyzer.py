# analysis/analyzer.py (обновленный с подсчетом маршрутов по нормам)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, CubicSpline
import warnings
warnings.filterwarnings('ignore')
import logging
from typing import List, Dict, Optional, Tuple

from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager
from core.norm_storage import NormStorage
from analysis.html_route_processor import HTMLRouteProcessor
from analysis.html_norm_processor import HTMLNormProcessor

# Настройка логирования
logger = logging.getLogger(__name__)

class InteractiveNormsAnalyzer:
    """Класс для интерактивного анализа норм расхода электроэнергии (обновленный с route_processor.py)"""
    
    def __init__(self):
        self.route_processor = HTMLRouteProcessor()
        self.norm_processor = HTMLNormProcessor()
        self.norm_storage = NormStorage()
        self.routes_df = None
        self.analyzed_results = {}
        self.sections_norms_map = {}  # Карта участков и их норм
        
        logger.info("Инициализирован обновленный анализатор норм с интеграцией route_processor.py")
    
    def load_routes_from_html(self, html_files: List[str]) -> bool:
        """Загружает маршруты из HTML файлов с полной обработкой как в route_processor.py"""
        logger.info(f"Загрузка маршрутов из {len(html_files)} HTML файлов")
        
        try:
            # Обрабатываем HTML файлы с полной интеграцией route_processor.py
            self.routes_df = self.route_processor.process_html_files(html_files)
            
            if self.routes_df.empty:
                logger.error("Не удалось загрузить маршруты из HTML файлов")
                return False
            
            logger.info(f"Загружено {len(self.routes_df)} записей маршрутов")
            self._log_routes_statistics()
            self._build_sections_norms_map()
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки маршрутов: {e}")
            return False
    
    def _build_sections_norms_map(self):
        """Строит карту участков и их норм"""
        if self.routes_df is None or self.routes_df.empty:
            return
        
        # Группируем по участкам и нормам
        sections_groups = self.routes_df.groupby('Наименование участка')['Номер нормы'].apply(lambda x: list(x.dropna().unique())).to_dict()
        
        self.sections_norms_map = {}
        for section, norms in sections_groups.items():
            self.sections_norms_map[section] = [str(norm) for norm in norms if str(norm) != 'nan']
        
        logger.info(f"Построена карта участков и норм: {len(self.sections_norms_map)} участков")
        
        # Логируем первые несколько для отладки
        for i, (section, norms) in enumerate(list(self.sections_norms_map.items())[:5]):
            logger.debug(f"  {section}: нормы {norms}")
    
    def load_norms_from_html(self, html_files: List[str]) -> bool:
        """Загружает нормы из HTML файлов"""
        logger.info(f"Загрузка норм из {len(html_files)} HTML файлов")
        
        try:
            # Обрабатываем HTML файлы норм
            new_norms = self.norm_processor.process_html_files(html_files)
            
            if not new_norms:
                logger.warning("Не найдено норм в HTML файлах")
                return False
            
            # Добавляем/обновляем нормы в хранилище
            update_results = self.norm_storage.add_or_update_norms(new_norms)
            
            # Логируем результаты
            stats = self.norm_processor.get_processing_stats()
            logger.info(f"Обработано норм: всего {stats['total_norms_found']}, "
                       f"новых {stats['new_norms']}, обновленных {stats['updated_norms']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки норм: {e}")
            return False
    
    def get_sections_list(self) -> List[str]:
        """Возвращает список доступных участков"""
        if self.routes_df is None or self.routes_df.empty:
            return []
        
        sections = self.routes_df['Наименование участка'].dropna().unique().tolist()
        logger.debug(f"Найдено участков: {len(sections)}")
        return sorted(sections)
    
    def get_norms_for_section(self, section_name: str) -> List[str]:
        """Возвращает список норм для участка"""
        return self.sections_norms_map.get(section_name, [])
    
    def get_norms_with_counts_for_section(self, section_name: str, single_section_only: bool = False) -> List[Tuple[str, int]]:
        """Возвращает список норм для участка с количеством маршрутов"""
        if self.routes_df is None or self.routes_df.empty:
            return []
        
        # Фильтруем данные по участку
        section_routes = self.routes_df[self.routes_df['Наименование участка'] == section_name].copy()
        
        if section_routes.empty:
            return []
        
        # Если нужны только маршруты с одним участком
        if single_section_only:
            # Подсчитываем количество участков для каждого маршрута
            route_section_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
            single_section_routes = route_section_counts[route_section_counts == 1].index
            
            # Фильтруем только маршруты с одним участком
            section_routes = section_routes.set_index(['Номер маршрута', 'Дата маршрута'])
            section_routes = section_routes.loc[section_routes.index.intersection(single_section_routes)]
            section_routes = section_routes.reset_index()
        
        if section_routes.empty:
            return []
        
        # Подсчитываем количество маршрутов для каждой нормы
        norm_counts = section_routes['Номер нормы'].value_counts()
        
        # Формируем список норм с количествами
        norms_with_counts = []
        for norm in self.sections_norms_map.get(section_name, []):
            count = norm_counts.get(int(norm) if norm.isdigit() else norm, 0)
            norms_with_counts.append((norm, count))
        
        # Сортируем по номеру нормы
        norms_with_counts.sort(key=lambda x: int(x[0]) if x[0].isdigit() else float('inf'))
        
        logger.debug(f"Нормы для участка '{section_name}' (только один участок: {single_section_only}): {norms_with_counts}")
        return norms_with_counts
    
    def get_routes_count_for_section(self, section_name: str, single_section_only: bool = False) -> int:
        """Возвращает общее количество маршрутов для участка"""
        if self.routes_df is None or self.routes_df.empty:
            return 0
        
        # Фильтруем данные по участку
        section_routes = self.routes_df[self.routes_df['Наименование участка'] == section_name].copy()
        
        if section_routes.empty:
            return 0
        
        # Если нужны только маршруты с одним участком
        if single_section_only:
            # Подсчитываем количество участков для каждого маршрута
            route_section_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
            single_section_routes = route_section_counts[route_section_counts == 1].index
            
            # Фильтруем только маршруты с одним участком
            section_routes = section_routes.set_index(['Номер маршрута', 'Дата маршрута'])
            section_routes = section_routes.loc[section_routes.index.intersection(single_section_routes)]
            
            return len(section_routes)
        
        return len(section_routes)
    
    def get_norm_routes_count_for_section(self, section_name: str, norm_id: str, single_section_only: bool = False) -> int:
        """Возвращает количество маршрутов для конкретной нормы в участке"""
        if self.routes_df is None or self.routes_df.empty:
            return 0
        
        # Фильтруем данные по участку и норме
        section_norm_routes = self.routes_df[
            (self.routes_df['Наименование участка'] == section_name) &
            (self.routes_df['Номер нормы'].astype(str) == str(norm_id))
        ].copy()
        
        if section_norm_routes.empty:
            return 0
        
        # Если нужны только маршруты с одним участком
        if single_section_only:
            # Подсчитываем количество участков для каждого маршрута
            route_section_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
            single_section_routes = route_section_counts[route_section_counts == 1].index
            
            # Фильтруем только маршруты с одним участком
            section_norm_routes = section_norm_routes.set_index(['Номер маршрута', 'Дата маршрута'])
            section_norm_routes = section_norm_routes.loc[section_norm_routes.index.intersection(single_section_routes)]
            
            return len(section_norm_routes)
        
        return len(section_norm_routes)
    
    def get_norm_info(self, norm_id: str) -> Optional[Dict]:
        """Возвращает информацию о норме"""
        norm_data = self.norm_storage.get_norm(norm_id)
        if not norm_data:
            return None
        
        # Формируем удобную информацию о норме
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
    
    def analyze_section(self, section_name: str, norm_id: Optional[str] = None,
                       single_section_only: bool = False,
                       locomotive_filter: Optional[LocomotiveFilter] = None,
                       coefficients_manager: Optional[LocomotiveCoefficientsManager] = None,
                       use_coefficients: bool = False) -> Tuple[Optional[go.Figure], Optional[Dict], Optional[str]]:
        """Анализирует участок с возможностью выбора конкретной нормы и фильтрации по одному участку"""
        logger.info(f"Анализ участка: {section_name}, норма: {norm_id}, только один участок: {single_section_only}")
        
        if self.routes_df is None or self.routes_df.empty:
            return None, None, "Данные маршрутов не загружены"
        
        try:
            # Фильтруем данные по участку
            section_routes = self.routes_df[
                self.routes_df['Наименование участка'] == section_name
            ].copy()
            
            if section_routes.empty:
                return None, None, f"Нет данных для участка {section_name}"
            
            # Фильтрация по маршрутам с одним участком
            if single_section_only:
                route_section_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
                single_section_routes = route_section_counts[route_section_counts == 1].index
                
                section_routes = section_routes.set_index(['Номер маршрута', 'Дата маршрута'])
                section_routes = section_routes.loc[section_routes.index.intersection(single_section_routes)]
                section_routes = section_routes.reset_index()
                
                if section_routes.empty:
                    return None, None, f"Нет маршрутов с одним участком для {section_name}"
                
                logger.debug(f"После фильтрации по одному участку осталось {len(section_routes)} маршрутов")
            
            # Если указана конкретная норма, фильтруем по ней
            if norm_id:
                section_routes = section_routes[
                    section_routes['Номер нормы'].astype(str) == str(norm_id)
                ]
                if section_routes.empty:
                    filter_text = " с одним участком" if single_section_only else ""
                    return None, None, f"Нет маршрутов{filter_text} для участка {section_name} с нормой {norm_id}"
            
            logger.debug(f"Найдено {len(section_routes)} маршрутов для участка {section_name}")
            
            # Применяем фильтр локомотивов
            if locomotive_filter:
                section_routes = locomotive_filter.filter_routes(section_routes)
                if section_routes.empty:
                    return None, None, "Нет данных после применения фильтра локомотивов"
                logger.debug(f"После фильтрации локомотивов осталось {len(section_routes)} маршрутов")
            
            # Применяем коэффициенты
            if use_coefficients and coefficients_manager:
                section_routes = self._apply_coefficients(section_routes, coefficients_manager)
                logger.debug("Применены коэффициенты к фактическому расходу")
            
            # Анализируем данные участка
            analyzed_data, norm_functions = self._analyze_section_data(section_name, section_routes, norm_id)
            
            if analyzed_data.empty:
                return None, None, f"Не удалось проанализировать участок {section_name}"
            
            # Создаем интерактивный график
            fig = self._create_interactive_plot(section_name, analyzed_data, norm_functions, norm_id, single_section_only)
            
            # Вычисляем статистику
            statistics = self._calculate_section_statistics(analyzed_data)
            
            # Сохраняем результаты
            analysis_key = f"{section_name}_{norm_id}_{single_section_only}" if norm_id else f"{section_name}_{single_section_only}"
            self.analyzed_results[analysis_key] = {
                'routes': analyzed_data,
                'norms': norm_functions,
                'statistics': statistics
            }
            
            logger.info(f"Анализ участка {section_name} завершен успешно")
            return fig, statistics, None
            
        except Exception as e:
            logger.error(f"Ошибка анализа участка {section_name}: {e}")
            return None, None, f"Ошибка анализа: {str(e)}"
    
    def _apply_coefficients(self, routes_df: pd.DataFrame, 
                          coefficients_manager: LocomotiveCoefficientsManager) -> pd.DataFrame:
        """Применяет коэффициенты к фактическому расходу"""
        routes_df = routes_df.copy()
        routes_df['Коэффициент'] = 1.0
        routes_df['Факт. удельный исходный'] = routes_df['Расход фактический']
        applied_count = 0
        
        for i, row in routes_df.iterrows():
            series = str(row.get('Серия локомотива', '')) if pd.notna(row.get('Серия локомотива')) else ''
            number = row.get('Номер локомотива')
            
            if series and pd.notna(number):
                try:
                    if isinstance(number, str):
                        number = int(number.lstrip('0')) if number.strip().lstrip('0') else 0
                    else:
                        number = int(number)
                    
                    coefficient = coefficients_manager.get_coefficient(series, number)
                    routes_df.at[i, 'Коэффициент'] = coefficient
                    
                    if coefficient != 1.0:
                        routes_df.at[i, 'Расход фактический'] = routes_df.at[i, 'Расход фактический'] / coefficient
                        applied_count += 1
                        
                        if applied_count <= 3:  # Логируем первые 3 для отладки
                            logger.debug(f"Применен коэффициент {coefficient:.3f} к локомотиву {series} №{number}")
                
                except (ValueError, TypeError) as e:
                    logger.debug(f"Ошибка обработки локомотива {series} №{number}: {e}")
                    continue
        
        logger.info(f"Применено коэффициентов: {applied_count}")
        return routes_df
    
    def _analyze_section_data(self, section_name: str, routes_df: pd.DataFrame, 
                             specific_norm_id: Optional[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """Анализирует данные участка с использованием норм из хранилища"""
        logger.debug(f"Анализ данных участка {section_name}")
        
        # Добавляем колонки для анализа
        routes_df = routes_df.copy()
        routes_df['Норма интерполированная'] = 0.0
        routes_df['Отклонение, %'] = 0.0
        routes_df['Статус'] = 'Не определен'
        
        # Получаем уникальные номера норм в данных
        if specific_norm_id:
            norm_numbers = [specific_norm_id]
        else:
            norm_numbers = routes_df['Номер нормы'].dropna().unique()
        
        logger.debug(f"Анализируемые номера норм: {norm_numbers}")
        
        # Создаем функции интерполяции для найденных норм
        norm_functions = {}
        for norm_number in norm_numbers:
            norm_number_str = str(int(norm_number)) if pd.notna(norm_number) else None
            if norm_number_str:
                norm_data = self.norm_storage.get_norm(norm_number_str)
                if norm_data and norm_data.get('points'):
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
                            logger.debug(f"Создана функция для нормы {norm_number_str}")
                    except Exception as e:
                        logger.error(f"Ошибка создания функции для нормы {norm_number_str}: {e}")
        
        if not norm_functions:
            logger.warning(f"Не найдено функций норм для участка {section_name}")
            return routes_df, {}
        
        # Интерполируем значения норм и вычисляем отклонения
        for i, row in routes_df.iterrows():
            norm_number = row.get('Номер нормы')
            if pd.notna(norm_number):
                norm_number_str = str(int(norm_number))
                
                if norm_number_str in norm_functions:
                    try:
                        # Вычисляем нажатие на ось из данных route_processor.py
                        axle_load = self._calculate_axle_load_from_data(row)
                        
                        if axle_load and axle_load > 0:
                            norm_func = norm_functions[norm_number_str]['function']
                            norm_value = float(norm_func(axle_load))
                            routes_df.loc[i, 'Норма интерполированная'] = norm_value
                            
                            # Используем "Факт уд" если доступен, иначе обычный расход
                            actual_value = row.get('Факт уд')
                            if pd.isna(actual_value) or actual_value is None:
                                actual_value = row.get('Расход фактический')
                            
                            if actual_value and norm_value > 0:
                                deviation = ((actual_value - norm_value) / norm_value) * 100
                                routes_df.loc[i, 'Отклонение, %'] = deviation
                                
                                # Определяем статус
                                if deviation < -30:
                                    routes_df.loc[i, 'Статус'] = 'Экономия сильная'
                                elif deviation < -20:
                                    routes_df.loc[i, 'Статус'] = 'Экономия средняя'
                                elif deviation < -5:
                                    routes_df.loc[i, 'Статус'] = 'Экономия слабая'
                                elif deviation <= 5:
                                    routes_df.loc[i, 'Статус'] = 'Норма'
                                elif deviation <= 20:
                                    routes_df.loc[i, 'Статус'] = 'Перерасход слабый'
                                elif deviation <= 30:
                                    routes_df.loc[i, 'Статус'] = 'Перерасход средний'
                                else:
                                    routes_df.loc[i, 'Статус'] = 'Перерасход сильный'
                        
                    except Exception as e:
                        logger.debug(f"Ошибка интерполяции для строки {i}: {e}")
                        continue
        
        logger.info(f"Проанализировано {len(routes_df)} записей для участка {section_name}")
        return routes_df, norm_functions
    
    def _calculate_axle_load_from_data(self, row: pd.Series) -> Optional[float]:
        """Вычисляет нажатие на ось из данных обработанных route_processor.py"""
        try:
            # Сначала пытаемся получить готовое значение из route_processor.py
            if 'Нажатие на ось' in row and pd.notna(row['Нажатие на ось']):
                axle_load = row['Нажатие на ось']
                if axle_load != "-" and isinstance(axle_load, (int, float)):
                    return float(axle_load)
            
            # Альтернативный расчет из БРУТТО/ОСИ
            brutto = row.get('БРУТТО')
            osi = row.get('ОСИ')
            
            if (pd.notna(brutto) and pd.notna(osi) and 
                brutto != "-" and osi != "-" and 
                isinstance(brutto, (int, float)) and isinstance(osi, (int, float)) and
                osi != 0):
                return float(brutto) / float(osi)
            
            # Если все остальное не работает, используем данные маршрута
            tkm_brutto = row.get('Ткм брутто')
            km = row.get('Км')
            
            if pd.notna(tkm_brutto) and pd.notna(km) and km != 0:
                # Это приблизительный расчет
                return float(tkm_brutto) / float(km) / 1000 * 20
            
            return None
            
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Ошибка расчета нажатия на ось: {e}")
            return None
    
    def _create_interactive_plot(self, section_name: str, routes_df: pd.DataFrame, 
                               norm_functions: Dict, specific_norm_id: Optional[str] = None,
                               single_section_only: bool = False) -> go.Figure:
        """Создает интерактивный график для участка"""
        title_suffix = f" (норма {specific_norm_id})" if specific_norm_id else ""
        filter_suffix = " [только один участок]" if single_section_only else ""
        logger.debug(f"Создание графика для участка {section_name}{title_suffix}{filter_suffix}")
        
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
        
        # Отрисовка кривых норм на верхнем графике
        for norm_id, norm_data in norm_functions.items():
            # Пропускаем норму если указана конкретная и это не она
            if specific_norm_id and norm_id != specific_norm_id:
                continue
                
            points = norm_data['points']
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            
            # Создаем интерполированную кривую
            x_interp = np.linspace(min(x_vals), max(x_vals), 100)
            norm_func = norm_data['function']
            y_interp = norm_func(x_interp)
            
            fig.add_trace(
                go.Scatter(
                    x=x_interp,
                    y=y_interp,
                    mode='lines',
                    name=f'Норма №{norm_id}',
                    line=dict(width=2),
                    hovertemplate='Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм'
                ), row=1, col=1
            )
            
            # Добавляем опорные точки
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    marker=dict(symbol='square', size=8, color='black'),
                    name=f'Опорные точки нормы №{norm_id}',
                    hovertemplate='Опорная точка<br>Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм'
                ), row=1, col=1
            )
        
        # Добавляем фактические точки маршрутов
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен'].copy()
        
        # Определяем цвета для статусов
        status_colors = {
            'Экономия сильная': '#006400',      # Темно-зеленый
            'Экономия средняя': '#228B22',      # Зеленый
            'Экономия слабая': '#32CD32',       # Светло-зеленый
            'Норма': '#FFD700',                 # Золотой
            'Перерасход слабый': '#FF8C00',     # Оранжевый
            'Перерасход средний': '#FF4500',    # Красно-оранжевый
            'Перерасход сильный': '#DC143C'     # Малиновый
        }
        
        for status in status_colors.keys():
            status_routes = valid_routes[valid_routes['Статус'] == status]
            if not status_routes.empty:
                x_values = []
                y_values = []
                hover_texts = []
                
                for _, route in status_routes.iterrows():
                    axle_load = self._calculate_axle_load_from_data(route)
                    
                    # Определяем значение расхода для отображения
                    consumption_value = route.get('Факт уд')
                    if pd.isna(consumption_value) or consumption_value is None:
                        consumption_value = route.get('Расход фактический')
                    
                    if axle_load and consumption_value:
                        x_values.append(axle_load)
                        y_values.append(consumption_value)
                        hover_texts.append(self._create_hover_text(route))
                
                if x_values:
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='markers',
                            name=f'{status} ({len(status_routes)})',
                            marker=dict(
                                color=status_colors[status],
                                size=8,
                                opacity=0.8,
                                line=dict(color='black', width=0.5)
                            ),
                            hovertemplate='%{text}',
                            text=hover_texts
                        ), row=1, col=1
                    )
        
        # Добавляем точки отклонений на нижний график
        self._add_deviation_points(fig, valid_routes)
        
        # Добавляем линии границ на нижний график
        self._add_boundary_lines(fig, valid_routes)
        
        # Настройка осей и layout
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
        
        return fig
    
    def _create_hover_text(self, route: pd.Series) -> str:
        """Создает текст для hover эффекта"""
        hover_text = (
            f"Маршрут №{route.get('Номер маршрута', 'N/A')}<br>"
            f"Дата: {route.get('Дата маршрута', 'N/A')}<br>"
            f"Локомотив: {route.get('Серия локомотива', '')} №{route.get('Номер локомотива', '')}<br>"
        )
        
        # Добавляем информацию о коэффициенте, если есть
        if 'Коэффициент' in route.index and pd.notna(route['Коэффициент']) and route['Коэффициент'] != 1.0:
            hover_text += f"Коэффициент: {route['Коэффициент']:.3f}<br>"
            if 'Факт. удельный исходный' in route.index and pd.notna(route['Факт. удельный исходный']):
                hover_text += f"Факт исходный: {route['Факт. удельный исходный']:.1f}<br>"
        
        axle_load = self._calculate_axle_load_from_data(route)
        
        # Определяем значение расхода
        consumption_value = route.get('Факт уд')
        if pd.isna(consumption_value) or consumption_value is None:
            consumption_value = route.get('Расход фактический')
        
        hover_text += (
            f"Нажатие: {axle_load:.2f} т/ось<br>" if axle_load else "Нажатие: N/A<br>"
            f"Факт: {consumption_value:.1f}<br>" if consumption_value else "Факт: N/A<br>"
            f"Норма: {route.get('Норма интерполированная', 'N/A'):.1f}<br>"
            f"Отклонение: {route.get('Отклонение, %', 'N/A'):.1f}%"
        )
        
        return hover_text
    
    def _add_deviation_points(self, fig: go.Figure, routes_df: pd.DataFrame):
        """Добавляет точки отклонений на нижний график"""
        # Группировка для нижнего графика по статусам
        status_colors = {
            'Экономия сильная': '#006400',
            'Экономия средняя': '#228B22',
            'Экономия слабая': '#32CD32',
            'Норма': '#FFD700',
            'Перерасход слабый': '#FF8C00',
            'Перерасход средний': '#FF4500',
            'Перерасход сильный': '#DC143C'
        }
        
        for status, color in status_colors.items():
            status_data = routes_df[routes_df['Статус'] == status]
            if len(status_data) > 0:
                hover_texts = []
                x_values = []
                y_values = []
                
                for _, route in status_data.iterrows():
                    axle_load = self._calculate_axle_load_from_data(route)
                    if axle_load:
                        x_values.append(axle_load)
                        y_values.append(route['Отклонение, %'])
                        hover_texts.append(self._create_hover_text(route))
                
                if x_values:
                    fig.add_trace(
                        go.Scatter(
                            x=x_values,
                            y=y_values,
                            mode='markers',
                            name=f'{status} ({len(status_data)})',
                            marker=dict(
                                color=color,
                                size=10,
                                opacity=0.8,
                                line=dict(color='black', width=0.5)
                            ),
                            hovertemplate='%{text}',
                            text=hover_texts
                        ), row=2, col=1
                    )
    
    def _add_boundary_lines(self, fig: go.Figure, routes_df: pd.DataFrame):
        """Добавляет граничные линии на нижний график"""
        if routes_df.empty:
            return
        
        # Вычисляем диапазон для линий
        axle_loads = []
        for _, route in routes_df.iterrows():
            axle_load = self._calculate_axle_load_from_data(route)
            if axle_load:
                axle_loads.append(axle_load)
        
        if not axle_loads:
            return
        
        x_range = [min(axle_loads) - 1, max(axle_loads) + 1]
        
        # Добавляем линии границ
        fig.add_trace(go.Scatter(x=x_range, y=[5, 5], mode='lines', 
                               line=dict(color='#FFD700', dash='dash', width=2), 
                               showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_range, y=[-5, -5], mode='lines', 
                               line=dict(color='#FFD700', dash='dash', width=2), 
                               showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_range, y=[20, 20], mode='lines', 
                               line=dict(color='#FF4500', dash='dot', width=2), 
                               showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_range, y=[-20, -20], mode='lines', 
                               line=dict(color='#FF4500', dash='dot', width=2), 
                               showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_range, y=[30, 30], mode='lines', 
                               line=dict(color='#DC143C', dash='dashdot', width=2), 
                               showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_range, y=[-30, -30], mode='lines', 
                               line=dict(color='#DC143C', dash='dashdot', width=2), 
                               showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=x_range, y=[0, 0], mode='lines', 
                               line=dict(color='black', width=1), 
                               showlegend=False, hoverinfo='skip'), row=2, col=1)
        
        # Добавляем зеленую зону нормы
        fig.add_trace(go.Scatter(x=x_range + x_range[::-1], y=[-5, -5, 5, 5], 
                               fill='toself', fillcolor='rgba(255, 215, 0, 0.1)', 
                               line=dict(color='rgba(255,255,255,0)'), 
                               showlegend=False, hoverinfo='skip'), row=2, col=1)
    
    def _calculate_section_statistics(self, routes_df: pd.DataFrame) -> Dict:
        """Вычисляет статистику для участка"""
        total = len(routes_df)
        valid_routes = routes_df[routes_df['Статус'] != 'Не определен']
        processed = len(valid_routes)
        
        if processed == 0:
            return {
                'total': total,
                'processed': processed,
                'economy': 0,
                'normal': 0,
                'overrun': 0,
                'mean_deviation': 0,
                'median_deviation': 0,
                'detailed_stats': {}
            }
        
        # Детальная статистика по статусам
        detailed_stats = {
            'economy_strong': len(valid_routes[valid_routes['Статус'] == 'Экономия сильная']),
            'economy_medium': len(valid_routes[valid_routes['Статус'] == 'Экономия средняя']),
            'economy_weak': len(valid_routes[valid_routes['Статус'] == 'Экономия слабая']),
            'normal': len(valid_routes[valid_routes['Статус'] == 'Норма']),
            'overrun_weak': len(valid_routes[valid_routes['Статус'] == 'Перерасход слабый']),
            'overrun_medium': len(valid_routes[valid_routes['Статус'] == 'Перерасход средний']),
            'overrun_strong': len(valid_routes[valid_routes['Статус'] == 'Перерасход сильный'])
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
    
    def _log_routes_statistics(self):
        """Логирует статистику загруженных маршрутов"""
        if self.routes_df is None or self.routes_df.empty:
            return
        
        stats = self.route_processor.get_processing_stats()
        
        logger.info("=== СТАТИСТИКА ЗАГРУЖЕННЫХ МАРШРУТОВ ===")
        logger.info(f"Всего файлов обработано: {stats['total_files']}")
        logger.info(f"Всего маршрутов найдено: {stats['total_routes_found']}")
        logger.info(f"Уникальных маршрутов: {stats['unique_routes']}")
        logger.info(f"Дубликатов: {stats['duplicates_total']}")
        logger.info(f"Маршрутов с равными расходами: {stats['routes_with_equal_rashod']}")
        logger.info(f"Обработано успешно: {stats['routes_processed']}")
        logger.info(f"Пропущено: {stats['routes_skipped']}")
        logger.info(f"Итоговых записей в DataFrame: {len(self.routes_df)}")
        
        # Статистика по участкам
        sections_count = self.routes_df['Наименование участка'].nunique()
        logger.info(f"Уникальных участков: {sections_count}")
        
        # Статистика по нормам
        norms_count = self.routes_df['Номер нормы'].nunique()
        logger.info(f"Уникальных норм: {norms_count}")
    
    def get_norm_storage_info(self) -> Dict:
        """Возвращает информацию о хранилище норм"""
        return self.norm_storage.get_storage_info()
    
    def export_routes_to_excel(self, output_file: str) -> bool:
        """Экспортирует маршруты в Excel с полным форматированием route_processor.py"""
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
    
    def get_section_routes_count(self, section_name: str) -> int:
        """Возвращает количество маршрутов для участка"""
        if self.routes_df is None or self.routes_df.empty:
            return 0
        
        return len(self.routes_df[self.routes_df['Наименование участка'] == section_name])
    
    def get_norm_routes_count(self, norm_id: str) -> int:
        """Возвращает количество маршрутов для нормы"""
        if self.routes_df is None or self.routes_df.empty:
            return 0
        
        return len(self.routes_df[self.routes_df['Номер нормы'].astype(str) == str(norm_id)])