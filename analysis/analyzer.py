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
        """Получает количество маршрутов для нормы на участке"""
        try:
            if self.routes_df is None or self.routes_df.empty:
                return 0
            
            # Фильтруем данные по участку
            section_routes = self.routes_df[self.routes_df['Наименование участка'] == section_name].copy()
            
            if section_routes.empty:
                return 0
            
            # Если нужны только маршруты с одним участком
            if single_section_only:
                route_section_counts = self.routes_df.groupby(['Номер маршрута', 'Дата маршрута']).size()
                single_section_routes = route_section_counts[route_section_counts == 1].index
                
                section_routes = section_routes.set_index(['Номер маршрута', 'Дата маршрута'])
                section_routes = section_routes.loc[section_routes.index.intersection(single_section_routes)]
                section_routes = section_routes.reset_index()
            
            # Фильтруем по номеру нормы
            norm_routes = section_routes[section_routes['Номер нормы'] == int(norm_id)]
            return len(norm_routes)
            
        except Exception as e:
            logger.error(f"Ошибка получения количества маршрутов для нормы {norm_id}: {e}")
            return 0
    
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
        logger.debug(f"Анализ участка {section_name}, маршрутов: {len(routes_df)}")
        
        # Получаем уникальные номера норм
        if specific_norm_id:
            norm_numbers = [specific_norm_id]
        else:
            norm_numbers = routes_df['Номер нормы'].dropna().unique()
        
        logger.debug(f"Найдены номера норм: {list(norm_numbers)}")
        
        # Создаем функции интерполяции для найденных норм
        norm_functions = {}
        for norm_number in norm_numbers:
            norm_number_str = str(int(norm_number)) if pd.notna(norm_number) else None
            if norm_number_str:
                norm_data = self.norm_storage.get_norm(norm_number_str)
                if norm_data and norm_data.get('points'):
                    try:
                        # Получаем базовые точки нормы
                        base_points = norm_data['points'].copy()
                        
                        # Добавляем дополнительные точки из маршрутов
                        additional_points = self._extract_additional_norm_points(routes_df, norm_number_str, norm_data.get('norm_type', 'Нажатие'))
                        
                        # Объединяем все точки
                        all_points = base_points + additional_points
                        
                        # Удаляем дубликаты и сортируем
                        all_points = self._remove_duplicate_points(all_points)
                        all_points.sort(key=lambda x: x[0])
                        
                        # Создаем функцию интерполяции с объединенными точками
                        func = self.norm_storage._create_interpolation_function(all_points)
                        
                        if func:
                            norm_functions[norm_number_str] = {
                                'function': func,
                                'points': all_points,  # Используем объединенные точки
                                'base_points': base_points,  # Сохраняем оригинальные для отображения
                                'additional_points': additional_points,  # Сохраняем дополнительные для отображения
                                'x_range': (
                                    min(p[0] for p in norm_data['points']),
                                    max(p[0] for p in norm_data['points'])
                                ),
                                'data': norm_data,
                                'norm_type': norm_data.get('norm_type', 'Нажатие')
                            }
                            logger.debug(f"Создана функция для нормы {norm_number_str}, тип: {norm_data.get('norm_type', 'Нажатие')}")
                    except Exception as e:
                        logger.error(f"Ошибка создания функции для нормы {norm_number_str}: {e}")
                else:
                    logger.warning(f"Норма {norm_number_str} не найдена в хранилище или не содержит точек")
        
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
                        # Определяем тип нормы из функций
                        norm_type = norm_functions[norm_number_str]['norm_type']
                        
                        # Выбираем параметр для интерполяции в зависимости от типа нормы
                        if norm_type == 'Вес':
                            # Для норм по весу используем БРУТТО
                            parameter_value = self._calculate_weight_from_data(row)
                            parameter_name = 'вес поезда (БРУТТО)'
                        else:
                            # Для норм по нажатию используем нажатие на ось
                            parameter_value = self._calculate_axle_load_from_data(row)
                            parameter_name = 'нажатие на ось'
                        
                        if parameter_value and parameter_value > 0:
                            norm_func = norm_functions[norm_number_str]['function']
                            norm_value = float(norm_func(parameter_value))
                            routes_df.loc[i, 'Норма интерполированная'] = norm_value
                            routes_df.loc[i, 'Параметр нормирования'] = parameter_name
                            routes_df.loc[i, 'Значение параметра'] = parameter_value
                            
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
        
        # Создаем базовую структуру графика
        fig = self._create_base_plot_structure(section_name, title_suffix, filter_suffix)
        
        # Определяем типы норм
        norm_types_used = self._get_norm_types_used(norm_functions)
        
        # Добавляем кривые норм на верхний график
        self._add_norm_curves(fig, norm_functions, routes_df, specific_norm_id, norm_types_used)
        
        # Добавляем точки маршрутов на верхний график
        self._add_route_points(fig, routes_df, norm_functions, norm_types_used)
        
        # Добавляем точки отклонений на нижний график
        self._add_deviation_points(fig, routes_df)
        
        # Добавляем граничные линии
        self._add_boundary_lines(fig, routes_df)
        
        # Настраиваем оси и макет
        self._configure_plot_layout(fig, norm_types_used)
        
        return fig

    def _create_base_plot_structure(self, section_name: str, title_suffix: str, filter_suffix: str) -> go.Figure:
        """Создает базовую структуру графика с двумя подграфиками"""
        return make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f"Нормы расхода для участка: {section_name}{title_suffix}{filter_suffix}",
                "Отклонение фактического расхода от нормы"
            )
        )

    def _get_norm_types_used(self, norm_functions: Dict) -> set:
        """Определяет используемые типы норм"""
        norm_types_used = set()
        for norm_data in norm_functions.values():
            norm_type = norm_data.get('norm_type', 'Нажатие')
            norm_types_used.add(norm_type)
        return norm_types_used

    def _add_norm_curves(self, fig: go.Figure, norm_functions: Dict, routes_df: pd.DataFrame, 
                        specific_norm_id: Optional[str], norm_types_used: set):
        """Добавляет кривые норм на верхний график"""
        logger.debug(f"Добавление {len(norm_functions)} норм на график")
        
        for norm_id, norm_data in norm_functions.items():
            # Пропускаем норму если указана конкретная и это не она
            if specific_norm_id and norm_id != specific_norm_id:
                continue
                
            self._add_single_norm_curve(fig, norm_id, norm_data, routes_df, norm_types_used)

    def _add_single_norm_curve(self, fig: go.Figure, norm_id: str, norm_data: Dict, 
                            routes_df: pd.DataFrame, norm_types_used: set):
        """Добавляет одну кривую нормы на график"""
        points = norm_data['points']
        norm_type = norm_data.get('norm_type', 'Нажатие')
        
        # Определяем название оси X
        x_axis_name = "Вес поезда БРУТТО, т" if norm_type == 'Вес' else "Нажатие на ось, т/ось"
        
        if len(points) == 1:
            self._add_constant_norm_curve(fig, norm_id, points[0], routes_df, norm_type, x_axis_name)
        else:
            self._add_interpolated_norm_curve(fig, norm_id, points, routes_df, norm_type, x_axis_name)
        
        # Добавляем дополнительные точки из маршрутов
        self._add_additional_norm_points(fig, norm_id, norm_type, routes_df, x_axis_name)

    def _add_constant_norm_curve(self, fig: go.Figure, norm_id: str, point: tuple, 
                            routes_df: pd.DataFrame, norm_type: str, x_axis_name: str):
        """Добавляет константную норму (одна точка)"""
        x_single, y_single = point
        
        # Определяем диапазон отображения на основе данных маршрутов
        x_data_values = self._get_x_values_for_norm(routes_df, norm_id, norm_type)
        
        if x_data_values:
            x_min_data, x_max_data = min(x_data_values), max(x_data_values)
            x_range_data = x_max_data - x_min_data
            x_start = max(x_min_data - x_range_data * 0.2, x_min_data * 0.8)
            x_end = x_max_data + x_range_data * 0.2
        else:
            x_start = max(x_single * 0.5, x_single - 100)
            x_end = x_single * 1.5 + 100
        
        x_const = np.linspace(x_start, x_end, 100)
        y_const = np.full_like(x_const, y_single)
        
        fig.add_trace(
            go.Scatter(
                x=x_const,
                y=y_const,
                mode='lines',
                name=f'Норма {norm_id} (константа)',
                line=dict(width=3, color='blue'),
                hovertemplate=f'<b>Норма {norm_id}</b><br>' +
                            f'{x_axis_name}: %{{x:.1f}}<br>' +
                            'Расход: %{y:.1f} кВт·ч/10⁴ ткм<extra></extra>'
            ),
            row=1, col=1
        )

    def _add_interpolated_norm_curve(self, fig: go.Figure, norm_id: str, points: list, 
                                routes_df: pd.DataFrame, norm_type: str, x_axis_name: str):
        """Добавляет интерполированную кривую нормы"""
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        
        # Расширяем диапазон для отображения
        x_min, x_max = min(x_vals), max(x_vals)
        x_range = x_max - x_min
        x_start = max(x_min - x_range * 0.3, x_min * 0.5)
        x_end = x_max + x_range * 0.3
        
        # Учитываем дополнительные точки из маршрутов
        additional_x = self._get_additional_points_for_norm(routes_df, norm_id, norm_type)
        if additional_x:
            x_start = min(x_start, min(additional_x) * 0.8)
            x_end = max(x_end, max(additional_x) * 1.2)
        
        # Создаем гладкую интерполяцию
        x_interp = np.linspace(x_start, x_end, 500)
        y_interp = self._interpolate_norm_values(x_vals, y_vals, x_interp)
        
        # Убираем недопустимые значения
        valid_mask = np.isfinite(y_interp) & (y_interp > 0)
        x_interp_clean = x_interp[valid_mask]
        y_interp_clean = y_interp[valid_mask]
        
        curve_name = f'Норма {norm_id} ({len(points)} точек)'
        
        fig.add_trace(
            go.Scatter(
                x=x_interp_clean,
                y=y_interp_clean,
                mode='lines',
                name=curve_name,
                line=dict(width=3, color='blue'),
                hovertemplate=f'<b>Норма {norm_id}</b><br>' +
                            f'{x_axis_name}: %{{x:.1f}}<br>' +
                            'Расход: %{y:.1f} кВт·ч/10⁴ ткм<extra></extra>'
            ),
            row=1, col=1
        )

    def _get_x_values_for_norm(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> list:
        """Получает X значения для данной нормы из маршрутов"""
        x_values = []
        
        for _, row in routes_df.iterrows():
            norm_number = row.get('Номер нормы')
            if pd.notna(norm_number) and str(int(norm_number)) == norm_id:
                if norm_type == 'Вес':
                    x_val = self._calculate_weight_from_data(row)
                else:
                    x_val = self._calculate_axle_load_from_data(row)
                
                if x_val and x_val > 0:
                    x_values.append(x_val)
        
        return x_values

    def _add_additional_norm_points(self, fig: go.Figure, norm_id: str, norm_type: str, 
                                routes_df: pd.DataFrame, x_axis_name: str):
        """Добавляет дополнительные точки норм из маршрутов"""
        additional_points_with_info = self._extract_additional_norm_points_with_route_info(
            routes_df, norm_id, norm_type)
        
        if additional_points_with_info:
            add_x = [point[0] for point in additional_points_with_info]
            add_y = [point[1] for point in additional_points_with_info]
            
            hover_texts = []
            for x, y, routes in additional_points_with_info:
                hover_text = (
                    f"<b>Из маршрута № {routes}</b><br>"
                    f"{x_axis_name}: {x:.1f}<br>"
                    f"Расход: {y:.1f} кВт·ч/10⁴ ткм"
                )
                hover_texts.append(hover_text)
            
            fig.add_trace(
                go.Scatter(
                    x=add_x,
                    y=add_y,
                    mode='markers',
                    name=f'Из маршрутов {norm_id} ({len(additional_points_with_info)})',
                    marker=dict(size=6, symbol='circle', opacity=0.7, color='orange'),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts
                ),
                row=1, col=1
            )

    def _add_route_points(self, fig: go.Figure, routes_df: pd.DataFrame, 
                        norm_functions: Dict, norm_types_used: set):
        """Добавляет точки маршрутов на верхний график"""
        status_colors = {
            'Экономия сильная': 'darkgreen',
            'Экономия средняя': 'green', 
            'Экономия слабая': 'lightgreen',
            'Норма': 'blue',
            'Перерасход слабый': 'orange',
            'Перерасход средний': 'darkorange',
            'Перерасход сильный': 'red'
        }
        
        for status, color in status_colors.items():
            status_data = routes_df[routes_df['Статус'] == status]
            
            if not status_data.empty:
                for norm_type in norm_types_used:
                    self._add_status_norm_type_points(
                        fig, status_data, norm_functions, norm_type, 
                        status, color, routes_df
                    )

    def _add_status_norm_type_points(self, fig: go.Figure, status_data: pd.DataFrame, 
                                norm_functions: Dict, norm_type: str, status: str, 
                                color: str, routes_df: pd.DataFrame):
        """Добавляет точки для конкретного статуса и типа нормы"""
        x_vals_routes = []
        y_vals_routes = []
        hover_texts = []
        custom_data_list = []
        
        for _, row in status_data.iterrows():
            route_data = self._process_single_route_point(
                row, norm_functions, norm_type, routes_df
            )
            
            if route_data:
                x_vals_routes.append(route_data['x_val'])
                y_vals_routes.append(route_data['y_val'])
                hover_texts.append(route_data['hover_text'])
                custom_data_list.append(route_data['custom_data'])
        
        if x_vals_routes:
            fig.add_trace(
                go.Scatter(
                    x=x_vals_routes,
                    y=y_vals_routes,
                    mode='markers',
                    name=f'{status} ({norm_type})',
                    marker=dict(color=color, size=6, opacity=0.7),
                    customdata=custom_data_list,
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts
                ),
                row=1, col=1
            )

    def _process_single_route_point(self, row: pd.Series, norm_functions: Dict, 
                                norm_type: str, routes_df: pd.DataFrame) -> Optional[Dict]:
        """Обрабатывает одну точку маршрута для графика"""
        norm_number = row.get('Номер нормы')
        if not pd.notna(norm_number):
            return None
        
        norm_number_str = str(int(norm_number))
        if norm_number_str not in norm_functions:
            return None
        
        current_norm_type = norm_functions[norm_number_str].get('norm_type', 'Нажатие')
        if current_norm_type != norm_type:
            return None
        
        # Определяем координаты
        if norm_type == 'Вес':
            x_val = self._calculate_weight_from_data(row)
            x_label = "Вес БРУТТО"
        else:
            x_val = self._calculate_axle_load_from_data(row)
            x_label = "Нажатие на ось"
        
        if not x_val or x_val <= 0:
            return None
        
        actual_value = row.get('Факт уд') or row.get('Расход фактический')
        if not actual_value:
            return None
        
        # Создаем информацию о маршруте для hover
        route_info = self._create_route_info_for_hover(row)
        route_title = " | ".join(route_info) if route_info else "Маршрут"
        
        # Создаем hover текст
        hover_text = (
            f"<b>{route_title}</b><br>"
            f"{x_label}: {x_val:.1f}<br>"
            f"Факт: {actual_value:.1f}<br>"
            f"Норма: {row.get('Норма интерполированная', 'N/A'):.1f}<br>"
            f"Расход фактический: {row.get('Расход фактический', 'N/A'):.1f}<br>"
            f"Расход по норме: {row.get('Расход по норме', 'N/A'):.1f}<br>"
            f"Отклонение: {row.get('Отклонение, %', 'N/A'):.1f}%<br>"
            f"Номер нормы: {norm_number_str}"
        )
        
        # Создаем полную информацию для клика
        custom_data = self._create_full_route_info(row, routes_df)
        
        return {
            'x_val': x_val,
            'y_val': actual_value,
            'hover_text': hover_text,
            'custom_data': custom_data
        }

    def _create_route_info_for_hover(self, row: pd.Series) -> list:
        """Создает краткую информацию о маршруте для hover"""
        route_info = []
        
        if 'Номер маршрута' in row and pd.notna(row['Номер маршрута']):
            route_info.append(f"Маршрут №{row['Номер маршрута']}")
        
        if 'Дата маршрута' in row and pd.notna(row['Дата маршрута']):
            route_info.append(f"Дата: {row['Дата маршрута']}")
        
        if 'Наименование участка' in row and pd.notna(row['Наименование участка']):
            route_info.append(f"Участок: {row['Наименование участка']}")
        elif 'Участок' in row and pd.notna(row['Участок']):
            route_info.append(f"Участок: {row['Участок']}")
        
        # Информация о локомотиве
        if 'Серия локомотива' in row and pd.notna(row['Серия локомотива']):
            loco_info = f"Локомотив: {row['Серия локомотива']}"
            if 'Номер локомотива' in row and pd.notna(row['Номер локомотива']):
                loco_info += f" №{row['Номер локомотива']}"
            route_info.append(loco_info)
        elif 'Серия ТПС' in row and pd.notna(row['Серия ТПС']):
            loco_info = f"ТПС: {row['Серия ТПС']}"
            if 'Номер ТПС' in row and pd.notna(row['Номер ТПС']):
                loco_info += f" №{row['Номер ТПС']}"
            route_info.append(loco_info)
        
        return route_info

    def _configure_plot_layout(self, fig: go.Figure, norm_types_used: set):
        """Настраивает оси и макет графика"""
        # Определяем подписи осей
        mixed_types = len(norm_types_used) > 1
        
        if mixed_types:
            x_axis_title = "Параметр нормирования (т/ось или т БРУТТО)"
        elif 'Вес' in norm_types_used:
            x_axis_title = "Вес поезда БРУТТО, т"
        else:
            x_axis_title = "Нажатие на ось, т/ось"
        
        fig.update_xaxes(title_text=x_axis_title, row=1, col=1)
        fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм", row=1, col=1)
        fig.update_xaxes(title_text=x_axis_title, row=2, col=1)
        fig.update_yaxes(title_text="Отклонение, %", row=2, col=1)
        
        # Добавляем линию нуля на график отклонений
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        fig.update_layout(
            height=800,
            hovermode='closest',
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

    def _interpolate_norm_values(self, x_vals: list, y_vals: list, x_interp: np.ndarray) -> np.ndarray:
        """Выполняет интерполяцию значений нормы (гипербола или линейная)"""
        try:
            # Пытаемся создать гиперболическую функцию Y = A/X + B
            if len(x_vals) >= 2:
                # Решаем систему уравнений для гиперболы
                A = np.array([[1/x_vals[i], 1] for i in range(len(x_vals))])
                b = np.array(y_vals)
                
                try:
                    params = np.linalg.lstsq(A, b, rcond=None)[0]
                    a_coef, b_coef = params
                    
                    # Проверяем качество аппроксимации
                    y_pred = a_coef / np.array(x_vals) + b_coef
                    mse = np.mean((np.array(y_vals) - y_pred) ** 2)
                    
                    if mse < 1000:  # Если гипербола хорошо подходит
                        return a_coef / x_interp + b_coef
                except:
                    pass
            
            # Если гипербола не подходит, используем сплайн
            from scipy.interpolate import interp1d
            f = interp1d(x_vals, y_vals, kind='cubic', fill_value='extrapolate')
            return f(x_interp)
            
        except Exception as e:
            logger.debug(f"Ошибка интерполяции: {e}, используем линейную")
            return np.interp(x_interp, x_vals, y_vals)
    
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
        
        # НОВОЕ: Добавляем информацию о расходах
        rashod_fact = route.get('Расход фактический')
        rashod_norm = route.get('Расход по норме')
        
        hover_text += (
            f"Нажатие: {axle_load:.2f} т/ось<br>" if axle_load else "Нажатие: N/A<br>"
            f"Факт: {consumption_value:.1f}<br>" if consumption_value else "Факт: N/A<br>"
            f"Норма: {route.get('Норма интерполированная', 'N/A'):.1f}<br>"
            f"Расход фактический: {rashod_fact:.1f}<br>" if pd.notna(rashod_fact) else "Расход фактический: N/A<br>"
            f"Расход по норме: {rashod_norm:.1f}<br>" if pd.notna(rashod_norm) else "Расход по норме: N/A<br>"
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
                    custom_data_list = []  # НОВОЕ: добавляем customdata и для нижнего графика
                    
                    for _, route in status_data.iterrows():
                        axle_load = self._calculate_axle_load_from_data(route)
                        if axle_load:
                            x_values.append(axle_load)
                            y_values.append(route['Отклонение, %'])
                            hover_texts.append(self._create_hover_text(route))
                            # НОВОЕ: добавляем полную информацию о маршруте
                            custom_data_list.append(self._create_full_route_info(route))
                    
                    if x_values:
                        fig.add_trace(
                            go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='markers',
                                name=f'{status} ({len(x_values)})',
                                marker=dict(
                                    color=color,
                                    size=10,
                                    opacity=0.8,
                                    line=dict(color='black', width=0.5)
                                ),
                                customdata=custom_data_list,  # НОВОЕ: добавляем customdata
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
    
    def _get_norm_type_from_storage(self, norm_id: str) -> str:
        """Определяет тип нормы из хранилища норм"""
        try:
            norm_data = self.norm_storage.get_norm(norm_id)
            if norm_data:
                return norm_data.get('norm_type', 'Нажатие')
            return 'Нажатие'
        except Exception as e:
            logger.debug(f"Не удалось определить тип нормы {norm_id}: {e}")
            return 'Нажатие'

    def _calculate_weight_from_data(self, row: pd.Series) -> Optional[float]:
        """Вычисляет вес поезда БРУТТО из данных"""
        try:
            # Пытаемся получить БРУТТО напрямую
            brutto = row.get('БРУТТО')
            if pd.notna(brutto) and brutto != "-" and isinstance(brutto, (int, float)):
                return float(brutto)
            
            # Альтернативный расчет из Ткм брутто и км
            tkm_brutto = row.get('Ткм брутто')
            km = row.get('Км')
            
            if pd.notna(tkm_brutto) and pd.notna(km) and km != 0:
                return float(tkm_brutto) / float(km)
            
            return None
                
        except (ValueError, TypeError, ZeroDivisionError) as e:
            logger.debug(f"Ошибка расчета веса поезда: {e}")
            return None
        
    def get_norm_info_with_type(self, norm_id: str) -> Optional[Dict]:
        """Получает информацию о норме включая ее тип"""
        try:
            norm_info = self.get_norm_info(norm_id)
            if norm_info:
                norm_info['norm_type'] = self._get_norm_type_from_storage(norm_id)
            return norm_info
        except Exception as e:
            logger.error(f"Ошибка получения информации о норме {norm_id}: {e}")
            return None

    def _get_additional_points_for_norm(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> List[float]:
        """Получает X-координаты дополнительных точек для нормы"""
        x_values = []
        
        for _, row in routes_df.iterrows():
            route_norm_id = row.get('Номер нормы')
            if pd.notna(route_norm_id) and str(int(route_norm_id)) == norm_id:
                if norm_type == 'Вес':
                    x_val = self._calculate_weight_from_data(row)
                else:
                    x_val = self._calculate_axle_load_from_data(row)
                if x_val and x_val > 0:
                    x_values.append(x_val)
        
        return x_values

    def _extract_additional_norm_points(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> List[Tuple[float, float]]:
        """Извлекает дополнительные точки норм из маршрутов"""
        points = []
        logger.debug(f"Поиск дополнительных точек для нормы {norm_id}, тип: {norm_type}")
        logger.debug(f"Доступные колонки: {list(routes_df.columns)}")
        
        # Возможные названия колонок для удельной нормы
        ud_norma_columns = [
            'Уд. норма, норма на 1 час ман. раб.',
            'Удельная норма',
            'Уд норма',
            'Норма на 1 час',
            'УД. НОРМА'
        ]
        
        # Находим реальную колонку удельной нормы
        ud_norma_col = None
        for col in ud_norma_columns:
            if col in routes_df.columns:
                ud_norma_col = col
                logger.debug(f"Найдена колонка удельной нормы: {col}")
                break
        
        if not ud_norma_col:
            logger.debug(f"Колонка удельной нормы не найдена среди: {ud_norma_columns}")
            return points
        
        # Извлекаем точки для конкретной нормы
        for _, row in routes_df.iterrows():
            try:
                route_norm_id = row.get('Номер нормы')
                if pd.notna(route_norm_id) and str(int(route_norm_id)) == norm_id:
                    
                    ud_norma = row.get(ud_norma_col)
                    
                    if pd.notna(ud_norma) and ud_norma != '' and ud_norma != '-':
                        try:
                            ud_norma_val = float(ud_norma)
                            
                            if ud_norma_val > 0:
                                # Для норм по весу используем расчет веса
                                if norm_type == 'Вес':
                                    x_val = self._calculate_weight_from_data(row)
                                else:
                                    x_val = self._calculate_axle_load_from_data(row)
                                
                                if x_val and x_val > 0:
                                    points.append((x_val, ud_norma_val))
                                    
                        except (ValueError, TypeError):
                            continue
                            
            except Exception as e:
                logger.debug(f"Ошибка обработки строки: {e}")
                continue
        
        # Убираем дубликаты и сортируем
        if points:
            points = list(set(points))
            points.sort(key=lambda x: x[0])
        
        logger.debug(f"Извлечено {len(points)} дополнительных точек для нормы {norm_id}")
        return points

    def _calculate_weight_from_data(self, row: pd.Series) -> Optional[float]:
        """Вычисляет вес поезда из данных маршрута"""
        try:
            # Сначала пробуем прямые колонки с весом
            weight_columns = ['БРУТТО', 'Вес БРУТТО', 'Вес поезда БРУТТО', 'Брутто']
            
            for col in weight_columns:
                if col in row.index:
                    brutto = row.get(col)
                    if pd.notna(brutto) and brutto != '-' and brutto != '':
                        try:
                            weight_val = float(brutto)
                            if weight_val > 0:
                                return weight_val
                        except (ValueError, TypeError):
                            continue
            
            # Если прямых данных нет, рассчитываем из Ткм брутто / км
            tkm_brutto = row.get('Ткм брутто')
            km = row.get('Км')
            
            if pd.notna(tkm_brutto) and pd.notna(km) and km != 0:
                try:
                    weight_calc = float(tkm_brutto) / float(km)
                    if weight_calc > 0:
                        return weight_calc
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Ошибка расчета веса: {e}")
            return None

    def _calculate_weight_from_data(self, row: pd.Series) -> Optional[float]:
        """Вычисляет вес поезда из данных маршрута"""
        try:
            # Сначала пробуем прямые колонки с весом
            weight_columns = ['БРУТТО', 'Вес БРУТТО', 'Вес поезда БРУТТО', 'Брутто']
            
            for col in weight_columns:
                if col in row.index:
                    brutto = row.get(col)
                    if pd.notna(brutto) and brutto != '-' and brutto != '':
                        try:
                            weight_val = float(brutto)
                            if weight_val > 0:
                                return weight_val
                        except (ValueError, TypeError):
                            continue
            
            # Если прямых данных нет, рассчитываем из Ткм брутто / км
            tkm_brutto = row.get('Ткм брутто')
            km = row.get('Км')
            
            if pd.notna(tkm_brutto) and pd.notna(km) and km != 0:
                try:
                    weight_calc = float(tkm_brutto) / float(km)
                    if weight_calc > 0:
                        return weight_calc
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            return None
            
        except Exception as e:
            logger.debug(f"Ошибка расчета веса: {e}")
            return None

    def _remove_duplicate_points(self, points: List[Tuple[float, float]], tolerance: float = 0.1) -> List[Tuple[float, float]]:
        """Удаляет дублирующиеся точки с учетом допуска"""
        if not points:
            return []
        
        unique_points = []
        for point in points:
            x, y = point
            # Проверяем, нет ли уже близкой точки по X
            is_duplicate = False
            for existing_point in unique_points:
                existing_x, existing_y = existing_point
                if abs(x - existing_x) <= tolerance:
                    # Если X близки, берем среднее значение Y
                    avg_y = (y + existing_y) / 2
                    unique_points[unique_points.index(existing_point)] = (existing_x, avg_y)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_points.append(point)
        
        return unique_points


    def _extract_additional_norm_points_with_route_info(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> List[Tuple[float, float, str]]:
        """Извлекает дополнительные точки норм из маршрутов с информацией о маршруте
        
        Args:
            routes_df: DataFrame с данными маршрутов
            norm_id: ID нормы
            norm_type: Тип нормы ('Вес' или 'Нажатие')
            
        Returns:
            List[Tuple[float, float, str]]: Список кортежей (x, y, route_numbers)
            где x - нагрузка, y - удельная норма, route_numbers - номера маршрутов
        """
        points_with_info = []
        logger.debug(f"Поиск дополнительных точек с информацией о маршруте для нормы {norm_id}, тип: {norm_type}")
        
        # Возможные названия колонок для удельной нормы
        ud_norma_columns = [
            'Уд. норма, норма на 1 час ман. раб.',
            'Удельная норма',
            'Уд норма',
            'Норма на 1 час',
            'УД. НОРМА'
        ]
        
        # Находим реальную колонку удельной нормы
        ud_norma_col = None
        for col in ud_norma_columns:
            if col in routes_df.columns:
                ud_norma_col = col
                break
        
        if not ud_norma_col:
            logger.debug(f"Колонка удельной нормы не найдена среди: {ud_norma_columns}")
            return points_with_info
        
        # Извлекаем точки для конкретной нормы с информацией о маршруте
        for _, row in routes_df.iterrows():
            try:
                route_norm_id = row.get('Номер нормы')
                if pd.notna(route_norm_id) and str(int(route_norm_id)) == norm_id:
                    
                    ud_norma = row.get(ud_norma_col)
                    
                    if pd.notna(ud_norma) and ud_norma != '' and ud_norma != '-':
                        try:
                            ud_norma_val = float(ud_norma)
                            
                            if ud_norma_val > 0:
                                # Для норм по весу используем расчет веса
                                if norm_type == 'Вес':
                                    x_val = self._calculate_weight_from_data(row)
                                else:
                                    x_val = self._calculate_axle_load_from_data(row)
                                
                                if x_val and x_val > 0:
                                    # Получаем номер маршрута
                                    route_number = row.get('Номер маршрута', 'N/A')
                                    points_with_info.append((x_val, ud_norma_val, str(route_number)))
                                    
                        except (ValueError, TypeError):
                            continue
                            
            except Exception as e:
                logger.debug(f"Ошибка обработки строки: {e}")
                continue
        
        # Группируем близкие точки и объединяем номера маршрутов
        if points_with_info:
            unique_points = {}
            for x, y, route in points_with_info:
                key = (round(x, 2), round(y, 1))  # Округляем для группировки близких точек
                if key not in unique_points:
                    unique_points[key] = (x, y, [route])
                else:
                    existing_x, existing_y, existing_routes = unique_points[key]
                    if route not in existing_routes:
                        existing_routes.append(route)
                    unique_points[key] = (existing_x, existing_y, existing_routes)
            
            # Преобразуем в финальный формат
            points_with_info = []
            for (x, y, routes) in unique_points.values():
                # Сортируем номера маршрутов и объединяем через запятую
                route_str = ', '.join(sorted(routes, key=lambda r: int(r) if r.isdigit() else float('inf')))
                points_with_info.append((x, y, route_str))
            
            points_with_info.sort(key=lambda x: x[0])  # Сортируем по x
            
            logger.info(f"✅ Найдено {len(points_with_info)} дополнительных точек с информацией о маршрутах для нормы {norm_id}")
        
        return points_with_info
    
    def _add_browser_mode_switcher(self, html_content: str) -> str:
        """Добавляет переключатель режимов и обработчик кликов в HTML файл браузера"""
        
        js_code = '''
        <div id="mode-switcher" style="
            position: fixed; 
            top: 10px; 
            right: 10px; 
            z-index: 1000; 
            background: white; 
            padding: 15px; 
            border: 2px solid #4a90e2; 
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            font-family: Arial, sans-serif;
            font-size: 14px;
        ">
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

        <!-- Модальное окно для полной информации о маршруте -->
        <div id="route-modal" style="
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
        ">
            <div style="
                background-color: white;
                margin: 2% auto;
                padding: 20px;
                border-radius: 10px;
                width: 95%;
                max-width: 1400px;
                max-height: 90%;
                overflow-y: auto;
                position: relative;
            ">
                <span id="close-modal" style="
                    position: absolute;
                    right: 20px;
                    top: 20px;
                    font-size: 28px;
                    font-weight: bold;
                    cursor: pointer;
                    color: #aaa;
                ">&times;</span>
                <div id="route-details"></div>
            </div>
        </div>

        <script>
        let originalData = {};
        let plotlyDiv = null;

        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(function() {
                plotlyDiv = document.querySelector('.js-plotly-plot') || document.querySelector('[data-plotly]');
                
                if (plotlyDiv && plotlyDiv.data) {
                    console.log('График найден, настройка обработчиков...');
                    saveOriginalData();
                    
                    document.querySelectorAll('input[name="display_mode"]').forEach(radio => {
                        radio.addEventListener('change', switchDisplayMode);
                    });
                    
                    plotlyDiv.on('plotly_click', handlePointClick);
                    
                    document.getElementById('close-modal').onclick = function() {
                        document.getElementById('route-modal').style.display = 'none';
                    };
                    
                    document.getElementById('route-modal').onclick = function(event) {
                        if (event.target === this) {
                            this.style.display = 'none';
                        }
                    };
                } else {
                    setTimeout(arguments.callee, 500);
                }
            }, 1000);
        });

        function saveOriginalData() {
            originalData = {};
            plotlyDiv.data.forEach((trace, index) => {
                if (trace.customdata && trace.y) {
                    originalData[index] = {
                        x: [...trace.x],
                        y: [...trace.y],
                        customdata: trace.customdata ? JSON.parse(JSON.stringify(trace.customdata)) : null
                    };
                }
            });
        }

        function switchDisplayMode() {
            const mode = document.querySelector('input[name="display_mode"]:checked').value;
            
            if (!plotlyDiv || !originalData) return;
            
            const update = {};
            let updatedTraces = 0;
            
            plotlyDiv.data.forEach((trace, index) => {
                if (originalData[index] && trace.customdata) {
                    if (mode === 'nf') {
                        const newY = trace.y.map((originalY, pointIndex) => {
                            const customData = trace.customdata[pointIndex];
                            
                            if (customData && 
                                customData.rashod_fact != null && 
                                customData.rashod_norm != null && 
                                customData.norm_interpolated != null &&
                                customData.rashod_norm > 0) {
                                
                                const deviationPercent = ((customData.rashod_fact - customData.rashod_norm) / customData.rashod_norm) * 100;
                                const adjustedY = customData.norm_interpolated * (1 + deviationPercent / 100);
                                return adjustedY;
                            }
                            return originalY;
                        });
                        
                        update['y[' + index + ']'] = newY;
                        updatedTraces++;
                    } else {
                        update['y[' + index + ']'] = originalData[index].y;
                        updatedTraces++;
                    }
                }
            });
            
            if (Object.keys(update).length > 0) {
                Plotly.restyle(plotlyDiv, update);
            }
        }
        
        function handlePointClick(data) {
            if (!data.points || data.points.length === 0) return;
            
            const point = data.points[0];
            const customData = point.customdata;
            
            if (!customData) return;
            
            showFullRouteInfo(customData);
        }
        
        function showFullRouteInfo(customData) {
            let detailsHtml = `<h2>Подробная информация о маршруте №${customData.route_number}</h2>`;
            
            // Основная информация
            detailsHtml += `<div style="margin-bottom: 20px;">`;
            detailsHtml += `<h3>Основная информация</h3>`;
            detailsHtml += `<table style="border-collapse: collapse; width: 50%; font-family: Arial;">`;
            
            detailsHtml += addTableRow('Номер маршрута', customData.route_number);
            detailsHtml += addTableRow('Дата маршрута', customData.route_date);
            detailsHtml += addTableRow('Дата поездки', customData.trip_date);
            detailsHtml += addTableRow('Табельный машиниста', customData.driver_tab);
            detailsHtml += addTableRow('Серия локомотива', customData.locomotive_series);
            detailsHtml += addTableRow('Номер локомотива', customData.locomotive_number);
            detailsHtml += addTableRow('Расход фактический, всего', customData.rashod_fact_total, customData.use_red_rashod);
            detailsHtml += addTableRow('Расход по норме, всего', customData.rashod_norm_total, customData.use_red_rashod);
            
            detailsHtml += `</table></div>`;
            
            // Информация по участкам
            detailsHtml += `<div style="margin-bottom: 20px;">`;
            detailsHtml += `<h3>Информация по участкам</h3>`;
            
            // Создаем горизонтальную таблицу как в Excel
            detailsHtml += `<table style="border-collapse: collapse; width: 100%; font-family: Arial; font-size: 11px;">`;
            
            // Заголовки
            detailsHtml += `<tr style="background-color: #f0f0f0; font-weight: bold;">`;
            const headers = [
                'Наименование участка', 'НЕТТО', 'БРУТТО', 'ОСИ', 'Номер нормы', 'Дв. тяга',
                'Ткм брутто', 'Км', 'Пр.', 'Расход фактический', 'Расход по норме',
                'Уд. норма, норма на 1 час ман. раб.', 'Нажатие на ось', 'Норма на работу',
                'Факт уд', 'Факт на работу', 'Норма на одиночное',
                'Простой с бригадой, мин., всего', 'Простой с бригадой, мин., норма',
                'Маневры, мин., всего', 'Маневры, мин., норма',
                'Трогание с места, случ., всего', 'Трогание с места, случ., норма',
                'Нагон опозданий, мин., всего', 'Нагон опозданий, мин., норма',
                'Ограничения скорости, случ., всего', 'Ограничения скорости, случ., норма',
                'На пересылаемые л-вы, всего', 'На пересылаемые л-вы, норма',
                'Количество дубликатов маршрута'
            ];
            
            headers.forEach(header => {
                detailsHtml += `<td style="padding: 4px; border: 1px solid #ddd; text-align: center; font-size: 10px;">${header}</td>`;
            });
            detailsHtml += `</tr>`;
            
            // Данные участка
            detailsHtml += `<tr>`;
            
            // Функция для определения стиля ячейки
            function getCellStyle(fieldName, customData) {
                let baseStyle = 'padding: 4px; border: 1px solid #ddd; text-align: center; font-size: 11px;';
                
                if (['НЕТТО', 'БРУТТО', 'ОСИ'].includes(fieldName) && customData.use_red_color) {
                    baseStyle += ' background-color: #ffcccc; color: #ff0000; font-weight: bold;';
                }
                else if (['Расход фактический', 'Расход по норме'].includes(fieldName) && customData.use_red_rashod) {
                    baseStyle += ' background-color: #ffcccc; color: #ff0000; font-weight: bold;';
                }
                
                return baseStyle;
            }
            
            // Добавляем значения в том же порядке что и заголовки
            const values = [
                customData.section_name,
                customData.netto,
                customData.brutto, 
                customData.osi,
                customData.norm_number,
                customData.movement_type,
                customData.tkm_brutto,
                customData.km,
                customData.pr,
                customData.rashod_fact,
                customData.rashod_norm,
                customData.ud_norma,
                customData.axle_load,
                customData.norma_work,
                customData.fact_ud,
                customData.fact_work,
                customData.norma_single,
                customData.idle_brigada_total,
                customData.idle_brigada_norm,
                customData.manevr_total,
                customData.manevr_norm,
                customData.start_total,
                customData.start_norm,
                customData.delay_total,
                customData.delay_norm,
                customData.speed_limit_total,
                customData.speed_limit_norm,
                customData.transfer_loco_total,
                customData.transfer_loco_norm,
                customData.duplicates_count
            ];
            
            values.forEach((value, index) => {
                const header = headers[index];
                const displayValue = (value !== null && value !== undefined && value !== 'N/A') ? value : '-';
                const cellStyle = getCellStyle(header, customData);
                
                detailsHtml += `<td style="${cellStyle}">${displayValue}</td>`;
            });
            
            detailsHtml += `</tr>`;
            detailsHtml += `</table></div>`;
            
            // Дополнительная информация анализа
            detailsHtml += `<div style="margin-bottom: 20px;">`;
            detailsHtml += `<h3>Результаты анализа</h3>`;
            detailsHtml += `<table style="border-collapse: collapse; width: 50%; font-family: Arial;">`;
            
            detailsHtml += addTableRow('Норма интерполированная', customData.norm_interpolated);
            detailsHtml += addTableRow('Отклонение, %', customData.deviation_percent);
            detailsHtml += addTableRow('Статус', customData.status);
            detailsHtml += addTableRow('Н=Ф', customData.n_equals_f);
            
            if (customData.coefficient && customData.coefficient !== 1.0) {
                detailsHtml += addTableRow('Коэффициент', customData.coefficient);
                if (customData.fact_ud_original) {
                    detailsHtml += addTableRow('Факт. удельный исходный', customData.fact_ud_original);
                }
            }
            
            detailsHtml += `</table></div>`;
            
            document.getElementById('route-details').innerHTML = detailsHtml;
            document.getElementById('route-modal').style.display = 'block';
        }
        
        function addTableRow(label, value, isRed = false) {
            const redStyle = isRed ? 'background-color: #ffcccc; color: #ff0000; font-weight: bold;' : '';
            const displayValue = (value !== null && value !== undefined && value !== 'N/A') ? value : '-';
            
            return `<tr style="border: 1px solid #ddd;">
                <td style="padding: 8px; border: 1px solid #ddd; background-color: #f5f5f5; font-weight: bold;">${label}</td>
                <td style="padding: 8px; border: 1px solid #ddd; ${redStyle}">${displayValue}</td>
            </tr>`;
        }
        </script>
        '''
        
        if '</body>' in html_content:
            html_content = html_content.replace('</body>', js_code + '\n</body>')
        else:
            html_content += js_code
        
        return html_content

    def _create_full_route_info(self, route: pd.Series, routes_df: pd.DataFrame = None) -> Dict:
        """Создает полную информацию о маршруте для клика и customdata"""
        
        # Используем переданный DataFrame или self.routes_df
        source_df = routes_df if routes_df is not None else getattr(self, 'routes_df', None)
        
        # Рассчитываем информацию по всем участкам маршрута
        route_number = route.get('Номер маршрута')
        route_date = route.get('Дата маршрута')
        
        all_sections_data = []
        rashod_fact_total = 0
        rashod_norm_total = 0
        
        if route_number and route_date and source_df is not None:
            # Находим все участки этого маршрута
            same_route_data = source_df[
                (source_df['Номер маршрута'] == route_number) & 
                (source_df['Дата маршрута'] == route_date)
            ].copy()
            
            if not same_route_data.empty:
                logger.debug(f"Найдено {len(same_route_data)} участков для маршрута {route_number}")
                
                # Собираем данные по каждому участку
                for _, section_row in same_route_data.iterrows():
                    section_info = {
                        'section_name': section_row.get('Наименование участка', 'N/A'),
                        'netto': section_row.get('НЕТТО', 'N/A'),
                        'brutto': section_row.get('БРУТТО', 'N/A'),
                        'osi': section_row.get('ОСИ', 'N/A'),
                        'norm_number': section_row.get('Номер нормы', 'N/A'),
                        'movement_type': section_row.get('Дв. тяга', 'N/A'),
                        'tkm_brutto': section_row.get('Ткм брутто', 'N/A'),
                        'km': section_row.get('Км', 'N/A'),
                        'pr': section_row.get('Пр.', 'N/A'),
                        'rashod_fact': section_row.get('Расход фактический', 'N/A'),
                        'rashod_norm': section_row.get('Расход по норме', 'N/A'),
                        'ud_norma': section_row.get('Уд. норма, норма на 1 час ман. раб.', 'N/A'),
                        'axle_load': section_row.get('Нажатие на ось', 'N/A'),
                        'norma_work': section_row.get('Норма на работу', 'N/A'),
                        'fact_ud': section_row.get('Факт уд', 'N/A'),
                        'fact_work': section_row.get('Факт на работу', 'N/A'),
                        'norma_single': section_row.get('Норма на одиночное', 'N/A'),
                        'idle_brigada_total': section_row.get('Простой с бригадой, мин., всего', 'N/A'),
                        'idle_brigada_norm': section_row.get('Простой с бригадой, мин., норма', 'N/A'),
                        'manevr_total': section_row.get('Маневры, мин., всего', 'N/A'),
                        'manevr_norm': section_row.get('Маневры, мин., норма', 'N/A'),
                        'start_total': section_row.get('Трогание с места, случ., всего', 'N/A'),
                        'start_norm': section_row.get('Трогание с места, случ., норма', 'N/A'),
                        'delay_total': section_row.get('Нагон опозданий, мин., всего', 'N/A'),
                        'delay_norm': section_row.get('Нагон опозданий, мин., норма', 'N/A'),
                        'speed_limit_total': section_row.get('Ограничения скорости, случ., всего', 'N/A'),
                        'speed_limit_norm': section_row.get('Ограничения скорости, случ., норма', 'N/A'),
                        'transfer_loco_total': section_row.get('На пересылаемые л-вы, всего', 'N/A'),
                        'transfer_loco_norm': section_row.get('На пересылаемые л-вы, норма', 'N/A'),
                        'duplicates_count': section_row.get('Количество дубликатов маршрута', 'N/A'),
                        'use_red_color': section_row.get('USE_RED_COLOR', False),
                        'use_red_rashod': section_row.get('USE_RED_RASHOD', False)
                    }
                    all_sections_data.append(section_info)
                    
                    # Суммируем числовые значения
                    if pd.notna(section_row.get('Расход фактический')):
                        rashod_fact_total += section_row.get('Расход фактический', 0)
                    if pd.notna(section_row.get('Расход по норме')):
                        rashod_norm_total += section_row.get('Расход по норме', 0)
            else:
                logger.debug(f"Не найдены участки для маршрута {route_number}, создаем данные из текущей строки")
        
        # Если не найдены участки, создаем из текущей строки
        if not all_sections_data:
            logger.debug("Создаем данные участка из текущей строки")
            section_info = {
                'section_name': route.get('Наименование участка', 'N/A'),
                'netto': route.get('НЕТТО', 'N/A'),
                'brutto': route.get('БРУТТО', 'N/A'),
                'osi': route.get('ОСИ', 'N/A'),
                'norm_number': route.get('Номер нормы', 'N/A'),
                'movement_type': route.get('Дв. тяга', 'N/A'),
                'tkm_brutto': route.get('Ткм брутто', 'N/A'),
                'km': route.get('Км', 'N/A'),
                'pr': route.get('Пр.', 'N/A'),
                'rashod_fact': route.get('Расход фактический', 'N/A'),
                'rashod_norm': route.get('Расход по норме', 'N/A'),
                'ud_norma': route.get('Уд. норма, норма на 1 час ман. раб.', 'N/A'),
                'axle_load': route.get('Нажатие на ось', 'N/A'),
                'norma_work': route.get('Норма на работу', 'N/A'),
                'fact_ud': route.get('Факт уд', 'N/A'),
                'fact_work': route.get('Факт на работу', 'N/A'),
                'norma_single': route.get('Норма на одиночное', 'N/A'),
                'idle_brigada_total': route.get('Простой с бригадой, мин., всего', 'N/A'),
                'idle_brigada_norm': route.get('Простой с бригадой, мин., норма', 'N/A'),
                'manevr_total': route.get('Маневры, мин., всего', 'N/A'),
                'manevr_norm': route.get('Маневры, мин., норма', 'N/A'),
                'start_total': route.get('Трогание с места, случ., всего', 'N/A'),
                'start_norm': route.get('Трогание с места, случ., норма', 'N/A'),
                'delay_total': route.get('Нагон опозданий, мин., всего', 'N/A'),
                'delay_norm': route.get('Нагон опозданий, мин., норма', 'N/A'),
                'speed_limit_total': route.get('Ограничения скорости, случ., всего', 'N/A'),
                'speed_limit_norm': route.get('Ограничения скорости, случ., норма', 'N/A'),
                'transfer_loco_total': route.get('На пересылаемые л-вы, всего', 'N/A'),
                'transfer_loco_norm': route.get('На пересылаемые л-вы, норма', 'N/A'),
                'duplicates_count': route.get('Количество дубликатов маршрута', 'N/A'),
                'use_red_color': route.get('USE_RED_COLOR', False),
                'use_red_rashod': route.get('USE_RED_RASHOD', False)
            }
            all_sections_data.append(section_info)
            
            # Устанавливаем итоговые расходы из текущей строки
            if pd.notna(route.get('Расход фактический')):
                rashod_fact_total = route.get('Расход фактический', 0)
            if pd.notna(route.get('Расход по норме')):
                rashod_norm_total = route.get('Расход по норме', 0)
        
        # Берем основную информацию из первой строки (текущего участка)
        route_info = {
            'route_number': route.get('Номер маршрута', 'N/A'),
            'route_date': route.get('Дата маршрута', 'N/A'),
            'trip_date': route.get('Дата поездки', 'N/A'),
            'driver_tab': route.get('Табельный машиниста', 'N/A'),
            'locomotive_series': route.get('Серия локомотива', 'N/A'),
            'locomotive_number': route.get('Номер локомотива', 'N/A'),
            
            # Суммарные расходы
            'rashod_fact_total': rashod_fact_total if rashod_fact_total > 0 else 'N/A',
            'rashod_norm_total': rashod_norm_total if rashod_norm_total > 0 else 'N/A',
            
            # Все участки маршрута
            'all_sections': all_sections_data,
            
            # Данные текущего участка для обратной совместимости
            'norm_interpolated': route.get('Норма интерполированная', 'N/A'),
            'deviation_percent': route.get('Отклонение, %', 'N/A'),
            'status': route.get('Статус', 'N/A'),
            'n_equals_f': route.get('Н=Ф', 'N/A'),
            
            # Коэффициенты, если есть
            'coefficient': route.get('Коэффициент', None),
            'fact_ud_original': route.get('Факт. удельный исходный', None)
        }
        
        logger.debug(f"Создана информация о маршруте {route_number} с {len(all_sections_data)} участками")
        
        return route_info