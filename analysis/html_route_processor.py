#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный процессор для обработки HTML файлов маршрутов.
Устранены ошибки с обработкой данных и добавлена лучшая обработка ошибок.
"""

from __future__ import annotations

import os
import re
import logging
import tempfile
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path

# Безопасный импорт pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("Pandas не установлен! Процессор маршрутов не будет работать.")

from .data_models import (
    RouteMetadata, LocoData, Yu7Data, RouteSection, 
    ProcessingStats, safe_float_conversion, safe_int_conversion
)
from .html_parser import FastHTMLParser

logger = logging.getLogger(__name__)

# Типы для Python 3.12
type HTMLContent = str
type RouteData = Dict[str, Any]

class HTMLRouteProcessor:
    """Исправленный процессор для обработки HTML файлов маршрутов."""
    
    def __init__(self):
        self.processed_routes = []
        self.processing_stats = ProcessingStats()
        self.routes_df = None
        self.html_parser = FastHTMLParser()
        
        # Предкомпилированные паттерны для производительности
        self._compile_patterns()
        
        if not PANDAS_AVAILABLE:
            logger.error("Pandas недоступен. Процессор не может работать без pandas.")
    
    def _compile_patterns(self):
        """Предкомпилирует regex паттерны для лучшей производительности."""
        self.patterns = {
            'route_start': re.compile(r'<table[^>]*><tr><th class=thl_common><font class=filter_key>\s*Маршрут\s*№:.*?<br><br><br>', re.DOTALL),
            'form_end': re.compile(r'</table>\s*</td>\s*</tr></table><form id=print_form>.*?(?=\n|$)', re.DOTALL),
            'vcht_filter': re.compile(r'<td class = itog2>" ВЧТ "</td>'),
            'date_clean': re.compile(r'<font class = rcp12 ><center>Дата получения:.*?</font>\s*<br>', re.DOTALL),
            'route_num_clean': re.compile(r'<font class = rcp12 ><center>Номер маршрута:.*?</font><br>', re.DOTALL),
            'numline_clean': re.compile(r'<tr class=tr_numline>.*?</tr>', re.DOTALL),
            'table_width': re.compile(r'<table width=\d+%')
        }
    
    # ================== УТИЛИТЫ ==================
    
    def try_convert_to_number(self, value: Any, force_int: bool = False) -> Optional[float]:
        """Преобразует значение в число с улучшенной обработкой ошибок."""
        if value is None:
            return None
            
        if PANDAS_AVAILABLE and hasattr(value, 'isna') and value.isna():
            return None
        
        try:
            # Очистка строки
            if isinstance(value, str):
                s = value.strip().replace(' ', '').replace('\xa0', '').replace('\u00a0', '')
                if s.endswith('.'):
                    s = s[:-1]
                s = s.replace(',', '.')
                
                if s in ('', 'nan', 'none', 'null'):
                    return None
            else:
                s = str(value)
            
            num = abs(float(s))  # Делаем число положительным
            return int(num) if force_int or (num == int(num) and not force_int) else num
            
        except (ValueError, TypeError, AttributeError):
            return None
    
    def safe_subtract(self, *values) -> Optional[float]:
        """Безопасное вычитание с проверкой на None/NaN."""
        valid_values = []
        for v in values:
            if v is not None:
                if PANDAS_AVAILABLE and not pd.isna(v):
                    valid_values.append(v)
                elif not PANDAS_AVAILABLE:
                    valid_values.append(v)
        
        if not valid_values:
            return None
        
        try:
            result = valid_values[0]
            for v in valid_values[1:]:
                result = result - v
            return abs(result)
        except (TypeError, ValueError):
            return None
    
    def safe_divide(self, numerator: Any, denominator: Any) -> Optional[float]:
        """Безопасное деление с проверкой на None/NaN и деление на ноль."""
        if numerator is None or denominator is None:
            return None
            
        if PANDAS_AVAILABLE:
            if pd.isna(numerator) or pd.isna(denominator):
                return None
        
        try:
            if float(denominator) == 0:
                return None
            return abs(float(numerator) / float(denominator))
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    
    def _read_file_with_encoding(self, file_path: Path) -> Optional[str]:
        """Читает файл с автоопределением кодировки."""
        encodings = ['cp1251', 'utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                logger.debug(f"Файл прочитан с кодировкой {encoding}")
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Ошибка чтения файла с кодировкой {encoding}: {e}")
                continue
        
        logger.error(f"Не удалось прочитать файл {file_path} ни с одной из кодировок")
        return None
    
    def process_html_files(self, html_files: List[str]) -> pd.DataFrame:
        """Обрабатывает список HTML файлов маршрутов с улучшенной обработкой ошибок."""
        if not PANDAS_AVAILABLE:
            logger.error("Pandas недоступен. Невозможно обработать файлы.")
            return pd.DataFrame()
        
        logger.info(f"Начинаем обработку {len(html_files)} HTML файлов маршрутов")
        
        self.processing_stats = ProcessingStats()
        self.processing_stats.total_files = len(html_files)
        all_routes_data = []
        
        # Обрабатываем каждый файл
        for file_path in html_files:
            logger.info(f"Обработка файла: {Path(file_path).name}")
            
            try:
                # Проверяем существование файла
                if not Path(file_path).exists():
                    logger.error(f"Файл не найден: {file_path}")
                    self.processing_stats.processing_errors += 1
                    continue
                
                # Читаем и очищаем файл
                cleaned_content = self._clean_html_file(file_path)
                if not cleaned_content:
                    logger.error(f"Не удалось очистить файл {file_path}")
                    self.processing_stats.processing_errors += 1
                    continue
                
                # Извлекаем маршруты из очищенного содержимого
                file_routes = self._extract_routes_from_html(cleaned_content)
                if file_routes:
                    all_routes_data.extend(file_routes)
                    self.processing_stats.processed_items += len(file_routes)
                else:
                    logger.warning(f"Не найдено маршрутов в файле {file_path}")
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {e}")
                self.processing_stats.processing_errors += 1
                continue
        
        # Создаем DataFrame
        if all_routes_data:
            try:
                self.routes_df = pd.DataFrame(all_routes_data)
                self._validate_and_clean_dataframe()
                self.processing_stats.total_items_found = len(all_routes_data)
                logger.info(f"Обработка завершена. Создано {len(self.routes_df)} записей маршрутов")
            except Exception as e:
                logger.error(f"Ошибка создания DataFrame: {e}")
                self.routes_df = pd.DataFrame()
        else:
            self.routes_df = pd.DataFrame()
            logger.warning("Не найдено данных маршрутов")
        
        return self.routes_df
    
    def _validate_and_clean_dataframe(self):
        """Валидирует и очищает DataFrame."""
        if self.routes_df is None or self.routes_df.empty:
            return
        
        try:
            initial_count = len(self.routes_df)
            
            # Удаляем строки с пустыми обязательными полями
            required_fields = ['Номер маршрута', 'Наименование участка']
            for field in required_fields:
                if field in self.routes_df.columns:
                    self.routes_df = self.routes_df[self.routes_df[field].notna()]
            
            # Конвертируем числовые поля
            numeric_fields = ['ТКМ брутто', 'КМ', 'ПР', 'Расход фактический', 'Расход по норме', 'Факт уд']
            for field in numeric_fields:
                if field in self.routes_df.columns:
                    self.routes_df[field] = pd.to_numeric(self.routes_df[field], errors='coerce')
            
            # Удаляем дубликаты
            before_dedup = len(self.routes_df)
            self.routes_df = self.routes_df.drop_duplicates(
                subset=['Номер маршрута', 'Дата маршрута', 'Наименование участка'],
                keep='first'
            )
            duplicates_removed = before_dedup - len(self.routes_df)
            
            final_count = len(self.routes_df)
            logger.info(f"Валидация DataFrame: {initial_count} -> {final_count} записей, удалено дубликатов: {duplicates_removed}")
            
        except Exception as e:
            logger.error(f"Ошибка валидации DataFrame: {e}")
    
    def _clean_html_file(self, input_file: str) -> Optional[str]:
        """Очищает HTML файл от лишнего кода."""
        logger.debug(f"Очистка HTML файла: {input_file}")
        
        input_path = Path(input_file)
        
        # Читаем файл с автоопределением кодировки
        html_content = self._read_file_with_encoding(input_path)
        if html_content is None:
            return None
        
        # Применяем очистку с помощью предкомпилированных паттернов
        cleaned_content = self._apply_cleaning_patterns(html_content)
        
        logger.debug(f"HTML файл очищен")
        return cleaned_content
    
    def _apply_cleaning_patterns(self, html_content: str) -> str:
        """Применяет паттерны очистки к HTML контенту."""
        try:
            # Удаляем лишние элементы
            for pattern_name, pattern in self.patterns.items():
                if pattern_name in ['date_clean', 'route_num_clean', 'numline_clean']:
                    html_content = pattern.sub('', html_content)
            
            # Исправляем ширину таблиц
            html_content = self.patterns['table_width'].sub('<table width=100%', html_content)
            
            return html_content
        except Exception as e:
            logger.error(f"Ошибка применения паттернов очистки: {e}")
            return html_content
    
    def _extract_routes_from_html(self, html_content: str) -> List[RouteData]:
        """Извлекает маршруты из HTML контента."""
        routes_data = []
        
        try:
            # Разбиваем HTML на секции маршрутов
            route_sections = self._split_into_route_sections(html_content)
            
            for i, route_section in enumerate(route_sections):
                try:
                    route_data = self._parse_route_section(route_section)
                    if route_data:
                        routes_data.extend(route_data)
                except Exception as e:
                    logger.error(f"Ошибка парсинга секции маршрута {i}: {e}")
                    continue
        except Exception as e:
            logger.error(f"Ошибка извлечения маршрутов: {e}")
        
        return routes_data
    
    def _split_into_route_sections(self, html_content: str) -> List[str]:
        """Разбивает HTML на секции отдельных маршрутов."""
        try:
            # Ищем начала маршрутов
            route_starts = list(self.patterns['route_start'].finditer(html_content))
            
            if not route_starts:
                logger.warning("Не найдены маркеры начала маршрутов")
                return [html_content]  # Возвращаем весь контент
            
            sections = []
            for i, match in enumerate(route_starts):
                start_pos = match.start()
                
                # Определяем конец секции
                if i + 1 < len(route_starts):
                    end_pos = route_starts[i + 1].start()
                else:
                    end_pos = len(html_content)
                
                section = html_content[start_pos:end_pos]
                sections.append(section)
            
            logger.debug(f"Разбито на {len(sections)} секций маршрутов")
            return sections
        except Exception as e:
            logger.error(f"Ошибка разбиения на секции: {e}")
            return [html_content]
    
    def _parse_route_section(self, route_section: str) -> List[RouteData]:
        """Парсит секцию одного маршрута и возвращает данные участков."""
        try:
            # Извлекаем метаданные маршрута
            metadata = self.html_parser.extract_route_header(route_section)
            
            # Извлекаем данные локомотива
            loco_data = self.html_parser.extract_loco_data(route_section)
            
            # Извлекаем данные Ю7
            yu7_data = self.html_parser.extract_yu7_data(route_section)
            
            # Парсим таблицу норм
            norm_sections = self.html_parser.parse_norm_table(route_section)
            
            # Парсим таблицу станций
            station_data = self.html_parser.parse_station_table(route_section)
            
            # Создаем записи для каждого участка
            route_records = []
            
            for section in norm_sections:
                try:
                    record = self._create_route_record(metadata, loco_data, yu7_data, section, station_data)
                    if record:
                        route_records.append(record)
                except Exception as e:
                    logger.debug(f"Ошибка создания записи для участка: {e}")
                    continue
            
            return route_records
        except Exception as e:
            logger.error(f"Ошибка парсинга секции маршрута: {e}")
            return []
    
    def _create_route_record(self, metadata: Optional[RouteMetadata], 
                           loco_data: LocoData, yu7_data: List[Yu7Data], 
                           section: RouteSection, station_data: Dict) -> Optional[RouteData]:
        """Создает запись маршрута для одного участка."""
        if not section.name:
            return None
        
        try:
            record = {
                # Метаданные маршрута
                'Номер маршрута': metadata.number if metadata else None,
                'Дата маршрута': metadata.date if metadata else None,
                'Депо': metadata.depot if metadata else None,
                'Идентификатор': metadata.identifier if metadata else None,
                
                # Данные локомотива
                'Серия локомотива': loco_data.series,
                'Номер локомотива': loco_data.number,
                
                # Данные участка
                'Наименование участка': section.name,
                'Номер нормы': section.norm_number,
                'ТКМ брутто': safe_float_conversion(section.tkm_brutto),
                'КМ': safe_float_conversion(section.km),
                'ПР': safe_float_conversion(section.pr),
                'Расход фактический': safe_float_conversion(section.rashod_fact),
                'Расход по норме': safe_float_conversion(section.rashod_norm),
                'Факт уд': safe_float_conversion(section.ud_norma),
                'Норма на работу': safe_float_conversion(section.norma_rabotu),
                'Норма на одиночное': safe_float_conversion(section.norma_odinochnoe),
            }
            
            # Добавляем данные Ю7 если есть
            if yu7_data:
                yu7 = yu7_data[0]  # Берем первую запись Ю7
                record['НЕТТО'] = safe_int_conversion(yu7.netto)
                record['БРУТТО'] = safe_int_conversion(yu7.brutto)
                record['ОСИ'] = safe_int_conversion(yu7.osi)
            
            # Добавляем данные станций если есть
            if section.name in station_data:
                section_station_data = station_data[section.name]
                for key, value in section_station_data.items():
                    record[key] = safe_float_conversion(value)
            
            return record
        except Exception as e:
            logger.error(f"Ошибка создания записи маршрута: {e}")
            return None
    
    def get_processing_stats(self) -> Dict:
        """Возвращает статистику обработки в формате словаря для совместимости."""
        try:
            base_stats = self.processing_stats.to_dict()
            
            # Добавляем специфичные для маршрутов поля
            base_stats.update({
                'total_routes_found': base_stats.get('total_items_found', 0),
                'routes_processed': base_stats.get('processed_items', 0),
                'routes_skipped': base_stats.get('skipped_items', 0),
                'unique_routes': len(self.routes_df) if self.routes_df is not None else 0,
                'duplicates_total': base_stats.get('total_items_found', 0) - (len(self.routes_df) if self.routes_df is not None else 0),
                'routes_with_equal_rashod': 0,  # Заполняется при анализе
                'output_rows': len(self.routes_df) if self.routes_df is not None else 0,
            })
            
            return base_stats
        except Exception as e:
            logger.error(f"Ошибка получения статистики: {e}")
            return {
                'total_files': 0,
                'total_routes_found': 0,
                'routes_processed': 0,
                'processing_errors': 0
            }
    
    def export_to_excel(self, df: pd.DataFrame, output_file: str) -> bool:
        """Экспортирует DataFrame в Excel файл с форматированием."""
        if not PANDAS_AVAILABLE:
            logger.error("Pandas недоступен для экспорта")
            return False
        
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Базовый экспорт
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Маршруты', index=False)
                
                # Получаем workbook и worksheet для форматирования
                try:
                    from openpyxl.styles import PatternFill
                    
                    workbook = writer.book
                    worksheet = writer.sheets['Маршруты']
                    
                    # Красная заливка для строк с проблемами
                    red_fill = PatternFill(start_color='FFCCCC', end_color='FFCCCC', fill_type='solid')
                    
                    # Применяем красное форматирование если есть соответствующие колонки
                    if 'USE_RED_COLOR' in df.columns:
                        for idx, use_red in enumerate(df['USE_RED_COLOR'], start=2):  # +2 для заголовка и 1-индексации
                            if use_red:
                                for col in range(1, len(df.columns) + 1):
                                    worksheet.cell(row=idx, column=col).fill = red_fill
                
                except ImportError:
                    logger.warning("openpyxl недоступен для форматирования, экспортируем без форматирования")
                except Exception as e:
                    logger.warning(f"Ошибка форматирования Excel: {e}")
            
            logger.info(f"Данные экспортированы в {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта в Excel: {e}")
            return False
    
    def validate_route_data(self, route_data: RouteData) -> bool:
        """Валидирует данные маршрута."""
        try:
            # Проверяем обязательные поля
            required_fields = ['Номер маршрута', 'Дата маршрута', 'Наименование участка']
            for field in required_fields:
                if field not in route_data or not route_data[field]:
                    return False
            
            # Проверяем числовые поля на корректность
            numeric_fields = ['ТКМ брутто', 'КМ', 'Расход фактический']
            for field in numeric_fields:
                value = route_data.get(field)
                if value is not None:
                    try:
                        num_val = float(value)
                        if num_val < 0:  # Отрицательные значения недопустимы
                            return False
                    except (ValueError, TypeError):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации маршрута: {e}")
            return False
    
    def get_routes_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по обработанным маршрутам."""
        try:
            if self.routes_df is None or self.routes_df.empty:
                return {'total_routes': 0, 'sections': [], 'processing_stats': self.get_processing_stats()}
            
            summary = {
                'total_routes': len(self.routes_df),
                'unique_routes': self.routes_df['Номер маршрута'].nunique() if 'Номер маршрута' in self.routes_df.columns else 0,
                'sections': sorted(self.routes_df['Наименование участка'].dropna().unique().tolist()) if 'Наименование участка' in self.routes_df.columns else [],
                'processing_stats': self.get_processing_stats()
            }
            
            # Добавляем диапазон дат если есть
            if 'Дата маршрута' in self.routes_df.columns:
                dates = self.routes_df['Дата маршрута'].dropna()
                if not dates.empty:
                    summary['date_range'] = {
                        'from': dates.min(),
                        'to': dates.max()
                    }
            
            return summary
        except Exception as e:
            logger.error(f"Ошибка получения сводки маршрутов: {e}")
            return {'total_routes': 0, 'sections': [], 'processing_stats': self.get_processing_stats()}
    
    def filter_routes_by_criteria(self, df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Фильтрует маршруты по заданным критериям."""
        if not PANDAS_AVAILABLE or df is None or df.empty:
            return df if df is not None else pd.DataFrame()
        
        try:
            filtered_df = df.copy()
            
            for field, value in criteria.items():
                if field in filtered_df.columns and value is not None:
                    if isinstance(value, (list, tuple)):
                        filtered_df = filtered_df[filtered_df[field].isin(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[field] == value]
            
            return filtered_df
        except Exception as e:
            logger.error(f"Ошибка фильтрации маршрутов: {e}")
            return df
    
    def clear_processed_data(self):
        """Очищает обработанные данные для экономии памяти."""
        try:
            self.processed_routes.clear()
            self.routes_df = None
            self.processing_stats = ProcessingStats()
            logger.info("Обработанные данные маршрутов очищены")
        except Exception as e:
            logger.error(f"Ошибка очистки данных: {e}")
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """Возвращает отчет о качестве данных."""
        if self.routes_df is None or self.routes_df.empty:
            return {'total_rows': 0, 'issues': []}
        
        try:
            report = {
                'total_rows': len(self.routes_df),
                'issues': [],
                'completeness': {},
                'validity': {}
            }
            
            # Проверяем полноту данных
            for column in self.routes_df.columns:
                null_count = self.routes_df[column].isna().sum()
                null_percent = (null_count / len(self.routes_df)) * 100
                report['completeness'][column] = {
                    'null_count': int(null_count),
                    'null_percent': round(null_percent, 2)
                }
                
                if null_percent > 50:
                    report['issues'].append(f"Колонка '{column}' имеет {null_percent:.1f}% пустых значений")
            
            # Проверяем валидность числовых полей
            numeric_fields = ['ТКМ брутто', 'КМ', 'Расход фактический']
            for field in numeric_fields:
                if field in self.routes_df.columns:
                    negative_count = (self.routes_df[field] < 0).sum()
                    if negative_count > 0:
                        report['issues'].append(f"Колонка '{field}' содержит {negative_count} отрицательных значений")
                    
                    report['validity'][field] = {
                        'negative_count': int(negative_count),
                        'zero_count': int((self.routes_df[field] == 0).sum())
                    }
            
            return report
        except Exception as e:
            logger.error(f"Ошибка создания отчета о качестве данных: {e}")
            return {'total_rows': 0, 'issues': ['Ошибка создания отчета'], 'completeness': {}, 'validity': {}}