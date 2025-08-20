#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный процессор для обработки HTML файлов маршрутов.
Использует современные возможности Python 3.12 для максимальной производительности.
"""

from __future__ import annotations

import os
import re
import logging
import pandas as pd
import tempfile
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from pathlib import Path

from .data_models import (
    RouteMetadata, LocoData, Yu7Data, RouteSection, 
    ProcessingStats
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
        """Преобразует значение в число, делая все числа положительными."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        
        s = str(value).strip().replace(' ', '').replace('\xa0', '').replace('\u00a0', '')
        if s.endswith('.'):
            s = s[:-1]
        s = s.replace(',', '.')
        
        if s in ('', 'nan', 'none'):
            return None
        
        try:
            num = abs(float(s))  # Делаем число положительным
            return int(num) if force_int or num == int(num) else num
        except:
            return None
    
    def safe_subtract(self, *values) -> Optional[float]:
        """Безопасное вычитание с проверкой на None/NaN, возвращает абсолютное значение."""
        valid_values = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
        
        if not valid_values:
            return None
        
        result = valid_values[0]
        for v in valid_values[1:]:
            result = result - v
        
        return abs(result)
    
    def safe_divide(self, numerator: Any, denominator: Any) -> Optional[float]:
        """Безопасное деление с проверкой на None/NaN и деление на ноль."""
        if numerator is None or denominator is None:
            return None
        if isinstance(numerator, float) and pd.isna(numerator):
            return None
        if isinstance(denominator, float) and pd.isna(denominator):
            return None
        if denominator == 0:
            return None
        
        return abs(numerator / denominator)
    
    def _read_file_with_encoding(self, file_path: Path) -> Optional[str]:
        """Читает файл с автоопределением кодировки."""
        encodings = ['cp1251', 'utf-8', 'latin-1']
        
        for encoding in encodings:
            try:
                content = file_path.read_text(encoding=encoding)
                logger.debug(f"Файл прочитан с кодировкой {encoding}")
                return content
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Не удалось прочитать файл {file_path} ни с одной из кодировок")
        return None
    
    def process_html_files(self, html_files: List[str]) -> pd.DataFrame:
        """Обрабатывает список HTML файлов маршрутов."""
        logger.info(f"Начинаем обработку {len(html_files)} HTML файлов маршрутов")
        
        self.processing_stats.total_files = len(html_files)
        all_routes_data = []
        
        # Обрабатываем каждый файл
        for file_path in html_files:
            logger.info(f"Обработка файла: {Path(file_path).name}")
            
            try:
                # Читаем и очищаем файл
                cleaned_content = self._clean_html_file(file_path)
                if not cleaned_content:
                    logger.error(f"Не удалось очистить файл {file_path}")
                    self.processing_stats.processing_errors += 1
                    continue
                
                # Извлекаем маршруты из очищенного содержимого
                file_routes = self._extract_routes_from_html(cleaned_content)
                all_routes_data.extend(file_routes)
                
                self.processing_stats.processed_items += len(file_routes)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {e}")
                self.processing_stats.processing_errors += 1
                continue
        
        # Создаем DataFrame
        if all_routes_data:
            self.routes_df = pd.DataFrame(all_routes_data)
            self.processing_stats.total_items_found = len(all_routes_data)
            logger.info(f"Обработка завершена. Создано {len(all_routes_data)} записей маршрутов")
        else:
            self.routes_df = pd.DataFrame()
            logger.warning("Не найдено данных маршрутов")
        
        return self.routes_df
    
    def _clean_html_file(self, input_file: str) -> Optional[str]:
        """Очищает HTML файл от лишнего кода."""
        logger.debug(f"Очистка HTML файла: {input_file}")
        
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Файл {input_file} не найден!")
            return None
        
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
        # Удаляем лишние элементы
        for pattern_name, pattern in self.patterns.items():
            if pattern_name in ['date_clean', 'route_num_clean', 'numline_clean']:
                html_content = pattern.sub('', html_content)
        
        # Исправляем ширину таблиц
        html_content = self.patterns['table_width'].sub('<table width=100%', html_content)
        
        return html_content
    
    def _extract_routes_from_html(self, html_content: str) -> List[RouteData]:
        """Извлекает маршруты из HTML контента."""
        routes_data = []
        
        # Разбиваем HTML на секции маршрутов
        route_sections = self._split_into_route_sections(html_content)
        
        for route_section in route_sections:
            try:
                route_data = self._parse_route_section(route_section)
                if route_data:
                    routes_data.extend(route_data)
            except Exception as e:
                logger.error(f"Ошибка парсинга секции маршрута: {e}")
                continue
        
        return routes_data
    
    def _split_into_route_sections(self, html_content: str) -> List[str]:
        """Разбивает HTML на секции отдельных маршрутов."""
        # Ищем начала маршрутов
        route_starts = list(self.patterns['route_start'].finditer(html_content))
        
        if not route_starts:
            return [html_content]  # Если паттерн не найден, возвращаем весь контент
        
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
        
        return sections
    
    def _parse_route_section(self, route_section: str) -> List[RouteData]:
        """Парсит секцию одного маршрута и возвращает данные участков."""
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
            record = self._create_route_record(metadata, loco_data, yu7_data, section, station_data)
            if record:
                route_records.append(record)
        
        return route_records
    
    def _create_route_record(self, metadata: Optional[RouteMetadata], 
                           loco_data: LocoData, yu7_data: List[Yu7Data], 
                           section: RouteSection, station_data: Dict) -> Optional[RouteData]:
        """Создает запись маршрута для одного участка."""
        if not section.name:
            return None
        
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
            'ТКМ брутто': section.tkm_brutto,
            'КМ': section.km,
            'ПР': section.pr,
            'Расход фактический': section.rashod_fact,
            'Расход по норме': section.rashod_norm,
            'Факт уд': section.ud_norma,
            'Норма на работу': section.norma_rabotu,
            'Норма на одиночное': section.norma_odinochnoe,
        }
        
        # Добавляем данные Ю7 если есть
        if yu7_data:
            yu7 = yu7_data[0]  # Берем первую запись Ю7
            record['НЕТТО'] = yu7.netto
            record['БРУТТО'] = yu7.brutto
            record['ОСИ'] = yu7.osi
        
        # Добавляем данные станций если есть
        section_station_data = station_data.get(section.name, {})
        record.update(section_station_data)
        
        return record
    
    def get_processing_stats(self) -> Dict:
        """Возвращает статистику обработки в формате словаря для совместимости."""
        return self.processing_stats.to_dict()
    
    def export_to_excel(self, df: pd.DataFrame, output_file: str) -> bool:
        """Экспортирует DataFrame в Excel файл."""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Маршруты', index=False)
            
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
            
            # Проверяем числовые поля
            numeric_fields = ['ТКМ брутто', 'КМ', 'Расход фактический']
            for field in numeric_fields:
                value = route_data.get(field)
                if value is not None:
                    try:
                        float(value)
                    except (ValueError, TypeError):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации маршрута: {e}")
            return False
    
    def get_routes_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по обработанным маршрутам."""
        if self.routes_df is None or self.routes_df.empty:
            return {'total_routes': 0, 'sections': [], 'processing_stats': self.get_processing_stats()}
        
        summary = {
            'total_routes': len(self.routes_df),
            'unique_routes': self.routes_df['Номер маршрута'].nunique(),
            'sections': sorted(self.routes_df['Наименование участка'].dropna().unique().tolist()),
            'date_range': {
                'from': self.routes_df['Дата маршрута'].min(),
                'to': self.routes_df['Дата маршрута'].max()
            },
            'processing_stats': self.get_processing_stats()
        }
        
        return summary
    
    def filter_routes_by_criteria(self, df: pd.DataFrame, criteria: Dict[str, Any]) -> pd.DataFrame:
        """Фильтрует маршруты по заданным критериям."""
        filtered_df = df.copy()
        
        for field, value in criteria.items():
            if field in filtered_df.columns and value is not None:
                if isinstance(value, (list, tuple)):
                    filtered_df = filtered_df[filtered_df[field].isin(value)]
                else:
                    filtered_df = filtered_df[filtered_df[field] == value]
        
        return filtered_df
    
    def clear_processed_data(self):
        """Очищает обработанные данные для экономии памяти."""
        self.processed_routes.clear()
        self.routes_df = None
        logger.info("Обработанные данные маршрутов очищены")