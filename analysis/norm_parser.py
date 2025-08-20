#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный быстрый HTML парсер для норм с улучшенной отладкой.
Исправлены проблемы с парсингом таблиц норм.
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Optional, Tuple

# Попробуем импортировать selectolax, если недоступно - используем BeautifulSoup
try:
    from selectolax.parser import HTMLParser
    USE_SELECTOLAX = True
except ImportError:
    from bs4 import BeautifulSoup
    USE_SELECTOLAX = False

from .data_models import (
    NormData, NormType, NormPoint, BaseNormData, 
    TableSection
)

logger = logging.getLogger(__name__)

# Типы для Python 3.12
type HTMLContent = str
type NormDict = Dict[str, NormData]

class FastNormParser:
    """Исправленный быстрый парсер HTML норм с улучшенной отладкой."""
    
    def __init__(self):
        # Предкомпилированные паттерны для производительности
        self._compile_patterns()
        
        # Мапинг типов норм
        self.norm_type_mapping = {
            'нагрузке на ось': NormType.AXLE_LOAD,
            'весу поезда': NormType.TRAIN_WEIGHT
        }
        
        if not USE_SELECTOLAX:
            logger.warning("selectolax не установлен, используем BeautifulSoup (медленнее)")
    
    def _compile_patterns(self):
        """Предкомпилирует regex паттерны для лучшей производительности."""
        self.patterns = {
            'norm_id': re.compile(r'id[=:](\d+)', re.IGNORECASE),
            'norm_id_alt': re.compile(r'(\d+)', re.IGNORECASE),
            'section1': re.compile(
                r'(<font class=rcp12><center><b>Удельные нормы электроэнергии и топлива по нагрузке на ось</b></center></font>.*?</table>.*?</table>)',
                re.DOTALL | re.IGNORECASE
            ),
            'section2': re.compile(
                r'(<font class=rcp12><center><b>Удельные нормы электроэнергии и топлива по весу поезда</b></center></font>.*?</table>.*?</table>)',
                re.DOTALL | re.IGNORECASE
            ),
            'number_clean': re.compile(r'[^\d.,+-]'),
            'whitespace': re.compile(r'\s+')
        }
    
    def extract_table_sections(self, html_content: str) -> List[TableSection]:
        """Извлекает секции таблиц из HTML с использованием быстрых regex."""
        sections = []
        
        # Поиск секции по нагрузке на ось
        match1 = self.patterns['section1'].search(html_content)
        if match1:
            sections.append(TableSection(
                title="Удельные нормы электроэнергии и топлива по нагрузке на ось",
                content=match1.group(1),
                norm_type=NormType.AXLE_LOAD
            ))
            logger.debug("✓ Найдена таблица 'по нагрузке на ось'")
        
        # Поиск секции по весу поезда
        match2 = self.patterns['section2'].search(html_content)
        if match2:
            sections.append(TableSection(
                title="Удельные нормы электроэнергии и топлива по весу поезда",
                content=match2.group(1),
                norm_type=NormType.TRAIN_WEIGHT
            ))
            logger.debug("✓ Найдена таблица 'по весу поезда'")
        
        if not sections:
            logger.warning("Искомые таблицы не найдены в HTML")
        
        return sections
    
    def parse_section_to_norms(self, section: TableSection) -> NormDict:
        """Парсит секцию таблицы в нормы с использованием selectolax или BeautifulSoup."""
        if USE_SELECTOLAX:
            return self._parse_section_selectolax(section)
        else:
            return self._parse_section_bs4(section)
    
    def _parse_section_selectolax(self, section: TableSection) -> NormDict:
        """Парсит секцию с использованием selectolax."""
        parser = HTMLParser(section.content)
        norms = {}
        
        logger.debug(f"Начинаем парсинг секции '{section.title}'")
        
        # Ищем все таблицы в секции
        tables = parser.css('table')
        logger.debug(f"Найдено таблиц: {len(tables)}")
        
        for table_idx, table in enumerate(tables):
            logger.debug(f"Обрабатываем таблицу {table_idx + 1}")
            
            # Получаем все строки таблицы
            rows = table.css('tr')
            logger.debug(f"Найдено строк в таблице: {len(rows)}")
            
            if len(rows) < 2:  # Минимум заголовок + 1 строка данных
                logger.debug("Недостаточно строк в таблице, пропускаем")
                continue
            
            # Первая строка - заголовок
            header_row = rows[0]
            headers = [self._clean_text(cell.text()) for cell in header_row.css('th, td')]
            logger.debug(f"Заголовки: {headers[:10]}...")  # Показываем первые 10
            
            if len(headers) < 10:  # Минимальное количество колонок для нормы
                logger.debug("Недостаточно колонок в заголовке, пропускаем таблицу")
                continue
            
            # Определяем диапазон числовых колонок
            numeric_start, numeric_end = self._find_numeric_range(headers)
            logger.debug(f"Числовые колонки: {numeric_start}-{numeric_end}")
            
            # Парсим строки данных (пропускаем заголовок)
            data_rows = rows[1:]
            logger.debug(f"Строк данных для обработки: {len(data_rows)}")
            
            for row_idx, row in enumerate(data_rows):
                try:
                    norm_data = self._parse_norm_row_selectolax(
                        row, headers, section.norm_type, numeric_start, numeric_end, row_idx
                    )
                    
                    if norm_data:
                        norms[norm_data.norm_id] = norm_data
                        logger.debug(f"✓ Обработана норма {norm_data.norm_id}")
                    else:
                        logger.debug(f"Строка {row_idx + 1}: норма не извлечена")
                        
                except Exception as e:
                    logger.warning(f"Ошибка парсинга строки {row_idx + 1}: {e}")
                    continue
        
        logger.info(f"Извлечено норм из секции '{section.title}': {len(norms)}")
        return norms
    
    def _parse_section_bs4(self, section: TableSection) -> NormDict:
        """Парсит секцию с использованием BeautifulSoup."""
        soup = BeautifulSoup(section.content, 'html.parser')
        norms = {}
        
        logger.debug(f"Начинаем парсинг секции '{section.title}' с BeautifulSoup")
        
        tables = soup.find_all('table')
        logger.debug(f"Найдено таблиц: {len(tables)}")
        
        for table_idx, table in enumerate(tables):
            rows = table.find_all('tr')
            logger.debug(f"Таблица {table_idx + 1}: найдено строк {len(rows)}")
            
            if len(rows) < 2:
                continue
            
            # Заголовки
            header_row = rows[0]
            headers = [self._clean_text(cell.get_text()) for cell in header_row.find_all(['th', 'td'])]
            logger.debug(f"Заголовки: {headers[:10]}...")
            
            if len(headers) < 10:
                continue
            
            numeric_start, numeric_end = self._find_numeric_range(headers)
            
            # Строки данных
            for row_idx, row in enumerate(rows[1:]):
                try:
                    norm_data = self._parse_norm_row_bs4(
                        row, headers, section.norm_type, numeric_start, numeric_end, row_idx
                    )
                    
                    if norm_data:
                        norms[norm_data.norm_id] = norm_data
                        logger.debug(f"✓ Обработана норма {norm_data.norm_id}")
                        
                except Exception as e:
                    logger.warning(f"Ошибка парсинга строки {row_idx + 1}: {e}")
                    continue
        
        logger.info(f"Извлечено норм из секции '{section.title}': {len(norms)}")
        return norms
    
    def _find_numeric_range(self, headers: List[str]) -> Tuple[int, int]:
        """Определяет диапазон числовых колонок с улучшенной логикой."""
        # Ищем начало числовых колонок
        numeric_start = 9  # Значение по умолчанию
        numeric_end = len(headers) - 2
        
        # Ищем колонку "Призн. алг. нормир." или аналогичную
        for i, header in enumerate(headers):
            header_lower = header.lower()
            if ('алг' in header_lower and 'нормир' in header_lower) or \
               ('признак' in header_lower and 'алг' in header_lower):
                numeric_start = i + 1
                break
        
        # Ищем конец числовых колонок (до дат)
        for i in range(len(headers) - 1, -1, -1):
            header_lower = headers[i].lower()
            if any(word in header_lower for word in ['дата', 'date', 'окончан', 'начал']):
                numeric_end = i
            else:
                break
        
        # Проверяем разумность диапазона
        if numeric_start >= numeric_end:
            # Если автоматическое определение не сработало, используем эвристику
            logger.warning("Не удалось автоматически определить диапазон числовых колонок")
            numeric_start = min(9, len(headers) // 2)
            numeric_end = max(numeric_start + 5, len(headers) - 2)
        
        logger.debug(f"Определен диапазон числовых колонок: {numeric_start}-{numeric_end}")
        return numeric_start, numeric_end
    
    def _parse_norm_row_selectolax(self, row, headers: List[str], norm_type: NormType, 
                                  numeric_start: int, numeric_end: int, row_idx: int = 0) -> Optional[NormData]:
        """Парсит строку нормы с использованием selectolax с улучшенной отладкой."""
        cells = row.css('td, th')
        
        logger.debug(f"Строка {row_idx + 1}: найдено ячеек {len(cells)}")
        
        if len(cells) < 5:  # Слишком мало ячеек
            logger.debug(f"Строка {row_idx + 1}: слишком мало ячеек ({len(cells)})")
            return None
        
        # Извлекаем все данные из ячеек
        raw_data = []
        for i, cell in enumerate(cells):
            cell_text = self._clean_text(cell.text())
            raw_data.append(cell_text)
            if i < 10:  # Показываем первые 10 ячеек для отладки
                logger.debug(f"  Ячейка {i}: '{cell_text}'")
        
        # Ищем ID нормы в первых нескольких ячейках
        norm_id = None
        for i in range(min(3, len(cells))):
            cell = cells[i]
            norm_id = self._extract_norm_id_selectolax(cell, i)
            if norm_id:
                logger.debug(f"Найден ID нормы: {norm_id} в ячейке {i}")
                break
        
        if not norm_id:
            # Пробуем извлечь ID из текста первых ячеек
            for i in range(min(3, len(raw_data))):
                text = raw_data[i]
                if text and text.isdigit():
                    norm_id = text
                    logger.debug(f"ID нормы из текста: {norm_id}")
                    break
        
        if not norm_id:
            logger.debug(f"Строка {row_idx + 1}: ID нормы не найден")
            return None
        
        # Парсим числовые данные (точки нормы)
        points = self._extract_norm_points(raw_data, headers, numeric_start, numeric_end)
        logger.debug(f"Строка {row_idx + 1}: извлечено точек {len(points)}")
        
        if len(points) < 2:
            logger.debug(f"Строка {row_idx + 1}: недостаточно точек для нормы ({len(points)})")
            return None
        
        # Создаем базовые данные
        base_data = self._create_base_data(raw_data)
        
        # Создаем объект нормы
        norm_data = NormData(
            norm_id=norm_id,
            norm_type=norm_type,
            description=f"Норма №{norm_id} ({norm_type.value})",
            points=points,
            base_data=base_data
        )
        
        logger.debug(f"✓ Создана норма {norm_id} с {len(points)} точками")
        return norm_data
    
    def _parse_norm_row_bs4(self, row, headers: List[str], norm_type: NormType, 
                           numeric_start: int, numeric_end: int, row_idx: int = 0) -> Optional[NormData]:
        """Парсит строку нормы с использованием BeautifulSoup."""
        cells = row.find_all(['td', 'th'])
        
        if len(cells) < 5:
            return None
        
        raw_data = [self._clean_text(cell.get_text()) for cell in cells]
        
        # Ищем ID нормы
        norm_id = None
        for i in range(min(3, len(cells))):
            norm_id = self._extract_norm_id_bs4(cells[i], i)
            if norm_id:
                break
        
        if not norm_id:
            for i in range(min(3, len(raw_data))):
                text = raw_data[i]
                if text and text.isdigit():
                    norm_id = text
                    break
        
        if not norm_id:
            return None
        
        points = self._extract_norm_points(raw_data, headers, numeric_start, numeric_end)
        
        if len(points) < 2:
            return None
        
        base_data = self._create_base_data(raw_data)
        
        norm_data = NormData(
            norm_id=norm_id,
            norm_type=norm_type,
            description=f"Норма №{norm_id} ({norm_type.value})",
            points=points,
            base_data=base_data
        )
        
        return norm_data
    
    def _extract_norm_id_selectolax(self, cell, cell_idx: int = 0) -> Optional[str]:
        """Улучшенное извлечение ID нормы из ячейки (selectolax версия)."""
        # Ищем ссылку в ячейке
        link = cell.css_first('a')
        if link:
            href = link.attributes.get('href', '')
            if href:
                # Ищем id= или id:
                match = self.patterns['norm_id'].search(href)
                if match:
                    return match.group(1)
                
                # Ищем любые числа в href
                match = self.patterns['norm_id_alt'].search(href)
                if match:
                    return match.group(1)
        
        # Ищем в тексте ячейки
        cell_text = self._clean_text(cell.text())
        if cell_text:
            # Проверяем, является ли текст числом
            if cell_text.isdigit() and len(cell_text) > 0:
                return cell_text
            
            # Ищем числа в тексте
            match = self.patterns['norm_id_alt'].search(cell_text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_norm_id_bs4(self, cell, cell_idx: int = 0) -> Optional[str]:
        """Улучшенное извлечение ID нормы из ячейки (BeautifulSoup версия)."""
        link = cell.find('a')
        if link:
            href = link.get('href', '')
            if href:
                match = self.patterns['norm_id'].search(href)
                if match:
                    return match.group(1)
                
                match = self.patterns['norm_id_alt'].search(href)
                if match:
                    return match.group(1)
        
        cell_text = self._clean_text(cell.get_text())
        if cell_text:
            if cell_text.isdigit() and len(cell_text) > 0:
                return cell_text
            
            match = self.patterns['norm_id_alt'].search(cell_text)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_norm_points(self, raw_data: List[str], headers: List[str], 
                           numeric_start: int, numeric_end: int) -> List[NormPoint]:
        """Извлекает точки нормы из числовых данных с улучшенной отладкой."""
        points = []
        
        logger.debug(f"Извлечение точек из диапазона {numeric_start}-{numeric_end}")
        
        for i in range(numeric_start, min(numeric_end, len(raw_data), len(headers))):
            if i >= len(raw_data) or not raw_data[i].strip():
                continue
            
            try:
                # Преобразуем заголовок в нагрузку
                header_value = self._to_number(headers[i])
                consumption_value = self._to_number(raw_data[i])
                
                logger.debug(f"  Колонка {i}: заголовок='{headers[i]}' -> {header_value}, "
                           f"значение='{raw_data[i]}' -> {consumption_value}")
                
                if header_value is not None and consumption_value is not None:
                    if header_value > 0 and consumption_value > 0:  # Проверяем положительные значения
                        points.append(NormPoint(
                            load=float(header_value),
                            consumption=float(consumption_value)
                        ))
                        logger.debug(f"    ✓ Добавлена точка: ({header_value}, {consumption_value})")
                    else:
                        logger.debug(f"    ✗ Пропущена точка: некорректные значения")
                else:
                    logger.debug(f"    ✗ Пропущена точка: не удалось преобразовать в числа")
                    
            except (ValueError, IndexError) as e:
                logger.debug(f"    ✗ Ошибка обработки колонки {i}: {e}")
                continue
        
        # Сортируем точки по нагрузке
        points.sort(key=lambda p: p.load)
        logger.debug(f"Итого извлечено и отсортировано точек: {len(points)}")
        
        return points
    
    def _create_base_data(self, raw_data: List[str]) -> BaseNormData:
        """Создает базовые данные нормы."""
        return BaseNormData(
            priznok_sost_tyag=self._to_number(raw_data[1]) if len(raw_data) > 1 else None,
            priznok_rek=self._to_number(raw_data[2]) if len(raw_data) > 2 else None,
            vid_dvizheniya=raw_data[3] if len(raw_data) > 3 else '',
            simvol_rod_raboty=self._to_number(raw_data[4]) if len(raw_data) > 4 else None,
            rps=self._to_number(raw_data[5]) if len(raw_data) > 5 else None,
            identif_gruppy=self._to_number(raw_data[6]) if len(raw_data) > 6 else None,
            priznok_sost=self._to_number(raw_data[7]) if len(raw_data) > 7 else None,
            priznok_alg=self._to_number(raw_data[8]) if len(raw_data) > 8 else None,
            date_start=raw_data[-2] if len(raw_data) >= 2 and raw_data[-2] else '',
            date_end=raw_data[-1] if len(raw_data) >= 1 and raw_data[-1] else ''
        )
    
    def _clean_text(self, text: str) -> str:
        """Быстрая очистка текста."""
        if not text:
            return ""
        
        # Заменяем неразрывные пробелы и множественные пробелы
        text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
        text = self.patterns['whitespace'].sub(' ', text)
        return text.strip()
    
    def _to_number(self, text: str) -> Optional[float]:
        """Быстрое преобразование текста в число с улучшенной отладкой."""
        if not text or not text.strip():
            return None
        
        # Очищаем текст
        cleaned = self._clean_text(text)
        if not cleaned:
            return None
        
        try:
            # Заменяем запятую на точку для десятичных чисел
            if ',' in cleaned:
                cleaned = cleaned.replace(',', '.')
            
            # Удаляем все нечисловые символы кроме точки и знаков
            cleaned = re.sub(r'[^\d.+-]', '', cleaned)
            
            if not cleaned or cleaned in ['+', '-', '.']:
                return None
            
            return float(cleaned)
            
        except (ValueError, TypeError):
            return None
    
    def create_cleaned_html(self, sections: List[TableSection]) -> str:
        """Создает очищенный HTML из найденных секций."""
        html_parts = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            '    <meta charset="utf-8">',
            '    <title>Удельные нормы электроэнергии и топлива</title>',
            '    <style>',
            '        body { font-family: Arial, sans-serif; margin: 20px; }',
            '        .rcp12 { font-size: 12px; }',
            '        .filter_key { font-weight: bold; }',
            '        .filter_value { color: blue; }',
            '        .tr_head { background-color: #e0e0e0; }',
            '        .thc { border: 1px solid #000; padding: 5px; text-align: center; }',
            '        .tdc_str1, .tdc_str2 { border: 1px solid #000; padding: 3px; text-align: center; }',
            '        .tdc_str1 { background-color: #f9f9f9; }',
            '        .tdc_str2 { background-color: #ffffff; }',
            '        table { border-collapse: collapse; margin: 20px auto; }',
            '        .link { color: blue; text-decoration: underline; }',
            '    </style>',
            '</head>',
            '<body>'
        ]
        
        # Добавляем найденные секции
        for section in sections:
            html_parts.append(section.content)
            html_parts.append('<br><br>')
        
        # Если секций нет, добавляем сообщение
        if not sections:
            html_parts.append('<h1>Искомые таблицы не найдены в файле</h1>')
        
        html_parts.extend([
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(html_parts)