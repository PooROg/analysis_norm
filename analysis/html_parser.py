#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный быстрый HTML парсер без циклических зависимостей.
Использует selectolax для высокой производительности (28x быстрее BeautifulSoup).
"""

from __future__ import annotations

import re
import logging
from typing import List, Dict, Tuple, Optional
from urllib.parse import urlparse, parse_qs

# Попробуем импортировать selectolax, если недоступно - используем BeautifulSoup
try:
    from selectolax.parser import HTMLParser
    USE_SELECTOLAX = True
except ImportError:
    from bs4 import BeautifulSoup
    USE_SELECTOLAX = False
    logger = logging.getLogger(__name__)
    logger.warning("selectolax не установлен, используем BeautifulSoup (медленнее)")

from .data_models import RouteMetadata, LocoData, Yu7Data, RouteSection

logger = logging.getLogger(__name__)

# Типы для Python 3.12
type HTMLContent = str
type ParsedData = Dict[str, any]

class FastHTMLParser:
    """Быстрый HTML парсер без циклических зависимостей."""
    
    def __init__(self):
        self.text_cleaners = [
            (re.compile(r'\xa0|\u00a0|&nbsp;'), ' '),
            (re.compile(r'\s+'), ' ')
        ]
    
    # Утилиты для работы с числами
    def try_convert_to_number(self, value, force_int: bool = False) -> Optional[float]:
        """Преобразует значение в число, делая все числа положительными."""
        if value is None or (hasattr(value, 'isna') and value.isna()):
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
    
    def clean_text(self, text: str) -> str:
        """Быстрая очистка текста с использованием предкомпилированных regex."""
        if not text:
            return ""
        
        for pattern, replacement in self.text_cleaners:
            text = pattern.sub(replacement, text)
        return text.strip()
    
    def extract_norm_url_from_href(self, href: str) -> Optional[str]:
        """Извлекает номер нормы из URL гиперссылки."""
        if not href:
            return None
        
        try:
            parsed = urlparse(href)
            params = parse_qs(parsed.query)
            if 'id_ntp_tax' in params:
                return params['id_ntp_tax'][0]
        except Exception as e:
            logger.debug(f"Ошибка парсинга URL {href}: {e}")
        
        # Альтернативный метод
        match = re.search(r'id_ntp_tax=(\d+)', href)
        return match.group(1) if match else None
    
    def extract_route_header(self, html_line: str) -> Optional[RouteMetadata]:
        """Извлекает метаданные маршрута из HTML строки."""
        if USE_SELECTOLAX:
            return self._extract_route_header_selectolax(html_line)
        else:
            return self._extract_route_header_bs4(html_line)
    
    def _extract_route_header_selectolax(self, html_line: str) -> Optional[RouteMetadata]:
        """Использует selectolax для извлечения заголовка."""
        parser = HTMLParser(html_line)
        
        header = parser.css_first('th.thl_common')
        if not header:
            return None
        
        header_text = header.text()
        metadata = {}
        
        # Номер маршрута
        route_spans = parser.css('font.filter_value')
        if route_spans:
            metadata['number'] = self.clean_text(route_spans[0].text())
        else:
            patterns = [r'Маршрут\s*№[:\s]*(\d+)', r'Route\s*№[:\s]*(\d+)']
            for pattern in patterns:
                match = re.search(pattern, header_text)
                if match:
                    metadata['number'] = match.group(1)
                    break
        
        # Дата
        match = re.search(r'(\d{2}\.\d{2}\.\d{4})', header_text)
        metadata['date'] = match.group(1) if match else None
        
        # Депо
        depot_patterns = [r'Депо:\s*([^И]+)', r'Depot:\s*([^A-Z]+)']
        for pattern in depot_patterns:
            match = re.search(pattern, header_text)
            if match:
                metadata['depot'] = match.group(1).strip()
                break
        
        # Идентификатор
        id_patterns = [r'Идентификатор:\s*(\d+)', r'Identifier:\s*(\d+)']
        for pattern in id_patterns:
            match = re.search(pattern, header_text)
            if match:
                metadata['identifier'] = match.group(1)
                break
        
        return RouteMetadata(**metadata)
    
    def _extract_route_header_bs4(self, html_line: str) -> Optional[RouteMetadata]:
        """Использует BeautifulSoup для извлечения заголовка."""
        soup = BeautifulSoup(html_line, 'html.parser')
        
        header = soup.find('th', class_='thl_common')
        if not header:
            return None
        
        header_text = header.get_text()
        metadata = {}
        
        # Аналогичная логика как в selectolax версии
        route_spans = soup.find_all('font', class_='filter_value')
        if route_spans:
            metadata['number'] = self.clean_text(route_spans[0].get_text())
        
        # Остальная логика аналогична
        match = re.search(r'(\d{2}\.\d{2}\.\d{4})', header_text)
        metadata['date'] = match.group(1) if match else None
        
        return RouteMetadata(**metadata)
    
    def extract_loco_data(self, html_content: str) -> LocoData:
        """Извлекает серию и номер локомотива из HTML."""
        if USE_SELECTOLAX:
            return self._extract_loco_data_selectolax(html_content)
        else:
            return self._extract_loco_data_bs4(html_content)
    
    def _extract_loco_data_selectolax(self, html_content: str) -> LocoData:
        """Использует selectolax для извлечения данных локомотива."""
        parser = HTMLParser(html_content)
        
        tu3_fonts = parser.css('font.itog2, font.itog3')
        
        for font in tu3_fonts:
            font_text = font.text().strip()
            if font_text in ['ТУ3', 'TU3'] or 'ТУ3' in font_text or 'TU3' in font_text:
                # Собираем следующие элементы
                data_elements = []
                current = font
                
                while current and len(data_elements) < 10:  # Ограничиваем поиск
                    next_sibling = current.next
                    if next_sibling and next_sibling.tag == 'font':
                        text = self.clean_text(next_sibling.text())
                        if text:
                            data_elements.append(text)
                        current = next_sibling
                    else:
                        break
                
                logger.debug(f"ТУ3 данные: {data_elements}")
                
                if len(data_elements) >= 4:
                    loco_number_raw = data_elements[2]
                    series_raw = data_elements[3]
                    
                    # Обрабатываем серию
                    series_part = series_raw.split(',')[0] if ',' in series_raw else series_raw
                    series_clean = re.sub(r'[^\d]', '', series_part)
                    
                    if len(series_clean) > 1:
                        processed_series = series_clean[:-1]
                    else:
                        processed_series = series_clean
                    
                    # Меняем местами серию и номер
                    return LocoData(series=loco_number_raw, number=processed_series)
        
        logger.debug("ТУ3 не найдено")
        return LocoData()
    
    def _extract_loco_data_bs4(self, html_content: str) -> LocoData:
        """Использует BeautifulSoup для извлечения данных локомотива."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        tu3_fonts = soup.find_all('font', class_=['itog2', 'itog3'])
        
        for font in tu3_fonts:
            font_text = font.get_text().strip()
            if 'ТУ3' in font_text or 'TU3' in font_text:
                # Аналогичная логика сбора данных
                data_elements = []
                current = font
                
                for _ in range(10):  # Ограничиваем поиск
                    next_sibling = current.find_next_sibling('font')
                    if next_sibling:
                        text = self.clean_text(next_sibling.get_text())
                        if text:
                            data_elements.append(text)
                        current = next_sibling
                    else:
                        break
                
                if len(data_elements) >= 4:
                    loco_number_raw = data_elements[2]
                    series_raw = data_elements[3]
                    
                    series_part = series_raw.split(',')[0] if ',' in series_raw else series_raw
                    series_clean = re.sub(r'[^\d]', '', series_part)
                    
                    if len(series_clean) > 1:
                        processed_series = series_clean[:-1]
                    else:
                        processed_series = series_clean
                    
                    return LocoData(series=loco_number_raw, number=processed_series)
        
        return LocoData()
    
    def extract_yu7_data(self, html_content: str) -> List[Yu7Data]:
        """Извлекает данные Ю7 из HTML."""
        if USE_SELECTOLAX:
            return self._extract_yu7_data_selectolax(html_content)
        else:
            return self._extract_yu7_data_bs4(html_content)
    
    def _extract_yu7_data_selectolax(self, html_content: str) -> List[Yu7Data]:
        """Использует selectolax для извлечения данных Ю7."""
        parser = HTMLParser(html_content)
        yu7_data = []
        
        all_fonts = parser.css('font')
        
        for font in all_fonts:
            font_text = font.text().strip()
            if any(pattern in font_text for pattern in ['Ю7', 'YU7', 'Yu7']):
                # Собираем данные аналогично ТУ3
                data_elements = []
                current = font
                
                while current and len(data_elements) < 20:
                    next_sibling = current.next
                    if next_sibling and next_sibling.tag == 'font':
                        text = self.clean_text(next_sibling.text())
                        if text:
                            data_elements.append(text)
                        current = next_sibling
                    else:
                        break
                
                logger.debug(f"Ю7 данные найдены: {data_elements[:15]}...")
                
                if len(data_elements) >= 9:
                    try:
                        netto = int(data_elements[7].strip())
                        brutto = int(data_elements[8].strip())
                        
                        # Ищем ОСИ
                        osi = None
                        for i in range(9, len(data_elements)):
                            element = data_elements[i]
                            if any(osi_pattern in element for osi_pattern in ['ОСИ', 'OSI']):
                                osi_match = re.search(r'(?:ОСИ|OSI)(\d+)', element)
                                if osi_match:
                                    osi = int(osi_match.group(1))
                                    break
                        
                        if osi is not None and brutto > 0:
                            yu7_data.append(Yu7Data(netto=netto, brutto=brutto, osi=osi))
                            logger.debug(f"Ю7 успешно обработано: НЕТТО={netto}, БРУТТО={brutto}, ОСИ={osi}")
                    
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Ошибка парсинга Ю7: {e}")
                        continue
        
        logger.debug(f"Всего найдено строк Ю7: {len(yu7_data)}")
        return yu7_data
    
    def _extract_yu7_data_bs4(self, html_content: str) -> List[Yu7Data]:
        """Использует BeautifulSoup для извлечения данных Ю7."""
        soup = BeautifulSoup(html_content, 'html.parser')
        yu7_data = []
        
        all_fonts = soup.find_all('font')
        
        for font in all_fonts:
            font_text = font.get_text().strip()
            if any(pattern in font_text for pattern in ['Ю7', 'YU7', 'Yu7']):
                # Аналогичная логика как в selectolax версии
                data_elements = []
                current = font
                
                for _ in range(20):
                    next_sibling = current.find_next_sibling('font')
                    if next_sibling:
                        text = self.clean_text(next_sibling.get_text())
                        if text:
                            data_elements.append(text)
                        current = next_sibling
                    else:
                        break
                
                if len(data_elements) >= 9:
                    try:
                        netto = int(data_elements[7].strip())
                        brutto = int(data_elements[8].strip())
                        
                        osi = None
                        for i in range(9, len(data_elements)):
                            element = data_elements[i]
                            if any(osi_pattern in element for osi_pattern in ['ОСИ', 'OSI']):
                                osi_match = re.search(r'(?:ОСИ|OSI)(\d+)', element)
                                if osi_match:
                                    osi = int(osi_match.group(1))
                                    break
                        
                        if osi is not None and brutto > 0:
                            yu7_data.append(Yu7Data(netto=netto, brutto=brutto, osi=osi))
                    
                    except (ValueError, IndexError):
                        continue
        
        return yu7_data
    
    def parse_norm_table(self, html_content: str) -> List[RouteSection]:
        """Парсит таблицу с нормами."""
        if USE_SELECTOLAX:
            return self._parse_norm_table_selectolax(html_content)
        else:
            return self._parse_norm_table_bs4(html_content)
    
    def _parse_norm_table_selectolax(self, html_content: str) -> List[RouteSection]:
        """Использует selectolax для парсинга таблицы норм."""
        parser = HTMLParser(html_content)
        norm_sections = []
        
        tables = parser.css('table')
        
        for table in tables:
            headers = table.css('th')
            header_texts = [self.clean_text(h.text()) for h in headers]
            
            # Проверяем, что это нужная таблица
            if any('Нормируемый участок' in h for h in header_texts):
                # Определяем индексы колонок
                col_indices = self._get_column_indices(header_texts)
                
                # Парсим строки данных
                rows = table.css('tr')
                for row in rows:
                    cells = row.css('td')
                    if not cells:
                        continue
                    
                    # Проверяем, что это не итоговая строка
                    first_cell_text = self.clean_text(cells[0].text())
                    if 'итого' in first_cell_text.lower():
                        continue
                    
                    section = self._parse_norm_row(cells, col_indices)
                    if section.name:
                        norm_sections.append(section)
        
        return norm_sections
    
    def _parse_norm_table_bs4(self, html_content: str) -> List[RouteSection]:
        """Использует BeautifulSoup для парсинга таблицы норм."""
        soup = BeautifulSoup(html_content, 'html.parser')
        norm_sections = []
        
        tables = soup.find_all('table')
        
        for table in tables:
            headers = table.find_all('th')
            header_texts = [self.clean_text(h.get_text()) for h in headers]
            
            if any('Нормируемый участок' in h for h in header_texts):
                col_indices = self._get_column_indices(header_texts)
                
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if not cells:
                        continue
                    
                    first_cell_text = self.clean_text(cells[0].get_text())
                    if 'итого' in first_cell_text.lower():
                        continue
                    
                    section = self._parse_norm_row_bs4(cells, col_indices)
                    if section.name:
                        norm_sections.append(section)
        
        return norm_sections
    
    def parse_station_table(self, html_content: str) -> Dict[str, Dict]:
        """Парсит таблицу со станциями и дополнительными данными."""
        if USE_SELECTOLAX:
            return self._parse_station_table_selectolax(html_content)
        else:
            return self._parse_station_table_bs4(html_content)
    
    def _parse_station_table_selectolax(self, html_content: str) -> Dict[str, Dict]:
        """Использует selectolax для парсинга таблицы станций."""
        parser = HTMLParser(html_content)
        station_sections = {}
        
        tables = parser.css('table')
        
        for table in tables:
            headers = table.css('th')
            header_texts = [self.clean_text(h.text()) for h in headers]
            
            # Проверяем, что это нужная таблица
            patterns = ['В том числе', 'In that number']
            if any(any(pattern in h for pattern in patterns) for h in header_texts):
                rows = table.css('tr')
                
                for row in rows:
                    cells = row.css('td')
                    if not cells:
                        continue
                    
                    # Первая ячейка - название участка
                    section_name = self.clean_text(cells[0].text())
                    if not section_name or any(pattern in section_name.lower() for pattern in ['итого', 'total']):
                        continue
                    
                    # Парсим остальные данные
                    data = self._parse_station_row(cells)
                    station_sections[section_name] = data
        
        return station_sections
    
    def _parse_station_table_bs4(self, html_content: str) -> Dict[str, Dict]:
        """Использует BeautifulSoup для парсинга таблицы станций."""
        soup = BeautifulSoup(html_content, 'html.parser')
        station_sections = {}
        
        tables = soup.find_all('table')
        
        for table in tables:
            headers = table.find_all('th')
            header_texts = [self.clean_text(h.get_text()) for h in headers]
            
            patterns = ['В том числе', 'In that number']
            if any(any(pattern in h for pattern in patterns) for h in header_texts):
                rows = table.find_all('tr')
                
                for row in rows:
                    cells = row.find_all('td')
                    if not cells:
                        continue
                    
                    section_name = self.clean_text(cells[0].get_text())
                    if not section_name or any(pattern in section_name.lower() for pattern in ['итого', 'total']):
                        continue
                    
                    data = self._parse_station_row_bs4(cells)
                    station_sections[section_name] = data
        
        return station_sections
    
    def _get_column_indices(self, header_texts: List[str]) -> Dict[str, int]:
        """Определяет индексы колонок в таблице норм."""
        col_indices = {}
        for idx, header in enumerate(header_texts):
            header_lower = header.lower()
            if 'участок' in header_lower and 'станция' in header_lower:
                col_indices['name'] = idx
            elif 'ткм брутто' in header_lower:
                col_indices['tkm_brutto'] = idx
            elif header_lower in ['км', 'км.']:
                col_indices['km'] = idx
            elif header_lower in ['пр.', 'пр']:
                col_indices['pr'] = idx
            elif 'расход фактический' in header_lower:
                col_indices['rashod_fact'] = idx
            elif 'расход по норме' in header_lower:
                col_indices['rashod_norm'] = idx
            elif 'уд. норма' in header_lower or 'норма на 1 час ман. раб.' in header_lower:
                col_indices['ud_norma'] = idx
            elif 'норма на работу' in header_lower:
                col_indices['norma_rabotu'] = idx
            elif 'норма на одиночное' in header_lower:
                col_indices['norma_odinochnoe'] = idx
        
        return col_indices
    
    def _parse_norm_row(self, cells, col_indices: Dict[str, int]) -> RouteSection:
        """Парсит строку данных из таблицы норм (selectolax версия)."""
        section = RouteSection()
        
        # Извлекаем данные по индексам
        for field, idx in col_indices.items():
            if idx < len(cells):
                cell = cells[idx]
                value = self.clean_text(cell.text())
                
                # Для удельной нормы также извлекаем URL и номер нормы
                if field == 'ud_norma':
                    link = cell.css_first('a')
                    if link and link.attributes.get('href'):
                        section.ud_norma_url = link.attributes['href']
                        norm_number = self.extract_norm_url_from_href(link.attributes['href'])
                        if norm_number:
                            section.norm_number = norm_number
                
                # Преобразуем числовые значения
                if field != 'name':
                    value = self.try_convert_to_number(value)
                
                setattr(section, field, value)
        
        return section
    
    def _parse_norm_row_bs4(self, cells, col_indices: Dict[str, int]) -> RouteSection:
        """Парсит строку данных из таблицы норм (BeautifulSoup версия)."""
        section = RouteSection()
        
        for field, idx in col_indices.items():
            if idx < len(cells):
                cell = cells[idx]
                value = self.clean_text(cell.get_text())
                
                if field == 'ud_norma':
                    link = cell.find('a')
                    if link and link.get('href'):
                        section.ud_norma_url = link.get('href')
                        norm_number = self.extract_norm_url_from_href(link.get('href'))
                        if norm_number:
                            section.norm_number = norm_number
                
                if field != 'name':
                    value = self.try_convert_to_number(value)
                
                setattr(section, field, value)
        
        return section
    
    def _parse_station_row(self, cells) -> Dict:
        """Парсит строку данных из таблицы станций (selectolax версия)."""
        data = {}
        
        field_mapping = [
            'prostoy_vsego', 'prostoy_norma', 'manevry_vsego', 'manevry_norma',
            'troganie_vsego', 'troganie_norma', 'nagon_vsego', 'nagon_norma',
            'ogranich_vsego', 'ogranich_norma', 'peresyl_vsego', 'peresyl_norma'
        ]
        
        for i, field in enumerate(field_mapping, 1):
            if i < len(cells):
                data[field] = self.try_convert_to_number(cells[i].text())
        
        return data
    
    def _parse_station_row_bs4(self, cells) -> Dict:
        """Парсит строку данных из таблицы станций (BeautifulSoup версия)."""
        data = {}
        
        field_mapping = [
            'prostoy_vsego', 'prostoy_norma', 'manevry_vsego', 'manevry_norma',
            'troganie_vsego', 'troganie_norma', 'nagon_vsego', 'nagon_norma',
            'ogranich_vsego', 'ogranich_norma', 'peresyl_vsego', 'peresyl_norma'
        ]
        
        for i, field in enumerate(field_mapping, 1):
            if i < len(cells):
                data[field] = self.try_convert_to_number(cells[i].get_text())
        
        return data