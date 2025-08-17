# analysis/html_parser.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

# Настройка логирования
logger = logging.getLogger(__name__)

class HTMLRouteParser:
    """Парсер HTML файлов с маршрутами из системы ИОММ"""
    
    def __init__(self):
        self.routes_data = []
        
    def parse_html_files(self, html_files: List[str]) -> pd.DataFrame:
        """Парсинг нескольких HTML файлов"""
        logger.info(f"Начало парсинга {len(html_files)} HTML файлов")
        
        all_routes = []
        
        for file_path in html_files:
            logger.info(f"Обработка файла: {file_path}")
            try:
                routes = self._parse_single_html_file(file_path)
                all_routes.extend(routes)
                logger.info(f"Из файла {file_path} извлечено {len(routes)} маршрутов")
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {e}")
                
        logger.info(f"Всего извлечено {len(all_routes)} маршрутов")
        
        if not all_routes:
            logger.warning("Не найдено ни одного маршрута")
            return pd.DataFrame()
            
        # Преобразование в DataFrame
        df = pd.DataFrame(all_routes)
        logger.info(f"Создан DataFrame с колонками: {list(df.columns)}")
        
        return df
    
    def _parse_single_html_file(self, file_path: str) -> List[Dict]:
        """Парсинг одного HTML файла"""
        # Пробуем разные кодировки
        content = None
        for encoding in ['utf-8', 'windows-1251', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
                    logger.debug(f"Файл прочитан с кодировкой {encoding}")
                    break
            except Exception as e:
                logger.debug(f"Не удалось прочитать файл с кодировкой {encoding}: {e}")
                continue
                
        if not content:
            logger.error(f"Не удалось прочитать файл {file_path}")
            return []
            
        soup = BeautifulSoup(content, 'html.parser')
        routes = []
        
        # Поиск заголовков маршрутов
        route_headers = self._find_route_headers(soup)
        logger.debug(f"Найдено {len(route_headers)} заголовков маршрутов")
        
        for header in route_headers:
            try:
                route_data = self._parse_route_from_header(header, soup)
                if route_data:
                    routes.extend(route_data)  # route_data теперь список записей
                    logger.debug(f"Обработан маршрут №{route_data[0].get('Номер маршрута', 'N/A') if route_data else 'N/A'}")
            except Exception as e:
                logger.error(f"Ошибка при парсинге маршрута: {e}")
                continue
                
        return routes
    
    def _find_route_headers(self, soup: BeautifulSoup) -> List:
        """Поиск заголовков с маршрутами"""
        headers = []
        
        # Ищем все элементы th с классом thl_common
        route_headers = soup.find_all('th', class_='thl_common')
        
        for header in route_headers:
            header_text = header.get_text()
            if 'Маршрут №' in header_text and 'Идентификатор' in header_text:
                headers.append(header)
                logger.debug(f"Найден заголовок маршрута: {header_text[:100]}...")
                
        return headers
    
    def _parse_route_from_header(self, header, soup: BeautifulSoup) -> List[Dict]:
        """Извлечение данных маршрута из заголовка"""
        try:
            header_text = header.get_text()
            
            # Извлекаем базовую информацию из заголовка
            route_info = self._parse_header_info(header_text)
            if not route_info:
                return []
                
            logger.debug(f"Найден маршрут: {route_info}")
            
            # Ищем таблицу с данными после заголовка
            data_table = self._find_data_table_after_header(header)
            
            if not data_table:
                logger.warning(f"Не найдена таблица данных для маршрута {route_info['Номер маршрута']}")
                return []
            
            # Парсим строки данных из таблицы
            route_records = self._parse_data_table(data_table, route_info)
            
            return route_records
                
        except Exception as e:
            logger.error(f"Ошибка при извлечении данных маршрута: {e}")
            return []
    
    def _parse_header_info(self, header_text: str) -> Optional[Dict]:
        """Парсинг информации из заголовка маршрута"""
        try:
            # Очищаем текст от лишних пробелов и символов
            clean_text = re.sub(r'\s+', ' ', header_text.strip())
            
            # Регулярные выражения для извлечения данных
            route_pattern = r'Маршрут\s*№:\s*(\d+)'
            date_pattern = r'Дата:\s*([\d.]+)'
            depot_pattern = r'Депо:\s*([^И]+?)(?=Идентификатор|$)'
            id_pattern = r'Идентификатор:\s*(\d+)'
            
            route_match = re.search(route_pattern, clean_text)
            date_match = re.search(date_pattern, clean_text)
            depot_match = re.search(depot_pattern, clean_text)
            id_match = re.search(id_pattern, clean_text)
            
            if not all([route_match, date_match, id_match]):
                logger.warning(f"Не все обязательные поля найдены в заголовке: {clean_text}")
                return None
                
            return {
                'Номер маршрута': int(route_match.group(1)),
                'Дата маршрута': date_match.group(1),
                'Депо': depot_match.group(1).strip() if depot_match else '',
                'Идентификатор': int(id_match.group(1))
            }
        except Exception as e:
            logger.error(f"Ошибка парсинга заголовка: {e}")
            return None
    
    def _find_data_table_after_header(self, header) -> Optional:
        """Поиск таблицы с данными после заголовка"""
        try:
            # Начинаем с родительской таблицы заголовка
            current_element = header.find_parent('table')
            if not current_element:
                return None
            
            # Ищем следующие элементы после таблицы заголовка
            next_element = current_element.find_next_sibling()
            search_count = 0
            
            while next_element and search_count < 10:
                if next_element.name == 'table':
                    # Проверяем, является ли это таблицей с данными о расходах
                    if self._is_data_table(next_element):
                        logger.debug("Найдена таблица с данными")
                        return next_element
                
                next_element = next_element.find_next_sibling()
                search_count += 1
                
            logger.warning("Таблица с данными не найдена")
            return None
                
        except Exception as e:
            logger.error(f"Ошибка при поиске таблицы данных: {e}")
            return None
    
    def _is_data_table(self, table) -> bool:
        """Проверяет, является ли таблица таблицей с данными о расходах"""
        try:
            # Ищем заголовки, характерные для таблицы данных
            headers = table.find_all('th')
            header_texts = [th.get_text().strip() for th in headers]
            
            # Ключевые слова, которые должны быть в заголовках таблицы данных
            keywords = ['участок', 'станция', 'расход', 'норма', 'брутто']
            
            header_text_combined = ' '.join(header_texts).lower()
            
            # Проверяем наличие ключевых слов
            keyword_count = sum(1 for keyword in keywords if keyword in header_text_combined)
            
            if keyword_count >= 3:  # Минимум 3 ключевых слова
                logger.debug(f"Найдена таблица данных с заголовками: {header_texts[:3]}...")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Ошибка при проверке таблицы: {e}")
            return False
    
    def _parse_data_table(self, table, route_info: Dict) -> List[Dict]:
        """Парсинг таблицы с данными о расходах"""
        try:
            rows = table.find_all('tr')
            data_records = []
            
            # Пропускаем заголовочные строки
            data_rows = []
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 5:  # Минимум 5 колонок для строки с данными
                    # Проверяем, что это не заголовок
                    first_cell_text = cells[0].get_text().strip()
                    if (not first_cell_text.lower() in ['нормируемый', 'участок', '1', 'итого'] and 
                        first_cell_text and 
                        not first_cell_text.isdigit() and
                        'итого' not in first_cell_text.lower()):
                        data_rows.append(cells)
            
            logger.debug(f"Найдено {len(data_rows)} строк данных")
            
            for cells in data_rows:
                try:
                    record = self._parse_data_row(cells, route_info)
                    if record:
                        data_records.append(record)
                except Exception as e:
                    logger.error(f"Ошибка при парсинге строки данных: {e}")
                    continue
            
            logger.debug(f"Создано {len(data_records)} записей данных")
            return data_records
            
        except Exception as e:
            logger.error(f"Ошибка парсинга таблицы данных: {e}")
            return []
    
    def _parse_data_row(self, cells, route_info: Dict) -> Optional[Dict]:
        """Парсинг строки с данными участка"""
        try:
            if len(cells) < 5:
                return None
                
            # Извлекаем название участка
            section_name = cells[0].get_text().strip()
            
            # Пропускаем итоговые строки и некорректные названия
            if (not section_name or 
                'итого' in section_name.lower() or 
                section_name.isdigit() or
                len(section_name) < 3):
                return None
            
            # Извлекаем числовые данные
            tkm_brutto = self._extract_number(cells[1].get_text()) if len(cells) > 1 else 0
            km = self._extract_number(cells[2].get_text()) if len(cells) > 2 else 0
            
            # Индекс может варьироваться, ищем колонки с расходами
            actual_consumption = 0
            norm_consumption = 0
            norm_number = None
            
            # Ищем колонки с фактическим и нормативным расходом
            for i, cell in enumerate(cells):
                cell_text = cell.get_text().strip()
                
                # Фактический расход (обычно в 4-5 колонке)
                if i in [3, 4] and self._is_number(cell_text):
                    actual_consumption = self._extract_number(cell_text)
                
                # Нормативный расход (обычно в 5-6 колонке)  
                if i in [4, 5] and self._is_number(cell_text):
                    if actual_consumption == 0:  # Если еще не нашли фактический
                        actual_consumption = self._extract_number(cell_text)
                    else:  # Если уже нашли фактический, это нормативный
                        norm_consumption = self._extract_number(cell_text)
                
                # Ищем ссылки с номером нормы
                links = cell.find_all('a')
                for link in links:
                    href = link.get('href', '')
                    if 'id_ntp_tax' in href:
                        norm_number = self._extract_norm_number(href)
            
            # Проверяем валидность данных
            if actual_consumption == 0:
                logger.debug(f"Пропускаем участок {section_name}: нет данных о расходе")
                return None
            
            # Вычисляем нагрузку на ось (примерная формула)
            if km > 0 and tkm_brutto > 0:
                load_per_axle = tkm_brutto / km / 1000 * 20  # Примерная формула
            else:
                load_per_axle = 20.0  # Значение по умолчанию
            
            record = {
                'Номер маршрута': route_info['Номер маршрута'],
                'Дата маршрута': route_info['Дата маршрута'],
                'Депо': route_info['Депо'],
                'Идентификатор': route_info['Идентификатор'],
                'Наименование участка': section_name,
                'БРУТТО': tkm_brutto / 1000 if tkm_brutto else 0,  # В тысячах
                'Фактический удельный': actual_consumption,
                'Номер нормы': norm_number,
                'Нажатие на ось': max(15.0, min(25.0, load_per_axle))  # Ограничиваем разумными пределами
            }
            
            logger.debug(f"Создана запись для участка: {section_name}")
            return record
            
        except Exception as e:
            logger.error(f"Ошибка парсинга строки участка: {e}")
            return None
    
    def _extract_number(self, text: str) -> float:
        """Извлечение числа из текста"""
        try:
            if not text or pd.isna(text):
                return 0.0
                
            # Очищаем текст
            cleaned_text = str(text).strip()
            
            # Убираем лишние символы, оставляем цифры, точки, запятые и минусы
            cleaned_text = re.sub(r'[^\d.,-]', '', cleaned_text)
            
            # Заменяем запятые на точки
            cleaned_text = cleaned_text.replace(',', '.')
            
            # Удаляем множественные точки
            cleaned_text = re.sub(r'\.+', '.', cleaned_text)
            
            # Ищем число
            number_match = re.search(r'-?\d*\.?\d+', cleaned_text)
            if number_match:
                return float(number_match.group())
                
        except (ValueError, TypeError):
            pass
            
        return 0.0
    
    def _is_number(self, text: str) -> bool:
        """Проверяет, содержит ли текст число"""
        try:
            cleaned = re.sub(r'[^\d.,-]', '', str(text).strip())
            return bool(re.search(r'\d', cleaned))
        except:
            return False
    
    def _extract_norm_number(self, href: str) -> Optional[int]:
        """Извлечение номера нормы из гиперссылки"""
        try:
            # Ищем id_ntp_tax в ссылке
            match = re.search(r'id_ntp_tax=(\d+)', href)
            if match:
                norm_number = int(match.group(1))
                logger.debug(f"Найден номер нормы: {norm_number}")
                return norm_number
        except Exception as e:
            logger.error(f"Ошибка извлечения номера нормы из {href}: {e}")
        return None