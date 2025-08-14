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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HTMLRouteParser:
    """Парсер HTML файлов с маршрутами"""
    
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
            
        # Фильтрация маршрутов по логике
        filtered_routes = self._filter_routes(all_routes)
        logger.info(f"После фильтрации осталось {len(filtered_routes)} маршрутов")
        
        # Преобразование в DataFrame
        df = pd.DataFrame(filtered_routes)
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
                    break
            except Exception as e:
                logger.debug(f"Не удалось прочитать файл с кодировкой {encoding}: {e}")
                continue
                
        if not content:
            logger.error(f"Не удалось прочитать файл {file_path}")
            return []
            
        soup = BeautifulSoup(content, 'html.parser')
        routes = []
        
        # Поиск блоков с информацией о маршрутах
        route_blocks = self._find_route_blocks(soup)
        logger.debug(f"Найдено {len(route_blocks)} блоков маршрутов")
        
        for block in route_blocks:
            try:
                route_data = self._extract_route_data_from_block(block)  # ✅ ИСПРАВЛЕНО
                if route_data:
                    routes.append(route_data)
                    logger.debug(f"Обработан маршрут №{route_data.get('Номер маршрута', 'N/A')}")
            except Exception as e:
                logger.error(f"Ошибка при парсинге блока маршрута: {e}")
                
        return routes
    
    def _find_route_blocks(self, soup: BeautifulSoup) -> List:
        """Поиск блоков с маршрутами в HTML"""
        route_blocks = []
        
        # Ищем заголовки с информацией о маршруте
        filter_headers = soup.find_all('th', class_='thl_common')
        
        for header in filter_headers:
            # Проверяем, содержит ли заголовок информацию о маршруте
            header_text = header.get_text()
            if 'Маршрут №' in header_text and 'Идентификатор' in header_text:
                # Возвращаем сам заголовок как блок
                route_blocks.append(header)
                    
        return route_blocks
    
    def _extract_route_data_from_block(self, header) -> Optional[Dict]:  # ✅ ПЕРЕИМЕНОВАНО
        """Извлечение данных маршрута из заголовка и следующих таблиц"""
        try:
            header_text = header.get_text()
            
            # Извлекаем базовую информацию из заголовка
            route_info = self._parse_header_info(header_text)
            if not route_info:
                return None
                
            logger.debug(f"Найден маршрут: {route_info}")
            
            # Ищем таблицы с данными после заголовка
            current_element = header.find_parent('table')
            if not current_element:
                return None
                
            # Ищем следующие таблицы с данными о расходах
            tables_data = self._find_route_tables(current_element)
            
            if tables_data:
                route_info.update(tables_data)
                return route_info
                
        except Exception as e:
            logger.error(f"Ошибка при извлечении данных маршрута: {e}")
            
        return None
    
    def _parse_header_info(self, header_text: str) -> Optional[Dict]:
        """Парсинг информации из заголовка маршрута"""
        try:
            # Регулярные выражения для извлечения данных
            route_pattern = r'Маршрут №:\s*(\d+)'
            date_pattern = r'Дата:\s*([\d.]+)'
            depot_pattern = r'Депо:\s*(\d+\s+[^И]*)'
            id_pattern = r'Идентификатор:\s*(\d+)'
            
            route_match = re.search(route_pattern, header_text)
            date_match = re.search(date_pattern, header_text)
            depot_match = re.search(depot_pattern, header_text)
            id_match = re.search(id_pattern, header_text)
            
            if not all([route_match, date_match, id_match]):
                logger.warning(f"Не все обязательные поля найдены в заголовке: {header_text}")
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
    
    def _find_route_tables(self, start_element) -> Optional[Dict]:
        """Поиск таблиц с данными маршрута"""
        try:
            # Ищем таблицы с данными о расходах
            next_sibling = start_element.find_next_sibling()
            tables_found = 0
            
            while next_sibling and tables_found < 10:  # Ограничиваем поиск
                if next_sibling.name == 'table':
                    # Проверяем, является ли это таблицей с данными о расходах
                    table_data = self._parse_consumption_table(next_sibling)
                    if table_data:
                        return table_data
                    tables_found += 1
                    
                next_sibling = next_sibling.find_next_sibling()
                
        except Exception as e:
            logger.error(f"Ошибка при поиске таблиц: {e}")
            
        return None
    
    def _parse_consumption_table(self, table) -> Optional[Dict]:
        """Парсинг таблицы с данными о расходах"""
        try:
            # Ищем строки с данными об участках
            rows = table.find_all('tr')
            
            sections_data = []
            
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 6:  # Минимальное количество ячеек для строки с данными
                    section_data = self._parse_section_row(cells)
                    if section_data:
                        sections_data.append(section_data)
                        
            if sections_data:
                logger.debug(f"Найдено {len(sections_data)} участков в таблице")
                return {'sections': sections_data}
                
        except Exception as e:
            logger.error(f"Ошибка парсинга таблицы расходов: {e}")
            
        return None
    
    def _parse_section_row(self, cells) -> Optional[Dict]:
        """Парсинг строки с данными участка"""
        try:
            # Проверяем, что это строка с данными участка
            section_name_cell = cells[0]
            section_name = section_name_cell.get_text().strip()
            
            # Игнорируем итоговые строки и заголовки
            if not section_name or 'Итого' in section_name or section_name.isdigit():
                return None
                
            # Извлекаем данные из ячеек
            tkm_brutto = self._extract_number(cells[1].get_text()) if len(cells) > 1 else 0
            km = self._extract_number(cells[2].get_text()) if len(cells) > 2 else 0
            actual_consumption = self._extract_number(cells[4].get_text()) if len(cells) > 4 else 0
            norm_consumption = self._extract_number(cells[5].get_text()) if len(cells) > 5 else 0
            
            # Извлекаем номер нормы из гиперссылки
            norm_number = None
            if len(cells) > 6:
                norm_link = cells[6].find('a')
                if norm_link and norm_link.get('href'):
                    norm_number = self._extract_norm_number(norm_link.get('href'))
            
            # Если номер нормы не найден в 7-й колонке, ищем в других
            if norm_number is None:
                for i, cell in enumerate(cells):
                    links = cell.find_all('a')
                    for link in links:
                        if link.get('href'):
                            extracted_norm = self._extract_norm_number(link.get('href'))
                            if extracted_norm:
                                norm_number = extracted_norm
                                break
                    if norm_number:
                        break
                        
            # Проверяем валидность данных
            if actual_consumption == 0 and norm_consumption == 0:
                return None
                
            # Игнорируем маршруты где расход по норме равен фактическому (с погрешностью)
            if actual_consumption != 0 and abs(actual_consumption - norm_consumption) < 1.0:
                logger.debug(f"Игнорируем участок {section_name}: фактический расход примерно равен нормативному")
                return None
                
            return {
                'Наименование участка': section_name,
                'Ткм брутто': tkm_brutto,
                'Км': km,
                'Фактический удельный': actual_consumption,
                'Расход по норме': norm_consumption,
                'Номер нормы': norm_number
            }
            
        except Exception as e:
            logger.error(f"Ошибка парсинга строки участка: {e}")
            return None
    
    def _extract_number(self, text: str) -> float:
        """Извлечение числа из текста"""
        try:
            # Убираем пробелы и заменяем запятые на точки
            cleaned_text = re.sub(r'[^\d.,-]', '', text.strip())
            cleaned_text = cleaned_text.replace(',', '.')
            
            # Ищем число
            number_match = re.search(r'-?\d+\.?\d*', cleaned_text)
            if number_match:
                return float(number_match.group())
        except:
            pass
        return 0.0
    
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
    
    def _filter_routes(self, routes: List[Dict]) -> List[Dict]:
        """Фильтрация маршрутов по логике выбора самых поздних версий"""
        logger.info("Начало фильтрации маршрутов")
        
        # Группируем маршруты по номеру и дате
        routes_groups = {}
        
        for route in routes:
            if 'sections' not in route:
                continue
                
            key = (route['Номер маршрута'], route['Дата маршрута'])
            if key not in routes_groups:
                routes_groups[key] = []
            routes_groups[key].append(route)
            
        logger.info(f"Сгруппировано {len(routes_groups)} уникальных маршрутов (номер+дата)")
        
        # Выбираем самую позднюю версию для каждой группы
        filtered_routes = []
        
        for key, group in routes_groups.items():
            logger.debug(f"Обработка группы {key}: {len(group)} версий")
            
            # Сортируем по идентификатору (чем больше, тем позднее)
            group.sort(key=lambda x: x['Идентификатор'], reverse=True)
            
            # Выбираем самую позднюю версию где есть различия в расходах
            selected_route = None
            
            for route in group:
                # Проверяем, есть ли участки с различающимися расходами
                valid_sections = []
                for section in route['sections']:
                    actual = section.get('Фактический удельный', 0)
                    norm = section.get('Расход по норме', 0)
                    if actual != 0 and norm != 0 and abs(actual - norm) > 1.0:  # Увеличили порог до 1.0
                        valid_sections.append(section)
                        
                if valid_sections:
                    route['sections'] = valid_sections
                    selected_route = route
                    logger.debug(f"Выбран маршрут с идентификатором {route['Идентификатор']}: {len(valid_sections)} участков")
                    break
                    
            if selected_route:
                # Создаем строки для каждого участка
                for section in selected_route['sections']:
                    # Вычисляем нагрузку на ось более корректно
                    tkm_brutto = section['Ткм брутто']
                    km = section['Км']
                    
                    # Примерный расчет нагрузки на ось
                    if km > 0 and tkm_brutto > 0:
                        # Простая формула: нагрузка = брутто / км (это грубое приближение)
                        load_per_axle = tkm_brutto / km
                    else:
                        load_per_axle = 20.0  # Значение по умолчанию
                    
                    row = {
                        'Номер маршрута': selected_route['Номер маршрута'],
                        'Дата маршрута': selected_route['Дата маршрута'],
                        'Депо': selected_route['Депо'],
                        'Идентификатор': selected_route['Идентификатор'],
                        'Наименование участка': section['Наименование участка'],
                        'БРУТТО': tkm_brutto / 1000 if tkm_brutto else 0,  # Переводим в тысячи
                        'Фактический удельный': section['Фактический удельный'],
                        'Номер нормы': section['Номер нормы'],
                        'Нажатие на ось': load_per_axle
                    }
                    filtered_routes.append(row)
                    
        logger.info(f"Финальная фильтрация: {len(filtered_routes)} записей")
        return filtered_routes