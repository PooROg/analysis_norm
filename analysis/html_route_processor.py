# analysis/html_route_processor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import logging
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from bs4 import BeautifulSoup
import tempfile
import shutil

# Настройка логирования
logger = logging.getLogger(__name__)

class HTMLRouteProcessor:
    """Процессор для обработки HTML файлов маршрутов"""
    
    def __init__(self):
        self.processed_routes = []
        self.processing_stats = {
            'total_files': 0,
            'total_routes_found': 0,
            'unique_routes': 0,
            'duplicates_total': 0,
            'routes_with_equal_rashod': 0,
            'routes_processed': 0,
            'routes_skipped': 0
        }
    
    def clean_html_file(self, input_file: str) -> str:
        """Очищает HTML файл от лишнего кода (адаптация из delet_code_marhrut.py)"""
        logger.info(f"Очистка HTML файла: {input_file}")
        
        # Проверяем существование файла
        if not os.path.exists(input_file):
            logger.error(f"Файл {input_file} не найден!")
            return None
        
        try:
            # Читаем файл с кодировкой cp1251
            with open(input_file, 'r', encoding='cp1251') as f:
                content = f.read()
            encoding_used = 'cp1251'
            logger.debug(f"Файл прочитан с кодировкой cp1251, размер: {len(content)} байт")
        except UnicodeDecodeError:
            # Пробуем другие кодировки
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                encoding_used = 'utf-8'
                logger.debug(f"Файл прочитан с кодировкой utf-8")
            except UnicodeDecodeError:
                with open(input_file, 'r', encoding='latin-1') as f:
                    content = f.read()
                encoding_used = 'latin-1'
                logger.debug(f"Файл прочитан с кодировкой latin-1")
        
        # Ищем начальный маркер
        start_marker = '<table align=center width="100%">'
        start_pos = content.find(start_marker)
        
        if start_pos == -1:
            logger.error(f"Начальный маркер не найден в файле {input_file}")
            return None
        
        # Ищем конечный маркер
        form_pattern = r'</table>\s*</td>\s*</tr></table><form id=print_form>.*?(?=\n|$)'
        form_match = re.search(form_pattern, content[start_pos:], re.DOTALL)
        
        if not form_match:
            logger.error(f"Конечный маркер не найден в файле {input_file}")
            return None
        
        # Позиция конца найденной строки с формой
        form_end_pos = start_pos + form_match.end()
        
        # Ищем следующую строку '</td></tr>' после формы
        remaining_content = content[form_end_pos:]
        lines = remaining_content.split('\n')
        
        end_pos = form_end_pos
        for i, line in enumerate(lines):
            if '</td></tr>' in line.strip():
                end_pos = form_end_pos + len('\n'.join(lines[:i+1]))
                break
        
        # Извлекаем нужную часть
        extracted_data = content[start_pos:end_pos + 1]
        
        # Разбиваем маршруты по отдельным строкам
        extracted_data = self._split_routes_to_lines(extracted_data)
        
        # Удаляем маршруты с " ВЧТ "
        extracted_data = self._remove_vcht_routes(extracted_data)
        
        # Очищаем HTML код от лишних элементов
        extracted_data = self._clean_html_content(extracted_data)
        
        # Создаем временный файл
        temp_file = tempfile.mktemp(suffix='.html')
        with open(temp_file, 'w', encoding=encoding_used) as f:
            f.write(extracted_data)
        
        logger.info(f"HTML файл очищен и сохранен во временный файл: {temp_file}")
        return temp_file
    
    def _split_routes_to_lines(self, content: str) -> str:
        """Разбивает содержимое HTML на отдельные строки для каждого маршрута"""
        logger.debug("Разбиваем маршруты по отдельным строкам...")
        
        # Паттерн для поиска полного маршрута от начала до конца
        route_pattern = r'(<table[^>]*><tr><th class=thl_common><font class=filter_key>\s*Маршрут\s*№:.*?<br><br><br>)'
        
        # Находим все маршруты
        routes = re.findall(route_pattern, content, flags=re.DOTALL)
        
        if not routes:
            logger.warning("Маршруты не найдены для разделения")
            return content
        
        logger.debug(f"Найдено маршрутов для разделения: {len(routes)}")
        
        # Находим код до первого маршрута
        first_route_start = content.find(routes[0])
        before_routes = content[:first_route_start]
        
        # Находим код после последнего маршрута  
        last_route_end = content.rfind(routes[-1]) + len(routes[-1])
        after_routes = content[last_route_end:]
        
        # Собираем результат
        result_lines = []
        
        if before_routes.strip():
            result_lines.append(before_routes.rstrip())
        
        result_lines.append("<!-- НАЧАЛО_ПЕРВОГО_МАРШРУТА -->")
        
        for route in routes:
            result_lines.append(route)
        
        result_lines.append("<!-- КОНЕЦ_ПОСЛЕДНЕГО_МАРШРУТА -->")
        
        if after_routes.strip():
            result_lines.append(after_routes.lstrip())
        
        logger.debug(f"Маршруты разделены на {len(routes)} отдельных строк")
        return '\n'.join(result_lines)
    
    def _remove_vcht_routes(self, content: str) -> str:
        """Удаляет строки с маршрутами, содержащими ' ВЧТ '"""
        logger.debug("Удаляем маршруты с ' ВЧТ '...")
        
        lines = content.split('\n')
        filtered_lines = []
        removed_routes = 0
        
        for line in lines:
            if '<td class = itog2>" ВЧТ "</td>' in line:
                removed_routes += 1
                logger.debug(f"Удален маршрут с ВЧТ (строка {len(filtered_lines) + 1})")
                continue
            filtered_lines.append(line)
        
        if removed_routes > 0:
            logger.info(f"Удалено {removed_routes} маршрутов с ' ВЧТ '")
        
        return '\n'.join(filtered_lines)
    
    def _clean_html_content(self, content: str) -> str:
        """Очищает HTML контент от лишних элементов"""
        logger.debug("Очищаем HTML код от лишних элементов...")
        
        original_size = len(content)
        
        # Удаляем лишние элементы (как в оригинальном коде)
        date_pattern = r'<font class = rcp12 ><center>Дата получения:.*?</font>\s*<br>'
        content = re.sub(date_pattern, '', content, flags=re.DOTALL)
        
        route_num_pattern = r'<font class = rcp12 ><center>Номер маршрута:.*?</font><br>'
        content = re.sub(route_num_pattern, '', content, flags=re.DOTALL)
        
        numline_pattern = r'<tr class=tr_numline>.*?</tr>'
        content = re.sub(numline_pattern, '', content, flags=re.DOTALL)
        
        # Удаляем атрибуты выравнивания и другие лишние элементы
        content = re.sub(r'\s+ALIGN=center', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\s+align=left', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\s+align=right', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<center>', '', content)
        content = re.sub(r'</center>', '', content)
        content = re.sub(r'<pre>', '', content)
        content = re.sub(r'</pre>', '', content)
        content = re.sub(r'>[ \t]+<', '><', content)
        
        cleaned_size = len(content)
        removed_bytes = original_size - cleaned_size
        
        logger.debug(f"Удалено {removed_bytes:,} байт лишнего кода ({removed_bytes/original_size*100:.1f}%)")
        
        return content
    
    def process_html_files(self, html_files: List[str]) -> pd.DataFrame:
        """Обрабатывает список HTML файлов маршрутов"""
        logger.info(f"Начинаем обработку {len(html_files)} HTML файлов")
        
        self.processing_stats['total_files'] = len(html_files)
        all_routes_data = []
        
        # Обрабатываем каждый файл
        for file_path in html_files:
            logger.info(f"Обработка файла: {os.path.basename(file_path)}")
            
            try:
                # Очищаем HTML файл
                cleaned_file = self.clean_html_file(file_path)
                if not cleaned_file:
                    logger.error(f"Не удалось очистить файл {file_path}")
                    continue
                
                # Обрабатываем очищенный файл
                file_routes = self._process_single_cleaned_file(cleaned_file)
                all_routes_data.extend(file_routes)
                
                # Удаляем временный файл
                try:
                    os.remove(cleaned_file)
                except:
                    pass
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла {file_path}: {e}")
                continue
        
        # Фильтруем и выбираем лучшие маршруты
        if all_routes_data:
            df = self._filter_and_select_best_routes(all_routes_data)
            logger.info(f"Обработка завершена. Получено {len(df)} итоговых записей")
            return df
        else:
            logger.warning("Не получено ни одной записи из всех файлов")
            return pd.DataFrame()
    
    def _process_single_cleaned_file(self, cleaned_file: str) -> List[Dict]:
        """Обрабатывает один очищенный HTML файл (адаптация из route_processor.py)"""
        logger.debug(f"Обработка очищенного файла: {cleaned_file}")
        
        try:
            with open(cleaned_file, 'r', encoding='cp1251') as f:
                html_content = f.read()
        except:
            with open(cleaned_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
        
        # Извлекаем маршруты из HTML
        routes = self._extract_routes_from_html(html_content)
        
        # Обрабатываем каждый маршрут
        routes_data = []
        for route_html, metadata in routes:
            try:
                # Проверяем на равные расходы
                rashod_equal = self._check_rashod_equal_html(route_html)
                
                # Парсим маршрут
                route_data = self._parse_html_route(route_html, metadata, rashod_equal)
                routes_data.extend(route_data)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке маршрута {metadata}: {e}")
                continue
        
        logger.debug(f"Из файла извлечено {len(routes_data)} записей")
        return routes_data
    
    def _extract_routes_from_html(self, html_content: str) -> List[Tuple[str, Dict]]:
        """Извлекает маршруты из HTML контента"""
        logger.debug("Извлекаем маршруты из HTML")
        
        # Ищем маркеры начала и конца маршрутов
        start_marker = "<!-- НАЧАЛО_ПЕРВОГО_МАРШРУТА -->"
        end_marker = "<!-- КОНЕЦ_ПОСЛЕДНЕГО_МАРШРУТА -->"
        
        start_pos = html_content.find(start_marker)
        end_pos = html_content.find(end_marker)
        
        if start_pos == -1 or end_pos == -1:
            logger.warning("Маркеры маршрутов не найдены, обрабатываем весь контент")
            routes_section = html_content
        else:
            routes_section = html_content[start_pos + len(start_marker):end_pos]
        
        # Разбиваем на строки
        lines = routes_section.strip().split('\n')
        routes = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Ищем маршруты
            if re.search(r'<table width=\d+%', line) and ('Маршрут №' in line or 'Маршрут' in line):
                # Извлекаем метаданные маршрута
                metadata = self._extract_route_header_from_html(line)
                if metadata:
                    routes.append((line, metadata))
                    logger.debug(f"Найден маршрут: №{metadata.get('number', 'N/A')}")
        
        logger.debug(f"Найдено маршрутов: {len(routes)}")
        return routes
    
    def _extract_route_header_from_html(self, html_line: str) -> Optional[Dict]:
        """Извлекает метаданные маршрута из HTML строки"""
        soup = BeautifulSoup(html_line, 'html.parser')
        
        header = soup.find('th', class_='thl_common')
        if not header:
            return None
        
        header_text = header.get_text()
        metadata = {}
        
        # Извлекаем данные с помощью регулярных выражений
        patterns = {
            'number': r'Маршрут\s*№[:\s]*(\d+)',
            'date': r'(\d{2}\.\d{2}\.\d{4})',
            'depot': r'Депо:\s*([^И]+)',
            'identifier': r'Идентификатор:\s*(\d+)'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, header_text)
            if match:
                metadata[field] = match.group(1).strip() if field == 'depot' else match.group(1)
        
        logger.debug(f"Извлечены метаданные: {metadata}")
        return metadata if metadata.get('number') and metadata.get('date') else None
    
    def _check_rashod_equal_html(self, route_html: str) -> bool:
        """Проверяет равны ли Расход по норме и Расход фактический"""
        soup = BeautifulSoup(route_html, 'html.parser')
        
        tables = soup.find_all('table', width='90%')
        
        for table in tables:
            headers = table.find_all('th')
            header_texts = [h.get_text().strip() for h in headers]
            
            if any('Нормируемый участок' in h for h in header_texts):
                rashod_fact_idx = None
                rashod_norm_idx = None
                
                for idx, header in enumerate(header_texts):
                    if 'Расход фактический' in header:
                        rashod_fact_idx = idx
                    elif 'Расход по норме' in header:
                        rashod_norm_idx = idx
                
                if rashod_fact_idx is not None and rashod_norm_idx is not None:
                    rows = table.find_all('tr')
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) > max(rashod_fact_idx, rashod_norm_idx):
                            fact_text = cells[rashod_fact_idx].get_text().strip()
                            norm_text = cells[rashod_norm_idx].get_text().strip()
                            
                            try:
                                fact_val = float(fact_text.replace(',', '.'))
                                norm_val = float(norm_text.replace(',', '.'))
                                
                                if abs(fact_val - norm_val) < 0.01:
                                    return True
                            except:
                                continue
        
        return False
    
    def _parse_html_route(self, route_html: str, metadata: Dict, rashod_equal: bool) -> List[Dict]:
        """Парсит данные маршрута из HTML (упрощенная версия)"""
        logger.debug(f"Парсим маршрут №{metadata.get('number')}")
        
        soup = BeautifulSoup(route_html, 'html.parser')
        
        # Базовые данные маршрута
        route_number = metadata.get('number')
        route_date = metadata.get('date')
        depot = metadata.get('depot', '')
        identifier = metadata.get('identifier')
        
        # Извлекаем данные локомотива (упрощенно)
        loco_series, loco_number = self._extract_loco_data_from_html(route_html)
        
        # Парсим таблицу с участками
        sections_data = self._parse_sections_table(soup)
        
        # Формируем результат
        output_rows = []
        for section in sections_data:
            row = {
                'Номер маршрута': route_number,
                'Дата маршрута': route_date,
                'Депо': depot,
                'Идентификатор': identifier,
                'Серия локомотива': loco_series,
                'Номер локомотива': loco_number,
                'Наименование участка': section.get('name'),
                'Номер нормы': section.get('norm_number'),
                'Ткм брутто': section.get('tkm_brutto'),
                'Км': section.get('km'),
                'Фактический удельный': section.get('rashod_fact'),
                'Расход по норме': section.get('rashod_norm'),
                'USE_RED_RASHOD': rashod_equal,
                # Добавляем другие необходимые поля...
            }
            output_rows.append(row)
        
        return output_rows
    
    def _extract_loco_data_from_html(self, html_content: str) -> Tuple[Optional[str], Optional[str]]:
        """Извлекает серию и номер локомотива (упрощенная версия)"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Ищем элементы с классом itog2 или itog3
        fonts = soup.find_all('font', class_=['itog2', 'itog3'])
        
        for font in fonts:
            text = font.get_text().strip()
            if text in ['ТУ3', 'TU3'] or 'ТУ3' in text:
                # Извлекаем данные локомотива
                current = font
                data_elements = []
                
                while current:
                    current = current.find_next_sibling()
                    if current and current.name == 'font':
                        text = current.get_text().strip()
                        if text:
                            data_elements.append(text)
                    elif current and current.name == 'br':
                        break
                    elif not current:
                        break
                
                if len(data_elements) >= 4:
                    # Структура: [номер_маршрута, депо, номер_лок, серия, ...]
                    loco_number_raw = data_elements[2]
                    series_raw = data_elements[3]
                    
                    # Обрабатываем серию
                    if ',' in series_raw:
                        series_part = series_raw.split(',')[0]
                    else:
                        series_part = series_raw
                    
                    series_clean = re.sub(r'[^\d]', '', series_part)
                    
                    if len(series_clean) > 1:
                        processed_series = series_clean[:-1]
                    else:
                        processed_series = series_clean
                    
                    # Меняем местами серию и номер
                    final_loco_series = loco_number_raw
                    final_loco_number = processed_series
                    
                    return final_loco_series, final_loco_number
        
        return None, None
    
    def _parse_sections_table(self, soup: BeautifulSoup) -> List[Dict]:
        """Парсит таблицу с участками"""
        sections = []
        
        tables = soup.find_all('table')
        
        for table in tables:
            headers = table.find_all('th')
            header_texts = [h.get_text().strip() for h in headers]
            
            # Проверяем, что это нужная таблица
            if any('Нормируемый участок' in h for h in header_texts):
                # Определяем индексы колонок
                col_indices = {}
                for idx, header in enumerate(header_texts):
                    header_lower = header.lower()
                    if 'участок' in header_lower:
                        col_indices['name'] = idx
                    elif 'ткм брутто' in header_lower:
                        col_indices['tkm_brutto'] = idx
                    elif header_lower == 'км' or header_lower == 'км.':
                        col_indices['km'] = idx
                    elif 'расход фактический' in header_lower:
                        col_indices['rashod_fact'] = idx
                    elif 'расход по норме' in header_lower:
                        col_indices['rashod_norm'] = idx
                    elif 'уд. норма' in header_lower:
                        col_indices['ud_norma'] = idx
                
                # Парсим строки данных
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if not cells:
                        continue
                    
                    # Проверяем, что это не итоговая строка
                    first_cell_text = cells[0].get_text().strip()
                    if 'итого' in first_cell_text.lower():
                        continue
                    
                    section = {}
                    
                    # Извлекаем данные по индексам
                    for field, idx in col_indices.items():
                        if idx < len(cells):
                            cell = cells[idx]
                            value = cell.get_text().strip()
                            
                            # Для удельной нормы также извлекаем URL и номер нормы
                            if field == 'ud_norma':
                                link = cell.find('a')
                                if link and link.get('href'):
                                    section['ud_norma_url'] = link.get('href')
                                    norm_number = self._extract_norm_number_from_href(link.get('href'))
                                    if norm_number:
                                        section['norm_number'] = norm_number
                            
                            # Преобразуем числовые значения
                            if field != 'name':
                                try:
                                    value = float(value.replace(',', '.'))
                                except:
                                    pass
                            
                            section[field] = value
                    
                    if section.get('name'):
                        sections.append(section)
        
        return sections
    
    def _extract_norm_number_from_href(self, href: str) -> Optional[str]:
        """Извлекает номер нормы из гиперссылки"""
        try:
            match = re.search(r'id_ntp_tax=(\d+)', href)
            if match:
                return match.group(1)
        except Exception as e:
            logger.debug(f"Ошибка извлечения номера нормы из {href}: {e}")
        return None
    
    def _filter_and_select_best_routes(self, all_routes_data: List[Dict]) -> pd.DataFrame:
        """Фильтрует и выбирает лучшие маршруты на основе идентификатора"""
        logger.info("Фильтруем и выбираем лучшие маршруты")
        
        if not all_routes_data:
            return pd.DataFrame()
        
        # Группируем маршруты по ключу (номер + дата)
        route_groups = defaultdict(list)
        
        for route_data in all_routes_data:
            if route_data.get('Номер маршрута') and route_data.get('Дата маршрута'):
                key = f"{route_data['Номер маршрута']}_{route_data['Дата маршрута']}"
                route_groups[key].append(route_data)
            else:
                self.processing_stats['routes_skipped'] += 1
        
        self.processing_stats['unique_routes'] = len(route_groups)
        
        # Выбираем лучший маршрут для каждой группы
        selected_routes = []
        
        for key, group in route_groups.items():
            logger.debug(f"Обработка группы маршрутов {key}: {len(group)} версий")
            
            if len(group) > 1:
                self.processing_stats['duplicates_total'] += len(group) - 1
            
            # Группируем по идентификатору (для исключения равных расходов)
            identifier_groups = defaultdict(list)
            for route in group:
                identifier = route.get('Идентификатор')
                if identifier:
                    identifier_groups[identifier].extend(group)
            
            # Фильтруем маршруты с равными расходами
            valid_identifiers = []
            equal_identifiers = []
            
            for identifier, routes in identifier_groups.items():
                has_equal_rashod = any(route.get('USE_RED_RASHOD', False) for route in routes)
                if has_equal_rashod:
                    equal_identifiers.append(identifier)
                    self.processing_stats['routes_with_equal_rashod'] += len(routes)
                else:
                    valid_identifiers.append(identifier)
            
            # Выбираем лучший идентификатор
            if valid_identifiers:
                best_identifier = max(valid_identifiers, key=int)
            elif equal_identifiers:
                best_identifier = max(equal_identifiers, key=int)
            else:
                continue
            
            # Добавляем маршруты с лучшим идентификатором
            for route in group:
                if route.get('Идентификатор') == best_identifier:
                    selected_routes.append(route)
                    break
        
        self.processing_stats['routes_processed'] = len(selected_routes)
        
        # Создаем DataFrame
        if selected_routes:
            df = pd.DataFrame(selected_routes)
            logger.info(f"Создан DataFrame с {len(df)} записями")
            return df
        else:
            logger.warning("Нет данных для создания DataFrame")
            return pd.DataFrame()
    
    def get_processing_stats(self) -> Dict:
        """Возвращает статистику обработки"""
        return self.processing_stats.copy()
    
    def export_to_excel(self, df: pd.DataFrame, output_file: str):
        """Экспортирует данные в Excel (опциональная функция)"""
        if df.empty:
            logger.warning("DataFrame пуст, нечего экспортировать")
            return False
        
        try:
            # Определяем порядок колонок
            columns = [
                'Номер маршрута', 'Дата маршрута', 'Серия локомотива', 'Номер локомотива',
                'Депо', 'Идентификатор', 'Наименование участка', 'Номер нормы',
                'Ткм брутто', 'Км', 'Фактический удельный', 'Расход по норме'
            ]
            
            # Переупорядочиваем колонки
            existing_columns = [col for col in columns if col in df.columns]
            df_export = df[existing_columns].copy()
            
            # Преобразуем дату
            if 'Дата маршрута' in df_export.columns:
                df_export['Дата маршрута'] = pd.to_datetime(
                    df_export['Дата маршрута'], 
                    format='%d.%m.%Y', 
                    errors='coerce'
                )
            
            # Сохраняем в Excel
            df_export.to_excel(output_file, index=False)
            logger.info(f"Данные экспортированы в {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка экспорта в Excel: {e}")
            return False
