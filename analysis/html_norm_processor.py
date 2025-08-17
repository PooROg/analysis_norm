# analysis/html_norm_processor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import logging
import tempfile
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup

# Настройка логирования
logger = logging.getLogger(__name__)

class HTMLNormProcessor:
    """Процессор для обработки HTML файлов норм"""
    
    def __init__(self):
        self.processed_norms = {}
        self.processing_stats = {
            'total_files': 0,
            'total_norms_found': 0,
            'new_norms': 0,
            'updated_norms': 0,
            'skipped_norms': 0
        }
    
    def clean_html_file(self, input_file: str) -> str:
        """Очищает HTML файл норм от лишнего кода (адаптация из delete_code_norm.py)"""
        logger.info(f"Очистка HTML файла норм: {input_file}")
        
        try:
            # Читаем HTML файл с кодировкой cp1251
            with open(input_file, 'r', encoding='cp1251') as f:
                html_content = f.read()
            logger.debug(f"Файл прочитан с кодировкой cp1251")
        except UnicodeDecodeError:
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                logger.debug(f"Файл прочитан с кодировкой utf-8")
            except Exception as e:
                logger.error(f"Ошибка чтения файла {input_file}: {e}")
                return None
        
        # Более точные паттерны для поиска полных секций с таблицами
        pattern1 = r'(<font class=rcp12><center><b>Удельные нормы электроэнергии и топлива по нагрузке на ось</b></center></font>.*?</table>.*?</table>)'
        pattern2 = r'(<font class=rcp12><center><b>Удельные нормы электроэнергии и топлива по весу поезда</b></center></font>.*?</table>.*?</table>)'
        
        # Ищем таблицы
        match1 = re.search(pattern1, html_content, re.DOTALL | re.IGNORECASE)
        match2 = re.search(pattern2, html_content, re.DOTALL | re.IGNORECASE)
        
        # Создаем новый HTML
        new_html = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Удельные нормы электроэнергии и топлива</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .rcp12 { font-size: 12px; }
        .filter_key { font-weight: bold; }
        .filter_value { color: blue; }
        .tr_head { background-color: #e0e0e0; }
        .thc { border: 1px solid #000; padding: 5px; text-align: center; }
        .tdc_str1, .tdc_str2 { border: 1px solid #000; padding: 3px; text-align: center; }
        .tdc_str1 { background-color: #f9f9f9; }
        .tdc_str2 { background-color: #ffffff; }
        table { border-collapse: collapse; margin: 20px auto; }
        .link { color: blue; text-decoration: underline; }
    </style>
</head>
<body>
'''
        
        if match1:
            new_html += match1.group(1) + '<br><br>\n'
            logger.debug("✓ Найдена таблица 'по нагрузке на ось'")
        
        if match2:
            new_html += match2.group(1) + '<br><br>\n'
            logger.debug("✓ Найдена таблица 'по весу поезда'")
        
        if not match1 and not match2:
            logger.warning("Искомые таблицы не найдены в файле!")
            new_html += '<h1>Искомые таблицы не найдены в файле</h1>\n'
        
        new_html += '''
</body>
</html>'''
        
        # Создаем временный файл
        temp_file = tempfile.mktemp(suffix='.html')
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(new_html)
        
        logger.info(f"HTML файл норм очищен и сохранен во временный файл: {temp_file}")
        return temp_file
    
    def process_html_files(self, html_files: List[str]) -> Dict[str, Dict]:
        """Обрабатывает список HTML файлов норм"""
        logger.info(f"Начинаем обработку {len(html_files)} HTML файлов норм")
        
        self.processing_stats['total_files'] = len(html_files)
        all_norms = {}
        
        # Обрабатываем каждый файл
        for file_path in html_files:
            logger.info(f"Обработка файла норм: {os.path.basename(file_path)}")
            
            try:
                # Очищаем HTML файл
                cleaned_file = self.clean_html_file(file_path)
                if not cleaned_file:
                    logger.error(f"Не удалось очистить файл {file_path}")
                    continue
                
                # Обрабатываем очищенный файл
                file_norms = self._extract_norms_from_cleaned_html(cleaned_file)
                
                # Объединяем нормы
                for norm_id, norm_data in file_norms.items():
                    if norm_id in all_norms:
                        logger.warning(f"Норма {norm_id} уже существует, перезаписываем")
                    all_norms[norm_id] = norm_data
                
                # Удаляем временный файл
                try:
                    os.remove(cleaned_file)
                except:
                    pass
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла норм {file_path}: {e}")
                continue
        
        self.processing_stats['total_norms_found'] = len(all_norms)
        logger.info(f"Обработка завершена. Найдено {len(all_norms)} норм")
        
        return all_norms
    
    def _extract_norms_from_cleaned_html(self, cleaned_file: str) -> Dict[str, Dict]:
        """Извлекает нормы из очищенного HTML файла (адаптация из html_to_excel_norm.py)"""
        logger.debug(f"Извлекаем нормы из файла: {cleaned_file}")
        
        # Читаем HTML файл
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Ищем обе таблицы
        norms_data = {}
        
        # Обработка таблицы по нагрузке на ось
        load_section = self._find_section_by_text(soup, 'нагрузке на ось')
        if load_section:
            load_norms = self._extract_norms_from_section(load_section, 'Нажатие')
            norms_data.update(load_norms)
            logger.debug(f"Найдено норм по нагрузке на ось: {len(load_norms)}")
        
        # Обработка таблицы по весу поезда
        weight_section = self._find_section_by_text(soup, 'весу поезда')
        if weight_section:
            weight_norms = self._extract_norms_from_section(weight_section, 'Вес')
            norms_data.update(weight_norms)
            logger.debug(f"Найдено норм по весу поезда: {len(weight_norms)}")
        
        logger.debug(f"Всего извлечено норм: {len(norms_data)}")
        return norms_data
    
    def _find_section_by_text(self, soup: BeautifulSoup, search_text: str):
        """Находит секцию по тексту заголовка"""
        for element in soup.find_all(text=True):
            if search_text in element:
                return element.parent
        return None
    
    def _extract_norms_from_section(self, section, norm_type: str) -> Dict[str, Dict]:
        """Извлекает нормы из секции"""
        norms = {}
        
        if not section:
            return norms
        
        # Ищем таблицу в секции
        current = section.parent
        for sibling in current.find_all_next('table'):
            if sibling.find('tr', class_='tr_head'):
                headers = self._get_table_headers(sibling)
                rows = sibling.find_all('tr')[1:]  # Пропускаем заголовок
                
                numeric_start = 9  # Начало после "Призн. алг. нормир."
                numeric_end = len(headers) - 2  # До колонок с датами
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) > 10:  # Проверяем что это строка с данными
                        norm_data = self._parse_norm_row(cells, headers, norm_type, numeric_start, numeric_end)
                        if norm_data and norm_data.get('norm_id'):
                            norms[norm_data['norm_id']] = norm_data
                break
        
        return norms
    
    def _get_table_headers(self, table) -> List[str]:
        """Получает заголовки таблицы"""
        headers = []
        header_row = table.find('tr', class_='tr_head')
        if header_row:
            for th in header_row.find_all('th'):
                header_text = self._clean_text(th.get_text())
                headers.append(header_text)
        return headers
    
    def _parse_norm_row(self, cells, headers: List[str], norm_type: str, 
                       numeric_start: int, numeric_end: int) -> Optional[Dict]:
        """Парсит строку нормы"""
        try:
            row_data = []
            norm_id = ""
            
            for i, cell in enumerate(cells):
                cell_text = self._clean_text(cell.get_text())
                row_data.append(cell_text)
                
                if i == 0:  # Первая колонка содержит ссылку
                    link = cell.find('a')
                    if link and link.get('href'):
                        norm_id = self._extract_norm_id_from_link(link['href'])
            
            if not norm_id or len(row_data) <= 10:
                return None
            
            # Извлекаем числовые данные
            numeric_data = {}
            for i in range(numeric_start, min(numeric_end, len(row_data), len(headers))):
                if i < len(row_data) and row_data[i].strip():
                    header_value = self._clean_text(headers[i])
                    try:
                        # Пытаемся преобразовать заголовок в число (нагрузка)
                        load_value = float(header_value.replace(',', '.'))
                        consumption_value = float(row_data[i].replace(',', '.'))
                        numeric_data[load_value] = consumption_value
                    except ValueError:
                        continue
            
            # Создаем данные нормы
            norm_data = {
                'norm_id': norm_id,
                'norm_type': norm_type,
                'description': f"Норма №{norm_id} ({norm_type})",
                'points': [(load, consumption) for load, consumption in sorted(numeric_data.items())],
                'base_data': {
                    'priznok_sost_tyag': self._convert_to_number(row_data[1]) if len(row_data) > 1 else None,
                    'priznok_rek': self._convert_to_number(row_data[2]) if len(row_data) > 2 else None,
                    'vid_dvizheniya': row_data[3] if len(row_data) > 3 else '',
                    'simvol_rod_raboty': self._convert_to_number(row_data[4]) if len(row_data) > 4 else None,
                    'rps': self._convert_to_number(row_data[5]) if len(row_data) > 5 else None,
                    'identif_gruppy': self._convert_to_number(row_data[6]) if len(row_data) > 6 else None,
                    'priznok_sost': self._convert_to_number(row_data[7]) if len(row_data) > 7 else None,
                    'priznok_alg': self._convert_to_number(row_data[8]) if len(row_data) > 8 else None,
                    'date_start': row_data[-2] if len(row_data) >= 2 and row_data[-2] else '',
                    'date_end': row_data[-1] if len(row_data) >= 1 and row_data[-1] else ''
                }
            }
            
            return norm_data
            
        except Exception as e:
            logger.error(f"Ошибка парсинга строки нормы: {e}")
            return None
    
    def _extract_norm_id_from_link(self, href: str) -> str:
        """Извлекает номер нормы из ссылки"""
        match = re.search(r'id=(\d+)', href)
        return match.group(1) if match else ""
    
    def _clean_text(self, text: str) -> str:
        """Очищает текст от лишних пробелов и символов"""
        if not text:
            return ""
        return text.strip().replace('\xa0', ' ').replace('&nbsp;', ' ').strip()
    
    def _convert_to_number(self, text: str):
        """Конвертирует текст в число, если возможно"""
        if not text or text.strip() == '':
            return None
        
        text = self._clean_text(text)
        
        try:
            # Сначала пробуем float
            if '.' in text or ',' in text:
                text = text.replace(',', '.')
                return float(text)
            else:
                return int(text)
        except ValueError:
            return text
    
    def compare_norms(self, new_norms: Dict[str, Dict], 
                     existing_norms: Dict[str, Dict]) -> Dict[str, str]:
        """Сравнивает новые нормы с существующими"""
        logger.info("Сравниваем новые нормы с существующими")
        
        comparison_result = {}
        
        for norm_id, new_norm in new_norms.items():
            if norm_id not in existing_norms:
                comparison_result[norm_id] = 'new'
                self.processing_stats['new_norms'] += 1
            else:
                existing_norm = existing_norms[norm_id]
                if self._norms_are_different(new_norm, existing_norm):
                    comparison_result[norm_id] = 'updated'
                    self.processing_stats['updated_norms'] += 1
                else:
                    comparison_result[norm_id] = 'unchanged'
        
        logger.info(f"Результат сравнения: новых {self.processing_stats['new_norms']}, "
                   f"обновленных {self.processing_stats['updated_norms']}")
        
        return comparison_result
    
    def _norms_are_different(self, norm1: Dict, norm2: Dict) -> bool:
        """Сравнивает две нормы на предмет различий"""
        try:
            # Сравниваем точки
            points1 = set(tuple(p) for p in norm1.get('points', []))
            points2 = set(tuple(p) for p in norm2.get('points', []))
            
            if points1 != points2:
                return True
            
            # Сравниваем базовые данные
            base1 = norm1.get('base_data', {})
            base2 = norm2.get('base_data', {})
            
            for key in base1:
                if base1.get(key) != base2.get(key):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка сравнения норм: {e}")
            return True  # В случае ошибки считаем, что нормы разные
    
    def get_processing_stats(self) -> Dict:
        """Возвращает статистику обработки"""
        return self.processing_stats.copy()
