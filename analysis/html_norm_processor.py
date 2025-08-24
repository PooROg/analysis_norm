# analysis/html_norm_processor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import re
from pathlib import Path
from typing import List, Dict, Optional

from bs4 import BeautifulSoup
from analysis.html_route_processor import normalize_text, read_text  # переиспользуем общие утилиты

# Настройка логирования
logger = logging.getLogger(__name__)


class HTMLNormProcessor:
    """Процессор для обработки HTML файлов норм.

    Сохраняем публичные методы, используемые внешним кодом:
    - process_html_files
    - compare_norms
    - _norms_are_different
    - get_processing_stats
    """

    def __init__(self):
        self.processed_norms: Dict[str, Dict] = {}
        self.processing_stats: Dict[str, int] = {
            'total_files': 0,
            'total_norms_found': 0,
            'new_norms': 0,
            'updated_norms': 0,
            'skipped_norms': 0
        }

    # --------------------------- Публичное API ---------------------------

    def process_html_files(self, html_files: List[str]) -> Dict[str, Dict]:
        """Обрабатывает список HTML файлов норм (очистка выполняется в памяти)."""
        logger.info("Начинаем обработку %d HTML файлов норм", len(html_files))

        # Сбрасываем статистику на запуск
        self.processing_stats.update({
            'total_files': len(html_files),
            'total_norms_found': 0,
            'new_norms': 0,
            'updated_norms': 0,
            'skipped_norms': 0
        })

        all_norms: Dict[str, Dict] = {}

        for file_path in html_files:
            logger.info("Обработка файла норм: %s", Path(file_path).name)
            try:
                html_content = self._read_text_with_fallbacks(file_path)
                if html_content is None:
                    logger.error("Не удалось прочитать файл %s", file_path)
                    continue

                # Очистка — строго по паттернам, но в памяти
                cleaned_html = self._clean_html_content(html_content)

                # Парсим уже очищенный HTML — как в проверенной версии
                norms_from_file = self._extract_norms_from_cleaned_html(cleaned_html)

                # Объединяем нормы
                for norm_id, norm_data in norms_from_file.items():
                    if norm_id in all_norms:
                        logger.warning("Норма %s уже существует, перезаписываем", norm_id)
                    all_norms[norm_id] = norm_data

            except Exception as e:
                logger.error("Ошибка при обработке файла норм %s: %s", file_path, e, exc_info=True)
                continue

        self.processing_stats['total_norms_found'] = len(all_norms)
        logger.info("Обработка завершена. Найдено норм: %d", len(all_norms))
        return all_norms

    def compare_norms(self, new_norms: Dict[str, Dict],
                      existing_norms: Dict[str, Dict]) -> Dict[str, str]:
        """Сравнивает новые нормы с существующими и обновляет статистику."""
        logger.info("Сравниваем новые нормы с существующими")

        # Обнуляем счетчики сравнений
        self.processing_stats['new_norms'] = 0
        self.processing_stats['updated_norms'] = 0

        comparison_result: Dict[str, str] = {}
        for norm_id, new_norm in new_norms.items():
            if norm_id not in existing_norms:
                comparison_result[norm_id] = 'new'
                self.processing_stats['new_norms'] += 1
            else:
                if self._norms_are_different(new_norm, existing_norms[norm_id]):
                    comparison_result[norm_id] = 'updated'
                    self.processing_stats['updated_norms'] += 1
                else:
                    comparison_result[norm_id] = 'unchanged'

        logger.info("Результат сравнения: новых %d, обновленных %d",
                    self.processing_stats['new_norms'], self.processing_stats['updated_norms'])
        return comparison_result

    def _norms_are_different(self, norm1: Dict, norm2: Dict) -> bool:
        """Сравнивает две нормы на предмет различий (по точкам и base_data)."""
        try:
            points1 = set(tuple(p) for p in norm1.get('points', []))
            points2 = set(tuple(p) for p in norm2.get('points', []))
            if points1 != points2:
                return True

            base1 = norm1.get('base_data', {})
            base2 = norm2.get('base_data', {})
            for key in base1:
                if base1.get(key) != base2.get(key):
                    return True

            return False
        except Exception as e:
            logger.error("Ошибка сравнения норм: %s", e, exc_info=True)
            return True  # консервативный подход: при ошибке считаем разными

    def get_processing_stats(self) -> Dict:
        """Возвращает копию статистики обработки."""
        return self.processing_stats.copy()

    # --------------------------- Очистка (в памяти) ---------------------------

    def _clean_html_content(self, html_content: str) -> str:
        """Очищает HTML от лишнего, оставляя только искомые секции с таблицами норм."""
        pattern1 = r'(<font class=rcp12><center><b>Удельные нормы электроэнергии и топлива по нагрузке на ось</b></center></font>.*?</table>.*?</table>)'
        pattern2 = r'(<font class=rcp12><center><b>Удельные нормы электроэнергии и топлива по весу поезда</b></center></font>.*?</table>.*?</table>)'

        match1 = re.search(pattern1, html_content, re.DOTALL | re.IGNORECASE)
        match2 = re.search(pattern2, html_content, re.DOTALL | re.IGNORECASE)

        new_html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '    <meta charset="utf-8">',
            "    <title>Удельные нормы электроэнергии и топлива</title>",
            "    <style>",
            "        body { font-family: Arial, sans-serif; margin: 20px; }",
            "        .rcp12 { font-size: 12px; }",
            "        .filter_key { font-weight: bold; }",
            "        .filter_value { color: blue; }",
            "        .tr_head { background-color: #e0e0e0; }",
            "        .thc { border: 1px solid #000; padding: 5px; text-align: center; }",
            "        .tdc_str1, .tdc_str2 { border: 1px solid #000; padding: 3px; text-align: center; }",
            "        .tdc_str1 { background-color: #f9f9f9; }",
            "        .tdc_str2 { background-color: #ffffff; }",
            "        table { border-collapse: collapse; margin: 20px auto; }",
            "        .link { color: blue; text-decoration: underline; }",
            "    </style>",
            "</head>",
            "<body>",
        ]

        if match1:
            new_html_parts.append(match1.group(1) + "<br><br>")
            logger.debug("✓ Найдена таблица 'по нагрузке на ось'")
        if match2:
            new_html_parts.append(match2.group(1) + "<br><br>")
            logger.debug("✓ Найдена таблица 'по весу поезда'")

        if not match1 and not match2:
            logger.warning("Искомые таблицы не найдены в файле!")
            new_html_parts.append("<h1>Искомые таблицы не найдены в файле</h1>")

        new_html_parts.extend(["</body>", "</html>"])
        return "\n".join(new_html_parts)

    # --------------------------- Извлечение норм ---------------------------

    def _extract_norms_from_cleaned_html(self, cleaned_html: str) -> Dict[str, Dict]:
        """Извлекает нормы из очищенного HTML."""
        soup = BeautifulSoup(cleaned_html, 'html.parser')

        norms_data: Dict[str, Dict] = {}

        # Обработка таблицы по нагрузке на ось
        load_section = self._find_section_by_text(soup, 'нагрузке на ось')
        if load_section:
            load_norms = self._extract_norms_from_section(load_section, 'Нажатие')
            norms_data.update(load_norms)
            logger.debug("Найдено норм по нагрузке на ось: %d", len(load_norms))

        # Обработка таблицы по весу поезда
        weight_section = self._find_section_by_text(soup, 'весу поезда')
        if weight_section:
            weight_norms = self._extract_norms_from_section(weight_section, 'Вес')
            norms_data.update(weight_norms)
            logger.debug("Найдено норм по весу поезда: %d", len(weight_norms))

        logger.debug("Всего извлечено норм: %d", len(norms_data))
        return norms_data

    def _find_section_by_text(self, soup: BeautifulSoup, search_text: str):
        """Находит секцию по тексту заголовка (как в исходном коде)."""
        for element in soup.find_all(text=True):
            if search_text in element:
                return element.parent
        return None

    def _extract_norms_from_section(self, section, norm_type: str) -> Dict[str, Dict]:
        """Извлекает нормы из секции: ближайшая таблица с tr.tr_head, numeric_start=9..len(headers)-2."""
        norms: Dict[str, Dict] = {}
        if not section:
            return norms

        current = section.parent
        for sibling in current.find_all_next('table'):
            if sibling.find('tr', class_='tr_head'):
                headers = self._get_table_headers(sibling)
                rows = sibling.find_all('tr')[1:]  # Пропускаем заголовок

                numeric_start = 9  # как в исходной логике
                numeric_end = len(headers) - 2  # до колонок с датами

                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) > 10:
                        norm_data = self._parse_norm_row(cells, headers, norm_type, numeric_start, numeric_end)
                        if norm_data and norm_data.get('norm_id'):
                            norms[norm_data['norm_id']] = norm_data
                        else:
                            self.processing_stats['skipped_norms'] += 1
                break

        return norms

    def _get_table_headers(self, table) -> List[str]:
        """Получает тексты заголовков из tr.tr_head."""
        headers: List[str] = []
        header_row = table.find('tr', class_='tr_head')
        if header_row:
            for th in header_row.find_all('th'):
                headers.append(self._clean_text(th.get_text()))
        return headers

    def _parse_norm_row(self, cells, headers: List[str], norm_type: str,
                        numeric_start: int, numeric_end: int) -> Optional[Dict]:
        """Парсит строку нормы: ищет id в ссылке и собирает точки между numeric_start..numeric_end."""
        try:
            row_data: List[str] = []
            norm_id = ""

            for i, cell in enumerate(cells):
                cell_text = self._clean_text(cell.get_text())
                row_data.append(cell_text)

                if i == 0:  # Первая колонка может содержать ссылку с id
                    link = cell.find('a')
                    if link and link.get('href'):
                        norm_id = self._extract_norm_id_from_link(link['href'])

            if not norm_id or len(row_data) <= 10:
                return None

            # Извлекаем числовые данные
            numeric_data: Dict[float, float] = {}
            upper = min(numeric_end, len(row_data), len(headers))
            for i in range(numeric_start, upper):
                if i < len(row_data) and row_data[i].strip():
                    header_value = self._clean_text(headers[i])
                    try:
                        load_value = float(header_value.replace(',', '.'))
                        consumption_value = float(row_data[i].replace(',', '.'))
                        numeric_data[load_value] = consumption_value
                    except ValueError:
                        continue

            if not numeric_data:
                return None

            # Базовые данные (позиционные поля)
            base_data = {
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

            points = [(load, cons) for load, cons in sorted(numeric_data.items())]
            return {
                'norm_id': norm_id,
                'norm_type': norm_type,
                'description': f'Норма №{norm_id} ({norm_type})',
                'points': points,
                'base_data': base_data
            }

        except Exception as e:
            logger.error("Ошибка парсинга строки нормы: %s", e, exc_info=True)
            return None

    # --------------------------- Утилиты ---------------------------

    def _read_text_with_fallbacks(self, path: str) -> Optional[str]:
        """Чтение HTML: приоритет cp1251, fallbacks utf-8/utf-8-sig (делегирует read_text)."""
        return read_text(path)

    def _extract_norm_id_from_link(self, href: str) -> str:
        """Извлекает номер нормы из ссылки (параметр id=...)."""
        match = re.search(r'id=(\d+)', href)
        return match.group(1) if match else ""

    def _clean_text(self, text: str) -> str:
        """Единая очистка текста (делегирование normalize_text)."""
        return normalize_text(text)

    def _convert_to_number(self, text: str):
        """Конвертирует текст в число (float/int), если возможно, иначе возвращает строку."""
        if not text or text.strip() == '':
            return None
        t = normalize_text(text)
        try:
            if ',' in t or '.' in t:
                return float(t.replace(',', '.'))
            return int(t)
        except ValueError:
            return t