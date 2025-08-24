# analysis/html_route_processor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

import pandas as pd
from bs4 import BeautifulSoup

# Настройка логирования
logger = logging.getLogger(__name__)


# Общие утилиты модуля: единая очистка текста и чтение файлов
def normalize_text(text: str) -> str:
    """Единая очистка текста от nbsp/мультипробелов по всему пакету analysis."""
    if not text:
        return ""
    text = text.replace('\xa0', ' ').replace('&nbsp;', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def read_text(path: str) -> Optional[str]:
    """Читает HTML как текст. Приоритет cp1251; безопасные fallbacks utf-8, utf-8-sig."""
    for enc in ('cp1251', 'utf-8', 'utf-8-sig'):
        try:
            with open(path, 'r', encoding=enc) as f:
                text = f.read()
            logger.debug("Файл %s прочитан с кодировкой %s", path, enc)
            return text
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error("Ошибка чтения файла %s: %s", path, e, exc_info=True)
            return None
    logger.warning("Не удалось корректно декодировать файл %s указанными кодировками", path)
    return None


class HTMLRouteProcessor:
    """Процессор для обработки HTML-файлов маршрутов с in-memory очисткой и парсингом."""

    def __init__(self):
        self.processed_routes: List[Dict] = []
        self.processing_stats: Dict[str, Any] = {
            'total_files': 0,
            'total_routes_found': 0,
            'unique_routes': 0,
            'duplicates_total': 0,
            'routes_with_equal_rashod': 0,
            'routes_processed': 0,
            'routes_skipped': 0,
            'output_rows': 0,
            'duplicate_details': {}
        }
        self.routes_df: Optional[pd.DataFrame] = None

    # ================== УТИЛИТЫ ==================

    def clean_text(self, text: str) -> str:
        """Очищает текст от лишних пробелов и символов (делегирует normalize_text)."""
        return normalize_text(text)

    def try_convert_to_number(self, value: Any, force_int: bool = False) -> Optional[float]:
        """Преобразует строку/число к числу; возврат только положительных значений."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        s = str(value).strip().replace(' ', '').replace('\xa0', '').replace('\u00a0', '')
        if s.endswith('.'):
            s = s[:-1]
        s = s.replace(',', '.')
        if s == '' or s.lower() in ('nan', 'none', '-'):
            return None
        try:
            num = abs(float(s))
            return int(num) if (force_int or num == int(num)) else num
        except Exception:
            return None

    def safe_subtract(self, *values) -> Optional[float]:
        """Безопасное вычитание ряда значений с пропусками; возвращает абсолютное значение."""
        valid = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
        if not valid:
            return None
        result = valid[0]
        for v in valid[1:]:
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

    # ================== ПАРСИНГ HTML ==================

    def extract_norm_url_from_href(self, href: str) -> Optional[str]:
        """Извлекает номер нормы из URL гиперссылки (id_ntp_tax=...)."""
        if not href:
            return None
        try:
            from urllib.parse import urlparse, parse_qs
            parsed = urlparse(href)
            params = parse_qs(parsed.query)
            if 'id_ntp_tax' in params:
                return params['id_ntp_tax'][0]
        except Exception as e:
            logger.debug("Ошибка парсинга URL %s: %s", href, e)
        match = re.search(r'id_ntp_tax=(\d+)', href)
        return match.group(1) if match else None

    def extract_route_header_from_html(self, html_line: str) -> Optional[Dict]:
        """Извлекает метаданные маршрута из HTML строки (th.thl_common)."""
        soup = BeautifulSoup(html_line, 'html.parser')
        header = soup.find('th', class_='thl_common')
        if not header:
            return None

        header_text = header.get_text()
        metadata: Dict[str, Any] = {}

        # Номер маршрута
        route_spans = header.find_all('font', class_='filter_value')
        if route_spans:
            metadata['number'] = self.clean_text(route_spans[0].get_text())
        else:
            for pat in (r'Маршрут\s*№[:\s]*(\d+)', r'Route\s*№[:\s]*(\d+)'):
                m = re.search(pat, header_text)
                if m:
                    metadata['number'] = m.group(1)
                    break

        # Дата маршрута
        m = re.search(r'(\d{2}\.\d{2}\.\d{4})', header_text)
        metadata['date'] = m.group(1) if m else None

        # Депо
        for pat in (r'Депо:\s*([^И]+)', r'Depot:\s*([^A-Z]+)'):
            m = re.search(pat, header_text)
            if m:
                metadata['depot'] = m.group(1).strip()
                break

        # Идентификатор
        for pat in (r'Идентификатор:\s*(\d+)', r'Identifier:\s*(\d+)'):
            m = re.search(pat, header_text)
            if m:
                metadata['identifier'] = m.group(1)
                break

        # ТУ3 (серия/номер/дата поездки/табельный)
        try:
            _, _, trip_date, driver_tab = self.extract_loco_data_from_html(html_line)
            metadata['trip_date'] = trip_date
            metadata['driver_tab'] = driver_tab
        except Exception:
            logger.debug("Не удалось извлечь дату поездки и табельный из ТУ3")

        return metadata

    def extract_loco_data_from_html(self, html_content: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Извлекает серию и номер локомотива, дату поездки и табельный номер машиниста из блока ТУ3."""
        soup = BeautifulSoup(html_content, 'html.parser')
        tu3_fonts = soup.find_all('font', class_=['itog2', 'itog3'])

        for font in tu3_fonts:
            text = font.get_text().strip()
            if text in ('ТУ3', 'TU3') or 'ТУ3' in text or 'TU3' in text:
                current = font
                data: List[str] = []
                while current:
                    current = current.find_next_sibling()
                    if current and current.name == 'font':
                        t = current.get_text().strip().replace('\xa0', ' ').replace('&nbsp;', ' ')
                        if t:
                            data.append(t)
                    elif current and current.name == 'br':
                        break
                    elif not current:
                        break

                logger.debug("ТУ3 данные: %s", data[:10])

                if len(data) >= 7:
                    # Исходная логика проекта: меняем местами серию/номер согласно формату источника
                    series_raw = data[2]            # серия локомотива
                    loco_number_raw = data[3]       # номер локомотива (строка с запятыми)
                    trip_date_raw = data[4]         # дата поездки
                    driver_tab_raw = data[5]        # табельный машиниста

                    series_part = loco_number_raw.split(',')[0] if ',' in loco_number_raw else loco_number_raw
                    series_clean = re.sub(r'[^\d]', '', series_part)
                    processed_series = series_clean[:-1] if len(series_clean) > 1 else series_clean

                    final_loco_series = series_raw
                    final_loco_number = processed_series

                    trip_date_clean = re.sub(r'[^\d]', '', trip_date_raw)
                    driver_tab_clean = re.sub(r'[^\d]', '', driver_tab_raw)
                    return final_loco_series, final_loco_number, trip_date_clean, driver_tab_clean

        logger.debug("ТУ3 не найдено")
        return None, None, None, None

    def extract_yu7_data(self, html_content: str) -> List[Tuple[int, int, int]]:
        """Извлекает данные Ю7: (НЕТТО, БРУТТО, ОСИ)."""
        soup = BeautifulSoup(html_content, 'html.parser')
        yu7_data: List[Tuple[int, int, int]] = []

        for font in soup.find_all('font'):
            ft = font.get_text().strip()
            if any(p in ft for p in ['Ю7', 'YU7', 'Yu7']):
                current = font
                data: List[str] = []
                while current:
                    current = current.find_next_sibling()
                    if current and current.name == 'font':
                        t = current.get_text().strip().replace('\xa0', ' ').replace('&nbsp;', ' ')
                        if t:
                            data.append(t)
                    elif current and current.name == 'br':
                        break
                    elif not current:
                        break

                logger.debug("Ю7 данные найдены: %s", data[:15])

                if len(data) >= 9:
                    try:
                        netto = int(data[7].strip())
                        brutto = int(data[8].strip())
                        osi = None
                        for element in data[9:]:
                            if any(p in element for p in ['ОСИ', 'OSI']):
                                m = re.search(r'(?:ОСИ|OSI)(\d+)', element)
                                if m:
                                    osi = int(m.group(1))
                                    break
                        if osi is not None and brutto > 0:
                            yu7_data.append((netto, brutto, osi))
                    except (ValueError, IndexError):
                        continue

        logger.debug("Всего найдено строк Ю7: %d", len(yu7_data))
        return yu7_data

    def find_matching_yu7(self, yu7_data: List[Tuple[int, int, int]], target_brutto: int,
                          allow_double: bool = True, tolerance_percent: float = 5.0
                          ) -> Tuple[Optional[Tuple[int, int, int]], bool, bool]:
        """Находит строку Ю7 с БРУТТО, близким к заданному (вкл. двойную тягу). Возврат: (match, is_double, is_approx)."""
        logger.debug("Поиск Ю7 для target_brutto=%s", target_brutto)

        # 1) Точное совпадение
        for netto, brutto, osi in yu7_data:
            if brutto == target_brutto:
                logger.info("✓ Найдено точное совпадение Ю7: НЕТТО=%s, БРУТТО=%s, ОСИ=%s", netto, brutto, osi)
                return (netto, brutto, osi), False, False

        # 2) Приближенное совпадение
        best_match, min_pct = None, float('inf')
        for netto, brutto, osi in yu7_data:
            diff = abs(brutto - target_brutto)
            pct = (diff / target_brutto) * 100
            if pct <= tolerance_percent and pct < min_pct:
                best_match, min_pct = (netto, brutto, osi), pct
        if best_match:
            logger.info("✓ Приближенное совпадение Ю7: target=%s, найдено=%s, отклонение=%.2f%%", target_brutto, best_match[1], min_pct)
            return best_match, False, True

        # 3) Двойная тяга
        if allow_double:
            double_target = target_brutto * 2
            for netto, brutto, osi in yu7_data:
                if brutto == double_target:
                    logger.info("✓ Точная двойная тяга Ю7: НЕТТО=%s, БРУТТО=%s, ОСИ=%s", netto, brutto, osi)
                    return (netto, brutto, osi), True, False

            best_double, min_double_pct = None, float('inf')
            for netto, brutto, osi in yu7_data:
                diff = abs(brutto - double_target)
                pct = (diff / double_target) * 100
                if pct <= tolerance_percent and pct < min_double_pct:
                    best_double, min_double_pct = (netto, brutto, osi), pct
            if best_double:
                logger.info("✓ Приближенная двойная тяга Ю7: отклонение=%.2f%%", min_double_pct)
                return best_double, True, True

        logger.warning("✗ Совпадение Ю7 не найдено для target_brutto=%s", target_brutto)
        return None, False, False

    # ================== ОБЪЕДИНЕНИЕ УЧАСТКОВ ==================

    def can_merge_sections(self, sections: List[Dict]) -> bool:
        """Можно ли объединить участки (одинаковая норма)."""
        norm_numbers: List[str] = []
        for section in sections:
            norm_num = section.get('norm_number')
            if norm_num is None and section.get('ud_norma_url'):
                norm_num = self.extract_norm_url_from_href(section.get('ud_norma_url'))
            if norm_num is not None:
                norm_numbers.append(norm_num)
        return len(set(norm_numbers)) <= 1

    def get_merged_norm_number(self, sections: List[Dict]) -> Optional[str]:
        """Возвращает номер нормы для объединенного участка."""
        for section in sections:
            if section.get('norm_number') is not None:
                return section['norm_number']
        for section in sections:
            num = self.extract_norm_url_from_href(section.get('ud_norma_url', ''))
            if num:
                return num
        return None

    def merge_sections(self, sections: List[Dict], yu7_data: List[Tuple[int, int, int]]) -> Optional[Dict]:
        """Объединяет участки с одинаковым названием и нормой."""
        if not sections:
            return None
        section_name = sections[0].get('name')
        logger.debug("Попытка объединения участка '%s': %d участков", section_name, len(sections))

        if not self.can_merge_sections(sections):
            logger.debug("Нельзя объединить участки '%s': разные номера норм", section_name)
            return None

        merged = sections[0].copy()
        # Список полей для суммирования
        sum_fields = [
            'tkm_brutto', 'km', 'rashod_fact', 'rashod_norm',
            'prostoy_vsego', 'prostoy_norma', 'manevry_vsego', 'manevry_norma',
            'troganie_vsego', 'troganie_norma', 'nagon_vsego', 'nagon_norma',
            'ogranich_vsego', 'ogranich_norma', 'peresyl_vsego', 'peresyl_norma'
        ]
        for field in sum_fields:
            total, has = 0.0, False
            for s in sections:
                v = s.get(field)
                if v is not None and v != '':
                    try:
                        total += float(v)
                        has = True
                    except (ValueError, TypeError):
                        pass
            merged[field] = total if has else None

        merged['norm_number'] = self.get_merged_norm_number(sections)

        # Подбор НЕТТО/БРУТТО/ОСИ
        tkm_brutto = merged.get('tkm_brutto')
        km = merged.get('km')
        if tkm_brutto and km and tkm_brutto > 0 and km > 0:
            target_brutto = round(tkm_brutto / km)
            match, is_double, is_appr = self.find_matching_yu7(yu7_data, target_brutto, allow_double=True)
            if match:
                netto, brutto, osi = match
                merged.update({
                    'netto': netto, 'brutto': brutto, 'osi': osi,
                    'use_red_color': is_appr, 'double_traction': "Да" if is_double else None,
                    'is_merged': True
                })
                logger.info("✓ Объединение участка '%s' успешно", section_name)
                return merged
            merged.update({'netto': "-", 'brutto': "-", 'osi': "-", 'use_red_color': True, 'double_traction': None, 'is_merged': True})
            logger.warning("✗ Для объединенного участка '%s' не найдены Ю7 данные", section_name)
            return merged

        logger.warning("✗ Недостаточно данных для объединения участка '%s'", section_name)
        return None

    def merge_identical_sections(self, norm_sections: List[Dict], station_sections: Dict[str, Dict],
                                 yu7_data: List[Tuple[int, int, int]]) -> List[Dict]:
        """Объединяет одинаковые участки в рамках одного маршрута."""
        logger.info("Начинаем объединение одинаковых участков")
        sections_by_name: Dict[str, List[Dict]] = defaultdict(list)
        for section in norm_sections:
            name = section.get('name')
            if name:
                sections_by_name[name].append(section)

        merged_sections: List[Dict] = []
        for section_name, sections in sections_by_name.items():
            if len(sections) == 1:
                merged_sections.append(sections[0])
                continue

            logger.info("Участок '%s': найдено %d одинаковых", section_name, len(sections))
            # Подмешиваем данные станций в каждый участок перед объединением
            for sec in sections:
                st = station_sections.get(section_name, {})
                sec.update({
                    'prostoy_vsego': st.get('prostoy_vsego'),
                    'prostoy_norma': st.get('prostoy_norma'),
                    'manevry_vsego': st.get('manevry_vsego'),
                    'manevry_norma': st.get('manevry_norma'),
                    'troganie_vsego': st.get('troganie_vsego'),
                    'troganie_norma': st.get('troganie_norma'),
                    'nagon_vsego': st.get('nagon_vsego'),
                    'nagon_norma': st.get('nagon_norma'),
                    'ogranich_vsego': st.get('ogranich_vsego'),
                    'ogranich_norma': st.get('ogranich_norma'),
                    'peresyl_vsego': st.get('peresyl_vsego'),
                    'peresyl_norma': st.get('peresyl_norma'),
                })
            merged = self.merge_sections(sections, yu7_data)
            merged_sections.append(merged if merged else sections[0])

        logger.info("Результат объединения: было %d участков, стало %d", len(norm_sections), len(merged_sections))
        return merged_sections

    # ================== ПАРСИНГ ТАБЛИЦ ==================

    def parse_norm_table(self, soup: BeautifulSoup) -> List[Dict]:
        """Парсит таблицу с нормируемыми участками и ключевыми параметрами."""
        result: List[Dict] = []
        for table in soup.find_all('table'):
            headers = table.find_all('th')
            header_texts = [self.clean_text(h.get_text()) for h in headers]
            if not any('Нормируемый участок' in h for h in header_texts):
                continue

            # Индексы колонок
            col_indices: Dict[str, int] = {}
            for idx, header in enumerate(header_texts):
                low = header.lower()
                if 'участок' in low and 'станция' in low:
                    col_indices['name'] = idx
                elif 'ткм брутто' in low:
                    col_indices['tkm_brutto'] = idx
                elif low in ('км', 'км.'):
                    col_indices['km'] = idx
                elif low in ('пр.', 'пр'):
                    col_indices['pr'] = idx
                elif 'расход фактический' in low:
                    col_indices['rashod_fact'] = idx
                elif 'расход по норме' in low:
                    col_indices['rashod_norm'] = idx
                elif 'уд. норма' in low or 'норма на 1 час ман. раб.' in low:
                    col_indices['ud_norma'] = idx
                elif 'норма на работу' in low:
                    col_indices['norma_rabotu'] = idx
                elif 'норма на одиночное' in low:
                    col_indices['norma_odinochnoe'] = idx

            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if not cells:
                    continue
                # Пропускаем итоговые строки
                first_cell_text = self.clean_text(cells[0].get_text())
                if 'итого' in first_cell_text.lower():
                    continue

                section: Dict[str, Any] = {}
                for field, idx in col_indices.items():
                    if idx < len(cells):
                        cell = cells[idx]
                        value = self.clean_text(cell.get_text())
                        if field == 'ud_norma':
                            link = cell.find('a')
                            if link and link.get('href'):
                                section['ud_norma_url'] = link.get('href')
                                num = self.extract_norm_url_from_href(link.get('href'))
                                if num:
                                    section['norm_number'] = num
                        section[field] = value if field == 'name' else self.try_convert_to_number(value)

                if section.get('name'):
                    result.append(section)
        return result

    def parse_station_table(self, soup: BeautifulSoup) -> Dict[str, Dict]:
        """Парсит таблицу со станциями (простой, маневры, трогания, и т.д.)."""
        station: Dict[str, Dict] = {}
        patterns = ['В том числе', 'In that number']

        for table in soup.find_all('table'):
            headers = table.find_all('th')
            header_texts = [self.clean_text(h.get_text()) for h in headers]
            if not any(any(p in h for p in patterns) for h in header_texts):
                continue

            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if not cells:
                    continue

                section_name = self.clean_text(cells[0].get_text())
                if not section_name or any(p in section_name.lower() for p in ['итого', 'total']):
                    continue

                fields = [
                    'prostoy_vsego', 'prostoy_norma',
                    'manevry_vsego', 'manevry_norma',
                    'troganie_vsego', 'troganie_norma',
                    'nagon_vsego', 'nagon_norma',
                    'ogranich_vsego', 'ogranich_norma',
                    'peresyl_vsego', 'peresyl_norma'
                ]
                data: Dict[str, Any] = {}
                for i, fld in enumerate(fields, start=1):
                    if i < len(cells):
                        data[fld] = self.try_convert_to_number(cells[i].get_text())
                station[section_name] = data

        return station

    def calculate_fact_na_rabotu(self, section: Dict, station_data: Dict) -> Optional[float]:
        """Вычисляет 'Факт на работу' как расход за вычетом нормируемых составляющих."""
        rashod_fact = section.get('rashod_fact')
        if rashod_fact is None:
            return None
        return self.safe_subtract(
            rashod_fact,
            station_data.get('prostoy_norma'),
            station_data.get('troganie_norma'),
            station_data.get('nagon_norma'),
            station_data.get('ogranich_norma'),
            station_data.get('peresyl_norma'),
        )

    def calculate_fact_ud(self, fact_na_rabotu: Optional[float], tkm_brutto: Optional[float]) -> Optional[float]:
        """Вычисляет 'Факт уд' как (факт на работу) / (ткм брутто/10000)."""
        if fact_na_rabotu is None or tkm_brutto is None:
            return None
        tkm_10000 = self.safe_divide(tkm_brutto, 10000)
        return self.safe_divide(fact_na_rabotu, tkm_10000) if tkm_10000 is not None else None

    # ================== ОЧИСТКА HTML (IN-MEMORY) ==================

    def _split_routes_to_lines(self, content: str) -> str:
        """Разбивает HTML на отдельные строки по маршрутам (для удобного последующего парсинга)."""
        logger.debug("Разбиваем маршруты по отдельным строкам...")
        route_pattern = r'(<table[^>]*><tr><th class=thl_common><font class=filter_key>\s*Маршрут\s*№:.*?<br><br><br>)'
        routes = re.findall(route_pattern, content, flags=re.DOTALL)
        if not routes:
            logger.warning("Маршруты не найдены для разделения")
            return content

        first_route_start = content.find(routes[0])
        before_routes = content[:first_route_start]
        last_route_end = content.rfind(routes[-1]) + len(routes[-1])
        after_routes = content[last_route_end:]

        result_lines = []
        if before_routes.strip():
            result_lines.append(before_routes.rstrip())
        result_lines.append("<!-- НАЧАЛО_ПЕРВОГО_МАРШРУТА -->")
        result_lines.extend(routes)
        result_lines.append("<!-- КОНЕЦ_ПОСЛЕДНЕГО_МАРШРУТА -->")
        if after_routes.strip():
            result_lines.append(after_routes.lstrip())

        logger.debug("Маршруты разделены на %d отдельных строк", len(routes))
        return '\n'.join(result_lines)

    def _remove_vcht_routes(self, content: str) -> str:
        """Удаляет строки с маршрутами, содержащими ' ВЧТ '."""
        logger.debug("Удаляем маршруты с ' ВЧТ '...")
        lines = content.split('\n')
        filtered: List[str] = []
        removed = 0
        for line in lines:
            if '<td class = itog2>" ВЧТ "</td>' in line:
                removed += 1
                continue
            filtered.append(line)
        if removed:
            logger.info("Удалено %d маршрутов с ' ВЧТ '", removed)
        return '\n'.join(filtered)

    def _clean_html_content(self, content: str) -> str:
        """Очищает HTML-код от лишних элементов (для стабильного парсинга)."""
        logger.debug("Очищаем HTML код от лишних элементов...")
        original_size = len(content)

        content = re.sub(r'<font class = rcp12 ><center>Дата получения:.*?</font>\s*<br>', '', content, flags=re.DOTALL)
        content = re.sub(r'<font class = rcp12 ><center>Номер маршрута:.*?</font><br>', '', content, flags=re.DOTALL)
        content = re.sub(r'<tr class=tr_numline>.*?</tr>', '', content, flags=re.DOTALL)
        content = re.sub(r'\s+ALIGN=center', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\s+align=left', '', content, flags=re.IGNORECASE)
        content = re.sub(r'\s+align=right', '', content, flags=re.IGNORECASE)
        content = re.sub(r'<center>', '', content)
        content = re.sub(r'</center>', '', content)
        content = re.sub(r'<pre>', '', content)
        content = re.sub(r'</pre>', '', content)
        content = re.sub(r'>[ \t]+<', '><', content)

        removed_bytes = original_size - len(content)
        logger.debug("Удалено %s байт лишнего кода (%.1f%%)", f"{removed_bytes:,}", removed_bytes / max(original_size, 1) * 100)
        return content

    # ================== ИЗВЛЕЧЕНИЕ И ОБРАБОТКА МАРШРУТОВ ==================

    def extract_routes_from_html(self, html_content: str) -> List[Tuple[str, Dict]]:
        """Извлекает маршруты из HTML-контента, возвращая пары (html_строка, метаданные)."""
        logger.info("Начинаем извлечение маршрутов из HTML")

        start_marker = "<!-- НАЧАЛО_ПЕРВОГО_МАРШРУТА -->"
        end_marker = "<!-- КОНЕЦ_ПОСЛЕДНЕГО_МАРШРУТА -->"
        start_pos = html_content.find(start_marker)
        end_pos = html_content.find(end_marker)
        routes_section = html_content if (start_pos == -1 or end_pos == -1) else html_content[start_pos + len(start_marker):end_pos]

        lines = routes_section.strip().split('\n')
        routes: List[Tuple[str, Dict]] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if re.search(r'<table width=\d+%', s) and ('Маршрут №' in s or 'Маршрут' in s):
                metadata = self.extract_route_header_from_html(s)
                if metadata:
                    routes.append((s, metadata))
                    logger.debug("Найден маршрут: №%s", metadata.get('number'))

        logger.info("Найдено маршрутов: %d", len(routes))
        return routes

    def check_rashod_equal_html(self, route_html: str) -> bool:
        """Проверяет, равны ли 'Расход по норме' и 'Расход фактический' в маршруте (хотя бы в одной строке)."""
        soup = BeautifulSoup(route_html, 'html.parser')
        for table in soup.find_all('table', width='90%'):
            headers = table.find_all('th')
            header_texts = [self.clean_text(h.get_text()) for h in headers]
            if not any('Нормируемый участок' in h for h in header_texts):
                continue

            rashod_fact_idx = rashod_norm_idx = None
            for idx, header in enumerate(header_texts):
                if 'Расход фактический' in header:
                    rashod_fact_idx = idx
                elif 'Расход по норме' in header:
                    rashod_norm_idx = idx

            if rashod_fact_idx is None or rashod_norm_idx is None:
                continue

            for row in table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) > max(rashod_fact_idx, rashod_norm_idx):
                    fact_val = self.try_convert_to_number(self.clean_text(cells[rashod_fact_idx].get_text()))
                    norm_val = self.try_convert_to_number(self.clean_text(cells[rashod_norm_idx].get_text()))
                    if fact_val is not None and norm_val is not None and abs(fact_val - norm_val) < 0.01:
                        return True
        return False

    def select_best_route(self, routes: List[Tuple[str, Dict]]) -> Optional[Tuple[str, Dict]]:
        """Выбирает лучший вариант маршрута из дубликатов: сначала без равных расходов, затем с макс. идентификатором."""
        if not routes:
            return None

        valid, equal = [], []
        for route_html, meta in routes:
            (equal if self.check_rashod_equal_html(route_html) else valid).append((route_html, meta))

        if valid:
            return max(valid, key=lambda x: int(x[1].get('identifier') or 0))
        if equal:
            return max(equal, key=lambda x: int(x[1].get('identifier') or 0))
        return None

    def parse_html_route(self, route_html: str, metadata: Dict,
                         has_equal_duplicates: bool = False,
                         rashod_equal: bool = False) -> List[Dict]:
        """Главная функция парсинга одного маршрута: собирает строки участков."""
        soup = BeautifulSoup(route_html, 'html.parser')

        route_number = metadata.get('number')
        route_date = metadata.get('date')
        depot = metadata.get('depot', '')
        identifier = metadata.get('identifier')

        loco_series, loco_number, trip_date, driver_tab = self.extract_loco_data_from_html(route_html)

        yu7_data = self.extract_yu7_data(route_html)
        default_netto = yu7_data[0][0] if yu7_data else None
        default_brutto = yu7_data[0][1] if yu7_data else None
        default_osi = yu7_data[0][2] if yu7_data else None

        norm_sections = self.parse_norm_table(soup)
        station_sections = self.parse_station_table(soup)
        merged_sections = self.merge_identical_sections(norm_sections, station_sections, yu7_data)

        rows: List[Dict] = []
        for section in merged_sections:
            name = section.get('name')
            if not name:
                continue

            station_data = {} if section.get('is_merged') else station_sections.get(name, {})

            norm_number = section.get('norm_number')
            if not norm_number and section.get('ud_norma_url'):
                norm_number = self.extract_norm_url_from_href(section.get('ud_norma_url'))

            netto, brutto, osi = section.get('netto'), section.get('brutto'), section.get('osi')
            use_red_color = section.get('use_red_color', False)
            double_traction = section.get('double_traction')

            if netto is None or brutto is None or osi is None:
                tkm_brutto = section.get('tkm_brutto')
                km = section.get('km')
                if tkm_brutto and km and tkm_brutto > 0 and km > 0:
                    target_brutto = round(tkm_brutto / km)
                    match, is_double, is_appr = self.find_matching_yu7(yu7_data, target_brutto, allow_double=True)
                    if match:
                        netto, brutto, osi = match
                        use_red_color = is_appr
                        if is_double:
                            double_traction = "Да"
                    else:
                        netto, brutto, osi, use_red_color = "-", "-", "-", True
                else:
                    netto, brutto, osi = default_netto, default_brutto, default_osi

            axle_load = None
            if (brutto and osi and brutto != "-" and osi != "-" and not use_red_color
                    and isinstance(brutto, (int, float)) and isinstance(osi, (int, float))):
                axle_load = brutto / osi
            elif brutto == "-" or osi == "-":
                axle_load = "-"

            fact_na_rabotu = self.calculate_fact_na_rabotu(section, station_data if not section.get('is_merged') else section)
            fact_ud = self.calculate_fact_ud(fact_na_rabotu, section.get('tkm_brutto'))

            row = {
                'Номер маршрута': route_number,
                'Дата маршрута': route_date,
                'Дата поездки': trip_date,
                'Табельный машиниста': driver_tab,
                'Депо': depot,
                'Идентификатор': identifier,
                'Серия локомотива': loco_series,
                'Номер локомотива': loco_number,
                'НЕТТО': netto,
                'БРУТТО': brutto,
                'ОСИ': osi,
                'USE_RED_COLOR': use_red_color,
                'USE_RED_RASHOD': rashod_equal,
                'Наименование участка': name,
                'Номер нормы': norm_number,
                'Дв. тяга': double_traction,
                'Ткм брутто': section.get('tkm_brutto'),
                'Км': section.get('km'),
                'Пр.': section.get('pr'),
                'Расход фактический': section.get('rashod_fact'),
                'Расход по норме': section.get('rashod_norm'),
                'Уд. норма, норма на 1 час ман. раб.': section.get('ud_norma'),
                'Нажатие на ось': axle_load,
                'Норма на работу': section.get('norma_rabotu'),
                'Факт уд': fact_ud,
                'Факт на работу': fact_na_rabotu,
                'Норма на одиночное': section.get('norma_odinochnoe'),
                'Простой с бригадой, мин., всего': section.get('prostoy_vsego') if section.get('is_merged') else station_data.get('prostoy_vsego'),
                'Простой с бригадой, мин., норма': section.get('prostoy_norma') if section.get('is_merged') else station_data.get('prostoy_norma'),
                'Маневры, мин., всего': section.get('manevry_vsego') if section.get('is_merged') else station_data.get('manevry_vsego'),
                'Маневры, мин., норма': section.get('manevry_norma') if section.get('is_merged') else station_data.get('manevry_norma'),
                'Трогание с места, случ., всего': section.get('troganie_vsego') if section.get('is_merged') else station_data.get('troganie_vsego'),
                'Трогание с места, случ., норма': section.get('troganie_norma') if section.get('is_merged') else station_data.get('troganie_norma'),
                'Нагон опозданий, мин., всего': section.get('nagon_vsego') if section.get('is_merged') else station_data.get('nagon_vsego'),
                'Нагон опозданий, мин., норма': section.get('nagon_norma') if section.get('is_merged') else station_data.get('nagon_norma'),
                'Ограничения скорости, случ., всего': section.get('ogranich_vsego') if section.get('is_merged') else station_data.get('ogranich_vsego'),
                'Ограничения скорости, случ., норма': section.get('ogranich_norma') if section.get('is_merged') else station_data.get('ogranich_norma'),
                'На пересылаемые л-вы, всего': section.get('peresyl_vsego') if section.get('is_merged') else station_data.get('peresyl_vsego'),
                'На пересылаемые л-вы, норма': section.get('peresyl_norma') if section.get('is_merged') else station_data.get('peresyl_norma'),
                'Количество дубликатов маршрута': '',
                'Н=Ф': 'Да' if has_equal_duplicates else None
            }
            rows.append(row)

        return rows

    # ================== ГЛАВНАЯ ФУНКЦИЯ ОБРАБОТКИ ==================

    def process_html_files(self, html_files: List[str]) -> pd.DataFrame:
        """Обрабатывает список HTML-файлов в памяти (без временных файлов)."""
        logger.info("Начинаем обработку %d HTML файлов", len(html_files))

        # Сброс статистики партии
        self.processing_stats.update({
            'total_files': len(html_files),
            'total_routes_found': 0,
            'unique_routes': 0,
            'duplicates_total': 0,
            'routes_with_equal_rashod': 0,
            'routes_processed': 0,
            'routes_skipped': 0,
            'output_rows': 0,
            'duplicate_details': {}
        })

        all_rows: List[Dict] = []

        for file_path in html_files:
            logger.info("Обработка файла: %s", os.path.basename(file_path))
            try:
                content = read_text(file_path)
                if content is None:
                    logger.error("Не удалось прочитать файл %s", file_path)
                    self.processing_stats['routes_skipped'] += 1
                    continue

                # Границы основного блока формы — маркеры гарантированы
                start_marker = '<table align=center width="100%">'
                start_pos = content.find(start_marker)
                if start_pos == -1:
                    logger.error("Начальный маркер не найден в файле %s", file_path)
                    self.processing_stats['routes_skipped'] += 1
                    continue

                form_pattern = r'</table>\s*</td>\s*</tr></table><form id=print_form>.*?(?=\n|$)'
                form_match = re.search(form_pattern, content[start_pos:], re.DOTALL)
                if not form_match:
                    logger.error("Конечный маркер не найден в файле %s", file_path)
                    self.processing_stats['routes_skipped'] += 1
                    continue

                form_end_pos = start_pos + form_match.end()
                remaining = content[form_end_pos:]
                end_pos = form_end_pos
                lines = remaining.split('\n')
                for i, line in enumerate(lines):
                    if '</td></tr>' in line.strip():
                        end_pos = form_end_pos + len('\n'.join(lines[:i + 1]))
                        break

                # Очистка в памяти: выделение блока → разбиение → фильтрация → чистка
                extracted = content[start_pos:end_pos + 1]
                extracted = self._split_routes_to_lines(extracted)
                extracted = self._remove_vcht_routes(extracted)
                extracted = self._clean_html_content(extracted)

                # Парсинг маршрутов и статистика
                df, stats = self._process_routes(extracted)
                # Агрегируем статистику
                for k in ('total_routes_found', 'unique_routes', 'duplicates_total',
                          'routes_with_equal_rashod', 'routes_processed', 'routes_skipped', 'output_rows'):
                    self.processing_stats[k] += stats.get(k, 0)

                if not df.empty:
                    all_rows.extend(df.to_dict('records'))

            except Exception as e:
                logger.error("Ошибка при обработке файла %s: %s", file_path, e, exc_info=True)
                self.processing_stats['routes_skipped'] += 1

        if all_rows:
            result_df = pd.DataFrame(all_rows)
            logger.info("Обработка завершена. Получено %d итоговых записей", len(result_df))
            self.routes_df = result_df
            return result_df

        logger.warning("Не получено ни одной записи из всех файлов")
        self.routes_df = pd.DataFrame()
        return self.routes_df

    def _process_routes(self, html_content: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Обрабатывает все маршруты из очищенного HTML-контента."""
        routes = self.extract_routes_from_html(html_content)

        stats = {
            'total_routes_found': len(routes),
            'unique_routes': 0,
            'duplicates_total': 0,
            'routes_with_equal_rashod': 0,
            'routes_skipped': 0,
            'routes_processed': 0,
            'output_rows': 0,
            'duplicate_details': {}
        }
        if not routes:
            logger.error("Маршруты не найдены в HTML")
            return pd.DataFrame(), stats

        # Группируем маршруты: номер + дата поездки + табельный
        route_groups: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)
        skipped_by_yu6 = 0

        for route_html, metadata in routes:
            if self.check_yu6_filter(route_html):
                skipped_by_yu6 += 1
                stats['routes_skipped'] += 1
                continue

            if metadata.get('number') and metadata.get('trip_date') and metadata.get('driver_tab'):
                key = f"{metadata['number']}_{metadata['trip_date']}_{metadata['driver_tab']}"
                route_groups[key].append((route_html, metadata))
            else:
                stats['routes_skipped'] += 1

        if skipped_by_yu6:
            logger.info("Пропущено %d маршрутов по фильтру Ю6", skipped_by_yu6)

        stats['unique_routes'] = len(route_groups)
        all_rows: List[Dict] = []
        duplicate_counts: Dict[str, int] = {}

        for key, group in route_groups.items():
            logger.info("Обработка маршрута %s, версий: %d", key, len(group))

            if len(group) > 1:
                duplicate_counts[key] = len(group) - 1
                stats['duplicates_total'] += len(group) - 1
                stats['duplicate_details'][key] = {
                    'versions': len(group),
                    'duplicates': len(group) - 1,
                    'identifiers': [g[1].get('identifier') for g in group]
                }

            equal_count = sum(1 for r in group if self.check_rashod_equal_html(r[0]))
            if equal_count > 0:
                stats['routes_with_equal_rashod'] += equal_count

            best_route = self.select_best_route(group)
            if not best_route:
                stats['routes_skipped'] += 1
                continue

            route_html, meta = best_route
            has_equal_duplicates = len(group) > 1 and any(self.check_rashod_equal_html(r[0]) for r in group)
            rashod_equal = self.check_rashod_equal_html(route_html)

            try:
                route_rows = self.parse_html_route(route_html, meta, has_equal_duplicates, rashod_equal)
                if key in duplicate_counts and route_rows:
                    route_rows[0]['Количество дубликатов маршрута'] = duplicate_counts[key]
                all_rows.extend(route_rows)
                stats['routes_processed'] += 1
                stats['output_rows'] += len(route_rows)
            except Exception as e:
                logger.error("Ошибка обработки маршрута %s: %s", key, e, exc_info=True)
                stats['routes_skipped'] += 1

        if all_rows:
            df = pd.DataFrame(all_rows)
            logger.info("Создан DataFrame с %d строками", len(df))
            return df, stats

        logger.warning("Нет данных для создания DataFrame")
        return pd.DataFrame(), stats

    # ================== ПУБЛИЧНЫЕ УТИЛИТЫ/ЭКСПОРТ ==================

    def get_processing_stats(self) -> Dict:
        """Возвращает статистику обработки."""
        return self.processing_stats.copy()

    def get_sections_list(self) -> List[str]:
        """Возвращает список участков из обработанных данных."""
        if self.routes_df is None or self.routes_df.empty:
            return []
        sections = self.routes_df['Наименование участка'].dropna().unique().tolist()
        logger.debug("Найдено участков: %d", len(sections))
        return sorted(sections)

    def get_section_data(self, section_name: str) -> pd.DataFrame:
        """Возвращает данные по конкретному участку."""
        if self.routes_df is None or self.routes_df.empty:
            return pd.DataFrame()
        return self.routes_df[self.routes_df['Наименование участка'] == section_name].copy()

    def get_norms_for_section(self, section_name: str) -> List[str]:
        """Возвращает список номеров норм для участка."""
        section_data = self.get_section_data(section_name)
        if section_data.empty:
            return []
        norms = section_data['Номер нормы'].dropna().unique().tolist()
        logger.debug("Нормы для участка '%s': %s", section_name, norms)
        return sorted([str(n) for n in norms])

    def export_to_excel(self, df: pd.DataFrame, output_file: str) -> bool:
        """Экспортирует данные в Excel с форматированием и подсветкой."""
        if df.empty:
            logger.warning("DataFrame пуст, нечего экспортировать")
            return False

        try:
            columns = [
                'Номер маршрута', 'Дата маршрута', 'Дата поездки', 'Табельный машиниста',
                'Серия локомотива', 'Номер локомотива',
                'НЕТТО', 'БРУТТО', 'ОСИ', 'Наименование участка', 'Номер нормы', 'Дв. тяга',
                'Ткм брутто', 'Км', 'Пр.', 'Расход фактический',
                'Расход по норме', 'Уд. норма, норма на 1 час ман. раб.', 'Нажатие на ось',
                'Норма на работу', 'Факт уд', 'Факт на работу', 'Норма на одиночное',
                'Простой с бригадой, мин., всего', 'Простой с бригадой, мин., норма',
                'Маневры, мин., всего', 'Маневры, мин., норма',
                'Трогание с места, случ., всего', 'Трогание с места, случ., норма',
                'Нагон опозданий, мин., всего', 'Нагон опозданий, мин., норма',
                'Ограничения скорости, случ., всего', 'Ограничения скорости, случ., норма',
                'На пересылаемые л-вы, всего', 'На пересылаемые л-вы, норма',
                'Количество дубликатов маршрута', 'Н=Ф',
                'USE_RED_COLOR', 'USE_RED_RASHOD'
            ]
            existing = [c for c in columns if c in df.columns]
            df_excel = df[existing].copy()

            # Преобразуем дату
            if 'Дата маршрута' in df_excel.columns:
                df_excel['Дата маршрута'] = pd.to_datetime(df_excel['Дата маршрута'], format='%d.%m.%Y', errors='coerce')

            # Логирование подсветок
            if 'USE_RED_COLOR' in df_excel.columns:
                logger.info("Строк с красным по НЕТТО/БРУТТО/ОСИ: %s", int(df_excel['USE_RED_COLOR'].fillna(False).sum()))
            if 'USE_RED_RASHOD' in df_excel.columns:
                logger.info("Строк с красным по расходам: %s", int(df_excel['USE_RED_RASHOD'].fillna(False).sum()))

            # Убираем служебные колонки для отображения, но оставляем их для форматирования
            display_cols = [c for c in existing if c not in ('USE_RED_COLOR', 'USE_RED_RASHOD')]
            df_display = df_excel[display_cols].copy()

            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df_display.to_excel(writer, index=False, sheet_name='Маршруты')
                ws = writer.sheets['Маршруты']
                self._apply_excel_formatting(ws, df_excel)

            logger.info("Данные экспортированы в %s", output_file)
            return True
        except Exception as e:
            logger.error("Ошибка экспорта в Excel: %s", e, exc_info=True)
            return False

    def _apply_excel_formatting(self, ws, df: pd.DataFrame) -> None:
        """Применяет форматирование к листу Excel: границы, подсветка, форматы, автоширина."""
        from openpyxl.utils import get_column_letter
        from openpyxl.styles import Font, Border, Side, PatternFill

        thin = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
        thick = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thick'))

        red_fill = PatternFill(start_color='FFCCCC', end_color='FFCCCC', fill_type='solid')
        red_font = Font(color='FF0000', bold=True)

        # Индексы колонок по названию
        col_idx: Dict[str, int] = {cell.value: i for i, cell in enumerate(ws[1], start=1)}
        netto_col = col_idx.get('НЕТТО')
        brutto_col = col_idx.get('БРУТТО')
        osi_col = col_idx.get('ОСИ')
        rf_col = col_idx.get('Расход фактический')
        rn_col = col_idx.get('Расход по норме')

        for r in range(2, ws.max_row + 1):
            route_num = ws.cell(row=r, column=1).value
            next_route = ws.cell(row=r + 1, column=1).value if r < ws.max_row else None

            dfi = r - 2  # индекс в df_excel (без заголовка)
            use_red_color = bool(df.iloc[dfi].get('USE_RED_COLOR', False)) if dfi < len(df) else False
            use_red_rashod = bool(df.iloc[dfi].get('USE_RED_RASHOD', False)) if dfi < len(df) else False

            for c in range(1, ws.max_column + 1):
                cell = ws.cell(row=r, column=c)
                cell.border = thick if route_num != next_route else thin

                if use_red_color and c in [netto_col, brutto_col, osi_col] and c is not None:
                    cell.fill = red_fill
                    cell.font = red_font
                if use_red_rashod and c in [rf_col, rn_col] and c is not None:
                    cell.fill = red_fill
                    cell.font = red_font

                if c > 4 and cell.value is not None and isinstance(cell.value, (int, float)) and cell.value != "-":
                    cell.number_format = '#,##0' if (isinstance(cell.value, int) or cell.value == int(cell.value)) else '#,##0.000'

        # Форматирование даты
        date_col = col_idx.get('Дата маршрута')
        if date_col:
            for r in range(2, ws.max_row + 1):
                dc = ws.cell(row=r, column=date_col)
                if dc.value:
                    dc.number_format = 'DD.MM.YYYY'

        # Автоширина
        for column in ws.columns:
            letter = get_column_letter(column[0].column)
            max_len = max((len(str(c.value)) if c.value is not None else 0) for c in column)
            ws.column_dimensions[letter].width = min(max_len + 2, 50)

    def check_yu6_filter(self, route_html: str) -> bool:
        """Фильтрует маршруты с Ю6, содержащим паттерн '1 2 ,0' или '1 3 ,0'."""
        soup = BeautifulSoup(route_html, 'html.parser')
        for font in soup.find_all('font'):
            ft = font.get_text().strip()
            if ft.startswith('Ю6'):
                current = font
                data: List[str] = []
                while current:
                    current = current.find_next_sibling()
                    if current and current.name == 'font':
                        t = current.get_text().strip().replace('\xa0', ' ').replace('&nbsp;', ' ')
                        if t:
                            data.append(t)
                    elif current and current.name == 'br':
                        break
                    elif not current:
                        break
                data_str = ' '.join(data)
                if re.search(r'1\s+2\s+,0', data_str) or re.search(r'1\s+3\s+,0', data_str):
                    logger.info("Маршрут с Ю6 ('1 2 ,0' или '1 3 ,0') — пропущен")
                    return True
        return False