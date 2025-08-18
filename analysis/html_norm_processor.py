# analysis/html_norm_processor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированный процессор HTML файлов норм с использованием современных возможностей Python 3.12.
Высокопроизводительная обработка норм с кэшированием и векторизацией.
"""

from __future__ import annotations
import re
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Iterator
from collections import defaultdict
import pandas as pd
import numpy as np
import logging

# Импорт оптимизированной альтернативы BeautifulSoup (в будущем можно заменить на selectolax)
from bs4 import BeautifulSoup

from .data_models import NormDefinition, ValidationResult, ProcessingStats
from .utils import TextCleaner, URLExtractor, HTMLCleaner, file_reader, temporary_file

logger = logging.getLogger(__name__)

type HTMLContent = str
type NormPoints = list[tuple[float, float]]
type NormDict = dict[str, any]

@dataclass(slots=True)
class NormProcessingConfig:
    """Конфигурация обработки норм с оптимизацией памяти."""
    min_work_threshold: float = 0.0
    max_points_per_norm: int = 50
    enable_validation: bool = True
    cache_size: int = 1000
    batch_size: int = 50

@dataclass(slots=True)
class NormProcessingResult:
    """Результат обработки норм."""
    processed_count: int = 0
    skipped_count: int = 0
    error_count: int = 0
    validation_errors: list[str] = field(default_factory=list)

class OptimizedHTMLNormProcessor:
    """
    Оптимизированный процессор HTML файлов норм.
    
    Ключевые улучшения:
    - Кэширование обработанных норм
    - Векторизованная валидация
    - Оптимизированный парсинг HTML
    - Современные структуры данных
    - Эффективное управление памятью
    """
    
    def __init__(self, config: Optional[NormProcessingConfig] = None):
        """Инициализирует процессор с конфигурацией."""
        self.config = config or NormProcessingConfig()
        self.norms_cache: dict[str, NormDefinition] = {}
        self.processing_stats: ProcessingStats = defaultdict(int)
        
        # Предкомпилированные регулярные выражения для оптимизации
        self._compile_patterns()
        
        logger.info("Optimized HTML norm processor initialized")
    
    def _compile_patterns(self) -> None:
        """Предкомпилирует регулярные выражения."""
        self.patterns = {
            'norm_table_1': re.compile(
                r'(<font class=rcp12><center><b>Удельные нормы электроэнергии и топлива по нагрузке на ось</b></center></font>.*?</table>.*?</table>)',
                re.DOTALL | re.IGNORECASE
            ),
            'norm_table_2': re.compile(
                r'(<font class=rcp12><center><b>Удельные нормы электроэнергии и топлива по весу поезда</b></center></font>.*?</table>.*?</table>)',
                re.DOTALL | re.IGNORECASE
            ),
            'series_from_sheet': re.compile(r'[А-ЯA-Z]+[\d]+[А-ЯA-Z]*'),
            'norm_id_link': re.compile(r'id=(\d+)'),
            'zavod_number': re.compile(r'завод.*?номер', re.IGNORECASE),
            'percent_value': re.compile(r'процент|%', re.IGNORECASE)
        }
    
    def process_html_files(self, html_files: list[Path]) -> bool:
        """
        Обрабатывает список HTML файлов норм с оптимизацией.
        
        Args:
            html_files: Список путей к HTML файлам норм
            
        Returns:
            True если обработка успешна
        """
        logger.info(f"Processing {len(html_files)} HTML norm files (optimized)")
        
        self.processing_stats.clear()
        self.processing_stats['total_files'] = len(html_files)
        
        total_processed = 0
        
        # Обрабатываем файлы батчами
        for batch in self._batch_files(html_files, self.config.batch_size):
            batch_result = self._process_norm_file_batch(batch)
            total_processed += batch_result.processed_count
            
            # Обновляем статистику
            self.processing_stats['total_norms_found'] += batch_result.processed_count
            self.processing_stats['skipped_norms'] += batch_result.skipped_count
            self.processing_stats['error_norms'] += batch_result.error_count
        
        logger.info(f"Norm processing completed: {total_processed} norms processed")
        return total_processed > 0
    
    def _batch_files(self, files: list[Path], batch_size: int) -> Iterator[list[Path]]:
        """Разбивает файлы на батчи."""
        for i in range(0, len(files), batch_size):
            yield files[i:i + batch_size]
    
    def _process_norm_file_batch(self, file_paths: list[Path]) -> NormProcessingResult:
        """Обрабатывает батч файлов норм."""
        batch_result = NormProcessingResult()
        
        for file_path in file_paths:
            try:
                file_result = self._process_single_norm_file_optimized(file_path)
                batch_result.processed_count += file_result.processed_count
                batch_result.skipped_count += file_result.skipped_count
                batch_result.error_count += file_result.error_count
                batch_result.validation_errors.extend(file_result.validation_errors)
                
                logger.debug(f"Processed norm file {file_path.name}: {file_result.processed_count} norms")
                
            except Exception as e:
                logger.error(f"Error processing norm file {file_path}: {e}")
                batch_result.error_count += 1
        
        return batch_result
    
    def _process_single_norm_file_optimized(self, file_path: Path) -> NormProcessingResult:
        """Оптимизированная обработка одного HTML файла норм."""
        result = NormProcessingResult()
        
        try:
            # Очищаем HTML файл
            with temporary_file(suffix='.html') as temp_file:
                cleaned_content = self._clean_norm_html_file_optimized(file_path, temp_file)
                
                if not cleaned_content:
                    result.error_count += 1
                    return result
                
                # Извлекаем нормы из очищенного файла
                norms_data = self._extract_norms_from_cleaned_html_optimized(cleaned_content)
                
                # Обрабатываем и кэшируем нормы
                for norm_id, norm_data in norms_data.items():
                    if self._validate_and_cache_norm(norm_id, norm_data):
                        result.processed_count += 1
                    else:
                        result.skipped_count += 1
                        result.validation_errors.append(f"Invalid norm {norm_id}")
        
        except Exception as e:
            logger.error(f"Error in single norm file processing: {e}")
            result.error_count += 1
        
        return result
    
    def _clean_norm_html_file_optimized(self, input_file: Path, output_file: Path) -> Optional[str]:
        """Оптимизированная очистка HTML файла норм."""
        try:
            with file_reader(input_file) as html_content:
                logger.debug(f"Reading norm file {input_file.name}, size: {len(html_content):,} bytes")
                
                # Ищем таблицы норм с использованием предкомпилированных паттернов
                norm_tables = []
                
                # Таблица по нагрузке на ось
                match1 = self.patterns['norm_table_1'].search(html_content)
                if match1:
                    norm_tables.append(match1.group(1))
                    logger.debug("Found 'load per axle' norm table")
                
                # Таблица по весу поезда
                match2 = self.patterns['norm_table_2'].search(html_content)
                if match2:
                    norm_tables.append(match2.group(1))
                    logger.debug("Found 'train weight' norm table")
                
                if not norm_tables:
                    logger.warning(f"No norm tables found in {input_file.name}")
                    return None
                
                # Создаем очищенный HTML
                cleaned_html = self._create_cleaned_norm_html(norm_tables)
                
                # Сохраняем во временный файл
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(cleaned_html)
                
                logger.debug(f"Norm HTML file cleaned: {input_file.name}")
                return cleaned_html
                
        except Exception as e:
            logger.error(f"Error cleaning norm HTML file {input_file}: {e}")
            return None
    
    def _create_cleaned_norm_html(self, norm_tables: list[str]) -> str:
        """Создает очищенный HTML с таблицами норм."""
        html_template = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Удельные нормы электроэнергии и топлива</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .rcp12 {{ font-size: 12px; }}
        .filter_key {{ font-weight: bold; }}
        .filter_value {{ color: blue; }}
        .tr_head {{ background-color: #e0e0e0; }}
        .thc {{ border: 1px solid #000; padding: 5px; text-align: center; }}
        .tdc_str1, .tdc_str2 {{ border: 1px solid #000; padding: 3px; text-align: center; }}
        .tdc_str1 {{ background-color: #f9f9f9; }}
        .tdc_str2 {{ background-color: #ffffff; }}
        table {{ border-collapse: collapse; margin: 20px auto; }}
        .link {{ color: blue; text-decoration: underline; }}
    </style>
</head>
<body>
{tables}
</body>
</html>'''
        
        tables_html = '<br><br>\n'.join(norm_tables)
        return html_template.format(tables=tables_html)
    
    def _extract_norms_from_cleaned_html_optimized(self, html_content: str) -> dict[str, NormDict]:
        """Оптимизированное извлечение норм из очищенного HTML."""
        logger.debug("Extracting norms from cleaned HTML (optimized)")
        
        soup = BeautifulSoup(html_content, 'html.parser')
        all_norms = {}
        
        # Обработка таблицы по нагрузке на ось
        load_section = self._find_section_by_text_optimized(soup, 'нагрузке на ось')
        if load_section:
            load_norms = self._extract_norms_from_section_optimized(load_section, 'Нажатие')
            all_norms.update(load_norms)
            logger.debug(f"Extracted {len(load_norms)} norms from load section")
        
        # Обработка таблицы по весу поезда
        weight_section = self._find_section_by_text_optimized(soup, 'весу поезда')
        if weight_section:
            weight_norms = self._extract_norms_from_section_optimized(weight_section, 'Вес')
            all_norms.update(weight_norms)
            logger.debug(f"Extracted {len(weight_norms)} norms from weight section")
        
        logger.debug(f"Total norms extracted: {len(all_norms)}")
        return all_norms
    
    def _find_section_by_text_optimized(self, soup: BeautifulSoup, search_text: str):
        """Оптимизированный поиск секции по тексту."""
        # Используем более эффективный поиск
        for element in soup.find_all(text=lambda text: text and search_text in text):
            return element.parent
        return None
    
    def _extract_norms_from_section_optimized(self, section, norm_type: str) -> dict[str, NormDict]:
        """Оптимизированное извлечение норм из секции."""
        norms = {}
        
        if not section:
            return norms
        
        # Ищем таблицу с данными норм
        current = section.parent
        for sibling in current.find_all_next('table'):
            if sibling.find('tr', class_='tr_head'):
                headers = self._get_table_headers_optimized(sibling)
                rows = sibling.find_all('tr')[1:]  # Пропускаем заголовок
                
                # Определяем диапазон числовых колонок
                numeric_start, numeric_end = self._find_numeric_columns_range(headers)
                
                # Векторизованная обработка строк
                batch_norms = self._process_norm_rows_vectorized(
                    rows, headers, norm_type, numeric_start, numeric_end
                )
                
                norms.update(batch_norms)
                break
        
        return norms
    
    def _get_table_headers_optimized(self, table) -> list[str]:
        """Оптимизированное получение заголовков таблицы."""
        headers = []
        header_row = table.find('tr', class_='tr_head')
        
        if header_row:
            # Используем list comprehension для оптимизации
            headers = [
                TextCleaner.clean_text(th.get_text()) 
                for th in header_row.find_all('th')
            ]
        
        return headers
    
    def _find_numeric_columns_range(self, headers: list[str]) -> tuple[int, int]:
        """Находит диапазон числовых колонок с нормами."""
        numeric_start = 9  # После "Призн. алг. нормир."
        numeric_end = len(headers) - 2  # До колонок с датами
        
        # Оптимизированный поиск начала числовых колонок
        for i, header in enumerate(headers):
            if any(keyword in header.lower() for keyword in ['алг', 'нормир']):
                numeric_start = i + 1
                break
        
        # Оптимизированный поиск конца числовых колонок
        for i in range(len(headers) - 1, -1, -1):
            header = headers[i]
            if any(keyword in header.lower() for keyword in ['дата', 'date']):
                numeric_end = i
            else:
                break
        
        return numeric_start, numeric_end
    
    def _process_norm_rows_vectorized(
        self, 
        rows, 
        headers: list[str], 
        norm_type: str,
        numeric_start: int, 
        numeric_end: int
    ) -> dict[str, NormDict]:
        """Векторизованная обработка строк норм."""
        norms = {}
        
        # Подготавливаем векторы для batch обработки
        row_data_batch = []
        
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) <= 10:  # Слишком мало данных
                continue
            
            row_data = self._extract_row_data_optimized(cells, headers, numeric_start, numeric_end)
            if row_data:
                row_data_batch.append(row_data)
        
        # Векторизованная обработка batch'а
        for row_data in row_data_batch:
            norm_data = self._create_norm_data_optimized(row_data, norm_type)
            if norm_data and norm_data.get('norm_id'):
                norms[norm_data['norm_id']] = norm_data
        
        return norms
    
    def _extract_row_data_optimized(
        self, 
        cells, 
        headers: list[str], 
        numeric_start: int, 
        numeric_end: int
    ) -> Optional[dict]:
        """Оптимизированное извлечение данных строки."""
        try:
            row_data = []
            norm_id = ""
            
            # Обрабатываем ячейки векторизованно
            for i, cell in enumerate(cells):
                cell_text = TextCleaner.clean_text(cell.get_text())
                row_data.append(cell_text)
                
                # Извлекаем ID нормы из первой ячейки
                if i == 0:
                    link = cell.find('a')
                    if link and link.get('href'):
                        norm_id = self._extract_norm_id_from_link_optimized(link['href'])
            
            if not norm_id or len(row_data) <= 10:
                return None
            
            # Векторизованное извлечение числовых данных
            numeric_data = self._extract_numeric_data_vectorized(
                row_data, headers, numeric_start, numeric_end
            )
            
            if not numeric_data:
                return None
            
            return {
                'norm_id': norm_id,
                'row_data': row_data,
                'numeric_data': numeric_data,
                'headers': headers
            }
            
        except Exception as e:
            logger.debug(f"Error extracting row data: {e}")
            return None
    
    def _extract_norm_id_from_link_optimized(self, href: str) -> str:
        """Оптимизированное извлечение ID нормы из ссылки."""
        match = self.patterns['norm_id_link'].search(href)
        return match.group(1) if match else ""
    
    def _extract_numeric_data_vectorized(
        self, 
        row_data: list[str], 
        headers: list[str], 
        numeric_start: int, 
        numeric_end: int
    ) -> dict[float, float]:
        """Векторизованное извлечение числовых данных норм."""
        numeric_data = {}
        
        # Векторизованная обработка числовых колонок
        for i in range(numeric_start, min(numeric_end, len(row_data), len(headers))):
            if not row_data[i].strip():
                continue
            
            try:
                # Векторизованное преобразование заголовка и значения
                header_value = TextCleaner.extract_number(headers[i])
                consumption_value = TextCleaner.extract_number(row_data[i])
                
                if header_value is not None and consumption_value is not None:
                    numeric_data[float(header_value)] = float(consumption_value)
                    
            except (ValueError, TypeError):
                continue
        
        return numeric_data
    
    def _create_norm_data_optimized(self, row_data: dict, norm_type: str) -> Optional[NormDict]:
        """Оптимизированное создание данных нормы."""
        try:
            norm_id = row_data['norm_id']
            numeric_data = row_data['numeric_data']
            raw_data = row_data['row_data']
            
            if not numeric_data:
                return None
            
            # Создаем объект нормы
            points = [(load, consumption) for load, consumption in sorted(numeric_data.items())]
            
            # Создаем базовые данные с векторизацией
            base_data = self._create_base_data_vectorized(raw_data)
            
            return {
                'norm_id': norm_id,
                'norm_type': norm_type,
                'description': f"Норма №{norm_id} ({norm_type})",
                'points': points,
                'base_data': base_data
            }
            
        except Exception as e:
            logger.debug(f"Error creating norm data: {e}")
            return None
    
    def _create_base_data_vectorized(self, raw_data: list[str]) -> dict:
        """Векторизованное создание базовых данных нормы."""
        base_fields = [
            'priznok_sost_tyag', 'priznok_rek', 'vid_dvizheniya', 
            'simvol_rod_raboty', 'rps', 'identif_gruppy',
            'priznok_sost', 'priznok_alg'
        ]
        
        base_data = {}
        
        # Векторизованная обработка полей
        for i, field in enumerate(base_fields, 1):
            if i < len(raw_data):
                value = self._convert_to_number_optimized(raw_data[i])
                if value is not None:
                    base_data[field] = value
        
        # Добавляем даты (последние две колонки)
        if len(raw_data) >= 2:
            base_data['date_start'] = raw_data[-2] if raw_data[-2] else ''
            base_data['date_end'] = raw_data[-1] if raw_data[-1] else ''
        
        return base_data
    
    def _convert_to_number_optimized(self, text: str):
        """Оптимизированное преобразование текста в число."""
        if not text or not text.strip():
            return None
        
        cleaned_text = TextCleaner.clean_text(text)
        
        # Быстрая проверка на число
        try:
            if '.' in cleaned_text or ',' in cleaned_text:
                return float(cleaned_text.replace(',', '.'))
            else:
                return int(cleaned_text)
        except (ValueError, TypeError):
            return cleaned_text  # Возвращаем как текст если не число
    
    def _validate_and_cache_norm(self, norm_id: str, norm_data: NormDict) -> bool:
        """Валидирует и кэширует норму."""
        try:
            # Создаем объект NormDefinition
            norm_definition = NormDefinition(
                norm_id=norm_id,
                points=norm_data['points'],
                description=norm_data.get('description', ''),
                norm_type=norm_data.get('norm_type', 'Unknown')
            )
            
            # Валидируем если включена валидация
            if self.config.enable_validation:
                is_valid, message = norm_definition.validate()
                if not is_valid:
                    logger.debug(f"Norm {norm_id} validation failed: {message}")
                    return False
            
            # Кэшируем норму
            self.norms_cache[norm_id] = norm_definition
            
            # Управление размером кэша
            if len(self.norms_cache) > self.config.cache_size:
                self._cleanup_cache()
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating/caching norm {norm_id}: {e}")
            return False
    
    def _cleanup_cache(self) -> None:
        """Очищает кэш до разумного размера."""
        if len(self.norms_cache) <= self.config.cache_size:
            return
        
        # Удаляем 20% старых записей (простая стратегия)
        items_to_remove = len(self.norms_cache) - int(self.config.cache_size * 0.8)
        
        # Удаляем первые элементы (как FIFO)
        keys_to_remove = list(self.norms_cache.keys())[:items_to_remove]
        
        for key in keys_to_remove:
            del self.norms_cache[key]
        
        logger.debug(f"Cache cleaned up: removed {items_to_remove} entries")
    
    def get_norm(self, norm_id: str) -> Optional[NormDict]:
        """Получает норму по ID."""
        norm_def = self.norms_cache.get(norm_id)
        
        if norm_def:
            return {
                'norm_id': norm_def.norm_id,
                'points': norm_def.points,
                'description': norm_def.description,
                'norm_type': norm_def.norm_type
            }
        
        return None
    
    def get_all_norms(self) -> dict[str, NormDict]:
        """Возвращает все кэшированные нормы."""
        result = {}
        
        for norm_id, norm_def in self.norms_cache.items():
            result[norm_id] = {
                'norm_id': norm_def.norm_id,
                'points': norm_def.points,
                'description': norm_def.description,
                'norm_type': norm_def.norm_type
            }
        
        return result
    
    def validate_norms(self) -> dict[str, ValidationResult]:
        """Валидирует все кэшированные нормы."""
        validation_results = {}
        
        for norm_id, norm_def in self.norms_cache.items():
            is_valid, message = norm_def.validate()
            validation_results[norm_id] = (is_valid, message)
        
        # Статистика валидации
        valid_count = sum(1 for valid, _ in validation_results.values() if valid)
        total_count = len(validation_results)
        
        logger.info(f"Norm validation completed: {valid_count}/{total_count} valid")
        
        return validation_results
    
    def get_storage_info(self) -> dict:
        """Возвращает информацию о хранилище норм."""
        return {
            'cached_norms': len(self.norms_cache),
            'cache_size_limit': self.config.cache_size,
            'memory_usage_mb': self._estimate_memory_usage(),
            'processing_stats': dict(self.processing_stats)
        }
    
    def get_storage_statistics(self) -> dict:
        """Возвращает статистику норм."""
        if not self.norms_cache:
            return {'total_norms': 0}
        
        # Векторизованный расчет статистики
        points_counts = [len(norm.points) for norm in self.norms_cache.values()]
        norm_types = [norm.norm_type for norm in self.norms_cache.values()]
        
        # Статистика по типам
        type_counts = pd.Series(norm_types).value_counts().to_dict()
        
        # Диапазоны значений
        all_loads = []
        all_consumptions = []
        
        for norm in self.norms_cache.values():
            if norm.points:
                loads, consumptions = zip(*norm.points)
                all_loads.extend(loads)
                all_consumptions.extend(consumptions)
        
        return {
            'total_norms': len(self.norms_cache),
            'by_type': type_counts,
            'avg_points_per_norm': np.mean(points_counts) if points_counts else 0,
            'load_range': {
                'min': min(all_loads) if all_loads else 0,
                'max': max(all_loads) if all_loads else 0
            },
            'consumption_range': {
                'min': min(all_consumptions) if all_consumptions else 0,
                'max': max(all_consumptions) if all_consumptions else 0
            },
            'points_distribution': pd.Series(points_counts).value_counts().to_dict()
        }
    
    def _estimate_memory_usage(self) -> float:
        """Оценивает потребление памяти кэшем в МБ."""
        if not self.norms_cache:
            return 0.0
        
        # Простая оценка: количество норм * средний размер нормы
        avg_points_per_norm = np.mean([len(norm.points) for norm in self.norms_cache.values()])
        
        # Примерный размер одной нормы в байтах
        bytes_per_norm = (
            100 +  # Базовые поля
            avg_points_per_norm * 16 +  # Точки (2 float по 8 байт)
            50  # Дополнительные данные
        )
        
        total_bytes = len(self.norms_cache) * bytes_per_norm
        return total_bytes / (1024 * 1024)  # Конвертируем в МБ
    
    def get_processing_stats(self) -> ProcessingStats:
        """Возвращает статистику обработки."""
        return dict(self.processing_stats)
    
    def clear_cache(self) -> None:
        """Очищает кэш норм."""
        cleared_count = len(self.norms_cache)
        self.norms_cache.clear()
        logger.info(f"Norm cache cleared: {cleared_count} norms removed")
    
    def export_norms_to_json(self, output_file: Path | str) -> bool:
        """Экспортирует нормы в JSON файл."""
        try:
            import json
            
            output_path = Path(output_file)
            
            # Подготавливаем данные для экспорта
            export_data = {
                'metadata': {
                    'total_norms': len(self.norms_cache),
                    'export_timestamp': pd.Timestamp.now().isoformat(),
                    'processing_stats': dict(self.processing_stats)
                },
                'norms': {}
            }
            
            # Конвертируем нормы в JSON-serializable формат
            for norm_id, norm_def in self.norms_cache.items():
                export_data['norms'][norm_id] = {
                    'norm_id': norm_def.norm_id,
                    'points': norm_def.points,
                    'description': norm_def.description,
                    'norm_type': norm_def.norm_type,
                    'load_range': norm_def.load_range,
                    'consumption_range': norm_def.consumption_range
                }
            
            # Сохраняем в файл
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Norms exported to JSON: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting norms to JSON: {e}")
            return False
    
    def import_norms_from_json(self, input_file: Path | str) -> bool:
        """Импортирует нормы из JSON файла."""
        try:
            import json
            
            input_path = Path(input_file)
            
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            imported_norms = data.get('norms', {})
            if not imported_norms:
                logger.warning("No norms found in JSON file")
                return False
            
            # Импортируем нормы
            imported_count = 0
            for norm_id, norm_data in imported_norms.items():
                try:
                    norm_definition = NormDefinition(
                        norm_id=norm_data['norm_id'],
                        points=norm_data['points'],
                        description=norm_data.get('description', ''),
                        norm_type=norm_data.get('norm_type', 'Unknown')
                    )
                    
                    self.norms_cache[norm_id] = norm_definition
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Error importing norm {norm_id}: {e}")
                    continue
            
            logger.info(f"Imported {imported_count} norms from JSON: {input_path}")
            return imported_count > 0
            
        except Exception as e:
            logger.error(f"Error importing norms from JSON: {e}")
            return False
