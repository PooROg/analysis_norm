#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный процессор для обработки HTML файлов норм.
Использует современные возможности Python 3.12 для максимальной производительности.
"""

from __future__ import annotations

import os
import logging
import tempfile
from typing import Dict, List, Optional
from pathlib import Path

from .data_models import NormData, NormType, ProcessingStats
from .norm_parser import FastNormParser

logger = logging.getLogger(__name__)

# Типы для Python 3.12
type HTMLContent = str
type NormDict = Dict[str, Any]

class HTMLNormProcessor:
    """Исправленный процессор для обработки HTML файлов норм."""
    
    def __init__(self):
        self.processed_norms: Dict[str, NormData] = {}
        self.processing_stats = ProcessingStats()
        self.norm_parser = FastNormParser()
    
    def clean_html_file(self, input_file: str) -> Optional[str]:
        """Очищает HTML файл норм от лишнего кода с улучшенной обработкой ошибок."""
        logger.info(f"Очистка HTML файла норм: {input_file}")
        
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"Файл {input_file} не найден!")
            return None
        
        # Читаем файл с автоопределением кодировки
        html_content = self._read_file_with_encoding(input_path)
        if html_content is None:
            return None
        
        # Извлекаем секции таблиц с использованием быстрого парсера
        sections = self.norm_parser.extract_table_sections(html_content)
        
        # Создаем очищенный HTML
        cleaned_html = self.norm_parser.create_cleaned_html(sections)
        
        # Создаем временный файл
        temp_file = tempfile.mktemp(suffix='.html')
        temp_path = Path(temp_file)
        temp_path.write_text(cleaned_html, encoding='utf-8')
        
        logger.info(f"HTML файл норм очищен и сохранен во временный файл: {temp_file}")
        return temp_file
    
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
    
    def process_html_files(self, html_files: List[str]) -> Dict[str, Dict]:
        """Обрабатывает список HTML файлов норм с улучшенной производительностью."""
        logger.info(f"Начинаем обработку {len(html_files)} HTML файлов норм")
        
        self.processing_stats.total_files = len(html_files)
        all_norms: Dict[str, NormData] = {}
        
        # Обрабатываем каждый файл
        for file_path in html_files:
            logger.info(f"Обработка файла норм: {Path(file_path).name}")
            
            try:
                # Очищаем HTML файл
                cleaned_file = self.clean_html_file(file_path)
                if not cleaned_file:
                    logger.error(f"Не удалось очистить файл {file_path}")
                    self.processing_stats.processing_errors += 1
                    continue
                
                # Обрабатываем очищенный файл
                file_norms = self._extract_norms_from_cleaned_html(cleaned_file)
                
                # Объединяем нормы
                for norm_id, norm_data in file_norms.items():
                    if norm_id in all_norms:
                        logger.warning(f"Норма {norm_id} уже существует, перезаписываем")
                    all_norms[norm_id] = norm_data
                
                # Удаляем временный файл
                self._safe_remove_file(cleaned_file)
                
            except Exception as e:
                logger.error(f"Ошибка при обработке файла норм {file_path}: {e}")
                self.processing_stats.processing_errors += 1
                continue
        
        self.processing_stats.total_items_found = len(all_norms)
        logger.info(f"Обработка завершена. Найдено {len(all_norms)} норм")
        
        # Сохраняем обработанные нормы
        self.processed_norms = all_norms
        
        # Преобразуем в старый формат для совместимости
        return {norm_id: norm.to_dict() for norm_id, norm in all_norms.items()}
    
    def _extract_norms_from_cleaned_html(self, cleaned_file: str) -> Dict[str, NormData]:
        """Извлекает нормы из очищенного HTML файла с использованием быстрого парсера."""
        logger.debug(f"Извлекаем нормы из файла: {cleaned_file}")
        
        # Читаем HTML файл
        html_content = Path(cleaned_file).read_text(encoding='utf-8')
        
        # Извлекаем секции таблиц
        sections = self.norm_parser.extract_table_sections(html_content)
        
        # Обрабатываем каждую секцию
        all_norms: Dict[str, NormData] = {}
        
        for section in sections:
            try:
                section_norms = self.norm_parser.parse_section_to_norms(section)
                all_norms.update(section_norms)
                logger.debug(f"Секция '{section.title}': найдено {len(section_norms)} норм")
            except Exception as e:
                logger.error(f"Ошибка обработки секции '{section.title}': {e}")
                self.processing_stats.processing_errors += 1
                continue
        
        logger.debug(f"Всего извлечено норм: {len(all_norms)}")
        return all_norms
    
    def _safe_remove_file(self, file_path: str):
        """Безопасно удаляет файл."""
        try:
            Path(file_path).unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Не удалось удалить временный файл {file_path}: {e}")
    
    def compare_norms(self, new_norms: Dict[str, Dict], 
                     existing_norms: Dict[str, Dict]) -> Dict[str, str]:
        """Сравнивает новые нормы с существующими с улучшенной логикой."""
        logger.info("Сравниваем новые нормы с существующими")
        
        comparison_result = {}
        
        for norm_id, new_norm in new_norms.items():
            try:
                if norm_id not in existing_norms:
                    comparison_result[norm_id] = 'new'
                    self.processing_stats.new_items += 1
                else:
                    existing_norm = existing_norms[norm_id]
                    if self._norms_are_different(new_norm, existing_norm):
                        comparison_result[norm_id] = 'updated'
                        self.processing_stats.updated_items += 1
                    else:
                        comparison_result[norm_id] = 'unchanged'
            except Exception as e:
                logger.error(f"Ошибка сравнения нормы {norm_id}: {e}")
                comparison_result[norm_id] = 'error'
                continue
        
        logger.info(
            f"Результат сравнения: "
            f"новых {self.processing_stats.new_items}, "
            f"обновленных {self.processing_stats.updated_items}"
        )
        
        return comparison_result
    
    def _norms_are_different(self, norm1: Dict, norm2: Dict) -> bool:
        """Сравнивает две нормы на предмет различий с улучшенным алгоритмом."""
        try:
            # Быстрая проверка по типу нормы
            if norm1.get('norm_type') != norm2.get('norm_type'):
                return True
            
            # Сравниваем точки с учетом плавающей точки
            points1 = set(
                (round(p[0], 6), round(p[1], 6)) 
                for p in norm1.get('points', [])
            )
            points2 = set(
                (round(p[0], 6), round(p[1], 6)) 
                for p in norm2.get('points', [])
            )
            
            if points1 != points2:
                return True
            
            # Сравниваем базовые данные
            base1 = norm1.get('base_data', {})
            base2 = norm2.get('base_data', {})
            
            # Сравниваем только значимые поля
            significant_fields = [
                'priznok_sost_tyag', 'priznok_rek', 'vid_dvizheniya',
                'simvol_rod_raboty', 'rps', 'identif_gruppy',
                'priznok_sost', 'priznok_alg'
            ]
            
            for field in significant_fields:
                val1 = base1.get(field)
                val2 = base2.get(field)
                
                # Для числовых значений используем приближенное сравнение
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) > 1e-6:
                        return True
                elif val1 != val2:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Ошибка сравнения норм: {e}")
            return True  # В случае ошибки считаем, что нормы разные
    
    def get_processing_stats(self) -> Dict:
        """Возвращает статистику обработки в формате словаря для совместимости."""
        stats = self.processing_stats.to_dict()
        # Добавляем специфичные для норм поля для обратной совместимости
        stats['total_norms_found'] = stats['total_items_found']
        stats['new_norms'] = stats['new_items']
        stats['updated_norms'] = stats['updated_items']
        return stats
    
    def get_norm_by_id(self, norm_id: str) -> Optional[Dict]:
        """Возвращает норму по ID."""
        norm = self.processed_norms.get(norm_id)
        return norm.to_dict() if norm else None
    
    def get_norms_by_type(self, norm_type: str) -> Dict[str, Dict]:
        """Возвращает нормы определенного типа."""
        try:
            target_type = NormType.AXLE_LOAD if norm_type == "Нажатие" else NormType.TRAIN_WEIGHT
            return {
                norm_id: norm.to_dict() 
                for norm_id, norm in self.processed_norms.items()
                if norm.norm_type == target_type
            }
        except Exception as e:
            logger.error(f"Ошибка фильтрации норм по типу {norm_type}: {e}")
            return {}
    
    def get_norm_statistics(self) -> Dict[str, int]:
        """Возвращает статистику по типам норм."""
        stats = {
            'total': len(self.processed_norms),
            'axle_load': 0,
            'train_weight': 0
        }
        
        for norm in self.processed_norms.values():
            if norm.norm_type == NormType.AXLE_LOAD:
                stats['axle_load'] += 1
            elif norm.norm_type == NormType.TRAIN_WEIGHT:
                stats['train_weight'] += 1
        
        return stats
    
    def validate_norm_data(self, norm_data: Dict) -> bool:
        """Валидирует данные нормы."""
        try:
            # Проверяем обязательные поля
            required_fields = ['norm_id', 'norm_type', 'points']
            for field in required_fields:
                if field not in norm_data:
                    return False
            
            # Проверяем структуру точек
            points = norm_data.get('points', [])
            if not isinstance(points, list):
                return False
            
            for point in points:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    return False
                if not all(isinstance(x, (int, float)) for x in point):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка валидации нормы: {e}")
            return False
    
    def export_norms_summary(self) -> Dict[str, any]:
        """Экспортирует сводку по нормам."""
        summary = {
            'total_norms': len(self.processed_norms),
            'by_type': {},
            'processing_stats': self.get_processing_stats(),
            'sample_norms': {}
        }
        
        # Группируем по типам
        for norm in self.processed_norms.values():
            norm_type = norm.norm_type.value
            if norm_type not in summary['by_type']:
                summary['by_type'][norm_type] = []
            summary['by_type'][norm_type].append(norm.norm_id)
        
        # Добавляем примеры норм (первые 3 из каждого типа)
        for norm_type, norm_ids in summary['by_type'].items():
            summary['sample_norms'][norm_type] = norm_ids[:3]
        
        return summary
    
    def clear_processed_data(self):
        """Очищает обработанные данные для экономии памяти."""
        self.processed_norms.clear()
        logger.info("Обработанные данные норм очищены")