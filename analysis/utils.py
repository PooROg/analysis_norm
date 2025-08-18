# analysis/utils.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Оптимизированные утилиты для анализа данных с использованием современных возможностей Python 3.12.
"""

from __future__ import annotations
import re
import tempfile
from pathlib import Path
from typing import Any, Optional, Iterator
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Типы для улучшения читаемости
type NumericValue = int | float
type TextContent = str
type CleanedText = str

class TextCleaner:
    """Оптимизированный очиститель текста с кэшированием паттернов."""
    
    _patterns: dict[str, re.Pattern[str]] = {}
    
    @classmethod
    def _get_pattern(cls, name: str, pattern: str) -> re.Pattern[str]:
        """Получает скомпилированный паттерн с кэшированием."""
        if name not in cls._patterns:
            cls._patterns[name] = re.compile(pattern, re.DOTALL | re.IGNORECASE)
        return cls._patterns[name]
    
    @classmethod
    def clean_text(cls, text: TextContent) -> CleanedText:
        """Очищает текст от лишних символов с оптимизацией."""
        if not text:
            return ""
        
        # Замена небезопасных символов одной операцией
        cleaned = text.replace('\xa0', ' ').replace('&nbsp;', ' ').replace('\u00a0', ' ')
        
        # Нормализация пробелов одним проходом
        whitespace_pattern = cls._get_pattern('whitespace', r'\s+')
        return whitespace_pattern.sub(' ', cleaned).strip()
    
    @classmethod
    def extract_number(cls, text: TextContent, force_int: bool = False) -> Optional[NumericValue]:
        """Извлекает число из текста с оптимизацией."""
        if not text:
            return None
        
        # Очистка за один проход
        cleaned = str(text).strip().replace(' ', '').replace(',', '.')
        if cleaned.endswith('.'):
            cleaned = cleaned[:-1]
        
        if not cleaned or cleaned.lower() in ('nan', 'none', ''):
            return None
        
        try:
            num = float(cleaned)
            # Делаем число положительным и возвращаем int если нужно
            num = abs(num)
            return int(num) if force_int or num == int(num) else num
        except (ValueError, OverflowError):
            return None

class MathUtils:
    """Оптимизированные математические утилиты."""
    
    @staticmethod
    def safe_divide(numerator: NumericValue, denominator: NumericValue) -> Optional[float]:
        """Безопасное деление с проверками."""
        if denominator == 0 or numerator is None or denominator is None:
            return None
        
        try:
            return abs(float(numerator) / float(denominator))
        except (ZeroDivisionError, ValueError, OverflowError):
            return None
    
    @staticmethod
    def safe_subtract(*values: NumericValue) -> Optional[float]:
        """Безопасное вычитание с возвратом абсолютного значения."""
        valid_values = [v for v in values if v is not None]
        
        if not valid_values:
            return None
        
        try:
            result = valid_values[0]
            for v in valid_values[1:]:
                result = result - v
            return abs(float(result))
        except (ValueError, OverflowError):
            return None
    
    @staticmethod
    def calculate_percentage_diff(value1: NumericValue, value2: NumericValue) -> Optional[float]:
        """Вычисляет процентное отклонение."""
        if value2 == 0 or value1 is None or value2 is None:
            return None
        
        try:
            return ((float(value1) - float(value2)) / float(value2)) * 100
        except (ZeroDivisionError, ValueError):
            return None

class URLExtractor:
    """Оптимизированный извлекатель данных из URL."""
    
    _url_patterns: dict[str, re.Pattern[str]] = {}
    
    @classmethod
    def _get_url_pattern(cls, name: str, pattern: str) -> re.Pattern[str]:
        """Получает URL паттерн с кэшированием."""
        if name not in cls._url_patterns:
            cls._url_patterns[name] = re.compile(pattern)
        return cls._url_patterns[name]
    
    @classmethod
    def extract_norm_id(cls, href: str) -> Optional[str]:
        """Извлекает ID нормы из URL."""
        if not href:
            return None
        
        # Оптимизированный поиск с кэшированным паттерном
        pattern = cls._get_url_pattern('norm_id', r'id_ntp_tax=(\d+)')
        match = pattern.search(href)
        
        if match:
            return match.group(1)
        
        # Fallback паттерн
        fallback_pattern = cls._get_url_pattern('norm_id_fallback', r'id=(\d+)')
        fallback_match = fallback_pattern.search(href)
        
        return fallback_match.group(1) if fallback_match else None

@contextmanager
def temporary_file(suffix: str = '.html', encoding: str = 'utf-8') -> Iterator[Path]:
    """Context manager для временных файлов с автоматической очисткой."""
    temp_file = None
    try:
        temp_file = tempfile.mktemp(suffix=suffix)
        temp_path = Path(temp_file)
        yield temp_path
    finally:
        if temp_file and Path(temp_file).exists():
            try:
                Path(temp_file).unlink()
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Failed to cleanup temporary file {temp_file}: {e}")

@contextmanager
def file_reader(file_path: Path | str, encodings: list[str] = None) -> Iterator[str]:
    """Context manager для чтения файлов с множественными кодировками."""
    if encodings is None:
        encodings = ['utf-8', 'cp1251', 'windows-1251', 'latin-1']
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    content = None
    used_encoding = None
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
                used_encoding = encoding
                break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        raise UnicodeDecodeError(f"Could not decode file {file_path} with any of: {encodings}")
    
    logger.debug(f"Read file {file_path} with encoding {used_encoding}")
    yield content

class HTMLCleaner:
    """Оптимизированный очиститель HTML с кэшированными паттернами."""
    
    _html_patterns: dict[str, re.Pattern[str]] = {}
    
    @classmethod
    def _get_html_pattern(cls, name: str, pattern: str, flags: int = re.DOTALL) -> re.Pattern[str]:
        """Получает HTML паттерн с кэшированием."""
        if name not in cls._html_patterns:
            cls._html_patterns[name] = re.compile(pattern, flags)
        return cls._html_patterns[name]
    
    @classmethod
    def clean_html_content(cls, content: str) -> str:
        """Очищает HTML контент от лишних элементов."""
        if not content:
            return content
        
        original_size = len(content)
        
        # Удаляем лишние элементы одним проходом
        patterns_to_remove = [
            ('date_info', r'<font class = rcp12 ><center>Дата получения:.*?</font>\s*<br>'),
            ('route_num', r'<font class = rcp12 ><center>Номер маршрута:.*?</font><br>'),
            ('numline', r'<tr class=tr_numline>.*?</tr>'),
        ]
        
        for name, pattern in patterns_to_remove:
            compiled_pattern = cls._get_html_pattern(name, pattern)
            content = compiled_pattern.sub('', content)
        
        # Очистка атрибутов одним проходом
        attribute_patterns = [
            ('align_center', r'\s+ALIGN=center'),
            ('align_left', r'\s+align=left'),
            ('align_right', r'\s+align=right'),
        ]
        
        for name, pattern in attribute_patterns:
            compiled_pattern = cls._get_html_pattern(name, pattern, re.IGNORECASE)
            content = compiled_pattern.sub('', content)
        
        # Удаление тегов одним проходом
        tags_to_remove = ['center', 'pre']
        for tag in tags_to_remove:
            pattern_name = f'remove_{tag}'
            open_pattern = cls._get_html_pattern(f'{pattern_name}_open', f'<{tag}>')
            close_pattern = cls._get_html_pattern(f'{pattern_name}_close', f'</{tag}>')
            content = open_pattern.sub('', content)
            content = close_pattern.sub('', content)
        
        # Нормализация пробелов между тегами
        space_pattern = cls._get_html_pattern('normalize_spaces', r'>[ \t]+<')
        content = space_pattern.sub('><', content)
        
        cleaned_size = len(content)
        removed_bytes = original_size - cleaned_size
        
        logger.debug(f"HTML cleaning: removed {removed_bytes:,} bytes ({removed_bytes/original_size*100:.1f}%)")
        
        return content

def find_content_boundaries(content: str, start_marker: str, end_marker: str) -> Optional[str]:
    """Находит контент между маркерами с оптимизацией."""
    if not content or not start_marker:
        return None
    
    start_pos = content.find(start_marker)
    if start_pos == -1:
        return None
    
    if not end_marker:
        return content[start_pos:]
    
    end_pos = content.find(end_marker, start_pos)
    if end_pos == -1:
        return content[start_pos:]
    
    return content[start_pos:end_pos + len(end_marker)]

def batch_process[T, R](items: list[T], processor: callable[[T], R], 
                        batch_size: int = 100) -> Iterator[list[R]]:
    """Обрабатывает элементы батчами для оптимизации памяти."""
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        results = []
        for item in batch:
            try:
                result = processor(item)
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing item in batch: {e}")
                continue
        
        if results:
            yield results

class ConfigManager:
    """Менеджер конфигурации с валидацией."""
    
    def __init__(self, default_config: dict[str, Any] = None):
        self._config = default_config or {}
    
    def get[T](self, key: str, default: T = None) -> T:
        """Получает значение конфигурации с типизацией."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Устанавливает значение конфигурации."""
        self._config[key] = value
    
    def update(self, config: dict[str, Any]) -> None:
        """Обновляет конфигурацию."""
        self._config.update(config)
    
    @property
    def config(self) -> dict[str, Any]:
        """Возвращает копию конфигурации."""
        return self._config.copy()
