# core/coefficients.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный менеджер коэффициентов с правильной интеграцией
Совмещает рабочую логику старого кода с новыми оптимизациями Python 3.12
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from functools import lru_cache, cached_property
from typing import Any
import logging
import re

logger = logging.getLogger(__name__)

# Python 3.12 type definitions
type LocomotiveID = tuple[str, int]
type CoefficientData = dict[str, float | int | str]
type SeriesPattern = str

@dataclass(slots=True, frozen=True)
class LocomotiveCoefficient:
    """Immutable coefficient data with slots optimization."""
    series: str
    number: int
    coefficient: float
    work_hours: float = 0.0
    efficiency_rating: str = field(default="normal", init=False)
    
    def __post_init__(self):
        # Validate coefficient range
        if not 0.1 <= self.coefficient <= 3.0:
            raise ValueError(f"Invalid coefficient: {self.coefficient}")
        
        # Set efficiency rating based on coefficient
        match self.coefficient:
            case c if c > 1.15:
                object.__setattr__(self, 'efficiency_rating', 'poor')
            case c if c > 1.05:
                object.__setattr__(self, 'efficiency_rating', 'below_average')
            case c if c < 0.85:
                object.__setattr__(self, 'efficiency_rating', 'excellent')
            case c if c < 0.95:
                object.__setattr__(self, 'efficiency_rating', 'good')
            case _:
                object.__setattr__(self, 'efficiency_rating', 'normal')
    
    @property
    def deviation_percent(self) -> float:
        """Deviation from baseline (1.0) in percentage."""
        return (self.coefficient - 1.0) * 100
    
    @property
    def locomotive_id(self) -> LocomotiveID:
        """Locomotive identifier tuple."""
        return (self.series, self.number)

class LocomotiveCoefficientsManager:
    """Исправленный менеджер коэффициентов с совместимостью со старым API"""
    
    # Common locomotive series patterns for fuzzy matching
    SERIES_PATTERNS = {
        'ВЛ80С': ['вл80с', 'вл-80с', 'вл 80с'],
        'ВЛ80Т': ['вл80т', 'вл-80т', 'вл 80т'],
        'ВЛ85': ['вл85', 'вл-85', 'вл 85'],
        'ЭП1М': ['эп1м', 'эп-1м', 'эп 1м'],
        '2ЭС5К': ['2эс5к', '2эс-5к', '2эс 5к', 'эс5к']
    }
    
    def __init__(self):
        self.file = None  # Совместимость со старым API
        self.file_path: Path | None = None
        self.coefficients: dict[LocomotiveID, LocomotiveCoefficient] = {}
        self.coef = {}  # Старый формат для совместимости
        self.data = {}  # Старый формат для совместимости
        self._series_name: str = ""
        self._stats_cache_valid = False
        self.debug_log = []
        
    def log_debug(self, message):
        """Добавление сообщения в лог отладки"""
        self.debug_log.append(message)
        logger.debug(message)
    
    @cached_property
    def loading_statistics(self) -> dict[str, Any]:
        """Cached loading statistics."""
        if not self.coefficients:
            return {}
        
        coeffs = list(self.coefficients.values())
        coeff_values = [c.coefficient for c in coeffs]
        
        return {
            'total_locomotives': len(coeffs),
            'series_name': self._series_name,
            'coefficient_mean': np.mean(coeff_values),
            'coefficient_std': np.std(coeff_values),
            'coefficient_min': np.min(coeff_values),
            'coefficient_max': np.max(coeff_values),
            'efficiency_distribution': self._get_efficiency_distribution(coeffs),
            'work_hours_available': sum(1 for c in coeffs if c.work_hours > 0)
        }
    
    def _get_efficiency_distribution(self, coeffs: list[LocomotiveCoefficient]) -> dict[str, int]:
        """Get distribution of efficiency ratings."""
        distribution = {}
        for coeff in coeffs:
            rating = coeff.efficiency_rating
            distribution[rating] = distribution.get(rating, 0) + 1
        return distribution
    
    def load_coefficients(self, file_path: Path | str, min_work_threshold: float = 0) -> bool:
        """Load coefficients using optimized Excel processing."""
        try:
            if isinstance(file_path, str):
                file_path = Path(file_path)
                
            logger.info(f"Loading coefficients from {file_path}")
            self.file = str(file_path)  # Совместимость
            self.file_path = file_path
            self.coefficients.clear()
            self.coef.clear()
            self.data.clear()
            self._stats_cache_valid = False
            self.debug_log.clear()
            
            # Fast Excel reading with optimized settings
            df = self._read_excel_optimized(file_path)
            
            # Extract series name from file or content
            self._series_name = self._extract_series_name(file_path, df)
            
            # Find required columns using fuzzy matching
            column_mapping = self._find_columns(df)
            if not column_mapping['locomotive'] or not column_mapping['coefficient']:
                logger.error("Required columns not found")
                return False
            
            # Process coefficients with vectorized operations
            processed_count = self._process_coefficients(
                df, column_mapping, min_work_threshold
            )
            
            if processed_count > 0:
                logger.info(f"Loaded {processed_count} coefficients for series {self._series_name}")
                # Clear cached property
                if hasattr(self, 'loading_statistics'):
                    delattr(self, 'loading_statistics')
                return True
            else:
                logger.warning("No valid coefficients found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load coefficients: {e}")
            return False
    
    def _read_excel_optimized(self, file_path: Path) -> pd.DataFrame:
        """Read Excel with optimized settings for performance."""
        # Try different engines for best performance
        for engine in ['openpyxl', 'xlrd']:
            try:
                return pd.read_excel(
                    file_path,
                    engine=engine,
                    dtype={
                        'Завод. номер секции ТПС': 'str',
                        'Процент': 'str'  # Read as string to handle various formats
                    },
                    skiprows=self._detect_header_row(file_path)
                )
            except Exception as e:
                logger.debug(f"Engine {engine} failed: {e}")
                continue
        
        # Fallback to default
        return pd.read_excel(file_path)
    
    def _detect_header_row(self, file_path: Path) -> int:
        """Detect the row containing headers."""
        try:
            # Read first few rows to find headers
            df_sample = pd.read_excel(file_path, nrows=10, header=None)
            
            for i, row in df_sample.iterrows():
                row_text = ' '.join(str(cell).lower() for cell in row if pd.notna(cell))
                if 'завод' in row_text and 'номер' in row_text:
                    return i
            
            return 0  # Default to first row
            
        except Exception:
            return 0
    
    def _find_columns(self, df: pd.DataFrame) -> dict[str, str | None]:
        """Find required columns using fuzzy matching."""
        mapping = {
            'locomotive': None,
            'coefficient': None,
            'work_hours': None
        }
        
        for col in df.columns:
            col_str = str(col).lower()
            
            # Locomotive number column
            if all(keyword in col_str for keyword in ['завод', 'номер', 'тпс']):
                mapping['locomotive'] = col
            
            # Coefficient column
            elif 'процент' in col_str:
                mapping['coefficient'] = col
            
            # Work hours column (in previous rows sometimes)
            elif all(keyword in col_str for keyword in ['работа', 'всего']):
                mapping['work_hours'] = col
        
        return mapping
    
    def _extract_series_name(self, file_path: Path, df: pd.DataFrame) -> str:
        """Extract locomotive series from file name or content."""
        # Try filename first
        filename = file_path.stem.upper()
        
        # Check against known patterns
        for series, patterns in self.SERIES_PATTERNS.items():
            if any(pattern.upper() in filename for pattern in patterns):
                return series
        
        # Try to extract from DataFrame content
        if not df.empty:
            # Check first few cells for series information
            for i in range(min(5, len(df))):
                for j in range(min(3, len(df.columns))):
                    cell_value = str(df.iloc[i, j]).upper()
                    for series, patterns in self.SERIES_PATTERNS.items():
                        if any(pattern.upper() in cell_value for pattern in patterns):
                            return series
        
        # Default fallback
        return 'ВЛ80С'
    
    def _process_coefficients(self, df: pd.DataFrame, 
                            column_mapping: dict[str, str | None], 
                            min_work_threshold: float) -> int:
        """Process coefficients using vectorized operations."""
        locomotive_col = column_mapping['locomotive']
        coefficient_col = column_mapping['coefficient']
        work_col = column_mapping['work_hours']
        
        # Filter valid data
        valid_mask = (
            df[locomotive_col].notna() & 
            df[coefficient_col].notna()
        )
        valid_df = df[valid_mask].copy()
        
        if valid_df.empty:
            return 0
        
        # Apply work hours filter if specified
        if work_col and min_work_threshold > 0:
            work_mask = valid_df[work_col].fillna(0) >= min_work_threshold
            valid_df = valid_df[work_mask]
        
        processed_count = 0
        series_data = []
        
        # Process each row
        for _, row in valid_df.iterrows():
            try:
                # Parse locomotive number
                number_str = str(row[locomotive_col]).strip()
                number = int(number_str.lstrip('0') or '0')
                
                if number <= 0:
                    continue
                
                # Parse coefficient
                coefficient = self._parse_coefficient_value(row[coefficient_col])
                if coefficient is None:
                    continue
                
                # Parse work hours
                work_hours = 0.0
                if work_col and pd.notna(row[work_col]):
                    try:
                        work_hours = float(row[work_col])
                    except (ValueError, TypeError):
                        pass
                
                # Create coefficient object
                locomotive_id = (self._series_name, number)
                coeff_obj = LocomotiveCoefficient(
                    series=self._series_name,
                    number=number,
                    coefficient=coefficient,
                    work_hours=work_hours
                )
                
                self.coefficients[locomotive_id] = coeff_obj
                
                # Старый формат для совместимости
                self.coef[locomotive_id] = coefficient
                
                # Добавляем в data для совместимости
                series_data.append({
                    'series': self._series_name,
                    'series_normalized': self._normalize_series_name(self._series_name),
                    'number': number,
                    'coefficient': coefficient,
                    'deviation_percent': (coefficient - 1) * 100,
                    'work_total': work_hours
                })
                
                processed_count += 1
                
            except (ValueError, TypeError) as e:
                logger.debug(f"Skipping invalid row: {e}")
                continue
        
        # Сохраняем данные для совместимости
        if series_data:
            self.data[self._series_name] = series_data
        
        return processed_count
    
    @staticmethod
    def _parse_coefficient_value(value: Any) -> float | None:
        """Parse coefficient value handling various formats."""
        if pd.isna(value):
            return None
        
        try:
            # Convert to string and clean
            str_value = str(value).strip()
            
            # Remove common artifacts
            str_value = str_value.replace('%', '').replace(',', '.')
            
            # Handle scientific notation
            if 'e' in str_value.lower():
                result = float(str_value)
            else:
                result = float(str_value)
            
            # If value is > 10, assume it's percentage and convert
            if result > 10:
                result = result / 100
            
            # Validate range
            if 0.1 <= result <= 3.0:
                return result
            
            logger.debug(f"Coefficient out of range: {result}")
            return None
            
        except (ValueError, TypeError):
            logger.debug(f"Cannot parse coefficient: {value}")
            return None
    
    def _normalize_series_name(self, series: str) -> str:
        """Normalize series name for fuzzy matching."""
        normalized = series.upper().replace('-', '').replace(' ', '').replace('.', '')
        return normalized
    
    @lru_cache(maxsize=1024)
    def get_coefficient(self, series: str, number: int) -> float:
        """Get coefficient with LRU caching for performance."""
        locomotive_id = (series, number)
        
        # Direct lookup first
        if locomotive_id in self.coefficients:
            return self.coefficients[locomotive_id].coefficient
        
        # Also check old format
        if locomotive_id in self.coef:
            return self.coef[locomotive_id]
        
        # Fuzzy matching for series variations
        normalized_series = self._normalize_series_name(series)
        
        for (stored_series, stored_number), coeff_obj in self.coefficients.items():
            if (stored_number == number and 
                self._normalize_series_name(stored_series) == normalized_series):
                return coeff_obj.coefficient
        
        # Check old format too
        for (stored_series, stored_number), coeff in self.coef.items():
            if (stored_number == number and 
                self._normalize_series_name(stored_series) == normalized_series):
                return coeff
        
        # No match found
        return 1.0
    
    def apply_coefficient_to_consumption(self, consumption: float, 
                                       series: str, number: int) -> float:
        """Apply coefficient to consumption value."""
        coefficient = self.get_coefficient(series, number)
        return consumption / coefficient if coefficient != 0 else consumption
    
    def get_locomotive_info(self, series: str, number: int) -> LocomotiveCoefficient | None:
        """Get full locomotive coefficient information."""
        locomotive_id = (series, number)
        return self.coefficients.get(locomotive_id)
    
    def get_all_locomotives(self) -> list[LocomotiveID]:
        """Get all locomotive IDs."""
        # Объединяем из обеих структур данных
        all_locomotives = set(self.coefficients.keys())
        all_locomotives.update(self.coef.keys())
        return list(all_locomotives)
    
    def get_series_list(self) -> list[str]:
        """Get list of available series."""
        series_set = set()
        for series, _ in self.coefficients.keys():
            series_set.add(series)
        for series, _ in self.coef.keys():
            series_set.add(series)
        # Также добавляем из data
        series_set.update(self.data.keys())
        return list(series_set)
    
    def get_locomotives_by_series(self, series: str) -> list[dict]:
        """Get all locomotives for a specific series."""
        # Сначала проверяем новый формат
        if series in self.data:
            return self.data[series]
        
        # Создаем из coefficients
        locomotives = []
        normalized_series = self._normalize_series_name(series)
        
        for (ser, num), coeff_obj in self.coefficients.items():
            if self._normalize_series_name(ser) == normalized_series:
                locomotives.append({
                    'series': ser,
                    'number': num,
                    'coefficient': coeff_obj.coefficient,
                    'work_total': coeff_obj.work_hours,
                    'deviation_percent': coeff_obj.deviation_percent
                })
        
        return locomotives
    
    def get_statistics(self) -> dict:
        """Get statistics about loaded coefficients."""
        if not self.coefficients and not self.coef:
            return {}
        
        # Get unique coefficients
        all_coeffs = {}
        for (series, number), coeff_obj in self.coefficients.items():
            all_coeffs[(series, number)] = coeff_obj.coefficient
        
        for (series, number), coeff in self.coef.items():
            if (series, number) not in all_coeffs:
                all_coeffs[(series, number)] = coeff
        
        if not all_coeffs:
            return {}
        
        coeffs = list(all_coeffs.values())
        deviations = [(c - 1.0) * 100 for c in coeffs]
        
        return {
            'total_locomotives': len(coeffs),
            'series_count': len(set(series for series, _ in all_coeffs.keys())),
            'avg_coefficient': np.mean(coeffs),
            'min_coefficient': min(coeffs),
            'max_coefficient': max(coeffs),
            'avg_deviation_percent': np.mean(deviations),
            'locomotives_above_norm': sum(1 for c in coeffs if c > 1.0),
            'locomotives_below_norm': sum(1 for c in coeffs if c < 1.0),
            'locomotives_at_norm': sum(1 for c in coeffs if abs(c - 1.0) < 0.001)
        }
    
    def clear_coefficients(self) -> None:
        """Clear all loaded coefficients."""
        self.coefficients.clear()
        self.coef.clear()
        self.data.clear()
        self.file_path = None
        self.file = None
        self._series_name = ""
        self._stats_cache_valid = False
        
        # Clear caches
        self.get_coefficient.cache_clear()
        
        if hasattr(self, 'loading_statistics'):
            delattr(self, 'loading_statistics')
        
        logger.info("Cleared all coefficients")
    
    def get_debug_log(self) -> list[str]:
        """Get debug log for troubleshooting."""
        return self.debug_log

# Создаем алиас для совместимости
CoefficientManager = LocomotiveCoefficientsManager