# core/filter.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный фильтр локомотивов с правильной интеграцией
Объединяет рабочую функциональность старого кода с новыми оптимизациями Python 3.12
"""

import pandas as pd
from dataclasses import dataclass
from typing import Protocol
from functools import lru_cache, cached_property
import logging

logger = logging.getLogger(__name__)

# Python 3.12 enhanced type system
type LocomotiveID = tuple[str, int]
type LocomotiveStats = dict[str, int | float]

@dataclass(slots=True, frozen=True)
class LocomotiveInfo:
    """Immutable locomotive information with slots optimization."""
    series: str
    number: int
    
    def __post_init__(self):
        if not self.series or self.number <= 0:
            raise ValueError("Invalid locomotive data")
    
    @property
    def display_name(self) -> str:
        """Human-readable locomotive name."""
        return f"{self.series} №{self.number}"

class DataFrameProcessor(Protocol):
    """Protocol for DataFrame processing strategies."""
    def process(self, df: pd.DataFrame) -> pd.DataFrame: ...

class LocomotiveFilter:
    """Исправленный фильтр локомотивов с полной совместимостью со старым API."""
    
    def __init__(self, routes_df: pd.DataFrame):
        self.routes_df = routes_df
        self.df = routes_df  # Алиас для совместимости
        self._validate_dataframe()
        
        # Новые атрибуты
        self.available_locomotives = self._extract_locomotives()
        self.selected: set[LocomotiveID] = set(self.available_locomotives)
        
        # Старые атрибуты для совместимости
        self.avl = list(self.available_locomotives)
        self.sel = set(self.available_locomotives)
        
        logger.info(f"Initialized filter with {len(self.available_locomotives)} locomotives")
    
    def _validate_dataframe(self) -> None:
        """Validate required columns exist."""
        required_cols = {'Серия локомотива', 'Номер локомотива'}
        if not required_cols.issubset(self.routes_df.columns):
            missing = required_cols - set(self.routes_df.columns)
            raise ValueError(f"Missing required columns: {missing}")
    
    @cached_property
    def locomotives_by_series(self) -> dict[str, list[int]]:
        """Group locomotives by series with caching."""
        series_dict = {}
        for series, number in self.available_locomotives:
            if series not in series_dict:
                series_dict[series] = []
            series_dict[series].append(number)
        
        # Sort numbers within each series
        for series in series_dict:
            series_dict[series].sort()
        
        return series_dict
    
    def _extract_locomotives(self) -> frozenset[LocomotiveID]:
        """Extract and cache unique locomotives using vectorized operations."""
        try:
            # Vectorized filtering for valid data
            valid_mask = (
                self.routes_df['Серия локомотива'].notna() & 
                self.routes_df['Номер локомотива'].notna()
            )
            valid_data = self.routes_df[valid_mask].copy()
            
            if valid_data.empty:
                logger.warning("No valid locomotive data found")
                return frozenset()
            
            # Vectorized processing
            valid_data['series_clean'] = valid_data['Серия локомотива'].astype(str)
            valid_data['number_clean'] = (
                valid_data['Номер локомотива']
                .astype(str)
                .str.lstrip('0')
                .replace('', '0')
                .astype(int)
            )
            
            # Extract unique locomotives
            locomotives = set()
            for _, row in valid_data[['series_clean', 'number_clean']].drop_duplicates().iterrows():
                if row['number_clean'] > 0:
                    locomotives.add((row['series_clean'], row['number_clean']))
            
            result = frozenset(locomotives)
            logger.info(f"Extracted {len(result)} unique locomotives")
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract locomotives: {e}")
            return frozenset()
    
    def get_locomotives_by_series(self) -> dict[str, list[int]]:
        """Get locomotives grouped by series - старый метод для совместимости."""
        return self.locomotives_by_series
    
    def set_selected_locomotives(self, selected: list[LocomotiveID] | set[LocomotiveID]) -> None:
        """Set selected locomotives with validation."""
        if isinstance(selected, list):
            selected = set(selected)
        
        # Validate selection
        invalid = selected - self.available_locomotives
        if invalid:
            logger.warning(f"Ignoring invalid locomotives: {invalid}")
            selected -= invalid
        
        self.selected = selected
        self.sel = selected  # Обновляем старый формат
        logger.info(f"Selected {len(selected)} locomotives")
    
    def toggle_locomotive(self, series: str, number: int) -> bool:
        """Toggle locomotive selection state."""
        locomotive_id = (series, number)
        if locomotive_id not in self.available_locomotives:
            logger.warning(f"Cannot toggle unknown locomotive: {locomotive_id}")
            return False
        
        if locomotive_id in self.selected:
            self.selected.remove(locomotive_id)
            self.sel.discard(locomotive_id)
            return False
        else:
            self.selected.add(locomotive_id)
            self.sel.add(locomotive_id)
            return True
    
    def select_all_in_series(self, series: str) -> int:
        """Select all locomotives in series."""
        count = 0
        for ser, num in self.available_locomotives:
            if ser == series:
                self.selected.add((ser, num))
                self.sel.add((ser, num))
                count += 1
        
        logger.info(f"Selected {count} locomotives in series {series}")
        return count
    
    def deselect_all_in_series(self, series: str) -> int:
        """Deselect all locomotives in series."""
        count = 0
        to_remove = set()
        for ser, num in self.selected:
            if ser == series:
                to_remove.add((ser, num))
                count += 1
        
        self.selected -= to_remove
        self.sel -= to_remove
        logger.info(f"Deselected {count} locomotives in series {series}")
        return count
    
    def select_all(self) -> None:
        """Select all available locomotives."""
        self.selected = set(self.available_locomotives)
        self.sel = set(self.available_locomotives)
        logger.info("Selected all locomotives")
    
    def deselect_all(self) -> None:
        """Deselect all locomotives."""
        self.selected.clear()
        self.sel.clear()
        logger.info("Deselected all locomotives")
    
    def filter_routes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter routes using vectorized operations for performance."""
        if not self.selected and not self.sel:
            logger.warning("No locomotives selected - returning empty DataFrame")
            return df.iloc[0:0].copy()
        
        # Используем актуальное множество выбранных локомотивов
        selected_set = self.selected if self.selected else self.sel
        
        try:
            # Create vectorized boolean mask
            mask = pd.Series([False] * len(df), index=df.index)
            
            # Group selected locomotives by series for efficient processing
            selected_by_series = {}
            for series, number in selected_set:
                if series not in selected_by_series:
                    selected_by_series[series] = set()
                selected_by_series[series].add(number)
            
            # Vectorized filtering by series
            for series, numbers in selected_by_series.items():
                series_mask = df['Серия локомотива'].astype(str) == series
                
                # Handle number matching with vectorized operations
                df_numbers = (
                    df['Номер локомотива']
                    .astype(str)
                    .str.lstrip('0')
                    .replace('', '0')
                    .astype(int)
                )
                number_mask = df_numbers.isin(numbers)
                
                mask |= (series_mask & number_mask)
            
            filtered_df = df[mask].copy()
            logger.info(f"Filtered {len(filtered_df)} routes from {len(df)} total")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Route filtering failed: {e}")
            return df.iloc[0:0].copy()
    
    @lru_cache(maxsize=64)
    def get_series_statistics(self, series: str) -> LocomotiveStats:
        """Get cached statistics for a series."""
        series_locomotives = [num for ser, num in self.available_locomotives if ser == series]
        selected_in_series = [num for ser, num in self.selected if ser == series]
        
        return {
            'total': len(series_locomotives),
            'selected': len(selected_in_series),
            'selection_percentage': len(selected_in_series) / len(series_locomotives) * 100 if series_locomotives else 0,
            'min_number': min(series_locomotives) if series_locomotives else 0,
            'max_number': max(series_locomotives) if series_locomotives else 0
        }
    
    def get_selection_statistics(self) -> LocomotiveStats:
        """Get overall selection statistics."""
        total_available = len(self.available_locomotives)
        total_selected = len(self.selected) if self.selected else len(self.sel)
        
        return {
            'total_available': total_available,
            'total_selected': total_selected,
            'selection_percentage': (total_selected / total_available * 100) if total_available else 0,
            'series_count': len(set(series for series, _ in self.available_locomotives)),
            'selected_series_count': len(set(series for series, _ in (self.selected or self.sel))),
            'unselected_count': total_available - total_selected
        }
    
    def export_selection(self) -> list[dict[str, str | int]]:
        """Export current selection for external use."""
        selected_set = self.selected if self.selected else self.sel
        return [
            {'series': series, 'number': number}
            for series, number in sorted(selected_set)
        ]
    
    def import_selection(self, selection_data: list[dict[str, str | int]]) -> bool:
        """Import selection from external data."""
        try:
            new_selection = set()
            for item in selection_data:
                locomotive_id = (str(item['series']), int(item['number']))
                if locomotive_id in self.available_locomotives:
                    new_selection.add(locomotive_id)
            
            self.selected = new_selection
            self.sel = new_selection
            logger.info(f"Imported selection of {len(new_selection)} locomotives")
            return True
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to import selection: {e}")
            return False