# Файл: core/filter.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from typing import List, Tuple, Dict, Set

class LocomotiveFilter:
    """Фильтр для выбора локомотивов"""
    
    def __init__(self, routes_df: pd.DataFrame):
        self.routes_df = routes_df
        self.available_locomotives = self._extract_locomotives()
        self.selected_locomotives = set(self.available_locomotives)  # По умолчанию все выбраны
        
    def _extract_locomotives(self) -> List[Tuple[str, int]]:
        locomotives = []
        if 'Серия локомотива' in self.routes_df.columns and 'Номер локомотива' in self.routes_df.columns:
            for _, row in self.routes_df.iterrows():
                series = row.get('Серия локомотива', '')
                number = row.get('Номер локомотива', 0)
                if pd.notna(series) and pd.notna(number):
                    try:
                        if isinstance(number, str):
                            number = int(number.lstrip('0')) if number.strip().lstrip('0') else 0
                        else:
                            number = int(number)
                        loco = (str(series), number)
                        if loco not in locomotives:
                            locomotives.append(loco)
                    except (ValueError, TypeError):
                        continue
        locomotives.sort(key=lambda x: (x[0], x[1]))
        return locomotives
    
    def get_locomotives_by_series(self) -> Dict[str, List[int]]:
        series_dict = {}
        for series, number in self.available_locomotives:
            if series not in series_dict:
                series_dict[series] = []
            series_dict[series].append(number)
        for series in series_dict:
            series_dict[series].sort()
        return series_dict
    
    def set_selected_locomotives(self, selected: List[Tuple[str, int]]):
        self.selected_locomotives = set(selected)
    
    def toggle_locomotive(self, series: str, number: int):
        loco = (series, number)
        if loco in self.selected_locomotives:
            self.selected_locomotives.remove(loco)
        else:
            self.selected_locomotives.add(loco)
    
    def select_all_in_series(self, series: str):
        for s, n in self.available_locomotives:
            if s == series:
                self.selected_locomotives.add((s, n))
    
    def deselect_all_in_series(self, series: str):
        for s, n in self.available_locomotives:
            if s == series:
                self.selected_locomотives.discard((s, n))
    
    def select_all(self):
        self.selected_locomotives = set(self.available_locomotives)
    
    def deselect_all(self):
        self.selected_locomотives = set()
    
    def filter_routes(self, routes_df: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_locomotives:
            return routes_df.iloc[0:0]
        mask = pd.Series([False] * len(routes_df))
        for idx, row in routes_df.iterrows():
            series = row.get('Серия локомотива', '')
            number = row.get('Номер локомотива', 0)
            if pd.notna(series) and pd.notna(number):
                try:
                    if isinstance(number, str):
                        number = int(number.lstrip('0')) if number.strip().lstrip('0') else 0
                    else:
                        number = int(number)
                    if (str(series), number) in self.selected_locomotives:
                        mask[idx] = True
                except (ValueError, TypeError):
                    continue
        return routes_df[mask]
    
    def get_selection_statistics(self) -> Dict:
        return {
            'total_available': len(self.available_locomotives),
            'total_selected': len(self.selected_locomотives),
            'series_count': len(set(s for s, _ in self.available_locomotives)),
            'selected_series': len(set(s for s, _ in self.selected_locomotives))
        }