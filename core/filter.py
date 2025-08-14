# core/filter.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from typing import List, Tuple, Dict

class LocomotiveFilter:
    """Фильтр для выбора локомотивов"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.avl = self._extract_locomotives()
        self.sel = set(self.avl)  # По умолчанию все выбраны
        
    def _extract_locomotives(self) -> List[Tuple[str, int]]:
        loc = []
        if 'Серия локомотива' in self.df.columns and 'Номер локомотива' in self.df.columns:
            for _, r in self.df.iterrows():
                s = r.get('Серия локомотива', '')
                n = r.get('Номер локомотива', 0)
                if pd.notna(s) and pd.notna(n):
                    try:
                        if isinstance(n, str):
                            n = int(n.lstrip('0')) if n.strip().lstrip('0') else 0
                        else:
                            n = int(n)
                        l = (str(s), n)
                        if l not in loc:
                            loc.append(l)
                    except (ValueError, TypeError):
                        continue
        loc.sort(key=lambda x: (x[0], x[1]))
        return loc
    
    def get_locomotives_by_series(self) -> Dict[str, List[int]]:
        sd = {}
        for s, n in self.avl:
            if s not in sd:
                sd[s] = []
            sd[s].append(n)
        for s in sd:
            sd[s].sort()
        return sd
    
    def set_selected_locomotives(self, sel: List[Tuple[str, int]]):
        self.sel = set(sel)
    
    def toggle_locomotive(self, s: str, n: int):
        l = (s, n)
        if l in self.sel:
            self.sel.remove(l)
        else:
            self.sel.add(l)
    
    def select_all_in_series(self, s: str):
        for se, n in self.avl:
            if se == s:
                self.sel.add((se, n))
    
    def deselect_all_in_series(self, s: str):
        for se, n in self.avl:
            if se == s:
                self.sel.discard((se, n))
    
    def select_all(self):
        self.sel = set(self.avl)
    
    def deselect_all(self):
        self.sel = set()
    
    def filter_routes(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.sel:
            return df.iloc[0:0]
        m = pd.Series([False] * len(df))
        for i, r in df.iterrows():
            s = r.get('Серия локомотива', '')
            n = r.get('Номер локомотива', 0)
            if pd.notna(s) and pd.notna(n):
                try:
                    if isinstance(n, str):
                        n = int(n.lstrip('0')) if n.strip().lstrip('0') else 0
                    else:
                        n = int(n)
                    if (str(s), n) in self.sel:
                        m[i] = True
                except (ValueError, TypeError):
                    continue
        return df[m]
    
    def get_selection_statistics(self) -> Dict:
        return {
            'total_available': len(self.avl),
            'total_selected': len(self.sel),
            'series_count': len(set(s for s, _ in self.avl)),
            'selected_series': len(set(s for s, _ in self.sel))
        }