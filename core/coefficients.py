# core/coefficients.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class LocomotiveCoefficientsManager:
    """Менеджер коэффициентов расхода локомотивов"""
    
    def __init__(self):
        self.file = None
        self.data = {}
        self.coef = {}  # {(серия, номер): коэффициент}
        
    def load_coefficients(self, fp: str) -> bool:
        """Загрузка коэффициентов из Excel файла"""
        try:
            self.file = fp
            self.data = {}
            self.coef = {}
            ef = pd.ExcelFile(fp)
            for sn in ef.sheet_names:
                df = pd.read_excel(fp, sheet_name=sn)
                lnc = None
                pc = None
                for c in df.columns:
                    if 'Завод. номер секции ТПС' in str(c) or 'номер' in str(c).lower():
                        lnc = c
                    if 'Процент' in str(c) or 'процент' in str(c).lower():
                        pc = c
                if lnc and pc:
                    vd = []
                    for _, r in df.iterrows():
                        try:
                            lns = str(r[lnc]).strip()
                            ln = int(lns.lstrip('0')) if lns.lstrip('0') else 0
                            pv = r[pc]
                            if pd.notna(pv):
                                if isinstance(pv, str):
                                    pv = float(pv.replace('%', '').replace(',', '.'))
                                else:
                                    pv = float(pv)
                                co = pv / 100.0
                                vd.append({
                                    'number': ln,
                                    'coefficient': co,
                                    'deviation_percent': pv - 100
                                })
                                self.coef[(sn, ln)] = co
                        except (ValueError, TypeError):
                            continue
                    if vd:
                        self.data[sn] = vd
            return bool(self.data)
        except Exception as e:
            print(f"Ошибка загрузки коэффициентов: {e}")
            return False
    
    def get_coefficient(self, s: str, n: int) -> float:
        k = (s, n)
        if k in self.coef:
            return self.coef[k]
        ns = s.upper().replace('-', '').replace(' ', '')
        for (se, nu), co in self.coef.items():
            if nu == n and se.upper().replace('-', '').replace(' ', '') == ns:
                return co
        return 1.0
    
    def apply_coefficient_to_consumption(self, c: float, s: str, n: int) -> float:
        co = self.get_coefficient(s, n)
        return c / co
    
    def get_all_locomotives(self) -> List[Tuple[str, int]]:
        return list(self.coef.keys())
    
    def get_series_list(self) -> List[str]:
        return list(self.data.keys())
    
    def get_locomotives_by_series(self, s: str) -> List[Dict]:
        return self.data.get(s, [])
    
    def get_statistics(self) -> Dict:
        if not self.coef:
            return {}
        co = list(self.coef.values())
        dv = [(c - 1.0) * 100 for c in co]
        return {
            'total_locomotives': len(co),
            'series_count': len(self.data),
            'avg_coefficient': np.mean(co),
            'min_coefficient': min(co),
            'max_coefficient': max(co),
            'avg_deviation_percent': np.mean(dv),
            'locomotives_above_norm': sum(1 for c in co if c > 1.0),
            'locomotives_below_norm': sum(1 for c in co if c < 1.0),
            'locomotives_at_norm': sum(1 for c in co if c == 1.0)
        }