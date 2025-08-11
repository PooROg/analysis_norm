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
                # Пропускаем первые 3 строки (фильтры, пустая, субзаголовки), header из 4-й строки
                df = pd.read_excel(fp, sheet_name=sn, skiprows=3)
                lnc = None
                pc = None
                # Поиск колонок в заголовках
                for c in df.columns:
                    if 'Завод. номер секции ТПС' in str(c) or 'номер' in str(c).lower():
                        lnc = c
                    if 'Процент' in str(c) or 'процент' in str(c).lower():
                        pc = c
                if lnc and pc:
                    # Извлечение серии из первой строки оригинального файла (фильтр)
                    # Читаем без skip для парсинга фильтра
                    filter_df = pd.read_excel(fp, sheet_name=sn, header=None)
                    filter_text = str(filter_df.iloc[0, 0])  # row1
                    series_name = 'ВЛ80С'  # По умолчанию
                    if 'электровоз.ВЛ80С' in filter_text:
                        series_name = 'ВЛ80С'
                    # Или парсите динамически: series_name = filter_text.split('электровоз.')[1].split(' ')[0] if 'электровоз.' in filter_text else sn
                    
                    vd = []
                    for _, r in df.iterrows():
                        try:
                            lns = str(r[lnc]).strip()
                            if not lns or pd.isna(lns):
                                continue
                            ln = int(lns.lstrip('0')) if lns.lstrip('0') else 0
                            pv = r[pc]
                            if pd.notna(pv):
                                if isinstance(pv, str):
                                    pv = float(pv.replace('%', '').replace(',', '.'))
                                else:
                                    pv = float(pv)
                                # Поскольку значения ~1 (e.g., 1.06), предполагаем, что pv уже коэффициент (не /100)
                                # Если в файле 107 для 107%, добавьте co = pv / 100
                                # Но по данным: co = pv (1.06 = 106%)
                                co = pv  # Измените на pv / 100, если значения >10
                                vd.append({
                                    'number': ln,
                                    'coefficient': co,
                                    'deviation_percent': (co - 1) * 100  # Отклонение в %
                                })
                                self.coef[(series_name, ln)] = co
                        except (ValueError, TypeError):
                            continue
                    if vd:
                        self.data[series_name] = vd
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