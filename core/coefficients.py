# Файл: core/coefficients.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class LocomotiveCoefficientsManager:
    """Менеджер коэффициентов расхода локомотивов"""
    
    def __init__(self):
        self.coefficients_file = None
        self.coefficients_data = {}
        self.locomotive_coefficients = {}  # {(серия, номер): коэффициент}
        
    def load_coefficients(self, file_path: str) -> bool:
        """Загрузка коэффициентов из Excel файла"""
        try:
            self.coefficients_file = file_path
            self.coefficients_data = {}
            self.locomotive_coefficients = {}
            excel_file = pd.ExcelFile(file_path)
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                loco_number_col = None
                percent_col = None
                for col in df.columns:
                    if 'Завод. номер секции ТПС' in str(col) or 'номер' in str(col).lower():
                        loco_number_col = col
                    if 'Процент' in str(col) or 'процент' in str(col).lower():
                        percent_col = col
                if loco_number_col and percent_col:
                    valid_data = []
                    for _, row in df.iterrows():
                        try:
                            loco_num_str = str(row[loco_number_col]).strip()
                            loco_num = int(loco_num_str.lstrip('0')) if loco_num_str.lstrip('0') else 0
                            percent_value = row[percent_col]
                            if pd.notna(percent_value):
                                if isinstance(percent_value, str):
                                    percent_value = float(percent_value.replace('%', '').replace(',', '.'))
                                else:
                                    percent_value = float(percent_value)
                                coefficient = percent_value / 100.0
                                valid_data.append({
                                    'number': loco_num,
                                    'coefficient': coefficient,
                                    'deviation_percent': percent_value - 100
                                })
                                self.locomotive_coefficients[(sheet_name, loco_num)] = coefficient
                        except (ValueError, TypeError):
                            continue
                    if valid_data:
                        self.coefficients_data[sheet_name] = valid_data
            return bool(self.coefficients_data)
        except Exception as e:
            print(f"Ошибка загрузки коэффициентов: {e}")
            return False
    
    def get_coefficient(self, series: str, number: int) -> float:
        key = (series, number)
        if key in self.locomotive_coefficients:
            return self.locomotive_coefficients[key]
        norm_series = series.upper().replace('-', '').replace(' ', '')
        for (s, n), coef in self.locomotive_coefficients.items():
            if n == number and s.upper().replace('-', '').replace(' ', '') == norm_series:
                return coef
        return 1.0
    
    def apply_coefficient_to_consumption(self, consumption: float, series: str, number: int) -> float:
        coefficient = self.get_coefficient(series, number)
        return consumption / coefficient
    
    def get_all_locomotives(self) -> List[Tuple[str, int]]:
        return list(self.locomotive_coefficients.keys())
    
    def get_series_list(self) -> List[str]:
        return list(self.coefficients_data.keys())
    
    def get_locomotives_by_series(self, series: str) -> List[Dict]:
        return self.coefficients_data.get(series, [])
    
    def get_statistics(self) -> Dict:
        if not self.locomotive_coefficients:
            return {}
        coefficients = list(self.locomotive_coefficients.values())
        deviations = [(c - 1.0) * 100 for c in coefficients]
        return {
            'total_locomotives': len(coefficients),
            'series_count': len(self.coefficients_data),
            'avg_coefficient': np.mean(coefficients),
            'min_coefficient': min(coefficients),
            'max_coefficient': max(coefficients),
            'avg_deviation_percent': np.mean(deviations),
            'locomotives_above_norm': sum(1 for c in coefficients if c > 1.0),
            'locomotives_below_norm': sum(1 for c in coefficients if c < 1.0),
            'locomotives_at_norm': sum(1 for c in coefficients if c == 1.0)
        }