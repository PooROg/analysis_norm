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
        
    def load_coefficients(self, fp: str, min_work_threshold: float = 0) -> bool:
        """Загрузка коэффициентов из Excel файла"""
        try:
            self.file = fp
            self.data = {}
            self.coef = {}
            
            # Читаем файл как данные без заголовков для анализа структуры
            df_raw = pd.read_excel(fp, header=None)
            
            # Извлекаем серию из фильтра в первой строке
            filter_text = str(df_raw.iloc[0, 0]) if len(df_raw) > 0 else ""
            
            series_name = 'ВЛ80С'  # По умолчанию
            if 'электровоз.' in filter_text:
                try:
                    series_part = filter_text.split('электровоз.')[1]
                    series_name = series_part.split(' ')[0]
                except:
                    series_name = 'ВЛ80С'
            
            # Ищем строку с заголовками (содержит "Завод. номер секции ТПС")
            header_row = None
            for i in range(min(10, len(df_raw))):  # Ищем в первых 10 строках
                row = df_raw.iloc[i]
                for j, cell in enumerate(row):
                    if pd.notna(cell) and 'Завод. номер секции ТПС' in str(cell):
                        header_row = i
                        break
                if header_row is not None:
                    break
            
            if header_row is None:
                print("Ошибка: не найдена строка с заголовками")
                return False
            
            # Читаем данные начиная со строки после заголовков
            df = pd.read_excel(fp, skiprows=header_row, header=0)
            
            # Поиск нужных колонок в заголовках
            locomotive_col = None
            work_col = None
            percent_col = None
            
            for col in df.columns:
                col_str = str(col).lower()
                if 'завод' in col_str and 'номер' in col_str and 'тпс' in col_str:
                    locomotive_col = col
                elif 'процент' in col_str:
                    percent_col = col
            
            # Ищем колонку работы в строке с подзаголовками
            work_headers_row = header_row - 1
            if work_headers_row >= 0 and work_headers_row < len(df_raw):
                work_headers = df_raw.iloc[work_headers_row]
                for i, header in enumerate(work_headers):
                    if pd.notna(header):
                        header_str = str(header).lower()
                        if 'работа' in header_str and 'всего' in header_str:
                            if i < len(df.columns):
                                work_col = df.columns[i]
                                break
            
            if not locomotive_col or not percent_col:
                print(f"Ошибка: не найдены необходимые колонки")
                return False
            
            vd = []
            processed_count = 0
            filtered_count = 0
            
            for idx, r in df.iterrows():
                try:
                    # Пропускаем строки без данных
                    if r.isna().all():
                        continue
                        
                    lns = str(r[locomotive_col]).strip()
                    if not lns or pd.isna(lns) or lns == 'nan' or lns == '':
                        continue
                    
                    # Убираем ведущие нули
                    try:
                        ln = int(lns.lstrip('0')) if lns.lstrip('0') else 0
                    except ValueError:
                        continue
                        
                    if ln == 0:
                        continue
                    
                    # Проверяем работу, если указан порог
                    if work_col and min_work_threshold > 0:
                        try:
                            work_value = float(r[work_col]) if pd.notna(r[work_col]) else 0
                            if work_value < min_work_threshold:
                                filtered_count += 1
                                continue
                        except (ValueError, TypeError):
                            work_value = 0
                    
                    pv = r[percent_col]
                    if pd.notna(pv):
                        try:
                            if isinstance(pv, str):
                                pv = float(pv.replace('%', '').replace(',', '.'))
                            else:
                                pv = float(pv)
                            
                            # Коэффициент уже в правильном виде
                            co = pv
                            work_val = float(r[work_col]) if work_col and pd.notna(r[work_col]) else 0
                            
                            vd.append({
                                'number': ln,
                                'coefficient': co,
                                'deviation_percent': (co - 1) * 100,
                                'work_total': work_val
                            })
                            self.coef[(series_name, ln)] = co
                            processed_count += 1
                                
                        except (ValueError, TypeError):
                            continue
                        
                except Exception:
                    continue
            
            if vd:
                self.data[series_name] = vd
                print(f"Загружено {processed_count} локомотивов серии {series_name}")
                if filtered_count > 0:
                    print(f"Отфильтровано {filtered_count} локомотивов с работой менее {min_work_threshold}")
                return True
            else:
                print("Нет данных для сохранения")
                return False
            
        except Exception as e:
            print(f"Ошибка загрузки коэффициентов: {e}")
            return False
    
    def get_coefficient(self, s: str, n: int) -> float:
        """Получение коэффициента для локомотива"""
        # Прямой поиск по ключу
        k = (s, n)
        if k in self.coef:
            return self.coef[k]
        
        # Поиск с нормализацией названия серии
        ns = s.upper().replace('-', '').replace(' ', '').replace('.', '')
        for (se, nu), co in self.coef.items():
            se_norm = se.upper().replace('-', '').replace(' ', '').replace('.', '')
            if nu == n and se_norm == ns:
                return co
        
        # Дополнительный поиск: если серия содержится в названии или наоборот
        for (se, nu), co in self.coef.items():
            if nu == n:
                # Проверяем, содержится ли одна серия в другой
                if (ns in se.upper() or se.upper() in ns) and len(ns) > 2 and len(se) > 2:
                    return co
        
        # Debug: если коэффициент не найден, выводим информацию (только для первых 3 случаев)
        if len(self.coef) > 0 and not hasattr(self, '_debug_count'):
            self._debug_count = 0
        
        if hasattr(self, '_debug_count') and self._debug_count < 3:
            self._debug_count += 1
            print(f"Debug: Коэффициент не найден для '{s}' №{n}")
            available_series = list(set(se for se, nu in self.coef.keys()))
            print(f"Debug: Доступные серии в коэффициентах: {available_series}")
        
        # Возвращаем 1.0 (100%) если коэффициент не найден
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