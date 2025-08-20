# core/coefficients.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re

class LocomotiveCoefficientsManager:
    """Менеджер коэффициентов расхода локомотивов"""
    
    def __init__(self):
        self.file = None
        self.data = {}
        self.coef = {}  # {(серия, номер): коэффициент}
        self.debug_log = []  # Для отладки
        
    def log_debug(self, message):
        """Добавление сообщения в лог отладки"""
        self.debug_log.append(message)
        print(f"[DEBUG] {message}")
    
    def normalize_series(self, series: str) -> str:
        """Нормализация названия серии локомотива"""
        # Извлекаем только буквы и цифры
        normalized = re.sub(r'[^А-ЯA-Zа-яa-z0-9]', '', str(series).upper())
        self.log_debug(f"Нормализация серии: '{series}' -> '{normalized}'")
        return normalized
    
    def load_coefficients(self, fp: str, min_work_threshold: float = 0) -> bool:
        """Загрузка коэффициентов из Excel файла (все листы)"""
        try:
            self.file = fp
            self.data = {}
            self.coef = {}
            self.debug_log = []
            
            self.log_debug(f"Начало загрузки файла: {fp}")
            self.log_debug(f"Порог минимальной работы: {min_work_threshold}")
            
            # Читаем все листы файла
            excel_file = pd.ExcelFile(fp)
            total_processed = 0
            
            self.log_debug(f"Найдено листов в файле: {len(excel_file.sheet_names)}")
            
            for sheet_name in excel_file.sheet_names:
                self.log_debug(f"\n--- Обработка листа: '{sheet_name}' ---")
                
                # Извлекаем серию из названия листа
                # Предполагаем, что название листа содержит серию локомотива
                series_name = sheet_name.strip()
                
                # Если в названии листа есть пробелы или дополнительный текст,
                # пытаемся извлечь серию (например, "ВЛ80С" из "Лист ВЛ80С")
                series_match = re.search(r'[А-ЯA-Z]+[\d]+[А-ЯA-Z]*', series_name)
                if series_match:
                    series_name = series_match.group()
                
                self.log_debug(f"Извлеченная серия из листа '{sheet_name}': '{series_name}'")
                
                try:
                    # Читаем данные листа
                    df_raw = pd.read_excel(fp, sheet_name=sheet_name, header=None)
                    
                    # Ищем строку с заголовками
                    header_row = None
                    for i in range(min(10, len(df_raw))):
                        row = df_raw.iloc[i]
                        for j, cell in enumerate(row):
                            if pd.notna(cell) and 'Завод' in str(cell) and 'номер' in str(cell).lower():
                                header_row = i
                                break
                        if header_row is not None:
                            break
                    
                    if header_row is None:
                        self.log_debug(f"Не найдена строка с заголовками на листе '{sheet_name}'")
                        continue
                    
                    # Читаем данные начиная со строки после заголовков
                    df = pd.read_excel(fp, sheet_name=sheet_name, skiprows=header_row, header=0)
                    
                    # Поиск нужных колонок
                    locomotive_col = None
                    work_col = None
                    percent_col = None
                    
                    for col in df.columns:
                        col_str = str(col).lower()
                        if 'завод' in col_str and 'номер' in col_str:
                            locomotive_col = col
                        elif 'процент' in col_str or '%' in str(col):
                            percent_col = col
                        elif 'работа' in col_str:
                            work_col = col
                    
                    if not locomotive_col or not percent_col:
                        self.log_debug(f"Не найдены необходимые колонки на листе '{sheet_name}'")
                        continue
                    
                    self.log_debug(f"Найдены колонки: локомотив='{locomotive_col}', процент='{percent_col}', работа='{work_col}'")
                    
                    vd = []
                    sheet_processed = 0
                    sheet_filtered = 0
                    
                    for idx, r in df.iterrows():
                        try:
                            # Пропускаем пустые строки
                            if r.isna().all():
                                continue
                            
                            lns = str(r[locomotive_col]).strip()
                            if not lns or pd.isna(lns) or lns == 'nan':
                                continue
                            
                            # Убираем ведущие нули и преобразуем в число
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
                                        sheet_filtered += 1
                                        self.log_debug(f"Локомотив {series_name} №{ln} отфильтрован: работа {work_value:.1f} < {min_work_threshold}")
                                        continue
                                except (ValueError, TypeError):
                                    work_value = 0
                            else:
                                work_value = 0
                            
                            # Получаем значение коэффициента
                            pv = r[percent_col]
                            if pd.notna(pv):
                                try:
                                    if isinstance(pv, str):
                                        # Убираем символ % и заменяем запятую на точку
                                        pv = pv.replace('%', '').replace(',', '.')
                                        co = float(pv)
                                    else:
                                        co = float(pv)
                                    
                                    # Если значение больше 10, предполагаем, что это проценты
                                    # и нужно перевести в коэффициент (например, 105% -> 1.05)
                                    if co > 10:
                                        co = co / 100
                                    
                                    # Нормализуем серию для сохранения
                                    normalized_series = self.normalize_series(series_name)
                                    
                                    vd.append({
                                        'series': series_name,
                                        'series_normalized': normalized_series,
                                        'number': ln,
                                        'coefficient': co,
                                        'deviation_percent': (co - 1) * 100,
                                        'work_total': work_value
                                    })
                                    
                                    # Сохраняем как с оригинальной серией, так и с нормализованной
                                    self.coef[(series_name, ln)] = co
                                    self.coef[(normalized_series, ln)] = co
                                    
                                    sheet_processed += 1
                                    
                                    if sheet_processed <= 3:  # Показываем первые 3 для отладки
                                        self.log_debug(f"Загружен: {series_name} №{ln} -> коэфф={co:.3f}, откл={(co-1)*100:.1f}%, работа={work_value:.0f}")
                                    
                                except (ValueError, TypeError) as e:
                                    self.log_debug(f"Ошибка обработки коэффициента для {series_name} №{ln}: {e}")
                                    continue
                        
                        except Exception as e:
                            self.log_debug(f"Ошибка обработки строки {idx}: {e}")
                            continue
                    
                    if vd:
                        self.data[series_name] = vd
                        self.log_debug(f"Лист '{sheet_name}': загружено {sheet_processed} локомотивов, отфильтровано {sheet_filtered}")
                        total_processed += sheet_processed
                    else:
                        self.log_debug(f"Лист '{sheet_name}': нет данных для сохранения")
                
                except Exception as e:
                    self.log_debug(f"Ошибка обработки листа '{sheet_name}': {e}")
                    continue
            
            self.log_debug(f"\n=== ИТОГО загружено {total_processed} локомотивов из {len(self.data)} серий ===")
            self.log_debug(f"Всего коэффициентов в словаре: {len(self.coef)}")
            
            # Выводим сводку по сериям
            for series_name, locomotives in self.data.items():
                self.log_debug(f"Серия '{series_name}': {len(locomotives)} локомотивов")
            
            return total_processed > 0
            
        except Exception as e:
            self.log_debug(f"КРИТИЧЕСКАЯ ОШИБКА загрузки коэффициентов: {e}")
            return False
    
    def get_coefficient(self, s: str, n: int) -> float:
        """Получение коэффициента для локомотива"""
        # Прямой поиск
        k = (s, n)
        if k in self.coef:
            self.log_debug(f"Найден коэффициент для {s} №{n}: {self.coef[k]:.3f} (прямой поиск)")
            return self.coef[k]
        
        # Поиск с нормализацией
        ns = self.normalize_series(s)
        k_norm = (ns, n)
        if k_norm in self.coef:
            self.log_debug(f"Найден коэффициент для {s} №{n}: {self.coef[k_norm]:.3f} (через нормализацию '{ns}')")
            return self.coef[k_norm]
        
        # Поиск по всем вариантам серий
        for (series, number), coeff in self.coef.items():
            if number == n:
                series_norm = self.normalize_series(series)
                if series_norm == ns:
                    self.log_debug(f"Найден коэффициент для {s} №{n}: {coeff:.3f} (через сопоставление '{series}' -> '{series_norm}')")
                    return coeff
        
        # Коэффициент не найден
        if not hasattr(self, '_not_found_logged'):
            self._not_found_logged = set()
        
        if (s, n) not in self._not_found_logged:
            self._not_found_logged.add((s, n))
            self.log_debug(f"ВНИМАНИЕ: Коэффициент НЕ НАЙДЕН для {s} №{n}, используется 1.0")
            
            # Показываем доступные серии для отладки
            if len(self._not_found_logged) <= 3:
                available_series = list(set(se for se, nu in self.coef.keys() if nu == n))
                if available_series:
                    self.log_debug(f"  Доступные серии для локомотива №{n}: {available_series}")
        
        return 1.0
    
    def apply_coefficient_to_consumption(self, c: float, s: str, n: int) -> float:
        """Применение коэффициента к расходу"""
        co = self.get_coefficient(s, n)
        result = c / co
        if co != 1.0:
            self.log_debug(f"Применен коэффициент {co:.3f} к расходу {c:.1f} для {s} №{n} -> результат {result:.1f}")
        return result
    
    def get_all_locomotives(self) -> List[Tuple[str, int]]:
        """Получение списка всех локомотивов"""
        # Возвращаем только уникальные пары (серия, номер)
        unique_locomotives = {}
        for (series, number), coeff in self.coef.items():
            # Используем оригинальные названия серий из data
            for original_series in self.data.keys():
                if self.normalize_series(original_series) == self.normalize_series(series):
                    unique_locomotives[(original_series, number)] = coeff
                    break
            else:
                unique_locomotives[(series, number)] = coeff
        
        return list(unique_locomotives.keys())
    
    def get_series_list(self) -> List[str]:
        """Получение списка серий"""
        return list(self.data.keys())
    
    def get_locomotives_by_series(self, s: str) -> List[Dict]:
        """Получение локомотивов по серии"""
        # Ищем как по прямому совпадению, так и по нормализованному
        if s in self.data:
            return self.data[s]
        
        # Поиск по нормализованной серии
        ns = self.normalize_series(s)
        for series, locomotives in self.data.items():
            if self.normalize_series(series) == ns:
                return locomotives
        
        return []
    
    def get_statistics(self) -> Dict:
        """Получение статистики по коэффициентам"""
        if not self.coef:
            return {}
        
        # Собираем уникальные коэффициенты
        unique_coeffs = {}
        for (series, number), coeff in self.coef.items():
            # Используем нормализованный ключ для исключения дубликатов
            key = (self.normalize_series(series), number)
            unique_coeffs[key] = coeff
        
        co = list(unique_coeffs.values())
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
            'locomotives_at_norm': sum(1 for c in co if abs(c - 1.0) < 0.001)
        }
    
    def get_debug_log(self) -> List[str]:
        """Получение лога отладки"""
        return self.debug_log