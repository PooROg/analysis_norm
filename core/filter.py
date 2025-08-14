# core/filter.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class LocomotiveFilter:
    """Фильтр для выбора локомотивов (обновлен для работы с HTML данными)"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.avl = self._extract_locomotives()
        self.sel = set(self.avl)  # По умолчанию все выбраны
        
    def _extract_locomotives(self) -> List[Tuple[str, int]]:
        """Извлечение локомотивов из DataFrame"""
        loc = []
        
        logger.debug(f"Извлечение локомотивов из DataFrame с колонками: {list(self.df.columns)}")
        
        # Проверяем наличие стандартных колонок
        if 'Серия локомотива' in self.df.columns and 'Номер локомотива' in self.df.columns:
            logger.debug("Найдены стандартные колонки локомотивов")
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
        else:
            # Попытка найти альтернативные колонки с информацией о локомотивах
            logger.debug("Стандартные колонки не найдены, ищем альтернативные")
            
            # Ищем колонки содержащие 'локомотив', 'серия', 'номер' и т.д.
            possible_series_cols = [col for col in self.df.columns if any(word in col.lower() for word in ['серия', 'series', 'локо'])]
            possible_number_cols = [col for col in self.df.columns if any(word in col.lower() for word in ['номер', 'number', 'num'])]
            
            if possible_series_cols and possible_number_cols:
                series_col = possible_series_cols[0]
                number_col = possible_number_cols[0]
                logger.debug(f"Используем альтернативные колонки: {series_col}, {number_col}")
                
                for _, r in self.df.iterrows():
                    s = r.get(series_col, '')
                    n = r.get(number_col, 0)
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
            else:
                # Если информация о локомотивах не найдена, создаем фиктивные записи
                logger.warning("Информация о локомотивах не найдена в HTML файлах")
                
                # Создаем фиктивные локомотивы на основе маршрутов
                if 'Номер маршрута' in self.df.columns:
                    unique_routes = self.df['Номер маршрута'].dropna().unique()
                    for route_num in unique_routes[:10]:  # Ограничиваем количество
                        # Создаем фиктивную серию и номер
                        fake_series = "HTML"
                        fake_number = int(route_num) % 10000  # Используем номер маршрута
                        l = (fake_series, fake_number)
                        if l not in loc:
                            loc.append(l)
                            
                    logger.info(f"Создано {len(loc)} фиктивных записей локомотивов на основе маршрутов")
                else:
                    # Создаем минимальный набор фиктивных локомотивов
                    logger.warning("Создаем минимальный набор фиктивных локомотивов")
                    for i in range(5):
                        loc.append(("HTML", i + 1))
        
        loc.sort(key=lambda x: (x[0], x[1]))
        logger.info(f"Извлечено {len(loc)} локомотивов")
        return loc
    
    def get_locomotives_by_series(self) -> Dict[str, List[int]]:
        """Группировка локомотивов по сериям"""
        sd = {}
        for s, n in self.avl:
            if s not in sd:
                sd[s] = []
            sd[s].append(n)
        for s in sd:
            sd[s].sort()
        return sd
    
    def set_selected_locomotives(self, sel: List[Tuple[str, int]]):
        """Установка выбранных локомотивов"""
        self.sel = set(sel)
    
    def toggle_locomotive(self, s: str, n: int):
        """Переключение выбора локомотива"""
        l = (s, n)
        if l in self.sel:
            self.sel.remove(l)
        else:
            self.sel.add(l)
    
    def select_all_in_series(self, s: str):
        """Выбор всех локомотивов серии"""
        for se, n in self.avl:
            if se == s:
                self.sel.add((se, n))
    
    def deselect_all_in_series(self, s: str):
        """Отмена выбора всех локомотивов серии"""
        for se, n in self.avl:
            if se == s:
                self.sel.discard((se, n))
    
    def select_all(self):
        """Выбор всех локомотивов"""
        self.sel = set(self.avl)
    
    def deselect_all(self):
        """Отмена выбора всех локомотивов"""
        self.sel = set()
    
    def filter_routes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрация маршрутов по выбранным локомотивам"""
        if not self.sel:
            logger.debug("Нет выбранных локомотивов, возвращаем пустой DataFrame")
            return df.iloc[0:0]
            
        # Если в данных нет информации о локомотивах, возвращаем все данные
        if 'Серия локомотива' not in df.columns and 'Номер локомотива' not in df.columns:
            logger.debug("В данных нет информации о локомотивах, возвращаем все записи")
            return df
            
        m = pd.Series([False] * len(df))
        
        # Попытка фильтрации по стандартным колонкам
        if 'Серия локомотива' in df.columns and 'Номер локомотива' in df.columns:
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
        else:
            # Если нет информации о локомотивах, возвращаем все записи
            logger.debug("Фильтрация невозможна - отсутствуют данные о локомотивах")
            return df
            
        filtered_df = df[m]
        logger.debug(f"Отфильтровано {len(filtered_df)} записей из {len(df)}")
        return filtered_df
    
    def get_selection_statistics(self) -> Dict:
        """Статистика выбора"""
        return {
            'total_available': len(self.avl),
            'total_selected': len(self.sel),
            'series_count': len(set(s for s, _ in self.avl)),
            'selected_series': len(set(s for s, _ in self.sel))
        }