# core/filter.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LocomotiveFilter:
    """Фильтр выбора локомотивов с поддержкой разных форматов входных данных."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._mode: str = 'minimal'  # 'standard' | 'guess' | 'depot' | 'minimal'
        self._series_col: str | None = None
        self._number_col: str | None = None

        self.avl: List[Tuple[str, int]] = self._extract_locomotives()
        # По умолчанию всё выбрано
        self.sel = set(self.avl)

    @staticmethod
    def _to_int_series(s: pd.Series) -> pd.Series:
        """Безопасное преобразование к целому."""
        vals = pd.to_numeric(s, errors='coerce')
        return vals.dropna().astype(np.int64)

    def _extract_locomotives(self) -> List[Tuple[str, int]]:
        """Извлекает уникальные локомотивы векторизованно и определяет режим ключей."""
        cols = list(self.df.columns)
        logger.debug(f"Извлечение локомотивов из DataFrame, колонки: {cols}")

        # 1) Стандартные колонки
        if 'Серия локомотива' in self.df.columns and 'Номер локомотива' in self.df.columns:
            self._mode = 'standard'
            self._series_col, self._number_col = 'Серия локомотива', 'Номер локомотива'
            sub = self.df[[self._series_col, self._number_col]].copy()
            sub[self._series_col] = sub[self._series_col].astype(str)
            sub[self._number_col] = pd.to_numeric(sub[self._number_col], errors='coerce')
            sub = sub.dropna()
            sub[self._number_col] = sub[self._number_col].astype(np.int64)
            pairs = sub.drop_duplicates().values.tolist()
            loc = [(str(s), int(n)) for s, n in pairs]

        else:
            # 2) Пытаемся угадать пары колонок
            possible_series_cols = [c for c in cols if any(k in str(c).lower() for k in ['серия', 'series', 'локо', 'депо'])]
            possible_number_cols = [c for c in cols if any(k in str(c).lower() for k in ['номер', 'number', 'num', 'маршрут'])]

            if possible_series_cols and possible_number_cols:
                self._mode = 'guess'
                self._series_col = possible_series_cols[0]
                self._number_col = possible_number_cols[0]
                sub = self.df[[self._series_col, self._number_col]].copy()
                sub[self._series_col] = sub[self._series_col].astype(str)
                sub[self._number_col] = pd.to_numeric(sub[self._number_col], errors='coerce')
                sub = sub.dropna()
                sub[self._number_col] = (sub[self._number_col] % 10000).astype(np.int64)
                pairs = sub.drop_duplicates().values.tolist()
                loc = [(str(s), int(n)) for s, n in pairs]

            elif 'Депо' in self.df.columns and 'Номер маршрута' in self.df.columns:
                # 3) Фиктивный ключ (депо/маршрут)
                self._mode = 'depot'
                depot = self.df['Депо'].astype(str).str.strip()
                depot_first = depot.str.split().str[0]
                depot_num = pd.to_numeric(depot_first, errors='coerce')
                series = np.where(depot_num.notna(), 'Д' + depot_num.astype(np.int64).astype(str), 'HTML')

                route = pd.to_numeric(self.df['Номер маршрута'], errors='coerce')
                number = (route % 1000).fillna(-1).astype(np.int64)

                sub = pd.DataFrame({'series': series, 'number': number})
                sub = sub[(sub['number'] >= 0)]
                pairs = sub.drop_duplicates().values.tolist()
                loc = [(str(s), int(n)) for s, n in pairs]

            else:
                # 4) Минимальный набор
                self._mode = 'minimal'
                loc = [('HTML', i + 1) for i in range(5)]

        # Упорядочиваем и возвращаем
        loc = list(set(loc))
        loc.sort(key=lambda x: (x[0], x[1]))
        logger.info(f"Извлечено {len(loc)} уникальных локомотивов (режим: {self._mode})")
        if loc:
            logger.debug(f"Примеры: {loc[:5]}")
        return loc

    def get_locomotives_by_series(self) -> Dict[str, List[int]]:
        """Группировка доступных локомотивов по сериям."""
        result: Dict[str, List[int]] = {}
        for s, n in self.avl:
            result.setdefault(s, []).append(n)
        for s in result:
            result[s].sort()
        return result

    def set_selected_locomotives(self, sel: List[Tuple[str, int]]):
        """Установка выбранных локомотивов."""
        self.sel = set((str(s), int(n)) for s, n in sel)
        logger.debug(f"Установлено {len(self.sel)} выбранных локомотивов")

    def select_all(self):
        """Выбрать все доступные локомотивы."""
        self.sel = set(self.avl)

    def deselect_all(self):
        """Снять выбор со всех локомотивов."""
        self.sel = set()

    # Вспомогательная функция для векторной фильтрации
    @staticmethod
    def _filter_by_join(df: pd.DataFrame, left_keys: List[str], right_pairs: List[Tuple]) -> pd.DataFrame:
        if not right_pairs:
            return df.iloc[0:0]
        sel_df = pd.DataFrame(right_pairs, columns=left_keys).drop_duplicates()
        tmp = df[left_keys].copy()
        tmp = tmp.reset_index().merge(sel_df, how='inner', on=left_keys)
        idx = tmp['index'].values
        return df.loc[idx].copy()

    def filter_routes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Фильтрация маршрутов по выбранным локомотивам (векторно).
        - Если нет выбранных — пустой результат.
        - Если в данных невозможно восстановить ключ — возвращаем исходный df (как в исходной логике).
        """
        if not self.sel:
            logger.debug("Нет выбранных локомотивов — возвращаем пустой DataFrame")
            return df.iloc[0:0]

        if self._mode == 'standard':
            if not {'Серия локомотива', 'Номер локомотива'}.issubset(df.columns):
                logger.debug("Стандартные колонки не найдены — возврат исходного df")
                return df
            work = df[['Серия локомотива', 'Номер локомотива']].copy()
            work['Серия локомотива'] = work['Серия локомотива'].astype(str)
            work['Номер локомотива'] = pd.to_numeric(work['Номер локомотива'], errors='coerce').astype('Int64')
            work = work.dropna()
            left = ['Серия локомотива', 'Номер локомотива']
            right = list(self.sel)
            return self._filter_by_join(work.join(df.drop(columns=left, errors='ignore')), left, right)

        if self._mode == 'guess':
            if self._series_col not in df.columns or self._number_col not in df.columns:
                logger.debug("Колонки для угадывания не найдены — возврат исходного df")
                return df
            work = df[[self._series_col, self._number_col]].copy()
            work[self._series_col] = work[self._series_col].astype(str)
            work[self._number_col] = pd.to_numeric(work[self._number_col], errors='coerce')
            work[self._number_col] = (work[self._number_col] % 10000).astype('Int64')
            work = work.dropna()
            left = [self._series_col, self._number_col]
            right = list(self.sel)
            return self._filter_by_join(work.join(df.drop(columns=left, errors='ignore')), left, right)

        if self._mode == 'depot':
            if not {'Депо', 'Номер маршрута'}.issubset(df.columns):
                logger.debug("Нет колонок депо/маршрут — возврат исходного df")
                return df
            depot = df['Депо'].astype(str).str.strip()
            depot_first = depot.str.split().str[0]
            depot_num = pd.to_numeric(depot_first, errors='coerce')
            series = np.where(depot_num.notna(), 'Д' + depot_num.astype(np.int64).astype(str), 'HTML')
            route = pd.to_numeric(df['Номер маршрута'], errors='coerce')
            number = (route % 1000).astype('Int64')
            work = pd.DataFrame({'series': series, 'number': number})
            work = work.dropna()
            left = ['series', 'number']
            right = list(self.sel)
            # Присоединяем рассчитанные ключи к исходным данным на основании индексов
            filtered = self._filter_by_join(work.join(df, how='left'), left, right)
            # Возвращаем исходные колонки
            return filtered[df.columns]

        # minimal — невозможно фильтровать осмысленно
        logger.debug("Минимальный режим — фильтрация невозможна, возвращаем исходный df")
        return df