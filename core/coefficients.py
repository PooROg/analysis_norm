# core/coefficients.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LocomotiveCoefficientsManager:
    """Менеджер коэффициентов расхода локомотивов."""

    def __init__(self):
        self.file: Optional[str] = None
        # Ключ: (нормализованная серия, номер), значение: коэффициент
        self._coef: Dict[Tuple[str, int], float] = {}
        # Данные по сериям: ключ — нормализованная серия, значение — список записей
        self._data: Dict[str, List[Dict]] = {}

    @staticmethod
    def normalize_series(series: str) -> str:
        """Нормализация названия серии локомотива: убираем все кроме букв/цифр, верхний регистр."""
        return re.sub(r'[^А-ЯA-Zа-яa-z0-9]', '', str(series).upper())

    def load_coefficients(self, fp: str, min_work_threshold: float = 0.0) -> bool:
        """Загрузка коэффициентов из Excel (все листы).
        Используются только необходимые параметры:
        - Заводской номер локомотива
        - Коэффициент ИЛИ Процент (приоритет у 'Коэффициент', 'Процент' — отдельный источник)
        - Работа (опционально, для фильтрации)
        """
        self.file = fp
        self._coef.clear()
        self._data.clear()

        try:
            excel = pd.ExcelFile(fp)
        except Exception as e:
            logger.error(f"Не удалось открыть Excel: {fp}: {e}")
            return False

        total_processed = 0

        for sheet in excel.sheet_names:
            # Извлекаем серию из имени листа: буквы + цифры (+буквы)
            series_name = sheet.strip()
            m = re.search(r'[А-ЯA-Z]+[\d]+[А-ЯA-Z]*', series_name)
            if m:
                series_name = m.group()
            series_norm = self.normalize_series(series_name)

            try:
                raw = pd.read_excel(excel, sheet_name=sheet, header=None)
                if raw.empty:
                    continue
            except Exception as e:
                logger.warning(f"Лист '{sheet}': ошибка чтения: {e}")
                continue

            # Определяем строку заголовков среди первых 10 строк
            header_row = None
            max_scan = min(10, len(raw))
            for i in range(max_scan):
                row = raw.iloc[i].astype(str).str.lower()
                if row.apply(lambda s: ('завод' in s) and ('номер' in s)).any():
                    header_row = i
                    break
            if header_row is None:
                logger.info(f"Лист '{sheet}': не найдена строка заголовков — пропуск")
                continue

            # Формируем DataFrame с заголовками из найденной строки
            headers = raw.iloc[header_row].tolist()
            df = raw.iloc[header_row + 1:].copy()
            df.columns = headers
            if df.empty:
                continue

            # Определяем нужные колонки
            cols_lower = {c: str(c).lower() for c in df.columns}
            locomotive_col = next(
                (c for c in df.columns if ('завод' in cols_lower[c] and 'номер' in cols_lower[c])),
                None,
            )
            # Колонка с коэффициентом (не проценты!)
            coefficient_col = next(
                (c for c in df.columns if ('коэффициент' in cols_lower[c] or 'коэфф' in cols_lower[c])),
                None,
            )
            # Колонка с процентами/%, отдельный источник
            percent_col = next(
                (c for c in df.columns if ('процент' in cols_lower[c] or '%' in str(c))),
                None,
            )
            work_col = next((c for c in df.columns if 'работа' in cols_lower[c]), None)

            if not locomotive_col or (not coefficient_col and not percent_col):
                logger.info(f"Лист '{sheet}': не найдены необходимые колонки — пропуск")
                continue

            # Парсим номера локомотивов
            num_raw = df[locomotive_col].astype(str).str.strip()
            numbers = pd.to_numeric(num_raw, errors='coerce')
            numbers = numbers.where(numbers > 0)  # Убираем 0 и NaN

            # Источник значений коэффициента:
            # 1) Если есть 'Коэффициент' — берем как есть (без деления на 100)
            # 2) Иначе используем 'Процент' с преобразованием: >10 => делим на 100
            if coefficient_col:
                coef_raw = df[coefficient_col].astype(str).str.replace(',', '.', regex=False).str.strip()
                coef = pd.to_numeric(coef_raw, errors='coerce')
                source = 'coefficient'
            else:
                p_raw = df[percent_col].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False).str.strip()
                coef = pd.to_numeric(p_raw, errors='coerce')
                coef = coef.where(coef <= 10, coef / 100)  # >10 считаем процентами
                source = 'percent'

            # Работа (опционально)
            if work_col and min_work_threshold > 0:
                work = pd.to_numeric(df[work_col], errors='coerce').fillna(0.0)
            else:
                work = pd.Series(0.0, index=df.index)

            mask_valid = numbers.notna() & coef.notna()
            if min_work_threshold > 0 and work_col:
                mask_valid &= work >= float(min_work_threshold)

            if not mask_valid.any():
                logger.info(f"Лист '{sheet}': нет валидных строк после фильтрации")
                continue

            res = pd.DataFrame(
                {
                    'series': series_name,
                    'series_normalized': series_norm,
                    'number': numbers.round().astype(int),
                    'coefficient': coef.astype(float),
                    'deviation_percent': (coef - 1.0) * 100.0,
                    'work_total': work.astype(float),
                }
            )[mask_valid]

            # Сохраняем: ключ (series_norm, number), «последний побеждает»
            items = res.to_dict(orient='records')
            if items:
                for rec in items:
                    self._coef[(series_norm, rec['number'])] = rec['coefficient']
                self._data.setdefault(series_norm, []).extend(items)
                total_processed += len(items)
                logger.debug(
                    f"Лист '{sheet}': загружено {len(items)} локомотивов (серия {series_name}, источник={source})"
                )

        logger.info(f"Загружено всего {total_processed} локомотивов из {len(self._data)} серий")
        return total_processed > 0

    def get_coefficient(self, series: str, number: int) -> float:
        """Возвращает коэффициент для локомотива. Если нет — 1.0."""
        key = (self.normalize_series(series), int(number))
        return float(self._coef.get(key, 1.0))

    def get_locomotives_by_series(self, series: str) -> List[Dict]:
        """Возвращает список записей по серии (по нормализованной серии)."""
        return self._data.get(self.normalize_series(series), [])

    def get_statistics(self) -> Dict:
        """Сводная статистика по коэффициентам."""
        if not self._coef:
            return {}
        coefs = np.array(list(self._coef.values()), dtype=float)
        deviations = (coefs - 1.0) * 100.0
        return {
            'total_locomotives': int(coefs.size),
            'series_count': int(len(self._data)),
            'avg_coefficient': float(np.mean(coefs)),
            'min_coefficient': float(np.min(coefs)),
            'max_coefficient': float(np.max(coefs)),
            'avg_deviation_percent': float(np.mean(deviations)),
            'locomotives_above_norm': int(np.sum(coefs > 1.0)),
            'locomotives_below_norm': int(np.sum(coefs < 1.0)),
            'locomotives_at_norm': int(np.sum(np.isclose(coefs, 1.0, atol=1e-3))),
        }