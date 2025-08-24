# core/norm_storage.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import os
import pickle
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class NormStorage:
    """Высокопроизводительное хранилище норм."""

    def __init__(self, storage_file: str = "norms_storage.pkl"):
        self.storage_file = storage_file
        self.norms_data: Dict[str, Dict] = {}
        self.norm_functions: Dict[str, Any] = {}  # Кэш интерполяционных функций
        self.metadata = {
            'version': '1.0',
            'total_norms': 0,
            'last_updated': None,
            'norm_types': {},
        }
        self.load_storage()

    def load_storage(self):
        """Загружает данные из файла хранилища."""
        if not os.path.exists(self.storage_file):
            logger.info(f"Файл хранилища {self.storage_file} не найден, создаем новое")
            return
        try:
            with open(self.storage_file, 'rb') as f:
                data = pickle.load(f)
            self.norms_data = data.get('norms_data', {})
            self.metadata = data.get('metadata', self.metadata)
            self._rebuild_interpolation_functions()
            logger.info(f"Загружено {len(self.norms_data)} норм из {self.storage_file}")
        except Exception as e:
            logger.error(f"Ошибка загрузки хранилища норм: {e}")
            self.norms_data = {}
            self.norm_functions = {}

    def save_storage(self):
        """Сохраняет данные в файл хранилища."""
        try:
            self.metadata['total_norms'] = len(self.norms_data)
            self.metadata['last_updated'] = datetime.now(timezone.utc).isoformat()

            norm_types: Dict[str, int] = {}
            for nd in self.norms_data.values():
                t = nd.get('norm_type', 'Unknown')
                norm_types[t] = norm_types.get(t, 0) + 1
            self.metadata['norm_types'] = norm_types

            data = {'norms_data': self.norms_data, 'metadata': self.metadata}
            with open(self.storage_file, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Хранилище норм сохранено в {self.storage_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения хранилища норм: {e}")

    def add_or_update_norms(self, new_norms: Dict[str, Dict]) -> Dict[str, str]:
        """Добавляет или обновляет нормы. Возвращает dict norm_id -> статус."""
        logger.info(f"Добавление/обновление {len(new_norms)} норм")
        update_results: Dict[str, str] = {}

        for norm_id, norm_data in new_norms.items():
            if norm_id in self.norms_data:
                if self._norms_are_different(norm_data, self.norms_data[norm_id]):
                    self.norms_data[norm_id] = norm_data
                    update_results[norm_id] = 'updated'
                else:
                    update_results[norm_id] = 'unchanged'
            else:
                self.norms_data[norm_id] = norm_data
                update_results[norm_id] = 'new'

        changed_ids = [nid for nid, status in update_results.items() if status in ('new', 'updated')]
        self._rebuild_interpolation_functions(changed_ids)
        self.save_storage()

        counts = Counter(update_results.values())
        logger.info(f"Результат обновления: {dict(counts)}")
        return update_results

    def get_norm(self, norm_id: str) -> Optional[Dict]:
        """Получить норму по ID."""
        return self.norms_data.get(norm_id)

    def get_all_norms(self) -> Dict[str, Dict]:
        """Получить все нормы."""
        return self.norms_data.copy()

    def _create_interpolation_function(self, points: List[Tuple[float, float]]):
        """Создает функцию интерполяции:
        - 1 точка  -> константа y = c
        - 2 точки  -> гипербола y = A/x + B (с устойчивым fallback на линейную интерполяцию)
        - 3+ точек -> подгонка гиперболы методом наименьших квадратов (fallback на первые 2 точки)
        Ограничение: X > 0 (иначе гипербола некорректна).
        """
        if len(points) < 1:
            raise ValueError("Недостаточно точек для интерполяции")

        pts = sorted(points, key=lambda t: t[0])
        x = np.array([p[0] for p in pts], dtype=float)
        y = np.array([p[1] for p in pts], dtype=float)

        if np.any(x <= 0):
            raise ValueError("Значения X должны быть положительными для гиперболы")

        if len(pts) == 1:
            c = float(y[0])
            return lambda z: float(c)

        if len(pts) == 2:
            x1, y1 = float(x[0]), float(y[0])
            x2, y2 = float(x[1]), float(y[1])
            try:
                if abs(x2 - x1) < 1e-12:
                    avg = (y1 + y2) / 2.0
                    return lambda z: float(avg)
                A = (y1 - y2) * x1 * x2 / (x2 - x1)
                B = (y2 * x2 - y1 * x1) / (x2 - x1)

                def f(z: float) -> float:
                    zf = float(z)
                    if zf <= 0:
                        return float(y1)
                    return float(A / zf + B)

                return f
            except Exception as e:
                logger.warning(f"Ошибка гиперболы (2 точки): {e}, fallback на линейную интерполяцию")
                return interp1d(x, y, kind='linear', fill_value='extrapolate', bounds_error=False)

        # 3+ точки
        try:
            from scipy.optimize import curve_fit

            def model(xx, A, B):
                return A / xx + B

            popt, _ = curve_fit(model, x, y, bounds=([-np.inf, -np.inf], [np.inf, np.inf]))
            A_opt, B_opt = popt

            def f(z: float) -> float:
                zf = float(z)
                if zf <= 0:
                    return float(y[0])
                return float(A_opt / zf + B_opt)

            return f
        except Exception as e:
            logger.warning(f"Ошибка подгонки гиперболы (3+): {e}, fallback на 2 точки")
            return self._create_interpolation_function(pts[:2])

    def _rebuild_interpolation_functions(self, norm_ids: Optional[List[str]] = None):
        """Пересоздает интерполяционные функции для всех или указанных норм."""
        if norm_ids is None:
            self.norm_functions.clear()
            norm_ids = list(self.norms_data.keys())
        logger.debug(f"Пересоздание функций интерполяции: {len(norm_ids)} норм")
        for norm_id in norm_ids:
            nd = self.norms_data.get(norm_id)
            try:
                pts = nd.get('points') if nd else None
                if pts:
                    self.norm_functions[norm_id] = self._create_interpolation_function(pts)
            except Exception as e:
                logger.error(f"Ошибка создания функции {norm_id}: {e}")

    def get_norm_function(self, norm_id: str):
        """Получает интерполяционную функцию для нормы (из кэша или построив)."""
        func = self.norm_functions.get(norm_id)
        if func is not None:
            return func
        nd = self.get_norm(norm_id)
        if nd and nd.get('points'):
            func = self._create_interpolation_function(nd['points'])
            self.norm_functions[norm_id] = func
            return func
        return None

    def interpolate_norm_value(self, norm_id: str, load_value: float) -> Optional[float]:
        """Интерполирует значение нормы для заданной нагрузки."""
        func = self.get_norm_function(norm_id)
        if func is None:
            return None
        try:
            return float(func(load_value))
        except Exception as e:
            logger.error(f"Ошибка интерполяции для {norm_id}: {e}")
            return None

    def search_norms(self, **criteria) -> Dict[str, Dict]:
        """Поиск норм по критериям (norm_type, norm_id_pattern, поля base_data)."""
        results: Dict[str, Dict] = {}
        for nid, nd in self.norms_data.items():
            ok = True
            for k, v in criteria.items():
                if k == 'norm_type':
                    if nd.get('norm_type') != v:
                        ok = False
                        break
                elif k == 'norm_id_pattern':
                    if v not in nid:
                        ok = False
                        break
                elif k in nd.get('base_data', {}):
                    if nd['base_data'].get(k) != v:
                        ok = False
                        break
            if ok:
                results[nid] = nd
        return results

    def get_norms_by_type(self, norm_type: str) -> Dict[str, Dict]:
        """Получает нормы по типу."""
        return self.search_norms(norm_type=norm_type)

    def get_storage_info(self) -> Dict:
        """Информация о хранилище."""
        return {
            **self.metadata,
            'storage_file': self.storage_file,
            'file_size_mb': (os.path.getsize(self.storage_file) / (1024 * 1024)) if os.path.exists(self.storage_file) else 0,
            'cached_functions': len(self.norm_functions),
        }

    def export_to_json(self, output_file: str) -> bool:
        """Экспортирует нормы в JSON."""
        try:
            export_data = {'metadata': self.metadata, 'norms': self.norms_data}
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Нормы экспортированы в JSON: {output_file}")
            return True
        except Exception as e:
            logger.error(f"Ошибка экспорта в JSON: {e}")
            return False

    def import_from_json(self, input_file: str) -> bool:
        """Импортирует нормы из JSON (с последующим обновлением кэша)."""
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            norms = data.get('norms', {})
            if not norms:
                logger.warning("В JSON нет норм для импорта")
                return False
            self.add_or_update_norms(norms)
            logger.info(f"Импортировано норм из JSON: {len(norms)}")
            return True
        except Exception as e:
            logger.error(f"Ошибка импорта из JSON: {e}")
            return False

    def validate_norms(self) -> Dict[str, List[str]]:
        """Валидация норм:
        - Мин. 1 точка
        - X > 0, Y > 0
        - Проверка построения функции
        """
        res = {'valid': [], 'invalid': [], 'warnings': []}
        for nid, nd in self.norms_data.items():
            try:
                points = nd.get('points', [])
                if len(points) < 1:
                    res['invalid'].append(f"Норма {nid}: нет точек")
                    continue
                if any((x <= 0 or y <= 0) for x, y in points):
                    res['invalid'].append(f"Норма {nid}: отрицательные или нулевые значения")
                    continue
                try:
                    self._create_interpolation_function(points)
                    res['valid'].append(nid)
                except Exception as e:
                    res['invalid'].append(f"Норма {nid}: ошибка интерполяции — {e}")
                    continue
                if len(points) > 20:
                    res['warnings'].append(f"Норма {nid}: много точек ({len(points)})")
                elif len(points) == 1:
                    res['warnings'].append(f"Норма {nid}: только одна точка (константа)")
            except Exception as e:
                res['invalid'].append(f"Норма {nid}: ошибка валидации — {e}")

        logger.info(f"Валидация: валидных={len(res['valid'])}, невалидных={len(res['invalid'])}, предупреждений={len(res['warnings'])}")
        return res

    def _norms_are_different(self, norm1: Dict, norm2: Dict) -> bool:
        """Сравнивает две нормы на предмет различий."""
        try:
            p1 = set(tuple(p) for p in norm1.get('points', []))
            p2 = set(tuple(p) for p in norm2.get('points', []))
            if p1 != p2:
                return True

            b1 = norm1.get('base_data', {}) or {}
            b2 = norm2.get('base_data', {}) or {}
            keys = set(b1.keys()) | set(b2.keys())
            for k in keys:
                if b1.get(k) != b2.get(k):
                    return True

            for k in ['norm_type', 'description']:
                if norm1.get(k) != norm2.get(k):
                    return True
            return False
        except Exception as e:
            logger.error(f"Ошибка сравнения норм: {e}")
            return True

    def cleanup_storage(self):
        """Очищает кэш функций для отсутствующих норм."""
        existing = set(self.norms_data.keys())
        cached = set(self.norm_functions.keys())
        for nid in (cached - existing):
            del self.norm_functions[nid]
        logger.debug(f"Очистка кэша: удалено {len(cached - existing)} функций")

    def get_norm_statistics(self) -> Dict:
        """Статистика по нормам: количество, распределение по типам/числу точек, диапазоны."""
        stats = {
            'total_norms': len(self.norms_data),
            'by_type': {},
            'points_distribution': {},
            'avg_points_per_norm': 0.0,
            'load_range': {'min': float('inf'), 'max': float('-inf')},
            'consumption_range': {'min': float('inf'), 'max': float('-inf')},
        }

        total_points = 0
        for nd in self.norms_data.values():
            t = nd.get('norm_type', 'Unknown')
            stats['by_type'][t] = stats['by_type'].get(t, 0) + 1

            pts = nd.get('points', []) or []
            total_points += len(pts)
            stats['points_distribution'][len(pts)] = stats['points_distribution'].get(len(pts), 0) + 1

            for load, cons in pts:
                stats['load_range']['min'] = min(stats['load_range']['min'], load)
                stats['load_range']['max'] = max(stats['load_range']['max'], load)
                stats['consumption_range']['min'] = min(stats['consumption_range']['min'], cons)
                stats['consumption_range']['max'] = max(stats['consumption_range']['max'], cons)

        if stats['total_norms'] > 0:
            stats['avg_points_per_norm'] = total_points / stats['total_norms']

        if stats['load_range']['min'] == float('inf'):
            stats['load_range'] = {'min': 0.0, 'max': 0.0}
        if stats['consumption_range']['min'] == float('inf'):
            stats['consumption_range'] = {'min': 0.0, 'max': 0.0}

        return stats