# analysis/analyzer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Основной анализатор норм расхода электроэнергии."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go

from analysis.data_analyzer import RouteDataAnalyzer, CoefficientsApplier
from analysis.html_route_processor import HTMLRouteProcessor
from analysis.html_norm_processor import HTMLNormProcessor
from analysis.visualization import PlotBuilder
from core.coefficients import LocomotiveCoefficientsManager
from core.filter import LocomotiveFilter
from core.norm_storage import NormStorage
from core.utils import extract_route_key, safe_int

logger = logging.getLogger(__name__)


class InteractiveNormsAnalyzer:
    """Интерактивный анализатор норм расхода электроэнергии."""

    def __init__(self):
        self.route_processor = HTMLRouteProcessor()
        self.norm_processor = HTMLNormProcessor()
        self.norm_storage = NormStorage()
        self.data_analyzer = RouteDataAnalyzer(self.norm_storage)
        self.plot_builder = PlotBuilder()
        self.plot_builder._analyzer = self 

        self.routes_df: Optional[pd.DataFrame] = None
        self.analyzed_results: Dict[str, Dict] = {}
        self.sections_norms_map: Dict[str, List[str]] = {}

        logger.info("Инициализирован анализатор норм")

    # ========================== Загрузка данных ==========================

    def load_routes_from_html(self, html_files: List[str]) -> bool:
        """Загружает маршруты из HTML файлов."""
        logger.info("Загрузка маршрутов из %d HTML файлов", len(html_files))
        
        try:
            self.routes_df = self.route_processor.process_html_files(html_files)
            if self.routes_df is None or self.routes_df.empty:
                logger.error("Не удалось загрузить маршруты")
                return False

            logger.info("Загружено записей: %d", len(self.routes_df))
            self._build_sections_norms_map()
            return True
            
        except Exception as e:
            logger.error("Ошибка загрузки маршрутов: %s", e, exc_info=True)
            return False

    def load_norms_from_html(self, html_files: List[str]) -> bool:
        """Загружает нормы из HTML файлов."""
        logger.info("Загрузка норм из %d HTML файлов", len(html_files))
        
        try:
            new_norms = self.norm_processor.process_html_files(html_files)
            if not new_norms:
                logger.warning("Не найдено норм в HTML файлах")
                return False

            self.norm_storage.add_or_update_norms(new_norms)
            stats = self.norm_processor.get_processing_stats()
            
            logger.info("Обработано норм: всего %d, новых %d, обновленных %d",
                       stats.get("total_norms_found", 0),
                       stats.get("new_norms", 0),
                       stats.get("updated_norms", 0))
            return True
            
        except Exception as e:
            logger.error("Ошибка загрузки норм: %s", e, exc_info=True)
            return False

    # ========================== Получение данных ==========================

    def get_sections_list(self) -> List[str]:
        """Возвращает список доступных участков."""
        if self.routes_df is None or self.routes_df.empty:
            return []
        return sorted(self.routes_df["Наименование участка"].dropna().unique().tolist())

    def get_norms_with_counts_for_section(self, section_name: str, single_section_only: bool = False) -> List[Tuple[str, int]]:
        """Возвращает список норм для участка с количествами маршрутов."""
        df = self._filter_section_routes(section_name, single_section_only)
        if df.empty:
            return []

        # Подсчитываем количество маршрутов по нормам
        norm_counts = df["Номер нормы"].dropna().apply(
            lambda x: str(safe_int(x))
        ).value_counts()

        # Получаем все известные нормы для участка
        known_norms = self.sections_norms_map.get(section_name, [])
        
        # Объединяем известные нормы с найденными
        all_norms = set(known_norms) | set(norm_counts.index)
        
        # Формируем результат с сортировкой
        result = [(norm, int(norm_counts.get(norm, 0))) for norm in all_norms]
        result.sort(key=lambda x: (safe_int(x[0]), x[0]))
        
        return result

    def get_routes_count_for_section(self, section_name: str, single_section_only: bool = False) -> int:
        """Возвращает общее количество маршрутов для участка."""
        df = self._filter_section_routes(section_name, single_section_only)
        return len(df)

    def get_norm_info(self, norm_id: str) -> Optional[Dict]:
        """Возвращает информацию о норме."""
        norm_data = self.norm_storage.get_norm(norm_id)
        if not norm_data:
            return None

        points = norm_data.get("points", [])
        
        info = {
            "norm_id": norm_id,
            "description": norm_data.get("description", f"Норма №{norm_id}"),
            "norm_type": norm_data.get("norm_type", "Неизвестно"),
            "points_count": len(points),
            "points": points,
            "base_data": norm_data.get("base_data", {}),
        }

        if points:
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            info["load_range"] = f"{min(x_vals):.1f} - {max(x_vals):.1f} т/ось"
            info["consumption_range"] = f"{min(y_vals):.1f} - {max(y_vals):.1f} кВт·ч/10⁴ ткм"
        else:
            info["load_range"] = "Нет данных"
            info["consumption_range"] = "Нет данных"

        return info

    # ========================== Анализ ==========================

    def analyze_section(
        self,
        section_name: str,
        norm_id: Optional[str] = None,
        single_section_only: bool = False,
        locomotive_filter: Optional[LocomotiveFilter] = None,
        coefficients_manager: Optional[LocomotiveCoefficientsManager] = None,
        use_coefficients: bool = False,
    ) -> Tuple[Optional[go.Figure], Optional[Dict], Optional[str]]:
        """Анализирует участок с построением интерактивного графика."""
        logger.info("Анализ участка: %s, норма: %s, только один участок: %s",
                   section_name, norm_id, single_section_only)

        if self.routes_df is None or self.routes_df.empty:
            return None, None, "Данные маршрутов не загружены"

        try:
            # Фильтрация данных
            section_routes = self._prepare_section_data(
                section_name, norm_id, single_section_only,
                locomotive_filter, coefficients_manager, use_coefficients
            )
            
            if section_routes.empty:
                return None, None, self._get_empty_data_message(section_name, norm_id, single_section_only)

            # Анализ данных
            analyzed_data, norm_functions = self.data_analyzer.analyze_section_data(
                section_name, section_routes, norm_id
            )
            
            if analyzed_data.empty:
                return None, None, f"Не удалось проанализировать участок {section_name}"

            # Построение графика
            fig = self.plot_builder.create_interactive_plot(
                section_name, analyzed_data, norm_functions, norm_id, single_section_only
            )
            
            # Статистика
            statistics = self.data_analyzer.calculate_statistics(analyzed_data)

            # Сохраняем результат
            result_key = f"{section_name}_{norm_id}_{single_section_only}"
            self.analyzed_results[result_key] = {
                "routes": analyzed_data,
                "norms": norm_functions,
                "statistics": statistics
            }

            logger.info("Анализ участка %s завершен", section_name)
            return fig, statistics, None

        except Exception as e:
            logger.error("Ошибка анализа участка %s: %s", section_name, e, exc_info=True)
            return None, None, f"Ошибка анализа: {str(e)}"

    # ========================== Экспорт и утилиты ==========================

    def export_routes_to_excel(self, output_file: str) -> bool:
        """Экспортирует данные маршрутов в Excel."""
        if self.routes_df is None or self.routes_df.empty:
            logger.warning("Нет данных для экспорта")
            return False
        return self.route_processor.export_to_excel(self.routes_df, output_file)

    def get_routes_data(self) -> pd.DataFrame:
        """Возвращает копию данных маршрутов."""
        return self.routes_df.copy() if self.routes_df is not None else pd.DataFrame()

    def get_norm_storage_info(self) -> Dict:
        """Возвращает информацию о хранилище норм."""
        return self.norm_storage.get_storage_info()

    def get_norm_storage_statistics(self) -> Dict:
        """Возвращает статистику хранилища норм."""
        return self.norm_storage.get_norm_statistics()

    def validate_norms_storage(self) -> Dict:
        """Валидирует хранилище норм."""
        return self.norm_storage.validate_norms()

    def add_browser_controls(self, html_content: str) -> str:
        """Добавляет браузерные контролы к HTML."""
        return self.plot_builder.add_browser_controls(html_content)

    # ========================== Внутренние методы ==========================

    def _build_sections_norms_map(self) -> None:
        """Строит карту участков -> список норм."""
        if self.routes_df is None or self.routes_df.empty:
            self.sections_norms_map = {}
            return

        # Группируем нормы по участкам
        grouped = self.routes_df.groupby("Наименование участка")["Номер нормы"].apply(
            lambda series: sorted({
                str(safe_int(x)) for x in series.dropna() 
                if pd.notna(x) and str(x).strip()
            })
        ).to_dict()
        
        self.sections_norms_map = grouped
        logger.info("Построена карта участков и норм: %d участков", len(self.sections_norms_map))

    def _filter_section_routes(self, section_name: str, single_section_only: bool = False) -> pd.DataFrame:
        """Фильтрует маршруты по участку и условию 'только один участок'."""
        if self.routes_df is None or self.routes_df.empty:
            return pd.DataFrame()

        # Базовая фильтрация по участку
        df = self.routes_df[self.routes_df["Наименование участка"] == section_name].copy()
        
        if df.empty or not single_section_only:
            return df

        # Фильтр "только один участок"
        route_counts = self.routes_df.groupby(["Номер маршрута", "Дата маршрута"]).size()
        single_section_routes = route_counts[route_counts == 1].index
        
        # Применяем фильтр
        df_indexed = df.set_index(["Номер маршрута", "Дата маршрута"])
        filtered_df = df_indexed.loc[df_indexed.index.intersection(single_section_routes)]
        
        return filtered_df.reset_index()

    def _prepare_section_data(
        self,
        section_name: str,
        norm_id: Optional[str],
        single_section_only: bool,
        locomotive_filter: Optional[LocomotiveFilter],
        coefficients_manager: Optional[LocomotiveCoefficientsManager],
        use_coefficients: bool,
    ) -> pd.DataFrame:
        """Подготавливает данные участка с применением всех фильтров."""
        # Базовая фильтрация
        df = self._filter_section_routes(section_name, single_section_only)
        if df.empty:
            return df

        # Фильтрация по норме
        if norm_id:
            norm_str = str(safe_int(norm_id))
            df = df[df["Номер нормы"].apply(lambda x: str(safe_int(x)) == norm_str)]
            if df.empty:
                return df

        # Фильтрация локомотивов
        if locomotive_filter:
            df = locomotive_filter.filter_routes(df)
            if df.empty:
                return df

        # Применение коэффициентов
        if use_coefficients and coefficients_manager:
            df = CoefficientsApplier.apply_coefficients(df, coefficients_manager)

        return df

    def _get_empty_data_message(self, section_name: str, norm_id: Optional[str], single_section_only: bool) -> str:
        """Формирует сообщение об отсутствии данных."""
        base_msg = f"Нет данных для участка {section_name}"
        
        if single_section_only:
            base_msg = f"Нет маршрутов с одним участком для {section_name}"
        
        if norm_id:
            suffix = " с одним участком" if single_section_only else ""
            base_msg = f"Нет маршрутов{suffix} для участка {section_name} с нормой {norm_id}"
        
        return base_msg

    # Метод для совместимости (используется в GUI)
    def _add_browser_mode_switcher(self, html_content: str) -> str:
        """Устаревший метод - делегируем в PlotBuilder."""
        return self.add_browser_controls(html_content)