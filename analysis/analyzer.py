# analysis/analyzer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from typing import List, Dict, Optional, Tuple, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager
from core.norm_storage import NormStorage
from analysis.html_route_processor import HTMLRouteProcessor
from analysis.html_norm_processor import HTMLNormProcessor

logger = logging.getLogger(__name__)


class InteractiveNormsAnalyzer:
    """Интерактивный анализ норм расхода электроэнергии."""

    def __init__(self):
        self.route_processor = HTMLRouteProcessor()
        self.norm_processor = HTMLNormProcessor()
        self.norm_storage = NormStorage()

        self.routes_df: Optional[pd.DataFrame] = None
        self.analyzed_results: Dict[str, Dict] = {}
        self.sections_norms_map: Dict[str, List[str]] = {}

        logger.info("Инициализирован анализатор норм")

    # -------------------------- Публичные API-методы --------------------------

    def load_routes_from_html(self, html_files: List[str]) -> bool:
        """Загрузка маршрутов из HTML с полной обработкой (как в route_processor)."""
        logger.info(f"Загрузка маршрутов из {len(html_files)} HTML файлов")
        try:
            self.routes_df = self.route_processor.process_html_files(html_files)
            if self.routes_df is None or self.routes_df.empty:
                logger.error("Не удалось загрузить маршруты из HTML файлов")
                return False

            logger.info(f"Загружено записей: {len(self.routes_df)}")
            self._log_routes_statistics()
            self._build_sections_norms_map()
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки маршрутов: {e}", exc_info=True)
            return False

    def load_norms_from_html(self, html_files: List[str]) -> bool:
        """Загрузка норм из HTML и помещение в хранилище норм."""
        logger.info(f"Загрузка норм из {len(html_files)} HTML файлов")
        try:
            new_norms = self.norm_processor.process_html_files(html_files)
            if not new_norms:
                logger.warning("Не найдено норм в HTML файлах")
                return False

            # Добавляем/обновляем нормы в хранилище
            self.norm_storage.add_or_update_norms(new_norms)

            stats = self.norm_processor.get_processing_stats()
            logger.info(
                "Обработано норм: всего %s, новых %s, обновленных %s",
                stats.get("total_norms_found", 0),
                stats.get("new_norms", 0),
                stats.get("updated_norms", 0),
            )
            return True
        except Exception as e:
            logger.error(f"Ошибка загрузки норм: {e}", exc_info=True)
            return False

    def get_sections_list(self) -> List[str]:
        """Список доступных участков."""
        if self.routes_df is None or self.routes_df.empty:
            return []
        sections = self.routes_df["Наименование участка"].dropna().unique().tolist()
        return sorted(sections)

    def get_norms_for_section(self, section_name: str) -> List[str]:
        """Список норм для участка."""
        return self.sections_norms_map.get(section_name, [])

    def get_norms_with_counts_for_section(
        self, section_name: str, single_section_only: bool = False
    ) -> List[Tuple[str, int]]:
        """Список норм для участка с количествами маршрутов."""
        df = self._filter_section_routes(section_name, single_section_only)
        if df.empty:
            return []

        counts = (
            df["Номер нормы"]
            .dropna()
            .apply(lambda x: str(int(x)) if str(x).isdigit() else str(x))
            .value_counts()
        )
        norms = self.get_norms_for_section(section_name)
        # Если карта норм не содержит все встречающиеся — добавим их
        for norm in counts.index:
            if norm not in norms:
                norms.append(norm)

        def sort_key(x: str) -> Tuple[int, float]:
            return (0, int(x)) if x.isdigit() else (1, float("inf"))

        return sorted([(n, int(counts.get(n, 0))) for n in set(norms)], key=lambda x: sort_key(x[0]))

    def get_routes_count_for_section(self, section_name: str, single_section_only: bool = False) -> int:
        """Общее количество строк-маршрутов для участка (как в исходной логике)."""
        df = self._filter_section_routes(section_name, single_section_only)
        return int(len(df))

    def get_norm_routes_count_for_section(
        self, section_name: str, norm_id: str, single_section_only: bool = False
    ) -> int:
        """Количество маршрутов для нормы на участке."""
        df = self._filter_section_routes(section_name, single_section_only)
        if df.empty:
            return 0
        try:
            # Сравниваем как строки — устойчивей к типам
            norm_str = str(int(norm_id)) if str(norm_id).isdigit() else str(norm_id)
            return int((df["Номер нормы"].apply(lambda x: str(int(x)) if pd.notna(x) and str(x).isdigit() else str(x)) == norm_str).sum())
        except Exception as e:
            logger.error(f"Ошибка получения количества маршрутов для нормы {norm_id}: {e}", exc_info=True)
            return 0

    def get_norm_info(self, norm_id: str) -> Optional[Dict]:
        """Информация о норме."""
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

    def analyze_section(
        self,
        section_name: str,
        norm_id: Optional[str] = None,
        single_section_only: bool = False,
        locomotive_filter: Optional[LocomotiveFilter] = None,
        coefficients_manager: Optional[LocomotiveCoefficientsManager] = None,
        use_coefficients: bool = False,
    ) -> Tuple[Optional[go.Figure], Optional[Dict], Optional[str]]:
        """Анализ участка с возможностью выбора нормы и фильтрации по одному участку."""
        logger.info(
            "Анализ участка: %s, норма: %s, только один участок: %s",
            section_name, norm_id, single_section_only
        )

        if self.routes_df is None or self.routes_df.empty:
            return None, None, "Данные маршрутов не загружены"

        try:
            # 1) Базовая фильтрация по участку (+ один участок при необходимости)
            section_routes = self._filter_section_routes(section_name, single_section_only)
            if section_routes.empty:
                msg = f"Нет данных для участка {section_name}"
                if single_section_only:
                    msg = f"Нет маршрутов с одним участком для {section_name}"
                return None, None, msg

            # 2) Фильтрация по норме (если указана)
            if norm_id:
                norm_str = str(int(norm_id)) if str(norm_id).isdigit() else str(norm_id)
                section_routes = section_routes[
                    section_routes["Номер нормы"].apply(lambda x: str(int(x)) if pd.notna(x) and str(x).isdigit() else str(x)) == norm_str
                ]
                if section_routes.empty:
                    suffix = " с одним участком" if single_section_only else ""
                    return None, None, f"Нет маршрутов{suffix} для участка {section_name} с нормой {norm_id}"

            # 3) Фильтр локомотивов (если есть)
            if locomotive_filter:
                section_routes = locomotive_filter.filter_routes(section_routes)
                if section_routes.empty:
                    return None, None, "Нет данных после применения фильтра локомотивов"

            # 4) Применяем коэффициенты (если нужно)
            if use_coefficients and coefficients_manager:
                section_routes = self._apply_coefficients(section_routes, coefficients_manager)

            # 5) Анализ норм и интерполяция
            analyzed_data, norm_functions = self._analyze_section_data(section_name, section_routes, norm_id)
            if analyzed_data.empty:
                return None, None, f"Не удалось проанализировать участок {section_name}"

            # 6) Построение графика и статистика
            fig = self._create_interactive_plot(section_name, analyzed_data, norm_functions, norm_id, single_section_only)
            statistics = self._calculate_section_statistics(analyzed_data)

            key = f"{section_name}_{norm_id}_{single_section_only}" if norm_id else f"{section_name}_{single_section_only}"
            self.analyzed_results[key] = {"routes": analyzed_data, "norms": norm_functions, "statistics": statistics}

            logger.info("Анализ участка %s завершен", section_name)
            return fig, statistics, None

        except Exception as e:
            logger.error(f"Ошибка анализа участка {section_name}: {e}", exc_info=True)
            return None, None, f"Ошибка анализа: {str(e)}"

    def get_norm_storage_info(self) -> Dict:
        """Информация о хранилище норм."""
        return self.norm_storage.get_storage_info()

    def export_routes_to_excel(self, output_file: str) -> bool:
        """Экспорт маршрутов в Excel с форматированием (route_processor)."""
        if self.routes_df is None or self.routes_df.empty:
            logger.warning("Нет данных маршрутов для экспорта")
            return False
        return self.route_processor.export_to_excel(self.routes_df, output_file)

    def validate_norms_storage(self) -> Dict:
        """Валидация хранилища норм."""
        return self.norm_storage.validate_norms()

    def get_norm_storage_statistics(self) -> Dict:
        """Статистика хранилища норм."""
        return self.norm_storage.get_norm_statistics()

    def get_routes_data(self) -> pd.DataFrame:
        """Полные данные маршрутов (копия)."""
        return self.routes_df.copy() if self.routes_df is not None else pd.DataFrame()

    def get_norm_routes_count(self, norm_id: str) -> int:
        """Количество маршрутов для нормы (во всех участках)."""
        if self.routes_df is None or self.routes_df.empty:
            return 0
        norm_str = str(int(norm_id)) if str(norm_id).isdigit() else str(norm_id)
        return int((self.routes_df["Номер нормы"].apply(
            lambda x: str(int(x)) if pd.notna(x) and str(x).isdigit() else str(x)
        ) == norm_str).sum())

    # Требуется внешним кодом (оставляем неизменным названием)
    def _add_browser_mode_switcher(self, html_content: str) -> str:
        """Добавляет переключатель режима отображения точек и модальное окно подробностей в HTML."""
        js_code = '''
        <div id="mode-switcher" style="
            position: fixed; top: 10px; right: 10px; z-index: 1000; background: white; padding: 15px;
            border: 2px solid #4a90e2; border-radius: 10px; box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            font-family: Arial, sans-serif; font-size: 14px;">
            <h4 style="margin: 0 0 12px 0; color: #333;">Режим отображения точек:</h4>
            <label style="display: block; margin-bottom: 8px; cursor: pointer;">
                <input type="radio" name="display_mode" value="work" checked style="margin-right: 8px;">
                <strong>Уд. на работу</strong> (текущий)
            </label>
            <label style="display: block; cursor: pointer;">
                <input type="radio" name="display_mode" value="nf" style="margin-right: 8px;">
                <strong>Н/Ф</strong> (по соотношению норма/факт)
            </label>
        </div>

        <div id="route-modal" style="display:none; position:fixed; z-index:2000; left:0; top:0; width:100%; height:100%;
            background-color:rgba(0,0,0,0.5);">
            <div id="modal-content" style="background-color:white; margin:2% auto; padding:20px; border-radius:10px;
                width:95%; max-width:1400px; max-height:90%; overflow-y:auto; position:relative;">
                <span id="close-modal" style="position:absolute; right:20px; top:15px; font-size:28px; font-weight:bold;
                    cursor:pointer; color:#aaa; user-select:none;">&times;</span>
                <div id="route-details"></div>
            </div>
        </div>

        <script>
        let plotlyDiv = null;
        let originalData = {};
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(initializePlotly, 1000);
            setupModal();
            setupModeSwitch();
        });
        function initializePlotly() {
            plotlyDiv = document.getElementsByClassName('plotly-graph-div')[0];
            if (!plotlyDiv) { setTimeout(initializePlotly, 2000); return; }
            if (!plotlyDiv.on)   { setTimeout(initializePlotly, 2000); return; }
            plotlyDiv.on('plotly_click', handlePointClick);
            if (plotlyDiv.data) {
                plotlyDiv.data.forEach((trace, index) => {
                    originalData[index] = { x:[...(trace.x||[])], y:[...(trace.y||[])],
                        customdata: trace.customdata ? JSON.parse(JSON.stringify(trace.customdata)) : null };
                });
            }
        }
        function setupModal() {
            const modal = document.getElementById('route-modal');
            const closeBtn = document.getElementById('close-modal');
            if (closeBtn) closeBtn.onclick = (e)=>{ e.preventDefault(); e.stopPropagation(); modal.style.display='none'; return false; };
            modal.onclick = (e)=>{ if (e.target===modal) modal.style.display='none'; };
            document.addEventListener('keydown', (e)=>{ if(e.key==='Escape' && modal.style.display==='block') modal.style.display='none';});
        }
        function setupModeSwitch() {
            document.querySelectorAll('input[name="display_mode"]').forEach(r => r.addEventListener('change', switchDisplayMode));
        }
        function switchDisplayMode() {
            const mode = document.querySelector('input[name="display_mode"]:checked').value;
            if (!plotlyDiv || !originalData) return;
            const update = {};
            plotlyDiv.data.forEach((trace, index) => {
                if (!originalData[index] || !trace.customdata) return;
                if (mode === 'nf') {
                    const newY = (trace.y || []).map((originalY, i) => {
                        const c = trace.customdata[i];
                        if (c && c.rashod_fact != null && c.rashod_norm != null && c.norm_interpolated != null && c.rashod_norm > 0) {
                            const dev = ((c.rashod_fact - c.rashod_norm) / c.rashod_norm) * 100;
                            return c.norm_interpolated * (1 + dev / 100);
                        }
                        return originalY;
                    });
                    update['y[' + index + ']'] = newY;
                } else {
                    update['y[' + index + ']'] = originalData[index].y;
                }
            });
            if (Object.keys(update).length) Plotly.restyle(plotlyDiv, update);
        }
        function handlePointClick(data) {
            if (!data.points || !data.points.length) return;
            const customData = data.points[0].customdata;
            if (!customData) return;
            showFullRouteInfo(customData);
        }
        function showFullRouteInfo(c) {
            let html = `<h2>Подробная информация о маршруте №${c.route_number}</h2>`;
            html += `<div style="margin-bottom:20px;"><h3>Основная информация</h3>`;
            html += `<table style="border-collapse:collapse;width:50%;font-family:Arial;">`;
            html += addTableRow('Номер маршрута', c.route_number);
            html += addTableRow('Дата маршрута', c.route_date);
            html += addTableRow('Дата поездки', c.trip_date);
            html += addTableRow('Табельный машиниста', c.driver_tab);
            html += addTableRow('Серия локомотива', c.locomotive_series);
            html += addTableRow('Номер локомотива', c.locomotive_number);
            html += addTableRow('Расход фактический, всего', c.rashod_fact_total, c.use_red_rashod);
            html += addTableRow('Расход по норме, всего', c.rashod_norm_total, c.use_red_rashod);
            html += `</table></div>`;

            html += `<div style="margin-bottom:20px;"><h3>Информация по участкам (всего: ${c.all_sections ? c.all_sections.length : 0})</h3>`;
            if (c.all_sections && c.all_sections.length) {
                html += `<div style="overflow-x:auto;max-width:100%;">`;
                html += `<table style="border-collapse:collapse;width:100%;font-family:Arial;font-size:11px;min-width:2000px;">`;
                const headers = [
                    'Наименование участка','НЕТТО','БРУТТО','ОСИ','Номер нормы','Дв. тяга','Ткм брутто','Км','Пр.',
                    'Расход фактический','Расход по норме','Уд. норма, норма на 1 час ман. раб.','Нажатие на ось','Норма на работу',
                    'Факт уд','Факт на работу','Норма на одиночное','Простой с бригадой, мин., всего','Простой с бригадой, мин., норма',
                    'Маневры, мин., всего','Маневры, мин., норма','Трогание с места, случ., всего','Трогание с места, случ., норма',
                    'Нагон опозданий, мин., всего','Нагон опозданий, мин., норма','Ограничения скорости, случ., всего','Ограничения скорости, случ., норма',
                    'На пересылаемые л-вы, всего','На пересылаемые л-вы, норма','Количество дубликатов маршрута'
                ];
                html += `<tr style="background-color:#f0f0f0;font-weight:bold;">` +
                        headers.map(h=>`<td style="padding:4px;border:1px solid #ddd;text-align:center;font-size:10px;white-space:nowrap;">${h}</td>`).join('') + `</tr>`;
                c.all_sections.forEach((s, i) => {
                    const cellStyle = (name) => {
                        let b='padding:4px;border:1px solid #ddd;text-align:center;font-size:11px;white-space:nowrap;';
                        if (['НЕТТО','БРУТТО','ОСИ'].includes(name) && s.use_red_color) b += ' background-color:#ffcccc;color:#f00;font-weight:bold;';
                        if (['Расход фактический','Расход по норме'].includes(name) && s.use_red_rashod) b += ' background-color:#ffcccc;color:#f00;font-weight:bold;';
                        return b;
                    };
                    const row = [
                        s.section_name,s.netto,s.brutto,s.osi,s.norm_number,s.movement_type,s.tkm_brutto,s.km,s.pr,
                        s.rashod_fact,s.rashod_norm,s.ud_norma,s.axle_load,s.norma_work,
                        s.fact_ud,s.fact_work,s.norma_single,s.idle_brigada_total,s.idle_brigada_norm,
                        s.manevr_total,s.manevr_norm,s.start_total,s.start_norm,s.delay_total,s.delay_norm,
                        s.speed_limit_total,s.speed_limit_norm,s.transfer_loco_total,s.transfer_loco_norm,s.duplicates_count
                    ];
                    html += `<tr style="background-color:${i%2===0?'#fff':'#f9f9f9'};">` +
                            row.map((v, idx)=>`<td style="${cellStyle(headers[idx])}">${(v??'-')!== 'N/A'? (v ?? '-') : '-'}</td>`).join('') +
                            `</tr>`;
                });
                html += `</table></div>`;
            } else {
                html += `<div style="padding:20px;background:#fff3cd;border:1px solid #ffeaa7;border-radius:5px;color:#856404;">
                        <strong>⚠️ Предупреждение:</strong> Информация о дополнительных участках маршрута недоступна.</div>`;
            }
            html += `</div>`;

            html += `<div style="margin-bottom:20px;"><h3>Результаты анализа (для текущего участка)</h3>
                    <table style="border-collapse:collapse;width:50%;font-family:Arial;">`;
            html += addTableRow('Норма интерполированная', c.norm_interpolated);
            html += addTableRow('Отклонение, %', c.deviation_percent);
            html += addTableRow('Статус', c.status);
            html += addTableRow('Н=Ф', c.n_equals_f);
            if (c.coefficient && c.coefficient !== 1.0) {
                html += addTableRow('Коэффициент', c.coefficient);
                if (c.fact_ud_original) html += addTableRow('Факт. удельный исходный', c.fact_ud_original);
            }
            html += `</table></div>`;

            document.getElementById('route-details').innerHTML = html;
            document.getElementById('route-modal').style.display = 'block';
        }
        function addTableRow(label, value, isRed = false) {
            const red = isRed ? 'background-color:#ffcccc;color:#f00;font-weight:bold;' : '';
            const v = (value !== null && value !== undefined && value !== 'N/A') ? value : '-';
            return `<tr style="border:1px solid #ddd;">
                <td style="padding:8px;border:1px solid #ddd;background-color:#f5f5f5;font-weight:bold;">${label}</td>
                <td style="padding:8px;border:1px solid #ddd;${red}">${v}</td></tr>`;
        }
        </script>
        '''
        return html_content.replace("</body>", f"{js_code}\n</body>") if "</body>" in html_content else html_content + js_code

    # -------------------------- Вспомогательные вычисления --------------------------

    def _build_sections_norms_map(self) -> None:
        """Строит карту участков -> список норм (строками)."""
        if self.routes_df is None or self.routes_df.empty:
            self.sections_norms_map = {}
            return

        def to_norm_str(x) -> Optional[str]:
            if pd.isna(x):
                return None
            s = str(x).strip()
            if s == "" or s.lower() == "nan":
                return None
            return str(int(s)) if s.isdigit() else s

        grouped = (
            self.routes_df.groupby("Наименование участка")["Номер нормы"]
            .apply(lambda s: sorted({n for n in (to_norm_str(x) for x in s) if n}))
            .to_dict()
        )
        self.sections_norms_map = grouped
        logger.info("Построена карта участков и норм: %s участков", len(self.sections_norms_map))

    def _filter_section_routes(self, section_name: str, single_section_only: bool) -> pd.DataFrame:
        """Фильтрация DataFrame по участку и (опционально) только маршруты с единственным участком."""
        if self.routes_df is None or self.routes_df.empty:
            return pd.DataFrame()

        df = self.routes_df[self.routes_df["Наименование участка"] == section_name].copy()
        if df.empty or not single_section_only:
            return df

        # Оставляем только те маршруты, где в исходных данных ровно один участок
        route_counts = (
            self.routes_df.groupby(["Номер маршрута", "Дата маршрута"])
            .size()
            .rename("cnt")
        )
        single_idx = route_counts[route_counts == 1].index
        df = df.set_index(["Номер маршрута", "Дата маршрута"]).loc[
            df.set_index(["Номер маршрута", "Дата маршрута"]).index.intersection(single_idx)
        ]
        return df.reset_index()

    def _apply_coefficients(self, routes_df: pd.DataFrame, manager: LocomotiveCoefficientsManager) -> pd.DataFrame:
        """Применяет коэффициенты к 'Расход фактический' и сохраняет исходное значение в 'Факт. удельный исходный'."""
        df = routes_df.copy()

        def resolve_number(val) -> Optional[int]:
            if pd.isna(val):
                return None
            try:
                if isinstance(val, str):
                    s = val.strip()
                    return int(s.lstrip("0") or "0")
                return int(val)
            except Exception:
                return None

        def coef_for_row(row) -> float:
            series = str(row.get("Серия локомотива", "") or "").strip()
            number = resolve_number(row.get("Номер локомотива"))
            if series and number is not None:
                try:
                    return float(manager.get_coefficient(series, number))
                except Exception:
                    return 1.0
            return 1.0

        df["Коэффициент"] = df.apply(coef_for_row, axis=1)
        mask = df["Коэффициент"].ne(1.0)
        df.loc[mask, "Факт. удельный исходный"] = df.loc[mask, "Расход фактический"]
        df.loc[mask, "Расход фактический"] = df.loc[mask, "Расход фактический"] / df.loc[mask, "Коэффициент"]

        applied_count = int(mask.sum())
        if applied_count:
            sample = df.loc[mask, ["Серия локомотива", "Номер локомотива", "Коэффициент"]].head(3)
            for _, r in sample.iterrows():
                logger.debug("Применен коэффициент %.3f к %s №%s", r["Коэффициент"], r["Серия локомотива"], r["Номер локомотива"])
        logger.info("Применено коэффициентов: %s", applied_count)
        return df

    def _analyze_section_data(
        self, section_name: str, routes_df: pd.DataFrame, specific_norm_id: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict]:
        """Интерполирует нормы, считает отклонения и статусы для строк маршрутов."""
        logger.debug("Анализ участка %s, строк: %s", section_name, len(routes_df))

        # Список норм для анализа
        if specific_norm_id:
            norm_numbers = [specific_norm_id]
        else:
            norm_numbers = routes_df["Номер нормы"].dropna().unique()

        norm_functions: Dict[str, Dict] = {}
        for nn in norm_numbers:
            norm_str = str(int(nn)) if pd.notna(nn) else None
            if not norm_str:
                continue

            norm_data = self.norm_storage.get_norm(norm_str)
            if not norm_data or not norm_data.get("points"):
                logger.warning("Норма %s не найдена или не содержит точек", norm_str)
                continue

            try:
                base_points = list(norm_data["points"])
                norm_type = norm_data.get("norm_type", "Нажатие")

                # Добавляем дополнительные точки из маршрутов (если есть уд. нормы в данных)
                additional_points = self._extract_additional_norm_points(routes_df, norm_str, norm_type)
                all_points = self._remove_duplicate_points(base_points + additional_points)
                all_points.sort(key=lambda x: x[0])

                # Создаем интерполяционную функцию (используем хранилище, т.к. оно знает нужную физику)
                func = self.norm_storage._create_interpolation_function(all_points)  # noqa: SLF001
                if func is None:
                    # Резерв: простая аппроксимация/интерполяция
                    func = self._fallback_interpolation(all_points)

                norm_functions[norm_str] = {
                    "function": func,
                    "points": all_points,
                    "base_points": base_points,
                    "additional_points": additional_points,
                    "x_range": (min(p[0] for p in base_points), max(p[0] for p in base_points)),
                    "data": norm_data,
                    "norm_type": norm_type,
                }
                logger.debug("Создана функция для нормы %s (тип: %s)", norm_str, norm_type)
            except Exception as e:
                logger.error("Ошибка создания функции для нормы %s: %s", norm_str, e, exc_info=True)

        if not norm_functions:
            logger.warning("Не найдено функций норм для участка %s", section_name)
            return routes_df, {}

        # Интерполяция и расчет отклонений
        routes_df = routes_df.copy()
        for idx, row in routes_df.iterrows():
            norm_number = row.get("Номер нормы")
            if not pd.notna(norm_number):
                continue

            norm_str = str(int(norm_number))
            nf = norm_functions.get(norm_str)
            if not nf:
                continue

            norm_type = nf.get("norm_type", "Нажатие")
            x_val = self._calculate_param(row, norm_type)
            if not (x_val and x_val > 0):
                continue

            try:
                norm_value = float(nf["function"](x_val))
            except Exception:
                continue

            routes_df.at[idx, "Норма интерполированная"] = norm_value
            routes_df.at[idx, "Параметр нормирования"] = "вес поезда (БРУТТО)" if norm_type == "Вес" else "нажатие на ось"
            routes_df.at[idx, "Значение параметра"] = x_val

            actual = row.get("Факт уд")
            if pd.isna(actual) or actual is None:
                actual = row.get("Расход фактический")

            if actual and norm_value > 0:
                deviation = ((float(actual) - norm_value) / norm_value) * 100.0
                routes_df.at[idx, "Отклонение, %"] = deviation
                routes_df.at[idx, "Статус"] = self._status_by_deviation(deviation)

        logger.info("Проанализировано %s строк для участка %s", len(routes_df), section_name)
        return routes_df, norm_functions

    # ------------------------------ Построение графика ------------------------------

    def _create_interactive_plot(
        self,
        section_name: str,
        routes_df: pd.DataFrame,
        norm_functions: Dict,
        specific_norm_id: Optional[str] = None,
        single_section_only: bool = False,
    ) -> go.Figure:
        """Создает интерактивный график: кривые норм, точки маршрутов, отклонения, границы."""
        title_suffix = f" (норма {specific_norm_id})" if specific_norm_id else ""
        filter_suffix = " [только один участок]" if single_section_only else ""

        fig = self._create_base_plot_structure(section_name, title_suffix, filter_suffix)
        norm_types_used = self._get_norm_types_used(norm_functions)

        # Кривые норм
        self._add_norm_curves(fig, norm_functions, routes_df, specific_norm_id, norm_types_used)
        # Точки маршрутов
        self._add_route_points(fig, routes_df, norm_functions, norm_types_used)
        # Точки отклонений и границы
        self._add_deviation_points(fig, routes_df)
        self._add_boundary_lines(fig, routes_df)

        self._configure_plot_layout(fig, norm_types_used)
        return fig

    def _create_base_plot_structure(self, section_name: str, title_suffix: str, filter_suffix: str) -> go.Figure:
        """Базовая структура с двумя подграфиками."""
        return make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f"Нормы расхода для участка: {section_name}{title_suffix}{filter_suffix}",
                "Отклонение фактического расхода от нормы",
            ),
        )

    def _get_norm_types_used(self, norm_functions: Dict) -> Set[str]:
        """Множество типов норм, используемых на графике."""
        return {nd.get("norm_type", "Нажатие") for nd in norm_functions.values()}

    def _add_norm_curves(
        self,
        fig: go.Figure,
        norm_functions: Dict,
        routes_df: pd.DataFrame,
        specific_norm_id: Optional[str],
        norm_types_used: Set[str],
    ) -> None:
        """Добавляет кривые норм и точки (базовые/дополнительные)."""
        for norm_id, norm_data in norm_functions.items():
            if specific_norm_id and norm_id != specific_norm_id:
                continue
            points = norm_data["points"]
            norm_type = norm_data.get("norm_type", "Нажатие")
            x_axis_name = "Вес поезда БРУТТО, т" if norm_type == "Вес" else "Нажатие на ось, т/ось"

            # Кривая нормы
            if len(points) == 1:
                self._add_constant_norm_curve(fig, norm_id, points[0], routes_df, norm_type, x_axis_name)
            else:
                self._add_interpolated_norm_curve(fig, norm_id, points, routes_df, norm_type, x_axis_name)

            # Базовые точки нормы
            self._add_base_norm_points(fig, norm_id, points, norm_type, x_axis_name)
            # Доп. точки из маршрутов
            self._add_additional_norm_points(fig, norm_id, norm_type, routes_df, x_axis_name)

    def _add_base_norm_points(self, fig: go.Figure, norm_id: str, points: List[Tuple[float, float]], norm_type: str, x_axis_name: str) -> None:
        """Добавляет базовые точки норм (синие квадраты)."""
        if not points:
            return
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]
        hover_texts = [
            f"<b>Базовая точка нормы {norm_id}</b><br>{x_axis_name}: {x:.1f}<br>Расход: {y:.1f} кВт·ч/10⁴ ткм<br><i>Источник: файл нормы</i>"
            for x, y in points
        ]
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=y_vals, mode="markers", name=f"Базовые точки нормы {norm_id} ({len(points)})",
                marker=dict(size=8, symbol="square", color="blue", opacity=0.9, line=dict(color="darkblue", width=1)),
                hovertemplate="%{text}<extra></extra>", text=hover_texts
            ),
            row=1, col=1,
        )

    def _add_constant_norm_curve(self, fig: go.Figure, norm_id: str, point: Tuple[float, float], routes_df: pd.DataFrame, norm_type: str, x_axis_name: str) -> None:
        """Добавляет константную норму (одна точка)."""
        x_single, y_single = point
        x_vals_from_data = self._get_additional_points_for_norm(routes_df, norm_id, norm_type)
        if x_vals_from_data:
            x_min, x_max = min(x_vals_from_data), max(x_vals_from_data)
            x_range = max(x_max - x_min, 1.0)
            x_start, x_end = max(x_min - x_range * 0.2, x_min * 0.8), x_max + x_range * 0.2
        else:
            x_start, x_end = max(x_single * 0.5, x_single - 100), x_single * 1.5 + 100

        x_const = np.linspace(x_start, x_end, 100)
        y_const = np.full_like(x_const, y_single)
        fig.add_trace(
            go.Scatter(
                x=x_const, y=y_const, mode="lines", name=f"Норма {norm_id} (константа)",
                line=dict(width=3, color="blue"),
                hovertemplate=f"<b>Норма {norm_id}</b><br>{x_axis_name}: %{{x:.1f}}<br>Расход: %{{y:.1f}} кВт·ч/10⁴ ткм<extra></extra>",
            ),
            row=1, col=1,
        )

    def _add_interpolated_norm_curve(self, fig: go.Figure, norm_id: str, points: List[Tuple[float, float]], routes_df: pd.DataFrame, norm_type: str, x_axis_name: str) -> None:
        """Добавляет интерполированную кривую нормы."""
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        x_min, x_max = min(x_vals), max(x_vals)
        x_range = max(x_max - x_min, 1.0)
        x_start, x_end = max(x_min - x_range * 0.3, x_min * 0.5), x_max + x_range * 0.3

        additional_x = self._get_additional_points_for_norm(routes_df, norm_id, norm_type)
        if additional_x:
            x_start = min(x_start, min(additional_x) * 0.8)
            x_end = max(x_end, max(additional_x) * 1.2)

        x_interp = np.linspace(x_start, x_end, 500)
        y_interp = self._interpolate_norm_values(x_vals, y_vals, x_interp)
        mask = np.isfinite(y_interp) & (y_interp > 0)
        fig.add_trace(
            go.Scatter(
                x=x_interp[mask], y=y_interp[mask], mode="lines", name=f"Норма {norm_id} ({len(points)} точек)",
                line=dict(width=3, color="blue"),
                hovertemplate=f"<b>Норма {norm_id}</b><br>{x_axis_name}: %{{x:.1f}}<br>Расход: %{{y:.1f}} кВт·ч/10⁴ ткм<extra></extra>",
            ),
            row=1, col=1,
        )

    def _add_additional_norm_points(self, fig: go.Figure, norm_id: str, norm_type: str, routes_df: pd.DataFrame, x_axis_name: str) -> None:
        """Добавляет дополнительные точки норм из маршрутов (оранжевые круги)."""
        pts = self._extract_additional_norm_points_with_route_info(routes_df, norm_id, norm_type)
        if not pts:
            return
        add_x = [p[0] for p in pts]
        add_y = [p[1] for p in pts]
        hover_texts = [f"<b>Из маршрута № {route}</b><br>{x_axis_name}: {x:.1f}<br>Расход: {y:.1f} кВт·ч/10⁴ ткм" for x, y, route in pts]
        fig.add_trace(
            go.Scatter(
                x=add_x, y=add_y, mode="markers", name=f"Из маршрутов {norm_id} ({len(pts)})",
                marker=dict(size=6, symbol="circle", opacity=0.7, color="orange"),
                hovertemplate="%{text}<extra></extra>", text=hover_texts,
            ),
            row=1, col=1,
        )

    def _add_route_points(self, fig: go.Figure, routes_df: pd.DataFrame, norm_functions: Dict, norm_types_used: Set[str]) -> None:
        """Добавляет точки маршрутов (верхний график) по статусам и типам норм."""
        status_colors = {
            "Экономия сильная": "darkgreen",
            "Экономия средняя": "green",
            "Экономия слабая": "lightgreen",
            "Норма": "blue",
            "Перерасход слабый": "orange",
            "Перерасход средний": "darkorange",
            "Перерасход сильный": "red",
        }
        for status, color in status_colors.items():
            sd = routes_df[routes_df["Статус"] == status]
            if sd.empty:
                continue
            for norm_type in norm_types_used:
                x_vals, y_vals, texts, custom = [], [], [], []
                for _, row in sd.iterrows():
                    payload = self._process_single_route_point(row, norm_functions, norm_type, routes_df)
                    if not payload:
                        continue
                    x_vals.append(payload["x_val"])
                    y_vals.append(payload["y_val"])
                    texts.append(payload["hover_text"])
                    custom.append(payload["custom_data"])
                if x_vals:
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals, y=y_vals, mode="markers", name=f"{status} ({norm_type})",
                            marker=dict(color=color, size=6, opacity=0.7),
                            customdata=custom, hovertemplate="%{text}<extra></extra>", text=texts,
                        ),
                        row=1, col=1,
                    )

    def _add_deviation_points(self, fig: go.Figure, routes_df: pd.DataFrame) -> None:
        """Добавляет точки отклонений на нижний график."""
        status_colors = {
            "Экономия сильная": "#006400",
            "Экономия средняя": "#228B22",
            "Экономия слабая": "#32CD32",
            "Норма": "#FFD700",
            "Перерасход слабый": "#FF8C00",
            "Перерасход средний": "#FF4500",
            "Перерасход сильный": "#DC143C",
        }
        for status, color in status_colors.items():
            sd = routes_df[routes_df["Статус"] == status]
            if sd.empty:
                continue
            x_vals, y_vals, texts, custom = [], [], [], []
            for _, r in sd.iterrows():
                axle = self._calculate_axle_load_from_data(r)
                if not axle:
                    continue
                x_vals.append(axle)
                y_vals.append(r.get("Отклонение, %"))
                texts.append(self._create_hover_text(r))
                custom.append(self._create_full_route_info(r))
            if x_vals:
                fig.add_trace(
                    go.Scatter(
                        x=x_vals, y=y_vals, mode="markers", name=f"{status} ({len(x_vals)})",
                        marker=dict(color=color, size=10, opacity=0.8, line=dict(color="black", width=0.5)),
                        customdata=custom, hovertemplate="%{text}",
                        text=texts,
                    ),
                    row=2, col=1,
                )

    def _add_boundary_lines(self, fig: go.Figure, routes_df: pd.DataFrame) -> None:
        """Добавляет граничные линии и зону нормы на нижней панели."""
        if routes_df.empty:
            return
        axle_loads = [self._calculate_axle_load_from_data(r) for _, r in routes_df.iterrows()]
        axle_loads = [x for x in axle_loads if x]
        if not axle_loads:
            return

        x_range = [min(axle_loads) - 1, max(axle_loads) + 1]
        lines = [
            (5, "#FFD700", "dash"), (-5, "#FFD700", "dash"),
            (20, "#FF4500", "dot"), (-20, "#FF4500", "dot"),
            (30, "#DC143C", "dashdot"), (-30, "#DC143C", "dashdot"),
            (0, "black", "solid"),
        ]
        for y, color, dash in lines:
            fig.add_trace(
                go.Scatter(x=x_range, y=[y, y], mode="lines", line=dict(color=color, dash=dash, width=2), showlegend=False, hoverinfo="skip"),
                row=2, col=1,
            )
        # Заливка зоны -5..5
        fig.add_trace(
            go.Scatter(
                x=x_range + x_range[::-1], y=[-5, -5, 5, 5],
                fill="toself", fillcolor="rgba(255,215,0,0.1)", line=dict(color="rgba(255,255,255,0)"),
                showlegend=False, hoverinfo="skip",
            ),
            row=2, col=1,
        )

    def _configure_plot_layout(self, fig: go.Figure, norm_types_used: Set[str]) -> None:
        """Оси, легенда, линии."""
        mixed = len(norm_types_used) > 1
        if mixed:
            x_title = "Параметр нормирования (т/ось или т БРУТТО)"
        elif "Вес" in norm_types_used:
            x_title = "Вес поезда БРУТТО, т"
        else:
            x_title = "Нажатие на ось, т/ось"

        fig.update_xaxes(title_text=x_title, row=1, col=1)
        fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм", row=1, col=1)
        fig.update_xaxes(title_text=x_title, row=2, col=1)
        fig.update_yaxes(title_text="Отклонение, %", row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        fig.update_layout(
            height=800, hovermode="closest",
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        )

    # ------------------------------ Низкоуровневые функции ------------------------------

    def _interpolate_norm_values(self, x_vals: List[float], y_vals: List[float], x_interp: np.ndarray) -> np.ndarray:
        """Интерполяция нормы: гиперболическая аппроксимация Y=A/X+B или сплайн/линейная."""
        try:
            if len(x_vals) >= 2:
                A = np.column_stack((1 / np.array(x_vals, dtype=float), np.ones(len(x_vals))))
                b = np.array(y_vals, dtype=float)
                params, *_ = np.linalg.lstsq(A, b, rcond=None)
                a, b_coef = params
                y_pred = a / np.array(x_vals, dtype=float) + b_coef
                mse = float(np.mean((np.array(y_vals, dtype=float) - y_pred) ** 2))
                if mse < 1000:  # порог пригодности гиперболы
                    return a / x_interp + b_coef

            # SciPy cubic если доступен, иначе линейная на numpy
            try:
                from scipy.interpolate import interp1d  # ленивый импорт, если доступен
                f = interp1d(x_vals, y_vals, kind="cubic", fill_value="extrapolate")
                return f(x_interp)
            except Exception:
                return np.interp(x_interp, x_vals, y_vals)
        except Exception as e:
            logger.debug("Ошибка интерполяции: %s, fallback на линейную", e)
            return np.interp(x_interp, x_vals, y_vals)

    def _fallback_interpolation(self, points: List[Tuple[float, float]]):
        """Резервная интерполяция на случай отсутствия функции в хранилище."""
        x_vals = [p[0] for p in points]
        y_vals = [p[1] for p in points]

        def f(x: float) -> float:
            return float(np.interp(x, x_vals, y_vals))

        return f

    def _process_single_route_point(
        self, row: pd.Series, norm_functions: Dict, norm_type: str, routes_df: pd.DataFrame
    ) -> Optional[Dict]:
        """Подготовка одной точки маршрута для верхнего графика."""
        norm_number = row.get("Номер нормы")
        if not pd.notna(norm_number):
            return None
        norm_str = str(int(norm_number))
        nf = norm_functions.get(norm_str)
        if not nf or nf.get("norm_type", "Нажатие") != norm_type:
            return None

        # Координаты
        x_val = self._calculate_param(row, norm_type)
        if not (x_val and x_val > 0):
            return None

        actual_value = row.get("Факт уд") or row.get("Расход фактический")
        if not actual_value:
            return None

        # Hover
        hover = self._build_hover_for_point(row, norm_str, x_val, actual_value)

        # Полная инфа в customdata (для клика)
        custom_data = self._create_full_route_info(row, routes_df)

        return {"x_val": x_val, "y_val": actual_value, "hover_text": hover, "custom_data": custom_data}

    def _build_hover_for_point(self, row: pd.Series, norm_str: str, x_val: float, actual_value: float) -> str:
        """Формирует hover-текст для точки маршрута (безопасное форматирование чисел)."""
        info = self._create_route_info_for_hover(row)
        route_title = " | ".join(info) if info else "Маршрут"

        norm_inter = self._fmt_num(row.get("Норма интерполированная"))
        dev = self._fmt_num(row.get("Отклонение, %"))
        rf = self._fmt_num(row.get("Расход фактический"))
        rn = self._fmt_num(row.get("Расход по норме"))
        av = self._fmt_num(actual_value)

        x_label = "Вес БРУТТО" if row.get("Параметр нормирования") == "вес поезда (БРУТТО)" else "Нажатие на ось"

        return (
            f"<b>{route_title}</b><br>"
            f"{x_label}: {x_val:.1f}<br>"
            f"Факт: {av}<br>"
            f"Норма: {norm_inter}<br>"
            f"Расход фактический: {rf}<br>"
            f"Расход по норме: {rn}<br>"
            f"Отклонение: {dev}%<br>"
            f"Номер нормы: {norm_str}"
        )

    def _create_route_info_for_hover(self, row: pd.Series) -> List[str]:
        """Краткая информация о маршруте для hover."""
        parts: List[str] = []
        if pd.notna(row.get("Номер маршрута")):
            parts.append(f"Маршрут №{row.get('Номер маршрута')}")
        if pd.notna(row.get("Дата маршрута")):
            parts.append(f"Дата: {row.get('Дата маршрута')}")
        section = row.get("Наименование участка") or row.get("Участок")
        if pd.notna(section):
            parts.append(f"Участок: {section}")
        series = row.get("Серия локомотива") or row.get("Серия ТПС")
        number = row.get("Номер локомотива") or row.get("Номер ТПС")
        if pd.notna(series):
            parts.append(f"{'Локомотив' if row.get('Серия локомотива') is not None else 'ТПС'}: {series}" + (f" №{number}" if pd.notna(number) else ""))
        return parts

    def _create_hover_text(self, route: pd.Series) -> str:
        """Текст для hover (нижний график), безопасное форматирование."""
        parts = [
            f"Маршрут №{route.get('Номер маршрута', 'N/A')}",
            f"Дата: {route.get('Дата маршрута', 'N/A')}",
        ]
        series = route.get("Серия локомотива", "")
        num = route.get("Номер локомотива", "")
        if series:
            parts.append(f"Локомотив: {series} №{num}")
        header = "<br>".join(parts) + "<br>"

        coef = route.get("Коэффициент")
        coef_block = ""
        if coef and not pd.isna(coef) and coef != 1.0:
            base = self._fmt_num(route.get("Факт. удельный исходный"))
            coef_block = f"Коэффициент: {coef:.3f}<br>" + (f"Факт исходный: {base}<br>" if base != "N/A" else "")

        axle = self._calculate_axle_load_from_data(route)
        norm_i = self._fmt_num(route.get("Норма интерполированная"))
        fact = self._fmt_num(route.get("Факт уд") or route.get("Расход фактический"))
        rf = self._fmt_num(route.get("Расход фактический"))
        rn = self._fmt_num(route.get("Расход по норме"))
        dev = self._fmt_num(route.get("Отклонение, %"))

        return (
            header
            + coef_block
            + (f"Нажатие: {axle:.2f} т/ось<br>" if axle else "Нажатие: N/A<br>")
            + f"Факт: {fact}<br>"
            + f"Норма: {norm_i}<br>"
            + f"Расход фактический: {rf}<br>"
            + f"Расход по норме: {rn}<br>"
            + f"Отклонение: {dev}%"
        )

    def _calculate_section_statistics(self, routes_df: pd.DataFrame) -> Dict:
        """Сводная статистика по статусам и отклонениям."""
        total = int(len(routes_df))
        valid_routes = routes_df[routes_df["Статус"] != "Не определен"] if "Статус" in routes_df else routes_df
        processed = int(len(valid_routes))
        if processed == 0:
            return {
                "total": total, "processed": 0, "economy": 0, "normal": 0, "overrun": 0,
                "mean_deviation": 0, "median_deviation": 0, "detailed_stats": {},
            }

        det = {
            "economy_strong": int((valid_routes["Статус"] == "Экономия сильная").sum()),
            "economy_medium": int((valid_routes["Статус"] == "Экономия средняя").sum()),
            "economy_weak": int((valid_routes["Статус"] == "Экономия слабая").sum()),
            "normal": int((valid_routes["Статус"] == "Норма").sum()),
            "overrun_weak": int((valid_routes["Статус"] == "Перерасход слабый").sum()),
            "overrun_medium": int((valid_routes["Статус"] == "Перерасход средний").sum()),
            "overrun_strong": int((valid_routes["Статус"] == "Перерасход сильный").sum()),
        }
        return {
            "total": total,
            "processed": processed,
            "economy": det["economy_strong"] + det["economy_medium"] + det["economy_weak"],
            "normal": det["normal"],
            "overrun": det["overrun_weak"] + det["overrun_medium"] + det["overrun_strong"],
            "mean_deviation": float(valid_routes["Отклонение, %"].mean()),
            "median_deviation": float(valid_routes["Отклонение, %"].median()),
            "detailed_stats": det,
        }

    def _log_routes_statistics(self) -> None:
        """Лог статистики загруженных маршрутов."""
        if self.routes_df is None or self.routes_df.empty:
            return
        stats = self.route_processor.get_processing_stats()
        logger.info("=== СТАТИСТИКА ЗАГРУЖЕННЫХ МАРШРУТОВ ===")
        logger.info("Всего файлов: %s", stats.get("total_files"))
        logger.info("Всего маршрутов: %s", stats.get("total_routes_found"))
        logger.info("Уникальных маршрутов: %s", stats.get("unique_routes"))
        logger.info("Дубликатов: %s", stats.get("duplicates_total"))
        logger.info("Маршрутов с равными расходами: %s", stats.get("routes_with_equal_rashod"))
        logger.info("Обработано успешно: %s", stats.get("routes_processed"))
        logger.info("Пропущено: %s", stats.get("routes_skipped"))
        logger.info("Итоговых записей в DataFrame: %s", len(self.routes_df))
        logger.info("Уникальных участков: %s", self.routes_df["Наименование участка"].nunique())
        logger.info("Уникальных норм: %s", self.routes_df["Номер нормы"].nunique())

    # ------------------------------ Вспомогательные утилиты ------------------------------

    def _get_norm_type_from_storage(self, norm_id: str) -> str:
        """Тип нормы из хранилища."""
        try:
            nd = self.norm_storage.get_norm(norm_id)
            return nd.get("norm_type", "Нажатие") if nd else "Нажатие"
        except Exception as e:
            logger.debug("Не удалось определить тип нормы %s: %s", norm_id, e)
            return "Нажатие"

    def get_norm_info_with_type(self, norm_id: str) -> Optional[Dict]:
        """Информация о норме с типом."""
        try:
            info = self.get_norm_info(norm_id)
            if info:
                info["norm_type"] = self._get_norm_type_from_storage(norm_id)
            return info
        except Exception as e:
            logger.error("Ошибка получения информации о норме %s: %s", norm_id, e, exc_info=True)
            return None

    def _get_additional_points_for_norm(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> List[float]:
        """X-координаты доп. точек (из маршрутов) для конкретной нормы."""
        x_values: List[float] = []
        for _, row in routes_df.iterrows():
            rid = row.get("Номер нормы")
            if not pd.notna(rid) or str(int(rid)) != norm_id:
                continue
            x = self._calculate_param(row, norm_type)
            if x and x > 0:
                x_values.append(x)
        return x_values

    def _extract_additional_norm_points(self, routes_df: pd.DataFrame, norm_id: str, norm_type: str) -> List[Tuple[float, float]]:
        """Извлекает дополнительные точки (x, y) норм из маршрутов на основе столбца с удельной нормой."""
        ud_col = self._get_ud_norma_column(routes_df)
        if not ud_col:
            return []
        points: List[Tuple[float, float]] = []
        for _, row in routes_df.iterrows():
            rid = row.get("Номер нормы")
            if not pd.notna(rid) or str(int(rid)) != norm_id:
                continue
            ud = row.get(ud_col)
            try:
                if pd.notna(ud) and str(ud).strip() not in ("", "-"):
                    y = float(ud)
                    if y > 0:
                        x = self._calculate_param(row, norm_type)
                        if x and x > 0:
                            points.append((x, y))
            except Exception:
                continue
        return points

    def _extract_additional_norm_points_with_route_info(
        self, routes_df: pd.DataFrame, norm_id: str, norm_type: str
    ) -> List[Tuple[float, float, str]]:
        """Извлекает доп. точки норм с информацией о маршрутах (x, y, 'route_numbers')."""
        ud_col = self._get_ud_norma_column(routes_df)
        if not ud_col:
            return []

        raw: List[Tuple[float, float, str]] = []
        for _, row in routes_df.iterrows():
            rid = row.get("Номер нормы")
            if not pd.notna(rid) or str(int(rid)) != norm_id:
                continue
            ud = row.get(ud_col)
            try:
                if pd.notna(ud) and str(ud).strip() not in ("", "-"):
                    y = float(ud)
                    if y > 0:
                        x = self._calculate_param(row, norm_type)
                        if x and x > 0:
                            route_number = row.get("Номер маршрута", "N/A")
                            raw.append((x, y, str(route_number)))
            except Exception:
                continue

        if not raw:
            return []

        # Группируем близкие точки, объединяем номера маршрутов
        buckets: Dict[Tuple[float, float], Tuple[float, float, List[str]]] = {}
        for x, y, route in raw:
            key = (round(x, 2), round(y, 1))
            if key not in buckets:
                buckets[key] = (x, y, [route])
            else:
                ex_x, ex_y, routes = buckets[key]
                if route not in routes:
                    routes.append(route)
                buckets[key] = (ex_x, ex_y, routes)

        result: List[Tuple[float, float, str]] = []
        for x, y, routes in buckets.values():
            try:
                routes_sorted = ", ".join(sorted(routes, key=lambda r: int(r) if str(r).isdigit() else 10**9))
            except Exception:
                routes_sorted = ", ".join(routes)
            result.append((x, y, routes_sorted))

        result.sort(key=lambda t: t[0])
        return result

    def _get_ud_norma_column(self, routes_df: pd.DataFrame) -> Optional[str]:
        """Определяет реальное имя колонки с удельной нормой."""
        candidates = [
            "Уд. норма, норма на 1 час ман. раб.",
            "Удельная норма",
            "Уд норма",
            "Норма на 1 час",
            "УД. НОРМА",
        ]
        for col in candidates:
            if col in routes_df.columns:
                return col
        logger.debug("Колонка удельной нормы не найдена среди: %s", candidates)
        return None

    def _remove_duplicate_points(self, points: List[Tuple[float, float]], tolerance: float = 0.1) -> List[Tuple[float, float]]:
        """Удаляет почти дублирующиеся точки по X (с усреднением Y)."""
        if not points:
            return []
        unique: List[Tuple[float, float]] = []
        for x, y in points:
            replaced = False
            for i, (ex, ey) in enumerate(unique):
                if abs(x - ex) <= tolerance:
                    unique[i] = (ex, (ey + y) / 2.0)
                    replaced = True
                    break
            if not replaced:
                unique.append((x, y))
        return unique

    def _calculate_param(self, row: pd.Series, norm_type: str) -> Optional[float]:
        """Выбор параметра нормирования по типу нормы."""
        return self._calculate_weight_from_data(row) if norm_type == "Вес" else self._calculate_axle_load_from_data(row)

    def _calculate_axle_load_from_data(self, row: pd.Series) -> Optional[float]:
        """Нажатие на ось: берем готовое значение или вычисляем по данным."""
        try:
            # 1) Готовое значение
            axle = row.get("Нажатие на ось")
            if pd.notna(axle) and axle != "-" and isinstance(axle, (int, float)):
                return float(axle)

            # 2) brutto / osi
            brutto = row.get("БРУТТО")
            osi = row.get("ОСИ")
            if (
                pd.notna(brutto) and pd.notna(osi)
                and brutto != "-" and osi != "-" and isinstance(brutto, (int, float)) and isinstance(osi, (int, float)) and float(osi) != 0.0
            ):
                return float(brutto) / float(osi)

            # 3) Приблизительный расчет из т-км и км
            tkm_brutto = row.get("Ткм брутто")
            km = row.get("Км")
            if pd.notna(tkm_brutto) and pd.notna(km) and float(km) != 0.0:
                # эмпирическое приближение
                return float(tkm_brutto) / float(km) / 1000.0 * 20.0
            return None
        except Exception as e:
            logger.debug("Ошибка расчета нажатия на ось: %s", e)
            return None

    def _calculate_weight_from_data(self, row: pd.Series) -> Optional[float]:
        """Вес поезда БРУТТО, прямые колонки или расчет из т-км/км."""
        try:
            for col in ("БРУТТО", "Вес БРУТТО", "Вес поезда БРУТТО", "Брутто"):
                if col in row.index:
                    brutto = row.get(col)
                    if pd.notna(brutto) and str(brutto).strip() not in ("", "-"):
                        val = float(brutto)
                        if val > 0:
                            return val
            tkm_brutto = row.get("Ткм брутто")
            km = row.get("Км")
            if pd.notna(tkm_brutto) and pd.notna(km) and float(km) != 0.0:
                val = float(tkm_brutto) / float(km)
                return val if val > 0 else None
            return None
        except Exception as e:
            logger.debug("Ошибка расчета веса поезда: %s", e)
            return None

    def _status_by_deviation(self, deviation: float) -> str:
        """Статус по порогам отклонения."""
        if deviation < -30:
            return "Экономия сильная"
        if deviation < -20:
            return "Экономия средняя"
        if deviation < -5:
            return "Экономия слабая"
        if deviation <= 5:
            return "Норма"
        if deviation <= 20:
            return "Перерасход слабый"
        if deviation <= 30:
            return "Перерасход средний"
        return "Перерасход сильный"

    def _fmt_num(self, val, ndigits: int = 1) -> str:
        """Безопасное форматирование числа."""
        try:
            f = float(val)
            return f"{f:.{ndigits}f}"
        except Exception:
            return "N/A"

    # ------------------------- Дополнительные (сохранены для совместимости) -------------------------

    def get_section_routes_count(self, section_name: str) -> int:
        """Количество строк маршрутов для участка (метод оставлен для совместимости)."""
        if self.routes_df is None or self.routes_df.empty:
            return 0
        return int((self.routes_df["Наименование участка"] == section_name).sum())

    # Полная информация о маршруте для кликов и customdata
    def _create_full_route_info(self, route: pd.Series, routes_df: pd.DataFrame = None) -> Dict:

        # Локальный помощник: безопасно привести к float или вернуть None
        def to_float(val):
            if val is None or (isinstance(val, float) and (val != val)):  # NaN
                return None
            try:
                if isinstance(val, str):
                    s = val.strip().replace(',', '.')
                    if not s or s in ('-', 'N/A'):
                        return None
                    return float(s)
                return float(val)
            except Exception:
                return None

        # Используем переданный DataFrame или self.routes_df
        source_df = routes_df if routes_df is not None else getattr(self, 'routes_df', None)

        route_number = route.get('Номер маршрута')
        route_date = route.get('Дата маршрута')

        all_sections_data = []
        rashod_fact_total = 0.0
        rashod_norm_total = 0.0

        if route_number and route_date and source_df is not None:
            same_route_data = source_df[
                (source_df['Номер маршрута'] == route_number) &
                (source_df['Дата маршрута'] == route_date)
            ].copy()

            if not same_route_data.empty:
                for _, section_row in same_route_data.iterrows():
                    section_info = {
                        'section_name': section_row.get('Наименование участка', 'N/A'),
                        'netto': section_row.get('НЕТТО', 'N/A'),
                        'brutto': section_row.get('БРУТТО', 'N/A'),
                        'osi': section_row.get('ОСИ', 'N/A'),
                        'norm_number': section_row.get('Номер нормы', 'N/A'),
                        'movement_type': section_row.get('Дв. тяга', 'N/A'),
                        'tkm_brutto': section_row.get('Ткм брутто', 'N/A'),
                        'km': section_row.get('Км', 'N/A'),
                        'pr': section_row.get('Пр.', 'N/A'),
                        'rashod_fact': section_row.get('Расход фактический', 'N/A'),
                        'rashod_norm': section_row.get('Расход по норме', 'N/A'),
                        'ud_norma': section_row.get('Уд. норма, норма на 1 час ман. раб.', 'N/A'),
                        'axle_load': section_row.get('Нажатие на ось', 'N/A'),
                        'norma_work': section_row.get('Норма на работу', 'N/A'),
                        'fact_ud': section_row.get('Факт уд', 'N/A'),
                        'fact_work': section_row.get('Факт на работу', 'N/A'),
                        'norma_single': section_row.get('Норма на одиночное', 'N/A'),
                        'idle_brigada_total': section_row.get('Простой с бригадой, мин., всего', 'N/A'),
                        'idle_brigada_norm': section_row.get('Простой с бригадой, мин., норма', 'N/A'),
                        'manevr_total': section_row.get('Маневры, мин., всего', 'N/A'),
                        'manevr_norm': section_row.get('Маневры, мин., норма', 'N/A'),
                        'start_total': section_row.get('Трогание с места, случ., всего', 'N/A'),
                        'start_norm': section_row.get('Трогание с места, случ., норма', 'N/A'),
                        'delay_total': section_row.get('Нагон опозданий, мин., всего', 'N/A'),
                        'delay_norm': section_row.get('Нагон опозданий, мин., норма', 'N/A'),
                        'speed_limit_total': section_row.get('Ограничения скорости, случ., всего', 'N/A'),
                        'speed_limit_norm': section_row.get('Ограничения скорости, случ., норма', 'N/A'),
                        'transfer_loco_total': section_row.get('На пересылаемые л-вы, всего', 'N/A'),
                        'transfer_loco_norm': section_row.get('На пересылаемые л-вы, норма', 'N/A'),
                        'duplicates_count': section_row.get('Количество дубликатов маршрута', 'N/A'),
                        'use_red_color': section_row.get('USE_RED_COLOR', False),
                        'use_red_rashod': section_row.get('USE_RED_RASHOD', False)
                    }
                    all_sections_data.append(section_info)

                    # Суммарные расходы (учитываем только числовые значения)
                    rf_val = to_float(section_row.get('Расход фактический'))
                    if rf_val and rf_val > 0:
                        rashod_fact_total += rf_val
                    rn_val = to_float(section_row.get('Расход по норме'))
                    if rn_val and rn_val > 0:
                        rashod_norm_total += rn_val
        else:
            # Fallback: хотя бы текущий участок
            current_section_info = {
                'section_name': route.get('Наименование участка', 'N/A'),
                'netto': route.get('НЕТТО', 'N/A'),
                'brutto': route.get('БРУТТО', 'N/A'),
                'osi': route.get('ОСИ', 'N/A'),
                'norm_number': route.get('Номер нормы', 'N/A'),
                'movement_type': route.get('Дв. тяга', 'N/A'),
                'tkm_brutto': route.get('Ткм брутто', 'N/A'),
                'km': route.get('Км', 'N/A'),
                'pr': route.get('Пр.', 'N/A'),
                'rashod_fact': route.get('Расход фактический', 'N/A'),
                'rashod_norm': route.get('Расход по норме', 'N/A'),
                'ud_norma': route.get('Уд. норма, норма на 1 час ман. раб.', 'N/A'),
                'axle_load': route.get('Нажатие на ось', 'N/A'),
                'norma_work': route.get('Норма на работу', 'N/A'),
                'fact_ud': route.get('Факт уд', 'N/A'),
                'fact_work': route.get('Факт на работу', 'N/A'),
                'norma_single': route.get('Норма на одиночное', 'N/A'),
                'idle_brigada_total': route.get('Простой с бригадой, мин., всего', 'N/A'),
                'idle_brigada_norm': route.get('Простой с бригадой, мин., норма', 'N/A'),
                'manevr_total': route.get('Маневры, мин., всего', 'N/A'),
                'manevr_norm': route.get('Маневры, мин., норма', 'N/A'),
                'start_total': route.get('Трогание с места, случ., всего', 'N/A'),
                'start_norm': route.get('Трогание с места, случ., норма', 'N/A'),
                'delay_total': route.get('Нагон опозданий, мин., всего', 'N/A'),
                'delay_norm': route.get('Нагон опозданий, мин., норма', 'N/A'),
                'speed_limit_total': route.get('Ограничения скорости, случ., всего', 'N/A'),
                'speed_limit_norm': route.get('Ограничения скорости, случ., норма', 'N/A'),
                'transfer_loco_total': route.get('На пересылаемые л-вы, всего', 'N/A'),
                'transfer_loco_norm': route.get('На пересылаемые л-вы, норма', 'N/A'),
                'duplicates_count': route.get('Количество дубликатов маршрута', 'N/A'),
                'use_red_color': route.get('USE_RED_COLOR', False),
                'use_red_rashod': route.get('USE_RED_RASHOD', False)
            }
            all_sections_data.append(current_section_info)

            # Итоги по текущему участку
            rf_val = to_float(route.get('Расход фактический'))
            if rf_val and rf_val > 0:
                rashod_fact_total = rf_val
            rn_val = to_float(route.get('Расход по норме'))
            if rn_val and rn_val > 0:
                rashod_norm_total = rn_val

        # Верхнеуровневые поля для JS-переключателя
        # Берем по текущей строке (текущему участку на графике); fallback: 'Факт уд', если 'Расход фактический' недоступен
        rashod_fact_point = to_float(route.get('Расход фактический'))
        if rashod_fact_point is None:
            rashod_fact_point = to_float(route.get('Факт уд'))
        rashod_norm_point = to_float(route.get('Расход по норме'))
        norm_interpolated_point = to_float(route.get('Норма интерполированная'))

        route_info = {
            'route_number': route.get('Номер маршрута', 'N/A'),
            'route_date': route.get('Дата маршрута', 'N/A'),
            'trip_date': route.get('Дата поездки', 'N/A'),
            'driver_tab': route.get('Табельный машиниста', 'N/A'),
            'locomotive_series': route.get('Серия локомотива', 'N/A'),
            'locomotive_number': route.get('Номер локомотива', 'N/A'),

            # Суммарные расходы по маршруту (для модального окна)
            'rashod_fact_total': rashod_fact_total if rashod_fact_total > 0 else 'N/A',
            'rashod_norm_total': rashod_norm_total if rashod_norm_total > 0 else 'N/A',

            # Все участки маршрута
            'all_sections': all_sections_data,

            # Данные текущего участка (для анализа/отображения)
            'norm_interpolated': norm_interpolated_point if norm_interpolated_point is not None else route.get('Норма интерполированная', 'N/A'),
            'deviation_percent': route.get('Отклонение, %', 'N/A'),
            'status': route.get('Статус', 'N/A'),
            'n_equals_f': route.get('Н=Ф', 'N/A'),

            # Коэффициенты (если применялись)
            'coefficient': route.get('Коэффициент', None),
            'fact_ud_original': route.get('Факт. удельный исходный', None),

            # Флаги для модального окна
            'use_red_rashod': route.get('USE_RED_RASHOD', False),

            # ВАЖНО: поля для JS 'Н/Ф' режима (числовые или None)
            'rashod_fact': rashod_fact_point,
            'rashod_norm': rashod_norm_point,
        }

        return route_info