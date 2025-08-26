# analysis/analyzer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ИСПРАВЛЕННЫЙ анализатор норм расхода электроэнергии с защитой от ошибок передачи данных."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analysis.data_analyzer import RouteDataAnalyzer, CoefficientsApplier
from analysis.html_route_processor import HTMLRouteProcessor
from analysis.html_norm_processor import HTMLNormProcessor
from core.coefficients import LocomotiveCoefficientsManager
from core.filter import LocomotiveFilter
from core.norm_storage import NormStorage
from core.utils import extract_route_key, safe_int, safe_float

logger = logging.getLogger(__name__)


class InteractiveNormsAnalyzer:
    """
    ИСПРАВЛЕННЫЙ интерактивный анализатор с robust data handling.
    Добавлена защита от ошибок передачи данных между компонентами.
    """

    def __init__(self):
        # Процессоры данных - инициализация с проверкой
        try:
            self.route_processor = HTMLRouteProcessor()
            self.norm_processor = HTMLNormProcessor()
            self.norm_storage = NormStorage()
            self.data_analyzer = RouteDataAnalyzer(self.norm_storage)
            
            logger.info("✓ Все процессоры инициализированы успешно")
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка инициализации процессоров: %s", e, exc_info=True)
            raise RuntimeError(f"Не удалось инициализировать анализатор: {e}")

        # Основные данные
        self.routes_df: Optional[pd.DataFrame] = None
        self.analyzed_results: Dict[str, Dict] = {}
        self.sections_norms_map: Dict[str, List[str]] = {}
        
        # Статистика и диагностика
        self.processing_stats = {
            'routes_loaded': False,
            'norms_loaded': False,
            'last_analysis': None,
            'total_routes': 0,
            'total_sections': 0,
            'total_norms': 0
        }

        logger.info("Анализатор инициализирован с защитой от ошибок")

    # ========================== Загрузка данных ==========================

    def load_routes_from_html(self, html_files: List[str]) -> bool:
        """ИСПРАВЛЕННАЯ загрузка маршрутов с детальной диагностикой."""
        logger.info("=== ЗАГРУЗКА МАРШРУТОВ ===")
        logger.info("Файлов для обработки: %d", len(html_files))
        
        if not html_files:
            logger.error("Список HTML файлов пуст")
            return False
            
        try:
            # Валидируем существование файлов
            valid_files = []
            for file_path in html_files:
                if Path(file_path).exists():
                    valid_files.append(file_path)
                else:
                    logger.warning("Файл не найден: %s", file_path)
                    
            if not valid_files:
                logger.error("Ни одного валидного файла не найдено")
                return False
                
            logger.info("Валидных файлов: %d из %d", len(valid_files), len(html_files))
            
            # Обрабатываем файлы
            self.routes_df = self.route_processor.process_html_files(valid_files)
            
            # Валидируем результат
            if self.routes_df is None:
                logger.error("process_html_files вернул None")
                return False
                
            if self.routes_df.empty:
                logger.error("DataFrame пуст после обработки")
                return False

            # Обновляем статистику
            self.processing_stats.update({
                'routes_loaded': True,
                'total_routes': len(self.routes_df),
                'total_sections': len(self.routes_df["Наименование участка"].dropna().unique())
            })

            # Строим карту участков -> нормы
            self._build_sections_norms_map_safe()

            logger.info("✓ ЗАГРУЗКА МАРШРУТОВ ЗАВЕРШЕНА")
            logger.info("Записей загружено: %d | Участков: %d", 
                       self.processing_stats['total_routes'], 
                       self.processing_stats['total_sections'])
            
            return True
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка загрузки маршрутов: %s", e, exc_info=True)
            self.routes_df = None
            self.processing_stats['routes_loaded'] = False
            return False

    def load_norms_from_html(self, html_files: List[str]) -> bool:
        """ИСПРАВЛЕННАЯ загрузка норм с валидацией."""
        logger.info("=== ЗАГРУЗКА НОРМ ===")
        logger.info("Файлов норм для обработки: %d", len(html_files))
        
        if not html_files:
            logger.error("Список HTML файлов норм пуст")
            return False
            
        try:
            # Валидируем файлы
            valid_files = [f for f in html_files if Path(f).exists()]
            if not valid_files:
                logger.error("Ни одного валидного файла норм не найдено")
                return False
                
            logger.info("Валидных файлов норм: %d", len(valid_files))
            
            # Обрабатываем нормы
            new_norms = self.norm_processor.process_html_files(valid_files)
            
            if not new_norms:
                logger.warning("Нормы не найдены в HTML файлах")
                return False

            # Добавляем в хранилище
            update_results = self.norm_storage.add_or_update_norms(new_norms)
            
            # Обновляем статистику
            self.processing_stats.update({
                'norms_loaded': True,
                'total_norms': len(new_norms)
            })
            
            # Статистика обновления
            stats = self.norm_processor.get_processing_stats()
            logger.info("✓ ЗАГРУЗКА НОРМ ЗАВЕРШЕНА")
            logger.info("Обработано: всего=%d, новых=%d, обновленных=%d",
                       stats.get("total_norms_found", 0),
                       stats.get("new_norms", 0), 
                       stats.get("updated_norms", 0))
            
            return True
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка загрузки норм: %s", e, exc_info=True)
            self.processing_stats['norms_loaded'] = False
            return False

    # ========================== Безопасное получение данных ==========================

    def get_sections_list(self) -> List[str]:
        """Безопасно возвращает список участков."""
        try:
            if self.routes_df is None or self.routes_df.empty:
                logger.debug("DataFrame маршрутов пуст")
                return []
                
            sections = self.routes_df["Наименование участка"].dropna().unique()
            section_list = sorted([str(s) for s in sections if str(s).strip()])
            
            logger.debug("Найдено участков: %d", len(section_list))
            return section_list
            
        except Exception as e:
            logger.error("Ошибка получения списка участков: %s", e)
            return []

    def get_norms_with_counts_for_section(self, section_name: str, single_section_only: bool = False) -> List[Tuple[str, int]]:
        """ИСПРАВЛЕННОЕ получение норм с количествами для участка."""
        logger.debug("Получение норм для участка: %s (один участок: %s)", section_name, single_section_only)
        
        try:
            # Получаем отфильтрованные маршруты
            df = self._filter_section_routes_safe(section_name, single_section_only)
            if df.empty:
                logger.debug("Нет маршрутов для участка %s", section_name)
                return []

            # Подсчитываем нормы с безопасным преобразованием
            norm_counts = {}
            
            for norm_raw in df["Номер нормы"].dropna():
                try:
                    norm_str = str(int(safe_float(norm_raw, 0)))
                    if norm_str != "0":  # Исключаем нулевые значения
                        norm_counts[norm_str] = norm_counts.get(norm_str, 0) + 1
                except (ValueError, TypeError):
                    continue
                    
            # Получаем дополнительные известные нормы
            known_norms = self.sections_norms_map.get(section_name, [])
            
            # Объединяем все нормы
            all_norms = set(norm_counts.keys()) | set(known_norms)
            
            # Формируем результат с сортировкой
            result = [(norm, norm_counts.get(norm, 0)) for norm in all_norms if norm and norm != "0"]
            result.sort(key=lambda x: (int(x[0]) if x[0].isdigit() else float('inf'), x[0]))
            
            logger.debug("Найдено норм для участка %s: %d", section_name, len(result))
            return result
            
        except Exception as e:
            logger.error("Ошибка получения норм для участка %s: %s", section_name, e)
            return []

    def get_routes_count_for_section(self, section_name: str, single_section_only: bool = False) -> int:
        """Безопасно возвращает количество маршрутов для участка."""
        try:
            df = self._filter_section_routes_safe(section_name, single_section_only)
            return len(df)
        except Exception as e:
            logger.error("Ошибка подсчета маршрутов для участка %s: %s", section_name, e)
            return 0

    def get_norm_info(self, norm_id: str) -> Optional[Dict]:
        """Безопасно возвращает информацию о норме."""
        try:
            norm_data = self.norm_storage.get_norm(str(norm_id))
            if not norm_data:
                logger.debug("Норма %s не найдена в хранилище", norm_id)
                return None

            points = norm_data.get("points", [])
            
            info = {
                "norm_id": str(norm_id),
                "description": norm_data.get("description", f"Норма №{norm_id}"),
                "norm_type": norm_data.get("norm_type", "Неизвестно"),
                "points_count": len(points),
                "points": points[:10],  # Ограничиваем для производительности
                "base_data": norm_data.get("base_data", {}),
            }

            # Безопасно вычисляем диапазоны
            if points:
                try:
                    x_vals = [float(p[0]) for p in points]
                    y_vals = [float(p[1]) for p in points]
                    info["load_range"] = f"{min(x_vals):.1f} - {max(x_vals):.1f} т/ось"
                    info["consumption_range"] = f"{min(y_vals):.1f} - {max(y_vals):.1f} кВт·ч/10⁴ ткм"
                except Exception:
                    info["load_range"] = "Ошибка расчета"
                    info["consumption_range"] = "Ошибка расчета"
            else:
                info["load_range"] = "Нет данных"
                info["consumption_range"] = "Нет данных"

            return info
            
        except Exception as e:
            logger.error("Ошибка получения информации о норме %s: %s", norm_id, e)
            return None

    # ========================== ИСПРАВЛЕННЫЙ анализ участка ==========================

    def analyze_section(
        self,
        section_name: str,
        norm_id: Optional[str] = None,
        single_section_only: bool = False,
        locomotive_filter: Optional[LocomotiveFilter] = None,
        coefficients_manager: Optional[LocomotiveCoefficientsManager] = None,
        use_coefficients: bool = False,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict], Optional[Dict], Optional[str]]:
        """
        ИСПРАВЛЕННЫЙ анализ участка с полной валидацией и обработкой ошибок.
        
        Returns:
            Tuple (analyzed_routes, norm_functions, statistics, error_message)
        """
        logger.info("=== АНАЛИЗ УЧАСТКА ===")
        logger.info("Участок: %s | Норма: %s | Один участок: %s | Коэффициенты: %s",
                   section_name, norm_id or "Все", single_section_only, use_coefficients)

        # ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ
        if not section_name or not section_name.strip():
            error = "Не указано название участка"
            logger.error(error)
            return None, None, None, error

        if self.routes_df is None or self.routes_df.empty:
            error = "Данные маршрутов не загружены или пусты"
            logger.error(error)
            return None, None, None, error

        try:
            # ЭТАП 1: Подготовка и фильтрация данных
            logger.info("Этап 1: Подготовка данных")
            section_routes = self._prepare_section_data_safe(
                section_name, norm_id, single_section_only,
                locomotive_filter, coefficients_manager, use_coefficients
            )
            
            if section_routes is None:
                error = "Критическая ошибка подготовки данных"
                logger.error(error)
                return None, None, None, error
                
            if section_routes.empty:
                error = self._get_empty_data_message(section_name, norm_id, single_section_only)
                logger.warning(error)
                return None, None, None, error

            logger.info("✓ Подготовлено записей для анализа: %d", len(section_routes))

            # ЭТАП 2: Анализ данных с интерполяцией
            logger.info("Этап 2: Анализ с интерполяцией норм")
            analyzed_data, norm_functions = self.data_analyzer.analyze_section_data(
                section_name, section_routes, norm_id
            )
            
            if analyzed_data is None or analyzed_data.empty:
                error = f"Анализатор не вернул данных для участка {section_name}"
                logger.error(error)
                return None, None, None, error
                
            if not norm_functions:
                error = f"Не найдены функции норм для участка {section_name}"
                logger.error(error)
                return None, None, None, error

            logger.info("✓ Анализ завершен: записей=%d, функций норм=%d", 
                       len(analyzed_data), len(norm_functions))

            # ЭТАП 3: Расчет статистики
            logger.info("Этап 3: Расчет статистики")
            statistics = self.data_analyzer.calculate_statistics(analyzed_data)
            
            if not statistics:
                logger.warning("Статистика не рассчитана")
                statistics = {'total': len(analyzed_data), 'processed': 0}

            # ЭТАП 4: Валидация результатов для графика
            logger.info("Этап 4: Валидация результатов")
            validation_result = self._validate_analysis_results(analyzed_data, norm_functions, statistics)
            
            if not validation_result['valid']:
                error = f"Валидация не пройдена: {validation_result['error']}"
                logger.error(error)
                return None, None, None, error

            # Сохраняем результат
            result_key = f"{section_name}_{norm_id}_{single_section_only}"
            self.analyzed_results[result_key] = {
                "routes": analyzed_data.copy(),  # Создаем копию для безопасности
                "norms": norm_functions.copy(),
                "statistics": statistics.copy()
            }
            
            self.processing_stats['last_analysis'] = result_key

            logger.info("✓ АНАЛИЗ УЧАСТКА ЗАВЕРШЕН УСПЕШНО")
            logger.info("Результат: записей=%d, норм=%d, статистика=%s", 
                       len(analyzed_data), len(norm_functions), 
                       f"обработано {statistics.get('processed', 0)} из {statistics.get('total', 0)}")
            
            return analyzed_data, norm_functions, statistics, None

        except Exception as e:
            error_msg = f"Критическая ошибка анализа участка {section_name}: {str(e)}"
            logger.error("КРИТИЧЕСКАЯ ОШИБКА АНАЛИЗА: %s", e, exc_info=True)
            return None, None, None, error_msg

    # ========================== Экспорт и утилиты ==========================

    def export_routes_to_excel(self, output_file: str) -> bool:
        """Безопасный экспорт данных маршрутов в Excel."""
        try:
            if self.routes_df is None or self.routes_df.empty:
                logger.warning("Нет данных для экспорта в Excel")
                return False
                
            return self.route_processor.export_to_excel(self.routes_df, output_file)
            
        except Exception as e:
            logger.error("Ошибка экспорта в Excel: %s", e)
            return False

    def get_routes_data(self) -> pd.DataFrame:
        """Безопасно возвращает копию данных маршрутов."""
        try:
            return self.routes_df.copy() if self.routes_df is not None else pd.DataFrame()
        except Exception as e:
            logger.error("Ошибка получения данных маршрутов: %s", e)
            return pd.DataFrame()

    def get_norm_storage_info(self) -> Dict:
        """Безопасно возвращает информацию о хранилище норм."""
        try:
            return self.norm_storage.get_storage_info()
        except Exception as e:
            logger.error("Ошибка получения информации о хранилище: %s", e)
            return {'error': str(e)}

    def get_norm_storage_statistics(self) -> Dict:
        """Безопасно возвращает статистику хранилища норм."""
        try:
            return self.norm_storage.get_norm_statistics()
        except Exception as e:
            logger.error("Ошибка получения статистики норм: %s", e)
            return {'error': str(e)}

    def validate_norms_storage(self) -> Dict:
        """Безопасно валидирует хранилище норм."""
        try:
            return self.norm_storage.validate_norms()
        except Exception as e:
            logger.error("Ошибка валидации норм: %s", e)
            return {'valid': [], 'invalid': [f'Ошибка валидации: {e}'], 'warnings': []}

    # ========================== ИСПРАВЛЕННЫЕ внутренние методы ==========================

    def _build_sections_norms_map_safe(self) -> None:
        """Безопасно строит карту участков -> список норм."""
        try:
            if self.routes_df is None or self.routes_df.empty:
                self.sections_norms_map = {}
                return

            # Группируем нормы по участкам с обработкой ошибок
            sections_norms = {}
            
            for section in self.routes_df["Наименование участка"].dropna().unique():
                section_str = str(section).strip()
                if not section_str:
                    continue
                    
                try:
                    section_df = self.routes_df[self.routes_df["Наименование участка"] == section]
                    norms = set()
                    
                    for norm_raw in section_df["Номер нормы"].dropna():
                        try:
                            norm_str = str(int(safe_float(norm_raw, 0)))
                            if norm_str != "0":
                                norms.add(norm_str)
                        except (ValueError, TypeError):
                            continue
                            
                    sections_norms[section_str] = sorted(list(norms), key=lambda x: int(x) if x.isdigit() else 0)
                    
                except Exception as section_error:
                    logger.warning("Ошибка обработки участка %s: %s", section, section_error)
                    sections_norms[section_str] = []
            
            self.sections_norms_map = sections_norms
            logger.info("✓ Карта участков->нормы построена: %d участков", len(self.sections_norms_map))
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка построения карты участков: %s", e, exc_info=True)
            self.sections_norms_map = {}

    def _filter_section_routes_safe(self, section_name: str, single_section_only: bool = False) -> pd.DataFrame:
        """ИСПРАВЛЕННАЯ фильтрация маршрутов по участку с защитой от ошибок."""
        try:
            if self.routes_df is None or self.routes_df.empty:
                return pd.DataFrame()

            # Базовая фильтрация по участку
            df = self.routes_df[self.routes_df["Наименование участка"] == section_name].copy()
            
            if df.empty:
                logger.debug("Участок %s не найден в данных", section_name)
                return df
                
            if not single_section_only:
                return df

            # Фильтр "только один участок" - с защитой от ошибок группировки
            try:
                # Группируем по маршруту и дате, считаем количество участков
                grouping_cols = ["Номер маршрута", "Дата маршрута"]
                
                # Проверяем наличие нужных колонок
                missing_cols = [col for col in grouping_cols if col not in self.routes_df.columns]
                if missing_cols:
                    logger.warning("Отсутствуют колонки для фильтрации: %s", missing_cols)
                    return df  # Возвращаем без фильтрации
                    
                route_counts = self.routes_df.groupby(grouping_cols).size()
                single_section_routes = route_counts[route_counts == 1].index
                
                if len(single_section_routes) == 0:
                    logger.debug("Нет маршрутов с одним участком")
                    return pd.DataFrame()
                
                # Применяем фильтр
                df_indexed = df.set_index(grouping_cols)
                available_indices = df_indexed.index.intersection(single_section_routes)
                
                if len(available_indices) == 0:
                    return pd.DataFrame()
                    
                filtered_df = df_indexed.loc[available_indices]
                return filtered_df.reset_index()
                
            except Exception as filter_error:
                logger.error("Ошибка фильтрации по одному участку: %s", filter_error)
                return df  # Fallback на нефильтрованные данные
                
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка фильтрации участка %s: %s", section_name, e, exc_info=True)
            return pd.DataFrame()

    def _prepare_section_data_safe(
        self,
        section_name: str,
        norm_id: Optional[str],
        single_section_only: bool,
        locomotive_filter: Optional[LocomotiveFilter],
        coefficients_manager: Optional[LocomotiveCoefficientsManager],
        use_coefficients: bool,
    ) -> Optional[pd.DataFrame]:
        """ИСПРАВЛЕННАЯ подготовка данных участка с поэтапной валидацией."""
        try:
            logger.debug("Подготовка данных участка %s", section_name)
            
            # Этап 1: Базовая фильтрация
            df = self._filter_section_routes_safe(section_name, single_section_only)
            if df is None:
                logger.error("Ошибка базовой фильтрации")
                return None
            if df.empty:
                logger.debug("Нет данных после базовой фильтрации")
                return df

            logger.debug("✓ После базовой фильтрации: %d записей", len(df))

            # Этап 2: Фильтрация по норме
            if norm_id:
                try:
                    norm_str = str(int(safe_float(norm_id, 0)))
                    if norm_str != "0":
                        original_count = len(df)
                        df = df[df["Номер нормы"].apply(lambda x: str(int(safe_float(x, 0))) == norm_str)]
                        logger.debug("✓ Фильтрация по норме %s: %d -> %d записей", norm_str, original_count, len(df))
                        
                        if df.empty:
                            logger.debug("Нет данных после фильтрации по норме %s", norm_str)
                            return df
                except Exception as norm_filter_error:
                    logger.error("Ошибка фильтрации по норме %s: %s", norm_id, norm_filter_error)
                    return None

            # Этап 3: Фильтрация локомотивов
            if locomotive_filter:
                try:
                    original_count = len(df)
                    df = locomotive_filter.filter_routes(df)
                    logger.debug("✓ Фильтрация локомотивов: %d -> %d записей", original_count, len(df))
                    
                    if df.empty:
                        logger.debug("Нет данных после фильтрации локомотивов")
                        return df
                except Exception as loco_filter_error:
                    logger.error("Ошибка фильтрации локомотивов: %s", loco_filter_error)
                    return None

            # Этап 4: Применение коэффициентов
            if use_coefficients and coefficients_manager:
                try:
                    original_count = len(df)
                    df = CoefficientsApplier.apply_coefficients(df, coefficients_manager)
                    logger.debug("✓ Применены коэффициенты: %d записей обработано", original_count)
                except Exception as coeff_error:
                    logger.error("Ошибка применения коэффициентов: %s", coeff_error)
                    # Продолжаем без коэффициентов
                    pass

            logger.info("✓ Подготовка данных завершена: %d финальных записей", len(df))
            return df

        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка подготовки данных участка %s: %s", section_name, e, exc_info=True)
            return None

    def _validate_analysis_results(self, routes_df: pd.DataFrame, norm_functions: Dict, statistics: Dict) -> Dict:
        """
        НОВЫЙ метод валидации результатов анализа перед передачей в график.
        Предотвращает передачу некорректных данных.
        """
        try:
            validation = {'valid': True, 'error': None, 'warnings': []}
            
            # 1. Валидация DataFrame
            if routes_df is None or routes_df.empty:
                validation['valid'] = False
                validation['error'] = "DataFrame маршрутов пуст"
                return validation
                
            # 2. Валидация обязательных колонок
            required_columns = ["Номер нормы", "Факт уд", "Статус"]
            missing_columns = [col for col in required_columns if col not in routes_df.columns]
            if missing_columns:
                validation['valid'] = False
                validation['error'] = f"Отсутствуют обязательные колонки: {missing_columns}"
                return validation
                
            # 3. Валидация функций норм
            if not norm_functions or not isinstance(norm_functions, dict):
                validation['valid'] = False
                validation['error'] = "Функции норм некорректны или отсутствуют"
                return validation
                
            # 4. Проверка наличия валидных точек для графика
            valid_points_count = 0
            for _, row in routes_df.iterrows():
                try:
                    # Проверяем X координату
                    brutto = safe_float(row.get("БРУТТО"))
                    osi = safe_float(row.get("ОСИ"))
                    axle_load = safe_float(row.get("Нажатие на ось"))
                    
                    x_val = axle_load if axle_load > 0 else (brutto / osi if brutto > 0 and osi > 0 else None)
                    
                    # Проверяем Y координату
                    y_val = safe_float(row.get("Факт уд"))
                    
                    if x_val and x_val > 0 and y_val and y_val > 0:
                        valid_points_count += 1
                        
                except Exception:
                    continue
                    
            if valid_points_count == 0:
                validation['valid'] = False
                validation['error'] = "Нет точек с валидными координатами для построения графика"
                return validation
                
            if valid_points_count < len(routes_df) * 0.5:  # Менее 50% валидных точек
                validation['warnings'].append(f"Только {valid_points_count} из {len(routes_df)} точек валидны")
                
            # 5. Валидация статистики
            if not statistics.get('processed', 0):
                validation['warnings'].append("Статистика показывает 0 обработанных записей")
                
            logger.info("✓ Валидация пройдена: валидных точек=%d из %d", valid_points_count, len(routes_df))
            return validation
            
        except Exception as e:
            logger.error("Ошибка валидации результатов: %s", e)
            return {'valid': False, 'error': f'Ошибка валидации: {e}'}

    def _get_empty_data_message(self, section_name: str, norm_id: Optional[str], single_section_only: bool) -> str:
        """Формирует информативное сообщение об отсутствии данных."""
        base_msg = f"Участок '{section_name}'"
        
        conditions = []
        if single_section_only:
            conditions.append("только маршруты с одним участком")
        if norm_id:
            conditions.append(f"норма {norm_id}")
            
        if conditions:
            base_msg += f" ({', '.join(conditions)})"
            
        base_msg += " - данных не найдено"
        
        # Добавляем диагностическую информацию
        try:
            total_for_section = len(self.routes_df[self.routes_df["Наименование участка"] == section_name])
            if total_for_section > 0:
                base_msg += f"\nВсего маршрутов для участка: {total_for_section}"
        except Exception:
            pass
            
        return base_msg

    def get_analyzer_statistics(self) -> Dict:
        """Возвращает общую статистику анализатора."""
        try:
            stats = self.processing_stats.copy()
            
            # Добавляем текущую информацию
            if self.routes_df is not None:
                stats['current_routes_count'] = len(self.routes_df)
                stats['current_sections_count'] = len(self.get_sections_list())
            else:
                stats['current_routes_count'] = 0
                stats['current_sections_count'] = 0
                
            # Информация о хранилище норм
            try:
                norm_stats = self.get_norm_storage_statistics()
                stats['norms_in_storage'] = norm_stats.get('total_norms', 0)
            except Exception:
                stats['norms_in_storage'] = 0
                
            return stats
            
        except Exception as e:
            logger.error("Ошибка получения статистики анализатора: %s", e)
            return {'error': str(e)}