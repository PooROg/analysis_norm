# gui/interface.py (обновленный)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Главный интерфейс приложения с встроенной визуализацией."""

from __future__ import annotations

import logging
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

from analysis.analyzer import InteractiveNormsAnalyzer
from core.coefficients import LocomotiveCoefficientsManager
from core.config import APP_CONFIG
from core.filter import LocomotiveFilter
from core.utils import format_number
from dialogs.selector import LocomotiveSelectorDialog
from gui.components import ControlSection, FileSection, VisualizationSection
from visualization import PlotWindow

logger = logging.getLogger(__name__)


class NormsAnalyzerGUI:
    """Главный класс интерфейса анализатора норм с встроенной визуализацией."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Анализатор норм расхода электроэнергии РЖД")
        self.root.geometry(f"{APP_CONFIG.default_window_size[0]}x{APP_CONFIG.default_window_size[1]}")

        # Основные компоненты
        self.analyzer = InteractiveNormsAnalyzer()
        self.locomotive_filter: Optional[LocomotiveFilter] = None
        self.coefficients_manager = LocomotiveCoefficientsManager()
        
        # Окно графика
        self.plot_window: Optional[PlotWindow] = None
        
        # Состояние
        self.use_coefficients = False
        self.exclude_low_work = False

        # Компоненты GUI
        self.file_section: Optional[FileSection] = None
        self.control_section: Optional[ControlSection] = None
        self.viz_section: Optional[VisualizationSection] = None
        self.log_text: Optional[tk.Text] = None

        # Инициализация
        self._setup_gui()
        self._setup_logging()
        self._connect_callbacks()

        # Настройка закрытия
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        logger.info("GUI инициализирован")

    def _setup_matplotlib_environment(self):
        """НОВЫЙ метод настройки matplotlib среды."""
        try:
            # Настройка matplotlib для GUI
            import matplotlib
            matplotlib.use('TkAgg', force=True)
            
            # Отключаем интерактивный режим
            import matplotlib.pyplot as plt
            plt.ioff()
            
            # Настройки DPI и качества
            import matplotlib.rcParams as rcParams
            rcParams['figure.dpi'] = 100
            rcParams['savefig.dpi'] = 300
            rcParams['font.family'] = 'sans-serif'
            rcParams['axes.grid'] = True
            rcParams['grid.alpha'] = 0.3
            
            logger.info("✓ matplotlib среда настроена для GUI")
            
        except Exception as e:
            logger.error("Ошибка настройки matplotlib среды: %s", e)

    def _setup_gui(self):
        """Настраивает интерфейс."""
        self._setup_matplotlib_environment()
        self._setup_styles()
        self._create_layout()

    def _setup_styles(self):
        """Настраивает стили интерфейса."""
        style = ttk.Style()
        style.theme_use('clam')
        
        styles_config = {
            'Title.TLabel': {'font': ('Arial', 14, 'bold')},
            'Header.TLabel': {'font': ('Arial', 11, 'bold')},
            'Success.TLabel': {'foreground': 'green'},
            'Error.TLabel': {'foreground': 'red'},
            'Warning.TLabel': {'foreground': 'orange'},
        }
        
        for style_name, config in styles_config.items():
            style.configure(style_name, **config)

    def _create_layout(self):
        """Создает основной layout."""
        # Главный контейнер
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка растяжения
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(2, weight=1)

        # Создание компонентов
        self.file_section = FileSection(main_container)
        file_frame = self.file_section.create_widgets()
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))

        self.control_section = ControlSection(main_container)
        control_frame = self.control_section.create_widgets()
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # ИЗМЕНЕНО: Обновленная секция визуализации
        self.viz_section = VisualizationSection(main_container)
        viz_frame = self.viz_section.create_widgets()
        viz_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Секция логов
        self._create_log_section(main_container)

    def _create_log_section(self, parent):
        """Создает секцию логирования."""
        log_frame = ttk.LabelFrame(parent, text="Журнал операций", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        ttk.Button(log_frame, text="Очистить логи", command=self._clear_logs).grid(
            row=1, column=0, columnspan=2, pady=(5, 0)
        )

    def _connect_callbacks(self):
        """Подключает callbacks между компонентами."""
        # File section callbacks
        self.file_section.on_routes_loaded = self._on_routes_loaded
        self.file_section.on_norms_loaded = self._on_norms_loaded

        # Control section callbacks
        self.control_section.on_section_selected = self._on_section_selected
        self.control_section.on_single_section_changed = self._on_single_section_changed
        self.control_section.on_analyze_clicked = self._on_analyze_clicked
        self.control_section.on_filter_clicked = self._on_filter_clicked
        self.control_section.on_edit_norms_clicked = self._on_edit_norms_clicked
        self.control_section.on_norm_selected = self._on_norm_selected

        # ИЗМЕНЕНО: Обновленные callbacks для встроенной визуализации
        self.viz_section.on_plot_open = self._open_integrated_plot
        self.viz_section.on_norm_storage_info = self._show_norm_storage_info
        self.viz_section.on_validate_norms = self._validate_norms
        self.viz_section.on_routes_statistics = self._show_routes_statistics

        # Export callbacks
        self.control_section.export_excel_btn.configure(command=self._export_to_excel)
        self.control_section.export_plot_btn.configure(command=self._export_plot)

    # ========================== Event Handlers ==========================

    def _on_routes_loaded(self, files: List[str]):
        """Обработчик загрузки маршрутов."""
        self.file_section.update_status("Загрузка маршрутов...", "warning")
        
        def load_routes():
            return self.analyzer.load_routes_from_html(files)
        
        def on_success(success: bool):
            if success:
                self.file_section.update_status("Маршруты загружены", "success")
                self._update_after_routes_loaded()
            else:
                self.file_section.update_status("Ошибка загрузки маршрутов", "error")
        
        self._run_async(load_routes, on_success)

    def _on_norms_loaded(self, files: List[str]):
        """Обработчик загрузки норм."""
        self.file_section.update_status("Загрузка норм...", "warning")
        
        def load_norms():
            return self.analyzer.load_norms_from_html(files)
        
        def on_success(success: bool):
            status_text = "Маршруты и нормы загружены" if success else "Ошибка загрузки норм"
            status_type = "success" if success else "error"
            self.file_section.update_status(status_text, status_type)
            
            if success:
                self.control_section.enable_buttons({"edit_norms": True})
        
        self._run_async(load_norms, on_success)

    def _on_section_selected(self, section_name: str):
        """Обработчик выбора участка."""
        if not section_name:
            return
        
        self._update_norms_and_section_info()
        logger.info("Выбран участок: %s", section_name)

    def _on_single_section_changed(self, single_section_only: bool):
        """Обработчик изменения фильтра по одному участку."""
        section = self.control_section.section_var.get()
        if section:
            self._update_norms_and_section_info()
            filter_status = "включен" if single_section_only else "выключен"
            logger.info("Фильтр 'только один участок' %s", filter_status)

    def _on_analyze_clicked(self):
        """Обработчик анализа участка."""
        section = self.control_section.section_var.get()
        if not section:
            messagebox.showwarning("Предупреждение", "Выберите участок для анализа")
            return

        norm_text = self.control_section.norm_var.get()
        norm_id = self._extract_norm_id_from_text(norm_text)
        single_section_only = self.control_section.single_section_only.get()

        logger.info("Начинаем анализ участка: %s", section)

        def analyze():
            return self.analyzer.analyze_section(
                section,
                norm_id=norm_id,
                single_section_only=single_section_only,
                locomotive_filter=self.locomotive_filter,
                coefficients_manager=self.coefficients_manager,
                use_coefficients=self.use_coefficients,
            )

        def on_success(result):
            routes_df, norm_functions, statistics, error = result
            self._update_analysis_results(routes_df, norm_functions, statistics, error, section, norm_id, single_section_only)

        self._run_async(analyze, on_success)

    def _on_filter_clicked(self):
        """Обработчик фильтра локомотивов."""
        if self.locomotive_filter is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные маршрутов")
            return

        dialog = LocomotiveSelectorDialog(self.root, self.locomotive_filter, self.coefficients_manager)
        self.root.wait_window(dialog.d)

        if dialog.res:
            self._handle_filter_results(dialog.res)

    def _on_edit_norms_clicked(self):
        """Обработчик редактирования норм."""
        section = self.control_section.section_var.get()
        if not section:
            messagebox.showwarning("Предупреждение", "Выберите участок для редактирования норм")
            return

        messagebox.showinfo(
            "Информация",
            "Функция редактирования норм будет реализована в следующих обновлениях.\n"
            "Сейчас нормы загружаются из HTML файлов и хранятся в высокопроизводительном хранилище.",
        )

    def _on_norm_selected(self, action: str):
        """Обработчик выбора нормы."""
        if action == "info":
            self._show_norm_info()

    # ========================== Business Logic ==========================

    def _update_after_routes_loaded(self):
        """Обновляет интерфейс после загрузки маршрутов."""
        # Обновляем список участков
        sections = self.analyzer.get_sections_list()
        self.control_section.update_sections(sections)

        # Активируем кнопки
        button_states = {
            "analyze": True,
            "filter": True,
            "export_excel": True,
        }
        self.control_section.enable_buttons(button_states)

        # Создаем фильтр локомотивов
        routes_data = self.analyzer.get_routes_data()
        if not routes_data.empty:
            self.locomotive_filter = LocomotiveFilter(routes_data)

        logger.info("Интерфейс обновлен после загрузки маршрутов")

    def _update_norms_and_section_info(self):
        """Обновляет список норм с количеством маршрутов и информацию об участке."""
        section = self.control_section.section_var.get()
        if not section:
            return

        single_section_filter = self.control_section.single_section_only.get()
        
        # Получаем нормы с количествами
        norms_with_counts = self.analyzer.get_norms_with_counts_for_section(section, single_section_filter)
        self.control_section.update_norms(norms_with_counts)

        # Обновляем информацию об участке
        total_routes = self.analyzer.get_routes_count_for_section(section, single_section_filter)
        filter_text = " (только один участок)" if single_section_filter else ""
        info_text = f"Участок: {total_routes} маршрутов{filter_text}"
        self.control_section.update_section_info(info_text)

    def _extract_norm_id_from_text(self, norm_text: str) -> Optional[str]:
        """Извлекает ID нормы из текста вида 'Норма 123 (45 маршрутов)'."""
        if not norm_text or norm_text == "Все нормы":
            return None
        
        import re
        match = re.search(r'Норма (\d+)', norm_text)
        return match.group(1) if match else None

    def _show_norm_info(self):
        """Показывает информацию о выбранной норме."""
        norm_text = self.control_section.norm_var.get()
        norm_id = self._extract_norm_id_from_text(norm_text)
        if not norm_id:
            return

        norm_info = self.analyzer.get_norm_info(norm_id)
        if not norm_info:
            messagebox.showwarning("Предупреждение", f"Информация о норме {norm_id} не найдена")
            return

        self._show_info_window(f"Информация о норме №{norm_id}", self._format_norm_info(norm_info))

    def _format_norm_info(self, norm_info: dict) -> str:
        """Форматирует информацию о норме."""
        content = f"ИНФОРМАЦИЯ О НОРМЕ №{norm_info['norm_id']}\n" + "=" * 50 + "\n\n"
        content += f"Описание: {norm_info['description']}\n"
        content += f"Тип нормы: {norm_info['norm_type']}\n"
        content += f"Количество точек: {norm_info['points_count']}\n"
        content += f"Диапазон нагрузки: {norm_info['load_range']}\n"
        content += f"Диапазон расхода: {norm_info['consumption_range']}\n\n"

        if norm_info['points']:
            content += "ТОЧКИ НОРМЫ:\n" + "-" * 30 + "\n"
            content += f"{'№':<3} {'Нагрузка, т/ось':<15} {'Расход, кВт·ч/10⁴ ткм':<20}\n"
            content += "-" * 50 + "\n"
            for i, (load, consumption) in enumerate(norm_info['points'], 1):
                content += f"{i:<3} {load:<15.2f} {consumption:<20.1f}\n"

        return content

    def _handle_filter_results(self, filter_results: dict):
        """Обрабатывает результаты фильтрации локомотивов."""
        self.use_coefficients = filter_results['use_coefficients']
        self.exclude_low_work = filter_results.get('exclude_low_work', False)
        self.coefficients_manager = filter_results['coefficients_manager']

        # Обновляем информацию о фильтре
        selected_count = len(filter_results['selected_locomotives'])
        self.control_section.update_filter_info(f"Выбрано локомотивов: {selected_count}")

        # Перезапускаем анализ если участок выбран
        if self.control_section.section_var.get():
            self._on_analyze_clicked()

    def _update_analysis_results(self, routes_df, norm_functions, statistics, error, section_name: str, 
                                norm_id: Optional[str], single_section_only: bool):
        """ИСПРАВЛЕННОЕ обновление результатов анализа с защитой от ошибок."""
        if error:
            logger.error("Ошибка анализа: %s", error)
            messagebox.showerror("Ошибка анализа", error)
            return

        if routes_df is None or norm_functions is None or statistics is None:
            error_msg = "Анализ вернул некорректные данные (None значения)"
            logger.error(error_msg)
            messagebox.showerror("Ошибка", error_msg)
            return

        try:
            # БЕЗОПАСНО сохраняем данные для графика с валидацией
            plot_data = {
                'section_name': str(section_name),
                'routes_df': routes_df.copy(),  # Создаем независимую копию
                'norm_functions': dict(norm_functions),  # Создаем независимую копию
                'specific_norm_id': str(norm_id) if norm_id else None,
                'single_section_only': bool(single_section_only)
            }
            
            # Дополнительная валидация перед сохранением
            if self._validate_plot_data(plot_data):
                self._current_plot_data = plot_data
                logger.info("✓ Данные графика сохранены и валидированы")
            else:
                logger.error("Валидация данных графика не пройдена")
                messagebox.showerror("Ошибка", "Данные анализа некорректны для построения графика")
                return

            # Обновляем интерфейс
            self.viz_section.enable_plot_button(True)
            self.control_section.enable_buttons({"export_plot": True})
            self.control_section.update_statistics(statistics)
            self.viz_section.update_plot_info(section_name, statistics, norm_id, single_section_only)

            # Логирование результата
            norm_text = f" с нормой {norm_id}" if norm_id else ""
            filter_text = " (только один участок)" if single_section_only else ""
            logger.info("✓ Анализ участка %s%s%s завершен успешно", section_name, norm_text, filter_text)
            logger.info("Результат: %d записей, %d функций норм, %d обработанных маршрутов",
                    len(routes_df), len(norm_functions), statistics.get('processed', 0))

        except Exception as e:
            error_msg = f"Ошибка сохранения результатов анализа: {str(e)}"
            logger.error("КРИТИЧЕСКАЯ ОШИБКА: %s", e, exc_info=True)
            messagebox.showerror("Критическая ошибка", error_msg)

    # ========================== Встроенная визуализация ==========================

    def _open_integrated_plot(self):
        """ИСПРАВЛЕННЫЙ метод открытия интегрированного графика БЕЗ THREADING."""
        if not hasattr(self, '_current_plot_data') or not self._current_plot_data:
            messagebox.showwarning("Предупреждение", "Сначала выполните анализ участка")
            return

        try:
            # Проверяем, не открыто ли уже окно графика
            if self.plot_window and self.plot_window.is_active():
                self.plot_window.bring_to_front()
                messagebox.showinfo("Информация", "Окно графика уже открыто")
                return

            logger.info("=== ОТКРЫТИЕ ВСТРОЕННОГО ГРАФИКА ===")
            
            # Валидация данных перед созданием окна
            plot_data = self._current_plot_data
            if not self._validate_plot_data(plot_data):
                messagebox.showerror("Ошибка", "Данные для графика некорректны")
                return
            
            # СИНХРОННОЕ создание окна графика
            logger.info("Создание окна графика...")
            self.plot_window = PlotWindow(self.root)
            
            # Создаем окно - может вызвать исключение
            plot_window_tk = self.plot_window.create_window()
            
            logger.info("✓ Окно создано, начинаем отображение графика")
            
            # СИНХРОННОЕ отображение графика - без threading!
            self.plot_window.show_plot(**plot_data)
            
            logger.info("✓ Встроенный график открыт успешно")

        except Exception as e:
            error_msg = f"Критическая ошибка открытия графика: {str(e)}"
            logger.error("КРИТИЧЕСКАЯ ОШИБКА: %s", e, exc_info=True)
            
            # Очищаем поврежденное окно
            if self.plot_window:
                try:
                    self.plot_window._on_window_close_safe()
                except Exception:
                    pass
                self.plot_window = None
                
            messagebox.showerror("Критическая ошибка", error_msg)

    def _validate_plot_data(self, plot_data: Dict) -> bool:
        """НОВЫЙ метод валидации данных графика перед передачей."""
        try:
            required_keys = ['section_name', 'routes_df', 'norm_functions']
            missing_keys = [key for key in required_keys if key not in plot_data]
            
            if missing_keys:
                logger.error("Отсутствуют ключевые данные для графика: %s", missing_keys)
                return False
                
            # Валидируем DataFrame
            routes_df = plot_data['routes_df']
            if routes_df is None or routes_df.empty:
                logger.error("DataFrame маршрутов пуст")
                return False
                
            # Валидируем функции норм
            norm_functions = plot_data['norm_functions']
            if not norm_functions or not isinstance(norm_functions, dict):
                logger.error("Функции норм некорректны")
                return False
                
            # Проверяем наличие обязательных колонок
            required_columns = ["Номер нормы", "Факт уд", "Статус"]
            missing_columns = [col for col in required_columns if col not in routes_df.columns]
            if missing_columns:
                logger.error("Отсутствуют обязательные колонки: %s", missing_columns)
                return False
                
            logger.info("✓ Валидация данных графика пройдена")
            return True
            
        except Exception as e:
            logger.error("Ошибка валидации данных графика: %s", e)
            return False
    
    # ========================== Utility Methods ==========================

    def _run_async(self, func, on_done, on_error=None):
        """Запускает функцию в фоне."""
        def worker():
            try:
                result = func()
                self.root.after(0, lambda: on_done(result))
            except Exception as e:
                logger.error("Ошибка фоновой операции: %s", e)
                if on_error:
                    self.root.after(0, lambda: on_error(e))
        
        threading.Thread(target=worker, daemon=True).start()

    def _show_info_window(self, title: str, content: str):
        """Показывает информационное окно."""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("700x600")
        window.transient(self.root)

        main_frame = ttk.Frame(window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        text_widget = tk.Text(main_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(main_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        text_widget.insert(1.0, content)
        text_widget.config(state='disabled')

        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(window, text="Закрыть", command=window.destroy).pack(pady=10)

    def _clear_logs(self):
        """Очищает журнал логов."""
        if self.log_text:
            self.log_text.delete(1.0, tk.END)

    # ========================== Export Methods ==========================

    def _export_to_excel(self):
        """Экспортирует данные в Excel."""
        routes_data = self.analyzer.get_routes_data()
        if routes_data.empty:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".xlsx", 
            filetypes=[("Excel files", "*.xlsx")]
        )
        
        if filename:
            try:
                success = self.analyzer.export_routes_to_excel(filename)
                if success:
                    logger.info("Данные экспортированы в %s", Path(filename).name)
                    messagebox.showinfo("Успех", f"Данные успешно экспортированы в Excel!\n\nФайл: {Path(filename).name}")
                else:
                    messagebox.showerror("Ошибка", "Не удалось экспортировать данные")
            except Exception as e:
                logger.error("Ошибка экспорта: %s", e)
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")

    def _export_plot(self):
        """Экспортирует график через окно графика."""
        if not self.plot_window or not self.plot_window.is_active():
            messagebox.showwarning("Предупреждение", "Сначала откройте график")
            return
            
        messagebox.showinfo("Информация", "Используйте кнопки экспорта в окне графика")
        self.plot_window.bring_to_front()

    # ========================== Info Methods ==========================

    def _show_norm_storage_info(self):
        """Показывает информацию о хранилище норм."""
        try:
            storage_info = self.analyzer.get_norm_storage_info()
            norm_stats = self.analyzer.get_norm_storage_statistics()
            content = self._format_storage_info(storage_info, norm_stats)
            self._show_info_window("Информация о хранилище норм", content)
        except Exception as e:
            logger.error("Ошибка получения информации о хранилище: %s", e)
            messagebox.showerror("Ошибка", f"Не удалось получить информацию: {str(e)}")

    def _format_storage_info(self, storage_info: dict, norm_stats: dict) -> str:
        """Форматирует информацию о хранилище норм."""
        info = "ИНФОРМАЦИЯ О ХРАНИЛИЩЕ НОРМ\n" + "=" * 50 + "\n\n"
        info += f"Файл хранилища: {storage_info.get('storage_file', 'N/A')}\n"
        info += f"Размер файла: {storage_info.get('file_size_mb', 0):.2f} MB\n"
        info += f"Версия: {storage_info.get('version', 'N/A')}\n"
        info += f"Последнее обновление: {storage_info.get('last_updated', 'N/A')}\n\n"

        info += "СТАТИСТИКА НОРМ:\n" + "-" * 30 + "\n"
        info += f"Всего норм: {norm_stats.get('total_norms', 0)}\n"
        info += f"Кэшированных функций: {storage_info.get('cached_functions', 0)}\n"
        info += f"Среднее количество точек на норму: {norm_stats.get('avg_points_per_norm', 0):.1f}\n\n"

        by_type = norm_stats.get('by_type', {})
        if by_type:
            info += "По типам норм:\n"
            for norm_type, count in by_type.items():
                info += f"  {norm_type}: {count}\n"

        return info

    def _validate_norms(self):
        """Валидирует нормы в хранилище."""
        try:
            validation_results = self.analyzer.validate_norms_storage()
            self._show_validation_results(validation_results)
        except Exception as e:
            logger.error("Ошибка валидации норм: %s", e)
            messagebox.showerror("Ошибка", f"Ошибка валидации: {str(e)}")

    def _show_validation_results(self, validation_results: dict):
        """Показывает результаты валидации."""
        window = tk.Toplevel(self.root)
        window.title("Результаты валидации норм")
        window.geometry("700x600")
        window.transient(self.root)

        notebook = ttk.Notebook(window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tabs = [
            ("Валидные", validation_results['valid']),
            ("Невалидные", validation_results['invalid']),
            ("Предупреждения", validation_results['warnings']),
        ]

        for title, lines in tabs:
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=f"{title} ({len(lines)})")
            
            text = tk.Text(frame, wrap=tk.WORD)
            scroll = ttk.Scrollbar(frame, command=text.yview)
            text.configure(yscrollcommand=scroll.set)
            
            text.insert(1.0, "\n".join(lines))
            text.config(state='disabled')
            
            text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(window, text="Закрыть", command=window.destroy).pack(pady=10)

    def _show_routes_statistics(self):
        """Показывает подробную статистику маршрутов."""
        routes_data = self.analyzer.get_routes_data()
        if routes_data.empty:
            messagebox.showwarning("Предупреждение", "Нет данных маршрутов")
            return

        try:
            content = self._format_routes_statistics(routes_data)
            self._show_info_window("Статистика обработанных маршрутов", content)
        except Exception as e:
            logger.error("Ошибка формирования статистики: %s", e)
            messagebox.showerror("Ошибка", f"Не удалось получить статистику: {str(e)}")

    def _format_routes_statistics(self, routes_data) -> str:
        """Форматирует статистику маршрутов."""
        processing_stats = self.analyzer.route_processor.get_processing_stats()

        content = "ПОДРОБНАЯ СТАТИСТИКА МАРШРУТОВ\n" + "=" * 60 + "\n\n"
        content += "ОБРАБОТКА ФАЙЛОВ:\n" + "-" * 30 + "\n"
        content += f"Файлов обработано: {processing_stats['total_files']}\n"
        content += f"Маршрутов найдено: {processing_stats['total_routes_found']}\n"
        content += f"Уникальных маршрутов: {processing_stats['unique_routes']}\n"
        content += f"Дубликатов удалено: {processing_stats['duplicates_total']}\n"
        content += f"Обработано успешно: {processing_stats['routes_processed']}\n"
        content += f"Итоговых записей: {processing_stats['output_rows']}\n\n"

        # Статистика по участкам
        sections_stats = routes_data.groupby('Наименование участка').size().sort_values(ascending=False)
        content += "СТАТИСТИКА ПО УЧАСТКАМ:\n" + "-" * 30 + "\n"
        content += f"Всего участков: {len(sections_stats)}\n"
        content += "Топ-10 участков по количеству маршрутов:\n"
        for section, count in sections_stats.head(10).items():
            content += f"  {section}: {count} маршрутов\n"

        return content

    def _setup_logging(self):
        """Настраивает логирование в GUI."""
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget: tk.Text):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                if not self.text_widget.winfo_exists():
                    return
                try:
                    msg = self.format(record)
                    self.text_widget.after(0, lambda: self._append_log(msg, record.levelname))
                except Exception:
                    pass

            def _append_log(self, msg, level):
                if not self.text_widget.winfo_exists():
                    return
                try:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_msg = f"[{timestamp}] {level}: {msg}\n"
                    self.text_widget.insert(tk.END, formatted_msg)
                    self.text_widget.see(tk.END)
                    
                    # Цветовое кодирование
                    color_map = {'ERROR': 'red', 'WARNING': 'orange', 'INFO': 'blue'}
                    if level in color_map:
                        self.text_widget.tag_add(level.lower(), "end-2l", "end-1l")
                        self.text_widget.tag_config(level.lower(), foreground=color_map[level])
                except Exception:
                    pass

        if self.log_text:
            gui_handler = GUILogHandler(self.log_text)
            gui_handler.setLevel(logging.INFO)
            gui_handler.setFormatter(logging.Formatter('%(name)s - %(message)s'))

            root_logger = logging.getLogger()
            root_logger.addHandler(gui_handler)

    def on_closing(self):
        """Обработчик закрытия окна."""
        try:
            # Закрываем окно графика если оно открыто
            if self.plot_window and self.plot_window.is_active():
                self.plot_window._on_window_close()
                
            logger.info("Приложение закрывается")
        except Exception as e:
            logger.error("Ошибка при закрытии: %s", e)
        finally:
            self.root.quit()
            self.root.destroy()