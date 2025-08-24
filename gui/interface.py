# gui/interface.py (обновленный с подсчетом маршрутов и фильтром по одному участку)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from plotly.offline import plot
import webbrowser
import os
import tempfile
from datetime import datetime
import threading
import logging
from typing import List, Dict, Optional

from analysis.analyzer import InteractiveNormsAnalyzer
from dialogs.selector import LocomotiveSelectorDialog
from dialogs.editor import NormEditorDialog, NormComparator
from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager

# Настройка логирования
logger = logging.getLogger(__name__)

class NormsAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор норм расхода электроэнергии РЖД (с подсчетом маршрутов)")
        self.root.geometry("1400x900")
        
        # Основные компоненты
        self.analyzer = InteractiveNormsAnalyzer()
        self.current_plot = None
        self.temp_html_file = None
        self.locomotive_filter = None
        self.coefficients_manager = LocomotiveCoefficientsManager()
        self.use_coefficients = False
        self.exclude_low_work = False
        
        # Пути к файлам
        self.route_html_files = []
        self.norm_html_files = []
        
        # Переменная для фильтра "только один участок"
        self.single_section_only = tk.BooleanVar(value=False)
        
        # Создаем интерфейс
        self.create_widgets()
        self.setup_styles()
        
        # Настройка логирования в GUI
        self.setup_logging()
        
        # Привязываем обработчик закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        logger.info("GUI инициализирован с подсчетом маршрутов и фильтром по одному участку")
    
    def setup_logging(self):
        """Настраивает логирование для отображения в GUI"""
        # Создаем handler для отображения логов в текстовом виджете
        class GUILogHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
            
            def emit(self, record):
                try:
                    msg = self.format(record)
                    # Безопасное обновление GUI из другого потока
                    self.text_widget.after(0, lambda: self._append_log(msg, record.levelname))
                except Exception:
                    pass
            
            def _append_log(self, msg, level):
                try:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    formatted_msg = f"[{timestamp}] {level}: {msg}\n"
                    
                    self.text_widget.insert(tk.END, formatted_msg)
                    self.text_widget.see(tk.END)
                    
                    # Цветовое кодирование
                    if level == 'ERROR':
                        self.text_widget.tag_add("error", f"end-2l", f"end-1l")
                        self.text_widget.tag_config("error", foreground="red")
                    elif level == 'WARNING':
                        self.text_widget.tag_add("warning", f"end-2l", f"end-1l")
                        self.text_widget.tag_config("warning", foreground="orange")
                    elif level == 'INFO':
                        self.text_widget.tag_add("info", f"end-2l", f"end-1l")
                        self.text_widget.tag_config("info", foreground="blue")
                except Exception:
                    pass
        
        # Добавляем handler к корневому логгеру
        gui_handler = GUILogHandler(self.log_text)
        gui_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s - %(message)s')
        gui_handler.setFormatter(formatter)
        
        # Добавляем к корневому логгеру
        root_logger = logging.getLogger()
        root_logger.addHandler(gui_handler)
        root_logger.setLevel(logging.INFO)
    
    def setup_styles(self):
        """Настраивает стили интерфейса"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 11, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        # Главный контейнер
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(2, weight=1)
        
        # Секция файлов
        self.create_files_section(main_container)
        
        # Секция управления
        self.create_control_section(main_container)
        
        # Секция визуализации
        self.create_visualization_section(main_container)
        
        # Секция логов
        self.create_log_section(main_container)
    
    def create_files_section(self, parent):
        """Создает секцию работы с файлами"""
        files_frame = ttk.LabelFrame(parent, text="Файлы данных", padding="10")
        files_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        files_frame.columnconfigure(1, weight=1)
        
        # HTML файлы маршрутов
        ttk.Label(files_frame, text="HTML файлы маршрутов:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.routes_label = ttk.Label(files_frame, text="Не выбраны", foreground="gray")
        self.routes_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Button(files_frame, text="Выбрать файлы", 
                  command=self.select_route_html_files).grid(row=0, column=2, padx=(0, 5))
        ttk.Button(files_frame, text="Очистить", 
                  command=self.clear_route_files).grid(row=0, column=3)
        
        # HTML файлы норм
        ttk.Label(files_frame, text="HTML файлы норм:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.norms_label = ttk.Label(files_frame, text="Не выбраны", foreground="gray")
        self.norms_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Button(files_frame, text="Выбрать файлы", 
                  command=self.select_norm_html_files).grid(row=1, column=2, padx=(0, 5), pady=(5, 0))
        ttk.Button(files_frame, text="Очистить", 
                  command=self.clear_norm_files).grid(row=1, column=3, pady=(5, 0))
        
        # Кнопки загрузки
        buttons_frame = ttk.Frame(files_frame)
        buttons_frame.grid(row=2, column=0, columnspan=4, pady=(10, 0))
        
        self.load_routes_button = ttk.Button(buttons_frame, text="Загрузить маршруты", 
                                           command=self.load_routes, state='disabled')
        self.load_routes_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.load_norms_button = ttk.Button(buttons_frame, text="Загрузить нормы", 
                                          command=self.load_norms, state='disabled')
        self.load_norms_button.pack(side=tk.LEFT, padx=(0, 5))
        
        # Статус загрузки
        self.load_status = ttk.Label(files_frame, text="", style='Success.TLabel')
        self.load_status.grid(row=3, column=0, columnspan=4, pady=(5, 0))
    
    def create_control_section(self, parent):
        """Создает секцию управления анализом"""
        control_frame = ttk.LabelFrame(parent, text="Управление анализом", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.rowconfigure(10, weight=1)
        
        # Выбор участка
        ttk.Label(control_frame, text="Участок:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.section_var = tk.StringVar()
        self.section_combo = ttk.Combobox(control_frame, textvariable=self.section_var, 
                                        state='readonly', width=40)
        self.section_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.section_combo.bind('<<ComboboxSelected>>', self.on_section_selected)
        
        # Фильтр по одному участку
        self.single_section_check = ttk.Checkbutton(
            control_frame,
            text="Только маршруты с одним участком",
            variable=self.single_section_only,
            command=self.on_single_section_filter_changed
        )
        self.single_section_check.grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        
        # Выбор нормы
        ttk.Label(control_frame, text="Норма (опционально):", style='Header.TLabel').grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        norm_selection_frame = ttk.Frame(control_frame)
        norm_selection_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        norm_selection_frame.columnconfigure(0, weight=1)
        
        self.norm_var = tk.StringVar()
        self.norm_combo = ttk.Combobox(norm_selection_frame, textvariable=self.norm_var, 
                                     state='readonly', width=30)
        self.norm_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.norm_combo.bind('<<ComboboxSelected>>', self.on_norm_selected)
        
        self.norm_info_button = ttk.Button(norm_selection_frame, text="Инфо о норме", 
                                         command=self.show_norm_info, state='disabled')
        self.norm_info_button.grid(row=0, column=1)
        
        # Информация о количестве маршрутов для участка
        self.section_info_label = ttk.Label(control_frame, text="", style='Warning.TLabel')
        self.section_info_label.grid(row=5, column=0, sticky=tk.W, pady=(0, 10))
        
        # Кнопки управления
        self.analyze_button = ttk.Button(control_frame, text="Анализировать участок", 
                                       command=self.analyze_section, state='disabled')
        self.analyze_button.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.filter_button = ttk.Button(control_frame, text="Фильтр локомотивов", 
                                      command=self.open_locomotive_filter, state='disabled')
        self.filter_button.grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.edit_norms_button = ttk.Button(control_frame, text="Редактировать нормы", 
                                          command=self.edit_norms, state='disabled')
        self.edit_norms_button.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Информация о фильтрах
        self.filter_info_label = ttk.Label(control_frame, text="", style='Warning.TLabel')
        self.filter_info_label.grid(row=9, column=0, sticky=tk.W, pady=(0, 5))
        
        # Статистика
        ttk.Label(control_frame, text="Статистика:", style='Header.TLabel').grid(row=10, column=0, sticky=tk.W, pady=(10, 5))
        self.stats_text = tk.Text(control_frame, width=45, height=8, wrap=tk.WORD)
        self.stats_text.grid(row=11, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        stats_scrollbar = ttk.Scrollbar(control_frame, orient='vertical', command=self.stats_text.yview)
        stats_scrollbar.grid(row=11, column=1, sticky=(tk.N, tk.S), pady=(0, 10))
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        # Кнопки экспорта
        export_frame = ttk.Frame(control_frame)
        export_frame.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.export_excel_button = ttk.Button(export_frame, text="Экспорт в Excel", 
                                            command=self.export_to_excel, state='disabled')
        self.export_excel_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.export_plot_button = ttk.Button(export_frame, text="Экспорт графика", 
                                           command=self.export_plot, state='disabled')
        self.export_plot_button.pack(side=tk.LEFT)
    
    def create_visualization_section(self, parent):
        """Создает секцию визуализации"""
        viz_frame = ttk.LabelFrame(parent, text="Визуализация", padding="10")
        viz_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        viz_frame.rowconfigure(2, weight=1)
        
        # Кнопка открытия графика
        self.view_plot_button = ttk.Button(viz_frame, text="Открыть график в браузере", 
                                         command=self.open_plot_in_browser, state='disabled')
        self.view_plot_button.pack(pady=(0, 10))
        
        # Информация о хранилище норм
        norm_info_frame = ttk.LabelFrame(viz_frame, text="Управление нормами", padding="5")
        norm_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.norm_storage_info_button = ttk.Button(norm_info_frame, text="Информация о хранилище норм", 
                                                 command=self.show_norm_storage_info)
        self.norm_storage_info_button.pack(pady=2)
        
        self.validate_norms_button = ttk.Button(norm_info_frame, text="Валидировать нормы", 
                                              command=self.validate_norms)
        self.validate_norms_button.pack(pady=2)
        
        # Информация о данных
        data_info_frame = ttk.LabelFrame(viz_frame, text="Информация о данных", padding="5")
        data_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.routes_info_button = ttk.Button(data_info_frame, text="Статистика маршрутов", 
                                           command=self.show_routes_statistics)
        self.routes_info_button.pack(pady=2)
        
        # Информация о графике
        self.plot_info = tk.Text(viz_frame, width=60, height=20, wrap=tk.WORD)
        self.plot_info.pack(fill=tk.BOTH, expand=True)
        
        # Инструкции по умолчанию
        self.update_plot_info_default()
    
    def create_log_section(self, parent):
        """Создает секцию логов"""
        log_frame = ttk.LabelFrame(parent, text="Журнал операций", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        
        # Текстовое поле для логов
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        log_scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        log_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        # Кнопка очистки логов
        ttk.Button(log_frame, text="Очистить логи", 
                  command=self.clear_logs).grid(row=1, column=0, columnspan=2, pady=(5, 0))
    
    def log(self, message: str, level: str = 'INFO'):
        """Добавляет сообщение в лог"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {level}: {message}\n"
        
        self.log_text.insert(tk.END, formatted_message)
        self.log_text.see(tk.END)
        
        # Цветовое кодирование
        if level == 'ERROR':
            self.log_text.tag_add("error", f"end-2l", f"end-1l")
            self.log_text.tag_config("error", foreground="red")
        elif level == 'SUCCESS':
            self.log_text.tag_add("success", f"end-2l", f"end-1l")
            self.log_text.tag_config("success", foreground="green")
        elif level == 'WARNING':
            self.log_text.tag_add("warning", f"end-2l", f"end-1l")
            self.log_text.tag_config("warning", foreground="orange")
    
    def clear_logs(self):
        """Очищает журнал логов"""
        self.log_text.delete(1.0, tk.END)
    
    # === МЕТОДЫ РАБОТЫ С ФАЙЛАМИ ===
    
    def select_route_html_files(self):
        """Выбирает HTML файлы маршрутов"""
        files = filedialog.askopenfilenames(
            title="Выберите HTML файлы маршрутов",
            filetypes=[("HTML files", "*.html *.htm"), ("All files", "*.*")]
        )
        
        if files:
            self.route_html_files = list(files)
            file_names = [os.path.basename(f) for f in files]
            
            if len(file_names) <= 3:
                display_text = ", ".join(file_names)
            else:
                display_text = f"{', '.join(file_names[:3])} и еще {len(file_names) - 3} файлов"
            
            self.routes_label.config(text=display_text, foreground="black")
            self.load_routes_button.config(state='normal')
            
            logger.info(f"Выбрано {len(files)} HTML файлов маршрутов")
    
    def select_norm_html_files(self):
        """Выбирает HTML файлы норм"""
        files = filedialog.askopenfilenames(
            title="Выберите HTML файлы норм",
            filetypes=[("HTML files", "*.html *.htm"), ("All files", "*.*")]
        )
        
        if files:
            self.norm_html_files = list(files)
            file_names = [os.path.basename(f) for f in files]
            
            if len(file_names) <= 3:
                display_text = ", ".join(file_names)
            else:
                display_text = f"{', '.join(file_names[:3])} и еще {len(file_names) - 3} файлов"
            
            self.norms_label.config(text=display_text, foreground="black")
            self.load_norms_button.config(state='normal')
            
            logger.info(f"Выбрано {len(files)} HTML файлов норм")
    
    def clear_route_files(self):
        """Очищает выбранные файлы маршрутов"""
        self.route_html_files = []
        self.routes_label.config(text="Не выбраны", foreground="gray")
        self.load_routes_button.config(state='disabled')
        logger.info("Очищен список файлов маршрутов")
    
    def clear_norm_files(self):
        """Очищает выбранные файлы норм"""
        self.norm_html_files = []
        self.norms_label.config(text="Не выбраны", foreground="gray")
        self.load_norms_button.config(state='disabled')
        logger.info("Очищен список файлов норм")
    
    def load_routes(self):
        """Загружает маршруты из HTML файлов"""
        if not self.route_html_files:
            messagebox.showwarning("Предупреждение", "Сначала выберите HTML файлы маршрутов")
            return
        
        self.load_routes_button.config(state='disabled')
        self.load_status.config(text="Загрузка маршрутов...", style='Warning.TLabel')
        
        # Запускаем загрузку в отдельном потоке
        threading.Thread(target=self._load_routes_thread, daemon=True).start()
    
    def _load_routes_thread(self):
        """Поток загрузки маршрутов"""
        try:
            success = self.analyzer.load_routes_from_html(self.route_html_files)
            self.root.after(0, self._update_routes_load_status, success)
        except Exception as e:
            logger.error(f"Ошибка загрузки маршрутов: {e}")
            self.root.after(0, self._update_routes_load_status, False)
    
    def _update_routes_load_status(self, success: bool):
        """Обновляет статус загрузки маршрутов"""
        if success:
            self.load_status.config(text="Маршруты загружены", style='Success.TLabel')
            
            # Обновляем список участков
            sections = self.analyzer.get_sections_list()
            self.section_combo['values'] = sections
            
            # Активируем кнопки
            self.analyze_button.config(state='normal')
            self.filter_button.config(state='normal')
            self.export_excel_button.config(state='normal')
            self.routes_info_button.config(state='normal')
            
            # Создаем фильтр локомотивов
            routes_data = self.analyzer.get_routes_data()
            if not routes_data.empty:
                self.locomotive_filter = LocomotiveFilter(routes_data)
            
            logger.info("Маршруты загружены успешно")
        else:
            self.load_status.config(text="Ошибка загрузки маршрутов", style='Error.TLabel')
            logger.error("Ошибка загрузки маршрутов")
        
        self.load_routes_button.config(state='normal')
    
    def load_norms(self):
        """Загружает нормы из HTML файлов"""
        if not self.norm_html_files:
            messagebox.showwarning("Предупреждение", "Сначала выберите HTML файлы норм")
            return
        
        self.load_norms_button.config(state='disabled')
        self.load_status.config(text="Загрузка норм...", style='Warning.TLabel')
        
        # Запускаем загрузку в отдельном потоке
        threading.Thread(target=self._load_norms_thread, daemon=True).start()
    
    def _load_norms_thread(self):
        """Поток загрузки норм"""
        try:
            success = self.analyzer.load_norms_from_html(self.norm_html_files)
            self.root.after(0, self._update_norms_load_status, success)
        except Exception as e:
            logger.error(f"Ошибка загрузки норм: {e}")
            self.root.after(0, self._update_norms_load_status, False)
    
    def _update_norms_load_status(self, success: bool):
        """Обновляет статус загрузки норм"""
        if success:
            current_text = self.load_status.cget('text')
            if "Маршруты загружены" in current_text:
                self.load_status.config(text="Маршруты и нормы загружены", style='Success.TLabel')
            else:
                self.load_status.config(text="Нормы загружены", style='Success.TLabel')
            
            # Активируем кнопку редактирования норм
            self.edit_norms_button.config(state='normal')
            
            logger.info("Нормы загружены успешно")
        else:
            logger.error("Ошибка загрузки норм")
        
        self.load_norms_button.config(state='normal')
    
    # === МЕТОДЫ АНАЛИЗА ===
    
    def on_section_selected(self, event=None):
        """Обработчик выбора участка"""
        section = self.section_var.get()
        if not section:
            return
        
        # Обновляем информацию о количестве маршрутов и нормы
        self._update_norms_and_section_info()
        
        logger.info(f"Выбран участок: {section}")
    
    def on_single_section_filter_changed(self):
        """Обработчик изменения фильтра по одному участку"""
        section = self.section_var.get()
        if section:
            # Обновляем информацию о количестве маршрутов и нормы
            self._update_norms_and_section_info()
            
            filter_status = "включен" if self.single_section_only.get() else "выключен"
            logger.info(f"Фильтр 'только один участок' {filter_status}")
    
    def _update_norms_and_section_info(self):
        """Обновляет список норм с количеством маршрутов и информацию об участке"""
        section = self.section_var.get()
        if not section:
            return
        
        single_section_filter = self.single_section_only.get()
        
        # Получаем нормы с количеством маршрутов
        norms_with_counts = self.analyzer.get_norms_with_counts_for_section(section, single_section_filter)
        
        # Формируем список для отображения
        norm_values = ["Все нормы"]
        for norm_id, count in norms_with_counts:
            norm_values.append(f"Норма {norm_id} ({count} маршрутов)")
        
        # Обновляем combo box
        self.norm_combo['values'] = norm_values
        self.norm_var.set("Все нормы")
        self.norm_info_button.config(state='disabled')
        
        # Обновляем информацию об участке
        total_routes = self.analyzer.get_routes_count_for_section(section, single_section_filter)
        filter_text = " (только один участок)" if single_section_filter else ""
        self.section_info_label.config(text=f"Участок: {total_routes} маршрутов{filter_text}")
        
        logger.debug(f"Обновлены нормы для участка {section}: {len(norms_with_counts)} норм, {total_routes} маршрутов")
    
    def on_norm_selected(self, event=None):
        """Обработчик выбора нормы"""
        norm_text = self.norm_var.get()
        if norm_text and norm_text != "Все нормы":
            self.norm_info_button.config(state='normal')
        else:
            self.norm_info_button.config(state='disabled')
    
    def _extract_norm_id_from_text(self, norm_text: str) -> Optional[str]:
        """Извлекает ID нормы из текста вида 'Норма 123 (45 маршрутов)'"""
        if not norm_text or norm_text == "Все нормы":
            return None
        
        # Ищем паттерн "Норма XXXX"
        import re
        match = re.search(r'Норма (\d+)', norm_text)
        return match.group(1) if match else None
    
    def show_norm_info(self):
        """Показывает информацию о выбранной норме"""
        norm_text = self.norm_var.get()
        norm_id = self._extract_norm_id_from_text(norm_text)
        
        if not norm_id:
            return
        
        norm_info = self.analyzer.get_norm_info(norm_id)
        if not norm_info:
            messagebox.showwarning("Предупреждение", f"Информация о норме {norm_id} не найдена")
            return
        
        # Создаем окно с информацией о норме
        info_window = tk.Toplevel(self.root)
        info_window.title(f"Информация о норме №{norm_id}")
        info_window.geometry("700x600")
        info_window.transient(self.root)
        
        # Основной фрейм
        main_frame = ttk.Frame(info_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Информационный текст
        info_text = tk.Text(main_frame, wrap=tk.WORD, font=("Consolas", 10))
        info_scrollbar = ttk.Scrollbar(main_frame, command=info_text.yview)
        info_text.config(yscrollcommand=info_scrollbar.set)
        
        # Формируем содержимое
        content = f"ИНФОРМАЦИЯ О НОРМЕ №{norm_id}\n" + "=" * 50 + "\n\n"
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
            
            content += "\n"
        
        # Базовые данные нормы
        if norm_info['base_data']:
            content += "ДОПОЛНИТЕЛЬНЫЕ ДАННЫЕ:\n" + "-" * 30 + "\n"
            for key, value in norm_info['base_data'].items():
                if value is not None and value != '':
                    content += f"{key}: {value}\n"
        
        # Статистика использования нормы
        section = self.section_var.get()
        single_section_filter = self.single_section_only.get()
        if section:
            routes_count = self.analyzer.get_norm_routes_count_for_section(section, norm_id, single_section_filter)
            content += f"\nИСПОЛЬЗОВАНИЕ НОРМЫ:\n" + "-" * 30 + "\n"
            filter_text = " (только один участок)" if single_section_filter else ""
            content += f"Маршрутов на участке '{section}'{filter_text}: {routes_count}\n"
        
        info_text.insert(1.0, content)
        info_text.config(state='disabled')
        
        # Размещение элементов
        info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Кнопка закрытия
        ttk.Button(info_window, text="Закрыть", 
                  command=info_window.destroy).pack(pady=10)
    
    def analyze_section(self):
        """Анализирует выбранный участок"""
        section = self.section_var.get()
        if not section:
            messagebox.showwarning("Предупреждение", "Выберите участок для анализа")
            return
        
        # Определяем норму для анализа
        norm_text = self.norm_var.get()
        norm_id = self._extract_norm_id_from_text(norm_text)
        
        single_section_filter = self.single_section_only.get()
        
        filter_info = " (только один участок)" if single_section_filter else ""
        norm_info = f" с нормой {norm_id}" if norm_id else ""
        
        logger.info(f"Начинаем анализ участка: {section}{norm_info}{filter_info}")
        
        # Запускаем анализ в отдельном потоке
        threading.Thread(target=self._analyze_section_thread, 
                        args=(section, norm_id, single_section_filter), daemon=True).start()
    
    def _analyze_section_thread(self, section_name: str, norm_id: Optional[str], single_section_only: bool):
        """Поток анализа участка"""
        try:
            fig, statistics, error = self.analyzer.analyze_section(
                section_name,
                norm_id=norm_id,
                single_section_only=single_section_only,
                locomotive_filter=self.locomotive_filter,
                coefficients_manager=self.coefficients_manager,
                use_coefficients=self.use_coefficients
            )
            
            self.root.after(0, self._update_analysis_results, fig, statistics, error, section_name, norm_id, single_section_only)
            
        except Exception as e:
            logger.error(f"Ошибка анализа участка {section_name}: {e}")
            self.root.after(0, self._update_analysis_results, None, None, str(e), section_name, norm_id, single_section_only)
    
    def _update_analysis_results(self, fig, statistics, error, section_name: str, norm_id: Optional[str], single_section_only: bool):
        """Обновляет результаты анализа"""
        if error:
            messagebox.showerror("Ошибка", error)
            return
        
        if fig is None or statistics is None:
            messagebox.showwarning("Предупреждение", "Не удалось выполнить анализ")
            return
        
        # Сохраняем график
        self.current_plot = fig
        self.temp_html_file = tempfile.mktemp(suffix='.html')
        
        # ИЗМЕНЕНИЕ: Создаем HTML с переключателем режимов
        from plotly.offline import plot
        
        # Получаем HTML строку вместо сохранения в файл
        html_string = plot(fig, output_type='div', include_plotlyjs=True, 
                        config={'displayModeBar': True, 'locale': 'ru'})
        
        # Создаем полный HTML документ с правильной кодировкой
        full_html = f'''<!DOCTYPE html>
    <html lang="ru">
    <head>
        <title>График анализа участка {section_name}</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body>
        {html_string}
    </body>
    </html>'''
        
        # Добавляем переключатель режимов
        full_html = self.analyzer._add_browser_mode_switcher(full_html)
        
        # Сохраняем с правильной кодировкой
        with open(self.temp_html_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        # Активируем кнопки и обновляем интерфейс
        self.view_plot_button.config(state='normal')
        self.export_plot_button.config(state='normal')
        
        self.update_statistics(statistics)
        self.update_plot_info(section_name, statistics, norm_id, single_section_only)
        
        norm_text = f" с нормой {norm_id}" if norm_id else ""
        filter_text = " (только один участок)" if single_section_only else ""
        logger.info(f"Анализ участка {section_name}{norm_text}{filter_text} завершен")
    
    def open_locomotive_filter(self):
        """Открывает диалог фильтра локомотивов"""
        if self.locomotive_filter is None:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные маршрутов")
            return
        
        dialog = LocomotiveSelectorDialog(self.root, self.locomotive_filter, self.coefficients_manager)
        self.root.wait_window(dialog.d)
        
        if dialog.res:
            self.use_coefficients = dialog.res['use_coefficients']
            self.exclude_low_work = dialog.res.get('exclude_low_work', False)
            self.coefficients_manager = dialog.res['coefficients_manager']
            
            # Обновляем информацию о фильтре
            selected_count = len(dialog.res['selected_locomotives'])
            self.filter_info_label.config(text=f"Выбрано локомотивов: {selected_count}")
            
            # Перезапускаем анализ если участок выбран
            if self.section_var.get():
                self.analyze_section()
    
    def edit_norms(self):
        """Открывает редактор норм"""
        section = self.section_var.get()
        if not section:
            messagebox.showwarning("Предупреждение", "Выберите участок для редактирования норм")
            return
        
        messagebox.showinfo("Информация", 
                          "Функция редактирования норм будет реализована в следующих обновлениях.\n"
                          "Сейчас нормы загружаются из HTML файлов и хранятся в высокопроизводительном хранилище.")
    
    def update_statistics(self, stats: Dict):
        """Обновляет отображение статистики"""
        self.stats_text.delete(1.0, tk.END)
        
        text = f"Всего маршрутов: {stats['total']}\n"
        text += f"Обработано: {stats['processed']}\n"
        text += f"Экономия: {stats['economy']} ({stats['economy']/stats['processed']*100 if stats['processed'] > 0 else 0:.1f}%)\n"
        text += f"В норме: {stats['normal']} ({stats['normal']/stats['processed']*100 if stats['processed'] > 0 else 0:.1f}%)\n"
        text += f"Перерасход: {stats['overrun']} ({stats['overrun']/stats['processed']*100 if stats['processed'] > 0 else 0:.1f}%)\n"
        text += f"Среднее отклонение: {stats['mean_deviation']:.1f}%\n"
        text += f"Медианное отклонение: {stats['median_deviation']:.1f}%\n\n"
        
        text += "Детально:\n"
        detailed = stats['detailed_stats']
        categories = {
            'economy_strong': 'Экономия сильная (>30%)',
            'economy_medium': 'Экономия средняя (20-30%)', 
            'economy_weak': 'Экономия слабая (5-20%)',
            'normal': 'Норма (±5%)',
            'overrun_weak': 'Перерасход слабый (5-20%)',
            'overrun_medium': 'Перерасход средний (20-30%)',
            'overrun_strong': 'Перерасход сильный (>30%)'
        }
        
        for key, name in categories.items():
            count = detailed.get(key, 0)
            percent = count / stats['processed'] * 100 if stats['processed'] > 0 else 0
            if count > 0:
                text += f"{name}: {count} ({percent:.1f}%)\n"
        
        self.stats_text.insert(1.0, text)
    
    def update_plot_info(self, section_name: str, stats: Dict, norm_id: Optional[str] = None, single_section_only: bool = False):
        """Обновляет информацию о графике"""
        self.plot_info.delete(1.0, tk.END)
        
        norm_text = f" (норма {norm_id})" if norm_id else ""
        filter_text = " [только один участок]" if single_section_only else ""
        
        text = "ИНТЕРАКТИВНЫЙ ГРАФИК\n" + "=" * 40 + "\n\n"
        text += f"Участок: {section_name}{norm_text}{filter_text}\n\n"
        text += "Возможности графика:\n"
        text += "• Наведите курсор на точку для просмотра подробной информации\n"
        text += "• Используйте колесо мыши для масштабирования\n"
        text += "• Зажмите левую кнопку мыши для перемещения\n"
        text += "• Двойной клик для сброса масштаба\n"
        text += "• Клик по легенде для скрытия/показа элементов\n\n"
        text += "Верхний график:\n"
        text += "• Линии - кривые норм\n"
        text += "• Квадраты - опорные точки норм\n"
        text += "• Цветные круги - фактические значения маршрутов\n"
        text += "  - Зеленые оттенки: экономия\n"
        text += "  - Золотой: норма (±5%)\n"
        text += "  - Оранжево-красные: перерасход\n\n"
        text += "Нижний график:\n"
        text += "• Точки сгруппированы по отклонениям от нормы\n"
        text += "• Золотая зона - допустимые отклонения (±5%)\n"
        text += "• Оранжевые линии - границы значительных отклонений (±20%)\n"
        text += "• Красные линии - границы критических отклонений (±30%)\n\n"
        
        # Информация о данных
        total_section_routes = self.analyzer.get_routes_count_for_section(section_name, False)
        filtered_section_routes = self.analyzer.get_routes_count_for_section(section_name, single_section_only)
        
        text += f"СТАТИСТИКА УЧАСТКА:\n" + "-" * 30 + "\n"
        text += f"Всего маршрутов участка: {total_section_routes}\n"
        if single_section_only:
            text += f"С одним участком: {filtered_section_routes}\n"
        text += f"Обработано в анализе: {stats['processed']}\n"
        
        if norm_id:
            norm_routes_count = self.analyzer.get_norm_routes_count_for_section(section_name, norm_id, single_section_only)
            text += f"Маршрутов с нормой {norm_id}: {norm_routes_count}\n"
        
        text += "\nДля просмотра в полноэкранном режиме\nнажмите 'Открыть график в браузере'"
        
        self.plot_info.insert(1.0, text)
    
    def update_plot_info_default(self):
        """Обновляет информацию о графике (по умолчанию)"""
        self.plot_info.delete(1.0, tk.END)
        
        text = "АНАЛИЗАТОР НОРМ С ПОДСЧЕТОМ МАРШРУТОВ\n" + "=" * 50 + "\n\n"
        text += "Новые возможности:\n\n"
        text += "1. Подсчет маршрутов по нормам\n"
        text += "   • В списке норм отображается количество маршрутов\n"
        text += "   • Формат: 'Норма 123 (45 маршрутов)'\n"
        text += "   • Обновляется автоматически при смене фильтров\n\n"
        text += "2. Фильтр по одному участку\n"
        text += "   • Галка 'Только маршруты с одним участком'\n"
        text += "   • Позволяет анализировать только маршруты\n"
        text += "     которые проходят только один участок\n"
        text += "   • Полезно для 'чистого' анализа норм\n\n"
        text += "3. Динамическое обновление информации\n"
        text += "   • Количество маршрутов обновляется при изменении фильтра\n"
        text += "   • Отображается общее количество маршрутов участка\n"
        text += "   • Показывается количество после фильтрации\n\n"
        text += "Для начала работы:\n\n"
        text += "1. Выберите и загрузите HTML файлы маршрутов\n"
        text += "2. Выберите и загрузите HTML файлы норм (опционально)\n"
        text += "3. Выберите участок для анализа\n"
        text += "4. Настройте фильтр 'только один участок' при необходимости\n"
        text += "5. Выберите конкретную норму или оставьте 'Все нормы'\n"
        text += "6. Анализируйте результаты на интерактивном графике\n\n"
        text += "ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ:\n" + "-" * 30 + "\n"
        text += "• Подробная информация о нормах (кнопка 'Инфо о норме')\n"
        text += "• Фильтрация локомотивов с коэффициентами\n"
        text += "• Экспорт в Excel с форматированием\n"
        text += "• Интерактивные графики с hover-эффектами\n"
        text += "• Расширенная статистика по всем категориям отклонений"
        
        self.plot_info.insert(1.0, text)
    
    def open_plot_in_browser(self):
        """Открывает график в браузере"""
        if self.temp_html_file and os.path.exists(self.temp_html_file):
            webbrowser.open(f'file://{os.path.abspath(self.temp_html_file)}')
            logger.info("График открыт в браузере")
        else:
            messagebox.showwarning("Предупреждение", "График не найден. Выполните анализ участка.")
    
    def export_to_excel(self):
        """Экспортирует данные в Excel с полным форматированием как в route_processor.py"""
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
                    logger.info(f"Данные экспортированы в {os.path.basename(filename)}")
                    messagebox.showinfo("Успех", 
                                      f"Данные успешно экспортированы в Excel!\n\n"
                                      f"Файл: {os.path.basename(filename)}\n"
                                      f"Записей: {len(routes_data)}\n"
                                      f"Включает: объединенные участки, красное форматирование,\n"
                                      f"все расчеты как в route_processor.py")
                else:
                    messagebox.showerror("Ошибка", "Не удалось экспортировать данные")
            except Exception as e:
                logger.error(f"Ошибка экспорта: {e}")
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
    
    def export_plot(self):
        """Экспортирует график"""
        if not self.current_plot:
            messagebox.showwarning("Предупреждение", "Нет графика для экспорта")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("PNG files", "*.png")]
        )
        
        if filename:
            try:
                if filename.endswith('.html'):
                    self.current_plot.write_html(filename)
                else:
                    self.current_plot.write_image(filename)
                
                logger.info(f"График экспортирован в {os.path.basename(filename)}")
                messagebox.showinfo("Успех", "График успешно экспортирован")
            except Exception as e:
                logger.error(f"Ошибка экспорта графика: {e}")
                messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}")
    
    def show_norm_storage_info(self):
        """Показывает информацию о хранилище норм"""
        try:
            storage_info = self.analyzer.get_norm_storage_info()
            norm_stats = self.analyzer.get_norm_storage_statistics()
            
            info_window = tk.Toplevel(self.root)
            info_window.title("Информация о хранилище норм")
            info_window.geometry("600x500")
            info_window.transient(self.root)
            
            # Создаем текстовое поле с прокруткой
            text_frame = ttk.Frame(info_window)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            info_text = tk.Text(text_frame, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(text_frame, command=info_text.yview)
            info_text.configure(yscrollcommand=scrollbar.set)
            
            # Формируем информацию
            info = "ИНФОРМАЦИЯ О ХРАНИЛИЩЕ НОРМ\n" + "=" * 50 + "\n\n"
            info += f"Файл хранилища: {storage_info.get('storage_file', 'N/A')}\n"
            info += f"Размер файла: {storage_info.get('file_size_mb', 0):.2f} MB\n"
            info += f"Версия: {storage_info.get('version', 'N/A')}\n"
            info += f"Последнее обновление: {storage_info.get('last_updated', 'N/A')}\n\n"
            
            info += "СТАТИСТИКА НОРМ:\n" + "-" * 30 + "\n"
            info += f"Всего норм: {norm_stats.get('total_norms', 0)}\n"
            info += f"Кэшированных функций: {storage_info.get('cached_functions', 0)}\n"
            info += f"Среднее количество точек на норму: {norm_stats.get('avg_points_per_norm', 0):.1f}\n\n"
            
            # Статистика по типам
            by_type = norm_stats.get('by_type', {})
            if by_type:
                info += "По типам норм:\n"
                for norm_type, count in by_type.items():
                    info += f"  {norm_type}: {count}\n"
                info += "\n"
            
            # Диапазоны значений
            load_range = norm_stats.get('load_range', {})
            consumption_range = norm_stats.get('consumption_range', {})
            
            info += "ДИАПАЗОНЫ ЗНАЧЕНИЙ:\n" + "-" * 30 + "\n"
            info += f"Нагрузка на ось: {load_range.get('min', 0):.1f} - {load_range.get('max', 0):.1f} т/ось\n"
            info += f"Удельный расход: {consumption_range.get('min', 0):.1f} - {consumption_range.get('max', 0):.1f} кВт·ч/10⁴ ткм\n\n"
            
            # Распределение точек
            points_dist = norm_stats.get('points_distribution', {})
            if points_dist:
                info += "РАСПРЕДЕЛЕНИЕ ПО КОЛИЧЕСТВУ ТОЧЕК:\n" + "-" * 30 + "\n"
                for points_count in sorted(points_dist.keys()):
                    norms_count = points_dist[points_count]
                    info += f"  {points_count} точек: {norms_count} норм\n"
            
            info_text.insert(1.0, info)
            info_text.config(state='disabled')
            
            info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Кнопка закрытия
            ttk.Button(info_window, text="Закрыть", 
                      command=info_window.destroy).pack(pady=10)
            
        except Exception as e:
            logger.error(f"Ошибка получения информации о хранилище: {e}")
            messagebox.showerror("Ошибка", f"Не удалось получить информацию: {str(e)}")
    
    def validate_norms(self):
        """Валидирует нормы в хранилище"""
        try:
            validation_results = self.analyzer.validate_norms_storage()
            
            # Создаем окно с результатами валидации
            validation_window = tk.Toplevel(self.root)
            validation_window.title("Результаты валидации норм")
            validation_window.geometry("700x600")
            validation_window.transient(self.root)
            
            # Создаем notebook для разных типов результатов
            notebook = ttk.Notebook(validation_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Вкладка с валидными нормами
            valid_frame = ttk.Frame(notebook)
            notebook.add(valid_frame, text=f"Валидные ({len(validation_results['valid'])})")
            
            valid_text = tk.Text(valid_frame, wrap=tk.WORD)
            valid_scroll = ttk.Scrollbar(valid_frame, command=valid_text.yview)
            valid_text.configure(yscrollcommand=valid_scroll.set)
            
            valid_text.insert(1.0, "\n".join(validation_results['valid']))
            valid_text.config(state='disabled')
            
            valid_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            valid_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Вкладка с невалидными нормами
            invalid_frame = ttk.Frame(notebook)
            notebook.add(invalid_frame, text=f"Невалидные ({len(validation_results['invalid'])})")
            
            invalid_text = tk.Text(invalid_frame, wrap=tk.WORD)
            invalid_scroll = ttk.Scrollbar(invalid_frame, command=invalid_text.yview)
            invalid_text.configure(yscrollcommand=invalid_scroll.set)
            
            invalid_text.insert(1.0, "\n".join(validation_results['invalid']))
            invalid_text.config(state='disabled')
            
            invalid_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            invalid_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Вкладка с предупреждениями
            warnings_frame = ttk.Frame(notebook)
            notebook.add(warnings_frame, text=f"Предупреждения ({len(validation_results['warnings'])})")
            
            warnings_text = tk.Text(warnings_frame, wrap=tk.WORD)
            warnings_scroll = ttk.Scrollbar(warnings_frame, command=warnings_text.yview)
            warnings_text.configure(yscrollcommand=warnings_scroll.set)
            
            warnings_text.insert(1.0, "\n".join(validation_results['warnings']))
            warnings_text.config(state='disabled')
            
            warnings_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            warnings_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Кнопка закрытия
            ttk.Button(validation_window, text="Закрыть", 
                      command=validation_window.destroy).pack(pady=10)
            
            logger.info(f"Валидация завершена: {len(validation_results['valid'])} валидных, "
                       f"{len(validation_results['invalid'])} невалидных норм")
            
        except Exception as e:
            logger.error(f"Ошибка валидации норм: {e}")
            messagebox.showerror("Ошибка", f"Ошибка валидации: {str(e)}")
    
    def show_routes_statistics(self):
        """Показывает подробную статистику маршрутов"""
        routes_data = self.analyzer.get_routes_data()
        if routes_data.empty:
            messagebox.showwarning("Предупреждение", "Нет данных маршрутов")
            return
        
        # Создаем окно со статистикой
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Статистика обработанных маршрутов")
        stats_window.geometry("800x600")
        stats_window.transient(self.root)
        
        # Основной фрейм
        main_frame = ttk.Frame(stats_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Текстовое поле со статистикой
        stats_text = tk.Text(main_frame, wrap=tk.WORD, font=("Consolas", 10))
        stats_scrollbar = ttk.Scrollbar(main_frame, command=stats_text.yview)
        stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        # Собираем статистику
        processing_stats = self.analyzer.route_processor.get_processing_stats()
        
        content = "ПОДРОБНАЯ СТАТИСТИКА МАРШРУТОВ\n" + "=" * 60 + "\n\n"
        
        # Общая статистика обработки
        content += "ОБРАБОТКА ФАЙЛОВ:\n" + "-" * 30 + "\n"
        content += f"Файлов обработано: {processing_stats['total_files']}\n"
        content += f"Маршрутов найдено: {processing_stats['total_routes_found']}\n"
        content += f"Уникальных маршрутов: {processing_stats['unique_routes']}\n"
        content += f"Дубликатов удалено: {processing_stats['duplicates_total']}\n"
        content += f"Маршрутов с равными расходами: {processing_stats['routes_with_equal_rashod']}\n"
        content += f"Обработано успешно: {processing_stats['routes_processed']}\n"
        content += f"Пропущено: {processing_stats['routes_skipped']}\n"
        content += f"Итоговых записей: {processing_stats['output_rows']}\n\n"
        
        # Статистика по участкам
        sections_stats = routes_data.groupby('Наименование участка').size().sort_values(ascending=False)
        content += "СТАТИСТИКА ПО УЧАСТКАМ:\n" + "-" * 30 + "\n"
        content += f"Всего участков: {len(sections_stats)}\n"
        content += "Топ-10 участков по количеству маршрутов:\n"
        for section, count in sections_stats.head(10).items():
            content += f"  {section}: {count} маршрутов\n"
        content += "\n"
        
        # Статистика по нормам
        norms_stats = routes_data['Номер нормы'].value_counts().head(10)
        content += "СТАТИСТИКА ПО НОРМАМ:\n" + "-" * 30 + "\n"
        content += f"Всего уникальных норм: {routes_data['Номер нормы'].nunique()}\n"
        content += "Топ-10 норм по использованию:\n"
        for norm, count in norms_stats.items():
            content += f"  Норма {norm}: {count} использований\n"
        content += "\n"
        
        # Статистика по локомотивам
        loco_stats = routes_data.groupby(['Серия локомотива', 'Номер локомотива']).size().sort_values(ascending=False)
        content += "СТАТИСТИКА ПО ЛОКОМОТИВАМ:\n" + "-" * 30 + "\n"
        content += f"Всего уникальных локомотивов: {len(loco_stats)}\n"
        content += "Топ-10 локомотивов по количеству маршрутов:\n"
        for (series, number), count in loco_stats.head(10).items():
            content += f"  {series} №{number}: {count} маршрутов\n"
        content += "\n"
        
        # Статистика по датам
        if 'Дата маршрута' in routes_data.columns:
            try:
                import pandas as pd
                dates_stats = pd.to_datetime(routes_data['Дата маршрута'], format='%d.%m.%Y', errors='coerce').dt.date.value_counts().sort_index()
                content += "СТАТИСТИКА ПО ДАТАМ:\n" + "-" * 30 + "\n"
                content += f"Диапазон дат: {dates_stats.index.min()} - {dates_stats.index.max()}\n"
                content += f"Дней с данными: {len(dates_stats)}\n"
                content += f"Среднее маршрутов в день: {dates_stats.mean():.1f}\n"
                content += f"Максимум маршрутов в день: {dates_stats.max()}\n\n"
            except:
                pass
        
        # Статистика по красным строкам (как в route_processor.py)
        if 'USE_RED_COLOR' in routes_data.columns:
            red_color_count = routes_data['USE_RED_COLOR'].sum()
            content += "СТАТИСТИКА КРАСНОГО ФОРМАТИРОВАНИЯ:\n" + "-" * 30 + "\n"
            content += f"Строк с красным НЕТТО/БРУТТО/ОСИ: {red_color_count}\n"
            content += f"Процент от общего: {red_color_count/len(routes_data)*100:.1f}%\n"
        
        if 'USE_RED_RASHOD' in routes_data.columns:
            red_rashod_count = routes_data['USE_RED_RASHOD'].sum()
            content += f"Строк с красными расходами: {red_rashod_count}\n"
            content += f"Процент от общего: {red_rashod_count/len(routes_data)*100:.1f}%\n"
        
        stats_text.insert(1.0, content)
        stats_text.config(state='disabled')
        
        # Размещение элементов
        stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Кнопка закрытия
        ttk.Button(stats_window, text="Закрыть", 
                  command=stats_window.destroy).pack(pady=10)
    
    def on_closing(self):
        """Обработчик закрытия окна"""
        try:
            # Удаляем временные файлы
            if self.temp_html_file and os.path.exists(self.temp_html_file):
                try:
                    os.remove(self.temp_html_file)
                except:
                    pass
            
            # Закрываем все дочерние окна
            for widget in self.root.winfo_children():
                if isinstance(widget, tk.Toplevel):
                    widget.destroy()
            
            logger.info("Приложение закрывается")
            
            # Закрываем главное окно
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"Ошибка при закрытии приложения: {e}")
            self.root.quit()
            self.root.destroy()