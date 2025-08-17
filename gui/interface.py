# gui/interface.py (обновленный)
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
        self.root.title("Анализатор норм расхода электроэнергии РЖД (HTML версия)")
        self.root.geometry("1400x800")
        
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
        
        # Создаем интерфейс
        self.create_widgets()
        self.setup_styles()
        
        # Настройка логирования в GUI
        self.setup_logging()
        
        # Привязываем обработчик закрытия окна
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        logger.info("GUI инициализирован")
    
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
        control_frame.rowconfigure(7, weight=1)
        
        # Выбор участка
        ttk.Label(control_frame, text="Участок:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.section_var = tk.StringVar()
        self.section_combo = ttk.Combobox(control_frame, textvariable=self.section_var, 
                                        state='readonly', width=40)
        self.section_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.section_combo.bind('<<ComboboxSelected>>', self.on_section_selected)
        
        # Кнопки управления
        self.analyze_button = ttk.Button(control_frame, text="Анализировать участок", 
                                       command=self.analyze_section, state='disabled')
        self.analyze_button.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.filter_button = ttk.Button(control_frame, text="Фильтр локомотивов", 
                                      command=self.open_locomotive_filter, state='disabled')
        self.filter_button.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.edit_norms_button = ttk.Button(control_frame, text="Редактировать нормы", 
                                          command=self.edit_norms, state='disabled')
        self.edit_norms_button.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Информация о фильтрах
        self.filter_info_label = ttk.Label(control_frame, text="", style='Warning.TLabel')
        self.filter_info_label.grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        
        # Статистика
        ttk.Label(control_frame, text="Статистика:", style='Header.TLabel').grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        self.stats_text = tk.Text(control_frame, width=45, height=12, wrap=tk.WORD)
        self.stats_text.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        stats_scrollbar = ttk.Scrollbar(control_frame, orient='vertical', command=self.stats_text.yview)
        stats_scrollbar.grid(row=7, column=1, sticky=(tk.N, tk.S), pady=(0, 10))
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        # Кнопки экспорта
        export_frame = ttk.Frame(control_frame)
        export_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
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
        norm_info_frame = ttk.LabelFrame(viz_frame, text="Информация о нормах", padding="5")
        norm_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.norm_info_button = ttk.Button(norm_info_frame, text="Информация о хранилище норм", 
                                         command=self.show_norm_storage_info)
        self.norm_info_button.pack(pady=5)
        
        self.validate_norms_button = ttk.Button(norm_info_frame, text="Валидировать нормы", 
                                              command=self.validate_norms)
        self.validate_norms_button.pack(pady=5)
        
        # Информация о графике
        self.plot_info = tk.Text(viz_frame, width=60, height=25, wrap=tk.WORD)
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
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
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
            
            # Создаем фильтр локомотивов
            if hasattr(self.analyzer, 'routes_df') and self.analyzer.routes_df is not None:
                self.locomotive_filter = LocomotiveFilter(self.analyzer.routes_df)
            
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
        self.analyze_section()
    
    def analyze_section(self):
        """Анализирует выбранный участок"""
        section = self.section_var.get()
        if not section:
            return
        
        logger.info(f"Начинаем анализ участка: {section}")
        
        # Запускаем анализ в отдельном потоке
        threading.Thread(target=self._analyze_section_thread, args=(section,), daemon=True).start()
    
    def _analyze_section_thread(self, section_name: str):
        """Поток анализа участка"""
        try:
            fig, statistics, error = self.analyzer.analyze_section(
                section_name,
                locomotive_filter=self.locomotive_filter,
                coefficients_manager=self.coefficients_manager,
                use_coefficients=self.use_coefficients
            )
            
            self.root.after(0, self._update_analysis_results, fig, statistics, error)
            
        except Exception as e:
            logger.error(f"Ошибка анализа участка {section_name}: {e}")
            self.root.after(0, self._update_analysis_results, None, None, str(e))
    
    def _update_analysis_results(self, fig, statistics, error):
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
        plot(fig, filename=self.temp_html_file, auto_open=False)
        
        # Активируем кнопки
        self.view_plot_button.config(state='normal')
        self.export_plot_button.config(state='normal')
        
        # Обновляем статистику
        self.update_statistics(statistics)
        
        # Обновляем информацию о графике
        self.update_plot_info(self.section_var.get(), statistics)
        
        logger.info(f"Анализ участка {self.section_var.get()} завершен")
    
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
        text += f"Среднее отклонение: {stats['mean_deviation']:.1f}%\n\n"
        
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
                text += f"{name}: {percent:.1f}%\n"
        
        self.stats_text.insert(1.0, text)
    
    def update_plot_info(self, section_name: str, stats: Dict):
        """Обновляет информацию о графике"""
        self.plot_info.delete(1.0, tk.END)
        
        text = "ИНТЕРАКТИВНЫЙ ГРАФИК\n" + "=" * 40 + "\n\n"
        text += f"Участок: {section_name}\n\n"
        text += "Возможности графика:\n"
        text += "• Наведите курсор на точку для просмотра подробной информации\n"
        text += "• Используйте колесо мыши для масштабирования\n"
        text += "• Зажмите левую кнопку мыши для перемещения\n"
        text += "• Двойной клик для сброса масштаба\n"
        text += "• Клик по легенде для скрытия/показа элементов\n\n"
        text += "Верхний график:\n"
        text += "• Линии - кривые норм\n"
        text += "• Квадраты - опорные точки норм\n"
        text += "• Круги - фактические значения маршрутов\n\n"
        text += "Нижний график:\n"
        text += "• Точки сгруппированы по отклонениям от нормы\n"
        text += "• Зеленая зона - допустимые отклонения (±5%)\n"
        text += "• Оранжевые линии - границы значительных отклонений (±20%)\n\n"
        text += "Для просмотра в полноэкранном режиме\nнажмите 'Открыть график в браузере'"
        
        self.plot_info.insert(1.0, text)
    
    def update_plot_info_default(self):
        """Обновляет информацию о графике (по умолчанию)"""
        self.plot_info.delete(1.0, tk.END)
        
        text = "АНАЛИЗАТОР НОРМ РАСХОДА ЭЛЕКТРОЭНЕРГИИ\n" + "=" * 45 + "\n\n"
        text += "Для начала работы:\n\n"
        text += "1. Выберите HTML файлы маршрутов\n"
        text += "   • Можно выбрать несколько файлов\n"
        text += "   • Файлы будут автоматически очищены от лишнего кода\n"
        text += "   • Дубликаты маршрутов будут отфильтрованы\n\n"
        text += "2. Выберите HTML файлы норм (опционально)\n"
        text += "   • Нормы будут добавлены в высокопроизводительное хранилище\n"
        text += "   • Существующие нормы будут обновлены при необходимости\n\n"
        text += "3. Загрузите выбранные файлы\n\n"
        text += "4. Выберите участок для анализа\n\n"
        text += "5. Настройте фильтры локомотивов (опционально)\n\n"
        text += "6. Анализируйте результаты на интерактивном графике\n\n"
        text += "Особенности HTML версии:\n"
        text += "• Автоматическая очистка HTML файлов\n"
        text += "• Извлечение номеров норм из участков\n"
        text += "• Фильтрация по идентификаторам маршрутов\n"
        text += "• Высокопроизводительное хранилище норм\n"
        text += "• Подробное логирование операций"
        
        self.plot_info.insert(1.0, text)
    
    def open_plot_in_browser(self):
        """Открывает график в браузере"""
        if self.temp_html_file and os.path.exists(self.temp_html_file):
            webbrowser.open(f'file://{os.path.abspath(self.temp_html_file)}')
            logger.info("График открыт в браузере")
        else:
            messagebox.showwarning("Предупреждение", "График не найден. Выполните анализ участка.")
    
    def export_to_excel(self):
        """Экспортирует данные в Excel"""
        if not hasattr(self.analyzer, 'routes_df') or self.analyzer.routes_df is None:
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
                    messagebox.showinfo("Успех", "Данные успешно экспортированы")
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
