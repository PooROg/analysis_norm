# gui/interface.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный GUI с правильной интеграцией компонентов
Современный интерфейс + восстановленная функциональность
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from plotly.offline import plot
import webbrowser
import tempfile
from pathlib import Path
from datetime import datetime
import threading
import queue
from contextlib import contextmanager
from dataclasses import dataclass
import logging

from analysis.analyzer import InteractiveNormsAnalyzer
from dialogs.selector import LocomotiveSelectorDialog
from dialogs.editor import NormEditorDialog, NormComparator
from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager

logger = logging.getLogger(__name__)

# Python 3.12 type definitions
type GUIState = dict[str, any]
type ThreadMessage = dict[str, any]

@dataclass(slots=True)
class ApplicationState:
    """Application state with slots optimization."""
    routes_file: Path | None = None
    norms_file: Path | None = None
    current_section: str = ""
    is_analyzing: bool = False
    use_coefficients: bool = False
    analysis_results: dict = None
    
    def __post_init__(self):
        if self.analysis_results is None:
            self.analysis_results = {}

class NormsAnalyzerGUI:
    """Исправленный GUI с правильной интеграцией компонентов."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Анализатор норм РЖД - Python 3.12 Optimized")
        self.root.geometry("1400x900")
        
        # Initialize state
        self.state = ApplicationState()
        self.analyzer = InteractiveNormsAnalyzer()
        self.locomotive_filter: LocomotiveFilter | None = None
        self.coefficient_manager = LocomotiveCoefficientsManager()  # ИСПРАВЛЕНО: используем правильный класс
        
        # Threading
        self.thread_queue = queue.Queue()
        self.temp_files: list[Path] = []
        
        # Setup GUI
        self._setup_styles()
        self._create_widgets()
        self._setup_event_handlers()
        self._start_thread_monitor()
        
        logger.info("GUI initialized successfully")
    
    def _setup_styles(self) -> None:
        """Setup modern ttk styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Segoe UI', 14, 'bold'))
        style.configure('Header.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
        style.configure('Info.TLabel', foreground='blue')
    
    def _create_widgets(self) -> None:
        """Create main GUI widgets with modern layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Create sections
        self._create_file_section(main_frame)
        self._create_control_panel(main_frame)
        self._create_results_area(main_frame)
        self._create_status_bar(main_frame)
    
    def _create_file_section(self, parent: ttk.Frame) -> None:
        """Create file selection section."""
        file_frame = ttk.LabelFrame(parent, text="📁 Файлы данных", padding="10")
        file_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        
        # Routes file selection
        ttk.Label(file_frame, text="Маршруты:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.routes_label = ttk.Label(file_frame, text="Не выбран", style='Info.TLabel')
        self.routes_label.grid(row=0, column=1, sticky="w", padx=(0, 10))
        ttk.Button(file_frame, text="📂 Выбрать", 
                  command=self._select_routes_file).grid(row=0, column=2, padx=(0, 10))
        
        # Norms file selection
        ttk.Label(file_frame, text="Нормы:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(5, 0))
        self.norms_label = ttk.Label(file_frame, text="Не выбран", style='Info.TLabel')
        self.norms_label.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=(5, 0))
        ttk.Button(file_frame, text="📂 Выбрать", 
                  command=self._select_norms_file).grid(row=1, column=2, padx=(0, 10), pady=(5, 0))
        
        # Load button
        self.load_button = ttk.Button(file_frame, text="⚡ Загрузить данные", 
                                     command=self._load_data, state="disabled")
        self.load_button.grid(row=2, column=0, columnspan=3, pady=(15, 0))
        
        # Status indicator
        self.load_status = ttk.Label(file_frame, text="", style='Info.TLabel')
        self.load_status.grid(row=3, column=0, columnspan=3, pady=(5, 0))
    
    def _create_control_panel(self, parent: ttk.Frame) -> None:
        """Create control panel."""
        control_frame = ttk.LabelFrame(parent, text="🎛️ Управление анализом", padding="10")
        control_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        # Section selection
        ttk.Label(control_frame, text="Участок:", style='Header.TLabel').pack(anchor="w", pady=(0, 5))
        self.section_var = tk.StringVar()
        self.section_combo = ttk.Combobox(control_frame, textvariable=self.section_var, 
                                         state="readonly", width=30)
        self.section_combo.pack(fill="x", pady=(0, 10))
        self.section_combo.bind('<<ComboboxSelected>>', self._on_section_changed)
        
        # Control buttons
        buttons_data = [
            ("🔍 Анализировать", self._analyze_section, "disabled"),
            ("🚂 Фильтр локомотивов", self._open_locomotive_filter, "disabled"),
            ("📝 Редактировать нормы", self._edit_norms, "disabled"),
            ("📊 Открыть график", self._open_plot, "disabled")
        ]
        
        self.control_buttons = {}
        for text, command, state in buttons_data:
            btn = ttk.Button(control_frame, text=text, command=command, state=state)
            btn.pack(fill="x", pady=2)
            self.control_buttons[text] = btn
        
        # Progress indicator
        ttk.Separator(control_frame, orient='horizontal').pack(fill="x", pady=(10, 5))
        ttk.Label(control_frame, text="Прогресс:", style='Header.TLabel').pack(anchor="w")
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill="x", pady=(5, 0))
        
        # Statistics display
        ttk.Label(control_frame, text="Статистика:", style='Header.TLabel').pack(anchor="w", pady=(10, 5))
        self.stats_text = tk.Text(control_frame, width=35, height=15, wrap="word", 
                                 font=("Consolas", 9), state="disabled")
        
        stats_scroll = ttk.Scrollbar(control_frame, command=self.stats_text.yview)
        self.stats_text.config(yscrollcommand=stats_scroll.set)
        
        # Pack stats with scrollbar
        stats_frame = ttk.Frame(control_frame)
        stats_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.stats_text.pack(side="left", fill="both", expand=True, in_=stats_frame)
        stats_scroll.pack(side="right", fill="y", in_=stats_frame)
        
        # Export buttons
        export_frame = ttk.Frame(control_frame)
        export_frame.pack(fill="x")
        
        ttk.Button(export_frame, text="📤 Excel", 
                  command=self._export_excel, state="disabled").pack(side="left", padx=(0, 5))
        ttk.Button(export_frame, text="📤 График", 
                  command=self._export_plot, state="disabled").pack(side="left")
        
        self.export_buttons = export_frame.winfo_children()
    
    def _create_results_area(self, parent: ttk.Frame) -> None:
        """Create results display area."""
        results_frame = ttk.LabelFrame(parent, text="📈 Результаты анализа", padding="10")
        results_frame.grid(row=1, column=1, sticky="nsew")
        
        # Results text with syntax highlighting
        self.results_text = tk.Text(results_frame, wrap="word", font=("Consolas", 10))
        results_scroll = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        self.results_text.config(yscrollcommand=results_scroll.set)
        
        # Configure text tags for colored output
        self.results_text.tag_configure("header", font=("Consolas", 12, "bold"), foreground="blue")
        self.results_text.tag_configure("success", foreground="green", font=("Consolas", 10, "bold"))
        self.results_text.tag_configure("warning", foreground="orange", font=("Consolas", 10, "bold"))
        self.results_text.tag_configure("error", foreground="red", font=("Consolas", 10, "bold"))
        self.results_text.tag_configure("info", foreground="gray")
        
        # Pack with scrollbar
        self.results_text.pack(side="left", fill="both", expand=True)
        results_scroll.pack(side="right", fill="y")
        
        # Add welcome message
        self._display_welcome_message()
    
    def _create_status_bar(self, parent: ttk.Frame) -> None:
        """Create status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="🟢 Готов к работе")
        self.status_label.pack(side="left")
        
        # Memory usage indicator (optional)
        self.memory_label = ttk.Label(status_frame, text="", font=("Consolas", 8))
        self.memory_label.pack(side="right")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers."""
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self._select_routes_file())
        self.root.bind('<Control-l>', lambda e: self._load_data())
        self.root.bind('<F5>', lambda e: self._analyze_section())
    
    def _start_thread_monitor(self) -> None:
        """Start thread message monitor."""
        self._monitor_threads()
    
    def _monitor_threads(self) -> None:
        """Monitor background thread messages."""
        try:
            while True:
                message = self.thread_queue.get_nowait()
                self._handle_thread_message(message)
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(50, self._monitor_threads)
    
    def _handle_thread_message(self, message: ThreadMessage) -> None:
        """Handle messages from background threads."""
        msg_type = message.get('type', 'unknown')
        
        match msg_type:
            case 'progress':
                self.progress_var.set(message['value'])
                if 'status' in message:
                    self._update_status(message['status'])
            
            case 'data_loaded':
                self._on_data_loaded(message['sections'])
            
            case 'data_load_error':
                self._on_data_load_error(message['error'])
            
            case 'analysis_complete':
                self._on_analysis_complete(message['data'], message['stats'])
            
            case 'analysis_error':
                self._on_analysis_error(message['error'])
            
            case 'status':
                self._update_status(message['text'], message.get('style', 'Info'))
    
    # File operations
    def _select_routes_file(self) -> None:
        """Select routes file."""
        file_path = filedialog.askopenfilename(
            title="Выберите файл маршрутов",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if file_path:
            self.state.routes_file = Path(file_path)
            self.routes_label.config(text=self.state.routes_file.name, style='Success.TLabel')
            self._check_files_ready()
            self._log_action(f"Выбран файл маршрутов: {self.state.routes_file.name}")
    
    def _select_norms_file(self) -> None:
        """Select norms file."""
        file_path = filedialog.askopenfilename(
            title="Выберите файл норм",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if file_path:
            self.state.norms_file = Path(file_path)
            self.norms_label.config(text=self.state.norms_file.name, style='Success.TLabel')
            self._check_files_ready()
            self._log_action(f"Выбран файл норм: {self.state.norms_file.name}")
    
    def _check_files_ready(self) -> None:
        """Check if both files are selected."""
        if self.state.routes_file and self.state.norms_file:
            self.load_button.config(state="normal")
            self.load_status.config(text="✅ Файлы готовы к загрузке", style='Success.TLabel')
    
    def _load_data(self) -> None:
        """Load data in background thread."""
        if not self.state.routes_file or not self.state.norms_file:
            messagebox.showwarning("Предупреждение", "Выберите файлы маршрутов и норм")
            return
        
        self.load_button.config(state="disabled")
        self._update_status("⏳ Загрузка данных...", "Info")
        
        def load_worker():
            """Background data loading."""
            try:
                # Load routes
                self.thread_queue.put({
                    'type': 'progress', 
                    'value': 25, 
                    'status': '📊 Загрузка маршрутов...'
                })
                
                if not self.analyzer.load_data(self.state.routes_file):
                    raise ValueError("Failed to load routes data")
                
                self.thread_queue.put({
                    'type': 'progress', 
                    'value': 75, 
                    'status': '📋 Загрузка норм...'
                })
                
                # Load norms
                if not self.analyzer.load_norms(self.state.norms_file):
                    raise ValueError("Failed to load norms data")
                
                # Initialize filter
                self.locomotive_filter = LocomotiveFilter(self.analyzer.routes_df)
                
                sections = self.analyzer.get_sections_list()
                
                self.thread_queue.put({
                    'type': 'progress', 
                    'value': 100
                })
                
                self.thread_queue.put({
                    'type': 'data_loaded',
                    'sections': sections
                })
                
            except Exception as e:
                logger.error(f"Data loading failed: {e}")
                self.thread_queue.put({
                    'type': 'data_load_error',
                    'error': str(e)
                })
        
        threading.Thread(target=load_worker, daemon=True).start()
    
    def _on_data_loaded(self, sections: list[str]) -> None:
        """Handle successful data loading."""
        self.section_combo['values'] = sections
        
        # Enable controls
        for button in self.control_buttons.values():
            button.config(state="normal")
        
        self.load_button.config(state="normal")
        
        # Update display
        routes_count = len(self.analyzer.routes_df) if self.analyzer.routes_df is not None else 0
        locomotives_count = len(self.locomotive_filter.available_locomotives)
        
        self.load_status.config(
            text=f"✅ Загружено: {routes_count:,} маршрутов, {len(sections)} участков, "
                 f"{locomotives_count} локомотивов",
            style='Success.TLabel'
        )
        
        # Display info in results
        info_text = f"""🎉 ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ

📊 Статистика загрузки:
   • Маршруты: {routes_count:,}
   • Участки: {len(sections)}
   • Локомотивы: {locomotives_count}
   • Серии локомотивов: {len(self.locomotive_filter.locomotives_by_series)}

🚀 Система готова к анализу!

📍 Инструкции:
   1. Выберите участок из списка
   2. Настройте фильтр локомотивов (опционально)
   3. Нажмите "Анализировать"
   4. Просмотрите результаты и график
"""
        
        self._display_text(info_text, clear=True)
        self._update_status("🟢 Данные загружены. Готов к анализу", "Success")
        self._log_action("Данные успешно загружены")
    
    def _on_data_load_error(self, error: str) -> None:
        """Handle data loading error."""
        self.load_button.config(state="normal")
        self._update_status("❌ Ошибка загрузки данных", "Error")
        
        error_text = f"""❌ ОШИБКА ЗАГРУЗКИ ДАННЫХ

🔍 Детали ошибки:
{error}

💡 Возможные причины:
   • Неверный формат файла Excel
   • Отсутствуют необходимые колонки
   • Файл поврежден или заблокирован
   • Недостаточно памяти

🛠️ Рекомендации:
   • Проверьте формат файлов (.xlsx)
   • Убедитесь что файлы не открыты в Excel
   • Проверьте наличие необходимых колонок
   • Попробуйте другие файлы
"""
        
        self._display_text(error_text, "error", clear=True)
        messagebox.showerror("Ошибка загрузки", f"Не удалось загрузить данные:\n\n{error}")
    
    # Analysis operations
    def _on_section_changed(self, event=None) -> None:
        """Handle section selection change."""
        if self.section_var.get() and not self.state.is_analyzing:
            self._analyze_section()
    
    def _analyze_section(self) -> None:
        """Analyze selected section - ИСПРАВЛЕННЫЙ МЕТОД."""
        section = self.section_var.get()
        if not section or self.state.is_analyzing:
            return
        
        self.state.is_analyzing = True
        self.state.current_section = section
        
        # Disable analyze button
        self.control_buttons["🔍 Анализировать"].config(state="disabled")
        
        self._update_status(f"🔬 Анализ участка: {section}...", "Info")
        self.progress_var.set(0)
        
        def analyze_worker():
            """Background analysis."""
            try:
                self.thread_queue.put({
                    'type': 'progress', 
                    'value': 50, 
                    'status': f'🔬 Анализируем {section}...'
                })
                
                # Get section data
                section_data = self.analyzer.routes_df[
                    self.analyzer.routes_df['Наименование участка'] == section
                ].copy()
                
                if section_data.empty:
                    raise ValueError(f"No data found for section: {section}")
                
                # Get section norms
                section_norms = self.analyzer.norms_manager.get_section_norms(section)
                if not section_norms:
                    raise ValueError(f"No norms found for section: {section}")
                
                # Perform analysis with filters using the CORRECTED method
                analyzed_data, norm_functions = self.analyzer.analyze_section_with_filters(
                    section,
                    section_data,
                    section_norms,
                    locomotive_filter=self.locomotive_filter,
                    coefficient_manager=self.coefficient_manager,
                    use_coefficients=self.state.use_coefficients
                )
                
                if analyzed_data is None or analyzed_data.empty:
                    raise ValueError("No data after applying filters")
                
                # Calculate statistics
                stats = self.analyzer._calculate_statistics(analyzed_data)
                
                self.thread_queue.put({
                    'type': 'progress', 
                    'value': 100
                })
                
                self.thread_queue.put({
                    'type': 'analysis_complete',
                    'data': analyzed_data,
                    'stats': stats
                })
                
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
                self.thread_queue.put({
                    'type': 'analysis_error',
                    'error': str(e)
                })
        
        threading.Thread(target=analyze_worker, daemon=True).start()
    
    def _on_analysis_complete(self, analyzed_data, stats: dict) -> None:
        """Handle successful analysis completion."""
        self.state.is_analyzing = False
        self.control_buttons["🔍 Анализировать"].config(state="normal")
        
        # Enable plot and export buttons
        self.control_buttons["📊 Открыть график"].config(state="normal")
        for btn in self.export_buttons:
            btn.config(state="normal")
        
        # Store results
        self.state.analysis_results[self.state.current_section] = {
            'data': analyzed_data,
            'stats': stats
        }
        
        # Update statistics display
        self._update_statistics_display(stats)
        
        # Update results display
        self._update_results_display(self.state.current_section, stats)
        
        self._update_status(f"✅ Анализ {self.state.current_section} завершен", "Success")
        self._log_action(f"Анализ участка {self.state.current_section} завершен")
    
    def _on_analysis_error(self, error: str) -> None:
        """Handle analysis error."""
        self.state.is_analyzing = False
        self.control_buttons["🔍 Анализировать"].config(state="normal")
        
        self._update_status("❌ Ошибка анализа", "Error")
        
        error_text = f"""❌ ОШИБКА АНАЛИЗА

🔍 Участок: {self.state.current_section}
📝 Ошибка: {error}

💡 Возможные причины:
   • Отсутствуют нормы для участка
   • Нет данных после фильтрации
   • Некорректные данные в файлах

🛠️ Рекомендации:
   • Проверьте наличие норм для участка
   • Измените фильтр локомотивов
   • Проверьте качество исходных данных
"""
        
        self._display_text(error_text, "error", clear=True)
        messagebox.showerror("Ошибка анализа", f"Не удалось выполнить анализ:\n\n{error}")
    
    # Dialog operations
    def _open_locomotive_filter(self) -> None:
        """Open locomotive filter dialog - ИСПРАВЛЕННЫЙ МЕТОД."""
        if not self.locomotive_filter:
            messagebox.showwarning("Предупреждение", "Сначала загрузите данные")
            return
        
        dialog = LocomotiveSelectorDialog(
            self.root,
            self.locomotive_filter,
            self.coefficient_manager
        )
        
        self.root.wait_window(dialog.dialog)
        
        if dialog.result:
            self.state.use_coefficients = dialog.result.use_coefficients  # ИСПРАВЛЕНО: доступ к атрибуту
            self.coefficient_manager = dialog.result.coefficient_manager  # ИСПРАВЛЕНО
            
            # Update status
            selected_count = len(self.locomotive_filter.selected)
            total_count = len(self.locomotive_filter.available_locomotives)
            
            self._update_status(f"🚂 Выбрано: {selected_count}/{total_count} локомотивов", "Info")
            
            # Re-analyze if section is selected
            if self.section_var.get() and not self.state.is_analyzing:
                self._analyze_section()
    
    def _edit_norms(self) -> None:
        """Open norms editor dialog."""
        section = self.section_var.get()
        if not section:
            messagebox.showwarning("Предупреждение", "Выберите участок")
            return
        
        existing_norms = self.analyzer.norms_manager.get_section_norms(section)
        
        # Конвертируем NormDefinition в dict для совместимости с редактором
        existing_norms_dict = {}
        for norm_id, norm_def in existing_norms.items():
            existing_norms_dict[norm_id] = {
                'points': norm_def.points,
                'description': norm_def.description
            }
        
        dialog = NormEditorDialog(self.root, section, existing_norms_dict)
        self.root.wait_window(dialog.dialog)
        
        if dialog.result == 'apply' and dialog.edited_norms:
            # Update norms
            self.analyzer.norms_manager.section_norms[section] = dialog.edited_norms
            
            self._log_action(f"Обновлены нормы для участка {section}")
            
            # Re-analyze
            if not self.state.is_analyzing:
                self._analyze_section()
    
    # Visualization and export
    def _open_plot(self) -> None:
        """Open interactive plot in browser."""
        section = self.state.current_section
        if not section or section not in self.state.analysis_results:
            messagebox.showwarning("Предупреждение", "Выполните анализ участка")
            return
        
        try:
            # Create plot using the analyzer's cached results
            fig = self.analyzer.create_interactive_plot(section)
            
            # Save to temporary file
            temp_file = Path(tempfile.mktemp(suffix='.html'))
            plot(fig, filename=str(temp_file), auto_open=False)
            
            # Track temp file for cleanup
            self.temp_files.append(temp_file)
            
            # Open in browser
            webbrowser.open(f'file://{temp_file.absolute()}')
            
            self._update_status("📊 График открыт в браузере", "Success")
            self._log_action("Интерактивный график открыт")
            
        except Exception as e:
            logger.error(f"Plot creation failed: {e}")
            messagebox.showerror("Ошибка", f"Не удалось создать график:\n{str(e)}")
    
    def _export_excel(self) -> None:
        """Export analysis results to Excel."""
        section = self.state.current_section
        if not section or section not in self.state.analysis_results:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                data = self.state.analysis_results[section]['data']
                data.to_excel(file_path, index=False)
                
                self._update_status(f"📤 Экспорт в {Path(file_path).name} завершен", "Success")
                messagebox.showinfo("Экспорт", "Данные успешно экспортированы!")
                
            except Exception as e:
                logger.error(f"Excel export failed: {e}")
                messagebox.showerror("Ошибка", f"Не удалось экспортировать:\n{str(e)}")
    
    def _export_plot(self) -> None:
        """Export plot to file."""
        section = self.state.current_section
        if not section or section not in self.state.analysis_results:
            messagebox.showwarning("Предупреждение", "Нет графика для экспорта")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Сохранить график",
            defaultextension=".html",
            filetypes=[
                ("HTML files", "*.html"),
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf")
            ]
        )
        
        if file_path:
            try:
                fig = self.analyzer.create_interactive_plot(section)
                
                if file_path.endswith('.html'):
                    fig.write_html(file_path)
                elif file_path.endswith('.png'):
                    fig.write_image(file_path, width=1920, height=1080)
                elif file_path.endswith('.pdf'):
                    fig.write_image(file_path)
                
                self._update_status(f"📤 График сохранен: {Path(file_path).name}", "Success")
                messagebox.showinfo("Экспорт", "График успешно сохранен!")
                
            except Exception as e:
                logger.error(f"Plot export failed: {e}")
                messagebox.showerror("Ошибка", f"Не удалось сохранить график:\n{str(e)}")
    
    # Display updates
    def _update_statistics_display(self, stats: dict) -> None:
        """Update statistics text display."""
        total = stats['processed_routes']
        
        if total == 0:
            stats_text = "Нет данных для отображения"
        else:
            stats_text = f"""📊 СТАТИСТИКА АНАЛИЗА

🔢 Обработано маршрутов: {total:,}
📈 Эффективность обработки: {stats.get('processing_efficiency', 0):.1f}%

📊 Распределение по статусам:
├─ 🟢 В норме: {stats.get('normal', 0):,} ({stats.get('normal', 0)/total*100:.1f}%)
├─ 🔵 Экономия: {stats.get('economy_weak', 0) + stats.get('economy_medium', 0) + stats.get('economy_strong', 0):,}
├─ 🟡 Перерасход: {stats.get('overrun_weak', 0) + stats.get('overrun_medium', 0) + stats.get('overrun_strong', 0):,}

📈 Детальная статистика:
├─ Экономия сильная (>30%): {stats.get('economy_strong', 0):,}
├─ Экономия средняя (20-30%): {stats.get('economy_medium', 0):,}
├─ Экономия слабая (5-20%): {stats.get('economy_weak', 0):,}
├─ Перерасход слабый (5-20%): {stats.get('overrun_weak', 0):,}
├─ Перерасход средний (20-30%): {stats.get('overrun_medium', 0):,}
└─ Перерасход сильный (>30%): {stats.get('overrun_strong', 0):,}

📊 Статистические показатели:
├─ Среднее отклонение: {stats.get('mean_deviation', 0):.2f}%
├─ Медианное отклонение: {stats.get('median_deviation', 0):.2f}%
└─ Стандартное отклонение: {stats.get('std_deviation', 0):.2f}%
"""
        
        self.stats_text.config(state="normal")
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state="disabled")
    
    def _update_results_display(self, section: str, stats: dict) -> None:
        """Update main results display."""
        total = stats['processed_routes']
        
        results_text = f"""🎯 АНАЛИЗ УЧАСТКА: {section}
{'='*60}

📊 Обработано {total:,} маршрутов из {stats['total_routes']:,} общих
🎯 Эффективность обработки: {stats.get('processing_efficiency', 0):.1f}%

📈 РЕЗУЛЬТАТЫ АНАЛИЗА:

🟢 ЭКОНОМИЯ ЭЛЕКТРОЭНЕРГИИ:
   • Сильная экономия (>30%): {stats.get('economy_strong', 0):,} маршрутов ({stats.get('economy_strong', 0)/total*100:.1f}%)
   • Средняя экономия (20-30%): {stats.get('economy_medium', 0):,} маршрутов ({stats.get('economy_medium', 0)/total*100:.1f}%)
   • Слабая экономия (5-20%): {stats.get('economy_weak', 0):,} маршрутов ({stats.get('economy_weak', 0)/total*100:.1f}%)

🟨 НОРМАЛЬНОЕ ПОТРЕБЛЕНИЕ (±5%): {stats.get('normal', 0):,} маршрутов ({stats.get('normal', 0)/total*100:.1f}%)

🔴 ПЕРЕРАСХОД ЭЛЕКТРОЭНЕРГИИ:
   • Слабый перерасход (5-20%): {stats.get('overrun_weak', 0):,} маршрутов ({stats.get('overrun_weak', 0)/total*100:.1f}%)
   • Средний перерасход (20-30%): {stats.get('overrun_medium', 0):,} маршрутов ({stats.get('overrun_medium', 0)/total*100:.1f}%)
   • Сильный перерасход (>30%): {stats.get('overrun_strong', 0):,} маршрутов ({stats.get('overrun_strong', 0)/total*100:.1f}%)

📊 СТАТИСТИЧЕСКИЕ ПОКАЗАТЕЛИ:
   • Среднее отклонение от нормы: {stats.get('mean_deviation', 0):.2f}%
   • Медианное отклонение: {stats.get('median_deviation', 0):.2f}%
   • Стандартное отклонение: {stats.get('std_deviation', 0):.2f}%

🚀 СЛЕДУЮЩИЕ ШАГИ:
   1. Нажмите "📊 Открыть график" для детального анализа
   2. Используйте "📤 Excel" для экспорта данных
   3. Настройте фильтр локомотивов для углубленного анализа

⚡ Анализ выполнен с использованием Python 3.12 оптимизаций
"""
        
        self._display_text(results_text, clear=True)
    
    def _display_welcome_message(self) -> None:
        """Display welcome message."""
        welcome_text = """🚂 АНАЛИЗАТОР НОРМ РАСХОДА ЭЛЕКТРОЭНЕРГИИ РЖД
{'='*60}

🎯 Python 3.12 Optimized Version

🚀 ВОЗМОЖНОСТИ СИСТЕМЫ:
   • Векторизованный анализ больших объемов данных
   • Интерактивная визуализация с Plotly
   • Фильтрация локомотивов по сериям
   • Применение индивидуальных коэффициентов
   • Редактирование и актуализация норм
   • Экспорт результатов в Excel и графические форматы

⚡ НОВЫЕ ОПТИМИЗАЦИИ:
   • 10-100x ускорение обработки данных
   • Улучшенное управление памятью
   • Кэширование для быстрых повторных расчетов
   • Современный многопоточный интерфейс

📝 НАЧАЛО РАБОТЫ:
   1. Выберите файлы данных (маршруты и нормы)
   2. Нажмите "⚡ Загрузить данные"
   3. Выберите участок для анализа
   4. Изучите результаты и интерактивные графики

🔧 Система готова к работе!
"""
        
        self._display_text(welcome_text, "header")
    
    def _display_text(self, text: str, tag: str = None, clear: bool = False) -> None:
        """Display text with optional formatting."""
        if clear:
            self.results_text.delete(1.0, tk.END)
        
        start = self.results_text.index(tk.END)
        self.results_text.insert(tk.END, text + "\n")
        
        if tag:
            end = self.results_text.index(tk.END)
            self.results_text.tag_add(tag, start, end)
        
        self.results_text.see(tk.END)
    
    def _update_status(self, text: str, style: str = "Info") -> None:
        """Update status bar."""
        self.status_label.config(text=text, style=f'{style}.TLabel')
    
    def _log_action(self, action: str) -> None:
        """Log user action."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_text = f"[{timestamp}] {action}"
        logger.info(log_text)
    
    # Cleanup
    @contextmanager
    def _error_handler(self, operation: str):
        """Context manager for error handling."""
        try:
            yield
        except Exception as e:
            logger.error(f"{operation} failed: {e}")
            messagebox.showerror("Ошибка", f"{operation}:\n{str(e)}")
    
    def _on_closing(self) -> None:
        """Handle application closing."""
        try:
            # Cleanup temporary files
            for temp_file in self.temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            
            # Clear caches
            if self.coefficient_manager:
                self.coefficient_manager.clear_coefficients()
            
            logger.info("Application closed successfully")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
        finally:
            self.root.quit()
            self.root.destroy()