# dialogs/selector.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный диалог выбора локомотивов с правильной интеграцией
Объединяет рабочую функциональность старого кода с новыми оптимизациями
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import logging

logger = logging.getLogger(__name__)

# Python 3.12 type definitions
type LocomotiveID = tuple[str, int]
type SelectionState = dict[LocomotiveID, bool]

@dataclass(slots=True)
class DialogResult:
    """Dialog result with slots optimization."""
    use_coefficients: bool = False
    exclude_low_work: bool = False
    coefficient_manager: object = None
    selected_locomotives: list[LocomotiveID] = field(default_factory=list)

class SelectionManager(Protocol):
    """Protocol for selection management strategies."""
    def update_selection(self, locomotive_id: LocomotiveID, selected: bool) -> None: ...
    def get_selection_count(self) -> tuple[int, int]: ...

class LocomotiveSelectorDialog:
    """Исправленный диалог выбора локомотивов с правильной интеграцией."""
    
    def __init__(self, parent: tk.Tk, locomotive_filter, coefficient_manager=None):
        self.parent = parent
        self.filter = locomotive_filter
        self.coefficient_manager = coefficient_manager
        self.result: DialogResult | None = None
        
        # State management
        self.selection_vars: dict[LocomotiveID, tk.BooleanVar] = {}
        self.series_vars: dict[str, tk.BooleanVar] = {}
        self.use_coefficients = tk.BooleanVar()
        self.exclude_low_work = tk.BooleanVar()
        
        # Create dialog
        self._create_dialog()
        self._populate_locomotives()
        self._load_current_selection()
        
        logger.info("Locomotive selector dialog initialized")
    
    def _create_dialog(self) -> None:
        """Create modern dialog interface."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("🚂 Выбор локомотивов и коэффициентов")
        self.dialog.geometry("900x700")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self._center_dialog()
        
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create sections
        self._create_coefficients_section(main_frame)
        self._create_selection_section(main_frame)
        self._create_button_section(main_frame)
        
        # Configure grid weights
        main_frame.rowconfigure(1, weight=1)
        main_frame.columnconfigure(0, weight=1)
    
    def _center_dialog(self) -> None:
        """Center dialog on parent window."""
        self.dialog.update_idletasks()
        
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.dialog.winfo_width()
        dialog_height = self.dialog.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.dialog.geometry(f"+{x}+{y}")
    
    def _create_coefficients_section(self, parent: ttk.Frame) -> None:
        """Create coefficients management section."""
        coeff_frame = ttk.LabelFrame(parent, text="⚙️ Коэффициенты расхода локомотивов", padding="10")
        coeff_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        
        # File management row
        file_row = ttk.Frame(coeff_frame)
        file_row.pack(fill="x", pady=(0, 10))
        
        ttk.Label(file_row, text="Файл коэффициентов:").pack(side="left", padx=(0, 10))
        
        self.coeff_file_label = ttk.Label(file_row, text="Не загружен", 
                                         foreground="gray", font=("Consolas", 9))
        self.coeff_file_label.pack(side="left", padx=(0, 15))
        
        ttk.Button(file_row, text="📂 Загрузить", 
                  command=self._load_coefficients).pack(side="left", padx=(0, 5))
        ttk.Button(file_row, text="🗑️ Очистить", 
                  command=self._clear_coefficients).pack(side="left")
        
        # Statistics row
        self.coeff_stats_label = ttk.Label(coeff_frame, text="", 
                                          foreground="blue", font=("Consolas", 9))
        self.coeff_stats_label.pack(fill="x", pady=(0, 10))
        
        # Options row
        options_row = ttk.Frame(coeff_frame)
        options_row.pack(fill="x")
        
        self.use_coeff_check = ttk.Checkbutton(
            options_row,
            text="✅ Применять коэффициенты при анализе",
            variable=self.use_coefficients,
            command=self._on_use_coefficients_changed
        )
        self.use_coeff_check.pack(side="left", padx=(0, 20))
        
        self.exclude_low_work_check = ttk.Checkbutton(
            options_row,
            text="⚡ Исключить локомотивы с работой < 200 10тыс.ткм",
            variable=self.exclude_low_work
        )
        self.exclude_low_work_check.pack(side="left")
        
        # Update display if coefficient manager already loaded
        if self.coefficient_manager and hasattr(self.coefficient_manager, 'coefficients'):
            if self.coefficient_manager.coefficients:
                self._update_coefficients_display()
    
    def _create_selection_section(self, parent: ttk.Frame) -> None:
        """Create locomotive selection section."""
        selection_frame = ttk.LabelFrame(parent, text="🚂 Выбор локомотивов для анализа", padding="10")
        selection_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        
        # Control buttons row
        controls_row = ttk.Frame(selection_frame)
        controls_row.pack(fill="x", pady=(0, 10))
        
        control_buttons = [
            ("✅ Выбрать все", self._select_all),
            ("❌ Снять все", self._deselect_all),
            ("🔄 Инвертировать", self._invert_selection)
        ]
        
        for text, command in control_buttons:
            ttk.Button(controls_row, text=text, command=command).pack(side="left", padx=(0, 10))
        
        # Selection counter
        self.selection_label = ttk.Label(controls_row, text="", 
                                        foreground="green", font=("Arial", 10, "bold"))
        self.selection_label.pack(side="left", padx=(20, 0))
        
        # Locomotive notebook
        self.locomotive_notebook = ttk.Notebook(selection_frame)
        self.locomotive_notebook.pack(fill="both", expand=True)
    
    def _create_button_section(self, parent: ttk.Frame) -> None:
        """Create dialog buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=2, column=0, sticky="ew")
        
        # Info label
        info_text = ("💡 Выберите локомотивы для включения в анализ. "
                    "Коэффициенты позволяют учесть индивидуальные характеристики.")
        info_label = ttk.Label(button_frame, text=info_text, 
                              foreground="gray", font=("Arial", 9))
        info_label.pack(pady=(0, 10))
        
        # Buttons
        buttons_row = ttk.Frame(button_frame)
        buttons_row.pack()
        
        ttk.Button(buttons_row, text="✅ Применить", 
                  command=self._apply_selection).pack(side="left", padx=(0, 10))
        ttk.Button(buttons_row, text="❌ Отмена", 
                  command=self._cancel).pack(side="left")
    
    def _populate_locomotives(self) -> None:
        """Populate locomotive selection interface efficiently."""
        if hasattr(self.filter, 'get_locomotives_by_series'):
            locomotives_by_series = self.filter.get_locomotives_by_series()
        else:
            # Старый формат
            locomotives_by_series = getattr(self.filter, 'locomotives_by_series', {})
        
        for series in sorted(locomotives_by_series.keys()):
            self._create_series_tab(series, locomotives_by_series[series])
        
        self._update_selection_counter()
    
    def _create_series_tab(self, series: str, locomotive_numbers: list[int]) -> None:
        """Create tab for locomotive series with optimized layout."""
        # Create tab frame
        tab_frame = ttk.Frame(self.locomotive_notebook)
        self.locomotive_notebook.add(tab_frame, text=f"{series} ({len(locomotive_numbers)})")
        
        # Series control frame
        series_control = ttk.Frame(tab_frame)
        series_control.pack(fill="x", padx=10, pady=(10, 5))
        
        # Series checkbox
        series_var = tk.BooleanVar(value=True)
        self.series_vars[series] = series_var
        
        series_check = ttk.Checkbutton(
            series_control,
            text=f"📋 Выбрать всю серию {series} ({len(locomotive_numbers)} локомотивов)",
            variable=series_var,
            command=lambda s=series: self._toggle_series(s)
        )
        series_check.pack(side="left")
        
        # Series statistics
        self._add_series_statistics(series_control, series, locomotive_numbers)
        
        # Scrollable locomotive list
        self._create_scrollable_locomotive_list(tab_frame, series, locomotive_numbers)
    
    def _add_series_statistics(self, parent: ttk.Frame, series: str, numbers: list[int]) -> None:
        """Add series statistics display."""
        stats_frame = ttk.Frame(parent)
        stats_frame.pack(side="right")
        
        # Calculate statistics
        total_count = len(numbers)
        min_num = min(numbers) if numbers else 0
        max_num = max(numbers) if numbers else 0
        
        # Count locomotives with coefficients
        coeff_count = 0
        if self.coefficient_manager:
            for num in numbers:
                coeff = self.coefficient_manager.get_coefficient(series, num)
                if coeff != 1.0:
                    coeff_count += 1
        
        stats_text = f"📊 №{min_num}-{max_num} | ⚙️ {coeff_count} с коэффициентами"
        ttk.Label(stats_frame, text=stats_text, 
                 foreground="blue", font=("Consolas", 8)).pack()
    
    def _create_scrollable_locomotive_list(self, parent: ttk.Frame, 
                                          series: str, numbers: list[int]) -> None:
        """Create scrollable list of locomotives with efficient rendering."""
        # Create scrollable container
        canvas = tk.Canvas(parent, highlightthickness=0, height=400)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Configure grid in scrollable frame
        scrollable_frame.columnconfigure(1, weight=1)
        
        # Headers
        headers = ["№", "Выбор", "Номер локомотива", "Коэффициент", "Эффективность"]
        for i, header in enumerate(headers):
            ttk.Label(scrollable_frame, text=header, 
                     font=("Arial", 9, "bold")).grid(row=0, column=i, padx=5, pady=5, sticky="w")
        
        # Add separator
        ttk.Separator(scrollable_frame, orient='horizontal').grid(
            row=1, column=0, columnspan=len(headers), sticky='ew', pady=2
        )
        
        # Add locomotives efficiently using grid
        for idx, number in enumerate(numbers):
            row = idx + 2  # Account for header and separator
            locomotive_id = (series, number)
            
            # Row number
            ttk.Label(scrollable_frame, text=f"{idx + 1}").grid(
                row=row, column=0, padx=5, pady=1, sticky="w"
            )
            
            # Selection checkbox
            var = tk.BooleanVar(value=True)
            self.selection_vars[locomotive_id] = var
            
            check = ttk.Checkbutton(
                scrollable_frame, 
                variable=var,
                command=self._update_selection_counter
            )
            check.grid(row=row, column=1, padx=5, pady=1)
            
            # Locomotive number (formatted)
            display_number = f"№{number:04d}" if number < 1000 else f"№{number}"
            ttk.Label(scrollable_frame, text=display_number).grid(
                row=row, column=2, padx=5, pady=1, sticky="w"
            )
            
            # Coefficient info
            self._add_coefficient_info(scrollable_frame, row, series, number)
        
        # Pack scrollable components
        canvas.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        canvas.bind("<MouseWheel>", _on_mousewheel)
    
    def _add_coefficient_info(self, parent: ttk.Frame, row: int, 
                             series: str, number: int) -> None:
        """Add coefficient information to locomotive row."""
        if not self.coefficient_manager:
            ttk.Label(parent, text="-").grid(row=row, column=3, padx=5, pady=1, sticky="w")
            ttk.Label(parent, text="-").grid(row=row, column=4, padx=5, pady=1, sticky="w")
            return
        
        # Get coefficient
        coeff = self.coefficient_manager.get_coefficient(series, number)
        locomotive_info = self.coefficient_manager.get_locomotive_info(series, number)
        
        # Coefficient display
        if coeff != 1.0:
            coeff_text = f"{coeff:.3f}"
            # Color based on coefficient value
            match coeff:
                case c if c > 1.15:
                    color = "red"
                case c if c > 1.05:
                    color = "orange"
                case c if c < 0.85:
                    color = "green"
                case c if c < 0.95:
                    color = "blue"
                case _:
                    color = "black"
        else:
            coeff_text = "1.000"
            color = "gray"
        
        coeff_label = ttk.Label(parent, text=coeff_text, foreground=color)
        coeff_label.grid(row=row, column=3, padx=5, pady=1, sticky="w")
        
        # Efficiency rating
        if locomotive_info:
            efficiency_text = locomotive_info.efficiency_rating.replace('_', ' ').title()
            efficiency_colors = {
                'Excellent': 'green',
                'Good': 'blue',
                'Normal': 'black',
                'Below Average': 'orange',
                'Poor': 'red'
            }
            efficiency_color = efficiency_colors.get(efficiency_text, 'black')
        else:
            efficiency_text = "Normal"
            efficiency_color = "gray"
        
        efficiency_label = ttk.Label(parent, text=efficiency_text, 
                                    foreground=efficiency_color, font=("Arial", 8))
        efficiency_label.grid(row=row, column=4, padx=5, pady=1, sticky="w")
    
    # Event handlers
    def _load_coefficients(self) -> None:
        """Load coefficients from file with progress feedback."""
        file_path = filedialog.askopenfilename(
            title="Выберите файл коэффициентов",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Get minimum work threshold
            min_work = 200 if self.exclude_low_work.get() else 0
            
            path_obj = Path(file_path)
            success = self.coefficient_manager.load_coefficients(path_obj, min_work)
            
            if success:
                self._update_coefficients_display()
                self._refresh_locomotive_display()
                
                stats = self.coefficient_manager.get_statistics()
                if stats:
                    messagebox.showinfo(
                        "Успех", 
                        f"Коэффициенты загружены успешно!\n\n"
                        f"Локомотивов: {stats['total_locomotives']}\n"
                        f"Средний коэффициент: {stats['avg_coefficient']:.3f}"
                    )
                
                logger.info(f"Coefficients loaded: {stats.get('total_locomotives', 0)} locomotives")
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить файл коэффициентов")
                
        except Exception as e:
            logger.error(f"Coefficient loading failed: {e}")
            messagebox.showerror("Ошибка", f"Ошибка при загрузке:\n{str(e)}")
    
    def _clear_coefficients(self) -> None:
        """Clear loaded coefficients."""
        if messagebox.askyesno("Подтверждение", "Очистить загруженные коэффициенты?"):
            self.coefficient_manager.clear_coefficients()
            self._update_coefficients_display()
            self._refresh_locomotive_display()
            self.use_coefficients.set(False)
            
            logger.info("Coefficients cleared")
    
    def _update_coefficients_display(self) -> None:
        """Update coefficients section display."""
        if hasattr(self.coefficient_manager, 'file_path') and self.coefficient_manager.file_path:
            filename = self.coefficient_manager.file_path.name
            self.coeff_file_label.config(text=filename, foreground="black")
            
            stats = self.coefficient_manager.get_statistics()
            if stats:
                stats_text = (
                    f"📊 Локомотивов: {stats['total_locomotives']} | "
                    f"Серий: {stats['series_count']} | "
                    f"Ср. коэфф.: {stats['avg_coefficient']:.3f} | "
                    f"Диапазон: {stats['min_coefficient']:.2f}-{stats['max_coefficient']:.2f}"
                )
                self.coeff_stats_label.config(text=stats_text)
            else:
                self.coeff_stats_label.config(text="")
        elif hasattr(self.coefficient_manager, 'file') and self.coefficient_manager.file:
            # Старый формат
            filename = Path(self.coefficient_manager.file).name
            self.coeff_file_label.config(text=filename, foreground="black")
            
            stats = self.coefficient_manager.get_statistics()
            if stats:
                stats_text = (
                    f"📊 Локомотивов: {stats['total_locomotives']} | "
                    f"Серий: {stats['series_count']} | "
                    f"Ср. коэфф.: {stats['avg_coefficient']:.3f}"
                )
                self.coeff_stats_label.config(text=stats_text)
        else:
            self.coeff_file_label.config(text="Не загружен", foreground="gray")
            self.coeff_stats_label.config(text="")
    
    def _refresh_locomotive_display(self) -> None:
        """Refresh locomotive display after coefficient changes."""
        # Store current selection
        current_selection = {
            loc_id: var.get() for loc_id, var in self.selection_vars.items()
        }
        
        # Clear and recreate notebook
        for tab in self.locomotive_notebook.tabs():
            self.locomotive_notebook.forget(tab)
        
        self.selection_vars.clear()
        self.series_vars.clear()
        
        # Repopulate with updated coefficient info
        self._populate_locomotives()
        
        # Restore selection
        for loc_id, selected in current_selection.items():
            if loc_id in self.selection_vars:
                self.selection_vars[loc_id].set(selected)
        
        self._update_selection_counter()
    
    def _on_use_coefficients_changed(self) -> None:
        """Handle use coefficients checkbox change."""
        if self.use_coefficients.get():
            # Check if coefficients are loaded
            has_coefficients = False
            if hasattr(self.coefficient_manager, 'coefficients'):
                has_coefficients = bool(self.coefficient_manager.coefficients)
            elif hasattr(self.coefficient_manager, 'coef'):
                has_coefficients = bool(self.coefficient_manager.coef)
            elif hasattr(self.coefficient_manager, 'data'):
                has_coefficients = bool(self.coefficient_manager.data)
            
            if not has_coefficients:
                messagebox.showwarning(
                    "Предупреждение", 
                    "Сначала загрузите файл с коэффициентами"
                )
                self.use_coefficients.set(False)
    
    # Selection management
    def _toggle_series(self, series: str) -> None:
        """Toggle selection for entire series."""
        is_selected = self.series_vars[series].get()
        
        for locomotive_id, var in self.selection_vars.items():
            if locomotive_id[0] == series:
                var.set(is_selected)
        
        self._update_selection_counter()
        logger.debug(f"Toggled series {series}: {is_selected}")
    
    def _select_all(self) -> None:
        """Select all locomotives."""
        for var in self.selection_vars.values():
            var.set(True)
        for var in self.series_vars.values():
            var.set(True)
        self._update_selection_counter()
        logger.debug("Selected all locomotives")
    
    def _deselect_all(self) -> None:
        """Deselect all locomotives."""
        for var in self.selection_vars.values():
            var.set(False)
        for var in self.series_vars.values():
            var.set(False)
        self._update_selection_counter()
        logger.debug("Deselected all locomotives")
    
    def _invert_selection(self) -> None:
        """Invert current selection."""
        for var in self.selection_vars.values():
            var.set(not var.get())
        self._update_series_checkboxes()
        self._update_selection_counter()
        logger.debug("Inverted locomotive selection")
    
    def _update_series_checkboxes(self) -> None:
        """Update series checkboxes based on individual selections."""
        for series, series_var in self.series_vars.items():
            # Check if all locomotives in series are selected
            series_locomotives = [
                loc_id for loc_id in self.selection_vars.keys() 
                if loc_id[0] == series
            ]
            
            if series_locomotives:
                all_selected = all(
                    self.selection_vars[loc_id].get() 
                    for loc_id in series_locomotives
                )
                series_var.set(all_selected)
    
    def _update_selection_counter(self) -> None:
        """Update selection counter display."""
        selected_count = sum(1 for var in self.selection_vars.values() if var.get())
        total_count = len(self.selection_vars)
        
        percentage = (selected_count / total_count * 100) if total_count > 0 else 0
        
        counter_text = f"📊 Выбрано: {selected_count:,} из {total_count:,} ({percentage:.1f}%)"
        self.selection_label.config(text=counter_text)
    
    def _load_current_selection(self) -> None:
        """Load current filter selection state."""
        if hasattr(self.filter, 'selected'):
            for locomotive_id, var in self.selection_vars.items():
                is_selected = locomotive_id in self.filter.selected
                var.set(is_selected)
        elif hasattr(self.filter, 'sel'):
            # Старый формат
            for locomotive_id, var in self.selection_vars.items():
                is_selected = locomotive_id in self.filter.sel
                var.set(is_selected)
        
        self._update_series_checkboxes()
        self._update_selection_counter()
    
    # Dialog actions
    def _apply_selection(self) -> None:
        """Apply current selection and close dialog."""
        try:
            # Collect selected locomotives
            selected_locomotives = [
                locomotive_id for locomotive_id, var in self.selection_vars.items()
                if var.get()
            ]
            
            # Update filter
            if hasattr(self.filter, 'set_selected_locomotives'):
                self.filter.set_selected_locomotives(selected_locomotives)
            elif hasattr(self.filter, 'sel'):
                # Старый формат
                self.filter.sel = set(selected_locomotives)
            
            # Create result - используем старый формат для совместимости
            self.result = {
                'use_coefficients': self.use_coefficients.get(),
                'exclude_low_work': self.exclude_low_work.get(),
                'coefficients_manager': self.coefficient_manager,
                'selected_locomotives': selected_locomotives
            }
            
            logger.info(f"Applied selection: {len(selected_locomotives)} locomotives, "
                       f"coefficients: {self.use_coefficients.get()}")
            
            self.dialog.destroy()
            
        except Exception as e:
            logger.error(f"Failed to apply selection: {e}")
            messagebox.showerror("Ошибка", f"Не удалось применить выбор:\n{str(e)}")
    
    def _cancel(self) -> None:
        """Cancel dialog without changes."""
        self.result = None
        logger.debug("Locomotive selection dialog cancelled")
        self.dialog.destroy()