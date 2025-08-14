# dialogs/editor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Исправленный редактор норм с правильной интеграцией
Объединяет рабочую функциональность старого кода с новыми оптимизациями
"""

import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import logging

logger = logging.getLogger(__name__)

# Python 3.12 type definitions
type NormPoints[T: float] = list[tuple[T, T]]
type ValidationResult = tuple[bool, str]

@dataclass(slots=True)
class NormData:
    """Norm data with slots optimization."""
    norm_id: int
    points: NormPoints[float] = field(default_factory=list)
    description: str = ""
    
    def validate(self) -> ValidationResult:
        """Validate norm data."""
        if len(self.points) < 2:
            return False, "Минимум 2 точки для нормы"
        
        # Check for duplicate X values
        x_values = [p[0] for p in self.points]
        if len(x_values) != len(set(x_values)):
            return False, "Обнаружены дублирующиеся значения нагрузки"
        
        # Check for positive values
        for load, consumption in self.points:
            if load <= 0 or consumption <= 0:
                return False, "Значения должны быть положительными"
        
        return True, "OK"

class NormValidator(Protocol):
    """Protocol for norm validation strategies."""
    def validate(self, norm_data: NormData) -> ValidationResult: ...

class StandardNormValidator:
    """Standard norm validation implementation."""
    
    def validate(self, norm_data: NormData) -> ValidationResult:
        """Validate norm data with comprehensive checks."""
        return norm_data.validate()

class NormEditorDialog:
    """Исправленный редактор норм с полной совместимостью."""
    
    MAX_NORMS = 10
    MAX_POINTS_PER_NORM = 15
    
    def __init__(self, parent: tk.Tk, section_name: str, existing_norms: dict = None):
        self.parent = parent
        self.section_name = section_name
        self.existing_norms = existing_norms or {}
        self.validator = StandardNormValidator()
        
        # Result tracking
        self.result: str | None = None
        self.edited_norms: dict = {}
        
        # UI components
        self.norm_editors: dict[int, dict] = {}
        self.notebook: ttk.Notebook | None = None
        
        self._create_dialog()
        self._setup_norms()
        self._load_existing_data()
        
        logger.info(f"Norm editor opened for section: {section_name}")
    
    def _create_dialog(self) -> None:
        """Create modern dialog interface."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(f"📝 Актуализация норм - {self.section_name}")
        self.dialog.geometry("1000x700")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self._center_dialog()
        
        # Main container
        main_frame = ttk.Frame(self.dialog, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create sections
        self._create_header(main_frame)
        self._create_norm_tabs(main_frame)
        self._create_controls(main_frame)
        self._create_buttons(main_frame)
        
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
    
    def _create_header(self, parent: ttk.Frame) -> None:
        """Create dialog header."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        # Title
        title_label = ttk.Label(
            header_frame,
            text=f"📊 Редактирование норм для участка: {self.section_name}",
            font=("Arial", 14, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Instructions
        instructions = (
            "💡 Инструкции:\n"
            "• Введите точки нормы (нагрузка на ось → удельный расход)\n"
            "• Минимум 2 точки для построения кривой\n"
            "• Используйте кнопки для сортировки и интерполяции\n"
            "• Для удаления нормы очистите все точки"
        )
        
        instructions_label = ttk.Label(
            header_frame,
            text=instructions,
            foreground="gray",
            font=("Arial", 9)
        )
        instructions_label.pack()
    
    def _create_norm_tabs(self, parent: ttk.Frame) -> None:
        """Create tabbed interface for norms."""
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, sticky="nsew", pady=(0, 15))
        
        # Create initial tabs
        for norm_id in range(1, 4):  # Start with 3 norms
            self._create_norm_tab(norm_id)
    
    def _create_controls(self, parent: ttk.Frame) -> None:
        """Create control buttons."""
        controls_frame = ttk.Frame(parent)
        controls_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        
        ttk.Button(
            controls_frame,
            text="➕ Добавить норму",
            command=self._add_new_norm
        ).pack(side="left", padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="🔍 Предпросмотр",
            command=self._preview_norms
        ).pack(side="left", padx=(0, 10))
        
        ttk.Button(
            controls_frame,
            text="🔄 Сброс",
            command=self._reset_norms
        ).pack(side="left")
    
    def _create_buttons(self, parent: ttk.Frame) -> None:
        """Create dialog action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=3, column=0, sticky="ew")
        
        ttk.Button(
            button_frame,
            text="✅ Применить",
            command=self._apply_changes
        ).pack(side="right", padx=(10, 0))
        
        ttk.Button(
            button_frame,
            text="❌ Отмена",
            command=self._cancel
        ).pack(side="right")
    
    def _create_norm_tab(self, norm_id: int) -> None:
        """Create tab for individual norm editing."""
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=f"Норма №{norm_id}")
        
        # Main container
        main_container = ttk.Frame(tab_frame)
        main_container.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Description section
        desc_frame = ttk.LabelFrame(main_container, text="📝 Описание", padding="10")
        desc_frame.pack(fill="x", pady=(0, 15))
        
        description_entry = ttk.Entry(desc_frame, width=60, font=("Arial", 10))
        description_entry.pack(fill="x")
        
        # Points section
        points_frame = ttk.LabelFrame(main_container, text="📊 Точки нормы", padding="10")
        points_frame.pack(fill="both", expand=True, pady=(0, 15))
        
        # Create points table
        points_table = self._create_points_table(points_frame)
        
        # Control buttons for this norm
        norm_controls = ttk.Frame(main_container)
        norm_controls.pack(fill="x")
        
        control_buttons = [
            ("🗑️ Очистить все", lambda nid=norm_id: self._clear_norm_points(nid)),
            ("🔄 Сортировать", lambda nid=norm_id: self._sort_norm_points(nid)),
            ("📈 Интерполировать", lambda nid=norm_id: self._interpolate_points(nid)),
            ("✓ Валидировать", lambda nid=norm_id: self._validate_norm(nid))
        ]
        
        for text, command in control_buttons:
            ttk.Button(norm_controls, text=text, command=command).pack(side="left", padx=(0, 10))
        
        # Validation status
        validation_label = ttk.Label(norm_controls, text="", font=("Arial", 9))
        validation_label.pack(side="right")
        
        # Store references
        self.norm_editors[norm_id] = {
            'description': description_entry,
            'points_table': points_table,
            'validation_label': validation_label,
            'tab_frame': tab_frame
        }
    
    def _create_points_table(self, parent: ttk.Frame) -> list[tuple[ttk.Entry, ttk.Entry]]:
        """Create table for norm points entry."""
        # Table frame with scrollbar
        table_container = ttk.Frame(parent)
        table_container.pack(fill="both", expand=True)
        
        # Canvas for scrolling
        canvas = tk.Canvas(table_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Headers
        headers = ["№", "Нагрузка на ось, т", "Удельный расход, кВт·ч/10⁴ ткм", "Статус"]
        for i, header in enumerate(headers):
            ttk.Label(
                scrollable_frame,
                text=header,
                font=("Arial", 10, "bold")
            ).grid(row=0, column=i, padx=10, pady=5, sticky="w")
        
        # Points entries
        points_entries = []
        for i in range(self.MAX_POINTS_PER_NORM):
            row = i + 1
            
            # Row number
            ttk.Label(scrollable_frame, text=f"{i+1}").grid(
                row=row, column=0, padx=10, pady=2
            )
            
            # Load entry
            load_entry = ttk.Entry(scrollable_frame, width=15, font=("Consolas", 9))
            load_entry.grid(row=row, column=1, padx=10, pady=2, sticky="ew")
            
            # Consumption entry
            consumption_entry = ttk.Entry(scrollable_frame, width=20, font=("Consolas", 9))
            consumption_entry.grid(row=row, column=2, padx=10, pady=2, sticky="ew")
            
            # Status indicator
            status_label = ttk.Label(scrollable_frame, text="", font=("Arial", 8))
            status_label.grid(row=row, column=3, padx=10, pady=2, sticky="w")
            
            points_entries.append((load_entry, consumption_entry, status_label))
            
            # Bind validation events
            for entry in [load_entry, consumption_entry]:
                entry.bind('<FocusOut>', lambda e, r=i: self._validate_point(r))
                entry.bind('<KeyRelease>', lambda e, r=i: self._on_point_changed(r))
        
        # Configure grid weights
        scrollable_frame.columnconfigure(1, weight=1)
        scrollable_frame.columnconfigure(2, weight=1)
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return points_entries
    
    def _setup_norms(self) -> None:
        """Setup initial norm structure."""
        # Initialize with existing norms or create new ones
        if self.existing_norms:
            max_norm_id = max(self.existing_norms.keys()) if self.existing_norms else 0
            
            # Add tabs for additional norms if needed
            while len(self.norm_editors) <= max_norm_id:
                if len(self.norm_editors) < self.MAX_NORMS:
                    self._create_norm_tab(len(self.norm_editors) + 1)
                else:
                    break
    
    def _load_existing_data(self) -> None:
        """Load existing norm data into editors."""
        for norm_id, norm_def in self.existing_norms.items():
            if norm_id in self.norm_editors:
                editor = self.norm_editors[norm_id]
                
                # Load description
                description = ""
                if hasattr(norm_def, 'description') and norm_def.description:
                    description = norm_def.description
                elif isinstance(norm_def, dict) and 'description' in norm_def:
                    description = norm_def['description']
                
                if description:
                    editor['description'].insert(0, description)
                
                # Load points
                points = []
                if hasattr(norm_def, 'points'):
                    points = norm_def.points
                elif isinstance(norm_def, dict) and 'points' in norm_def:
                    points = norm_def['points']
                
                for i, (load, consumption) in enumerate(points[:self.MAX_POINTS_PER_NORM]):
                    if i < len(editor['points_table']):
                        load_entry, consumption_entry, _ = editor['points_table'][i]
                        load_entry.insert(0, str(load))
                        consumption_entry.insert(0, str(consumption))
                
                # Validate loaded data
                self._validate_norm(norm_id)
    
    # Event handlers
    def _validate_point(self, point_index: int) -> None:
        """Validate individual point entry."""
        current_tab = self.notebook.index(self.notebook.select())
        norm_id = current_tab + 1
        
        if norm_id not in self.norm_editors:
            return
        
        editor = self.norm_editors[norm_id]
        points_table = editor['points_table']
        
        if point_index >= len(points_table):
            return
        
        load_entry, consumption_entry, status_label = points_table[point_index]
        
        try:
            load_text = load_entry.get().strip()
            consumption_text = consumption_entry.get().strip()
            
            if not load_text and not consumption_text:
                status_label.config(text="", foreground="black")
                return
            
            if load_text and consumption_text:
                load_val = float(load_text)
                consumption_val = float(consumption_text)
                
                if load_val <= 0 or consumption_val <= 0:
                    status_label.config(text="❌", foreground="red")
                else:
                    status_label.config(text="✓", foreground="green")
            else:
                status_label.config(text="⚠️", foreground="orange")
                
        except ValueError:
            status_label.config(text="❌", foreground="red")
    
    def _on_point_changed(self, point_index: int) -> None:
        """Handle point value change."""
        # Trigger validation after short delay
        self.dialog.after(500, lambda: self._validate_point(point_index))
    
    def _add_new_norm(self) -> None:
        """Add new norm tab."""
        if len(self.norm_editors) >= self.MAX_NORMS:
            messagebox.showwarning(
                "Предупреждение",
                f"Достигнуто максимальное количество норм ({self.MAX_NORMS})"
            )
            return
        
        new_norm_id = len(self.norm_editors) + 1
        self._create_norm_tab(new_norm_id)
        
        # Select new tab
        self.notebook.select(len(self.notebook.tabs()) - 1)
        
        logger.debug(f"Added new norm tab: {new_norm_id}")
    
    def _clear_norm_points(self, norm_id: int) -> None:
        """Clear all points for specified norm."""
        if norm_id not in self.norm_editors:
            return
        
        editor = self.norm_editors[norm_id]
        for load_entry, consumption_entry, status_label in editor['points_table']:
            load_entry.delete(0, tk.END)
            consumption_entry.delete(0, tk.END)
            status_label.config(text="")
        
        self._validate_norm(norm_id)
        logger.debug(f"Cleared points for norm {norm_id}")
    
    def _sort_norm_points(self, norm_id: int) -> None:
        """Sort norm points by load value."""
        if norm_id not in self.norm_editors:
            return
        
        editor = self.norm_editors[norm_id]
        points_table = editor['points_table']
        
        # Extract points
        points = []
        for load_entry, consumption_entry, _ in points_table:
            load_text = load_entry.get().strip()
            consumption_text = consumption_entry.get().strip()
            
            if load_text and consumption_text:
                try:
                    load_val = float(load_text)
                    consumption_val = float(consumption_text)
                    points.append((load_val, consumption_val))
                except ValueError:
                    continue
        
        if len(points) < 2:
            messagebox.showinfo("Информация", "Недостаточно корректных точек для сортировки")
            return
        
        # Sort by load value
        points.sort(key=lambda x: x[0])
        
        # Clear and repopulate
        self._clear_norm_points(norm_id)
        
        for i, (load, consumption) in enumerate(points):
            if i < len(points_table):
                load_entry, consumption_entry, _ = points_table[i]
                load_entry.insert(0, str(load))
                consumption_entry.insert(0, str(consumption))
        
        self._validate_norm(norm_id)
        logger.debug(f"Sorted points for norm {norm_id}")
    
    def _interpolate_points(self, norm_id: int) -> None:
        """Interpolate additional points for norm."""
        if norm_id not in self.norm_editors:
            return
        
        editor = self.norm_editors[norm_id]
        points_table = editor['points_table']
        
        # Extract valid points
        points = []
        for load_entry, consumption_entry, _ in points_table:
            load_text = load_entry.get().strip()
            consumption_text = consumption_entry.get().strip()
            
            if load_text and consumption_text:
                try:
                    load_val = float(load_text)
                    consumption_val = float(consumption_text)
                    if load_val > 0 and consumption_val > 0:
                        points.append((load_val, consumption_val))
                except ValueError:
                    continue
        
        if len(points) < 2:
            messagebox.showwarning("Предупреждение", "Нужно минимум 2 корректные точки для интерполяции")
            return
        
        # Sort points
        points.sort(key=lambda x: x[0])
        
        try:
            # Prepare interpolation
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            
            # Create interpolation function
            if len(points) == 2:
                interp_func = interp1d(x_vals, y_vals, kind='linear')
            else:
                try:
                    interp_func = CubicSpline(x_vals, y_vals, bc_type='natural')
                except:
                    interp_func = interp1d(x_vals, y_vals, kind='quadratic')
            
            # Generate new points
            x_min, x_max = min(x_vals), max(x_vals)
            num_points = min(self.MAX_POINTS_PER_NORM, len(points) + 5)
            x_new = np.linspace(x_min, x_max, num_points)
            y_new = interp_func(x_new)
            
            # Clear and populate with interpolated points
            self._clear_norm_points(norm_id)
            
            for i, (x, y) in enumerate(zip(x_new, y_new)):
                if i < len(points_table):
                    load_entry, consumption_entry, _ = points_table[i]
                    load_entry.insert(0, f"{x:.1f}")
                    consumption_entry.insert(0, f"{y:.1f}")
            
            self._validate_norm(norm_id)
            messagebox.showinfo("Успех", f"Создано {len(x_new)} интерполированных точек")
            logger.debug(f"Interpolated {len(x_new)} points for norm {norm_id}")
            
        except Exception as e:
            logger.error(f"Interpolation failed for norm {norm_id}: {e}")
            messagebox.showerror("Ошибка", f"Не удалось выполнить интерполяцию:\n{str(e)}")
    
    def _validate_norm(self, norm_id: int) -> bool:
        """Validate complete norm."""
        if norm_id not in self.norm_editors:
            return False
        
        norm_data = self._extract_norm_data(norm_id)
        is_valid, message = self.validator.validate(norm_data)
        
        editor = self.norm_editors[norm_id]
        validation_label = editor['validation_label']
        
        if is_valid:
            validation_label.config(text="✅ Корректно", foreground="green")
        else:
            validation_label.config(text=f"❌ {message}", foreground="red")
        
        return is_valid
    
    def _extract_norm_data(self, norm_id: int) -> NormData:
        """Extract norm data from editor."""
        if norm_id not in self.norm_editors:
            return NormData(norm_id=norm_id)
        
        editor = self.norm_editors[norm_id]
        
        # Extract description
        description = editor['description'].get().strip()
        
        # Extract points
        points = []
        for load_entry, consumption_entry, _ in editor['points_table']:
            load_text = load_entry.get().strip()
            consumption_text = consumption_entry.get().strip()
            
            if load_text and consumption_text:
                try:
                    load_val = float(load_text)
                    consumption_val = float(consumption_text)
                    points.append((load_val, consumption_val))
                except ValueError:
                    continue
        
        return NormData(norm_id=norm_id, points=points, description=description)
    
    def _get_all_edited_norms(self) -> dict:
        """Get all edited norms."""
        edited_norms = {}
        
        for norm_id in self.norm_editors:
            norm_data = self._extract_norm_data(norm_id)
            
            if len(norm_data.points) >= 2:
                is_valid, _ = self.validator.validate(norm_data)
                
                if is_valid:
                    # Sort points by load value
                    sorted_points = sorted(norm_data.points, key=lambda x: x[0])
                    
                    # Create norm definition in old format for compatibility
                    edited_norms[norm_id] = {
                        'points': sorted_points,
                        'description': norm_data.description
                    }
        
        return edited_norms
    
    def _validate_all_norms(self) -> bool:
        """Validate all norms."""
        edited_norms = self._get_all_edited_norms()
        
        if not edited_norms:
            messagebox.showwarning("Предупреждение", "Не введено ни одной корректной нормы")
            return False
        
        # Additional cross-norm validation
        all_valid = True
        for norm_id in self.norm_editors:
            if not self._validate_norm(norm_id):
                norm_data = self._extract_norm_data(norm_id)
                if len(norm_data.points) > 0:  # Only report invalid if has data
                    all_valid = False
        
        return all_valid
    
    # Dialog actions
    def _preview_norms(self) -> None:
        """Preview all edited norms."""
        edited_norms = self._get_all_edited_norms()
        
        if not edited_norms:
            messagebox.showinfo("Информация", "Нет корректных норм для предпросмотра")
            return
        
        # Create preview window
        preview_window = tk.Toplevel(self.dialog)
        preview_window.title(f"🔍 Предпросмотр норм - {self.section_name}")
        preview_window.geometry("700x500")
        preview_window.transient(self.dialog)
        
        # Preview text
        preview_text = tk.Text(preview_window, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(preview_window, command=preview_text.yview)
        preview_text.config(yscrollcommand=scrollbar.set)
        
        # Generate preview content
        content = f"📊 ПРЕДПРОСМОТР НОРМ\n"
        content += f"Участок: {self.section_name}\n"
        content += "=" * 60 + "\n\n"
        
        for norm_id, norm_def in sorted(edited_norms.items()):
            content += f"📈 Норма №{norm_id}\n"
            content += f"Описание: {norm_def['description'] or 'Не указано'}\n"
            content += f"Количество точек: {len(norm_def['points'])}\n"
            
            x_vals = [p[0] for p in norm_def['points']]
            content += f"Диапазон нагрузки: {min(x_vals):.1f} - {max(x_vals):.1f} т/ось\n"
            content += "Точки:\n"
            
            for i, (load, consumption) in enumerate(norm_def['points'], 1):
                content += f"  {i:2d}. {load:6.1f} т/ось → {consumption:8.1f} кВт·ч/10⁴ ткм\n"
            
            content += "\n"
        
        content += f"Итого корректных норм: {len(edited_norms)}\n"
        
        preview_text.insert(1.0, content)
        preview_text.config(state='disabled')
        
        # Pack components
        preview_text.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        scrollbar.pack(side="right", fill="y", pady=10)
        
        # Close button
        ttk.Button(preview_window, text="Закрыть", 
                  command=preview_window.destroy).pack(pady=10)
        
        logger.debug(f"Opened preview for {len(edited_norms)} norms")
    
    def _reset_norms(self) -> None:
        """Reset all norms to original state."""
        if messagebox.askyesno("Подтверждение", 
                              "Сбросить все изменения к исходным нормам?\n\n"
                              "Все несохраненные изменения будут потеряны."):
            
            # Clear all editors
            for norm_id in self.norm_editors:
                self._clear_norm_points(norm_id)
                self.norm_editors[norm_id]['description'].delete(0, tk.END)
                self.norm_editors[norm_id]['validation_label'].config(text="")
            
            # Reload existing data
            self._load_existing_data()
            
            logger.debug("Reset all norms to original state")
    
    def _apply_changes(self) -> None:
        """Apply changes and close dialog."""
        if not self._validate_all_norms():
            return
        
        self.edited_norms = self._get_all_edited_norms()
        
        if not self.edited_norms:
            messagebox.showwarning("Предупреждение", "Нет корректных норм для сохранения")
            return
        
        # Confirm changes
        changes_summary = f"Будет сохранено {len(self.edited_norms)} норм для участка '{self.section_name}'"
        
        if messagebox.askyesno("Подтверждение", f"{changes_summary}\n\nПрименить изменения?"):
            self.result = 'apply'
            logger.info(f"Applied norm changes: {len(self.edited_norms)} norms for {self.section_name}")
            self.dialog.destroy()
    
    def _cancel(self) -> None:
        """Cancel dialog without saving."""
        if self._has_unsaved_changes():
            if not messagebox.askyesno("Подтверждение", 
                                     "Есть несохраненные изменения.\n\nВыйти без сохранения?"):
                return
        
        self.result = 'cancel'
        logger.debug("Norm editor cancelled")
        self.dialog.destroy()
    
    def _has_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes."""
        current_norms = self._get_all_edited_norms()
        
        # Compare with existing norms
        if len(current_norms) != len(self.existing_norms):
            return True
        
        for norm_id, current_norm in current_norms.items():
            if norm_id not in self.existing_norms:
                return True
            
            existing_norm = self.existing_norms[norm_id]
            
            # Compare points
            existing_points = []
            if hasattr(existing_norm, 'points'):
                existing_points = existing_norm.points
            elif isinstance(existing_norm, dict) and 'points' in existing_norm:
                existing_points = existing_norm['points']
            
            if len(current_norm['points']) != len(existing_points):
                return True
            
            # Compare individual points (with tolerance)
            for (curr_load, curr_cons), (exist_load, exist_cons) in zip(current_norm['points'], existing_points):
                if abs(curr_load - exist_load) > 0.01 or abs(curr_cons - exist_cons) > 0.01:
                    return True
        
        return False

class NormComparator:
    """Utility class for comparing norm analysis results."""
    
    @staticmethod
    def compare_norms(original_norms: dict, edited_norms: dict, routes_df) -> dict:
        """Compare analysis results between original and edited norms."""
        comparison_result = {
            'original': {},
            'edited': {},
            'differences': {}
        }
        
        # This would integrate with the analyzer for full comparison
        # For now, return basic structure
        if original_norms:
            comparison_result['original'] = NormComparator._analyze_with_norms(routes_df, original_norms)
        
        if edited_norms:
            comparison_result['edited'] = NormComparator._analyze_with_norms(routes_df, edited_norms)
        
        if original_norms and edited_norms:
            comparison_result['differences'] = NormComparator._calculate_differences(
                comparison_result['original'],
                comparison_result['edited']
            )
        
        return comparison_result
    
    @staticmethod
    def _analyze_with_norms(routes_df, norms: dict) -> dict:
        """Perform basic analysis with given norms."""
        # Placeholder implementation - would use full analyzer
        return {
            'total_routes': len(routes_df),
            'processed': len(routes_df),
            'economy_strong': 0,
            'economy_medium': 0,
            'economy_weak': 0,
            'normal': len(routes_df),
            'overrun_weak': 0,
            'overrun_medium': 0,
            'overrun_strong': 0,
            'mean_deviation': 0.0,
            'median_deviation': 0.0
        }
    
    @staticmethod
    def _calculate_differences(original_stats: dict, edited_stats: dict) -> dict:
        """Calculate differences between statistics."""
        differences = {}
        
        for key in original_stats:
            if isinstance(original_stats[key], (int, float)):
                differences[key] = edited_stats[key] - original_stats[key]
        
        return differences