# Файл: dialogs/selector.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager

class LocomotiveSelectorDialog:
    """Диалог выбора локомотивов для анализа"""
    
    def __init__(self, parent, locomotive_filter, coefficients_manager=None):
        self.parent = parent
        self.filter = locomotive_filter
        self.coefficients_manager = coefficients_manager or LocomotiveCoefficientsManager()
        self.result = None
        self.checkbox_vars = {}
        self.series_vars = {}
        self.use_coefficients = tk.BooleanVar(value=False)
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Выбор локомотивов и коэффициентов")
        self.dialog.geometry("900x700")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.create_widgets()
        self.load_current_selection()
        self.center_window()
    
    def center_window(self):
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        coef_frame = ttk.LabelFrame(main_frame, text="Коэффициенты расхода локомотивов", padding="10")
        coef_frame.pack(fill=tk.X, pady=(0, 10))
        file_frame = ttk.Frame(coef_frame)
        file_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(file_frame, text="Файл коэффициентов:").pack(side=tk.LEFT, padx=(0, 5))
        self.coef_file_label = ttk.Label(file_frame, text="Не загружен", foreground="gray")
        self.coef_file_label.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Загрузить", command=self.load_coefficients_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(file_frame, text="Очистить", command=self.clear_coefficients).pack(side=tk.LEFT)
        self.coef_stats_label = ttk.Label(coef_frame, text="", foreground="blue")
        self.coef_stats_label.pack(fill=tk.X, pady=(5, 0))
        self.use_coef_check = ttk.Checkbutton(
            coef_frame,
            text="Применять коэффициенты при анализе",
            variable=self.use_coefficients,
            command=self.on_use_coefficients_changed
        )
        self.use_coef_check.pack(pady=(5, 0))
        selection_frame = ttk.LabelFrame(main_frame, text="Выбор локомотивов для анализа", padding="10")
        selection_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        control_frame = ttk.Frame(selection_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(control_frame, text="Выбрать все", command=self.select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Снять все", command=self.deselect_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Инвертировать", command=self.invert_selection).pack(side=tk.LEFT, padx=(0, 10))
        self.selection_label = ttk.Label(control_frame, text="", foreground="green")
        self.selection_label.pack(side=tk.LEFT, padx=(10, 0))
        self.notebook = ttk.Notebook(selection_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.create_series_tabs()
        info_frame = ttk.Frame(main_frame)
        info_frame.pack(fill=tk.X)
        info_text = ("• Выберите локомотивы, которые нужно включить в анализ\n"
                    "• Загрузите файл коэффициентов для учета индивидуальных характеристик\n"
                    "• При включенных коэффициентах фактический расход будет скорректирован")
        info_label = ttk.Label(info_frame, text=info_text, foreground="gray")
        info_label.pack(pady=(0, 10))
        button_frame = ttk.Frame(info_frame)
        button_frame.pack()
        ttk.Button(button_frame, text="Применить", command=self.apply_selection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Отмена", command=self.cancel).pack(side=tk.LEFT)
    
    def create_series_tabs(self):
        locomotives_by_series = self.filter.get_locomotives_by_series()
        for series in sorted(locomotives_by_series.keys()):
            tab_frame = ttk.Frame(self.notebook)
            self.notebook.add(tab_frame, text=series)
            canvas = tk.Canvas(tab_frame, highlightthickness=0)
            scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            series_var = tk.BooleanVar(value=True)
            self.series_vars[series] = series_var
            series_check = ttk.Checkbutton(
                scrollable_frame,
                text=f"Выбрать всю серию {series}",
                variable=series_var,
                command=lambda s=series: self.toggle_series(s)
            )
            series_check.grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(5, 10))
            ttk.Label(scrollable_frame, text="№", font=('Arial', 9, 'bold')).grid(row=1, column=0, padx=5, pady=2)
            ttk.Label(scrollable_frame, text="Выбор", font=('Arial', 9, 'bold')).grid(row=1, column=1, padx=5, pady=2)
            ttk.Label(scrollable_frame, text="Номер", font=('Arial', 9, 'bold')).grid(row=1, column=2, padx=5, pady=2)
            ttk.Label(scrollable_frame, text="Коэффициент", font=('Arial', 9, 'bold')).grid(row=1, column=3, padx=5, pady=2)
            ttk.Separator(scrollable_frame, orient='horizontal').grid(row=2, column=0, columnspan=4, sticky='ew', pady=2)
            numbers = locomotives_by_series[series]
            for i, number in enumerate(numbers):
                row_num = i + 3
                ttk.Label(scrollable_frame, text=f"{i+1}").grid(row=row_num, column=0, padx=5, pady=1)
                var = tk.BooleanVar(value=True)
                self.checkbox_vars[(series, number)] = var
                check = ttk.Checkbutton(scrollable_frame, variable=var, command=lambda: self.update_selection_count())
                check.grid(row=row_num, column=1, padx=5, pady=1)
                ttk.Label(scrollable_frame, text=f"{number:04d}").grid(row=row_num, column=2, padx=5, pady=1)
                coef = self.coefficients_manager.get_coefficient(series, number)
                coef_text = f"{coef:.3f}" if coef != 1.0 else "-"
                coef_color = "red" if coef > 1.05 else "green" if coef < 0.95 else "black"
                coef_label = ttk.Label(scrollable_frame, text=coef_text, foreground=coef_color)
                coef_label.grid(row=row_num, column=3, padx=5, pady=1)
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
    
    def toggle_series(self, series):
        is_selected = self.series_vars[series].get()
        for (s, number), var in self.checkbox_vars.items():
            if s == series:
                var.set(is_selected)
        self.update_selection_count()
    
    def select_all(self):
        for var in self.checkbox_vars.values():
            var.set(True)
        for var in self.series_vars.values():
            var.set(True)
        self.update_selection_count()
    
    def deselect_all(self):
        for var in self.checkbox_vars.values():
            var.set(False)
        for var in self.series_vars.values():
            var.set(False)
        self.update_selection_count()
    
    def invert_selection(self):
        for var in self.checkbox_vars.values():
            var.set(not var.get())
        self.update_selection_count()
        self.update_series_checkboxes()
    
    def update_series_checkboxes(self):
        for series, series_var in self.series_vars.items():
            all_selected = True
            for (s, _), var in self.checkbox_vars.items():
                if s == series and not var.get():
                    all_selected = False
                    break
            series_var.set(all_selected)
    
    def update_selection_count(self):
        selected = sum(1 for var in self.checkbox_vars.values() if var.get())
        total = len(self.checkbox_vars)
        self.selection_label.config(text=f"Выбрано: {selected} из {total}")
    
    def load_coefficients_file(self):
        filename = filedialog.askopenfilename(title="Выберите файл коэффициентов", filetypes=[("Excel files", "*.xlsx *.xls")])
        if filename:
            if self.coefficients_manager.load_coefficients(filename):
                self.coef_file_label.config(text=filename.split('/')[-1], foreground="black")
                stats = self.coefficients_manager.get_statistics()
                if stats:
                    stats_text = f"Загружено: {stats['total_locomotives']} локомотивов, {stats['series_count']} серий. Средн. откл.: {stats['avg_deviation_percent']:.1f}%"
                    self.coef_stats_label.config(text=stats_text)
                messagebox.showinfo("Успех", f"Коэффициенты загружены успешно!\nЛокомотивов: {stats['total_locomotives']}")
                self.refresh_coefficients_display()
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить файл коэффициентов")
    
    def clear_coefficients(self):
        self.coefficients_manager = LocomotiveCoefficientsManager()
        self.coef_file_label.config(text="Не загружен", foreground="gray")
        self.coef_stats_label.config(text="")
        self.use_coefficients.set(False)
        self.refresh_coefficients_display()
    
    def refresh_coefficients_display(self):
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
        self.checkbox_vars.clear()
        self.series_vars.clear()
        self.create_series_tabs()
        self.load_current_selection()
    
    def on_use_coefficients_changed(self):
        if self.use_coefficients.get() and not self.coefficients_manager.coefficients_data:
            messagebox.showwarning("Предупреждение", "Сначала загрузите файл с коэффициентами")
            self.use_coefficients.set(False)
    
    def load_current_selection(self):
        for (series, number), var in self.checkbox_vars.items():
            is_selected = (series, number) in self.filter.selected_locomotives
            var.set(is_selected)
        self.update_series_checkboxes()
        self.update_selection_count()
    
    def apply_selection(self):
        selected = []
        for (series, number), var in self.checkbox_vars.items():
            if var.get():
                selected.append((series, number))
        self.filter.set_selected_locomотives(selected)
        self.result = {
            'selected_locomotives': selected,
            'use_coefficients': self.use_coefficients.get(),
            'coefficients_manager': self.coefficients_manager
        }
        self.dialog.destroy()
    
    def cancel(self):
        self.result = None
        self.dialog.destroy()