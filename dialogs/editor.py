# Файл: dialogs/editor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

class NormEditorDialog:
    """Диалог редактирования норм участка"""
    
    def __init__(self, parent, section_name, existing_norms=None):
        self.parent = parent
        self.section_name = section_name
        self.existing_norms = existing_norms or {}
        self.edited_norms = {}
        self.result = None
        self.norm_entries = {}
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(f"Актуализация норм - {section_name}")
        self.dialog.geometry("800x600")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.create_widgets()
        self.load_existing_norms()
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
        title_label = ttk.Label(main_frame, text=f"Редактирование норм для участка: {self.section_name}", font=('Arial', 12, 'bold'))
        title_label.pack(pady=(0, 10))
        info_text = ("Введите точки нормы (нагрузка на ось → удельный расход).\n"
                    "Минимум 2 точки для построения кривой.\n"
                    "Для удаления нормы очистите все точки.")
        info_label = ttk.Label(main_frame, text=info_text, foreground='gray')
        info_label.pack(pady=(0, 10))
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        add_norm_btn = ttk.Button(main_frame, text="+ Добавить норму", command=self.add_new_norm)
        add_norm_btn.pack(pady=(0, 10))
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(button_frame, text="Применить", command=self.apply_changes).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(button_frame, text="Отмена", command=self.cancel).pack(side=tk.RIGHT)
        ttk.Button(button_frame, text="Сброс", command=self.reset_norms).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Предпросмотр", command=self.preview_norms).pack(side=tk.LEFT, padx=(5, 0))
        for i in range(1, 4):
            self.create_norm_tab(i)
    
    def create_norm_tab(self, norm_id):
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=f"Норма №{norm_id}")
        self.norm_tabs[norm_id] = tab_frame
        desc_frame = ttk.Frame(tab_frame)
        desc_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(desc_frame, text="Описание:").pack(side=tk.LEFT, padx=(0, 5))
        desc_entry = ttk.Entry(desc_frame, width=50)
        desc_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        points_frame = ttk.LabelFrame(tab_frame, text="Точки нормы", padding="10")
        points_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(points_frame, text="№", width=5).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(points_frame, text="Нагрузка на ось, т", width=20).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(points_frame, text="Удельный расход, кВт·ч/изм", width=25).grid(row=0, column=2, padx=5, pady=5)
        entries = {'description': desc_entry, 'points': []}
        for i in range(10):
            ttk.Label(points_frame, text=f"{i+1}").grid(row=i+1, column=0, padx=5, pady=2)
            load_entry = ttk.Entry(points_frame, width=20)
            load_entry.grid(row=i+1, column=1, padx=5, pady=2)
            consumption_entry = ttk.Entry(points_frame, width=25)
            consumption_entry.grid(row=i+1, column=2, padx=5, pady=2)
            entries['points'].append((load_entry, consumption_entry))
        self.norm_entries[norm_id] = entries
        btn_frame = ttk.Frame(points_frame)
        btn_frame.grid(row=11, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(btn_frame, text="Очистить все", command=lambda nid=norm_id: self.clear_norm_points(nid)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Сортировать", command=lambda nid=norm_id: self.sort_norm_points(nid)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Интерполировать", command=lambda nid=norm_id: self.interpolate_points(nid)).pack(side=tk.LEFT, padx=5)
    
    def add_new_norm(self):
        new_id = len(self.norm_tabs) + 1
        if new_id <= 10:
            self.create_norm_tab(new_id)
            self.notebook.select(len(self.notebook.tabs()) - 1)
        else:
            messagebox.showwarning("Предупреждение", "Достигнуто максимальное количество норм (10)")
    
    def load_existing_norms(self):
        for norm_id, norm_data in self.existing_norms.items():
            if norm_id in self.norm_entries:
                entries = self.norm_entries[norm_id]
                if 'description' in norm_data:
                    entries['description'].insert(0, norm_data['description'])
                if 'points' in norm_data:
                    points = norm_data['points']
                    for i, (load, consumption) in enumerate(points[:10]):
                        entries['points'][i][0].insert(0, str(load))
                        entries['points'][i][1].insert(0, str(consumption))
    
    def clear_norm_points(self, norm_id):
        if norm_id in self.norm_entries:
            entries = self.norm_entries[norm_id]
            for load_entry, consumption_entry in entries['points']:
                load_entry.delete(0, tk.END)
                consumption_entry.delete(0, tk.END)
    
    def sort_norm_points(self, norm_id):
        if norm_id in self.norm_entries:
            entries = self.norm_entries[norm_id]
            points = []
            for load_entry, consumption_entry in entries['points']:
                load_val = load_entry.get().strip()
                consumption_val = consumption_entry.get().strip()
                if load_val and consumption_val:
                    try:
                        points.append((float(load_val), float(consumption_val)))
                    except ValueError:
                        continue
            points.sort(key=lambda x: x[0])
            self.clear_norm_points(norm_id)
            for i, (load, consumption) in enumerate(points):
                entries['points'][i][0].insert(0, str(load))
                entries['points'][i][1].insert(0, str(consumption))
    
    def interpolate_points(self, norm_id):
        if norm_id in self.norm_entries:
            entries = self.norm_entries[norm_id]
            points = []
            for load_entry, consumption_entry in entries['points']:
                load_val = load_entry.get().strip()
                consumption_val = consumption_entry.get().strip()
                if load_val and consumption_val:
                    try:
                        points.append((float(load_val), float(consumption_val)))
                    except ValueError:
                        continue
            if len(points) < 2:
                messagebox.showwarning("Предупреждение", "Нужно минимум 2 точки для интерполяции")
                return
            points.sort(key=lambda x: x[0])
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            if len(points) == 2:
                interp_func = interp1d(x_vals, y_vals, kind='linear')
            else:
                try:
                    interp_func = CubicSpline(x_vals, y_vals)
                except:
                    interp_func = interp1d(x_vals, y_vals, kind='quadratic')
            x_new = np.linspace(min(x_vals), max(x_vals), min(10, len(points) + 3))
            y_new = interp_func(x_new)
            self.clear_norm_points(norm_id)
            for i, (x, y) in enumerate(zip(x_new, y_new)):
                if i < 10:
                    entries['points'][i][0].insert(0, f"{x:.1f}")
                    entries['points'][i][1].insert(0, f"{y:.1f}")
    
    def get_edited_norms(self):
        edited_norms = {}
        for norm_id, entries in self.norm_entries.items():
            points = []
            for load_entry, consumption_entry in entries['points']:
                load_val = load_entry.get().strip()
                consumption_val = consumption_entry.get().strip()
                if load_val and consumption_val:
                    try:
                        points.append((float(load_val), float(consumption_val)))
                    except ValueError:
                        continue
            if len(points) >= 2:
                points.sort(key=lambda x: x[0])
                edited_norms[norm_id] = {
                    'points': points,
                    'description': entries['description'].get().strip()
                }
        return edited_norms
    
    def validate_norms(self):
        edited_norms = self.get_edited_norms()
        if not edited_norms:
            messagebox.showwarning("Предупреждение", "Не введено ни одной корректной нормы")
            return False
        for norm_id, norm_data in edited_norms.items():
            points = norm_data['points']
            x_vals = [p[0] for p in points]
            if len(x_vals) != len(set(x_vals)):
                messagebox.showwarning("Предупреждение", f"Норма №{norm_id}: обнаружены дублирующиеся значения нагрузки")
                return False
            for load, consumption in points:
                if load <= 0 or consumption <= 0:
                    messagebox.showwarning("Предупреждение", f"Норма №{norm_id}: значения должны быть положительными")
                    return False
        return True
    
    def preview_norms(self):
        if not self.validate_norms():
            return
        edited_norms = self.get_edited_norms()
        preview_window = tk.Toplevel(self.dialog)
        preview_window.title("Предпросмотр норм")
        preview_window.geometry("600x400")
        preview_window.transient(self.dialog)
        text_widget = tk.Text(preview_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        preview_text = f"Участок: {self.section_name}\n" + "=" * 50 + "\n\n"
        for norm_id, norm_data in sorted(edited_norms.items()):
            preview_text += f"Норма №{norm_id}\n"
            if norm_data['description']:
                preview_text += f"Описание: {norm_data['description']}\n"
            preview_text += f"Количество точек: {len(norm_data['points'])}\nТочки:\n"
            for i, (load, consumption) in enumerate(norm_data['points'], 1):
                preview_text += f"  {i}. Нагрузка: {load:.1f} т/ось → Расход: {consumption:.1f} кВт·ч/изм\n"
            preview_text += "\n"
        text_widget.insert(1.0, preview_text)
        text_widget.config(state='disabled')
        ttk.Button(preview_window, text="Закрыть", command=preview_window.destroy).pack(pady=10)
    
    def apply_changes(self):
        if not self.validate_norms():
            return
        self.edited_norms = self.get_edited_norms()
        self.result = 'apply'
        self.dialog.destroy()
    
    def cancel(self):
        self.result = 'cancel'
        self.dialog.destroy()
    
    def reset_norms(self):
        response = messagebox.askyesno("Подтверждение", "Сбросить все изменения к исходным нормам?")
        if response:
            for norm_id in self.norm_entries:
                self.clear_norm_points(norm_id)
                self.norm_entries[norm_id]['description'].delete(0, tk.END)
            self.load_existing_norms()

class NormComparator:
    """Класс для сравнения норм"""
    
    @staticmethod
    def compare_norms(original_norms, edited_norms, routes_df):
        comparison_results = {
            'original': {},
            'edited': {},
            'differences': {}
        }
        if original_norms:
            orig_stats = NormComparator._analyze_with_norms(routes_df, original_norms)
            comparison_results['original'] = orig_stats
        if edited_norms:
            edited_stats = NormComparator._analyze_with_norms(routes_df, edited_norms)
            comparison_results['edited'] = edited_stats
        if original_norms and edited_norms:
            comparison_results['differences'] = NormComparator._calculate_differences(
                comparison_results['original'], 
                comparison_results['edited']
            )
        return comparison_results
    
    @staticmethod
    def _analyze_with_norms(routes_df, norms):
        stats = {
            'total_routes': len(routes_df),
            'processed': 0,
            'economy_strong': 0,
            'economy_medium': 0,
            'economy_weak': 0,
            'normal': 0,
            'overrun_weak': 0,
            'overrun_medium': 0,
            'overrun_strong': 0,
            'mean_deviation': 0,
            'median_deviation': 0
        }
        deviations = []
        norm_functions = {}
        for norm_id, norm_data in norms.items():
            points = norm_data['points']
            x_vals = [p[0] for p in points]
            y_vals = [p[1] for p in points]
            if len(points) == 2:
                interp_func = interp1d(x_vals, y_vals, kind='linear', fill_value='extrapolate', bounds_error=False)
            else:
                try:
                    interp_func = CubicSpline(x_vals, y_vals)
                except:
                    interp_func = interp1d(x_vals, y_vals, kind='quadratic', fill_value='extrapolate', bounds_error=False)
            norm_functions[norm_id] = interp_func
        for _, row in routes_df.iterrows():
            norm_id = row['Номер нормы']
            if norm_id in norm_functions:
                try:
                    norm_value = float(norm_functions[norm_id](row['Нажатие на ось']))
                    if norm_value > 0:
                        deviation = ((row['Фактический удельный'] - norm_value) / norm_value) * 100
                        deviations.append(deviation)
                        stats['processed'] += 1
                        if deviation >= 30:
                            stats['economy_strong'] += 1
                        elif deviation >= 20:
                            stats['economy_medium'] += 1
                        elif deviation >= 5:
                            stats['economy_weak'] += 1
                        elif deviation >= -5:
                            stats['normal'] += 1
                        elif deviation >= -20:
                            stats['overrun_weak'] += 1
                        elif deviation >= -30:
                            stats['overrun_medium'] += 1
                        else:
                            stats['overrun_strong'] += 1
                except:
                    continue
        if deviations:
            stats['mean_deviation'] = np.mean(deviations)
            stats['median_deviation'] = np.median(deviations)
        return stats
    
    @staticmethod
    def _calculate_differences(original_stats, edited_stats):
        differences = {}
        for key in original_stats:
            if isinstance(original_stats[key], (int, float)):
                differences[key] = edited_stats[key] - original_stats[key]
        return differences