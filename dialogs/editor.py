# dialogs/editor.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d, CubicSpline

class NormEditorDialog:
    """Диалог редактирования норм участка"""
    
    def __init__(self, p, sn, en=None):
        self.p = p
        self.sn = sn
        self.en = en or {}
        self.ed = {}
        self.res = None
        self.ne = {}
        self.nt = {}
        self.d = tk.Toplevel(p)
        self.d.title(f"Актуализация норм - {sn}")
        self.d.geometry("800x600")
        self.d.transient(p)
        self.d.grab_set()
        self.create_widgets()
        self.load_existing_norms()
        self.center_window()
    
    def center_window(self):
        self.d.update_idletasks()
        w = self.d.winfo_width()
        h = self.d.winfo_height()
        x = (self.d.winfo_screenwidth() // 2) - (w // 2)
        y = (self.d.winfo_screenheight() // 2) - (h // 2)
        self.d.geometry(f'{w}x{h}+{x}+{y}')
    
    def create_widgets(self):
        mf = ttk.Frame(self.d, padding="10")
        mf.pack(fill=tk.BOTH, expand=True)
        tl = ttk.Label(mf, text=f"Редактирование норм для участка: {self.sn}", font=('Arial', 12, 'bold'))
        tl.pack(pady=(0, 10))
        it = ("Введите точки нормы (нагрузка на ось → удельный расход).\n"
                    "Минимум 2 точки для построения кривой.\n"
                    "Для удаления нормы очистите все точки.")
        il = ttk.Label(mf, text=it, foreground='gray')
        il.pack(pady=(0, 10))
        self.nb = ttk.Notebook(mf)
        self.nb.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.nt = {}
        anb = ttk.Button(mf, text="+ Добавить норму", command=self.add_new_norm)
        anb.pack(pady=(0, 10))
        bf = ttk.Frame(mf)
        bf.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(bf, text="Применить", command=self.apply_changes).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(bf, text="Отмена", command=self.cancel).pack(side=tk.RIGHT)
        ttk.Button(bf, text="Сброс", command=self.reset_norms).pack(side=tk.LEFT)
        ttk.Button(bf, text="Предпросмотр", command=self.preview_norms).pack(side=tk.LEFT, padx=(5, 0))
        for i in range(1, 4):
            self.create_norm_tab(i)
    
    def create_norm_tab(self, ni):
        tf = ttk.Frame(self.nb)
        self.nb.add(tf, text=f"Норма №{ni}")
        self.nt[ni] = tf
        df = ttk.Frame(tf)
        df.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(df, text="Описание:").pack(side=tk.LEFT, padx=(0, 5))
        de = ttk.Entry(df, width=50)
        de.pack(side=tk.LEFT, fill=tk.X, expand=True)
        pf = ttk.LabelFrame(tf, text="Точки нормы", padding="10")
        pf.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(pf, text="№", width=5).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(pf, text="Нагрузка на ось, т", width=20).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(pf, text="Удельный расход, кВт·ч/изм", width=25).grid(row=0, column=2, padx=5, pady=5)
        ent = {'description': de, 'points': []}
        for i in range(10):
            ttk.Label(pf, text=f"{i+1}").grid(row=i+1, column=0, padx=5, pady=2)
            le = ttk.Entry(pf, width=20)
            le.grid(row=i+1, column=1, padx=5, pady=2)
            ce = ttk.Entry(pf, width=25)
            ce.grid(row=i+1, column=2, padx=5, pady=2)
            ent['points'].append((le, ce))
        self.ne[ni] = ent
        btnf = ttk.Frame(pf)
        btnf.grid(row=11, column=0, columnspan=3, pady=(10, 0))
        ttk.Button(btnf, text="Очистить все", command=lambda nid=ni: self.clear_norm_points(nid)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btnf, text="Сортировать", command=lambda nid=ni: self.sort_norm_points(nid)).pack(side=tk.LEFT, padx=5)
        ttk.Button(btnf, text="Интерполировать", command=lambda nid=ni: self.interpolate_points(nid)).pack(side=tk.LEFT, padx=5)
    
    def add_new_norm(self):
        nid = len(self.nt) + 1
        if nid <= 10:
            self.create_norm_tab(nid)
            self.nb.select(len(self.nb.tabs()) - 1)
        else:
            messagebox.showwarning("Предупреждение", "Достигнуто максимальное количество норм (10)")
    
    def load_existing_norms(self):
        for ni, nd in self.en.items():
            if ni in self.ne:
                ent = self.ne[ni]
                if 'description' in nd:
                    ent['description'].insert(0, nd['description'])
                if 'points' in nd:
                    pts = nd['points']
                    for i, (l, c) in enumerate(pts[:10]):
                        ent['points'][i][0].insert(0, str(l))
                        ent['points'][i][1].insert(0, str(c))
    
    def clear_norm_points(self, ni):
        if ni in self.ne:
            ent = self.ne[ni]
            for le, ce in ent['points']:
                le.delete(0, tk.END)
                ce.delete(0, tk.END)
    
    def sort_norm_points(self, ni):
        if ni in self.ne:
            ent = self.ne[ni]
            pts = []
            for le, ce in ent['points']:
                lv = le.get().strip()
                cv = ce.get().strip()
                if lv and cv:
                    try:
                        pts.append((float(lv), float(cv)))
                    except ValueError:
                        continue
            pts.sort(key=lambda x: x[0])
            self.clear_norm_points(ni)
            for i, (l, c) in enumerate(pts):
                ent['points'][i][0].insert(0, str(l))
                ent['points'][i][1].insert(0, str(c))
    
    def interpolate_points(self, ni):
        if ni in self.ne:
            ent = self.ne[ni]
            pts = []
            for le, ce in ent['points']:
                lv = le.get().strip()
                cv = ce.get().strip()
                if lv and cv:
                    try:
                        pts.append((float(lv), float(cv)))
                    except ValueError:
                        continue
            if len(pts) < 2:
                messagebox.showwarning("Предупреждение", "Нужно минимум 2 точки для интерполяции")
                return
            pts.sort(key=lambda x: x[0])
            xv = [p[0] for p in pts]
            yv = [p[1] for p in pts]
            if len(pts) == 2:
                inf = interp1d(xv, yv, kind='linear')
            else:
                try:
                    inf = CubicSpline(xv, yv)
                except:
                    inf = interp1d(xv, yv, kind='quadratic')
            xn = np.linspace(min(xv), max(xv), min(10, len(pts) + 3))
            yn = inf(xn)
            self.clear_norm_points(ni)
            for i, (x, y) in enumerate(zip(xn, yn)):
                if i < 10:
                    ent['points'][i][0].insert(0, f"{x:.1f}")
                    ent['points'][i][1].insert(0, f"{y:.1f}")
    
    def get_edited_norms(self):
        en = {}
        for ni, ent in self.ne.items():
            pts = []
            for le, ce in ent['points']:
                lv = le.get().strip()
                cv = ce.get().strip()
                if lv and cv:
                    try:
                        pts.append((float(lv), float(cv)))
                    except ValueError:
                        continue
            if len(pts) >= 2:
                pts.sort(key=lambda x: x[0])
                en[ni] = {
                    'points': pts,
                    'description': ent['description'].get().strip()
                }
        return en
    
    def validate_norms(self):
        en = self.get_edited_norms()
        if not en:
            messagebox.showwarning("Предупреждение", "Не введено ни одной корректной нормы")
            return False
        for ni, nd in en.items():
            pts = nd['points']
            xv = [p[0] for p in pts]
            if len(xv) != len(set(xv)):
                messagebox.showwarning("Предупреждение", f"Норма №{ni}: обнаружены дублирующиеся значения нагрузки")
                return False
            for l, c in pts:
                if l <= 0 or c <= 0:
                    messagebox.showwarning("Предупреждение", f"Норма №{ni}: значения должны быть положительными")
                    return False
        return True
    
    def preview_norms(self):
        if not self.validate_norms():
            return
        en = self.get_edited_norms()
        pw = tk.Toplevel(self.d)
        pw.title("Предпросмотр норм")
        pw.geometry("600x400")
        pw.transient(self.d)
        tw = tk.Text(pw, wrap=tk.WORD, padx=10, pady=10)
        tw.pack(fill=tk.BOTH, expand=True)
        pt = f"Участок: {self.sn}\n" + "=" * 50 + "\n\n"
        for ni, nd in sorted(en.items()):
            pt += f"Норма №{ni}\n"
            if nd['description']:
                pt += f"Описание: {nd['description']}\n"
            pt += f"Количество точек: {len(nd['points'])}\nТочки:\n"
            for i, (l, c) in enumerate(nd['points'], 1):
                pt += f"  {i}. Нагрузка: {l:.1f} т/ось → Расход: {c:.1f} кВт·ч/изм\n"
            pt += "\n"
        tw.insert(1.0, pt)
        tw.config(state='disabled')
        ttk.Button(pw, text="Закрыть", command=pw.destroy).pack(pady=10)
    
    def apply_changes(self):
        if not self.validate_norms():
            return
        self.ed = self.get_edited_norms()
        self.res = 'apply'
        self.d.destroy()
    
    def cancel(self):
        self.res = 'cancel'
        self.d.destroy()
    
    def reset_norms(self):
        resp = messagebox.askyesno("Подтверждение", "Сбросить все изменения к исходным нормам?")
        if resp:
            for ni in self.ne:
                self.clear_norm_points(ni)
                self.ne[ni]['description'].delete(0, tk.END)
            self.load_existing_norms()

class NormComparator:
    """Класс для сравнения норм"""
    
    @staticmethod
    def compare_norms(on, en, rdf):
        cr = {
            'original': {},
            'edited': {},
            'differences': {}
        }
        if on:
            os = NormComparator._analyze_with_norms(rdf, on)
            cr['original'] = os
        if en:
            es = NormComparator._analyze_with_norms(rdf, en)
            cr['edited'] = es
        if on and en:
            cr['differences'] = NormComparator._calculate_differences(
                cr['original'], 
                cr['edited']
            )
        return cr
    
    @staticmethod
    def _analyze_with_norms(rdf, norms):
        st = {
            'total_routes': len(rdf),
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
        dev = []
        nf = {}
        for ni, nd in norms.items():
            pts = nd['points']
            xv = [p[0] for p in pts]
            yv = [p[1] for p in pts]
            if len(pts) == 2:
                inf = interp1d(xv, yv, kind='linear', fill_value='extrapolate', bounds_error=False)
            else:
                try:
                    inf = CubicSpline(xv, yv)
                except:
                    inf = interp1d(xv, yv, kind='quadratic', fill_value='extrapolate', bounds_error=False)
            nf[ni] = inf
        for _, r in rdf.iterrows():
            ni = r['Номер нормы']
            if ni in nf:
                try:
                    nv = float(nf[ni](r['Нажатие на ось']))
                    if nv > 0:
                        d = ((r['Фактический удельный'] - nv) / nv) * 100
                        dev.append(d)
                        st['processed'] += 1
                        if d >= 30:
                            st['economy_strong'] += 1
                        elif d >= 20:
                            st['economy_medium'] += 1
                        elif d >= 5:
                            st['economy_weak'] += 1
                        elif d >= -5:
                            st['normal'] += 1
                        elif d >= -20:
                            st['overrun_weak'] += 1
                        elif d >= -30:
                            st['overrun_medium'] += 1
                        else:
                            st['overrun_strong'] += 1
                except:
                    continue
        if dev:
            st['mean_deviation'] = np.mean(dev)
            st['median_deviation'] = np.median(dev)
        return st
    
    @staticmethod
    def _calculate_differences(os, es):
        diff = {}
        for k in os:
            if isinstance(os[k], (int, float)):
                diff[k] = es[k] - os[k]
        return diff