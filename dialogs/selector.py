# dialogs/selector.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager

class LocomotiveSelectorDialog:
    """Диалог выбора локомотивов для анализа"""
    
    def __init__(self, p, lf, cm=None):
        self.p = p
        self.f = lf
        self.cm = cm or LocomotiveCoefficientsManager()
        self.res = None
        self.cv = {}
        self.sv = {}
        self.uc = tk.BooleanVar(value=False)
        self.exclude_low_work = tk.BooleanVar(value=False)  # Новая переменная
        self.d = tk.Toplevel(p)
        self.d.title("Выбор локомотивов и коэффициентов")
        self.d.geometry("900x700")
        self.d.transient(p)
        self.d.grab_set()
        self.create_widgets()
        self.load_current_selection()
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
        cf = ttk.LabelFrame(mf, text="Коэффициенты расхода локомотивов", padding="10")
        cf.pack(fill=tk.X, pady=(0, 10))
        ff = ttk.Frame(cf)
        ff.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(ff, text="Файл коэффициентов:").pack(side=tk.LEFT, padx=(0, 5))
        self.cfl = ttk.Label(ff, text="Не загружен", foreground="gray")
        self.cfl.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(ff, text="Загрузить", command=self.load_coefficients_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ff, text="Очистить", command=self.clear_coefficients).pack(side=tk.LEFT)
        self.csl = ttk.Label(cf, text="", foreground="blue")
        self.csl.pack(fill=tk.X, pady=(5, 0))
        
        self.ucc = ttk.Checkbutton(
            cf,
            text="Применять коэффициенты при анализе",
            variable=self.uc,
            command=self.on_use_coefficients_changed
        )
        self.ucc.pack(pady=(5, 0))
        
        # Добавляем новую галку для исключения локомотивов с малой работой
        self.elwc = ttk.Checkbutton(
            cf,
            text="Исключить при анализе локомотивы с менее 200 10тыс.ткм брутто",
            variable=self.exclude_low_work
        )
        self.elwc.pack(pady=(5, 0))
        
        sf = ttk.LabelFrame(mf, text="Выбор локомотивов для анализа", padding="10")
        sf.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        ctf = ttk.Frame(sf)
        ctf.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(ctf, text="Выбрать все", command=self.select_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ctf, text="Снять все", command=self.deselect_all).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(ctf, text="Инвертировать", command=self.invert_selection).pack(side=tk.LEFT, padx=(0, 10))
        self.sl = ttk.Label(ctf, text="", foreground="green")
        self.sl.pack(side=tk.LEFT, padx=(10, 0))
        self.nb = ttk.Notebook(sf)
        self.nb.pack(fill=tk.BOTH, expand=True)
        self.create_series_tabs()
        inf = ttk.Frame(mf)
        inf.pack(fill=tk.X)
        it = ("• Выберите локомотивы, которые нужно включить в анализ\n"
                    "• Загрузите файл коэффициентов для учета индивидуальных характеристик\n"
                    "• При включенных коэффициентах фактический расход будет скорректирован")
        il = ttk.Label(inf, text=it, foreground="gray")
        il.pack(pady=(0, 10))
        bf = ttk.Frame(inf)
        bf.pack()
        ttk.Button(bf, text="Применить", command=self.apply_selection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(bf, text="Отмена", command=self.cancel).pack(side=tk.LEFT)
    
    def create_series_tabs(self):
        lbs = self.f.get_locomotives_by_series()
        for s in sorted(lbs.keys()):
            tf = ttk.Frame(self.nb)
            self.nb.add(tf, text=s)
            cn = tk.Canvas(tf, highlightthickness=0)
            sb = ttk.Scrollbar(tf, orient="vertical", command=cn.yview)
            scf = ttk.Frame(cn)
            scf.bind("<Configure>", lambda e: cn.configure(scrollregion=cn.bbox("all")))
            cn.create_window((0, 0), window=scf, anchor="nw")
            cn.configure(yscrollcommand=sb.set)
            sv = tk.BooleanVar(value=True)
            self.sv[s] = sv
            sc = ttk.Checkbutton(
                scf,
                text=f"Выбрать всю серию {s}",
                variable=sv,
                command=lambda se=s: self.toggle_series(se)
            )
            sc.grid(row=0, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(5, 10))
            ttk.Label(scf, text="№", font=('Arial', 9, 'bold')).grid(row=1, column=0, padx=5, pady=2)
            ttk.Label(scf, text="Выбор", font=('Arial', 9, 'bold')).grid(row=1, column=1, padx=5, pady=2)
            ttk.Label(scf, text="Номер", font=('Arial', 9, 'bold')).grid(row=1, column=2, padx=5, pady=2)
            ttk.Label(scf, text="Коэффициент", font=('Arial', 9, 'bold')).grid(row=1, column=3, padx=5, pady=2)
            ttk.Separator(scf, orient='horizontal').grid(row=2, column=0, columnspan=4, sticky='ew', pady=2)
            nums = lbs[s]
            for i, n in enumerate(nums):
                rn = i + 3
                ttk.Label(scf, text=f"{i+1}").grid(row=rn, column=0, padx=5, pady=1)
                v = tk.BooleanVar(value=True)
                self.cv[(s, n)] = v
                c = ttk.Checkbutton(scf, variable=v, command=lambda: self.update_selection_count())
                c.grid(row=rn, column=1, padx=5, pady=1)
                # Убираем ведущие нули при отображении
                display_num = str(n) if n >= 1000 else str(n)
                ttk.Label(scf, text=display_num).grid(row=rn, column=2, padx=5, pady=1)
                co = self.cm.get_coefficient(s, n)
                ct = f"{co:.3f}" if co != 1.0 else "-"
                cc = "red" if co > 1.05 else "green" if co < 0.95 else "black"
                cl = ttk.Label(scf, text=ct, foreground=cc)
                cl.grid(row=rn, column=3, padx=5, pady=1)
            cn.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
    
    def toggle_series(self, s):
        is_sel = self.sv[s].get()
        for (se, n), v in self.cv.items():
            if se == s:
                v.set(is_sel)
        self.update_selection_count()
    
    def select_all(self):
        for v in self.cv.values():
            v.set(True)
        for v in self.sv.values():
            v.set(True)
        self.update_selection_count()
    
    def deselect_all(self):
        for v in self.cv.values():
            v.set(False)
        for v in self.sv.values():
            v.set(False)
        self.update_selection_count()
    
    def invert_selection(self):
        for v in self.cv.values():
            v.set(not v.get())
        self.update_selection_count()
        self.update_series_checkboxes()
    
    def update_series_checkboxes(self):
        for s, sv in self.sv.items():
            all_sel = True
            for (se, _), v in self.cv.items():
                if se == s and not v.get():
                    all_sel = False
                    break
            sv.set(all_sel)
    
    def update_selection_count(self):
        sel = sum(1 for v in self.cv.values() if v.get())
        tot = len(self.cv)
        self.sl.config(text=f"Выбрано: {sel} из {tot}")
    
    def load_coefficients_file(self):
        fn = filedialog.askopenfilename(title="Выберите файл коэффициентов", filetypes=[("Excel files", "*.xlsx *.xls")])
        if fn:
            # Получаем порог фильтрации
            min_work = 200 if self.exclude_low_work.get() else 0
            
            if self.cm.load_coefficients(fn, min_work):
                self.cfl.config(text=fn.split('/')[-1], foreground="black")
                st = self.cm.get_statistics()
                if st:
                    st_txt = f"Загружено: {st['total_locomotives']} локомотивов, {st['series_count']} серий. Средн. откл.: {st['avg_deviation_percent']:.1f}%"
                    self.csl.config(text=st_txt)
                messagebox.showinfo("Успех", f"Коэффициенты загружены успешно!\nЛокомотивов: {st['total_locomotives']}")
                # Обновляем интерфейс
                self.refresh_coefficients_display()
            else:
                messagebox.showerror("Ошибка", "Не удалось загрузить файл коэффициентов")
    
    def clear_coefficients(self):
        self.cm = LocomotiveCoefficientsManager()
        self.cfl.config(text="Не загружен", foreground="gray")
        self.csl.config(text="")
        self.uc.set(False)
        self.refresh_coefficients_display()
    
    def refresh_coefficients_display(self):
        # Сохраняем текущий выбор
        current_selection = {}
        for (s, n), v in self.cv.items():
            current_selection[(s, n)] = v.get()
        
        # Полностью пересоздаем все вкладки для обновления коэффициентов
        for tab_id in self.nb.tabs():
            self.nb.forget(tab_id)
        
        # Очищаем старые данные
        self.cv.clear()
        self.sv.clear()
        
        # Создаем вкладки заново с обновленными коэффициентами
        self.create_series_tabs()
        
        # Восстанавливаем выбор
        for (s, n), selected in current_selection.items():
            if (s, n) in self.cv:
                self.cv[(s, n)].set(selected)
        
        self.update_series_checkboxes()
        self.update_selection_count()
    
    def on_use_coefficients_changed(self):
        if self.uc.get() and not self.cm.data:
            messagebox.showwarning("Предупреждение", "Сначала загрузите файл с коэффициентами")
            self.uc.set(False)
    
    def load_current_selection(self):
        for (s, n), v in self.cv.items():
            is_sel = (s, n) in self.f.sel
            v.set(is_sel)
        self.update_series_checkboxes()
        self.update_selection_count()
    
    def apply_selection(self):
        sel = []
        for (s, n), v in self.cv.items():
            if v.get():
                sel.append((s, n))
        self.f.set_selected_locomotives(sel)
        self.res = {
            'selected_locomotives': sel,
            'use_coefficients': self.uc.get(),
            'exclude_low_work': self.exclude_low_work.get(),
            'coefficients_manager': self.cm
        }
        self.d.destroy()
    
    def cancel(self):
        self.res = None
        self.d.destroy()