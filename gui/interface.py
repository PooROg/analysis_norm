# gui/interface.py
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
from analysis.analyzer import InteractiveNormsAnalyzer
from dialogs.selector import LocomotiveSelectorDialog
from dialogs.editor import NormEditorDialog, NormComparator
from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager

class NormsAnalyzerGUI:
    def __init__(self, r):
        self.r = r
        self.r.title("Анализатор норм расхода электроэнергии РЖД")
        self.r.geometry("1200x700")
        self.a = None
        self.cp = None
        self.th = None
        self.lf = None
        self.cm = LocomotiveCoefficientsManager()
        self.uc = False
        self.elw = False  # exclude_low_work
        self.rf = 'Processed_Routes.xlsx'
        self.nf = 'Нормы участков.xlsx'
        self.create_widgets()
        self.setup_styles()
        
        # Привязываем обработчик закрытия окна
        self.r.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        s.configure('Header.TLabel', font=('Arial', 11, 'bold'))
        s.configure('Success.TLabel', foreground='green')
        s.configure('Error.TLabel', foreground='red')
        s.configure('Warning.TLabel', foreground='orange')
    
    def create_widgets(self):
        mc = ttk.Frame(self.r, padding="10")
        mc.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.r.columnconfigure(0, weight=1)
        self.r.rowconfigure(0, weight=1)
        mc.columnconfigure(1, weight=1)
        mc.rowconfigure(2, weight=1)
        ff = ttk.LabelFrame(mc, text="Файлы данных", padding="10")
        ff.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(ff, text="Файл маршрутов:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.rl = ttk.Label(ff, text=self.rf)
        self.rl.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Button(ff, text="Выбрать", command=self.select_routes_file).grid(row=0, column=2, padx=(0, 10))
        ttk.Label(ff, text="Файл норм:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.nl = ttk.Label(ff, text=self.nf)
        self.nl.grid(row=1, column=1, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Button(ff, text="Выбрать", command=self.select_norms_file).grid(row=1, column=2, padx=(0, 10), pady=(5, 0))
        self.lb = ttk.Button(ff, text="Загрузить данные", command=self.load_data)
        self.lb.grid(row=2, column=0, columnspan=3, pady=(10, 0))
        self.ls = ttk.Label(ff, text="", style='Success.TLabel')
        self.ls.grid(row=3, column=0, columnspan=3, pady=(5, 0))
        cf = ttk.LabelFrame(mc, text="Управление анализом", padding="10")
        cf.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        ttk.Label(cf, text="Участок:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.sv = tk.StringVar()
        self.sc = ttk.Combobox(cf, textvariable=self.sv, state='readonly', width=30)
        self.sc.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.sc.bind('<<ComboboxSelected>>', self.on_section_selected)
        self.ab = ttk.Button(cf, text="Анализировать участок", command=self.analyze_section, state='disabled')
        self.ab.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.fb = ttk.Button(cf, text="Фильтр локомотивов", command=self.open_locomotive_filter, state='disabled')
        self.fb.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.enb = ttk.Button(cf, text="Редактировать нормы", command=self.edit_norms, state='disabled')
        self.enb.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.fil = ttk.Label(cf, text="", style='Warning.TLabel')
        self.fil.grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Label(cf, text="Статистика:", style='Header.TLabel').grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        self.st = tk.Text(cf, width=35, height=10, wrap=tk.WORD)
        self.st.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        ss = ttk.Scrollbar(cf, orient='vertical', command=self.st.yview)
        ss.grid(row=7, column=1, sticky=(tk.N, tk.S), pady=(0, 10))
        self.st.configure(yscrollcommand=ss.set)
        ef = ttk.Frame(cf)
        ef.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self.eeb = ttk.Button(ef, text="Экспорт в Excel", command=self.export_to_excel, state='disabled')
        self.eeb.pack(side=tk.LEFT, padx=(0, 5))
        self.epb = ttk.Button(ef, text="Экспорт графика", command=self.export_plot, state='disabled')
        self.epb.pack(side=tk.LEFT)
        cf.rowconfigure(7, weight=1)
        pf = ttk.LabelFrame(mc, text="Визуализация", padding="10")
        pf.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.vb = ttk.Button(pf, text="Открыть график в браузере", command=self.open_plot_in_browser, state='disabled')
        self.vb.pack(pady=(0, 10))
        self.pi = tk.Text(pf, width=60, height=25, wrap=tk.WORD)
        self.pi.pack(fill=tk.BOTH, expand=True)
        lf = ttk.LabelFrame(mc, text="Журнал операций", padding="5")
        lf.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        self.lt = tk.Text(lf, height=8, wrap=tk.WORD)
        self.lt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lsc = ttk.Scrollbar(lf, orient='vertical', command=self.lt.yview)
        lsc.pack(side=tk.RIGHT, fill=tk.Y)
        self.lt.configure(yscrollcommand=lsc.set)
        self.log("Программа запущена. Загрузите файлы для начала анализа.")
    
    def log(self, msg, lvl='INFO'):
        ts = datetime.now().strftime("%H:%M:%S")
        self.lt.insert(tk.END, f"[{ts}] {lvl}: {msg}\n")
        self.lt.see(tk.END)
        if lvl == 'ERROR':
            self.lt.tag_add("error", f"end-2l", f"end-1l")
            self.lt.tag_config("error", foreground="red")
        elif lvl == 'SUCCESS':
            self.lt.tag_add("success", f"end-2l", f"end-1l")
            self.lt.tag_config("success", foreground="green")
        elif lvl == 'WARNING':
            self.lt.tag_add("warning", f"end-2l", f"end-1l")
            self.lt.tag_config("warning", foreground="orange")
    
    def select_routes_file(self):
        fn = filedialog.askopenfilename(title="Выберите файл маршрутов", filetypes=[("Excel files", "*.xlsx *.xls")])
        if fn:
            self.rf = fn
            self.rl.config(text=os.path.basename(fn))
            self.log(f"Выбран файл маршрутов: {os.path.basename(fn)}")
    
    def select_norms_file(self):
        fn = filedialog.askopenfilename(title="Выберите файл норм", filetypes=[("Excel files", "*.xlsx *.xls")])
        if fn:
            self.nf = fn
            self.nl.config(text=os.path.basename(fn))
            self.log(f"Выбран файл норм: {os.path.basename(fn)}")
    
    def load_data(self):
        self.lb.config(state='disabled')
        threading.Thread(target=self._load_data_thread).start()
    
    def _load_data_thread(self):
        self.a = InteractiveNormsAnalyzer(self.rf, self.nf)
        rl = self.a.load_data()
        nl = self.a.load_norms()
        self.r.after(0, self._update_load_status, rl and nl)
    
    def _update_load_status(self, suc):
        if suc:
            self.sc['values'] = self.a.get_sections_list()
            self.ab['state'] = 'normal'
            self.fb['state'] = 'normal'
            self.enb['state'] = 'normal'
            self.lf = LocomotiveFilter(self.a.rdf)
            self.log("Данные загружены успешно", 'SUCCESS')
            self.ls.config(text="Данные загружены", style='Success.TLabel')
        else:
            self.log("Ошибка загрузки данных", 'ERROR')
            self.ls.config(text="Ошибка загрузки", style='Error.TLabel')
        self.lb.config(state='normal')
    
    def on_section_selected(self, e=None):
        self.analyze_section()
    
    def analyze_section(self):
        sec = self.sv.get()
        if not sec:
            return
        
        # Передаем фильтр локомотивов и настройки коэффициентов в анализатор
        if hasattr(self, 'lf') and self.lf:
            # Создаем временный анализатор для учета фильтров
            sr = self.a.rdf[self.a.rdf['Наименование участка'] == sec].copy()
            if sr.empty:
                messagebox.showerror("Ошибка", "Нет маршрутов для участка")
                return
            
            ra, nf = self.a.analyze_section_with_filters(
                sec, sr, self.a.nd[sec], 
                self.lf, 
                getattr(self, 'cm', None), 
                getattr(self, 'uc', False)
            )
            
            if ra is None or ra.empty:
                messagebox.showwarning("Предупреждение", "Нет данных после применения фильтров")
                return
            
            fig = self.a.create_interactive_plot(sec, ra, nf)
        else:
            fig, st, err = self.a.analyze_single_section(sec)
            if err:
                messagebox.showerror("Ошибка", err)
                return
        
        self.cp = fig
        self.th = tempfile.mktemp(suffix='.html')
        plot(fig, filename=self.th, auto_open=False)
        self.vb['state'] = 'normal'
        self.eeb['state'] = 'normal'
        self.epb['state'] = 'normal'
        
        # Пересчитываем статистику для отфильтрованных данных
        if hasattr(self, 'lf') and self.lf:
            vr = ra[ra['Статус'] != 'Не определен']
            ds = {
                'economy_strong': len(vr[vr['Отклонение, %'] >= 30]),
                'economy_medium': len(vr[(vr['Отклонение, %'] >= 20) & (vr['Отклонение, %'] < 30)]),
                'economy_weak': len(vr[(vr['Отклонение, %'] >= 5) & (vr['Отклонение, %'] < 20)]),
                'normal': len(vr[(vr['Отклонение, %'] >= -5) & (vr['Отклонение, %'] < 5)]),
                'overrun_weak': len(vr[(vr['Отклонение, %'] >= -20) & (vr['Отклонение, %'] < -5)]),
                'overrun_medium': len(vr[(vr['Отклонение, %'] >= -30) & (vr['Отклонение, %'] < -20)]),
                'overrun_strong': len(vr[vr['Отклонение, %'] < -30])
            }
            st = {
                'total': len(ra),
                'processed': len(vr),
                'economy': ds['economy_strong'] + ds['economy_medium'] + ds['economy_weak'],
                'normal': ds['normal'],
                'overrun': ds['overrun_weak'] + ds['overrun_medium'] + ds['overrun_strong'],
                'mean_deviation': vr['Отклонение, %'].mean() if len(vr) > 0 else 0,
                'detailed_stats': ds
            }
        
        self.update_statistics(st)
        self.update_plot_info(sec, st)
        self.log(f"Анализ участка {sec} завершен")
    
    def open_locomotive_filter(self):
        d = LocomotiveSelectorDialog(self.r, self.lf, self.cm)
        self.r.wait_window(d.d)
        if d.res:
            self.uc = d.res['use_coefficients']
            self.elw = d.res.get('exclude_low_work', False)
            self.cm = d.res['coefficients_manager']
            self.fil.config(text=f"Выбрано локомотивов: {len(d.res['selected_locomotives'])}")
            self.analyze_section()
    
    def edit_norms(self):
        sec = self.sv.get()
        if not sec:
            messagebox.showwarning("Предупреждение", "Выберите участок для редактирования норм")
            return
        en = self.a.nd.get(sec)
        ed = NormEditorDialog(self.r, sec, en)
        self.r.wait_window(ed.d)
        if ed.res == 'apply' and ed.ed:
            self.a.nd[sec] = ed.ed
            self.show_comparison(sec, en, ed.ed)
            self.analyze_section()
        else:
            self.log("Редактирование норм отменено")
    
    def show_comparison(self, sec, on, en):
        sr = self.a.rdf[self.a.rdf['Наименование участка'] == sec].copy()
        if sr.empty:
            return
        comp = NormComparator.compare_norms(on, en, sr)
        cw = tk.Toplevel(self.r)
        cw.title(f"Сравнение норм - {sec}")
        cw.geometry("600x500")
        cw.transient(self.r)
        tl = ttk.Label(cw, text="Сравнение результатов анализа", font=('Arial', 12, 'bold'))
        tl.pack(pady=10)
        tw = tk.Text(cw, wrap=tk.WORD, padx=10, pady=10)
        tw.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        ct = f"Участок: {sec}\n" + "=" * 50 + "\n\n"
        if 'original' in comp and comp['original']:
            ct += "ИСХОДНЫЕ НОРМЫ:\n" + "-" * 30 + "\n"
            o = comp['original']
            ct += f"Обработано маршрутов: {o['processed']}\n"
            ct += f"Экономия сильная (+30% и более): {o['economy_strong']}\n"
            ct += f"Экономия средняя (+20% до +30%): {o['economy_medium']}\n"
            ct += f"Экономия слабая (+5% до +20%): {o['economy_weak']}\n"
            ct += f"Норма (-5% до +5%): {o['normal']}\n"
            ct += f"Перерасход слабый (-5% до -20%): {o['overrun_weak']}\n"
            ct += f"Перерасход средний (-20% до -30%): {o['overrun_medium']}\n"
            ct += f"Перерасход сильный (-30% и менее): {o['overrun_strong']}\n"
            ct += f"Среднее отклонение: {o['mean_deviation']:.1f}%\n\n"
        if 'edited' in comp and comp['edited']:
            ct += "АКТУАЛИЗИРОВАННЫЕ НОРМЫ:\n" + "-" * 30 + "\n"
            e = comp['edited']
            ct += f"Обработано маршрутов: {e['processed']}\n"
            ct += f"Экономия сильная (+30% и более): {e['economy_strong']}\n"
            ct += f"Экономия средняя (+20% до +30%): {e['economy_medium']}\n"
            ct += f"Экономия слабая (+5% до +20%): {e['economy_weak']}\n"
            ct += f"Норма (-5% до +5%): {e['normal']}\n"
            ct += f"Перерасход слабый (-5% до -20%): {e['overrun_weak']}\n"
            ct += f"Перерасход средний (-20% до -30%): {e['overrun_medium']}\n"
            ct += f"Перерасход сильный (-30% и менее): {e['overrun_strong']}\n"
            ct += f"Среднее отклонение: {e['mean_deviation']:.1f}%\n\n"
        if 'differences' in comp and comp['differences']:
            ct += "ИЗМЕНЕНИЯ:\n" + "-" * 30 + "\n"
            d = comp['differences']
            for k, v in d.items():
                if k == 'mean_deviation':
                    ct += f"Изменение среднего отклонения: {v:+.1f}%\n"
                elif k == 'normal':
                    ct += f"Изменение маршрутов в норме: {v:+d}\n"
                elif 'economy' in k:
                    ct += f"Изменение экономии ({k}): {v:+d}\n"
                elif 'overrun' in k:
                    ct += f"Изменение перерасхода ({k}): {v:+d}\n"
        tw.insert(1.0, ct)
        tw.config(state='disabled')
        ttk.Button(cw, text="Закрыть", command=cw.destroy).pack(pady=10)
    
    def update_statistics(self, st):
        self.st.delete(1.0, tk.END)
        txt = f"Всего маршрутов: {st['total']}\n"
        txt += f"Обработано: {st['processed']}\n"
        txt += f"Экономия: {st['economy']} ({st['economy']/st['processed']*100 if st['processed'] > 0 else 0:.1f}%)\n"
        txt += f"В норме: {st['normal']} ({st['normal']/st['processed']*100 if st['processed'] > 0 else 0:.1f}%)\n"
        txt += f"Перерасход: {st['overrun']} ({st['overrun']/st['processed']*100 if st['processed'] > 0 else 0:.1f}%)\n"
        txt += f"Среднее отклонение: {st['mean_deviation']:.1f}%\n\n"
        txt += "Детально:\n"
        dt = st['detailed_stats']
        for cat in ['economy_strong', 'economy_medium', 'economy_weak', 'normal', 'overrun_weak', 'overrun_medium', 'overrun_strong']:
            cnt = dt.get(cat, 0)
            pct = cnt / st['processed'] * 100 if st['processed'] > 0 else 0
            cn = cat.replace('_', ' ').title()
            if cnt > 0:
                txt += f"{cn}: {pct:.1f}%\n"
        self.st.insert(1.0, txt)
    
    def update_plot_info(self, sec, st):
        self.pi.delete(1.0, tk.END)
        txt = "ИНТЕРАКТИВНЫЙ ГРАФИК\n" + "=" * 40 + "\n\n"
        txt += f"Участок: {sec}\n\n"
        txt += "Возможности графика:\n"
        txt += "• Наведите курсор на точку для просмотра подробной информации\n"
        txt += "• Используйте колесо мыши для масштабирования\n"
        txt += "• Зажмите левую кнопку мыши для перемещения\n"
        txt += "• Двойной клик для сброса масштаба\n"
        txt += "• Клик по легенде для скрытия/показа элементов\n\n"
        txt += "Верхний график:\n"
        txt += "• Линии - кривые норм\n"
        txt += "• Квадраты - опорные точки норм\n"
        txt += "• Круги - фактические значения маршрутов\n\n"
        txt += "Нижний график:\n"
        txt += "• Зеленые точки - экономия (< -5%)\n"
        txt += "• Желтые точки - в пределах нормы (±5%)\n"
        txt += "• Красные точки - перерасход (> 5%)\n"
        txt += "• Оранжевые линии - границы допустимых отклонений\n\n"
        txt += "Для просмотра в полноэкранном режиме\nнажмите 'Открыть график в браузере'"
        self.pi.insert(1.0, txt)
    
    def open_plot_in_browser(self):
        if self.th and os.path.exists(self.th):
            webbrowser.open(f'file://{os.path.abspath(self.th)}')
            self.log("График открыт в браузере")
        else:
            messagebox.showwarning("Предупреждение", "График не найден. Выполните анализ участка.")
    
    def export_to_excel(self):
        if not self.a or not self.a.ar:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if fn:
            try:
                sec = self.sv.get()
                if sec in self.a.ar:
                    df = self.a.ar[sec]['routes']
                    df.to_excel(fn, index=False)
                    self.log(f"Данные экспортированы в {os.path.basename(fn)}", 'SUCCESS')
                    messagebox.showinfo("Успех", "Данные успешно экспортированы")
            except Exception as e:
                self.log(f"Ошибка экспорта: {str(e)}", 'ERROR')
                messagebox.showerror("Ошибка", f"Не удалось экспортировать данные:\n{str(e)}")
    
    def export_plot(self):
        if not self.cp:
            messagebox.showwarning("Предупреждение", "Нет графика для экспорта")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html"), ("PNG files", "*.png")])
        if fn:
            try:
                if fn.endswith('.html'):
                    self.cp.write_html(fn)
                else:
                    self.cp.write_image(fn)
                self.log(f"График экспортирован в {os.path.basename(fn)}", 'SUCCESS')
                messagebox.showinfo("Успех", "График успешно экспортирован")
            except Exception as e:
                self.log(f"Ошибка экспорта: {str(e)}", 'ERROR')
                messagebox.showerror("Ошибка", f"Не удалось экспортировать график:\n{str(e)}")
    
    def on_closing(self):
        """Обработчик закрытия окна - ИСПРАВЛЕНО"""
        # Удаляем временные файлы
        if self.th and os.path.exists(self.th):
            try:
                os.remove(self.th)
            except:
                pass
        
        # ВАЖНО: Закрываем все открытые диалоги и уничтожаем главное окно
        for widget in self.r.winfo_children():
            if isinstance(widget, tk.Toplevel):
                widget.destroy()
        
        # Уничтожаем главное окно - это исправляет проблему с незакрывающейся программой
        self.r.quit()
        self.r.destroy()