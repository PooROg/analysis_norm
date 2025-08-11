# Файл: gui/interface.py
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
    def __init__(self, root):
        self.root = root
        self.root.title("Анализатор норм расхода электроэнергии РЖД")
        self.root.geometry("1200x700")
        self.analyzer = None
        self.current_plot = None
        self.temp_html_file = None
        self.locomotive_filter = None
        self.coefficients_manager = LocomotiveCoefficientsManager()
        self.use_coefficients = False
        self.routes_file = 'Processed_Routes.xlsx'
        self.norms_file = 'Нормы участков.xlsx'
        self.create_widgets()
        self.setup_styles()
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Arial', 14, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 11, 'bold'))
        style.configure('Success.TLabel', foreground='green')
        style.configure('Error.TLabel', foreground='red')
        style.configure('Warning.TLabel', foreground='orange')
    
    def create_widgets(self):
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=1)
        main_container.rowconfigure(2, weight=1)
        files_frame = ttk.LabelFrame(main_container, text="Файлы данных", padding="10")
        files_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(files_frame, text="Файл маршрутов:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.routes_label = ttk.Label(files_frame, text=self.routes_file)
        self.routes_label.grid(row=0, column=1, sticky=tk.W, padx=(0, 10))
        ttk.Button(files_frame, text="Выбрать", command=self.select_routes_file).grid(row=0, column=2, padx=(0, 10))
        ttk.Label(files_frame, text="Файл норм:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.norms_label = ttk.Label(files_frame, text=self.norms_file)
        self.norms_label.grid(row=1, column=1, sticky=tk.W, padx=(0, 10), pady=(5, 0))
        ttk.Button(files_frame, text="Выбрать", command=self.select_norms_file).grid(row=1, column=2, padx=(0, 10), pady=(5, 0))
        self.load_button = ttk.Button(files_frame, text="Загрузить данные", command=self.load_data)
        self.load_button.grid(row=2, column=0, columnspan=3, pady=(10, 0))
        self.load_status = ttk.Label(files_frame, text="", style='Success.TLabel')
        self.load_status.grid(row=3, column=0, columnspan=3, pady=(5, 0))
        control_frame = ttk.LabelFrame(main_container, text="Управление анализом", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        ttk.Label(control_frame, text="Участок:", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.section_var = tk.StringVar()
        self.section_combo = ttk.Combobox(control_frame, textvariable=self.section_var, state='readonly', width=30)
        self.section_combo.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.section_combo.bind('<<ComboboxSelected>>', self.on_section_selected)
        self.analyze_button = ttk.Button(control_frame, text="Анализировать участок", command=self.analyze_section, state='disabled')
        self.analyze_button.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.filter_button = ttk.Button(control_frame, text="Фильтр локомотивов", command=self.open_locomotive_filter, state='disabled')
        self.filter_button.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.edit_norms_button = ttk.Button(control_frame, text="Редактировать нормы", command=self.edit_norms, state='disabled')
        self.edit_norms_button.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.filter_info_label = ttk.Label(control_frame, text="", style='Warning.TLabel')
        self.filter_info_label.grid(row=5, column=0, sticky=tk.W, pady=(0, 5))
        ttk.Label(control_frame, text="Статистика:", style='Header.TLabel').grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        self.stats_text = tk.Text(control_frame, width=35, height=10, wrap=tk.WORD)
        self.stats_text.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        stats_scroll = ttk.Scrollbar(control_frame, orient='vertical', command=self.stats_text.yview)
        stats_scroll.grid(row=7, column=1, sticky=(tk.N, tk.S), pady=(0, 10))
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        export_frame = ttk.Frame(control_frame)
        export_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        self.export_excel_btn = ttk.Button(export_frame, text="Экспорт в Excel", command=self.export_to_excel, state='disabled')
        self.export_excel_btn.pack(side=tk.LEFT, padx=(0, 5))
        self.export_plot_btn = ttk.Button(export_frame, text="Экспорт графика", command=self.export_plot, state='disabled')
        self.export_plot_btn.pack(side=tk.LEFT)
        control_frame.rowconfigure(7, weight=1)
        plot_frame = ttk.LabelFrame(main_container, text="Визуализация", padding="10")
        plot_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.view_button = ttk.Button(plot_frame, text="Открыть график в браузере", command=self.open_plot_in_browser, state='disabled')
        self.view_button.pack(pady=(0, 10))
        self.plot_info = tk.Text(plot_frame, width=60, height=25, wrap=tk.WORD)
        self.plot_info.pack(fill=tk.BOTH, expand=True)
        log_frame = ttk.LabelFrame(main_container, text="Журнал операций", padding="5")
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log("Программа запущена. Загрузите файлы для начала анализа.")
    
    def log(self, message, level='INFO'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {level}: {message}\n")
        self.log_text.see(tk.END)
        if level == 'ERROR':
            self.log_text.tag_add("error", f"end-2l", f"end-1l")
            self.log_text.tag_config("error", foreground="red")
        elif level == 'SUCCESS':
            self.log_text.tag_add("success", f"end-2l", f"end-1l")
            self.log_text.tag_config("success", foreground="green")
        elif level == 'WARNING':
            self.log_text.tag_add("warning", f"end-2l", f"end-1l")
            self.log_text.tag_config("warning", foreground="orange")
    
    def select_routes_file(self):
        filename = filedialog.askopenfilename(title="Выберите файл маршрутов", filetypes=[("Excel files", "*.xlsx *.xls")])
        if filename:
            self.routes_file = filename
            self.routes_label.config(text=os.path.basename(filename))
            self.log(f"Выбран файл маршрутов: {os.path.basename(filename)}")
    
    def select_norms_file(self):
        filename = filedialog.askopenfilename(title="Выберите файл норм", filetypes=[("Excel files", "*.xlsx *.xls")])
        if filename:
            self.norms_file = filename
            self.norms_label.config(text=os.path.basename(filename))
            self.log(f"Выбран файл норм: {os.path.basename(filename)}")
    
    def load_data(self):
        self.load_button.config(state='disabled')
        threading.Thread(target=self._load_data_thread).start()
    
    def _load_data_thread(self):
        self.analyzer = InteractiveNormsAnalyzer(self.routes_file, self.norms_file)
        routes_loaded = self.analyzer.load_data()
        norms_loaded = self.analyzer.load_norms()
        self.root.after(0, self._update_load_status, routes_loaded and norms_loaded)
    
    def _update_load_status(self, success):
        if success:
            self.section_combo['values'] = self.analyzer.get_sections_list()
            self.analyze_button['state'] = 'normal'
            self.filter_button['state'] = 'normal'
            self.edit_norms_button['state'] = 'normal'
            self.locomotive_filter = LocomotiveFilter(self.analyzer.routes_df)
            self.log("Данные загружены успешно", 'SUCCESS')
            self.load_status.config(text="Данные загружены", style='Success.TLabel')
        else:
            self.log("Ошибка загрузки данных", 'ERROR')
            self.load_status.config(text="Ошибка загрузки", style='Error.TLabel')
        self.load_button.config(state='normal')
    
    def on_section_selected(self, event=None):
        self.analyze_section()
    
    def analyze_section(self):
        section = self.section_var.get()
        if not section:
            return
        fig, stats, error = self.analyzer.analyze_single_section(section)
        if error:
            messagebox.showerror("Ошибка", error)
            return
        self.current_plot = fig
        self.temp_html_file = tempfile.mktemp(suffix='.html')
        plot(fig, filename=self.temp_html_file, auto_open=False)
        self.view_button['state'] = 'normal'
        self.export_excel_btn['state'] = 'normal'
        self.export_plot_btn['state'] = 'normal'
        self.update_statistics(stats)
        self.update_plot_info(section, stats)
        self.log(f"Анализ участка {section} завершен")
    
    def open_locomotive_filter(self):
        dialog = LocomotiveSelectorDialog(self.root, self.locomotive_filter, self.coefficients_manager)
        self.root.wait_window(dialog.dialog)
        if dialog.result:
            self.use_coefficients = dialog.result['use_coefficients']
            self.coefficients_manager = dialog.result['coefficients_manager']
            self.filter_info_label.config(text=f"Выбрано локомотивов: {len(dialog.result['selected_locomотives'])}")
            self.analyze_section()
    
    def edit_norms(self):
        section = self.section_var.get()
        if not section:
            messagebox.showwarning("Предупреждение", "Выберите участок для редактирования норм")
            return
        existing_norms = self.analyzer.norms_data.get(section)
        editor = NormEditorDialog(self.root, section, existing_norms)
        self.root.wait_window(editor.dialog)
        if editor.result == 'apply' and editor.edited_norms:
            self.analyzer.norms_data[section] = editor.edited_norms
            self.show_comparison(section, existing_norms, editor.edited_norms)
            self.analyze_section()
        else:
            self.log("Редактирование норм отменено")
    
    def show_comparison(self, section, original_norms, edited_norms):
        section_routes = self.analyzer.routes_df[self.analyzer.routes_df['Наименование участка'] == section].copy()
        if section_routes.empty:
            return
        comparison = NormComparator.compare_norms(original_norms, edited_norms, section_routes)
        comp_window = tk.Toplevel(self.root)
        comp_window.title(f"Сравнение норм - {section}")
        comp_window.geometry("600x500")
        comp_window.transient(self.root)
        title_label = ttk.Label(comp_window, text="Сравнение результатов анализа", font=('Arial', 12, 'bold'))
        title_label.pack(pady=10)
        text_widget = tk.Text(comp_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        comp_text = f"Участок: {section}\n" + "=" * 50 + "\n\n"
        if 'original' in comparison and comparison['original']:
            comp_text += "ИСХОДНЫЕ НОРМЫ:\n" + "-" * 30 + "\n"
            orig = comparison['original']
            comp_text += f"Обработано маршрутов: {orig['processed']}\n"
            comp_text += f"Экономия сильная (+30% и более): {orig['economy_strong']}\n"
            comp_text += f"Экономия средняя (+20% до +30%): {orig['economy_medium']}\n"
            comp_text += f"Экономия слабая (+5% до +20%): {orig['economy_weak']}\n"
            comp_text += f"Норма (-5% до +5%): {orig['normal']}\n"
            comp_text += f"Перерасход слабый (-5% до -20%): {orig['overrun_weak']}\n"
            comp_text += f"Перерасход средний (-20% до -30%): {orig['overrun_medium']}\n"
            comp_text += f"Перерасход сильный (-30% и менее): {orig['overrun_strong']}\n"
            comp_text += f"Среднее отклонение: {orig['mean_deviation']:.1f}%\n\n"
        if 'edited' in comparison and comparison['edited']:
            comp_text += "АКТУАЛИЗИРОВАННЫЕ НОРМЫ:\n" + "-" * 30 + "\n"
            edited = comparison['edited']
            comp_text += f"Обработано маршрутов: {edited['processed']}\n"
            comp_text += f"Экономия сильная (+30% и более): {edited['economy_strong']}\n"
            comp_text += f"Экономия средняя (+20% до +30%): {edited['economy_medium']}\n"
            comp_text += f"Экономия слабая (+5% до +20%): {edited['economy_weak']}\n"
            comp_text += f"Норма (-5% до +5%): {edited['normal']}\n"
            comp_text += f"Перерасход слабый (-5% до -20%): {edited['overrun_weak']}\n"
            comp_text += f"Перерасход средний (-20% до -30%): {edited['overrun_medium']}\n"
            comp_text += f"Перерасход сильный (-30% и менее): {edited['overrun_strong']}\n"
            comp_text += f"Среднее отклонение: {edited['mean_deviation']:.1f}%\n\n"
        if 'differences' in comparison and comparison['differences']:
            comp_text += "ИЗМЕНЕНИЯ:\n" + "-" * 30 + "\n"
            diff = comparison['differences']
            for key, value in diff.items():
                if key == 'mean_deviation':
                    comp_text += f"Изменение среднего отклонения: {value:+.1f}%\n"
                elif key == 'normal':
                    comp_text += f"Изменение маршрутов в норме: {value:+d}\n"
                elif 'economy' in key:
                    comp_text += f"Изменение экономии ({key}): {value:+d}\n"
                elif 'overrun' in key:
                    comp_text += f"Изменение перерасхода ({key}): {value:+d}\n"
        text_widget.insert(1.0, comp_text)
        text_widget.config(state='disabled')
        ttk.Button(comp_window, text="Закрыть", command=comp_window.destroy).pack(pady=10)
    
    def update_statistics(self, stats):
        self.stats_text.delete(1.0, tk.END)
        text = f"Всего маршрутов: {stats['total']}\n"
        text += f"Обработано: {stats['processed']}\n"
        text += f"Экономия: {stats['economy']} ({stats['economy']/stats['processed']*100 if stats['processed'] > 0 else 0:.1f}%)\n"
        text += f"В норме: {stats['normal']} ({stats['normal']/stats['processed']*100 if stats['processed'] > 0 else 0:.1f}%)\n"
        text += f"Перерасход: {stats['overrun']} ({stats['overrun']/stats['processed']*100 if stats['processed'] > 0 else 0:.1f}%)\n"
        text += f"Среднее отклонение: {stats['mean_deviation']:.1f}%\n\n"
        text += "Детально:\n"
        detailed = stats['detailed_stats']
        for category in ['economy_strong', 'economy_medium', 'economy_weak', 'normal', 'overrun_weak', 'overrun_medium', 'overrun_strong']:
            count = detailed.get(category, 0)
            percent = count / stats['processed'] * 100 if stats['processed'] > 0 else 0
            category_name = category.replace('_', ' ').title()
            if count > 0:
                text += f"{category_name}: {percent:.1f}%\n"
        self.stats_text.insert(1.0, text)
    
    def update_plot_info(self, section, stats):
        self.plot_info.delete(1.0, tk.END)
        text = "ИНТЕРАКТИВНЫЙ ГРАФИК\n" + "=" * 40 + "\n\n"
        text += f"Участок: {section}\n\n"
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
        text += "• Зеленые точки - экономия (< -5%)\n"
        text += "• Желтые точки - в пределах нормы (±5%)\n"
        text += "• Красные точки - перерасход (> 5%)\n"
        text += "• Оранжевые линии - границы допустимых отклонений\n\n"
        text += "Для просмотра в полноэкранном режиме\nнажмите 'Открыть график в браузере'"
        self.plot_info.insert(1.0, text)
    
    def open_plot_in_browser(self):
        if self.temp_html_file and os.path.exists(self.temp_html_file):
            webbrowser.open(f'file://{os.path.abspath(self.temp_html_file)}')
            self.log("График открыт в браузере")
        else:
            messagebox.showwarning("Предупреждение", "График не найден. Выполните анализ участка.")
    
    def export_to_excel(self):
        if not self.analyzer or not self.analyzer.analysis_results:
            messagebox.showwarning("Предупреждение", "Нет данных для экспорта")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
        if filename:
            try:
                # Предполагаем, что анализ_results имеет routes
                section = self.section_var.get()
                if section in self.analyzer.analysis_results:
                    df = self.analyzer.analysis_results[section]['routes']
                    df.to_excel(filename, index=False)
                    self.log(f"Данные экспортированы в {os.path.basename(filename)}", 'SUCCESS')
                    messagebox.showinfo("Успех", "Данные успешно экспортированы")
            except Exception as e:
                self.log(f"Ошибка экспорта: {str(e)}", 'ERROR')
                messagebox.showerror("Ошибка", f"Не удалось экспортировать данные:\n{str(e)}")
    
    def export_plot(self):
        if not self.current_plot:
            messagebox.showwarning("Предупреждение", "Нет графика для экспорта")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".html", filetypes=[("HTML files", "*.html"), ("PNG files", "*.png")])
        if filename:
            try:
                if filename.endswith('.html'):
                    self.current_plot.write_html(filename)
                else:
                    self.current_plot.write_image(filename)
                self.log(f"График экспортирован в {os.path.basename(filename)}", 'SUCCESS')
                messagebox.showinfo("Успех", "График успешно экспортирован")
            except Exception as e:
                self.log(f"Ошибка экспорта: {str(e)}", 'ERROR')
                messagebox.showerror("Ошибка", f"Не удалось экспортировать график:\n{str(e)}")
    
    def on_closing(self):
        if self.temp_html_file and os.path.exists(self.temp_html_file):
            try:
                os.remove(self.temp_html_file)
            except:
                pass
        self.root.destroy()