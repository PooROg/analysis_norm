# dialogs/selector.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Tuple

from core.coefficients import LocomotiveCoefficientsManager


class LocomotiveSelectorDialog:
    """Диалог выбора локомотивов для анализа."""

    def __init__(self, p, lf, cm: LocomotiveCoefficientsManager | None = None):
        self.p = p
        self.f = lf
        self.cm = cm or LocomotiveCoefficientsManager()

        # Состояние
        self.res = None
        self.uc = tk.BooleanVar(value=False)                # применять коэффициенты
        self.exclude_low_work = tk.BooleanVar(value=False)  # исключать локомотивы с низкой работой

        # Деревья и состояния выбора
        self.trees: Dict[str, ttk.Treeview] = {}                   # серия -> дерево
        self.series_vars: Dict[str, tk.BooleanVar] = {}            # серия -> чекбокс "вся серия"
        self.selected: set[Tuple[str, int]] = set()                # выбранные (серия, номер)

        # UI
        self.d = tk.Toplevel(p)
        self.d.title("Выбор локомотивов и коэффициентов")
        self.d.geometry("900x700")
        self.d.transient(p)
        self.d.grab_set()

        self.create_widgets()          # публичный метод
        self.load_current_selection()  # синхронизация с фильтром (по умолчанию все выбраны)
        self.center_window()

    # -------------------- Вспомогательные методы UI --------------------

    def center_window(self):
        self.d.update_idletasks()
        w, h = self.d.winfo_width(), self.d.winfo_height()
        x = (self.d.winfo_screenwidth() // 2) - (w // 2)
        y = (self.d.winfo_screenheight() // 2) - (h // 2)
        self.d.geometry(f"{w}x{h}+{x}+{y}")

    def create_widgets(self):
        """Создание основных виджетов диалога (публичный API)."""
        mf = ttk.Frame(self.d, padding="10")
        mf.pack(fill=tk.BOTH, expand=True)

        # Блок коэффициентов
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
            command=self.on_use_coefficients_changed,
        )
        self.ucc.pack(pady=(5, 0))

        self.elwc = ttk.Checkbutton(
            cf,
            text="Исключить при анализе локомотивы с менее 200 10тыс.ткм брутто",
            variable=self.exclude_low_work,
        )
        self.elwc.pack(pady=(5, 0))

        # Блок выбора локомотивов
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

        # Подсказка и кнопки действий
        inf = ttk.Frame(mf)
        inf.pack(fill=tk.X)
        hint = (
            "• Выберите локомотивы, которые нужно включить в анализ\n"
            "• Загрузите файл коэффициентов для учета индивидуальных характеристик\n"
            "• При включенных коэффициентах фактический расход будет скорректирован"
        )
        ttk.Label(inf, text=hint, foreground="gray").pack(pady=(0, 10))
        bf = ttk.Frame(inf)
        bf.pack()
        ttk.Button(bf, text="Применить", command=self.apply_selection).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(bf, text="Отмена", command=self.cancel).pack(side=tk.LEFT)

    def create_series_tabs(self):
        """Создает вкладки по сериям с Treeview (быстро и масштабируемо)."""
        # Очистка старых вкладок
        for tab in list(self.nb.tabs()):
            self.nb.forget(tab)
        self.trees.clear()
        self.series_vars.clear()

        by_series = self.f.get_locomotives_by_series()
        for series in sorted(by_series.keys()):
            numbers = sorted(by_series[series])
            self._build_series_tab(series, numbers)

        self.update_selection_count()

    def _build_series_tab(self, series: str, numbers: List[int]):
        """Строит вкладку серии и наполняет её данными."""
        frame = ttk.Frame(self.nb)
        self.nb.add(frame, text=series)

        # Верхний чекбокс "вся серия"
        sv = tk.BooleanVar(value=True)
        self.series_vars[series] = sv
        ttk.Checkbutton(
            frame,
            text=f"Выбрать всю серию {series}",
            variable=sv,
            command=lambda se=series: self.toggle_series(se),
        ).pack(anchor=tk.W, padx=5, pady=(5, 5))

        # Дерево с данными
        columns = ("idx", "sel", "num", "coef")
        tree = ttk.Treeview(frame, columns=columns, show="headings", height=18)
        self.trees[series] = tree

        tree.heading("idx", text="№")
        tree.heading("sel", text="Выбор")
        tree.heading("num", text="Номер")
        tree.heading("coef", text="Коэффициент")

        tree.column("idx", width=60, anchor=tk.E, stretch=False)
        tree.column("sel", width=80, anchor=tk.CENTER, stretch=False)
        tree.column("num", width=120, anchor=tk.CENTER, stretch=False)
        tree.column("coef", width=140, anchor=tk.CENTER, stretch=False)

        # Цветовые теги (целая строка окрашивается — достаточно для визуального различия)
        tree.tag_configure("above_norm", foreground="red")
        tree.tag_configure("below_norm", foreground="green")
        tree.tag_configure("norm", foreground="black")

        # Скроллбар
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Заполнение строк
        for i, n in enumerate(numbers, start=1):
            coef = self.cm.get_coefficient(series, n)
            coef_text = f"{coef:.3f}" if coef != 1.0 else "-"
            tag = "above_norm" if coef > 1.05 else "below_norm" if coef < 0.95 else "norm"
            # По умолчанию ставим галочку — дальше load_current_selection приведёт к реальному состоянию
            tree.insert("", "end", iid=f"{series}:{n}", values=(i, "✓", n, coef_text), tags=(tag,))

        # Обработка кликов: переключаем выбор по клику на колонке "Выбор" и по пробелу
        tree.bind("<Button-1>", lambda e, s=series: self._on_tree_click(e, s), add="+")
        tree.bind("<space>", lambda e, s=series: self._on_tree_space(e, s), add="+")

    # -------------------- Обработчики Treeview --------------------

    def _on_tree_click(self, event, series: str):
        tree = self.trees[series]
        region = tree.identify("region", event.x, event.y)
        if region != "cell":
            return  # клики вне ячеек игнорируем
        col = tree.identify_column(event.x)
        if col != "#2":  # реагируем только на колонку "Выбор"
            return
        iid = tree.identify_row(event.y)
        if not iid:
            return
        number = int(tree.set(iid, "num"))
        current = (series, number) in self.selected
        self._set_row_selected(series, number, not current)

    def _on_tree_space(self, event, series: str):
        """Пробел переключает текущую строку."""
        tree = self.trees[series]
        focus = tree.focus()
        if not focus:
            return
        number = int(tree.set(focus, "num"))
        current = (series, number) in self.selected
        self._set_row_selected(series, number, not current)

    # -------------------- Логика выбора --------------------

    def _set_row_selected(self, series: str, number: int, is_selected: bool):
        """Устанавливает состояние выбора для одной строки и обновляет UI."""
        key = (series, number)
        if is_selected:
            self.selected.add(key)
        else:
            self.selected.discard(key)

        tree = self.trees[series]
        iid = f"{series}:{number}"
        if iid in tree.get_children(""):
            tree.set(iid, "sel", "✓" if is_selected else "")

        # Обновляем чекбокс серии и счётчик
        self._recalculate_series_checkbox(series)
        self.update_selection_count()

    def _recalculate_series_checkbox(self, series: str):
        """Обновляет чекбокс серии: отмечен, если все локомотивы этой серии выбраны."""
        tree = self.trees[series]
        items = tree.get_children("")
        all_selected = True
        for iid in items:
            n = int(tree.set(iid, "num"))
            if (series, n) not in self.selected:
                all_selected = False
                break
        self.series_vars[series].set(all_selected)

    def toggle_series(self, series: str):
        """Отметить/снять всю серию согласно чекбоксу серии."""
        want = self.series_vars[series].get()
        tree = self.trees[series]
        for iid in tree.get_children(""):
            n = int(tree.set(iid, "num"))
            self._set_row_selected(series, n, want)

    def select_all(self):
        """Выбрать все локомотивы (публичный API)."""
        self.selected.clear()
        # Массовое включение быстрее через один проход
        for series, tree in self.trees.items():
            for iid in tree.get_children(""):
                n = int(tree.set(iid, "num"))
                self.selected.add((series, n))
                tree.set(iid, "sel", "✓")
            self.series_vars[series].set(True)
        self.update_selection_count()

    def deselect_all(self):
        """Снять выбор со всех локомотивов (публичный API)."""
        self.selected.clear()
        for series, tree in self.trees.items():
            for iid in tree.get_children(""):
                tree.set(iid, "sel", "")
            self.series_vars[series].set(False)
        self.update_selection_count()

    def invert_selection(self):
        """Инвертировать выбор по всем сериям."""
        new_selected: set[Tuple[str, int]] = set()
        for series, tree in self.trees.items():
            for iid in tree.get_children(""):
                n = int(tree.set(iid, "num"))
                key = (series, n)
                now = key in self.selected
                will = not now
                if will:
                    new_selected.add(key)
                tree.set(iid, "sel", "✓" if will else "")
            self.series_vars[series].set(self._series_all_selected(series, new_selected))
        self.selected = new_selected
        self.update_selection_count()

    def _series_all_selected(self, series: str, selected_set: set[Tuple[str, int]]) -> bool:
        """Проверяет, выбраны ли все локомотивы серии с заданным набором выбора."""
        tree = self.trees[series]
        for iid in tree.get_children(""):
            n = int(tree.set(iid, "num"))
            if (series, n) not in selected_set:
                return False
        return True

    def update_series_checkboxes(self):
        """Синхронизировать чекбоксы серий на основе текущего выбора."""
        for series in self.trees:
            self._recalculate_series_checkbox(series)

    def update_selection_count(self):
        """Обновляет метку с количеством выбранных локомотивов."""
        total = sum(len(t.get_children("")) for t in self.trees.values())
        selected = len(self.selected)
        self.sl.config(text=f"Выбрано: {selected} из {total}")

    # -------------------- Работа с коэффициентами --------------------

    def load_coefficients_file(self):
        """Загрузить файл коэффициентов и обновить UI."""
        fn = filedialog.askopenfilename(
            title="Выберите файл коэффициентов",
            filetypes=[("Excel files", "*.xlsx *.xls")],
        )
        if not fn:
            return

        min_work = 200 if self.exclude_low_work.get() else 0
        if self.cm.load_coefficients(fn, min_work):
            from pathlib import Path
            self.cfl.config(text=Path(fn).name, foreground="black")

            st = self.cm.get_statistics() or {}
            if st:
                self.csl.config(
                    text=f"Загружено: {st['total_locomotives']} локомотивов, "
                        f"{st['series_count']} серий. "
                        f"Средн. откл.: {st['avg_deviation_percent']:.1f}%"
                )
                messagebox.showinfo(
                    "Успех",
                    f"Коэффициенты загружены успешно!\nЛокомотивов: {st['total_locomotives']}",
                )
            else:
                self.csl.config(text="Коэффициенты загружены")
                messagebox.showinfo("Успех", "Коэффициенты загружены успешно!")

            # Точечное обновление — меняем только колонку коэффициентов и цвета
            self.update_coefficients_in_place()
            # Пересчёт метки выбора (количество выбранных не меняется)
            self.update_selection_count()
        else:
            messagebox.showerror("Ошибка", "Не удалось загрузить файл коэффициентов")

    def clear_coefficients(self):
        """Сбросить загруженные коэффициенты и обновить UI."""
        self.cm = LocomotiveCoefficientsManager()
        self.cfl.config(text="Не загружен", foreground="gray")
        self.csl.config(text="")
        self.uc.set(False)

        self.update_coefficients_in_place()
        self.update_selection_count()

    def _restore_selection(self, selection: set[Tuple[str, int]]):
        """Восстановить состояние выбора после перестройки UI."""
        self.selected.clear()
        for series, tree in self.trees.items():
            for iid in tree.get_children(""):
                n = int(tree.set(iid, "num"))
                key = (series, n)
                is_sel = key in selection
                if is_sel:
                    self.selected.add(key)
                tree.set(iid, "sel", "✓" if is_sel else "")
            self.series_vars[series].set(self._series_all_selected(series, self.selected))
        self.update_selection_count()

    def on_use_coefficients_changed(self):
        """Проверка перед включением учёта коэффициентов."""
        if self.uc.get() and not self.cm.get_statistics():
            messagebox.showwarning("Предупреждение", "Сначала загрузите файл с коэффициентами")
            self.uc.set(False)

    # -------------------- Применение/отмена --------------------

    def load_current_selection(self):
        """Инициализирует выбор из фильтра (по умолчанию там выбраны все)."""
        sel = getattr(self.f, "sel", set())
        if not sel:
            # Если фильтр ничего не содержит — считаем, что выбрано всё
            for series, tree in self.trees.items():
                for iid in tree.get_children(""):
                    n = int(tree.set(iid, "num"))
                    self.selected.add((series, n))
                    tree.set(iid, "sel", "✓")
                self.series_vars[series].set(True)
        else:
            # Восстановить из фильтра
            for series, tree in self.trees.items():
                for iid in tree.get_children(""):
                    n = int(tree.set(iid, "num"))
                    key = (series, n)
                    is_sel = key in sel
                    if is_sel:
                        self.selected.add(key)
                    tree.set(iid, "sel", "✓" if is_sel else "")
                self.series_vars[series].set(self._series_all_selected(series, self.selected))
        self.update_selection_count()

    def apply_selection(self):
        """Сохранить выбор и закрыть диалог."""
        # Без сортировки: состав и порядок определит внешний код (как вы и просили)
        sel = list(self.selected)
        self.f.set_selected_locomotives(sel)
        self.res = {
            "selected_locomotives": sel,
            "use_coefficients": self.uc.get(),
            "exclude_low_work": self.exclude_low_work.get(),
            "coefficients_manager": self.cm,
        }
        self.d.destroy()
        
    def cancel(self):
        """Отменить изменения и закрыть диалог."""
        self.res = None
        self.d.destroy()

    def update_coefficients_in_place(self):
        """Обновляет только колонку 'Коэффициент' и цвет строк без пересоздания вкладок."""
        # Обновляем коэффициенты по всем сериям и строкам
        for series, tree in self.trees.items():
            # Теги окраски уже сконфигурированы в _build_series_tab
            for iid in tree.get_children(""):
                number = int(tree.set(iid, "num"))
                coef = self.cm.get_coefficient(series, number)
                coef_text = f"{coef:.3f}" if coef != 1.0 else "-"
                tag = "above_norm" if coef > 1.05 else "below_norm" if coef < 0.95 else "norm"
                tree.set(iid, "coef", coef_text)
                tree.item(iid, tags=(tag,))  # обновляем цветовую метку только для этой строки