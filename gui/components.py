# gui/components.py (обновленный)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Компоненты GUI интерфейса с поддержкой встроенной визуализации."""

from __future__ import annotations

import tkinter as tk
from abc import ABC, abstractmethod
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, List, Optional
from datetime import datetime
import logging

from core.config import APP_CONFIG
from core.utils import format_number


class GUIComponent(ABC):
    """Базовый класс для компонентов GUI."""
    
    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.frame = None
    
    @abstractmethod
    def create_widgets(self) -> tk.Widget:
        """Создает виджеты компонента."""
        pass


class FileSection(GUIComponent):
    """Секция работы с файлами."""
    
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        self.route_files: List[str] = []
        self.norm_files: List[str] = []
        
        # Callbacks
        self.on_routes_loaded: Optional[Callable] = None
        self.on_norms_loaded: Optional[Callable] = None
    
    def create_widgets(self) -> tk.Widget:
        """Создает секцию работы с файлами."""
        self.frame = ttk.LabelFrame(self.parent, text="Файлы данных", padding="10")
        self.frame.columnconfigure(1, weight=1)
        
        # HTML файлы маршрутов
        self._create_file_row(
            row=0,
            label="HTML файлы маршрутов:",
            attr_name="routes",
            load_callback=self._load_routes
        )
        
        # HTML файлы норм
        self._create_file_row(
            row=1, 
            label="HTML файлы норм:",
            attr_name="norms",
            load_callback=self._load_norms
        )
        
        # Кнопки загрузки
        self._create_load_buttons()
        
        # Статус
        self.load_status = ttk.Label(self.frame, text="", style='Success.TLabel')
        self.load_status.grid(row=3, column=0, columnspan=4, pady=(5, 0))
        
        return self.frame
    
    def _create_file_row(self, row: int, label: str, attr_name: str, load_callback: Callable):
        """Создает строку для выбора файлов."""
        ttk.Label(self.frame, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 5))
        
        label_widget = ttk.Label(self.frame, text="Не выбраны", foreground="gray")
        label_widget.grid(row=row, column=1, sticky=tk.W, padx=(0, 10))
        
        choose_btn = ttk.Button(
            self.frame, 
            text="Выбрать файлы",
            command=lambda: self._select_files(attr_name, label_widget, load_callback)
        )
        choose_btn.grid(row=row, column=2, padx=(0, 5))
        
        clear_btn = ttk.Button(
            self.frame,
            text="Очистить", 
            command=lambda: self._clear_files(attr_name, label_widget)
        )
        clear_btn.grid(row=row, column=3)
        
        # Сохраняем ссылки на виджеты
        setattr(self, f"{attr_name}_label", label_widget)
        setattr(self, f"{attr_name}_choose_btn", choose_btn)
    
    def _create_load_buttons(self):
        """Создает кнопки загрузки."""
        buttons_frame = ttk.Frame(self.frame)
        buttons_frame.grid(row=2, column=0, columnspan=4, pady=(10, 0))
        
        self.load_routes_btn = ttk.Button(
            buttons_frame,
            text="Загрузить маршруты",
            command=self._load_routes,
            state='disabled'
        )
        self.load_routes_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.load_norms_btn = ttk.Button(
            buttons_frame,
            text="Загрузить нормы", 
            command=self._load_norms,
            state='disabled'
        )
        self.load_norms_btn.pack(side=tk.LEFT, padx=(0, 5))
    
    def _select_files(self, file_type: str, label_widget: ttk.Label, load_callback: Callable):
        """Выбирает HTML файлы."""
        kind_label = "маршрутов" if file_type == "routes" else "норм"
        
        files = filedialog.askopenfilenames(
            title=f"Выберите HTML файлы {kind_label}",
            filetypes=[("HTML files", "*.html *.htm"), ("All files", "*.*")],
        )
        
        if not files:
            return
        
        # Сохраняем файлы
        if file_type == "routes":
            self.route_files = list(files)
        else:
            self.norm_files = list(files)
        
        # Обновляем UI
        file_names = [Path(f).name for f in files]
        display_text = (
            ", ".join(file_names) if len(file_names) <= 3 
            else f"{', '.join(file_names[:3])} и еще {len(file_names) - 3} файлов"
        )
        
        label_widget.config(text=display_text, foreground="black")
        
        # Активируем кнопку загрузки
        load_btn = getattr(self, f"load_{file_type}_btn")
        load_btn.config(state='normal')
    
    def _clear_files(self, file_type: str, label_widget: ttk.Label):
        """Очищает выбранные файлы."""
        if file_type == "routes":
            self.route_files = []
        else:
            self.norm_files = []
        
        label_widget.config(text="Не выбраны", foreground="gray")
        
        load_btn = getattr(self, f"load_{file_type}_btn")
        load_btn.config(state='disabled')
    
    def _load_routes(self):
        """Загружает маршруты."""
        if not self.route_files:
            messagebox.showwarning("Предупреждение", "Сначала выберите HTML файлы маршрутов")
            return
        
        if self.on_routes_loaded:
            self.on_routes_loaded(self.route_files)
    
    def _load_norms(self):
        """Загружает нормы."""
        if not self.norm_files:
            messagebox.showwarning("Предупреждение", "Сначала выберите HTML файлы норм")
            return
        
        if self.on_norms_loaded:
            self.on_norms_loaded(self.norm_files)
    
    def update_status(self, message: str, status_type: str = "info"):
        """Обновляет статус загрузки."""
        style_map = {
            "success": "Success.TLabel",
            "error": "Error.TLabel", 
            "warning": "Warning.TLabel",
            "info": "TLabel"
        }
        
        style = style_map.get(status_type, "TLabel")
        self.load_status.config(text=message, style=style)


class ControlSection(GUIComponent):
    """Секция управления анализом."""
    
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        
        # Переменные
        self.section_var = tk.StringVar()
        self.norm_var = tk.StringVar()
        self.single_section_only = tk.BooleanVar(value=False)
        
        # Callbacks
        self.on_section_selected: Optional[Callable] = None
        self.on_norm_selected: Optional[Callable] = None
        self.on_analyze_clicked: Optional[Callable] = None
        self.on_filter_clicked: Optional[Callable] = None
        self.on_edit_norms_clicked: Optional[Callable] = None
        self.on_single_section_changed: Optional[Callable] = None
    
    def create_widgets(self) -> tk.Widget:
        """Создает секцию управления."""
        self.frame = ttk.LabelFrame(self.parent, text="Управление анализом", padding="10")
        self.frame.rowconfigure(12, weight=1)  # Растягиваем статистику
        
        row = 0
        
        # Выбор участка
        row = self._create_section_selection(row)
        
        # Фильтр по одному участку
        row = self._create_single_section_filter(row)
        
        # Выбор нормы
        row = self._create_norm_selection(row)
        
        # Информация об участке
        row = self._create_section_info(row)
        
        # Кнопки управления
        row = self._create_action_buttons(row)
        
        # Информация о фильтрах
        row = self._create_filter_info(row)
        
        # Статистика
        row = self._create_statistics_section(row)
        
        # Кнопки экспорта
        self._create_export_buttons(row)
        
        return self.frame
    
    def _create_section_selection(self, row: int) -> int:
        """Создает выбор участка."""
        ttk.Label(self.frame, text="Участок:", style='Header.TLabel').grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        
        self.section_combo = ttk.Combobox(
            self.frame, textvariable=self.section_var, state='readonly', width=40
        )
        self.section_combo.grid(row=row+1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.section_combo.bind('<<ComboboxSelected>>', self._on_section_change)
        
        return row + 2
    
    def _create_single_section_filter(self, row: int) -> int:
        """Создает фильтр по одному участку."""
        self.single_section_check = ttk.Checkbutton(
            self.frame,
            text="Только маршруты с одним участком",
            variable=self.single_section_only,
            command=self._on_single_section_change,
        )
        self.single_section_check.grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        
        return row + 1
    
    def _create_norm_selection(self, row: int) -> int:
        """Создает выбор нормы."""
        ttk.Label(self.frame, text="Норма (опционально):", style='Header.TLabel').grid(
            row=row, column=0, sticky=tk.W, pady=(0, 5)
        )
        
        norm_frame = ttk.Frame(self.frame)
        norm_frame.grid(row=row+1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        norm_frame.columnconfigure(0, weight=1)
        
        self.norm_combo = ttk.Combobox(
            norm_frame, textvariable=self.norm_var, state='readonly', width=30
        )
        self.norm_combo.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        self.norm_combo.bind('<<ComboboxSelected>>', self._on_norm_change)
        
        self.norm_info_btn = ttk.Button(
            norm_frame, text="Инфо о норме", state='disabled',
            command=lambda: self.on_norm_selected and self.on_norm_selected("info")
        )
        self.norm_info_btn.grid(row=0, column=1)
        
        return row + 2
    
    def _create_section_info(self, row: int) -> int:
        """Создает информацию о количестве маршрутов."""
        self.section_info_label = ttk.Label(self.frame, text="", style='Warning.TLabel')
        self.section_info_label.grid(row=row, column=0, sticky=tk.W, pady=(0, 10))
        
        return row + 1
    
    def _create_action_buttons(self, row: int) -> int:
        """Создает основные кнопки действий."""
        buttons = [
            ("Анализировать участок", "analyze", self._on_analyze),
            ("Фильтр локомотивов", "filter", self._on_filter),
            ("Редактировать нормы", "edit_norms", self._on_edit_norms),
        ]
        
        for text, name, command in buttons:
            btn = ttk.Button(self.frame, text=text, command=command, state='disabled')
            btn.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
            setattr(self, f"{name}_btn", btn)
            row += 1
        
        return row
    
    def _create_filter_info(self, row: int) -> int:
        """Создает информацию о фильтрах."""
        self.filter_info_label = ttk.Label(self.frame, text="", style='Warning.TLabel')
        self.filter_info_label.grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        
        return row + 1
    
    def _create_statistics_section(self, row: int) -> int:
        """Создает секцию статистики."""
        ttk.Label(self.frame, text="Статистика:", style='Header.TLabel').grid(
            row=row, column=0, sticky=tk.W, pady=(10, 5)
        )
        
        self.stats_text = tk.Text(self.frame, width=45, height=8, wrap=tk.WORD)
        self.stats_text.grid(row=row+1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        stats_scrollbar = ttk.Scrollbar(self.frame, orient='vertical', command=self.stats_text.yview)
        stats_scrollbar.grid(row=row+1, column=1, sticky=(tk.N, tk.S), pady=(0, 10))
        self.stats_text.configure(yscrollcommand=stats_scrollbar.set)
        
        return row + 2
    
    def _create_export_buttons(self, row: int):
        """Создает кнопки экспорта."""
        export_frame = ttk.Frame(self.frame)
        export_frame.grid(row=row, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.export_excel_btn = ttk.Button(
            export_frame, text="Экспорт в Excel", state='disabled'
        )
        self.export_excel_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        self.export_plot_btn = ttk.Button(
            export_frame, text="Экспорт графика", state='disabled'
        )
        self.export_plot_btn.pack(side=tk.LEFT)
    
    # Event handlers
    def _on_section_change(self, event=None):
        if self.on_section_selected:
            self.on_section_selected(self.section_var.get())
    
    def _on_norm_change(self, event=None):
        norm_text = self.norm_var.get()
        self.norm_info_btn.config(state='normal' if norm_text and norm_text != "Все нормы" else 'disabled')
    
    def _on_single_section_change(self):
        if self.on_single_section_changed:
            self.on_single_section_changed(self.single_section_only.get())
    
    def _on_analyze(self):
        if self.on_analyze_clicked:
            self.on_analyze_clicked()
    
    def _on_filter(self):
        if self.on_filter_clicked:
            self.on_filter_clicked()
    
    def _on_edit_norms(self):
        if self.on_edit_norms_clicked:
            self.on_edit_norms_clicked()
    
    # Public methods for updating UI
    def update_sections(self, sections: List[str]):
        """Обновляет список участков."""
        self.section_combo['values'] = sections
    
    def update_norms(self, norms_with_counts: List[tuple[str, int]]):
        """Обновляет список норм с количествами."""
        norm_values = ["Все нормы"]
        for norm_id, count in norms_with_counts:
            norm_values.append(f"Норма {norm_id} ({count} маршрутов)")
        
        self.norm_combo['values'] = norm_values
        self.norm_var.set("Все нормы")
    
    def update_section_info(self, message: str):
        """Обновляет информацию об участке."""
        self.section_info_label.config(text=message)
    
    def update_filter_info(self, message: str):
        """Обновляет информацию о фильтрах."""
        self.filter_info_label.config(text=message)
    
    def update_statistics(self, stats: dict):
        """Обновляет статистику."""
        self.stats_text.delete(1.0, tk.END)
        
        if not stats:
            return
        
        processed = max(stats.get('processed', 1), 1)  # Избегаем деления на 0
        
        text = (
            f"Всего маршрутов: {stats.get('total', 0)}\n"
            f"Обработано: {stats.get('processed', 0)}\n"
            f"Экономия: {stats.get('economy', 0)} ({stats.get('economy', 0)/processed*100:.1f}%)\n"
            f"В норме: {stats.get('normal', 0)} ({stats.get('normal', 0)/processed*100:.1f}%)\n"
            f"Перерасход: {stats.get('overrun', 0)} ({stats.get('overrun', 0)/processed*100:.1f}%)\n"
            f"Среднее отклонение: {format_number(stats.get('mean_deviation', 0))}%\n"
            f"Медианное отклонение: {format_number(stats.get('median_deviation', 0))}%\n\n"
        )
        
        detailed = stats.get('detailed_stats', {})
        if detailed:
            text += "Детально:\n"
            categories = {
                'economy_strong': 'Экономия сильная (>30%)',
                'economy_medium': 'Экономия средняя (20-30%)',
                'economy_weak': 'Экономия слабая (5-20%)',
                'normal': 'Норма (±5%)',
                'overrun_weak': 'Перерасход слабый (5-20%)',
                'overrun_medium': 'Перерасход средний (20-30%)',
                'overrun_strong': 'Перерасход сильный (>30%)',
            }
            
            for key, name in categories.items():
                count = detailed.get(key, 0)
                if count > 0:
                    percent = count / processed * 100
                    text += f"{name}: {count} ({percent:.1f}%)\n"
        
        self.stats_text.insert(1.0, text)
    
    def enable_buttons(self, button_states: dict[str, bool]):
        """Включает/выключает кнопки."""
        for button_name, enabled in button_states.items():
            btn = getattr(self, f"{button_name}_btn", None)
            if btn:
                btn.config(state='normal' if enabled else 'disabled')


class VisualizationSection(GUIComponent):
    """ОБНОВЛЕННАЯ секция визуализации для встроенного графика."""
    
    def __init__(self, parent: tk.Widget):
        super().__init__(parent)
        
        # Callbacks
        self.on_plot_open: Optional[Callable] = None
        self.on_norm_storage_info: Optional[Callable] = None
        self.on_validate_norms: Optional[Callable] = None
        self.on_routes_statistics: Optional[Callable] = None
    
    def create_widgets(self) -> tk.Widget:
        """Создает секцию визуализации."""
        self.frame = ttk.LabelFrame(self.parent, text="Встроенная визуализация", padding="10")
        self.frame.rowconfigure(4, weight=1)  # Растягиваем текстовое поле
        
        # ИЗМЕНЕНО: Кнопка для встроенного графика
        self.view_plot_btn = ttk.Button(
            self.frame,
            text="📊 Открыть интерактивный график",
            state='disabled',
            command=lambda: self.on_plot_open and self.on_plot_open()
        )
        self.view_plot_btn.pack(pady=(0, 10))
        
        # Преимущества встроенного графика
        advantages_frame = ttk.LabelFrame(self.frame, text="Преимущества встроенного графика", padding="5")
        advantages_frame.pack(fill=tk.X, pady=(0, 10))
        
        advantages_text = (
            "✓ Высокая производительность с matplotlib\n"
            "✓ Стабильное переключение режимов (Уд. на работу ↔ Н/Ф)\n"  
            "✓ Интерактивность: клик по точке → детали маршрута\n"
            "✓ Встроенные инструменты: масштаб, панорамирование\n"
            "✓ Экспорт в PNG, PDF, SVG с высоким разрешением"
        )
        
        ttk.Label(advantages_frame, text=advantages_text, font=("Arial", 9)).pack(anchor=tk.W)
        
        # Информация о хранилище норм
        self._create_norms_management_section()
        
        # Информация о данных
        self._create_data_info_section()
        
        # Информация о графике
        self.plot_info = tk.Text(self.frame, width=60, height=20, wrap=tk.WORD)
        self.plot_info.pack(fill=tk.BOTH, expand=True)
        
        # Устанавливаем обновленные инструкции
        self.show_default_instructions()
        
        return self.frame
    
    def _create_norms_management_section(self):
        """Создает секцию управления нормами."""
        norm_info_frame = ttk.LabelFrame(self.frame, text="Управление нормами", padding="5")
        norm_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            norm_info_frame,
            text="Информация о хранилище норм",
            command=lambda: self.on_norm_storage_info and self.on_norm_storage_info()
        ).pack(pady=2)
        
        ttk.Button(
            norm_info_frame,
            text="Валидировать нормы",
            command=lambda: self.on_validate_norms and self.on_validate_norms()
        ).pack(pady=2)
    
    def _create_data_info_section(self):
        """Создает секцию информации о данных."""
        data_info_frame = ttk.LabelFrame(self.frame, text="Информация о данных", padding="5")
        data_info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            data_info_frame,
            text="Статистика маршрутов",
            command=lambda: self.on_routes_statistics and self.on_routes_statistics()
        ).pack(pady=2)

    def update_plot_creation_status(self, status: str, is_error: bool = False):
        """НОВЫЙ метод обновления статуса создания графика."""
        try:
            if hasattr(self, 'plot_info') and self.plot_info:
                # Очищаем и показываем статус
                self.plot_info.delete(1.0, tk.END)
                
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                status_text = f"""СТАТУС СОЗДАНИЯ ГРАФИКА
    ========================

    ⏰ Время: {timestamp}
    📊 Статус: {status}

    {'❌ Обнаружена ошибка!' if is_error else '✅ Выполняется...'}

    Подождите завершения операции или проверьте
    журнал операций для подробностей.
    """
                
                self.plot_info.insert(1.0, status_text)
                self.plot_info.see(1.0)
                
        except Exception as e:
            logging.error("Ошибка обновления статуса графика: %s", e)

    def show_default_instructions(self):
        """ОБНОВЛЕННЫЕ инструкции для встроенного графика."""
        instructions = """ВСТРОЕННЫЙ ИНТЕРАКТИВНЫЙ ГРАФИК
=======================================================

НОВЫЕ ВОЗМОЖНОСТИ v2.0:

🎯 СТАБИЛЬНОЕ ПЕРЕКЛЮЧЕНИЕ РЕЖИМОВ:
   • "Уд. на работу" - стандартный режим анализа
   • "Н/Ф" - соотношение норма/факт
   • Точки ГАРАНТИРОВАННО возвращаются на исходные позиции
   • Никаких артефактов браузерной визуализации

🖱️ ИНТЕРАКТИВНОСТЬ:
   • Клик по любой точке → подробности маршрута
   • Инструменты matplotlib: масштаб, панорамирование
   • Кнопка "Сбросить масштаб" для быстрого возврата
   • Hover эффекты (при наведении мыши)

📊 ВЫСОКАЯ ПРОИЗВОДИТЕЛЬНОСТЬ:
   • Нативный matplotlib с tkinter backend
   • Мгновенное переключение режимов
   • Эффективное кэширование вычислений
   • Обработка больших объемов данных

💾 ЭКСПОРТ БЕЗ ПОТЕРЬ:
   • PNG (300 DPI) - для документов
   • PDF (векторный) - для печати
   • SVG (масштабируемый) - для веб

🔧 РАСШИРЕННЫЕ ВОЗМОЖНОСТИ:
   • Два подграфика: нормы + отклонения
   • Цветовые зоны отклонений
   • Легенда с количеством точек по статусам
   • Автоматическое определение типа нормы (вес/нажатие)

ИНСТРУКЦИЯ:
1. Загрузите HTML файлы маршрутов и норм
2. Выберите участок и настройте фильтры
3. Нажмите "Анализировать участок"
4. Откройте график кнопкой "📊 Открыть интерактивный график"
5. Переключайте режимы отображения в окне графика
6. Кликайте по точкам для просмотра деталей
7. Экспортируйте результаты в нужном формате

💡 Встроенный график решает все проблемы браузерной
   визуализации и обеспечивает профессиональное
   качество анализа данных."""
        
        self.plot_info.delete(1.0, tk.END)
        self.plot_info.insert(1.0, instructions)
    
    def update_plot_info(self, section_name: str, stats: dict, norm_id: Optional[str] = None, 
                         single_section_only: bool = False):
        """Обновляет информацию о графике."""
        self.plot_info.delete(1.0, tk.END)
        
        norm_text = f" (норма {norm_id})" if norm_id else ""
        filter_text = " [только один участок]" if single_section_only else ""
        
        info_text = f"""ГОТОВ К ВИЗУАЛИЗАЦИИ
========================================

Участок: {section_name}{norm_text}{filter_text}

📈 ДОСТУПНЫЕ ФУНКЦИИ В ГРАФИКЕ:

🖱️ Интерактивность:
- Клик по точке → детали маршрута
- Колесо мыши → масштабирование  
- Перетаскивание → панорамирование
- Двойной клик → сброс масштаба

🔄 Режимы отображения:
- "Уд. на работу": стандартный анализ
- "Н/Ф": коэффициентный анализ
- Плавное переключение без артефактов

📊 Структура графика:
Верхний: Нормы и фактические значения
- Синие квадраты: базовые точки норм
- Золотые квадраты: точки из маршрутов
- Синие линии: интерполированные кривые
- Цветные точки: статусы маршрутов

Нижний: Анализ отклонений
- Золотая зона: норма (±5%)
- Оранжевые линии: значимые отклонения (±20%)  
- Красные линии: критические отклонения (±30%)

📋 СТАТИСТИКА АНАЛИЗА:
Обработано маршрутов: {stats.get('processed', 0)}
Экономия: {stats.get('economy', 0)} ({(stats.get('economy', 0)/max(stats.get('processed', 1), 1)*100):.1f}%)
В норме: {stats.get('normal', 0)} ({(stats.get('normal', 0)/max(stats.get('processed', 1), 1)*100):.1f}%)
Перерасход: {stats.get('overrun', 0)} ({(stats.get('overrun', 0)/max(stats.get('processed', 1), 1)*100):.1f}%)

🚀 Нажмите "📊 Открыть интерактивный график"
   для начала работы!"""
        
        self.plot_info.insert(1.0, info_text)
    
    def enable_plot_button(self, enabled: bool):
        """Включает/выключает кнопку просмотра графика."""
        self.view_plot_btn.config(state='normal' if enabled else 'disabled')