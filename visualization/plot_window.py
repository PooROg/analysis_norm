# visualization/plot_window.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Окно отображения интерактивного графика."""

import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Optional
import threading

import matplotlib
matplotlib.use('TkAgg')  # Обязательно до импорта pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .interactive_plot import InteractivePlot
from .plot_modes import DisplayMode

logger = logging.getLogger(__name__)


class PlotWindow:
    """
    Окно для отображения интерактивного графика анализа норм.
    Включает контролы переключения режимов и экспорта.
    """
    
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.window: Optional[tk.Toplevel] = None
        self.plot: Optional[InteractivePlot] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None
        
        # Контролы режима отображения
        self.mode_var = tk.StringVar(value=DisplayMode.WORK.value)
        
        # Данные текущего графика
        self._current_data: Dict = {}
        
        # Флаг инициализации
        self._initialized = False
        
    def create_window(self) -> tk.Toplevel:
        """
        Создает окно графика с элементами управления.
        
        Returns:
            Созданное окно Toplevel
        """
        logger.info("Создание окна графика")
        
        # Создаем окно
        self.window = tk.Toplevel(self.parent)
        self.window.title("Интерактивный график анализа норм")
        self.window.geometry("1200x800")
        
        # Привязываем к родительскому окну
        self.window.transient(self.parent)
        self.window.grab_set()  # Делаем модальным
        
        # Обработчик закрытия окна
        self.window.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Создаем интерфейс
        self._create_interface()
        
        # Центрируем окно
        self._center_window()
        
        self._initialized = True
        logger.info("Окно графика создано")
        
        return self.window
        
    def _create_interface(self) -> None:
        """Создает интерфейс окна графика."""
        # Главный контейнер
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Панель управления (сверху)
        self._create_control_panel(main_frame)
        
        # Область графика
        self._create_plot_area(main_frame)
        
        # Панель кнопок (снизу)
        self._create_button_panel(main_frame)
        
    def _create_control_panel(self, parent: ttk.Frame) -> None:
        """Создает панель управления с переключателем режимов."""
        control_frame = ttk.LabelFrame(parent, text="Управление отображением", padding="10")
        control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Переключатель режимов
        mode_frame = ttk.Frame(control_frame)
        mode_frame.pack(side=tk.LEFT)
        
        ttk.Label(mode_frame, text="Режим отображения точек:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        # Радио кнопки для режимов
        work_radio = ttk.Radiobutton(
            mode_frame,
            text="Уд. на работу (текущий)",
            variable=self.mode_var,
            value=DisplayMode.WORK.value,
            command=self._on_mode_changed
        )
        work_radio.pack(side=tk.LEFT, padx=(0, 15))
        
        nf_radio = ttk.Radiobutton(
            mode_frame,
            text="Н/Ф (по соотношению норма/факт)", 
            variable=self.mode_var,
            value=DisplayMode.NF_RATIO.value,
            command=self._on_mode_changed
        )
        nf_radio.pack(side=tk.LEFT)
        
        # Разделитель
        ttk.Separator(control_frame, orient='vertical').pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        # Информация о текущем режиме
        info_frame = ttk.Frame(control_frame)
        info_frame.pack(side=tk.LEFT)
        
        self.info_label = ttk.Label(
            info_frame, 
            text="Текущий режим: Уд. на работу",
            foreground="blue",
            font=("Arial", 9)
        )
        self.info_label.pack()
        
    def _create_plot_area(self, parent: ttk.Frame) -> None:
        """Создает область для отображения графика."""
        plot_frame = ttk.Frame(parent)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Создаем фигуру matplotlib
        self.figure = Figure(figsize=(12, 8), dpi=100)
        self.figure.patch.set_facecolor('white')
        
        # Создаем интерактивный график
        self.plot = InteractivePlot(self.figure)
        
        # Создаем canvas для встраивания в tkinter
        self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
        self.canvas.draw()
        
        # Размещаем canvas
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Добавляем панель инструментов matplotlib
        toolbar_frame = ttk.Frame(plot_frame)
        toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
    def _create_button_panel(self, parent: ttk.Frame) -> None:
        """Создает панель с кнопками действий."""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Кнопки экспорта
        export_frame = ttk.LabelFrame(button_frame, text="Экспорт", padding="5")
        export_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            export_frame,
            text="Сохранить как PNG",
            command=lambda: self._export_plot("png")
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            export_frame,
            text="Сохранить как PDF", 
            command=lambda: self._export_plot("pdf")
        ).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(
            export_frame,
            text="Сохранить как SVG",
            command=lambda: self._export_plot("svg")
        ).pack(side=tk.LEFT, padx=2)
        
        # Кнопки управления
        control_frame = ttk.Frame(button_frame)
        control_frame.pack(side=tk.RIGHT)
        
        ttk.Button(
            control_frame,
            text="Сбросить масштаб",
            command=self._reset_zoom
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Обновить график",
            command=self._refresh_plot
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="Закрыть",
            command=self._on_window_close
        ).pack(side=tk.LEFT, padx=5)
        
    def _center_window(self) -> None:
        """Центрирует окно относительно родительского."""
        if not self.window:
            return
            
        self.window.update_idletasks()
        
        # Получаем размеры
        w = self.window.winfo_width()
        h = self.window.winfo_height()
        
        # Получаем размеры родительского окна
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_w = self.parent.winfo_width()
        parent_h = self.parent.winfo_height()
        
        # Вычисляем координаты для центрирования
        x = parent_x + (parent_w // 2) - (w // 2)
        y = parent_y + (parent_h // 2) - (h // 2)
        
        # Убеждаемся что окно остается на экране
        screen_w = self.window.winfo_screenwidth()
        screen_h = self.window.winfo_screenheight()
        
        x = max(0, min(x, screen_w - w))
        y = max(0, min(y, screen_h - h))
        
        self.window.geometry(f"{w}x{h}+{x}+{y}")
        
    def show_plot(
        self,
        section_name: str,
        routes_df,
        norm_functions: Dict,
        specific_norm_id: Optional[str] = None,
        single_section_only: bool = False
    ) -> None:
        """
        Отображает график в окне.
        
        Args:
            section_name: Название участка
            routes_df: DataFrame с данными маршрутов
            norm_functions: Функции интерполяции норм
            specific_norm_id: ID конкретной нормы (опционально)
            single_section_only: Только маршруты с одним участком
        """
        logger.info("Отображение графика для участка: %s", section_name)
        
        if not self._initialized or not self.window or not self.plot:
            logger.error("Окно графика не инициализировано")
            return
            
        # Сохраняем параметры для возможного обновления
        self._current_data = {
            'section_name': section_name,
            'routes_df': routes_df,
            'norm_functions': norm_functions,
            'specific_norm_id': specific_norm_id,
            'single_section_only': single_section_only
        }
        
        try:
            # Создаем график в отдельном потоке для отзывчивости UI
            self._show_loading_indicator()
            
            # Запускаем создание графика в фоне
            threading.Thread(
                target=self._create_plot_thread,
                daemon=True
            ).start()
            
        except Exception as e:
            logger.error("Ошибка отображения графика: %s", e, exc_info=True)
            messagebox.showerror("Ошибка", f"Не удалось отобразить график: {str(e)}")
            
    def _show_loading_indicator(self) -> None:
        """Показывает индикатор загрузки."""
        if self.plot and self.plot.ax1:
            self.plot.ax1.clear()
            self.plot.ax1.text(
                0.5, 0.5, 
                "Создание графика...\nПожалуйста, подождите",
                ha='center', va='center',
                transform=self.plot.ax1.transAxes,
                fontsize=14
            )
            
        if self.canvas:
            self.canvas.draw_idle()
            
    def _create_plot_thread(self) -> None:
        """Создает график в фоновом потоке."""
        try:
            # Создаем график
            self.plot.create_plot(**self._current_data)
            
            # Обновляем UI в главном потоке
            if self.window:
                self.window.after(0, self._on_plot_created)
                
        except Exception as e:
            logger.error("Ошибка создания графика в потоке: %s", e, exc_info=True)
            if self.window:
                self.window.after(0, lambda: self._on_plot_error(str(e)))
                
    def _on_plot_created(self) -> None:
        """Вызывается после успешного создания графика."""
        if self.canvas:
            self.canvas.draw_idle()
        logger.info("График успешно создан и отображен")
        
    def _on_plot_error(self, error_message: str) -> None:
        """Вызывается при ошибке создания графика."""
        messagebox.showerror("Ошибка", f"Ошибка создания графика: {error_message}")
        
    def _on_mode_changed(self) -> None:
        """Обработчик изменения режима отображения."""
        if not self._initialized or not self.plot:
            return
            
        try:
            new_mode = DisplayMode(self.mode_var.get())
            logger.info("Переключение режима на: %s", new_mode.value)
            
            # Переключаем режим в графике
            self.plot.switch_display_mode(new_mode)
            
            # Обновляем информационную метку
            mode_label = self.plot.mode_manager.get_mode_label(new_mode)
            self.info_label.config(text=f"Текущий режим: {mode_label}")
            
        except Exception as e:
            logger.error("Ошибка переключения режима: %s", e, exc_info=True)
            messagebox.showerror("Ошибка", f"Ошибка переключения режима: {str(e)}")
            
            # Возвращаем предыдущий режим
            self.mode_var.set(DisplayMode.WORK.value)
            
    def _export_plot(self, format_type: str) -> None:
        """
        Экспортирует график в указанном формате.
        
        Args:
            format_type: Формат файла ('png', 'pdf', 'svg')
        """
        if not self.plot:
            messagebox.showwarning("Предупреждение", "Нет графика для экспорта")
            return
            
        # Определяем расширение и фильтры файлов
        extensions = {
            'png': ('PNG files', '*.png'),
            'pdf': ('PDF files', '*.pdf'), 
            'svg': ('SVG files', '*.svg')
        }
        
        if format_type not in extensions:
            messagebox.showerror("Ошибка", f"Неподдерживаемый формат: {format_type}")
            return
            
        # Диалог сохранения файла
        file_types = [extensions[format_type], ('All files', '*.*')]
        
        filename = filedialog.asksaveasfilename(
            title=f"Сохранить график как {format_type.upper()}",
            defaultextension=f".{format_type}",
            filetypes=file_types,
            parent=self.window
        )
        
        if not filename:
            return
            
        try:
            # Определяем DPI в зависимости от формата
            dpi = 300 if format_type == 'png' else 150
            
            success = self.plot.export_plot(filename, dpi=dpi)
            
            if success:
                messagebox.showinfo(
                    "Успех", 
                    f"График успешно сохранен в файл:\n{Path(filename).name}",
                    parent=self.window
                )
            else:
                messagebox.showerror("Ошибка", "Не удалось сохранить график", parent=self.window)
                
        except Exception as e:
            logger.error("Ошибка экспорта графика: %s", e, exc_info=True)
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}", parent=self.window)
            
    def _reset_zoom(self) -> None:
        """Сбрасывает масштабирование графика."""
        if not self.plot:
            return
            
        try:
            # Автоматическое масштабирование
            self.plot.ax1.relim()
            self.plot.ax1.autoscale_view()
            self.plot.ax2.relim()
            self.plot.ax2.autoscale_view()
            
            if self.canvas:
                self.canvas.draw_idle()
                
            logger.info("Масштаб графика сброшен")
            
        except Exception as e:
            logger.error("Ошибка сброса масштаба: %s", e)
            messagebox.showerror("Ошибка", f"Ошибка сброса масштаба: {str(e)}", parent=self.window)
            
    def _refresh_plot(self) -> None:
        """Обновляет график с текущими данными."""
        if not self._current_data:
            messagebox.showwarning("Предупреждение", "Нет данных для обновления", parent=self.window)
            return
            
        logger.info("Обновление графика")
        
        try:
            # Пересоздаем график с теми же параметрами
            self.show_plot(**self._current_data)
            
        except Exception as e:
            logger.error("Ошибка обновления графика: %s", e, exc_info=True)
            messagebox.showerror("Ошибка", f"Ошибка обновления: {str(e)}", parent=self.window)
            
    def _on_window_close(self) -> None:
        """Обработчик закрытия окна."""
        logger.info("Закрытие окна графика")
        
        try:
            # Очищаем ресурсы
            if self.plot:
                self.plot = None
                
            if self.canvas:
                self.canvas = None
                
            if self.toolbar:
                self.toolbar = None
                
            # Закрываем окно
            if self.window:
                self.window.grab_release()  # Снимаем модальность
                self.window.destroy()
                self.window = None
                
            self._initialized = False
            
        except Exception as e:
            logger.error("Ошибка при закрытии окна: %s", e)
            
    def is_active(self) -> bool:
        """
        Проверяет, активно ли окно графика.
        
        Returns:
            True если окно открыто и активно
        """
        return (self._initialized and 
                self.window is not None and 
                self.window.winfo_exists())
                
    def bring_to_front(self) -> None:
        """Выводит окно на передний план."""
        if self.is_active():
            self.window.lift()
            self.window.focus_force()