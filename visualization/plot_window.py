# visualization/plot_window.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ИСПРАВЛЕННОЕ окно отображения интерактивного графика БЕЗ THREADING."""

import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Dict, Optional
import traceback

import matplotlib
matplotlib.use('TkAgg')  # Принудительно устанавливаем backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .interactive_plot import InteractivePlot
from .plot_modes import DisplayMode

logger = logging.getLogger(__name__)


class PlotWindow:
    """
    ИСПРАВЛЕННОЕ окно для отображения интерактивного графика.
    Убран threading, добавлена синхронная обработка с progress feedback.
    """
    
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self.window: Optional[tk.Toplevel] = None
        self.plot: Optional[InteractivePlot] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.toolbar: Optional[NavigationToolbar2Tk] = None
        
        # Контролы режима отображения
        self.mode_var = tk.StringVar(value=DisplayMode.WORK.value)
        
        # Progress индикация
        self.progress_var = tk.StringVar(value="")
        self.progress_bar: Optional[ttk.Progressbar] = None
        
        # Данные текущего графика
        self._current_data: Dict = {}
        
        # Флаги состояния
        self._initialized = False
        self._plot_created = False
        
        logger.info("PlotWindow инициализирован")
        
    def create_window(self) -> tk.Toplevel:
        """
        ИСПРАВЛЕННОЕ создание окна с robust error handling.
        
        Returns:
            Созданное окно Toplevel
        """
        logger.info("=== СОЗДАНИЕ ОКНА ГРАФИКА ===")
        
        try:
            # Создаем окно
            self.window = tk.Toplevel(self.parent)
            self.window.title("Интерактивный график анализа норм")
            self.window.geometry("1400x900")  # Увеличенный размер
            
            # Настройка окна
            self.window.transient(self.parent)
            self.window.grab_set()  # Модальное окно
            self.window.protocol("WM_DELETE_WINDOW", self._on_window_close_safe)
            
            # Создаем интерфейс
            self._create_interface_safe()
            
            # Центрируем окно
            self._center_window_safe()
            
            # Создаем matplotlib компоненты
            self._create_matplotlib_components_safe()
            
            self._initialized = True
            logger.info("✓ Окно графика создано успешно")
            
            return self.window
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка создания окна: %s", e, exc_info=True)
            if self.window:
                self.window.destroy()
            raise RuntimeError(f"Не удалось создать окно графика: {e}")
            
    def _create_interface_safe(self) -> None:
        """Безопасно создает интерфейс окна."""
        try:
            # Главный контейнер
            main_frame = ttk.Frame(self.window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Панель управления (сверху)
            self._create_control_panel_safe(main_frame)
            
            # Progress bar
            self._create_progress_panel_safe(main_frame)
            
            # Область графика (растягивается)
            self._create_plot_area_safe(main_frame)
            
            # Панель кнопок (снизу)
            self._create_button_panel_safe(main_frame)
            
            logger.debug("✓ Интерфейс создан")
            
        except Exception as e:
            logger.error("Ошибка создания интерфейса: %s", e, exc_info=True)
            raise
            
    def _create_control_panel_safe(self, parent: ttk.Frame) -> None:
        """Безопасно создает панель управления."""
        try:
            control_frame = ttk.LabelFrame(parent, text="Управление отображением", padding="10")
            control_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Переключатель режимов
            mode_frame = ttk.Frame(control_frame)
            mode_frame.pack(side=tk.LEFT)
            
            ttk.Label(mode_frame, text="Режим отображения:", 
                     font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(0, 15))
            
            # Радио кнопки
            work_radio = ttk.Radiobutton(
                mode_frame,
                text="🎯 Уд. на работу (исходные данные)",
                variable=self.mode_var,
                value=DisplayMode.WORK.value,
                command=self._on_mode_changed_safe
            )
            work_radio.pack(side=tk.LEFT, padx=(0, 20))
            
            nf_radio = ttk.Radiobutton(
                mode_frame,
                text="📊 Н/Ф (соотношение норма/факт)", 
                variable=self.mode_var,
                value=DisplayMode.NF_RATIO.value,
                command=self._on_mode_changed_safe
            )
            nf_radio.pack(side=tk.LEFT)
            
            # Разделитель
            ttk.Separator(control_frame, orient='vertical').pack(
                side=tk.LEFT, fill=tk.Y, padx=25
            )
            
            # Информация о текущем режиме
            info_frame = ttk.Frame(control_frame)
            info_frame.pack(side=tk.LEFT)
            
            self.info_label = ttk.Label(
                info_frame, 
                text="Текущий: Уд. на работу",
                foreground="blue",
                font=("Arial", 10, "bold")
            )
            self.info_label.pack()
            
        except Exception as e:
            logger.error("Ошибка создания панели управления: %s", e)
            
    def _create_progress_panel_safe(self, parent: ttk.Frame) -> None:
        """Создает панель прогресса."""
        try:
            progress_frame = ttk.Frame(parent)
            progress_frame.pack(fill=tk.X, pady=(0, 10))
            
            self.progress_bar = ttk.Progressbar(
                progress_frame, 
                mode='indeterminate',
                length=400
            )
            self.progress_bar.pack(side=tk.LEFT, padx=(0, 10))
            
            self.progress_label = ttk.Label(progress_frame, textvariable=self.progress_var)
            self.progress_label.pack(side=tk.LEFT)
            
        except Exception as e:
            logger.error("Ошибка создания панели прогресса: %s", e)
            
    def _create_plot_area_safe(self, parent: ttk.Frame) -> None:
        """Безопасно создает область для matplotlib графика."""
        try:
            plot_frame = ttk.Frame(parent)
            plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
            
            # Создаем matplotlib Figure с правильными настройками
            self.figure = Figure(
                figsize=(14, 9), 
                dpi=100,
                facecolor='white',
                edgecolor='black',
                tight_layout=True
            )
            
            # Создаем интерактивный график
            self.plot = InteractivePlot(self.figure)
            
            logger.debug("✓ matplotlib Figure и InteractivePlot созданы")
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка создания области графика: %s", e, exc_info=True)
            raise
            
    def _create_matplotlib_components_safe(self) -> None:
        """Безопасно создает matplotlib компоненты для встраивания в tkinter."""
        try:
            if not self.figure:
                raise RuntimeError("Figure не инициализирован")
                
            # Находим plot_frame
            plot_frame = None
            for child in self.window.winfo_children():
                if isinstance(child, ttk.Frame):
                    for grandchild in child.winfo_children():
                        if isinstance(grandchild, ttk.Frame):
                            # Это наш plot_frame (второй ttk.Frame в main_frame)
                            if grandchild != child.winfo_children()[0]:  # не control_frame
                                plot_frame = grandchild
                                break
                    if plot_frame:
                        break
                        
            if not plot_frame:
                raise RuntimeError("Не найден plot_frame")
                
            # Создаем Canvas для встраивания matplotlib в tkinter
            self.canvas = FigureCanvasTkAgg(self.figure, plot_frame)
            
            # Размещаем canvas
            canvas_widget = self.canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            # Панель инструментов matplotlib
            toolbar_frame = ttk.Frame(plot_frame)
            toolbar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
            
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            self.toolbar.update()
            
            # Первоначальная отрисовка
            self.canvas.draw()
            
            logger.info("✓ matplotlib компоненты встроены в tkinter")
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка создания matplotlib компонентов: %s", e, exc_info=True)
            raise
            
    def _create_button_panel_safe(self, parent: ttk.Frame) -> None:
        """Безопасно создает панель кнопок."""
        try:
            button_frame = ttk.Frame(parent)
            button_frame.pack(fill=tk.X)
            
            # Кнопки экспорта
            export_frame = ttk.LabelFrame(button_frame, text="Экспорт графика", padding="5")
            export_frame.pack(side=tk.LEFT, padx=(0, 15))
            
            export_buttons = [
                ("💾 PNG", "png"), ("📄 PDF", "pdf"), ("🎨 SVG", "svg")
            ]
            
            for text, fmt in export_buttons:
                ttk.Button(
                    export_frame,
                    text=text,
                    command=lambda f=fmt: self._export_plot_safe(f)
                ).pack(side=tk.LEFT, padx=3)
            
            # Кнопки управления
            control_frame = ttk.LabelFrame(button_frame, text="Управление", padding="5")
            control_frame.pack(side=tk.LEFT, padx=(0, 15))
            
            control_buttons = [
                ("🔄 Обновить", self._refresh_plot_safe),
                ("🎯 Сбросить масштаб", self._reset_zoom_safe),
                ("❌ Закрыть", self._on_window_close_safe)
            ]
            
            for text, command in control_buttons:
                ttk.Button(control_frame, text=text, command=command).pack(side=tk.LEFT, padx=3)
            
        except Exception as e:
            logger.error("Ошибка создания панели кнопок: %s", e)
            
    def _center_window_safe(self) -> None:
        """Безопасно центрирует окно."""
        try:
            self.window.update_idletasks()
            
            # Получаем размеры
            w, h = self.window.winfo_width(), self.window.winfo_height()
            
            # Центрирование относительно экрана
            screen_w = self.window.winfo_screenwidth()
            screen_h = self.window.winfo_screenheight()
            
            x = (screen_w // 2) - (w // 2)
            y = (screen_h // 2) - (h // 2)
            
            # Убеждаемся что окно остается на экране
            x = max(0, min(x, screen_w - w))
            y = max(0, min(y, screen_h - h))
            
            self.window.geometry(f"{w}x{h}+{x}+{y}")
            logger.debug("✓ Окно отцентрировано")
            
        except Exception as e:
            logger.error("Ошибка центрирования окна: %s", e)
            
    def show_plot(
        self,
        section_name: str,
        routes_df,
        norm_functions: Dict,
        specific_norm_id: Optional[str] = None,
        single_section_only: bool = False
    ) -> None:
        """
        ИСПРАВЛЕННОЕ отображение графика - БЕЗ THREADING.
        Все операции выполняются синхронно с progress feedback.
        """
        logger.info("=== ПОКАЗ ГРАФИКА ===")
        logger.info("Участок: %s | Норма: %s | Один участок: %s", 
                   section_name, specific_norm_id or "Все", single_section_only)
        
        if not self._initialized or not self.window or not self.plot:
            error_msg = "Окно графика не инициализировано"
            logger.error(error_msg)
            messagebox.showerror("Критическая ошибка", error_msg)
            return
            
        # Валидация входных данных
        try:
            if routes_df is None or routes_df.empty:
                raise ValueError("DataFrame с маршрутами пуст")
            if not norm_functions:
                raise ValueError("Функции норм не переданы")
                
            logger.info("✓ Валидация входных данных пройдена")
            
        except Exception as validation_error:
            error_msg = f"Ошибка валидации данных: {validation_error}"
            logger.error(error_msg)
            messagebox.showerror("Ошибка данных", error_msg)
            return
            
        # Сохраняем параметры
        self._current_data = {
            'section_name': section_name,
            'routes_df': routes_df,
            'norm_functions': norm_functions,
            'specific_norm_id': specific_norm_id,
            'single_section_only': single_section_only
        }
        
        # СИНХРОННОЕ создание графика с progress feedback
        try:
            self._create_plot_with_progress()
            
        except Exception as e:
            logger.error("КРИТИЧЕСКАЯ ошибка создания графика: %s", e, exc_info=True)
            self._show_error_message(f"Не удалось создать график: {str(e)}")
            
    def _create_plot_with_progress(self) -> None:
        """
        НОВЫЙ метод создания графика с синхронным progress feedback.
        Заменяет threading подход на поэтапное выполнение.
        """
        steps = [
            ("Подготовка данных...", self._prepare_plot_data),
            ("Создание структуры графика...", self._create_plot_structure),
            ("Добавление элементов...", self._add_plot_elements),
            ("Финализация графика...", self._finalize_plot),
        ]
        
        self._show_progress(True)
        
        def execute_step(step_index: int):
            try:
                if step_index >= len(steps):
                    # Завершение
                    self._plot_created = True
                    self._show_progress(False)
                    self._on_plot_completed()
                    return
                    
                step_text, step_func = steps[step_index]
                self._update_progress(step_text)
                
                # Выполняем шаг
                step_func()
                
                # Планируем следующий шаг через tkinter.after для отзывчивости UI
                self.window.after(50, lambda: execute_step(step_index + 1))
                
            except Exception as e:
                self._show_progress(False)
                logger.error("Ошибка на шаге %d ('%s'): %s", step_index, steps[step_index][0] if step_index < len(steps) else "unknown", e, exc_info=True)
                self._show_error_message(f"Ошибка создания графика: {str(e)}")
                
        # Запускаем выполнение первого шага
        execute_step(0)
        
    def _prepare_plot_data(self) -> None:
        """Шаг 1: Подготовка данных."""
        logger.debug("Шаг 1: Подготовка данных")
        # Данные уже подготовлены в self._current_data
        # Здесь можем добавить дополнительную валидацию если нужно
        pass
        
    def _create_plot_structure(self) -> None:
        """Шаг 2: Создание структуры графика."""
        logger.debug("Шаг 2: Создание структуры графика")
        if self.plot:
            self.plot._clear_all_plot_data()  # Очищаем предыдущий график
            
    def _add_plot_elements(self) -> None:
        """Шаг 3: Добавление элементов графика."""
        logger.debug("Шаг 3: Добавление элементов")
        if self.plot:
            self.plot.create_plot(**self._current_data)
            
    def _finalize_plot(self) -> None:
        """Шаг 4: Финализация графика."""
        logger.debug("Шаг 4: Финализация")
        if self.canvas:
            self.canvas.draw()
            
    def _show_progress(self, show: bool) -> None:
        """Показывает/скрывает progress bar."""
        try:
            if self.progress_bar:
                if show:
                    self.progress_bar.start(10)  # анимация
                else:
                    self.progress_bar.stop()
                    self.progress_var.set("График готов")
                    
        except Exception as e:
            logger.error("Ошибка управления progress bar: %s", e)
            
    def _update_progress(self, message: str) -> None:
        """Обновляет сообщение прогресса."""
        try:
            self.progress_var.set(message)
            self.window.update_idletasks()  # Принудительное обновление UI
            
        except Exception as e:
            logger.error("Ошибка обновления прогресса: %s", e)
            
    def _on_plot_completed(self) -> None:
        """Вызывается после успешного создания графика."""
        try:
            logger.info("✓ График успешно создан и отображен")
            
            # Обновляем UI
            self._update_progress("График готов к использованию")
            
            # Показываем краткую справку
            self._show_usage_hint()
            
        except Exception as e:
            logger.error("Ошибка завершения создания графика: %s", e)
            
    def _show_usage_hint(self) -> None:
        """Показывает подсказку по использованию."""
        try:
            hint_text = (
                "График готов!\n\n"
                "💡 Советы по использованию:\n"
                "• Переключайте режимы радио-кнопками\n"  
                "• Кликайте по точкам для деталей\n"
                "• Используйте колесо мыши для масштабирования\n"
                "• Экспортируйте в высоком качестве"
            )
            
            messagebox.showinfo("График готов", hint_text, parent=self.window)
            
        except Exception as e:
            logger.error("Ошибка показа подсказки: %s", e)
            
    def _show_error_message(self, message: str) -> None:
        """Показывает сообщение об ошибке."""
        try:
            messagebox.showerror("Ошибка графика", message, parent=self.window)
        except Exception:
            pass
            
    def _on_mode_changed_safe(self) -> None:
        """ИСПРАВЛЕННЫЙ обработчик изменения режима без threading."""
        if not self._initialized or not self.plot or not self._plot_created:
            return
            
        try:
            new_mode = DisplayMode(self.mode_var.get())
            logger.info("Смена режима на: %s", new_mode.value)
            
            # Показываем прогресс
            self.progress_var.set("Переключение режима...")
            self.window.update_idletasks()
            
            # Переключаем режим - синхронно!
            self.plot.switch_display_mode(new_mode)
            
            # Обновляем информационную метку
            mode_label = self._get_mode_label(new_mode)
            self.info_label.config(text=f"Текущий: {mode_label}")
            
            # Обновляем прогресс
            self.progress_var.set(f"Режим: {mode_label}")
            
            logger.info("✓ Режим переключен успешно")
            
        except Exception as e:
            logger.error("Ошибка переключения режима: %s", e, exc_info=True)
            
            # Показываем ошибку и возвращаем предыдущий режим
            messagebox.showerror("Ошибка", f"Ошибка переключения режима: {str(e)}", parent=self.window)
            
            # Возвращаем к WORK режиму как безопасному fallback
            self.mode_var.set(DisplayMode.WORK.value)
            self.info_label.config(text="Текущий: Уд. на работу")
            
    def _get_mode_label(self, mode: DisplayMode) -> str:
        """Получает человекочитаемое название режима."""
        labels = {
            DisplayMode.WORK: "Уд. на работу",
            DisplayMode.NF_RATIO: "Н/Ф (норма/факт)"
        }
        return labels.get(mode, str(mode.value))
            
    def _export_plot_safe(self, format_type: str) -> None:
        """Безопасный экспорт графика."""
        if not self.plot or not self._plot_created:
            messagebox.showwarning("Предупреждение", "Сначала создайте график", parent=self.window)
            return
            
        # Настройки форматов
        format_configs = {
            'png': {'ext': '.png', 'desc': 'PNG files', 'dpi': 300},
            'pdf': {'ext': '.pdf', 'desc': 'PDF files', 'dpi': 150}, 
            'svg': {'ext': '.svg', 'desc': 'SVG files', 'dpi': 150}
        }
        
        if format_type not in format_configs:
            messagebox.showerror("Ошибка", f"Неподдерживаемый формат: {format_type}", parent=self.window)
            return
            
        config = format_configs[format_type]
        
        # Диалог сохранения
        filename = filedialog.asksaveasfilename(
            title=f"Сохранить график как {format_type.upper()}",
            defaultextension=config['ext'],
            filetypes=[(config['desc'], f"*{config['ext']}"), ('All files', '*.*')],
            parent=self.window
        )
        
        if not filename:
            return
            
        try:
            success = self.plot.export_plot(filename, dpi=config['dpi'])
            
            if success:
                messagebox.showinfo(
                    "Успех", 
                    f"График сохранен:\n{Path(filename).name}",
                    parent=self.window
                )
            else:
                messagebox.showerror("Ошибка", "Не удалось сохранить график", parent=self.window)
                
        except Exception as e:
            logger.error("Ошибка экспорта: %s", e, exc_info=True)
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {str(e)}", parent=self.window)
            
    def _reset_zoom_safe(self) -> None:
        """Безопасный сброс масштаба."""
        if not self.plot:
            return
            
        try:
            # Автомасштабирование обеих осей
            for ax in [self.plot.ax1, self.plot.ax2]:
                if ax:
                    ax.relim()
                    ax.autoscale_view()
                    
            if self.canvas:
                self.canvas.draw_idle()
                
            self.progress_var.set("Масштаб сброшен")
            logger.info("✓ Масштаб графика сброшен")
            
        except Exception as e:
            logger.error("Ошибка сброса масштаба: %s", e)
            messagebox.showerror("Ошибка", f"Ошибка сброса масштаба: {str(e)}", parent=self.window)
            
    def _refresh_plot_safe(self) -> None:
        """Безопасное обновление графика."""
        if not self._current_data:
            messagebox.showwarning("Предупреждение", "Нет данных для обновления", parent=self.window)
            return
            
        logger.info("Обновление графика")
        
        try:
            # Пересоздаем график с теми же параметрами
            self._plot_created = False  # Сбрасываем флаг
            self.show_plot(**self._current_data)
            
        except Exception as e:
            logger.error("Ошибка обновления графика: %s", e, exc_info=True)
            messagebox.showerror("Ошибка", f"Ошибка обновления: {str(e)}", parent=self.window)
            
    def _on_window_close_safe(self) -> None:
        """Безопасное закрытие окна с полной очисткой ресурсов."""
        logger.info("Закрытие окна графика")
        
        try:
            # Останавливаем progress bar
            self._show_progress(False)
            
            # Очищаем matplotlib ресурсы
            if self.plot:
                self.plot = None
                
            if self.canvas:
                try:
                    self.canvas.get_tk_widget().destroy()
                except Exception:
                    pass
                self.canvas = None
                
            if self.toolbar:
                try:
                    self.toolbar.destroy()
                except Exception:
                    pass
                self.toolbar = None
                
            if self.figure:
                try:
                    plt.close(self.figure)
                except Exception:
                    pass
                self.figure = None
                
            # Закрываем окно
            if self.window:
                try:
                    self.window.grab_release()
                    self.window.destroy()
                except Exception as e:
                    logger.error("Ошибка уничтожения окна: %s", e)
                self.window = None
                
            # Сбрасываем флаги
            self._initialized = False
            self._plot_created = False
            
            logger.info("✓ Окно графика закрыто, ресурсы освобождены")
            
        except Exception as e:
            logger.error("Ошибка при закрытии окна: %s", e, exc_info=True)
            # В любом случае пытаемся закрыть окно
            try:
                if self.window:
                    self.window.destroy()
            except Exception:
                pass
                
    def is_active(self) -> bool:
        """Проверяет, активно ли окно графика."""
        try:
            return (self._initialized and 
                    self.window is not None and 
                    self.window.winfo_exists())
        except Exception:
            return False
            
    def bring_to_front(self) -> None:
        """Выводит окно на передний план."""
        try:
            if self.is_active():
                self.window.lift()
                self.window.focus_force()
                logger.debug("Окно графика выведено на передний план")
        except Exception as e:
            logger.error("Ошибка вывода окна на передний план: %s", e)
            
    # ========================== Дополнительные утилиты ==========================
            
    def get_plot_statistics(self) -> Dict:
        """Возвращает статистику текущего графика."""
        try:
            if not self.plot or not self._plot_created:
                return {'status': 'not_created'}
                
            stats = {
                'status': 'active',
                'current_mode': self.plot.mode_manager.get_current_mode().value,
                'traces_count': len(self.plot._traces_data),
                'norm_curves': len(self.plot._norm_lines),
                'norm_points': len(self.plot._norm_points),
            }
            
            # Добавляем статистику от менеджера режимов
            mode_stats = self.plot.mode_manager.get_statistics()
            stats.update(mode_stats)
            
            return stats
            
        except Exception as e:
            logger.error("Ошибка получения статистики графика: %s", e)
            return {'status': 'error', 'error': str(e)}
            
    def validate_plot_state(self) -> bool:
        """
        Валидирует состояние графика и пытается восстановить при проблемах.
        
        Returns:
            True если график в порядке или восстановлен
        """
        try:
            if not self.is_active():
                return False
                
            if not self.plot or not self._plot_created:
                return False
                
            # Проверяем основные компоненты
            if not self.plot.ax1 or not self.plot.ax2:
                logger.warning("Отсутствуют subplot'ы, пытаемся восстановить")
                self.plot._initialize_plots()
                
            # Проверяем canvas
            if not self.canvas:
                logger.warning("Canvas отсутствует")
                return False
                
            # Проверяем данные трасс
            if not self.plot._traces_data:
                logger.warning("Данные трасс отсутствуют")
                return False
                
            return True
            
        except Exception as e:
            logger.error("Ошибка валидации состояния графика: %s", e)
            return False
        
    def _handle_matplotlib_errors(self) -> None:
        """НОВЫЙ метод обработки специфических ошибок matplotlib."""
        try:
            # Проверяем backend
            current_backend = matplotlib.get_backend()
            if current_backend != 'TkAgg':
                logger.warning("Неправильный matplotlib backend: %s, переключаем на TkAgg", current_backend)
                matplotlib.use('TkAgg', force=True)
                
            # Проверяем состояние Figure
            if not self.figure:
                raise RuntimeError("Figure не инициализирован")
                
            # Проверяем subplot'ы
            if not self.plot or not self.plot.ax1 or not self.plot.ax2:
                logger.warning("Subplot'ы не инициализированы, восстанавливаем")
                if self.plot:
                    self.plot._initialize_plots()
                
        except Exception as e:
            logger.error("Ошибка обработки matplotlib ошибок: %s", e)
            raise