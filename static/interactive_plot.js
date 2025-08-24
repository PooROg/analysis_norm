// static/interactive_plot.js
class PlotModeController {
    constructor() {
        this.plotlyDiv = null;
        this.originalData = {};
        this.init();
    }

    init() {
        document.addEventListener('DOMContentLoaded', () => {
            setTimeout(() => this.initializePlotly(), 1000);
            this.setupModal();
            this.setupModeSwitch();
        });
    }

    initializePlotly() {
        this.plotlyDiv = document.getElementsByClassName('plotly-graph-div')[0];
        if (!this.plotlyDiv) {
            setTimeout(() => this.initializePlotly(), 2000);
            return;
        }
        if (!this.plotlyDiv.on) {
            setTimeout(() => this.initializePlotly(), 2000);
            return;
        }
        
        console.log("Инициализация Plotly...");
        this.plotlyDiv.on('plotly_click', (data) => this.handlePointClick(data));
        
        // ИСПРАВЛЕНО: сохранение исходных данных
        if (this.plotlyDiv.data) {
            this.plotlyDiv.data.forEach((trace, index) => {
                this.originalData[index] = {
                    x: [...(trace.x || [])],
                    y: [...(trace.y || [])],
                    customdata: trace.customdata ? JSON.parse(JSON.stringify(trace.customdata)) : null
                };
                console.log(`Сохранены данные трассы ${index}: ${trace.x?.length || 0} точек`);
            });
            console.log("Исходные данные сохранены для", Object.keys(this.originalData).length, "трасс");
        }
    }

    setupModal() {
        const modal = document.getElementById('route-modal');
        const closeBtn = document.getElementById('close-modal');
        
        if (closeBtn) {
            closeBtn.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                modal.style.display = 'none';
                return false;
            };
        }
        
        if (modal) {
            modal.onclick = (e) => {
                if (e.target === modal) modal.style.display = 'none';
            };
        }
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal && modal.style.display === 'block') {
                modal.style.display = 'none';
            }
        });
    }

    setupModeSwitch() {
        document.querySelectorAll('input[name="display_mode"]').forEach(radio => {
            radio.addEventListener('change', () => this.switchDisplayMode());
        });
    }

    switchDisplayMode() {
        const mode = document.querySelector('input[name="display_mode"]:checked')?.value;
        if (!mode || !this.plotlyDiv || !this.originalData) {
            console.log("Переключение режима невозможно:", { mode, plotlyDiv: !!this.plotlyDiv, originalData: !!Object.keys(this.originalData).length });
            return;
        }
    
        console.log("Переключение на режим:", mode);
        const update = {};
        let updatedTraces = 0;
    
        this.plotlyDiv.data.forEach((trace, index) => {
            if (!this.originalData[index] || !trace.customdata) {
                return;
            }
    
            // Проверяем, что это трасса с точками маршрутов
            if (!trace.name || 
                trace.name.includes("Норма") || 
                trace.name.includes("Базовые точки") ||
                trace.name.includes("Из маршрутов") ||
                trace.mode === "lines") {
                return;
            }
    
            console.log(`Обработка трассы ${index}: ${trace.name}, точек: ${trace.customdata.length}`);
    
            if (mode === 'nf') {
                // ИСПРАВЛЕННЫЙ режим Н/Ф с использованием СУММАРНЫХ данных
                const newY = this.originalData[index].y.map((originalY, i) => {
                    const customPoint = trace.customdata[i];
                    if (!customPoint) {
                        return originalY;
                    }
    
                    // ИСПРАВЛЕНО: используем суммарные значения для всего маршрута
                    const rashod_fact = this.safeFloat(customPoint.rashod_fact_total);
                    const rashod_norm = this.safeFloat(customPoint.rashod_norm_total);
                    const norm_interpolated = this.safeFloat(customPoint.norm_interpolated);
    
                    // Если нет суммарных данных, используем данные участка
                    const fact_val = rashod_fact > 0 ? rashod_fact : this.safeFloat(customPoint.rashod_fact);
                    const norm_val = rashod_norm > 0 ? rashod_norm : this.safeFloat(customPoint.rashod_norm);
    
                    if (!fact_val || !norm_val || !norm_interpolated || norm_val <= 0) {
                        console.log(`Точка ${i}: недостаточно данных (fact_total: ${rashod_fact}, norm_total: ${rashod_norm}, fact: ${fact_val}, norm: ${norm_val}, interp: ${norm_interpolated})`);
                        return originalY;
                    }
    
                    // Правильный расчет процентного отклонения от СУММАРНЫХ значений
                    const deviationPercent = ((fact_val - norm_val) / norm_val) * 100;
                    
                    // Применяем отклонение к интерполированной норме текущего участка
                    const adjustedY = norm_interpolated * (1 + deviationPercent / 100);
                    
                    console.log(`Точка ${i}: маршрут (fact: ${fact_val}, norm: ${norm_val}, откл: ${deviationPercent.toFixed(1)}%) -> ${originalY.toFixed(2)} -> ${adjustedY.toFixed(2)}`);
                    return adjustedY;
                });
                
                update[`y[${index}]`] = newY;
                updatedTraces++;
            } else {
                // Режим "Уд. на работу" - возвращаем исходные значения
                update[`y[${index}]`] = [...this.originalData[index].y];
                updatedTraces++;
            }
        });
    
        if (Object.keys(update).length > 0) {
            console.log(`Обновляем ${updatedTraces} трасс`);
            try {
                Plotly.restyle(this.plotlyDiv, update);
                console.log("График успешно обновлен");
            } catch (error) {
                console.error("Ошибка обновления графика:", error);
            }
        } else {
            console.log("Нет данных для обновления");
        }
    }

    safeFloat(value) {
        if (value === null || value === undefined || value === 'N/A' || value === '' || value === '-') {
            return 0;
        }
        
        // Преобразуем к строке и очищаем
        let strValue = String(value).replace(',', '.');
        const num = parseFloat(strValue);
        return isNaN(num) ? 0 : num;
    }

    handlePointClick(data) {
        if (!data.points?.length) return;
        const customData = data.points[0].customdata;
        if (!customData) {
            console.log("Нет customData для точки");
            return;
        }
        console.log("Клик по точке, данные:", customData);
        this.showFullRouteInfo(customData);
    }

    showFullRouteInfo(routeData) {
        const modalContent = this.buildRouteInfoHTML(routeData);
        const detailsElement = document.getElementById('route-details');
        const modalElement = document.getElementById('route-modal');
        
        if (detailsElement && modalElement) {
            detailsElement.innerHTML = modalContent;
            modalElement.style.display = 'block';
        }
    }

    buildRouteInfoHTML(c) {
        return `
            <h2>Подробная информация о маршруте №${c.route_number || 'N/A'}</h2>
            ${this.buildBasicInfoTable(c)}
            ${this.buildSectionsTable(c)}
            ${this.buildAnalysisTable(c)}
        `;
    }

    buildBasicInfoTable(c) {
        const basicFields = [
            ['Номер маршрута', c.route_number],
            ['Дата маршрута', c.route_date],
            ['Дата поездки', c.trip_date],
            ['Табельный машиниста', c.driver_tab],
            ['Серия локомотива', c.locomotive_series],
            ['Номер локомотива', c.locomotive_number],
            ['Расход фактический, всего', c.rashod_fact_total, c.use_red_rashod],
            ['Расход по норме, всего', c.rashod_norm_total, c.use_red_rashod]
        ];

        const rows = basicFields.map(([label, value, isRed = false]) => 
            this.buildTableRow(label, value, isRed)
        ).join('');

        return `
            <div style="margin-bottom:20px;">
                <h3>Основная информация</h3>
                <table style="border-collapse:collapse;width:50%;font-family:Arial;">
                    ${rows}
                </table>
            </div>
        `;
    }

    buildSectionsTable(c) {
        if (!c.all_sections?.length) {
            return `
                <div style="margin-bottom:20px;">
                    <h3>Информация по участкам</h3>
                    <div style="padding:20px;background:#fff3cd;border:1px solid #ffeaa7;border-radius:5px;color:#856404;">
                        <strong>⚠️ Предупреждение:</strong> Информация о дополнительных участках маршрута недоступна.
                    </div>
                </div>
            `;
        }
    
        // ИСПРАВЛЕННЫЙ список заголовков согласно требованиям
        const headers = [
            'Наименование участка', 'НЕТТО', 'БРУТТО', 'ОСИ', 'Номер нормы', 'Дв. тяга',
            'Ткм брутто', 'Км', 'Пр.', 'Расход фактический', 'Расход по норме',
            'Уд. норма, норма на 1 час ман. раб.', 'Нажатие на ось', 'Норма на работу',
            'Факт уд', 'Факт на работу', 'Норма на одиночное',
            'Простой с бригадой, мин., всего', 'Простой с бригадой, мин., норма',
            'Маневры, мин., всего', 'Маневры, мин., норма',
            'Трогание с места, случ., всего', 'Трогание с места, случ., норма',
            'Нагон опозданий, мин., всего', 'Нагон опозданий, мин., норма',
            'Ограничения скорости, случ., всего', 'Ограничения скорости, случ., норма',
            'На пересылаемые л-вы, всего', 'На пересылаемые л-вы, норма',
            'Количество дубликатов маршрута'
        ];
    
        const headerRow = headers.map(h => 
            `<td style="padding:4px;border:1px solid #ddd;text-align:center;font-size:9px;white-space:nowrap;font-weight:bold;">${h}</td>`
        ).join('');
    
        const dataRows = c.all_sections.map((section, i) => {
            const rowData = [
                section.section_name, section.netto, section.brutto, section.osi,
                section.norm_number, section.movement_type, section.tkm_brutto, section.km,
                section.pr, section.rashod_fact, section.rashod_norm, section.ud_norma,
                section.axle_load, section.norma_work, section.fact_ud, section.fact_work,
                section.norma_single, section.idle_brigada_total, section.idle_brigada_norm,
                section.manevr_total, section.manevr_norm, section.start_total, section.start_norm,
                section.delay_total, section.delay_norm, section.speed_limit_total, section.speed_limit_norm,
                section.transfer_loco_total, section.transfer_loco_norm, section.duplicates_count
            ];
    
            const cells = rowData.map((value, idx) => {
                let style = 'padding:4px;border:1px solid #ddd;text-align:center;font-size:10px;white-space:nowrap;';
                
                if (['НЕТТО', 'БРУТТО', 'ОСИ'].includes(headers[idx]) && section.use_red_color) {
                    style += ' background-color:#ffcccc;color:#f00;font-weight:bold;';
                }
                if (['Расход фактический', 'Расход по норме'].includes(headers[idx]) && section.use_red_rashod) {
                    style += ' background-color:#ffcccc;color:#f00;font-weight:bold;';
                }
    
                const displayValue = (value ?? '-') !== 'N/A' ? (value ?? '-') : '-';
                return `<td style="${style}">${displayValue}</td>`;
            }).join('');
    
            const bgColor = i % 2 === 0 ? '#fff' : '#f9f9f9';
            return `<tr style="background-color:${bgColor};">${cells}</tr>`;
        }).join('');
    
        // Итоговая строка с суммами
        let totalsRow = '';
        if (c.totals) {
            const totalsData = [
                'ИТОГО', '-', '-', '-', '-', '-',  // Первые 6 колонок
                c.totals.tkm_brutto > 0 ? c.totals.tkm_brutto.toFixed(1) : '-',
                c.totals.km > 0 ? c.totals.km.toFixed(1) : '-',
                c.totals.pr > 0 ? c.totals.pr.toFixed(1) : '-',
                c.totals.rashod_fact > 0 ? c.totals.rashod_fact.toFixed(1) : '-',
                c.totals.rashod_norm > 0 ? c.totals.rashod_norm.toFixed(1) : '-',
                c.totals.ud_norma > 0 ? c.totals.ud_norma.toFixed(1) : '-',
                c.totals.axle_load > 0 ? c.totals.axle_load.toFixed(1) : '-',
                c.totals.norma_work > 0 ? c.totals.norma_work.toFixed(1) : '-',
                c.totals.fact_ud > 0 ? c.totals.fact_ud.toFixed(1) : '-',
                c.totals.fact_work > 0 ? c.totals.fact_work.toFixed(1) : '-',
                c.totals.norma_single > 0 ? c.totals.norma_single.toFixed(1) : '-',
                c.totals.idle_brigada_total > 0 ? c.totals.idle_brigada_total.toFixed(1) : '-',
                c.totals.idle_brigada_norm > 0 ? c.totals.idle_brigada_norm.toFixed(1) : '-',
                c.totals.manevr_total > 0 ? c.totals.manevr_total.toFixed(1) : '-',
                c.totals.manevr_norm > 0 ? c.totals.manevr_norm.toFixed(1) : '-',
                c.totals.start_total > 0 ? c.totals.start_total.toFixed(1) : '-',
                c.totals.start_norm > 0 ? c.totals.start_norm.toFixed(1) : '-',
                c.totals.delay_total > 0 ? c.totals.delay_total.toFixed(1) : '-',
                c.totals.delay_norm > 0 ? c.totals.delay_norm.toFixed(1) : '-',
                c.totals.speed_limit_total > 0 ? c.totals.speed_limit_total.toFixed(1) : '-',
                c.totals.speed_limit_norm > 0 ? c.totals.speed_limit_norm.toFixed(1) : '-',
                c.totals.transfer_loco_total > 0 ? c.totals.transfer_loco_total.toFixed(1) : '-',
                c.totals.transfer_loco_norm > 0 ? c.totals.transfer_loco_norm.toFixed(1) : '-',
                '-'  // Количество дубликатов
            ];
    
            const totalsCells = totalsData.map(value => 
                `<td style="padding:4px;border:1px solid #ddd;text-align:center;font-size:10px;white-space:nowrap;font-weight:bold;background-color:#e6f3ff;">${value}</td>`
            ).join('');
    
            totalsRow = `<tr style="background-color:#e6f3ff;">${totalsCells}</tr>`;
        }
    
        return `
            <div style="margin-bottom:20px;">
                <h3>Информация по участкам (всего: ${c.all_sections.length})</h3>
                <div style="overflow-x:auto;max-width:100%;">
                    <table style="border-collapse:collapse;width:100%;font-family:Arial;font-size:10px;min-width:2500px;">
                        <tr style="background-color:#f0f0f0;">${headerRow}</tr>
                        ${dataRows}
                        ${totalsRow}
                    </table>
                </div>
            </div>
        `;
    }

    buildAnalysisTable(c) {
        const analysisFields = [
            ['Норма интерполированная', c.norm_interpolated],
            ['Отклонение, %', c.deviation_percent],
            ['Статус', c.status],
            ['Н=Ф', c.n_equals_f]
        ];

        if (c.coefficient && c.coefficient !== 1.0) {
            analysisFields.push(['Коэффициент', c.coefficient]);
            if (c.fact_ud_original) {
                analysisFields.push(['Факт. удельный исходный', c.fact_ud_original]);
            }
        }

        const rows = analysisFields.map(([label, value]) => 
            this.buildTableRow(label, value)
        ).join('');

        return `
            <div style="margin-bottom:20px;">
                <h3>Результаты анализа (для текущего участка)</h3>
                <table style="border-collapse:collapse;width:50%;font-family:Arial;">
                    ${rows}
                </table>
            </div>
        `;
    }

    buildTableRow(label, value, isRed = false) {
        const redStyle = isRed ? 'background-color:#ffcccc;color:#f00;font-weight:bold;' : '';
        const displayValue = (value !== null && value !== undefined && value !== 'N/A') ? value : '-';
        
        return `
            <tr style="border:1px solid #ddd;">
                <td style="padding:8px;border:1px solid #ddd;background-color:#f5f5f5;font-weight:bold;">${label}</td>
                <td style="padding:8px;border:1px solid #ddd;${redStyle}">${displayValue}</td>
            </tr>
        `;
    }
}

// Инициализация
new PlotModeController();