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
        
        // ИСПРАВЛЕНО: принудительное сохранение ВСЕХ трасс
        this.saveOriginalData();
    }

    // ИСПРАВЛЕННЫЙ метод - основная проблема была здесь
    saveOriginalData() {
        console.log("=== ПРИНУДИТЕЛЬНОЕ СОХРАНЕНИЕ ВСЕХ ТРАСС ===");
        
        if (!this.plotlyDiv || !this.plotlyDiv.data) {
            console.log("❌ Данные графика недоступны, повторная попытка через 1 секунду");
            setTimeout(() => this.saveOriginalData(), 1000);
            return;
        }
    
        console.log("Всего трасс для сохранения:", this.plotlyDiv.data.length);
        
        // ОЧИЩАЕМ старые данные
        this.originalData = {};
        
        // ПРИНУДИТЕЛЬНО сохраняем ВСЕ трассы
        this.plotlyDiv.data.forEach((trace, index) => {
            console.log(`Сохранение трассы ${index}:`);
            console.log(`  - Имя: ${trace.name || 'undefined'}`);
            console.log(`  - Режим: ${trace.mode || 'undefined'}`);
            console.log(`  - X точек: ${trace.x?.length || 0}`);
            console.log(`  - Y точек: ${trace.y?.length || 0}`);
            console.log(`  - customdata: ${trace.customdata ? 'есть' : 'нет'} (${trace.customdata?.length || 0} элементов)`);
            
            // ИСПРАВЛЕНО: Безопасное копирование массивов Plotly
            this.originalData[index] = {
                x: this.safeArrayCopy(trace.x),
                y: this.safeArrayCopy(trace.y),
                customdata: this.safeDeepCopy(trace.customdata),
                name: trace.name || `trace_${index}`,
                mode: trace.mode || 'markers'
            };
            
            console.log(`  ✅ Сохранено: x=${this.originalData[index].x.length}, y=${this.originalData[index].y.length}`);
        });
        
        console.log(`🎉 Все ${Object.keys(this.originalData).length} трасс сохранены принудительно`);
        
        // ПРОВЕРЯЕМ что трассы маршрутов сохранились
        const routeTraces = [];
        Object.keys(this.originalData).forEach(index => {
            const data = this.originalData[index];
            const name = data.name || '';
            if (name.includes('Экономия') || name.includes('Перерасход') || name === 'Норма') {
                routeTraces.push(`${index}: ${name} (${data.y.length} точек)`);
            }
        });
        
        console.log("🔍 Трассы маршрутов в originalData:", routeTraces);
        
        if (routeTraces.length === 0) {
            console.log("⚠️ КРИТИЧНО: Трассы маршрутов не найдены, повтор через 2 секунды");
            setTimeout(() => this.saveOriginalData(), 2000);
        } else {
            console.log("✅ Трассы маршрутов успешно сохранены");
        }
    }

    // НОВЫЙ метод: Безопасное копирование массивов Plotly
    safeArrayCopy(data) {
        if (!data) return [];
        
        try {
            // Проверяем является ли это массивом
            if (Array.isArray(data)) {
                return [...data];
            }
            
            // Проверяем TypedArray (Float32Array, Float64Array и т.д.)
            if (data.constructor && data.constructor.name.includes('Array')) {
                return Array.from(data);
            }
            
            // Проверяем есть ли length и можно ли итерировать
            if (data.length !== undefined && typeof data[Symbol.iterator] === 'function') {
                return Array.from(data);
            }
            
            // Попытка конвертации через Object.values если это объект с числовыми ключами
            if (typeof data === 'object' && data !== null) {
                const keys = Object.keys(data);
                if (keys.length > 0 && keys.every(k => !isNaN(k))) {
                    return keys.map(k => data[k]);
                }
            }
            
            console.warn("Не удалось определить тип данных:", typeof data, data);
            return [];
            
        } catch (error) {
            console.error("Ошибка копирования массива:", error);
            return [];
        }
    }

    // НОВЫЙ метод: Безопасное глубокое копирование
    safeDeepCopy(data) {
        if (!data) return null;
        try {
            return JSON.parse(JSON.stringify(data));
        } catch (error) {
            console.warn("Ошибка глубокого копирования:", error);
            return null;
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
            console.log("Переключение режима невозможно:", { 
                mode, 
                plotlyDiv: !!this.plotlyDiv, 
                originalData: !!Object.keys(this.originalData).length 
            });
            return;
        }
    
        // ДОБАВЛЕНО: Проверка что исходные данные актуальны
        if (Object.keys(this.originalData).length < this.plotlyDiv.data.length) {
            console.log("⚠️ КРИТИЧНО: Исходные данные неполные, принудительное пересохранение");
            this.saveOriginalData();
            setTimeout(() => this.switchDisplayMode(), 1000);
            return;
        }
    
        console.log("Переключение на режим:", mode);
        console.log("Всего трасс в графике:", this.plotlyDiv.data.length);
        console.log("Исходных данных сохранено:", Object.keys(this.originalData).length);
        
        const update = {};
        let updatedTraces = 0;
    
        this.plotlyDiv.data.forEach((trace, index) => {
            console.log(`\n=== Обработка трассы ${index} ===`);
            console.log("Имя трассы:", trace.name);
            console.log("Есть исходные данные:", !!this.originalData[index]);
            console.log("Есть customdata:", !!trace.customdata);
            
            // КРИТИЧНО: Проверяем что это трасса маршрутов
            const traceName = trace.name || '';
            const isRouteTrace = (
                traceName.includes('Экономия') || 
                traceName.includes('Перерасход') || 
                traceName === 'Норма'
            ) && !traceName.includes('('); // Исключаем дубли с скобками
            
            console.log("Это трасса маршрутов:", isRouteTrace);
            
            if (!this.originalData[index]) {
                console.log(`❌ Трасса ${index} пропущена: нет исходных данных`);
                return;
            }
            
            if (!isRouteTrace) {
                console.log(`❌ Трасса ${index} пропущена: не является трассой маршрутов`);
                return;
            }
            
            if (!trace.customdata) {
                console.log(`❌ Трасса ${index} пропущена: нет customdata`);
                return;
            }
    
            console.log(`✅ Трасса ${index} проходит все проверки - обрабатываем`);
    
            if (mode === 'nf') {
                // Режим Н/Ф
                const newY = this.originalData[index].y.map((originalY, i) => {
                    const customPoint = trace.customdata[i];
                    if (!customPoint) return originalY;
    
                    const expected_nf_y = this.safeFloat(customPoint.expected_nf_y);
                    if (expected_nf_y > 0) {
                        console.log(`Точка ${i}: используем предрасчитанное значение ${expected_nf_y.toFixed(2)}`);
                        return expected_nf_y;
                    }
    
                    const rashod_fact_total = this.safeFloat(customPoint.rashod_fact_total);
                    const rashod_norm_total = this.safeFloat(customPoint.rashod_norm_total);
                    const ud_norma_original = this.safeFloat(customPoint.ud_norma_original);
    
                    if (rashod_fact_total > 0 && rashod_norm_total > 0 && ud_norma_original > 0) {
                        const adjustedY = (rashod_fact_total / rashod_norm_total) * ud_norma_original;
                        console.log(`Точка ${i}: расчет ${originalY.toFixed(2)} -> ${adjustedY.toFixed(2)}`);
                        return adjustedY;
                    }
    
                    return originalY;
                });
                
                update[`y[${index}]`] = newY;
                updatedTraces++;
            } else {
                // Режим "Уд. на работу"
                update[`y[${index}]`] = this.safeArrayCopy(this.originalData[index].y);
                updatedTraces++;
            }
        });
    
        console.log(`\n🔄 Обновляем ${updatedTraces} трасс`);
        
        if (Object.keys(update).length > 0) {
            try {
                Plotly.restyle(this.plotlyDiv, update);
                console.log("✅ График успешно обновлен");
            } catch (error) {
                console.error("❌ Ошибка обновления графика:", error);
            }
        } else {
            console.log("❌ Нет данных для обновления");
        }
    }

    safeFloat(value) {
        if (value === null || value === undefined || value === 'N/A' || value === '' || value === '-') {
            return 0;
        }
        
        let strValue = String(value).replace(',', '.');
        const num = parseFloat(strValue);
        return isNaN(num) ? 0 : num;
    }

    handlePointClick(data) {
        if (!data.points?.length) return;
        
        const customData = data.points[0].customdata;
        console.log("Тип customData:", typeof customData);
        console.log("customData:", customData);
        
        if (!customData) {
            console.error("Нет customData для точки");
            return;
        }
        
        if (typeof customData === 'string') {
            console.error("customData является строкой вместо объекта:", customData.substring(0, 200));
            return;
        }
        
        if (typeof customData !== 'object') {
            console.error("customData имеет неправильный тип:", typeof customData);
            return;
        }
        
        console.log("customData корректен, показываем модальное окно");
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
                <table style="border-collapse:collapse;width:70%;font-family:Arial;">
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
    
        return `
            <div style="margin-bottom:20px;">
                <h3>Информация по участкам (всего: ${c.all_sections.length})</h3>
                <div style="overflow-x:auto;max-width:100%;">
                    <table style="border-collapse:collapse;width:100%;font-family:Arial;font-size:10px;min-width:2500px;">
                        <tr style="background-color:#f0f0f0;">${headerRow}</tr>
                        ${dataRows}
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

        if (c.coefficient_section !== null && c.coefficient_section !== undefined) {
            analysisFields.push(['Коэффициент участка (Факт/Норма)', c.coefficient_section.toFixed(6)]);
        }
        
        if (c.coefficient_route !== null && c.coefficient_route !== undefined) {
            analysisFields.push(['Коэффициент маршрута (Расх.факт.всего/Расх.норма.всего)', c.coefficient_route.toFixed(6)]);
        }
        
        if (c.expected_nf_y !== null && c.expected_nf_y !== undefined) {
            analysisFields.push(['Ожидаемая Y в режиме Н/Ф', c.expected_nf_y.toFixed(3)]);
        }

        const rows = analysisFields.map(([label, value]) => 
            this.buildTableRow(label, value)
        ).join('');

        return `
            <div style="margin-bottom:20px;">
                <h3>Результаты анализа (для текущего участка)</h3>
                <table style="border-collapse:collapse;width:70%;font-family:Arial;">
                    ${rows}
                </table>
            </div>
        `;
    }
    
}

// Инициализация
new PlotModeController();