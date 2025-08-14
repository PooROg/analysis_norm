# analysis/analyzer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d, CubicSpline
import warnings
warnings.filterwarnings('ignore')
from core.filter import LocomotiveFilter
from core.coefficients import LocomotiveCoefficientsManager

class InteractiveNormsAnalyzer:
    """Класс для интерактивного анализа норм расхода электроэнергии"""
    
    def __init__(self, rf='Processed_Routes.xlsx', nf='Нормы участков.xlsx'):
        self.rf = rf
        self.nf = nf
        self.rdf = None
        self.nd = {}
        self.ar = {}
        
    def load_data(self):
        try:
            print("\n[1] Загрузка данных маршрутов...")
            self.rdf = pd.read_excel(self.rf)
            print("    Фильтрация маршрутов с одним участком...")
            rc = self.rdf.groupby(['Номер маршрута', 'Дата маршрута']).size()
            sr = rc[rc == 1].index
            self.rdf = self.rdf.set_index(['Номер маршрута', 'Дата маршрута']).loc[sr].reset_index()
            if 'Нажатие на ось' not in self.rdf.columns:
                self.rdf['Нажатие на ось'] = self.rdf['БРУТТО'] / self.rdf['ОСИ']
            ic = len(self.rdf)
            self.rdf = self.rdf[self.rdf['Номер нормы'].notna()]
            self.rdf['Номер нормы'] = self.rdf['Номер нормы'].astype(int)
            print(f"    Загружено маршрутов: {ic}")
            print(f"    С указанной нормой: {len(self.rdf)}")
            print(f"    Уникальных участков: {self.rdf['Наименование участка'].nunique()}")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return False
    
    def load_norms(self):
        try:
            print("\n[2] Загрузка норм...")
            ef = pd.ExcelFile(self.nf)
            for sn in ef.sheet_names:
                print(f"\n    Обработка листа: {sn}")
                df = pd.read_excel(self.nf, sheet_name=sn, header=None)
                shn = self.parse_norms_from_sheet(df, sn)
                if shn:
                    self.nd[sn] = shn
                    print(f"      Найдено норм: {len(shn)}")
                    for ni, nd in shn.items():
                        print(f"        Норма №{ni}: {len(nd['points'])} точек")
            if not self.nd:
                print("❌ Не найдено ни одной нормы в файле")
                return False
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки норм: {e}")
            return False
    
    def parse_norms_from_sheet(self, df, sn):
        norms = {}
        for ri in range(len(df)):
            fc = str(df.iloc[ri, 0]) if pd.notna(df.iloc[ri, 0]) else ""
            if "Норма №" in fc:
                try:
                    nn = int(fc.split("№")[1].strip())
                    lr = df.iloc[ri + 2, 1:]
                    cr = df.iloc[ri + 3, 1:]
                    pts = []
                    for i in range(len(lr)):
                        if pd.notna(lr.iloc[i]) and pd.notna(cr.iloc[i]):
                            try:
                                lv = float(lr.iloc[i])
                                cv = float(cr.iloc[i])
                                pts.append((lv, cv))
                            except:
                                continue
                    if len(pts) >= 2:
                        pts.sort(key=lambda x: x[0])
                        norms[nn] = {
                            'points': pts,
                            'description': str(df.iloc[ri + 1, 0]) if pd.notna(df.iloc[ri + 1, 0]) else ""
                        }
                except:
                    continue
        return norms
    
    def analyze_section(self, sn, rdf, norms):
        nf = {}
        for ni, nd in norms.items():
            pts = nd['points']
            xv = [p[0] for p in pts]
            yv = [p[1] for p in pts]
            if len(pts) == 2:
                inf = interp1d(xv, yv, kind='linear', fill_value='extrapolate', bounds_error=False)
            else:
                try:
                    inf = CubicSpline(xv, yv, bc_type='natural')
                except:
                    inf = interp1d(xv, yv, kind='quadratic' if len(pts) > 2 else 'linear', fill_value='extrapolate', bounds_error=False)
            nf[ni] = {
                'function': inf,
                'points': pts,
                'x_range': (min(xv), max(xv))
            }
        rdf['Норма интерполированная'] = 0.0
        rdf['Отклонение, %'] = 0.0
        rdf['Статус'] = 'Не определен'
        for i, r in rdf.iterrows():
            ni = r['Номер нормы']
            if ni in nf:
                try:
                    xv = r['Нажатие на ось']
                    nfunc = nf[ni]['function']
                    nval = float(nfunc(xv))
                    rdf.loc[i, 'Норма интерполированная'] = nval
                    fval = r['Фактический удельный']
                    if nval > 0:
                        dev = ((fval - nval) / nval) * 100
                        rdf.loc[i, 'Отклонение, %'] = dev
                        if dev < -5:
                            rdf.loc[i, 'Статус'] = 'Экономия'
                        elif dev > 5:
                            rdf.loc[i, 'Статус'] = 'Перерасход'
                        else:
                            rdf.loc[i, 'Статус'] = 'Норма'
                except:
                    continue
        self.ar[sn] = {
            'routes': rdf,
            'norms': norms,
            'norm_functions': nf
        }
        return rdf, nf
    
    def analyze_section_with_filters(self, sn, rdf, norms, lf=None, cm=None, uc=False):
        print(f"Debug: analyze_section_with_filters вызван. uc={uc}, cm есть коэфф={bool(cm and cm.coef)}")
        if lf:
            rdf = lf.filter_routes(rdf)
            if rdf.empty:
                return rdf, None
        rdf = rdf.copy()
        if uc and cm and cm.coef:  # Проверяем что коэффициенты загружены
            print(f"Debug: Применяем коэффициенты. Загружено коэфф: {len(cm.coef)}")
            rdf['Коэффициент'] = 1.0
            rdf['Факт. удельный исходный'] = rdf['Фактический удельный']
            applied_count = 0
            for i, r in rdf.iterrows():
                if 'Серия локомотива' in rdf.columns and 'Номер локомотива' in rdf.columns:
                    s = str(r['Серия локомотива']) if pd.notna(r['Серия локомотива']) else ''
                    n = r['Номер локомотива']
                    if s and pd.notna(n):
                        try:
                            if isinstance(n, str):
                                n = int(n.lstrip('0')) if n.strip().lstrip('0') else 0
                            else:
                                n = int(n)
                            co = cm.get_coefficient(s, n)
                            rdf.at[i, 'Коэффициент'] = co
                            if co != 1.0:  # Применяем коэффициент только если он не равен 1.0
                                rdf.at[i, 'Фактический удельный'] = rdf.at[i, 'Фактический удельный'] / co
                                applied_count += 1
                                if applied_count <= 3:  # Показываем первые 3 для отладки
                                    print(f"Debug: Применен коэфф {co:.3f} к локомотиву {s} №{n}")
                        except (ValueError, TypeError) as e:
                            print(f"Debug: Ошибка обработки локомотива: {e}")
                            continue
            print(f"Debug: Применено коэффициентов: {applied_count}")
        else:
            print("Debug: Коэффициенты НЕ применяются")
        return self.analyze_section(sn, rdf, norms)
    
    def create_interactive_plot(self, sn, ra, nf):
        section_name = sn
        routes = ra
        norms = nf
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=(
                f"Нормы расхода для участка: {section_name}",
                "Отклонение фактического расхода от нормы"
            )
        )

        # Отрисовка кривых норм на верхнем графике
        for ni, nf_item in norms.items():
            pts = nf_item['points']
            xv = [p[0] for p in pts]
            yv = [p[1] for p in pts]
            x_interp = np.linspace(min(xv), max(xv), 100)
            nfunc = nf_item['function']
            y_interp = nfunc(x_interp)
            fig.add_trace(
                go.Scatter(
                    x=x_interp,
                    y=y_interp,
                    mode='lines',
                    name=f'Норма №{ni}',
                    line=dict(width=2),
                    hovertemplate='Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм'
                ), row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=xv,
                    y=yv,
                    mode='markers',
                    marker=dict(symbol='square', size=8, color='black'),
                    name=f'Опорные точки нормы №{ni}',
                    hovertemplate='Опорная точка<br>Нажатие: %{x:.2f} т/ось<br>Норма: %{y:.1f} кВт·ч/10⁴ ткм'
                ), row=1, col=1
            )

        # Определение vr один раз
        vr = routes[routes['Статус'] != 'Не определен']

        # Добавление фактических точек маршрутов на верхний график
        for _, r in vr.iterrows():
            # Цвет точки в зависимости от отклонения
            if r['Отклонение, %'] >= 30:
                color = '#7C3AED'
            elif r['Отклонение, %'] >= 20:
                color = '#9333EA'
            elif r['Отклонение, %'] >= 5:
                color = '#06B6D4'
            elif r['Отклонение, %'] >= -5:
                color = '#22C55E'
            elif r['Отклонение, %'] >= -20:
                color = '#EAB308'
            elif r['Отклонение, %'] >= -30:
                color = '#F97316'
            else:
                color = '#DC2626'
            
            hover_text = (
                f"Маршрут №{r['Номер маршрута']}<br>"
                f"Дата: {r['Дата маршрута']}<br>"
                f"Локомотив: {r.get('Серия локомотива', '')} №{r.get('Номер локомотива', '')}<br>"
            )
            
            # Добавляем информацию о коэффициенте, если он есть
            if 'Коэффициент' in r.index and pd.notna(r['Коэффициент']) and r['Коэффициент'] != 1.0:
                hover_text += f"Коэффициент: {r['Коэффициент']:.3f}<br>"
                if 'Факт. удельный исходный' in r.index and pd.notna(r['Факт. удельный исходный']):
                    hover_text += f"Факт исходный: {r['Факт. удельный исходный']:.1f}<br>"
            
            hover_text += (
                f"Нажатие: {r['Нажатие на ось']:.2f} т/ось<br>"
                f"Факт: {r['Фактический удельный']:.1f}<br>"
                f"Норма: {r['Норма интерполированная']:.1f}<br>"
                f"Отклонение: {r['Отклонение, %']:.1f}%"
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[r['Нажатие на ось']],
                    y=[r['Фактический удельный']],
                    mode='markers',
                    marker=dict(color=color, size=8, opacity=0.8, line=dict(color='black', width=0.5)),
                    hovertemplate=hover_text,
                    showlegend=False
                ), row=1, col=1
            )

        # Группировка для нижнего графика
        dg = {
            'Экономия +30% и более': vr[vr['Отклонение, %'] >= 30],
            'Экономия +20% до +30%': vr[(vr['Отклонение, %'] >= 20) & (vr['Отклонение, %'] < 30)],
            'Экономия +5% до +20%': vr[(vr['Отклонение, %'] >= 5) & (vr['Отклонение, %'] < 20)],
            'Норма -5% до +5%': vr[(vr['Отклонение, %'] >= -5) & (vr['Отклонение, %'] < 5)],
            'Перерасход -5% до -20%': vr[(vr['Отклонение, %'] >= -20) & (vr['Отклонение, %'] < -5)],
            'Перерасход -20% до -30%': vr[(vr['Отклонение, %'] >= -30) & (vr['Отклонение, %'] < -20)],
            'Перерасход -30% и менее': vr[vr['Отклонение, %'] < -30]
        }
        gc = {
            'Экономия +30% и более': '#7C3AED',
            'Экономия +20% до +30%': '#9333EA',
            'Экономия +5% до +20%': '#06B6D4',
            'Норма -5% до +5%': '#22C55E',
            'Перерасход -5% до -20%': '#EAB308',
            'Перерасход -20% до -30%': '#F97316',
            'Перерасход -30% и менее': '#DC2626'
        }
        
        for gn, gd in dg.items():
            if len(gd) > 0:
                ht = []
                for _, r in gd.iterrows():
                    txt = f"Маршрут №{r['Номер маршрута']}<br>Дата: {r['Дата маршрута']}<br>Норма №{r['Номер нормы']}<br>"
                    if 'Серия локомотива' in r and pd.notna(r['Серия локомотива']):
                        txt += f"Локомотив: {r['Серия локомотива']} "
                    if 'Номер локомотива' in r and pd.notna(r['Номер локомотива']):
                        txt += f"№{r['Номер локомотива']}<br>"
                    else:
                        txt += "<br>"
                    
                    # Добавляем информацию о коэффициенте
                    if 'Коэффициент' in r.index and pd.notna(r['Коэффициент']) and r['Коэффициент'] != 1.0:
                        txt += f"Коэффициент: {r['Коэффициент']:.3f}<br>"
                        if 'Факт. удельный исходный' in r.index and pd.notna(r['Факт. удельный исходный']):
                            txt += f"Факт исходный: {r['Факт. удельный исходный']:.1f}<br>"
                    
                    txt += f"Нажатие: {r['Нажатие на ось']:.2f} т/ось<br>Отклонение: {r['Отклонение, %']:.1f}%"
                    ht.append(txt)
                fig.add_trace(
                    go.Scatter(
                        x=gd['Нажатие на ось'], y=gd['Отклонение, %'],
                        mode='markers',
                        name=f'{gn} ({len(gd)})',
                        marker=dict(color=gc[gn], size=10, opacity=0.8, line=dict(color='black', width=0.5)),
                        hovertemplate='%{text}',
                        text=ht
                    ), row=2, col=1
                )

        # Добавление линий границ на нижний график
        xr = [vr['Нажатие на ось'].min() - 1, vr['Нажатие на ось'].max() + 1]
        fig.add_trace(go.Scatter(x=xr, y=[5, 5], mode='lines', line=dict(color='#22C55E', dash='dash', width=2), showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=xr, y=[-5, -5], mode='lines', line=dict(color='#22C55E', dash='dash', width=2), showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=xr, y=[20, 20], mode='lines', line=dict(color='#F97316', dash='dot', width=2), showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=xr, y=[-20, -20], mode='lines', line=dict(color='#F97316', dash='dot', width=2), showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=xr, y=[0, 0], mode='lines', line=dict(color='black', width=1), showlegend=False, hoverinfo='skip'), row=2, col=1)
        fig.add_trace(go.Scatter(x=xr + xr[::-1], y=[-5, -5, 5, 5], fill='toself', fillcolor='rgba(34, 197, 94, 0.1)', line=dict(color='rgba(255,255,255,0)'), showlegend=False, hoverinfo='skip'), row=2, col=1)

        # Обновление осей и layout
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=1, col=1)
        fig.update_yaxes(title_text="Удельный расход, кВт·ч/10⁴ ткм брутто", row=1, col=1)
        fig.update_xaxes(title_text="Нажатие на ось, т/ось", row=2, col=1)
        fig.update_yaxes(title_text="Отклонение от нормы, %", row=2, col=1)
        fig.update_layout(height=1000, showlegend=True, hovermode='closest', template='plotly_white', legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02))
        return fig
    
    def get_sections_list(self):
        if self.rdf is None:
            return []
        return self.rdf['Наименование участка'].unique().tolist()
    
    def analyze_single_section(self, sn):
        if sn not in self.nd:
            return None, None, "Нормы для участка не найдены"
        sr = self.rdf[self.rdf['Наименование участка'] == sn].copy()
        if sr.empty:
            return None, None, "Нет маршрутов для участка"
        ra, nf = self.analyze_section(sn, sr, self.nd[sn])
        fig = self.create_interactive_plot(sn, ra, nf)
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
        return fig, st, None