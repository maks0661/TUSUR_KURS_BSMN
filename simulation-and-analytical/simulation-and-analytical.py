import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams['font.family'] = 'DejaVu Sans' 
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
# --- Параметры тепловой модели (подраздел 2.3) ----------------
C1, C2 = 150.0, 6.0            # теплоемкости оболочки и образца, Дж/К
Rc, Rth = 20.0, 60.0 # тепловые сопротивления оболочки и среды, К/Вт
dP = 0.01                      # ступенька мощности нагревателя, Вт
t_end = 12540.0                # длительность моделирования = 4·Ta, с
KdP = Rc * dP                  # статический отклик K·dP = 20·0,01 = 0,2 К
# Матрицы системы (2.7.2): dtheta/dt = A·theta + B·u, u = [dP, theta_a]
A = np.array([[-(1/Rc + 1/Rth)/C1, 1/(Rth*C1)],
              [1/(Rth*C2),        -1/(Rth*C2)]])
B = np.array([[1/C1, 1/(Rc*C1)], [0.0, 0.0]])
# Постоянные времени – корни характеристического полинома a2*s^2 + a1*s + 1
a1 = Rc*(C1 + C2) + Rth*C2                   # a1 = 3480 с
a2 = Rc * Rth * C1 * C2                      # a2 = 1,08·10^6 с^2
Ta = (a1 + np.sqrt(a1*a1 - 4*a2)) / 2        # Ta = 3135,6 с
Tb = (a1 - np.sqrt(a1*a1 - 4*a2)) / 2        # Tb = 344,4 с
# Общая сетка времени с шагом 1 с – на ней сравниваются все три решения
t = np.linspace(0.0, t_end, int(t_end) + 1)
def load_xcos_csv(fname, t_grid):
    def parse(line):
        cands = [line.split(';'), line.split('\t'), line.split(',')]
        if len(line.split()) >= 2:
            cands.append(line.split())
        for parts in cands:
            vals, ok = [], True
            for p in parts:
                p = p.strip().strip('"')
                if p.count(',') == 1 and '.' not in p:   # десятичная запятая
                    p = p.replace(',', '.')
                try:
                    vals.append(float(p))
                except ValueError: # шапка/текст – пропустить
                    ok = False
                    break
            if ok and len(vals) >= 2:
                return vals
        return None
    if not os.path.exists(fname):
        return None
    rows = [v for v in (parse(s) for s in
            open(fname, encoding='utf-8-sig', errors='replace')) if v]
    why = None
    th_i = None
    if len(rows) < 1000:
        why = f'числовых строк всего {len(rows)}, а для записи с шагом 1 с нужно ~{int(t_end)+1}'
    if why is None:
        ncol = max(len(r) for r in rows)
        arr = np.array([r for r in rows if len(r) == ncol])
        t_xc, th_raw = arr[:, 0], arr[:, -1]
        if abs(t_xc[0]) > 5:
            why = f'запись начинается с t = {t_xc[0]:.0f} с, а не с 0'
        elif t_xc[-1] < 0.95 * t_end:
            why = (f'запись кончается на t = {t_xc[-1]:.0f} с, а моделирование шло '
                   f'до {t_end:.0f} с – в первом столбце не время моделирования?')
        elif np.any(np.diff(t_xc) <= 0):
            why = 'время не возрастает'
    if why is None:
        th_i = np.interp(t_grid, t_xc, th_raw)
        i63 = int(np.searchsorted(t_grid, 3500.0))
        if abs(th_i[i63] - 0.632 * KdP) > 0.02:
            why = (f'theta2(3500 c) = {th_i[i63]:.4f} К, а ожидалось '
                   f'{0.632*KdP:.4f} К (63,2 %) — записан другой сигнал?')
        elif not 0.9 * KdP < th_raw.max() < 1.05 * KdP:
            why = f'max(theta2) = {th_raw.max():.4f} К, а ожидалось ~{KdP:.2f} К'
    if why is not None:
        head = '\n'.join('    ' + s for s in
                open(fname,encoding='utf-8-sig', errors='replace').read().splitlines()[:3])
        return None
    print(f'* экспорт Xcos подхвачен: {len(rows)} строк, {ncol} столбцов, '
       f't = {t_xc[0]:.0f}…{t_xc[-1]:.0f} с, theta2 до {th_raw.max():.4f} К')
    return th_i
def rhs(t, x):
    """Правая часть системы (2.7.2); входы u = [dP; theta_a] = [0,01; 0]."""
    return A @ x + B @ np.array([dP, 0.0])
# 1) Аналитическое решение (2.5.1)
th_an = KdP * (1 - (Ta * np.exp(-t / Ta) - Tb * np.exp(-t / Tb)) / (Ta - Tb))
# 2) Расчетная ветвь: метод РК45, относительная и абсолютная точность 10^-10
sol_py = solve_ivp(rhs, [0, t_end], [0.0, 0.0], method='RK45', t_eval=t,
                   rtol=1e-10, atol=1e-13)
th_py = sol_py.y[1]

# 3) Имитационная ветвь: экспорт блочной модели Xcos (LSodar, точность 10^-8) либо, при его отсутствии или неподходящем формате, решатель LSODA
th_xc = load_xcos_csv('/content/xcos_export_2.csv', t)
if th_xc is None:
    sol_xc = solve_ivp(rhs, [0, t_end], [0.0, 0.0], method='LSODA', t_eval=t, rtol=1e-8, atol=1e-10)
    th_xc = sol_xc.y[1]
# Контрольные значения: theta2 в точках таблиц 2.7 и 2.8, отклонения delta_max
for tk in (600, 1750, 3500, 7000, 12540):
    i = int(tk)
    print(f't = {tk:6d} c:  theta2 = {th_an[i]:.4f} (аналит.)   '
          f'{th_py[i]:.4f} (РК45)   {th_xc[i]:.4f} (Xcos)')
d_py = np.abs(th_an - th_py)
d_xc = np.abs(th_an - th_xc)
print(f'дельта_max (РК45, 10^-10): {d_py.max():.1e} К')
print(f'дельта_max (Xcos, 10^-8):  {d_xc.max():.1e} К')
BLUE, ORANGE, GREY = '#1f77b4', '#d95f02', '#8c8c8c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 6.2), dpi=200,
                               constrained_layout=True)
# --- а) переходные характеристики -------------------
ax1.plot(t, th_an, color=BLUE, lw=1.8, label='аналитическое решение (2.5.1)')
ax1.plot(t[::250], th_py[::250], 'o', ms=3.2, color='green',
         label='численное интегрирование (Python, РК45)')
ax1.plot(t, th_xc, color=ORANGE, lw=1.0, ls='--',
         label='блочная модель (Xcos, LSodar)')
ax1.axhline(KdP, color=GREY, lw=0.8, ls=':') # статика 0,2 К
ax1.axhline(0.632 * KdP, color=GREY, lw=0.8, ls=':') # уровень 63,2 %
ax1.axvline(3500, color=GREY, lw=0.8, ls=':') # t = T + tau = 3500 с
ax1.annotate('63,2 %: t = 3500 с', xy=(3500, 0.632 * KdP),
             xytext=(4300, 0.055), fontsize=8)
ax1.text(0.985, 0.1, r'$\theta_2^\infty = K\Delta P$ = 0,2 К',
         transform=ax1.transAxes, fontsize=9, ha='right')
ax1.set_ylabel(r'$\theta_2$, К', fontsize=9)
ax1.set_xlabel('t, с', fontsize=9)
ax1.set_xlim(0, t_end)
ax1.set_ylim(0, 0.225)
ax1.legend(fontsize=8, loc='lower right')
ax1.text(0.02, 0.92, 'а)', transform=ax1.transAxes, fontsize=10)
# --- б) отклонения численных решений от (2.5.1), нК ----
ax2.semilogy(t[1:], np.maximum(d_py[1:] * 1e9, 1e-3), color='green', lw=1.2,
             label='Python (РК45), точность 10$^{-10}$')
ax2.semilogy(t[1:], np.maximum(d_xc[1:] * 1e9, 1e-3), color=ORANGE, lw=1.2,
             label='Xcos (LSodar), точность 10$^{-8}$')
ax2.set_ylabel(r'$|\theta_2^{\mathrm{анал}} - \theta_2|$, нК', fontsize=9)
ax2.set_xlabel('t, с', fontsize=9)
ax2.set_xlim(0, t_end)
ax2.set_ylim(5e-4, 3e1)
ax2.legend(fontsize=8, loc='upper right')
ax2.text(0.02, 0.92, 'б)', transform=ax2.transAxes, fontsize=10)
ax2.grid(True, which='both', lw=0.3, alpha=0.4)
fig.savefig('risunok_2_10_new.png', dpi=200)
print('рисунок сохранен: risunok_2_10_new.png')
