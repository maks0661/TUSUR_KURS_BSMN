import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

# ----- 1. Параметры полной модели -----
C1, C2 = 150.0, 6.0          # теплоемкости узлов, Дж/К
R_c, R_th = 20.0, 60.0       # тепловые сопротивления, К/Вт
dP = 0.01                    # ступенька мощности нагревателя, Вт (1 % от Q1_inf)

# Характеристическое уравнение полной модели: a2*s^2 + a1*s + 1 = 0 (п. 2.3)
a1 = R_c * (C1 + C2) + R_th * C2       # с
a2 = R_th * R_c * C1 * C2              # с^2
sD = math.sqrt(a1 * a1 - 4.0 * a2)     # корень дискриминанта, с
T_a = 0.5 * (a1 + sD)                  # большая постоянная времени (tau1 из п. 2.3), с
T_b = 0.5 * (a1 - sD)                  # меньшая постоянная времени (tau2 из п. 2.3), с
K = R_c                                # коэффициент передачи канала (статика), К/Вт
theta_inf = K * dP                     # установившееся приращение theta2, К

print('1. Параметры полной модели (таблица 2.3, средние значения)')
print(f'   a1 = {a1:.0f} с, a2 = {a2:.2e} с2;  '
      f'T_a = {T_a:.1f} с, T_b = {T_b:.1f} с')
print(f'   K = R_c = {K:.0f} К/Вт;  ступенька dP = {dP} Вт -> '
      f'K*dP = {theta_inf:.3f} К')

# ----- 2. Переходная характеристика (2.4.5) -----
# Система (2.4.5) в отклонениях от исходного установившегося режима:
#     C1*th1' = dP - (th1 - th2)/R_th - th1/R_c
#     C2*th2' = (th1 - th2)/R_th
def model(_t, y):
    th1, th2 = y
    return [(dP - (th1 - th2) / R_th - th1 / R_c) / C1,
            (th1 - th2) / R_th / C2]

T_END = 4.0 * T_a                           # длительность эксперимента, с
t_grid = np.arange(0.0, T_END + 0.5, 1.0)   # сетка регистрации с шагом 1 с
sol = solve_ivp(model, [0.0, T_END], [0.0, 0.0], t_eval=t_grid,
                rtol=1e-10, atol=1e-14, max_step=5.0)
theta2 = sol.y[1]                           # кривая разгона theta2(t), К

def theta2_full(t):
    """Аналитическое решение (2.5.1), К."""
    t = np.asarray(t, dtype=float)
    return K * dP * (1.0 - (T_a * np.exp(-t / T_a)
                            - T_b * np.exp(-t / T_b)) / (T_a - T_b))

def fopdt(t, T, tau, Kx=K):
    """Переходная характеристика FOPDT-модели (2.5.6), К."""
    t = np.asarray(t, dtype=float)
    return Kx * dP * np.where(t < tau, 0.0, 1.0 - np.exp(-(t - tau) / T))

def dev_max(T, tau):
    """Максимум отклонения FOPDT от кривой разгона: % от K*dP и время, с."""
    d = np.abs(theta2 - fopdt(t_grid, T, tau))
    return d.max() / theta_inf * 100.0, t_grid[d.argmax()]

err = np.abs(theta2 - theta2_full(t_grid)).max()
print('2. Переходная характеристика: численное интегрирование (2.4.5)')
print(f'   длительность 4*T_a = {T_END:.0f} с ({T_END / 3600:.1f} ч), шаг 1 с;')
print(f'   контроль по (2.5.1): max |численное - аналитическое| = {err:.1e} К '
      f'({err / theta_inf * 100:.1e} % от K*dP)')

# ----- 3. Метод касательной: (2.5.2) и (2.5.3) -----
t_pi = T_a * T_b * math.log(T_a / T_b) / (T_a - T_b)   # точка перегиба t_п, с
th_pi = float(theta2_full(t_pi))                       # theta2(t_п), К
th_pi_d = K * dP * (math.exp(-t_pi / T_a)
                    - math.exp(-t_pi / T_b)) / (T_a - T_b)   # наклон, К/с
tau_k = t_pi - th_pi / th_pi_d                  # пересечение касательной с осью t
T_k = t_pi + (theta_inf - th_pi) / th_pi_d - tau_k   # пересечение с асимптотой
d_k, _t = dev_max(T_k, tau_k)
print('3. Метод касательной в точке перегиба')
print(f'   t_п = {t_pi:.0f} с, theta2(t_п) = {th_pi:.4f} К '
      f'({th_pi / theta_inf * 100:.1f} % установившегося приращения)')
print(f'   tau_к = {tau_k:.0f} с, T_к = {T_k:.0f} с (сумма {tau_k + T_k:.0f} с); '
      f'delta_max = {d_k:.1f} %')

# ----- 4. Двухточечный метод (28,3/63,2 %) -----
h1 = 1.0 - math.exp(-1.0 / 3.0) # 0.2GitHub835 - уровень 28,3 %
h2 = 1.0 - math.exp(-1.0) # 0.6321 - уровень 63,2 %
t1 = t_grid[np.argmax(theta2 >= h1 * theta_inf)]
t2 = t_grid[np.argmax(theta2 >= h2 * theta_inf)]
T_2 = 1.5 * (t2 - t1) # T = (t2-t1)/ln[(1-h1)/(1-h2)]; здесь ln(...) = 2/3
tau_2 = t2 - T_2
d_2, _t = dev_max(T_2, tau_2)
print('4. Двухточечный метод (уровни 28,3 % и 63,2 %)')
print(f'   t1 = {t1:.0f} с, t2 = {t2:.0f} с  ->  T = {T_2:.0f} с, '
      f'tau = {tau_2:.0f} с; delta_max = {d_2:.1f} %')

# ----- 5. МНК (уточнение) -----
def residual(x):
    T, tau = x
    return fopdt(t_grid, T, tau) - theta2

res = least_squares(residual, [T_2, tau_2],
                    bounds=([0.5 * T_a, 0.0], [2.0 * T_a, 1.5 * T_b]),
                    xtol=1e-12, ftol=1e-12)
T_mnk, tau_ob = res.x
d_m, _t = dev_max(T_mnk, tau_ob)
print('5. МНК (K зафиксирован на уровне статики K = R_c)')
print(f'   T = {T_mnk:.0f} с, tau_об = {tau_ob:.0f} с ({res.nfev} вычислений '
      f'функции); delta_max = {d_m:.1f} %')

# Контроль: если K не фиксировать, МНК возвращает K = R_c
res_k = least_squares(lambda x: fopdt(t_grid, x[0], x[1], Kx=x[2]) - theta2,
                      [T_2, tau_2, K],
                      bounds=([0.5 * T_a, 0.0, 5.0], [2.0 * T_a, 1.5 * T_b, 50.0]))
print(f'   контроль (K свободен): K = {res_k.x[2]:.2f} К/Вт '
      f'({(res_k.x[2] - K) / K * 100:+.2f} % к R_c)')

# ----- 6. Рабочие значения и контрольные точки -----
tau_tr = 5.0 # запаздывание тракта (таблица 2.3), с
tau_w = tau_ob + tau_tr # рабочее запаздывание, с
d_w, t_w = dev_max(T_mnk, tau_w)
rms = math.sqrt(float(np.mean((theta2 - fopdt(t_grid, T_mnk, tau_w)) ** 2))) \
      / theta_inf * 100.0
print('6. Сводка (таблица 2.4) и контрольные точки')
print('   метод              T, с   tau, с   delta_max, %')
print(f'   касательная       {T_k:6.0f}  {tau_k:7.0f}      {d_k:4.1f}')
print(f'   две точки         {T_2:6.0f}  {tau_2:7.0f}      {d_2:4.1f}')
print(f'   МНК               {T_mnk:6.0f}  {tau_ob:7.0f}      {d_m:4.1f}')
print(f'   МНК + tau_тр      {T_mnk:6.0f}  {tau_w:7.0f}      {d_w:4.1f}   (СКО {rms:.1f} %)')
print(f'   контроль: tau + T = {tau_w + T_mnk:.0f} с против t(63,2 %) = {t2:.0f} с;')
print(f'   максимум отклонения {d_w:.1f} % достигается при t = {t_w:.0f} с (t ~ tau)')
print(f'   итог (2.5.6): W(s) = {K:.0f}*exp(-{tau_w:.0f}*s)/({T_mnk:.0f}*s + 1)')

# ----- 7. Графики (рисунок 2.6) -----
PLOT = True # построить рисунок 2.6; False - без графики
if PLOT:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   constrained_layout=True)
    tf = np.linspace(0.0, T_END, 2000)
    ax1.plot(tf, theta2_full(tf), label='полная модель (2.4.5)')
    ax1.plot(tf, fopdt(tf, T_mnk, tau_w), '--', label='FOPDT (2.5.6)')
    ax1.axhline(theta_inf, ls=':', lw=0.8, color='grey')
    ax1.set_xlabel('t, с'); ax1.set_ylabel(r'$\theta_2$, К')
    ax1.set_title('а) полный переходный процесс'); ax1.legend()

    m = t_grid <= 4200.0 # начальный участок (рис. 2.6, б)
    ax2.plot(t_grid[m], theta2[m], label='полная модель')
    tl = np.array([0.0, 4200.0])
    ax2.plot(tl, th_pi + th_pi_d * (tl - t_pi), '--', label='касательная')
    ax2.axhline(h1 * theta_inf, ls=':', lw=0.8, color='grey')
    ax2.axhline(h2 * theta_inf, ls=':', lw=0.8, color='grey')
    ax2.plot([t1, t2], [h1 * theta_inf, h2 * theta_inf], 'o',
             label='уровни 28,3 % / 63,2 %')
    ax2.set_xlabel('t, с'); ax2.set_ylabel(r'$\theta_2$, К')
    ax2.set_title('б) начальный участок'); ax2.legend()

    fig.savefig('risunok_2_6.png', dpi=300)