import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# ----- Параметры тепловой модели (подраздел 2.3) -----
C1, C2 = 150.0, 6.0           # теплоемкости тела и среды-образца, Дж/К
Rc, Rth = 20.0, 60.0          # тепловые сопротивления оболочки и среды, К/Вт
a1 = Rc * (C1 + C2) + Rth * C2              # коэффициент a1 = 3480 с
a2 = Rth * Rc * C1 * C2                     # коэффициент a2 = 1.08e6 с^2
tau1 = (a1 + np.sqrt(a1 * a1 - 4 * a2)) / 2  # постоянная T1 = 3135,6 с
tau2 = (a1 - np.sqrt(a1 * a1 - 4 * a2)) / 2  # постоянная T2 = 344,4 с
Tf, tauf = 3186.0, 314.0 # параметры FOPDT-аппроксимации (подраздел 2.6.2)

def step_full(t, K=1.0):
    """Точная реакция полной модели на ступеньку theta_a = K*1(t): (2.6.9)."""
    t = np.asarray(t, float)
    return K * (1 - (tau1 * np.exp(-t / tau1) - tau2 * np.exp(-t / tau2))
                / (tau1 - tau2))

def step_fopdt(t, K=1.0):
    """Реакция FOPDT-аппроксимации: запаздывание tauf + апериодическое звено."""
    t = np.asarray(t, float)
    return np.where(t >= tauf, K * (1 - np.exp(-(t - tauf) / Tf)), 0.0)

def sim(theta_a_func, t_end, n=6000, x0=(0.0, 0.0)):
    """Численное решение уравнения (2.6.7) при mu = 0 методом РК45.
    Уравнение a2*th'' + a1*th' + th = theta_a(t) сведено к системе двух
    уравнений первого порядка; x = [th, dth] = [theta_2, dtheta_2/dt].
    """
    def rhs(t, x):
        th, dth = x
        return [dth, (theta_a_func(t) - a1 * dth - th) / a2]

    sol = solve_ivp(rhs, [0, t_end], list(x0), t_eval=np.linspace(0, t_end, n),
                    rtol=1e-9, atol=1e-11)
    return sol.t, sol.y[0]

BLUE, ORANGE, GREY = '#1f77b4', '#d95f02', '#8c8c8c'
fig, axs = plt.subplots(3, 1, figsize=(6.3, 7.6), dpi=200, constrained_layout=True)

# ----- а) ступенька +10 К -----
ax = axs[0]
t = np.linspace(0, 14000, 5000)
ax.plot(t, step_full(t, 10), color=BLUE, lw=1.6,
        label='полная модель (2.6.9)')
ax.plot(t, step_fopdt(t, 10), color=ORANGE, lw=1.3, ls='--',
        label='FOPDT-аппроксимация')
ax.axhline(10, color=GREY, lw=0.8, ls=':')

t63 = np.interp(6.32, step_full(t, 10), t)   # момент достижения 63,2 %
ax.plot([t63], [6.32], 'o', ms=4, color='green')
ax.axhline(6.32, color=GREY, lw=0.6, ls=':')
ax.axvline(t63, color=GREY, lw=0.6, ls=':')
ax.annotate('63,2 %', xy=(t63, 6.32), xytext=(t63 + 700, 5.6), fontsize=8)
ax.text(0.975, 0.12, r'$\theta_2^{\infty} = 10$ К',
        transform=ax.transAxes, fontsize=9, ha='right')
ax.set_ylabel(r'$\theta_2$, К', fontsize=9)
ax.set_xlabel('t, с', fontsize=9)
ax.set_xlim(0, 14000)
ax.legend(fontsize=8, loc='lower right')
ax.text(0.02, 0.9, 'а)', transform=ax.transAxes, fontsize=10)

# ----- б) импульс 10 К длительностью 600 с -----
ax = axs[1]
t_end = 8000
tp, th2 = sim(lambda t: 10.0 if t <= 600 else 0.0, t_end, n=8001)
ax.plot(tp, np.where(tp <= 600, 10.0, 0.0), color=GREY, lw=1.2, ls='-.',
        label=r'$\theta_a$ (возмущение)')
ax.plot(tp, th2, color=BLUE, lw=1.6, label=r'$\theta_2$ (реакция)')
ipk = th2.argmax()
ax.plot([tp[ipk]], [th2[ipk]], 'o', ms=4, color='green')
ax.annotate(f'пик {th2[ipk]:.2f} К', xy=(tp[ipk], th2[ipk]),
            xytext=(tp[ipk] + 900, th2[ipk] + 0.3), fontsize=8)
ax.set_ylabel(r'$\theta$, К', fontsize=9)
ax.set_xlabel('t, с', fontsize=9)
ax.set_xlim(0, t_end)
ax.set_ylim(-0.4, 11)
ax.legend(fontsize=8, loc='center right')
ax.text(0.02, 0.9, 'б)', transform=ax.transAxes, fontsize=10)

# ----- в) суточная гармоника с амплитудой 5 К -----
ax = axs[2]
w = 2 * np.pi / 86400.0 # частота суточной гармоники, рад/с
t_end = 86400 * 3
tg, th2g = sim(lambda t: 5.0 * np.sin(w * t), t_end, n=12000)

# теоретические частотные характеристики (2.6.10)
A = 1.0 / np.sqrt((1 + (w * tau1) ** 2) * (1 + (w * tau2) ** 2))
phi = -(np.degrees(np.arctan(w * tau1)) + np.degrees(np.arctan(w * tau2)))
lag = -phi / np.degrees(w) # фазовое запаздывание, с

ax.plot(tg / 3600, 5 * np.sin(w * tg), color=GREY, lw=1.1, ls='-.',
        label=r'$\theta_a$ (возмущение)')
ax.plot(tg / 3600, th2g, color=BLUE, lw=1.6, label=r'$\theta_2$ (реакция)')
ax.text(0.985, 0.06, f'A = {A:.3f}, запаздывание ≈ {lag/60:.0f} мин',
        transform=ax.transAxes, fontsize=8.5, ha='right')
ax.set_ylabel(r'$\theta$, К', fontsize=9)
ax.set_xlabel('t, ч', fontsize=9)
ax.set_xlim(0, 72)
ax.set_ylim(-6.2, 6.2)
ax.legend(fontsize=8, loc='upper right')
ax.text(0.02, 0.9, 'в)', transform=ax.transAxes, fontsize=10)
fig.savefig('risunok_2_8.png', dpi=200)
