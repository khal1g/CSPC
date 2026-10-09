import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

data = np.genfromtxt('freefall.csv', delimiter=',', skip_header=1)
t = data[:, 0]  # время (с)
y = data[:, 1]  # высота (м)

v = np.gradient(y, t) 
a = np.gradient(v, t)  

mean_a = np.mean(a)
std_a = np.std(a)

print(f"Среднее ускорение: {mean_a:.2f} м/с^2")
print(f"Стандартное отклонение ускорения: {std_a:.2f} м/с^2")

v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]

y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]

max_diff = np.max(np.abs(y - y_rec))
print(f"Максимальная разница между исходной и восстановленной координатой: {max_diff:.4f} м")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

ax1.plot(t, y, label='Измеренная высота (y)', color='blue')
ax1.set_ylabel('Высота y (м)')
ax1.set_title('Анализ движения свободно падающего тела')
ax1.grid(True)
ax1.legend()

ax2.plot(t, v, label='Скорость (v)', color='orange')
ax2.set_ylabel('Скорость v (м/с)')
ax2.grid(True)
ax2.legend()

ax3.plot(t, a, label='Вычисленное ускорение (a)', color='red', alpha=0.6)
ax3.axhline(-9.81, color='black', linestyle='--', label='Теоретическое g (-9.81 м/с²)')
ax3.set_xlabel('Время t (с)')
ax3.set_ylabel('Ускорение a (м/с²)')
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.savefig('motion.png')
plt.show()

try:
    data_2d = np.genfromtxt('trajectory.csv', delimiter=',', skip_header=1)
    t_2d = data_2d[:, 0]
    x_2d = data_2d[:, 1]
    y_2d = data_2d[:, 2]

    vx = np.gradient(x_2d, t_2d)
    vy = np.gradient(y_2d, t_2d)
    speed = np.sqrt(vx**2 + vy**2)

    fig_2d, (ax_traj, ax_speed) = plt.subplots(1, 2, figsize=(12, 5))

    ax_traj.plot(x_2d, y_2d, 'g.-')
    ax_traj.set_title('2D Траектория (x vs y)')
    ax_traj.set_xlabel('x (м)')
    ax_traj.set_ylabel('y (м)')
    ax_traj.grid(True)

    ax_speed.plot(t_2d, speed, 'm-')
    ax_speed.set_title('Полная скорость со временем')
    ax_speed.set_xlabel('Время t (с)')
    ax_speed.set_ylabel('Скорость (м/с)')
    ax_speed.grid(True)

    plt.tight_layout()
    plt.savefig('trajectory_analysis.png')
    plt.show()
except Exception as e:
    print(f"Бонусное задание не выполнено или файл trajectory.csv не найден: {e}")