import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)
t = data[:, 0]
c_measured = data[:, 1]

C0 = c_measured[0]

def total_error(k):
    k_val = k[0]
    c_model = C0 * np.exp(-k_val * t)
    return np.sum((c_measured - c_model)**2)

result = minimize(total_error, x0=[0.5], method="SLSQP", bounds=[(0, 5)])
k_fitted = result.x[0]

print(f"Подобранная константа скорости k = {k_fitted:.4f}")[cite: 2]

t_dense = np.linspace(min(t), max(t), 200)
c_fit = C0 * np.exp(-k_fitted * t_dense)

plt.figure(figsize=(8, 5))
plt.scatter(t, c_measured, color="red", label="Измерения (данные)")
plt.plot(t_dense, c_fit, color="blue", label=f"Модель (k = {k_fitted:.4f})")
plt.xlabel("Время (t)")
plt.ylabel("Концентрация (C)")
plt.title("Кинетика реакции (Подгонка скорости)")
plt.legend()
plt.grid(True)
plt.savefig("kinetics.png")
plt.show()[cite: 2]