import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton, minimize

K = 1.0 

def k_imbalance(x):
    x = np.clip(x, 0.0001, 0.9999)
    return ((2 * x)**2) / ((1 - x) * (1 - x)) - K

x_newton = newton(k_imbalance, x0=0.5)

res_slsqp = minimize(lambda x: k_imbalance(x[0])**2, x0=[0.5], method="SLSQP", bounds=[(0, 1)])
x_slsqp = res_slsqp.x[0]

print(f"Равновесие x (Ньютон) = {x_newton:.4f}")[cite: 2]
print(f"Равновесие x (SLSQP) = {x_slsqp:.4f}")[cite: 2]
print(f"Равновесное количество HI = {2 * x_newton:.2f} моль")[cite: 2]

x_vals = np.linspace(0, 0.99, 200)
h2_vals = 1 - x_vals
i2_vals = 1 - x_vals
hi_vals = 2 * x_vals

plt.figure(figsize=(8, 5))
plt.plot(x_vals, h2_vals, label="H2", color="blue")
plt.plot(x_vals, i2_vals, label="I2", color="green")
plt.plot(x_vals, hi_vals, label="HI", color="red")
plt.axvline(x_newton, color="black", linestyle="--", label=f"Точка равновесия x={x_newton:.2f}")
plt.xlabel("Степень превращения (x)")
plt.ylabel("Количество вещества (моль)")
plt.title("Химическое равновесие")
plt.legend()
plt.grid(True)
plt.savefig("equilibrium.png")
plt.show()[cite: 2]