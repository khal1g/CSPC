import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, newton

K = 1.0


def k_imbalance(x):
  x = np.clip(x, 0.0001, 0.9999)
  return ((2 * x) ** 2) / ((1 - x) * (1 - x)) - K


x_newton = newton(k_imbalance, x0=0.5)

res_slsqp = minimize(
    lambda x: k_imbalance(x[0]) ** 2, x0=[0.5], method="SLSQP", bounds=[(0, 1)]
)
x_slsqp = res_slsqp.x[0]

print(f"Equilibrium x (Newton) = {x_newton:.4f}")
print(f"Equilibrium x (SLSQP) = {x_slsqp:.4f}")
print(f"Equilibrium amount of HI = {2 * x_newton:.2f} mol")

x_vals = np.linspace(0, 0.99, 200)
h2_vals = 1 - x_vals
i2_vals = 1 - x_vals
hi_vals = 2 * x_vals

plt.figure(figsize=(8, 5))
plt.plot(x_vals, h2_vals, label="H2", color="blue")
plt.plot(x_vals, i2_vals, label="I2", color="green")
plt.plot(x_vals, hi_vals, label="HI", color="red")
plt.axvline(
    x_newton,
    color="black",
    linestyle="--",
    label=f"Equilibrium point x={x_newton:.2f}",
)
plt.xlabel("Extent of reaction (x)")
plt.ylabel("Amount of substance (mol)")
plt.title("Chemical Equilibrium")
plt.legend()
plt.grid(True)
plt.savefig("equilibrium.png")
plt.show()