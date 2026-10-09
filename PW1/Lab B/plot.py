import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt("decay_observed.csv", delimiter=",", skiprows=1)
t = data[:, 0]
observed = data[:, 1]

LAMBDA = 0.3
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)

ax1.scatter(t, observed, color="blue", label="Наблюдения", alpha=0.7)
ax1.set_title("Экспериментальные данные")
ax1.set_xlabel("Время (t)")
ax1.set_ylabel("Число отсчетов (N)")
ax1.grid(True)
ax1.legend()

ax2.plot(t, analytical, color="red", label=r"$N_0 e^{-\lambda t}$", linewidth=2)
ax2.set_title("Аналитический закон")
ax2.set_xlabel("Время (t)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()

plt.savefig("figure.png", dpi=300)
print("Изображение figure.png успешно сгенерировано.")