import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
V = data[:, 0]
ph = data[:, 1]

slope = np.gradient(ph, V)

max_idx = np.argmax(slope)
equivalence_volume = V[max_idx]

print(f"Точка эквивалентности (объем): {equivalence_volume:.2f} мл")[cite: 3]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(V, ph, color="purple")
ax1.axvline(equivalence_volume, color="red", linestyle="--", label=f"Эквивалентность: {equivalence_volume:.1f} мл")
ax1.set_xlabel("Объем основания (мл)")
ax1.set_ylabel("pH")
ax1.setTitle("Кривая титрования (pH)")
ax1.legend()
ax1.grid(True)

ax2.plot(V, slope, color="orange")
ax2.axvline(equivalence_volume, color="red", linestyle="--")
ax2.set_xlabel("Объем основания (мл)")
ax2.set_ylabel("dpH / dV (Наклон)")
ax2.setTitle("Производная кривой pH")
ax2.grid(True)

plt.tight_layout()
plt.savefig("titration.png")
plt.show()[cite: 3]