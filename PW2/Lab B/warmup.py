import numpy as np
from scipy.optimize import minimize, newton

def f(x):
    return (x - 3)**2 + 1

def f_prime(x):
    return 2 * (x - 3)

print("--- 2A: Простая функция ---")
x = 0.0
lr = 0.1
for _ in range(50):
    x = x - lr * f_prime(x)
print(f"Градиентный спуск: x = {x:.4f}")

x_newton = newton(f_prime, x0=0.0)
print(f"Метод Ньютона: x = {x_newton:.4f}")

res_slsqp = minimize(f, x0=[0.0], method="SLSQP")
print(f"SLSQP: x = {res_slsqp.x[0]:.4f}\n")


def g(x):
    return x**4 - 3*x**2 + x + 5

def g_prime(x):
    return 4*x**3 - 6*x + 1

def g_double_prime(x):
    return 12*x**2 - 6

print("--- 2B: Сложный ландшафт ---")
for x0 in [0.0, 2.0]:
    print(f"Стартовая точка x0 = {x0}:")
    
    x_gd = x0
    for _ in range(100):
        x_gd = x_gd - 0.05 * g_prime(x_gd)
    print(f"  Градиентный спуск: x = {x_gd:.4f}")
    
    try:
        x_n = newton(g_prime, x0=x0)
        g2 = g_double_prime(x_n)
        status = "минимум" if g2 > 0 else "максимум/другая точка"
        print(f"  Метод Ньютона: x = {x_n:.4f} (g'' = {g2:.2f}, это {status})")
    except Exception as e:
        print(f"  Метод Ньютона не сошелся: {e}")
        
    res_s = minimize(g, x0=[x0], method="SLSQP")
    print(f"  SLSQP: x = {res_s.x[0]:.4f}")