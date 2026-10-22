# Apéndice Técnico: Resolución de Ecuaciones No Lineales con SciPy

---

## 1. El Problema de Raíz en Esquemas Temporales Implícitos

Al discretizar un sistema diferencial ordinario $\frac{dy}{dt} = f(t, y)$ mediante el método de Euler implícito, la regla de actualización es:

$$y_{n+1} = y_n + \Delta t f(t_{n+1}, y_{n+1})$$

Dado que la incógnita $y_{n+1}$ aparece en ambos lados de la igualdad y está dentro del campo no lineal $f$, definir el estado futuro equivale a resolver un **problema de búsqueda de raíces** para la función residual vectorial:

$$F(y^*) = y^* - y_n - \Delta t f(t_{n+1}, y^*) = 0$$

donde $y^* = y_{n+1}$ es el vector solución buscado.

En problemas simples con no linealidades cuadráticas (como el modelo logístico 1D o el modelo SIR desacoplado), es posible despejar analíticamente una raíz cerrada. Sin embargo, en modelos no lineales generales de dos o más dimensiones, este despeje analítico no es factible y es indispensable emplear algoritmos numéricos iterativos (como el método de Newton-Raphson o variantes de Powell).

---

## 2. La Herramienta `scipy.optimize.root`

La librería científica SciPy proporciona la función `scipy.optimize.root` para resolver sistemas algebraicos no lineales de la forma $F(x) = 0$.

Los argumentos esenciales de `root` son:
- `fun(x, *args)`: Función que toma el vector candidato $x$ y retorna el vector residual $F(x)$. La solución exacta cumple $F(x) = 0$.
- `x0`: Estimación inicial (*initial guess*).
- `args`: Tupla de parámetros adicionales que se pasan directamente a `fun`.

> **Principio de Continuidad Temporal (*Warm Start*)**: Al integrar una ecuación diferencial en el tiempo, el estado futuro $y_{n+1}$ suele estar muy cerca del estado actual $y_n$ si el paso $\Delta t$ es moderado. Por tanto, elegir $x_0 = y_n$ como estimación inicial garantiza una convergencia rápida y robusta del resolvedor no lineal.

---

## 3. Ejemplo Básico: Ecuación Escalar No Lineal

Resolvamos la ecuación $x^3 + x - 5 = 0$:

```python
from scipy.optimize import root

def residual_1d(x):
    return x**3 + x - 5.0

# Estimación inicial x0 = 1.0
sol = root(residual_1d, x0=1.0)

print(f"Raíz encontrada: {sol.x[0]:.6f}")
print(f"Residual F(x):   {sol.fun[0]:.2e}")
print(f"¿Convergió?:     {sol.success}")
```

---

## 4. Arquetipo de Implementación para Euler Implícito

A continuación se presenta un arquetipo genérico y reutilizable para integrar un paso temporal con Euler implícito resolviendo el residuo con `scipy.optimize.root`:

```python
import numpy as np
from scipy.optimize import root

def paso_euler_implicito(f_ode, y_n, dt):
    """
    Avanza un paso temporal mediante Euler implícito:
    y_{n+1} - y_n - dt * f(y_{n+1}) = 0
    """
    def residual(y_cand):
        # Diferencia entre el candidato y el paso implícito
        return y_cand - y_n - dt * f_ode(y_cand)

    # Warm start: usamos y_n como estimación inicial
    res = root(residual, x0=y_n, method="hybr")

    if not res.success:
        raise RuntimeError(f"Fallo de convergencia en el paso implícito: {res.message}")

    return res.x
```

### Ejemplo de uso en un bucle temporal (Modelo Logístico):

```python
# Definición de la física f(x) = C * x * (1 - x)
def f_logistica(y):
    x = y[0]
    return np.array([2.0 * x * (1.0 - x)])

# Simulación temporal paso a paso
dt = 0.1
n_steps = 50
x_trayectoria = [0.05]

for n in range(n_steps):
    x_actual = np.array([x_trayectoria[-1]])
    x_siguiente = paso_euler_implicito(f_logistica, x_actual, dt)
    x_trayectoria.append(float(x_siguiente[0]))
```

Este arquetipo es directamente aplicable a sistemas de más variables (como el modelo SIR), simplemente haciendo que `f_ode` acepte un vector `y = [S, I, R]` y retorne el vector de derivadas `[dS, dI, dR]`.
