# Apéndice Teórico-Práctico: El Método de Runge-Kutta y su Implementación con SciPy

---

## 1. Forma Algebraica del Método Clásico de Runge-Kutta (RK4)

Consideremos el problema de valor inicial para un sistema de ecuaciones diferenciales ordinarias:

$$\frac{dy}{dt} = f(t, y), \quad y(t_0) = y_0$$

donde $y(t) \in \mathbb{R}^d$ es el vector de estado y $f: \mathbb{R} \times \mathbb{R}^d \to \mathbb{R}^d$ describe el campo de velocidades.

Para avanzar el estado desde el tiempo discreto $t_n$ hasta $t_{n+1} = t_n + \Delta t$, el método clásico de Runge-Kutta de cuarto orden (RK4) evalúa la derivada en cuatro etapas intermedias:

$$k_1 = f(t_n, y_n)$$

$$k_2 = f\left(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2} k_1\right)$$

$$k_3 = f\left(t_n + \frac{\Delta t}{2}, y_n + \frac{\Delta t}{2} k_2\right)$$

$$k_4 = f(t_n + \Delta t, y_n + \Delta t k_3)$$

La actualización del estado se obtiene como un promedio ponderado de estas cuatro pendientes:

$$y_{n+1} = y_n + \frac{\Delta t}{6} \left( k_1 + 2 k_2 + 2 k_3 + k_4 \right)$$

---

## 2. Tasa de Convergencia y Adaptabilidad

- **Orden de precisión**: RK4 es un método de orden 4. El error de discretización global escala proporcionalmente a $(\Delta t)^4$. Al reducir el paso temporal a la mitad ($\Delta t \to \Delta t / 2$), el error global disminuye aproximadamente por un factor de $2^4 = 16$.
- **Control adaptativo de paso (RK45)**: En las librerías científicas modernas (como SciPy), el integrador estándar por defecto es `RK45` (fórmula de Dormand-Prince). Este algoritmo calcula simultáneamente aproximaciones de orden 4 y orden 5 utilizando evaluaciones compartidas. La diferencia entre ambas soluciones proporciona una estimación continua del error local, lo que permite al integrador aumentar o reducir automáticamente el tamaño del paso $\Delta t$ para satisfacer tolerancias relativas (`rtol`) y absolutas (`atol`) prefijadas.

---

## 3. Implementación con `scipy.integrate.solve_ivp`

La función `solve_ivp` requiere los siguientes argumentos básicos:
- `fun(t, y)`: Función que calcula el vector de derivadas. Debe recibir el tiempo escalar `t` y el arreglo unidimensional `y`, retornando una lista o arreglo con las derivadas en el mismo orden.
- `t_span`: Tupla `(t_inicial, t_final)` con el intervalo de integración.
- `y0`: Lista o arreglo con las condiciones iniciales en $t = t_{\text{inicial}}$.
- `t_eval`: Arreglo de instantes temporales donde se desea almacenar la solución.
- `method`: Algoritmo de integración (por defecto `"RK45"`).

### Ejemplo 1: Modelo Logístico 1D (del Cuaderno 1)

$$\frac{dx}{dt} = C x (1 - x), \quad x(0) = x_0$$

```python
import numpy as np
from scipy.integrate import solve_ivp

# 1. Definición de la ecuación diferencial
def ode_logistica(t, y, C=2.0):
    x = y[0]
    dxdt = C * x * (1.0 - x)
    return [dxdt]

# 2. Parámetros de simulación
t_span = (0.0, 8.0)
y0 = [0.05]
t_eval = np.linspace(0.0, 8.0, 100)

# 3. Integración con RK45
sol = solve_ivp(ode_logistica, t_span, y0, t_eval=t_eval, method="RK45")

# 4. Extracción de la trayectoria
t = sol.t
x = sol.y[0]
```

---

### Ejemplo 2: Sistema Abstracto de Dos Variables (2D)

Consideremos un sistema lineal acoplado:

$$\frac{dy_1}{dt} = -a y_1 + b y_2$$

$$\frac{dy_2}{dt} = c y_1 - d y_2$$

con condiciones iniciales $y_1(0) = 1.0$, $y_2(0) = 0.0$ y parámetros $a = 1.0, b = 0.5, c = 0.2, d = 0.8$:

```python
import numpy as np
from scipy.integrate import solve_ivp

# 1. Definición del sistema vectorial
def sistema_2d(t, y, a=1.0, b=0.5, c=0.2, d=0.8):
    y1, y2 = y
    dy1 = -a * y1 + b * y2
    dy2 =  c * y1 - d * y2
    return [dy1, dy2]

# 2. Parámetros de simulación
t_span = (0.0, 10.0)
y0 = [1.0, 0.0]
t_eval = np.linspace(0.0, 10.0, 200)

# 3. Integración con RK45
sol = solve_ivp(sistema_2d, t_span, y0, t_eval=t_eval, method="RK45")

# 4. Extracción de componentes
t = sol.t
y1 = sol.y[0]
y2 = sol.y[1]
```

### Extensión al Modelo SIR (3D)

La estructura para el modelo SIR es análoga: el vector de estado contiene tres variables `y = [S, I, R]` y la función retorna `[ds, di, dr]`. Los componentes de la solución se extraen directamente como `sol.y[0]` para $S(t)$, `sol.y[1]` para $I(t)$ y `sol.y[2]` para $R(t)$.
