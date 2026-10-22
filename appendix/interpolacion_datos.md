# Apéndice Técnico: Interpolación de Datos en Python

---

## 1. El Problema de Interpolación Unidimensional

En el análisis de sistemas dinámicos y ciencia de datos, con frecuencia se dispone de una serie de observaciones o simulaciones evaluadas en una malla discreta de puntos:

$$\{(x_0, y_0), (x_1, y_1), \dots, (x_N, y_N)\}$$

El objetivo de la **interpolación unidimensional** es construir una función continua $\hat{y}(x)$ que pase exactamente por los puntos conocidos ($\hat{y}(x_i) = y_i$) para estimar el valor del sistema en puntos no muestreados $x^* \in [x_0, x_N]$.

En modelamiento diferencial, la interpolación es indispensable para:
1. Comparar simulaciones numéricas calculadas con pasos temporales distintos ($\Delta t_1 \neq \Delta t_2$).
2. Alinear temporalmente registros experimentales discretos (por ejemplo, reportes clínicos diarios) con curvas continuas de alta resolución.
3. Evaluar funciones de pérdida en problemas inversos cuando los tiempos de simulación no coinciden exactamente con los instantes de observación.

---

## 2. Interpolación Lineal Rápida con NumPy (`np.interp`)

NumPy ofrece la función nativa `np.interp`, implementada en C para máxima velocidad de ejecución.

### Sintaxis Básica

```python
y_nuevo = np.interp(x_nuevo, x_conocido, y_conocido)
```

- `x_nuevo`: Punto o arreglo de puntos donde se desea evaluar la interpolación.
- `x_conocido`: Arreglo de coordenadas independientes de los datos originales (debe estar ordenado de forma estrictamente creciente).
- `y_conocido`: Arreglo de valores dependientes correspondientes a `x_conocido`.

### Ejemplo Práctico

Supongamos que tenemos una trayectoria de referencia continua calculada con un paso fino $\Delta t = 0.01$, y deseamos muestrearla en días enteros $t = 0, 1, 2, \dots, 10$:

```python
import numpy as np

# Datos continuos de alta resolución (malla fina)
t_fino = np.linspace(0.0, 10.0, 1001)
y_fino = np.sin(t_fino)

# Instantes discretos donde necesitamos los valores
t_dias = np.arange(0.0, 11.0, 1.0)

# Interpolación lineal directa
y_dias = np.interp(t_dias, t_fino, y_fino)

print("Tiempos diarios:   ", t_dias[:5])
print("Valores calculados:", np.round(y_dias[:5], 4))
```

---

## 3. Interpolación de Mayor Suavidad con SciPy (`scipy.interpolate`)

Cuando la función interpolada debe tener derivadas continuas (por ejemplo, para calcular velocidades o aceleraciones sin quiebres artificiales), la interpolación lineal a trozos puede no ser suficiente. SciPy proporciona herramientas de orden superior:

### Splines Cúbicos (`CubicSpline`)

El método `CubicSpline` construye polinomios cúbicos independientes entre cada par de puntos adyacentes, asegurando que tanto la función como su primera y segunda derivada sean estrictamente continuas ($C^2$):

```python
from scipy.interpolate import CubicSpline

# Ajustar el interpolador cúbico
cs = CubicSpline(t_fino, y_fino)

# Evaluar en los nuevos puntos o en una malla arbitraria
y_spline = cs(t_dias)

# Además, CubicSpline permite evaluar directamente la derivada analítica:
dy_dt = cs(t_dias, nu=1)
```

---

## 4. Comparación: `t_eval` en `solve_ivp` frente a `np.interp`

Al trabajar con resolvedores de ecuaciones diferenciales como `scipy.integrate.solve_ivp`:

1. **Vía directa con `t_eval`**: Al invocar `solve_ivp(..., t_eval=t_deseado)`, el integrador utiliza internamente un polinomio interpolador continuo de alto orden (solución densa) para entregar los valores exactamente en los instantes `t_deseado`. Es la opción más limpia y eficiente cuando se genera la simulación desde cero.
2. **Vía `np.interp`**: Si los datos ya fueron generados previamente, provienen de un archivo externo o se encuentran en una malla densa precalculada, `np.interp` permite submuestrear o remuestrear la serie en una sola línea de código sin necesidad de reinvocar el resolvedor numérico.
