# Apéndice Técnico: Resolución de Ecuaciones No Lineales con SciPy

---

## 1. El Problema de Raíz en Esquemas Temporales Implícitos

Al discretizar una ecuación diferencial ordinaria $\frac{dy}{dt} = f(t, y)$ mediante el método de Euler implícito, la regla de avance temporal entre el instante $t_n$ y el instante $t_{n+1} = t_n + \Delta t$ es:

$$\frac{y_{n+1} - y_n}{\Delta t} = f(t_{n+1}, y_{n+1})$$

Multiplicando por el paso temporal $\Delta t$ y despejando todos los términos hacia un solo lado de la igualdad, se obtiene la **ecuación algebraica no lineal que debe satisfacerse en cada paso temporal**:

$$R(y_{n+1}) = y_{n+1} - y_n - \Delta t f(t_{n+1}, y_{n+1}) = 0$$

donde $R(y_{n+1})$ es la **función residual**. Encontrar el estado futuro $y_{n+1}$ equivale exactamente a encontrar la raíz de $R$.

### Aclaración Conceptual Fundamental: Avance Temporal vs Puntos de Equilibrio
Es muy importante no confundir dos conceptos matemáticos distintos:
1. **Puntos de equilibrio de la EDO:** Se obtienen al igualar el lado derecho de la ecuación diferencial continua a cero, $f(y) = 0$. En estos puntos, la derivada temporal se anula ($\frac{dy}{dt} = 0$) y el sistema permanece en reposo indefinidamente (por ejemplo, $x = 0$ o $x = 1$ en el modelo logístico).
2. **Ecuación de avance de Euler implícito:** **No** iguala el lado derecho $f(y)$ a cero. Lo que iguala a cero es el **residuo temporal** $R(y_{n+1}) = y_{n+1} - y_n - \Delta t f(t_{n+1}, y_{n+1}) = 0$. Esta ecuación define la evolución dinámica paso a paso a lo largo del tiempo, aun cuando el sistema esté muy lejos del equilibrio.

---

## 2. Deducción Algebraica Explícita: Caso 1D y Sistema SIR

### Caso 1: Modelo Logístico 1D (Notebook 1)
La ecuación diferencial logística es:

$$\frac{dx}{dt} = C x (1 - x)$$

Aplicando la discretización de Euler implícito:

$$\frac{x_{n+1} - x_n}{\Delta t} = C x_{n+1} (1 - x_{n+1})$$

Efectuamos el cálculo algebraico paso a paso:
1. Multiplicamos ambos miembros por $\Delta t$:
   $$x_{n+1} - x_n = \Delta t C x_{n+1} - \Delta t C x_{n+1}^2$$
2. Trasladamos todos los términos al miembro izquierdo e igualamos a cero:
   $$(\Delta t C) x_{n+1}^2 + (1 - \Delta t C) x_{n+1} - x_n = 0$$

Esta es exactamente la ecuación cuadrática $a x_{n+1}^2 + b x_{n+1} + c = 0$ utilizada en el **Laboratorio 1**, con:
$$a = \Delta t C, \quad b = 1 - \Delta t C, \quad c = -x_n$$

Resolver esta ecuación cuadrática analíticamente o resolver $R(x_{n+1}) = 0$ con un resolvedor numérico no lineal son dos caminos matemáticamente equivalentes para determinar $x_{n+1}$.

### Caso 2: Sistema Epidemiológico SIR (Notebook 2)
El sistema SIR discretizado con Euler implícito es:

$$S_{n+1} = S_n - \Delta t \beta S_{n+1} I_{n+1} \implies S_{n+1} = \frac{S_n}{1 + \Delta t \beta I_{n+1}}$$
$$I_{n+1} = I_n + \Delta t (\beta S_{n+1} I_{n+1} - \gamma I_{n+1})$$
$$R_{n+1} = R_n + \Delta t \gamma I_{n+1}$$

Sustituyendo la expresión de $S_{n+1}$ en la ecuación de $I_{n+1}$ e igualando a cero:

$$h(I_{n+1}) = I_{n+1} - I_n - \Delta t \left( \beta \frac{S_n}{1 + \Delta t \beta I_{n+1}} I_{n+1} - \gamma I_{n+1} \right) = 0$$

Al multiplicar por $(1 + \Delta t \beta I_{n+1})$ se obtiene la ecuación cuadrática $A I_{n+1}^2 + B I_{n+1} + C = 0$ del **Laboratorio 2**. Sin embargo, con un resolvedor no lineal en Python podemos encontrar $I_{n+1}$ directamente a partir de $h(I_{n+1}) = 0$ sin necesidad de multiplicar ni despejar polinomios a mano.

---

## 3. La Herramienta `scipy.optimize.root`

La librería científica SciPy proporciona la función `scipy.optimize.root` para calcular raíces de funciones vectoriales:

```python
from scipy.optimize import root

# Sintaxis fundamental:
sol = root(fun, x0, method="hybr")
```

Argumentos principales:
- `fun(x)`: Función que recibe el vector candidato $x$ y retorna el vector residual $R(x)$.
- `x0`: Estimación inicial (*initial guess*).
- `method`: Algoritmo numérico (por defecto `"hybr"`, que combina el método de Powell con Newton-Raphson).

> **Estrategia de Warm Start:** En integración temporal, la solución en el paso futuro $y_{n+1}$ se encuentra muy cerca del estado actual $y_n$ si el paso $\Delta t$ es moderado. Por tanto, utilizar $x_0 = y_n$ como estimación inicial garantiza convergencia en apenas 2 a 4 iteraciones.

---

## 4. Ejemplo Práctico: Comparación entre Fórmula Analítica y `root`

El siguiente código implementa el paso de Euler implícito para la ecuación logística de dos maneras:
1. Mediante la raíz analítica de la fórmula cuadrática.
2. Mediante `scipy.optimize.root` resolviendo la ecuación residual implícita.

```python
import numpy as np
from scipy.optimize import root

# Parámetros físicos y temporales
C = 2.0
dt = 0.1
x_n = 0.40  # Estado actual en el paso n

# -------------------------------------------------------------
# Enfoque 1: Fórmula analítica cuadrática (usada en Laboratorio 1)
# (dt * C) * x^2 + (1 - dt * C) * x - x_n = 0
# -------------------------------------------------------------
a = dt * C
b = 1.0 - dt * C
c = -x_n
disc = b**2 - 4.0 * a * c
x_next_analitico = 2.0 * x_n / (b + np.sqrt(disc))

# -------------------------------------------------------------
# Enfoque 2: Resolución numérica con scipy.optimize.root
# Definimos el residuo temporal implícito R(x_{n+1}) = 0
# -------------------------------------------------------------
def residuo_euler_implicito(x_candidato):
    x_cand = x_candidato[0]
    # x_{n+1} - x_n - dt * f(x_{n+1})
    f_ode = C * x_cand * (1.0 - x_cand)
    return [x_cand - x_n - dt * f_ode]

# Warm start con el valor actual x_n
sol = root(residuo_euler_implicito, x0=[x_n])
x_next_root = sol.x[0]

# Comparación numérica
print(f"Solución analítica: {x_next_analitico:.10f}")
print(f"Solución con root:  {x_next_root:.10f}")
print(f"Diferencia:         {abs(x_next_analitico - x_next_root):.2e}")
assert np.isclose(x_next_analitico, x_next_root, atol=1e-12)
```

Salida esperada:
```text
Solución analítica: 0.4494897428
Solución con root:  0.4494897428
Diferencia:         0.00e+00
```

---

## 5. Arquetipo General para Sistemas Multivariables (Modelo SIR)

Para sistemas de varias variables donde el despeje algebraico manual es complejo o inviable, se define la función residual vectorial para todo el vector de estado $y = [S, I, R]^T$:

```python
def paso_sir_implicito(S_n, I_n, R_n, beta, gamma, dt):
    """Avanza un paso de Euler implícito en el modelo SIR resolviendo R(y) = 0."""
    y_actual = np.array([S_n, I_n, R_n], dtype=np.float64)

    def residual_sir(y_cand):
        s, i, r = y_cand
        # Evaluamos el campo vectorial en el candidato y_{n+1}
        ds = -beta * s * i
        di = beta * s * i - gamma * i
        dr = gamma * i
        
        # Ecuación de Euler implícito: y_{n+1} - y_n - dt * f(y_{n+1}) = 0
        return [
            s - S_n - dt * ds,
            i - I_n - dt * di,
            r - R_n - dt * dr
        ]

    # Warm start: estimación inicial igual al estado actual
    res = root(residual_sir, x0=y_actual, method="hybr")
    if not res.success:
        raise RuntimeError("Convergencia fallida en el paso implícito.")
    
    return res.x[0], res.x[1], res.x[2]
```

Este arquetipo permite incorporar términos no lineales arbitrarios (por ejemplo, saturación en el contagio $\frac{\beta S I}{1 + \alpha I}$ o dinámica demográfica) sin necesidad de recalcular raíces cuadráticas analíticas.
