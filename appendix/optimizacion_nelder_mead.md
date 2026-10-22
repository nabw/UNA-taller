# Apéndice Técnico: Optimización Libre de Derivadas con Nelder-Mead en SciPy

---

## 1. El Algoritmo Símplex de Nelder-Mead

En problemas inversos de ciencia de datos, a menudo se busca calibrar parámetros $\theta \in \mathbb{R}^d$ minimizando una función de pérdida o discrepancia $L(\theta)$. Cuando evaluar $L(\theta)$ requiere ejecutar un simulador numérico complejo (como la integración temporal de un sistema de EDOs), calcular gradientes analíticos $\nabla L(\theta)$ puede ser impracticable o computacionalmente costoso.

El método de **Nelder-Mead** (1965) es un algoritmo de optimización heurístico directo que **no requiere derivadas**. En un espacio de dimensión $d$, el método mantiene una figura geométrica compuesta por $d+1$ vértices denominada **símplex**:
- Para $d = 1$: un intervalo (2 puntos).
- Para $d = 2$: un triángulo (3 vértices).
- Para $d = 3$: un tetraedro (4 vértices).

### Dinámica Geométrica de Cada Iteración
1. Se evalúa la función objetivo en los $d+1$ vértices del símplex y se ordenan de mejor a peor valor de pérdida.
2. Se calcula el centroide de todos los vértices excepto el peor.
3. Se proyecta el peor vértice a través del centroide mediante transformaciones geométricas elementales:
   - **Reflexión**: Intenta alejarse del peor punto.
   - **Expansión**: Si la reflexión mejora sustancialmente, se acelera en esa dirección favorable.
   - **Contracción**: Si la reflexión no mejora, se prueban puntos intermedios más cercanos al centroide.
   - **Encogimiento (*Shrink*)**: Si ninguna maniobra tiene éxito, todo el símplex se contrae hacia el mejor vértice actual.

El proceso se repite hasta que el tamaño del símplex o la diferencia en los valores de la función cae por debajo de tolerancias numéricas prefijadas (`xatol`, `fatol`).

---

## 2. Implementación con `scipy.optimize.minimize`

En Python, la biblioteca SciPy implementa este método dentro de `scipy.optimize.minimize` con el argumento `method="Nelder-Mead"`.

### Argumentos Principales
- `fun`: Función escalar que recibe el vector $x$ y retorna el valor de pérdida float $L(x)$.
- `x0`: Estimación inicial (vector o lista de longitud $d$).
- `method`: Cadena de texto `"Nelder-Mead"`.
- `callback`: Función opcional que se invoca en cada iteración recibiendo el estado actual $x_k$, útil para registrar la trayectoria de optimización.
- `options`: Diccionario de configuración, típicamente `{"maxiter": 500, "xatol": 1e-6, "fatol": 1e-6}`.

---

## 3. Ejemplo Arquetípico Mínimo (Función 2D)

Minimicemos la función cuadrática convexa:

$$f(x_1, x_2) = (x_1 - 0.40)^2 + 5 (x_2 - 0.10)^2$$

cuyo mínimo global teórico se encuentra en $\theta^* = (0.40, 0.10)$:

```python
import numpy as np
from scipy.optimize import minimize

# 1. Definición de la función objetivo
def funcion_objetivo(theta):
    x1, x2 = theta[0], theta[1]
    return float((x1 - 0.40) ** 2 + 5.0 * (x2 - 0.10) ** 2)

# 2. Punto inicial y registro de trayectoria
theta_0 = [0.20, 0.20]
trayectoria = [np.array(theta_0, dtype=np.float64)]

def registrar_paso(xk):
    trayectoria.append(np.array(xk, dtype=np.float64))

# 3. Optimización con Nelder-Mead
res = minimize(
    funcion_objetivo,
    x0=theta_0,
    method="Nelder-Mead",
    callback=registrar_paso,
    options={"maxiter": 200, "xatol": 1e-6, "fatol": 1e-6}
)

# 4. Inspección de resultados
print(f"¿Convergió exitosamente?: {res.success}")
print(f"Parámetros estimados:     x1 = {res.x[0]:.4f}, x2 = {res.x[1]:.4f}")
print(f"Valor óptimo de pérdida:  {res.fun:.2e}")
print(f"Número de iteraciones:    {res.nit}")
print(f"Puntos en la trayectoria: {len(trayectoria)}")
```

---

## 4. Aplicación en Calibración de Modelos Diferenciales

En el problema inverso epidemiológico (Laboratorio 4), la función objetivo evalúa la discrepancia cuadrática entre las observaciones clínicas $I^{\text{obs}}_k$ y la simulación numérica $I^{\text{sim}}(t_k; \beta, \gamma)$. El algoritmo de Nelder-Mead ajusta dinámicamente el par $(\beta, \gamma)$ recorriendo el paisaje de pérdida hasta encontrar el punto de mínimo error sin requerir derivar el simulador diferencial.
