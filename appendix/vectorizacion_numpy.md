# Apéndice Teórico: Vectorización y Optimización de Rendimiento en NumPy

## 1. Introducción al Modelo de Ejecución de NumPy

El lenguaje Python es un lenguaje interpretado de tipado dinámico. Cuando se ejecutan bucles explícitos (`for`) procesando elementos individuales de una lista o arreglo, el intérprete realiza múltiples verificaciones en cada iteración:
- Búsqueda dinámica de tipos de datos.
- Resolución de métodos y atributos en tiempo de ejecución.
- Verificación de límites en los contenedores.

Estas operaciones introducen una sobrecarga computacional significativa que degrada el rendimiento al simular modelos numéricos con un número elevado de pasos temporales o espaciales.

---

## 2. Prealocación en Memoria Contigua de C

La biblioteca NumPy resuelve esta limitación estructurando los datos en **arreglos multidimensionales homogéneos** (`ndarray`). 

- **Memoria contigua:** A diferencia de las listas estándar de Python (que almacenan punteros dispersos en memoria apuntando a objetos arbitrarios), los arreglos de NumPy almacenan los datos en bloques de memoria contigua, reflejando directamente el modelo de asignación de memoria de lenguajes compilados como C.
- **Prealocación:** Al declarar un arreglo, se reserva el espacio exacto en memoria requerido para toda la estructura de datos. Esto elimina la necesidad de redimensionamientos dinámicos o reasignaciones de memoria durante la ejecución de las simulaciones.

---

## 3. Eliminación de Verificaciones Dinámicas y Bucle en el Intérprete

Cuando se opera sobre arreglos de NumPy mediante funciones vectorizadas (por ejemplo, operaciones aritméticas aplicadas directamente sobre el arreglo completo), el control del bucle de procesamiento se delega a rutinas internas implementadas en código compilado de C o Fortran (`BLAS`, `LAPACK`).

Esto produce los siguientes beneficios de rendimiento:
1. **Eliminación del bucle en Python:** El intérprete de Python deja de ejecutar iteraciones individuales, transfiriendo el control al código máquina nativo.
2. **Ausencia de chequeos de tipos:** Como los elementos del arreglo comparten un único tipo de dato (`dtype`), las operaciones aritméticas se ejecutan directamente a nivel de hardware sin comprobaciones previas de compatibilidad de tipos.

---

## 4. Operaciones Vectorizadas en Bloques y Localidad de Caché

Las operaciones vectorizadas aprovechan la jerarquía de memoria del procesador (memoria caché L1/L2/L3). Al acceder a datos contiguos en bloques:
- Se maximiza la localidad espacial y temporal de las referencias a memoria.
- Se minimizan los fallos de caché (*cache misses*).
- Se habilitan las optimizaciones a nivel de instrucciones de vectorización por hardware (SIMD: *Single Instruction, Multiple Data*), permitiendo procesar múltiples puntos de la malla o de la serie temporal en un solo ciclo de reloj del procesador.

El uso correcto de estas prácticas garantiza que las simulaciones numéricas de modelos epidemiológicos mantengan una ejecución eficiente y escalable.

---

## 5. Ejemplo Práctico: Comparación de Tiempos de Ejecución

Para cuantificar el impacto de la vectorización, comparamos el cálculo de la función de pérdida cuadrática entre una serie simulada y una observada ($L = \sum_{k=1}^N (y_k^{\text{sim}} - y_k^{\text{obs}})^2$) sobre una malla de $N = 2\,000\,000$ puntos, evaluando dos implementaciones:
1. Bucle explícito `for` sobre listas nativas de Python.
2. Operación vectorial en un bloque contiguo con NumPy.

```python
import time
import numpy as np

# Generar datos sintéticos de prueba (2 millones de puntos)
N = 2_000_000
rng = np.random.default_rng(42)
y_sim_arr = rng.uniform(0.0, 1.0, N)
y_obs_arr = rng.uniform(0.0, 1.0, N)

# Listas estándar de Python para comparar contra el bucle interpretado
y_sim_list = list(y_sim_arr)
y_obs_list = list(y_obs_arr)

# -------------------------------------------------------------
# Método 1: Bucle for explícito en Python puro
# -------------------------------------------------------------
t0 = time.perf_counter()
suma_cuadratica_py = 0.0
for i in range(N):
    diff = y_sim_list[i] - y_obs_list[i]
    suma_cuadratica_py += diff * diff
t_py = time.perf_counter() - t0

# -------------------------------------------------------------
# Método 2: Operación vectorizada en NumPy (memoria contigua C)
# -------------------------------------------------------------
t0 = time.perf_counter()
suma_cuadratica_np = np.sum((y_sim_arr - y_obs_arr) ** 2)
t_np = time.perf_counter() - t0

# -------------------------------------------------------------
# Resultados y factor de aceleración
# -------------------------------------------------------------
print(f"Bucle for en Python puro: {t_py * 1000:7.2f} ms")
print(f"NumPy vectorizado:        {t_np * 1000:7.2f} ms")
print(f"Factor de aceleración:    {t_py / t_np:7.1f}x más rápido")

assert np.isclose(suma_cuadratica_py, suma_cuadratica_np, rtol=1e-10)
```

### Resultados Típicos de Rendimiento:

| Implementación | Tiempo de Ejecución ($N = 2 \times 10^6$) | Factor de Aceleración |
| :--- | :---: | :---: |
| Bucle `for` en Python estándar | $\approx 410\text{ ms}$ | $1.0\times$ (base) |
| Operación vectorizada en NumPy | $\approx 6.2\text{ ms}$ | **$\approx 65\times$ más rápido** |

### Conclusión Práctica
En algoritmos iterativos de optimización (como Nelder-Mead en el **Laboratorio 4**), donde la función de pérdida se evalúa cientos de veces para explorar el espacio de parámetros, la vectorización reduce el tiempo total de cómputo de varios minutos a unos pocos segundos.
