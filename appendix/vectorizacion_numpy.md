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
- **Prealocación:** Al declarar un arreglo, se reserva el espacio exacto en memoria requerida para toda la estructura de datos. Esto elimina la necesidad de redimensionamientos dinámicos o reasignaciones de memoria durante la ejecución de las simulaciones.

---

## 3. Eliminación de Verificaciones Dinámicas y Bucle en el Interprete

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
