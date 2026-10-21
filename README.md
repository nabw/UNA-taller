# Taller: Discretización de Sistemas Dinámicos y Problemas Inversos en Ciencia de Datos

**Curso:** Métodos Numéricos / Modelamiento Matemático  
**Carrera:** Ingeniería Matemática en Ciencia de Datos (IMCD)  
**Institución:** Universidad Nacional (UNA), Costa Rica  
**Nivel:** 2.º Año | **Duración:** 3 horas

---

## Cuadernos de Trabajo

El taller está dividido en cuatro cuadernos autocontenidos y progresivos. Cada cuaderno dispone de una versión para estudiantes (con funciones incompletas y pruebas de verificación) y una versión docente con la solución completa:

| Cuaderno | Temática | Versión Estudiante | Versión Solución |
| :--- | :--- | :---: | :---: |
| **01: Modelo Logístico** | EDO 1D, Euler explícito, implícito e IMEX, estabilidad según el paso temporal | [Estudiante](notebooks/estudiante/01_modelo_logistico.ipynb) | [Solución](notebooks/solucion/01_modelo_logistico.ipynb) |
| **02: Modelo SIR y Estabilidad** | Modelo epidemiológico SIR, conservación de población total, comparación de esquemas y fallas de Euler explícito | [Estudiante](notebooks/estudiante/02_modelo_sir_estabilidad.ipynb) | [Solución](notebooks/solucion/02_modelo_sir_estabilidad.ipynb) |
| **03: Datos Sintéticos y Ruido** | Simulación de referencia con RK45, muestreo discreto, ruido observacional Gaussiano y análisis de residuos | [Estudiante](notebooks/estudiante/03_datos_sinteticos_ruido.ipynb) | [Solución](notebooks/solucion/03_datos_sinteticos_ruido.ipynb) |
| **04: Calibración y Problema Inverso** | Ajuste de parámetros por mínimos cuadrados, optimización Nelder-Mead y análisis del sesgo de discretización | [Estudiante](notebooks/estudiante/04_calibracion_problema_inverso.ipynb) | [Solución](notebooks/solucion/04_calibracion_problema_inverso.ipynb) |

---

## Instalación y Ejecución Local

### 1. Clonar el repositorio e instalar dependencias

```bash
git clone https://github.com/nabw/UNA-taller.git
cd UNA-taller

# Instalar dependencias científicas
pip install -r requirements.txt

# (Opcional) Instalar en modo editable para importar el paquete src
pip install -e .
```

### 2. Ejecutar los scripts en terminal

Cada bloque temático cuenta con un script independiente en la carpeta `scripts/` que muestra resultados numéricos por consola y genera figuras en formato PNG:

```bash
# 1. Ejemplo 1D logístico (Euler explícito vs implícito vs IMEX)
python3 scripts/01_ejemplo_logistico.py

# 2. Problema directo SIR (Conservación de masa y sobreestimación con paso grande)
python3 scripts/02_simulacion_sir.py

# 3. Datos sintéticos con ruido Gaussiano y análisis de residuos
python3 scripts/03_datos_sinteticos.py

# 4. Problema inverso (Nelder-Mead, curvas de nivel y sesgo por Delta t)
python3 scripts/04_calibracion_inversa.py
```

---

## Apéndices Complementarios

En la carpeta `appendix/` se incluyen documentos teóricos y prácticos complementarios para profundizar en aspectos específicos del taller:

- **[Estabilidad Temporal](appendix/estabilidad_temporal.md):** Formalización de la estabilidad lineal con la ecuación de prueba de Dahlquist, el factor de amplificación y la deducción del Euler implícito cuadrático para SIR.
- **[Guía de Vectorización en NumPy](appendix/vectorizacion_numpy.md):** Fundamentos prácticos sobre prealocación en memoria contigua de C, eliminación de verificaciones del intérprete y operaciones en bloques.
- **[Ruido de Conteo y Distribución de Poisson](appendix/ruido_conteo_poisson.md):** Justificación matemática de por qué el ruido Gaussiano es solo una aproximación pedagógica inicial y detalles del modelo de Poisson para conteos epidemiológicos discretos.

---

## Estructura del Repositorio

```text
UNA-taller/
├── src/                               # Funciones numéricas y de visualización
│   ├── __init__.py                    # Exportación del paquete
│   ├── solvers.py                     # Solvers 1D y SIR (Explícito, Implícito, IMEX)
│   ├── synthetic.py                   # Generación de referencia y modelos de ruido
│   ├── inverse.py                     # Función de pérdida, grilla 2D y Nelder-Mead
│   └── plotting.py                    # Gráficos con Matplotlib
├── scripts/                           # Scripts ejecutables en consola
│   ├── 01_ejemplo_logistico.py        # Simulación y estabilidad 1D
│   ├── 02_simulacion_sir.py           # Simulación SIR y paso temporal crítico
│   ├── 03_datos_sinteticos.py         # Muestreo con ruido observacional Gaussiano
│   ├── 04_calibracion_inversa.py      # Calibración Nelder-Mead y sesgo de Delta t
│   └── figura_*.png                   # Gráficos generados
├── appendix/                          # Documentos teóricos complementarios
│   ├── estabilidad_temporal.md        # Estabilidad lineal y deducción SIR implícito
│   ├── vectorizacion_numpy.md         # Rendimiento y memoria en NumPy
│   └── ruido_conteo_poisson.md        # Comparación teórica de ruido Gaussiano vs Poisson
├── notebooks/                         # Cuadernos interactivos Jupyter
│   ├── estudiante/                    # Versión con ejercicios para estudiantes
│   │   ├── 01_modelo_logistico.ipynb
│   │   ├── 02_modelo_sir_estabilidad.ipynb
│   │   ├── 03_datos_sinteticos_ruido.ipynb
│   │   └── 04_calibracion_problema_inverso.ipynb
│   └── solucion/                      # Versión con soluciones completas
│       ├── 01_modelo_logistico.ipynb
│       ├── 02_modelo_sir_estabilidad.ipynb
│       ├── 03_datos_sinteticos_ruido.ipynb
│       └── 04_calibracion_problema_inverso.ipynb
├── requirements.txt                   # Dependencias científicas
├── pyproject.toml                     # Configuración de paquete pip
└── README.md                          # Guía del taller
```
