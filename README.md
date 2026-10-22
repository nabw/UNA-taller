# Taller: Discretización de Sistemas Dinámicos y Problemas Inversos en Ciencia de Datos

**Curso:** Modelamiento Matemático  
**Carrera:** Ingeniería Matemática en Ciencia de Datos (IMCD)  
**Institución:** Universidad Nacional (UNA), Costa Rica  
**Nivel:** 2.º Año | **Duración:** 3 horas

---

## Cuadernos de Trabajo (Acceso en Google Colab)

El taller está dividido en cuatro cuadernos autocontenidos y progresivos. Puedes abrirlos y ejecutarlos directamente en Google Colab con un clic, sin necesidad de instalar nada en tu computadora:

| Cuaderno | Temática | Versión Estudiante | Versión Solución |
| :--- | :--- | :---: | :---: |
| **01: Modelo Logístico** | EDO 1D, Euler explícito, implícito e IMEX, estabilidad según el paso temporal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/estudiante/01_modelo_logistico.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/solucion/01_modelo_logistico.ipynb) |
| **02: Modelo SIR y Estabilidad** | Modelo epidemiológico SIR, conservación de población total, comparación de esquemas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/estudiante/02_modelo_sir_estabilidad.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/solucion/02_modelo_sir_estabilidad.ipynb) |
| **03: Datos Sintéticos y Ruido** | Simulación de referencia con RK45, muestreo discreto, ruido observacional Gaussiano y análisis de residuos | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/estudiante/03_datos_sinteticos_ruido.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/solucion/03_datos_sinteticos_ruido.ipynb) |
| **04: Calibración y Problema Inverso** | Ajuste de parámetros por mínimos cuadrados, optimización Nelder-Mead y análisis del sesgo de discretización | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/estudiante/04_calibracion_problema_inverso.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/solucion/04_calibracion_problema_inverso.ipynb) |

---

## Instalación y Ejecución Local (Opcional)

Ejecutar los cuadernos en Google Colab es la opción recomendada para el taller. De manera opcional, si prefieres trabajar en tu máquina local o ejecutar los scripts de consola:

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
- **[Método de Runge-Kutta y SciPy](appendix/runge_kutta_scipy.md):** Formulación algebraica de RK4, tasa de convergencia de orden 4 y ejemplos prácticos de uso de `scipy.integrate.solve_ivp` para modelos 1D y sistemas 2D.
- **[Resolución de Ecuaciones No Lineales](appendix/solucion_no_lineal_scipy.md):** Formulación de problemas de raíz en integración implícita y arquetipo de resolución con `scipy.optimize.root`.
- **[Interpolación de Datos en Python](appendix/interpolacion_datos.md):** Métodos de submuestreo y alineamiento temporal con `np.interp` y splines cúbicos de SciPy.
- **[Guía de Vectorización en NumPy](appendix/vectorizacion_numpy.md):** Fundamentos prácticos sobre prealocación en memoria contigua de C, eliminación de verificaciones del intérprete y operaciones en bloques.
- **[Ruido de Conteo y Distribución de Poisson](appendix/ruido_conteo_poisson.md):** Justificación matemática de por qué el ruido Gaussiano es solo una aproximación inicial y detalles del modelo de Poisson para conteos epidemiológicos discretos.

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
│   ├── runge_kutta_scipy.md           # Formulación algebraica de RK4 y solve_ivp
│   ├── solucion_no_lineal_scipy.md    # Ecuaciones no lineales implícitas con SciPy
│   ├── interpolacion_datos.md         # Submuestreo e interpolación con NumPy y SciPy
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
