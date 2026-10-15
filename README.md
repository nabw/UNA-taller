# Laboratorio: Discretización de Sistemas Dinámicos y Problemas Inversos en Ciencia de Datos

**Curso:** Métodos Numéricos / Ciencia de Datos  
**Carrera:** Ingeniería Matemática en Ciencia de Datos (IMCD)  
**Institución:** Universidad Nacional (UNA), Costa Rica  
**Nivel:** 3.er Año | **Duración:** 3 horas (180 minutos)

---

## 🚀 Acceso en Google Colab (1 Clic)

Puedes ejecutar los laboratorios directamente en la nube sin instalar nada localmente:

| Cuaderno | Descripción | Enlace directo |
| :--- | :--- | :---: |
| **01_laboratorio_estudiante.ipynb** | Versión de trabajo para estudiantes (con stubs `# TU CÓDIGO AQUÍ` y celdas `assert`) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/01_laboratorio_estudiante.ipynb) |
| **01_laboratorio_solucion.ipynb** | Versión completa docente con soluciones, gráficos y respuestas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nabw/UNA-taller/blob/main/notebooks/01_laboratorio_solucion.ipynb) |

---

## 🛠️ Instalación y Ejecución Local

### 1. Clonar e Instalar Entorno
```bash
git clone https://github.com/nabw/UNA-taller.git
cd UNA-taller

# Instalar dependencias
pip install -r requirements.txt

# (Opcional) Instalar en modo editable para importar directamente desde cualquier lugar
pip install -e .
```

### 2. Ejecutar Scripts Crudos en Terminal
Cada bloque temático dispone de un script autocontenido en `scripts/` que genera métricas por consola y guarda figuras explicativas:

```bash
# 1. Ejemplo 1D Logística (IMEX vs Explícito vs Implícito)
python3 scripts/01_ejemplo_logistico.py

# 2. Problema directo SIR (Conservación de invariante y rigidez)
python3 scripts/02_simulacion_sir.py

# 3. Verdad de campo y ruidos (Gaussiano vs Poisson)
python3 scripts/03_datos_sinteticos.py

# 4. Problema inverso (Nelder-Mead, curvas de nivel y sesgo por Delta t)
python3 scripts/04_calibracion_inversa.py
```

### 3. Dashboard Interactivo (Streamlit)
Para explorar dinámicamente en clase con controles deslizantes (sliders de $\beta, \gamma, \Delta t$, visualización en vivo de la trayectoria del símplex y sesgo de discretización):

```bash
streamlit run app.py
```

---

## 📁 Estructura del Repositorio

```text
UNA-taller/
├── src/                               # Módulos numéricos y de graficación
│   ├── __init__.py                    # Exportación unificada de funciones
│   ├── solvers.py                     # Solvers 1D y SIR (Explícito, Implícito, IMEX)
│   ├── synthetic.py                   # Generación de verdad de campo y modelos de ruido
│   ├── inverse.py                     # Pérdida, grilla 2D, Nelder-Mead y sesgo
│   └── plotting.py                    # Visualizaciones didácticas con Matplotlib
├── scripts/                           # Scripts crudos para terminal
│   ├── 01_ejemplo_logistico.py        # Demo 1D logística y límite de estabilidad
│   ├── 02_simulacion_sir.py           # Simulación SIR y análisis de rigidez
│   ├── 03_datos_sinteticos.py         # Muestreo y modelos de ruido clínico
│   ├── 04_calibracion_inversa.py      # Ajuste libre de derivadas y sesgo por Delta t
│   └── figura_*.png                   # Gráficos generados
├── appendix/                          # Documentos teóricos y prácticos complementarios
│   ├── estabilidad_temporal.md        # Análisis formal de estabilidad lineal (Dahlquist)
│   └── vectorizacion_numpy.md         # Guía de rendimiento: bucles vs comprensiones vs NumPy
├── notebooks/                         # Cuadernos Jupyter interactivos
│   ├── 01_laboratorio_estudiante.ipynb# Versión guiada para el estudiante
│   └── 01_laboratorio_solucion.ipynb  # Versión completa con soluciones y figuras
├── app.py                             # Aplicación web interactiva en Streamlit
├── requirements.txt                   # Dependencias científicas
├── pyproject.toml                     # Metadata de empaquetado pip
└── README.md                          # Guía del laboratorio
```
