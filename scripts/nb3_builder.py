"""
Constructor del Cuaderno 3: Datos Sintéticos y Ruido Observacional en el Modelo SIR.
Produce celdas para las versiones 'estudiante' y 'solucion'.
"""

from typing import List, Dict, Any


def build_nb3_cells(is_solution: bool) -> List[Dict[str, Any]]:
    cells = []

    # Celda 1: Markdown - Introducción al Laboratorio in Silico
    cells.append({
        "type": "markdown",
        "source": (
            "# Laboratorio 3: Generación de Datos Sintéticos y Modelado de Ruido Observacional\n\n"
            "## 1. Introducción: El Laboratorio in Silico\n\n"
            "En la práctica de la ciencia de datos aplicada a sistemas dinámicos, rara vez se dispone "
            "de trayectorias continuas exactas y libres de error. Los registros clínicos y epidemiológicos "
            "se recolectan a intervalos discretos de tiempo (por ejemplo, reportes hospitalarios diarios) "
            "y están afectados por diversas fuentes de incertidumbre: errores de muestreo, subnotificación, "
            "sensibilidad imperfecta de las pruebas diagnósticas y demoras en el procesamiento de datos.\n\n"
            "Para desarrollar y validar algoritmos de calibración e inferencia de parámetros (problemas inversos), "
            "es necesario contar con un entorno de prueba controlado: un **laboratorio in silico**. "
            "En este laboratorio generamos observaciones sintéticas a partir de un modelo mecanístico cuya "
            "**verdad de campo** (*Ground Truth*) conocemos con precisión absoluta. Esto permite cuantificar "
            "de forma rigurosa la exactitud, robustez y sesgo de los métodos de estimación.\n\n"
            "## Objetivos de Aprendizaje\n\n"
            "1. Simular la verdad de campo continua del modelo SIR ($\\beta = 0.4$, $\\gamma = 0.1$) mediante RK45 adaptativo en una malla densa.\n"
            "2. Implementar el submuestreo temporal discreto en días enteros $t_k \\in \\{0, 1, 2, \\dots, 50\\}$ usando interpolación.\n"
            "3. Modelar el proceso observacional incorporando ruido Gaussiano $\\mathcal{N}(0, \\sigma^2)$ con "
            "truncamiento a no-negatividad física: $I^{\\text{obs}} = \\max(0, I + \\varepsilon)$.\n"
            "4. Efectuar un análisis estadístico riguroso de los residuos observacionales (media, desviación estándar, banda $\\pm 2\\sigma$).\n"
            "5. Visualizar las observaciones clínicas y caracterizar la incertidumbre instrumental."
        )
    })

    # Celda 2: Código - Configuración de entorno
    cells.append({
        "type": "code",
        "source": (
            "# Celda de configuración y librerías científicas\n"
            "import os\n"
            "os.environ.setdefault(\"MPLCONFIGDIR\", \"/tmp/matplotlib\")\n"
            "from typing import Any\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "from scipy.integrate import solve_ivp\n"
            "from scipy import stats\n\n"
            "plt.style.use(\"seaborn-v0_8-whitegrid\" if \"seaborn-v0_8-whitegrid\" in plt.style.available else \"default\")\n"
            "plt.rcParams.update({\n"
            "    \"font.size\": 11,\n"
            "    \"axes.labelsize\": 12,\n"
            "    \"axes.titlesize\": 12,\n"
            "    \"figure.titlesize\": 13,\n"
            "    \"lines.linewidth\": 2.0,\n"
            "    \"grid.alpha\": 0.4\n"
            "})\n\n"
            "print(\"Configuración inicial completada.\")"
        )
    })

    # Celda 3: Markdown - Verdad de Campo Continua y Malla Densa
    cells.append({
        "type": "markdown",
        "source": (
            "## 2. Generación de la Verdad de Campo (Ground Truth)\n\n"
            "Consideramos el sistema continuo SIR:\n\n"
            "$$\\frac{dS}{dt} = -\\beta S I$$\n"
            "$$\\frac{dI}{dt} = \\beta S I - \\gamma I$$\n"
            "$$\\frac{dR}{dt} = \\gamma I$$\n\n"
            "con parámetros nominales verdaderos:\n"
            "- $\\beta^* = 0.40$ (tasa de transmisión)\n"
            "- $\\gamma^* = 0.10$ (tasa de recuperación)\n"
            "- $R_0^* = \\beta^* / \\gamma^* = 4.00$\n"
            "- Condiciones iniciales: $S(0) = 0.99$, $I(0) = 0.01$, $R(0) = 0.00$\n"
            "- Horizonte temporal: $T = 50.0$ días\n\n"
            "### ¿Qué significa evaluar en una 'Malla Densa'?\n\n"
            "En simulación numérica, una **malla densa** significa simplemente utilizar un **paso temporal chiquito** "
            "(en nuestro caso, $\\Delta t_{\\text{eval}} = 0.005$ días, lo que produce más de 10,000 puntos en el intervalo $[0, 50]$). "
            "Esto permite que la trayectoria calculada por el integrador adaptativo `solve_ivp` sea visual y analíticamente "
            "indistinguible de una curva continua suave.\n\n"
            "**Instrucciones de implementación para `solve_ivp`:**\n"
            "1. Construye el vector de tiempos densos con `np.linspace(0.0, T, n_points)`, donde `n_points = int(np.round(T / dt_eval)) + 1`.\n"
            "2. Invoca el integrador con:\n"
            "   ```python\n"
            "   sol = solve_ivp(sir_ode, t_span=(0.0, T), y0=y0, t_eval=t_eval, method='RK45', rtol=1e-10, atol=1e-12)\n"
            "   ```\n"
            "3. Retorna `sol.t, sol.y[0], sol.y[1], sol.y[2]`.\n\n"
            "### Especificación de la Función a Implementar\n"
            "- **Firma:** `generar_verdad_campo(beta: float = 0.4, gamma: float = 0.1, S0: float = 0.99, I0: float = 0.01, T: float = 50.0, dt_eval: float = 0.005) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`\n"
            "- **Parámetros de entrada:**\n"
            "  - `beta` (`float`, opcional, por defecto `0.4`): Tasa efectiva de transmisión $\\beta$.\n"
            "  - `gamma` (`float`, opcional, por defecto `0.1`): Tasa media de recuperación $\\gamma$.\n"
            "  - `S0` (`float`, opcional, por defecto `0.99`): Fracción inicial de susceptibles.\n"
            "  - `I0` (`float`, opcional, por defecto `0.01`): Fracción inicial de infectados.\n"
            "  - `T` (`float`, opcional, por defecto `50.0`): Tiempo final de simulación.\n"
            "  - `dt_eval` (`float`, opcional, por defecto `0.005`): Paso temporal de evaluación en la malla densa.\n"
            "- **Salida esperada:**\n"
            "  - `tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`: Cuádrupla `(t, S, I, R)` con arreglos unidimensionales "
            "de NumPy calculados por `solve_ivp` con método `RK45`."
        )
    })

    # Celda 4: Código - Función generar_verdad_campo
    if is_solution:
        code_gt = (
            "def generar_verdad_campo(\n"
            "    beta: float = 0.4,\n"
            "    gamma: float = 0.1,\n"
            "    S0: float = 0.99,\n"
            "    I0: float = 0.01,\n"
            "    T: float = 50.0,\n"
            "    dt_eval: float = 0.005\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Genera la solución continua de referencia (Ground Truth) para el modelo SIR.\n"
            "    \"\"\"\n"
            "    R0_val = 1.0 - S0 - I0\n"
            "    y0 = np.array([S0, I0, R0_val], dtype=np.float64)\n\n"
            "    def sir_ode(t: float, y: np.ndarray) -> np.ndarray:\n"
            "        s, i, r = y\n"
            "        ds = -beta * s * i\n"
            "        di = beta * s * i - gamma * i\n"
            "        dr = gamma * i\n"
            "        return np.array([ds, di, dr], dtype=np.float64)\n\n"
            "    n_points = int(np.round(T / dt_eval)) + 1\n"
            "    t_eval = np.linspace(0.0, T, n_points)\n\n"
            "    sol = solve_ivp(\n"
            "        sir_ode,\n"
            "        t_span=(0.0, T),\n"
            "        y0=y0,\n"
            "        t_eval=t_eval,\n"
            "        method=\"RK45\",\n"
            "        rtol=1e-10,\n"
            "        atol=1e-12\n"
            "    )\n"
            "    return sol.t, sol.y[0], sol.y[1], sol.y[2]"
        )
    else:
        code_gt = (
            "def generar_verdad_campo(\n"
            "    beta: float = 0.4,\n"
            "    gamma: float = 0.1,\n"
            "    S0: float = 0.99,\n"
            "    I0: float = 0.01,\n"
            "    T: float = 50.0,\n"
            "    dt_eval: float = 0.005\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Genera la solución continua de referencia (Ground Truth) para el modelo SIR.\n"
            "    \"\"\"\n"
            "    # TU CÓDIGO AQUÍ\n"
            "    ..."
        )
    cells.append({"type": "code", "source": code_gt})

    # Celda 5: Código - Pruebas unitarias generar_verdad_campo
    cells.append({
        "type": "code",
        "source": (
            "# Pruebas de validación para generar_verdad_campo\n"
            "t_ref, S_ref, I_ref, R_ref = generar_verdad_campo(0.4, 0.1, 0.99, 0.01, 50.0)\n"
            "assert np.isclose(S_ref[0], 0.99) and np.isclose(I_ref[0], 0.01), \"Condiciones iniciales incorrectas.\"\n"
            "assert np.max(np.abs(S_ref + I_ref + R_ref - 1.0)) < 1e-6, \"Conservación de masa no satisfecha.\"\n"
            "print(\"Validación de generar_verdad_campo: EXITOSA.\")"
        )
    })

    # Celda 6: Markdown - Muestreo en Tiempos de Reporte Discreto
    cells.append({
        "type": "markdown",
        "source": (
            "## 3. Muestreo en Tiempos de Reporte Clínico Discreto\n\n"
            "En los sistemas de vigilancia epidemiológica, las notificaciones se registran en "
            "días enteros: $t_k \\in \\{0, 1, 2, \\dots, 50\\}$. Para obtener la serie temporal "
            "limpia $I^{\\text{clean}}(t_k)$, existen dos enfoques:\n\n"
            "1. **Vía directa de SciPy**: `solve_ivp` permite especificar directamente `t_eval = np.arange(0.0, T + 1.0, 1.0)` "
            "para que el integrador entregue la solución evaluada exclusivamente en los días de reporte.\n"
            "2. **Vía interpolación 1D**: Si ya contamos con la trayectoria continua sobre la malla densa $(t_{\\text{ref}}, I_{\\text{ref}})$, "
            "podemos submuestrear en los días requeridos utilizando la función de interpolación lineal de NumPy:\n"
            "   ```python\n"
            "   t_dias = np.arange(0.0, T_final + 1.0, 1.0)\n"
            "   I_clean = np.interp(t_dias, t_ref, I_ref)\n"
            "   ```\n"
            "   donde `np.interp(x_nuevo, x_conocido, y_conocido)` calcula rápidamente los valores interpolados.\n\n"
            "Aprovechamos la simplicidad de `np.interp` para desacoplar el submuestreo de la integración continua. "
            "Para profundizar en técnicas de interpolación en Python (incluyendo splines cúbicos), consulta el "
            "[Apéndice Técnico: Interpolación de Datos en Python](https://github.com/nabw/UNA-taller/blob/main/appendix/interpolacion_datos.md).\n\n"
            "### Especificación de la Función a Implementar\n"
            "- **Firma:** `muestrear_dias_enteros(t_ref: np.ndarray, I_ref: np.ndarray, t_dias: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]`\n"
            "- **Parámetros de entrada:**\n"
            "  - `t_ref` (`np.ndarray`): Malla temporal continua densa.\n"
            "  - `I_ref` (`np.ndarray`): Trayectoria continua de infectados sobre la malla densa.\n"
            "  - `t_dias` (`np.ndarray`, opcional): Vector de instantes discretos de reporte. Si es `None`, se genera como `np.arange(0.0, T_final + 1.0, 1.0)`.\n"
            "- **Salida esperada:**\n"
            "  - `tuple[np.ndarray, np.ndarray]`: Tupla `(t_dias, I_clean)` donde `t_dias` es el arreglo de días discretos "
            "y `I_clean` es el arreglo de valores interpolados de infectados en dichos días."
        )
    })

    # Celda 7: Código - Función muestrear_dias_enteros
    if is_solution:
        code_sample = (
            "def muestrear_dias_enteros(\n"
            "    t_ref: np.ndarray,\n"
            "    I_ref: np.ndarray,\n"
            "    t_dias: np.ndarray | None = None\n"
            ") -> tuple[np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Submuestrea la serie continua de infectados sobre instantes de tiempo discretos.\n"
            "    \"\"\"\n"
            "    if t_dias is None:\n"
            "        T_final = float(t_ref[-1])\n"
            "        t_dias = np.arange(0.0, T_final + 1.0, 1.0)\n"
            "    else:\n"
            "        t_dias = np.asarray(t_dias, dtype=np.float64)\n\n"
            "    I_clean = np.interp(t_dias, t_ref, I_ref)\n"
            "    return t_dias, I_clean"
        )
    else:
        code_sample = (
            "def muestrear_dias_enteros(\n"
            "    t_ref: np.ndarray,\n"
            "    I_ref: np.ndarray,\n"
            "    t_dias: np.ndarray | None = None\n"
            ") -> tuple[np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Submuestrea la serie continua de infectados sobre instantes de tiempo discretos.\n"
            "    \"\"\"\n"
            "    # TU CÓDIGO AQUÍ\n"
            "    ..."
        )
    cells.append({"type": "code", "source": code_sample})

    # Celda 8: Código - Pruebas unitarias muestrear_dias_enteros
    cells.append({
        "type": "code",
        "source": (
            "# Pruebas de validación para muestrear_dias_enteros\n"
            "t_dias, I_clean = muestrear_dias_enteros(t_ref, I_ref)\n"
            "assert len(t_dias) == 51, f\"Se esperaban 51 puntos diarios (0 a 50), obtenidos {len(t_dias)}\"\n"
            "assert np.isclose(t_dias[0], 0.0) and np.isclose(t_dias[-1], 50.0), \"Los extremos temporales no coinciden.\"\n"
            "assert np.all(I_clean >= 0.0) and np.all(I_clean <= 1.0), \"Valores fuera del rango físico [0, 1].\"\n"
            "print(\"Validación de muestrear_dias_enteros: EXITOSA.\")"
        )
    })

    # Celda 9: Markdown - Ruido Gaussiano y Truncamiento Físico
    cells.append({
        "type": "markdown",
        "source": (
            "## 4. Proceso de Medición: Ruido Gaussiano y Truncamiento Físico\n\n"
            "Para modelar el error observacional aditivo, perturbamos las mediciones con variables "
            "aleatorias normales independientes e idénticamente distribuidas:\n\n"
            "$$\\varepsilon_k \\sim \\mathcal{N}(0, \\sigma^2), \\quad \\sigma = 0.02$$\n\n"
            "Dado que $I(t)$ representa la proporción de la población infectada, una observación "
            "negativa es físicamente inadmisible. Se suele imponer una proyección no lineal mediante truncamiento inferior:\n\n"
            "$$I^{\\text{obs}}_k = \\max(0, I^{\\text{clean}}(t_k) + \\varepsilon_k)$$\n\n"
            "### Comandos de Python para Generar Ruido y Truncar\n\n"
            "Para implementar esta función en Python se utilizan los siguientes comandos:\n"
            "```python\n"
            "# 1. Inicializar el generador pseudoaleatorio con la semilla indicada\n"
            "rng = np.random.default_rng(seed)\n\n"
            "# 2. Generar ruido normal con media 0 y desviación estándar sigma\n"
            "noise = rng.normal(loc=0.0, scale=sigma, size=I_clean.shape)\n\n"
            "# 3. Sumar el ruido y truncar valores negativos a cero con np.maximum\n"
            "I_noisy = np.maximum(0.0, I_clean + noise)\n"
            "```\n\n"
            "### Advertencia Teórica sobre el Truncamiento\n\n"
            "Aunque la operación `np.maximum(0.0, ...)` previene valores negativos, **desde el punto de vista estadístico "
            "este truncamiento no es una buena práctica teórica**: deforma artificialmente la distribución normal cerca de cero "
            "(acumulando una masa de probabilidad no nula en 0 y sesgando la media de los residuos hacia valores positivos).\n\n"
            "En la práctica epidemiológica real, los datos hospitalarios provienen de conteos enteros de individuos "
            "($C_k \\in \\{0, 1, 2, \\dots\\}$), cuya naturaleza discreta, varianza variable y no-negatividad natural "
            "se modelan rigurosamente mediante procesos de **Poisson**. Para una explicación matemática detallada de "
            "por qué Poisson es superior al ruido Gaussiano truncado, consulta el "
            "[Apéndice Teórico: Ruido de Conteo y Distribución de Poisson](https://github.com/nabw/UNA-taller/blob/main/appendix/ruido_conteo_poisson.md).\n\n"
            "### Especificación de la Función a Implementar\n"
            "- **Firma:** `agregar_ruido_gaussiano(I_clean: np.ndarray, sigma: float = 0.02, seed: int = 42) -> np.ndarray`\n"
            "- **Parámetros de entrada:**\n"
            "  - `I_clean` (`np.ndarray`): Arreglo unidimensional con las observaciones limpias submuestreadas.\n"
            "  - `sigma` (`float`, opcional, por defecto `0.02`): Desviación estándar del ruido normal aditivo $\\sigma$.\n"
            "  - `seed` (`int`, opcional, por defecto `42`): Semilla entera para reproducibilidad del generador `np.random.default_rng`.\n"
            "- **Salida esperada:**\n"
            "  - `np.ndarray`: Arreglo unidimensional de igual longitud que `I_clean` con las observaciones perturbadas y proyectadas a no-negatividad."
        )
    })

    # Celda 10: Código - Función agregar_ruido_gaussiano
    if is_solution:
        code_noise = (
            "def agregar_ruido_gaussiano(\n"
            "    I_clean: np.ndarray,\n"
            "    sigma: float = 0.02,\n"
            "    seed: int = 42\n"
            ") -> np.ndarray:\n"
            "    \"\"\"\n"
            "    Agrega ruido aditivo Gaussiano N(0, sigma^2) con proyección a no-negatividad.\n"
            "    \"\"\"\n"
            "    I_arr = np.asarray(I_clean, dtype=np.float64)\n"
            "    rng = np.random.default_rng(seed)\n"
            "    noise = rng.normal(loc=0.0, scale=sigma, size=I_arr.shape)\n"
            "    I_noisy = np.maximum(0.0, I_arr + noise)\n"
            "    return I_noisy"
        )
    else:
        code_noise = (
            "def agregar_ruido_gaussiano(\n"
            "    I_clean: np.ndarray,\n"
            "    sigma: float = 0.02,\n"
            "    seed: int = 42\n"
            ") -> np.ndarray:\n"
            "    \"\"\"\n"
            "    Agrega ruido aditivo Gaussiano N(0, sigma^2) con proyección a no-negatividad.\n"
            "    \"\"\"\n"
            "    # TU CÓDIGO AQUÍ\n"
            "    ..."
        )
    cells.append({"type": "code", "source": code_noise})

    # Celda 11: Código - Pruebas unitarias agregar_ruido_gaussiano
    cells.append({
        "type": "code",
        "source": (
            "# Pruebas de validación para agregar_ruido_gaussiano\n"
            "I_obs_test = agregar_ruido_gaussiano(I_clean, sigma=0.02, seed=42)\n"
            "assert I_obs_test.shape == I_clean.shape, \"La forma del arreglo no coincide.\"\n"
            "assert np.all(I_obs_test >= 0.0), \"Violación de no-negatividad física.\"\n"
            "# Verificar determinismo mediante semilla fija\n"
            "I_obs_test2 = agregar_ruido_gaussiano(I_clean, sigma=0.02, seed=42)\n"
            "assert np.array_equal(I_obs_test, I_obs_test2), \"El generador debe ser determinista con semilla fija.\"\n"
            "print(\"Validación de agregar_ruido_gaussiano: EXITOSA.\")"
        )
    })

    # Celda 12: Markdown - Análisis Estadístico de Residuos
    cells.append({
        "type": "markdown",
        "source": (
            "## 5. Análisis Estadístico de Residuos Observacionales\n\n"
            "Definimos el residuo observacional en cada instante muestral como la desviación "
            "entre el valor observado y el valor real subyacente:\n\n"
            "$$e_k = I^{\\text{obs}}_k - I^{\\text{clean}}(t_k)$$\n\n"
            "Propiedades estadísticas a verificar y comandos en Python / SciPy:\n\n"
            "1. **Media muestral**: $\\bar{e} = \\frac{1}{M} \\sum_{k=1}^M e_k \\approx 0$, "
            "lo que confirma la ausencia de sesgo sistemático en el proceso de captura.\n"
            "   ```python\n"
            "   media = float(np.mean(residuos))\n"
            "   ```\n"
            "2. **Desviación estándar muestral**: $s_e = \\sqrt{\\frac{1}{M-1} \\sum_{k=1}^M (e_k - \\bar{e})^2} \\approx \\sigma = 0.02$.\n"
            "   En NumPy se debe especificar `ddof=1` para aplicar la corrección de Bessel (divisor $M-1$):\n"
            "   ```python\n"
            "   std = float(np.std(residuos, ddof=1))\n"
            "   # O de forma equivalente con SciPy: float(stats.tstd(residuos))\n"
            "   ```\n"
            "3. **Banda de confianza $\\pm 2\\sigma$**: Para una distribución normal, aproximadamente el $95.4\\%$ "
            "de las observaciones deben residir en $[-\\text{2}\\sigma, +\\text{2}\\sigma]$:\n"
            "   ```python\n"
            "   dentro_banda = np.abs(residuos) <= 2.0 * sigma\n"
            "   pct_banda = float(np.mean(dentro_banda)) * 100.0\n"
            "   ```\n\n"
            "### Especificación de la Función a Implementar\n"
            "- **Firma:** `analizar_residuos(I_clean: np.ndarray, I_obs: np.ndarray, sigma: float = 0.02) -> dict[str, Any]`\n"
            "- **Parámetros de entrada:**\n"
            "  - `I_clean` (`np.ndarray`): Serie limpia subyacente de referencia.\n"
            "  - `I_obs` (`np.ndarray`): Serie observada ruidosa.\n"
            "  - `sigma` (`float`, opcional, por defecto `0.02`): Desviación estándar nominal del ruido.\n"
            "- **Salida esperada:**\n"
            "  - `dict[str, Any]`: Diccionario con las claves exactas:\n"
            "    - `'residuos'`: arreglo `np.ndarray` con $e_k = I^{\\text{obs}}_k - I^{\\text{clean}}_k$.\n"
            "    - `'media'`: valor `float` de la media muestral $\\bar{e}$.\n"
            "    - `'desviacion_std'`: valor `float` de la desviación estándar muestral ($ddof=1$).\n"
            "    - `'porcentaje_2sigma'`: valor `float` con el porcentaje de observaciones dentro de $[-\\text{2}\\sigma, +\\text{2}\\sigma]$."
        )
    })

    # Celda 13: Código - Función analizar_residuos
    if is_solution:
        code_res = (
            "def analizar_residuos(\n"
            "    I_clean: np.ndarray,\n"
            "    I_obs: np.ndarray,\n"
            "    sigma: float = 0.02\n"
            ") -> dict[str, Any]:\n"
            "    \"\"\"\n"
            "    Calcula los estadísticos descriptivos de los residuos observacionales.\n"
            "    \"\"\"\n"
            "    residuos = np.asarray(I_obs, dtype=np.float64) - np.asarray(I_clean, dtype=np.float64)\n"
            "    media = float(np.mean(residuos))\n"
            "    std = float(np.std(residuos, ddof=1))\n"
            "    dentro_banda = np.abs(residuos) <= 2.0 * sigma\n"
            "    pct_banda = float(np.mean(dentro_banda)) * 100.0\n\n"
            "    return {\n"
            "        \"residuos\": residuos,\n"
            "        \"media\": media,\n"
            "        \"desviacion_std\": std,\n"
            "        \"porcentaje_2sigma\": pct_banda\n"
            "    }"
        )
    else:
        code_res = (
            "def analizar_residuos(\n"
            "    I_clean: np.ndarray,\n"
            "    I_obs: np.ndarray,\n"
            "    sigma: float = 0.02\n"
            ") -> dict[str, Any]:\n"
            "    \"\"\"\n"
            "    Calcula los estadísticos descriptivos de los residuos observacionales.\n"
            "    \"\"\"\n"
            "    # TU CÓDIGO AQUÍ\n"
            "    ..."
        )
    cells.append({"type": "code", "source": code_res})

    # Celda 14: Código - Pruebas unitarias analizar_residuos
    cells.append({
        "type": "code",
        "source": (
            "# Pruebas de validación para analizar_residuos\n"
            "stats_res = analizar_residuos(I_clean, I_obs_test, sigma=0.02)\n"
            "assert isinstance(stats_res, dict), \"La función debe retornar un diccionario.\"\n"
            "assert \"media\" in stats_res and \"desviacion_std\" in stats_res and \"porcentaje_2sigma\" in stats_res\n"
            "assert abs(stats_res[\"media\"]) < 0.01, f\"Media residual excesiva: {stats_res['media']}\"\n"
            "assert 0.01 < stats_res[\"desviacion_std\"] < 0.03, f\"Desviación residual anormal: {stats_res['desviacion_std']}\"\n"
            "print(\"Validación de analizar_residuos: EXITOSA.\")"
        )
    })

    # Celda 15: Markdown - Simulación completa y visualización
    cells.append({
        "type": "markdown",
        "source": (
            "## 6. Generación Completa y Caracterización Gráfica\n\n"
            "Ejecutamos el flujo completo de generación con la semilla $42$ y $\\sigma = 0.02$. "
            "Construimos dos gráficos científicos:\n"
            "1. **Curva de referencia continua frente a las observaciones discretas**, "
            "incluyendo la banda de dispersión $\\pm 2\\sigma$.\n"
            "2. **Serie temporal de los residuos**, contrastada contra las bandas de control $\\pm 2\\sigma$."
        )
    })

    # Celda 16: Código - Generación de datos y tabla de estadísticos
    cells.append({
        "type": "code",
        "source": (
            "sigma = 0.02\n"
            "seed = 42\n\n"
            "t_dias, I_clean = muestrear_dias_enteros(t_ref, I_ref)\n"
            "I_obs = agregar_ruido_gaussiano(I_clean, sigma=sigma, seed=seed)\n"
            "analisis = analizar_residuos(I_clean, I_obs, sigma=sigma)\n\n"
            "print(\"=\" * 65)\n"
            "print(\"ANÁLISIS ESTADÍSTICO DE RESIDUOS OBSERVACIONALES\")\n"
            "print(\"=\" * 65)\n"
            "print(f\"Número de observaciones discretas: {len(t_dias)}\")\n"
            "print(f\"Desviación estándar nominal (sigma): {sigma:.4f}\")\n"
            "print(f\"Media muestral de residuos:        {analisis['media']:+.5f}\")\n"
            "print(f\"Desviación estándar muestral (s):  {analisis['desviacion_std']:.5f}\")\n"
            "print(f\"Observaciones en banda +- 2*sigma: {analisis['porcentaje_2sigma']:.1f}% (esperado: ~95.4%)\")\n"
            "print(\"=\" * 65)\n"
        )
    })

    # Celda 17: Código - Figuras científicas
    cells.append({
        "type": "code",
        "source": (
            "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n\n"
            "# Panel 1: Curva de verdad de campo y observaciones ruidosas\n"
            "ax1 = axes[0]\n"
            "ax1.plot(t_ref, I_ref, 'k-', linewidth=2.0, label=\"Verdad de Campo I(t) [RK45]\", zorder=2)\n"
            "ax1.fill_between(\n"
            "    t_dias,\n"
            "    np.maximum(0.0, I_clean - 2 * sigma),\n"
            "    I_clean + 2 * sigma,\n"
            "    color=\"#9ecae1\",\n"
            "    alpha=0.45,\n"
            "    label=\"Banda de Incertidumbre +- 2*sigma (95.4%)\",\n"
            "    zorder=1\n"
            ")\n"
            "ax1.scatter(\n"
            "    t_dias, I_obs, color=\"#de2d26\", s=28, facecolors=\"none\", edgecolors=\"#de2d26\",\n"
            "    linewidths=1.5, label=\"Observaciones Clínicas Sintéticas\", zorder=3\n"
            ")\n"
            "ax1.set_title(\"Serie Temporal: Dinámica Continua vs Observaciones\")\n"
            "ax1.set_xlabel(\"Tiempo (días)\")\n"
            "ax1.set_ylabel(\"Proporción de Infectados I(t)\")\n"
            "ax1.legend(frameon=True, loc=\"upper right\")\n\n"
            "# Panel 2: Serie temporal de residuos con bandas de control\n"
            "ax2 = axes[1]\n"
            "residuos = analisis[\"residuos\"]\n"
            "ax2.axhline(0.0, color=\"black\", linestyle=\"-\", linewidth=1.0)\n"
            "ax2.axhline(+2 * sigma, color=\"#e6550d\", linestyle=\"--\", linewidth=1.5, label=\"+- 2*sigma (+- 0.04)\")\n"
            "ax2.axhline(-2 * sigma, color=\"#e6550d\", linestyle=\"--\", linewidth=1.5)\n"
            "markerline, stemlines, baseline = ax2.stem(t_dias, residuos, linefmt=\"b-\", markerfmt=\"bo\", basefmt=\"k-\")\n"
            "plt.setp(stemlines, linewidth=1.2, color=\"#3182bd\")\n"
            "plt.setp(markerline, markersize=4, color=\"#3182bd\")\n"
            "plt.setp(baseline, linewidth=0.8, color=\"gray\")\n"
            "ax2.set_title(\"Residuos Observacionales: e_k = I_obs - I_clean\")\n"
            "ax2.set_xlabel(\"Tiempo (días)\")\n"
            "ax2.set_ylabel(\"Residuo\")\n"
            "ax2.legend(frameon=True, loc=\"upper right\")\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    })

    # Celda 18: Markdown - Conclusiones y Síntesis del Laboratorio in Silico
    cells.append({
        "type": "markdown",
        "source": (
            "## 7. Conclusiones y Síntesis del Laboratorio in Silico\n\n"
            "1. **El Valor Metodológico del Ground Truth**:\n"
            "   Disponer de un entorno in silico donde la dinámica continua y los parámetros verdaderos $(\\beta^*, \\gamma^*)$ "
            "son conocidos con exactitud matemática es el estándar de oro para validar algoritmos de inferencia y estimación.\n\n"
            "2. **Caracterización de la Incertidumbre Observacional**:\n"
            "   El análisis de residuos confirmó que las observaciones sintéticas respetan las propiedades del error instrumental: "
            "media muestral nula (ausencia de sesgo sistemático), dispersión nominal acorde a $\\sigma = 0.02$ y más del $95\\%$ "
            "de los puntos dentro de la banda de control $\\pm 2\\sigma$.\n\n"
            "3. **Conexión con el Problema Inverso**:\n"
            "   Estas observaciones ruidosas $I^{\\text{obs}}_k$ simulan fielmente una serie temporal clínica real. "
            "En el **Laboratorio 4**, utilizaremos este conjunto de datos sintéticos como insumo para resolver el problema "
            "inverso: estimar $(\\beta, \\gamma)$ mediante optimización y evaluar el impacto del sesgo de discretización."
        )
    })

    return cells
