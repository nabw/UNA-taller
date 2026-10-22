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
            "1. Simular la verdad de campo continua del modelo SIR ($\\beta = 0.4$, $\\gamma = 0.1$) mediante RK45 adaptativo.\n"
            "2. Implementar el submuestreo temporal discreto en días enteros $t_k \\in \\{0, 1, 2, \\dots, 50\\}$.\n"
            "3. Modelar el proceso observacional incorporando ruido Gaussiano $\\mathcal{N}(0, \\sigma^2)$ con "
            "proyección a no-negatividad física: $I^{\\text{obs}} = \\max(0, I + \\varepsilon)$.\n"
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
            "from scipy.integrate import solve_ivp\n\n"
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

    # Celda 3: Markdown - Verdad de Campo Continua
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
            "Generamos la trayectoria continua evaluada en una malla densa con RK45 adaptativo."
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
            "días enteros: $t_k \\in \\{0, 1, 2, \\dots, 50\\}$. "
            "Interpolar la trayectoria continua de referencia sobre estos puntos discretos produce "
            "la serie temporal libre de ruido, denominada $I^{\\text{clean}}(t_k)$."
        )
    })

    # Celda 7: Código - Función muestrear_dias_enteros
    if is_solution:
        code_sample = (
            "def muestrear_dias_enteros(\n"
            "    t_ref: np.ndarray,\n"
            "    I_ref: np.ndarray,\n"
            "    t_dias: np.ndarray = None\n"
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
            "    t_dias: np.ndarray = None\n"
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

    # Celda 9: Markdown - Ruido Gaussiano y Proyección
    cells.append({
        "type": "markdown",
        "source": (
            "## 4. Proceso de Medición: Ruido Gaussiano y Proyección Física\n\n"
            "Para modelar el error observacional aditivo, perturbamos las mediciones con variables "
            "aleatorias normales independientes e idénticamente distribuidas:\n\n"
            "$$\\varepsilon_k \\sim \\mathcal{N}(0, \\sigma^2), \\quad \\sigma = 0.02$$\n\n"
            "Dado que $I(t)$ representa la proporción de la población infectada, una observación "
            "negativa es físicamente inadmisible. Imponemos la proyección no lineal sobre el conjunto de "
            "estados válidos mediante la operación de truncamiento inferior:\n\n"
            "$$I^{\\text{obs}}_k = \\max(0, I^{\\text{clean}}(t_k) + \\varepsilon_k)$$\n\n"
            "Para garantizar reproducibilidad científica estricta, fijamos la semilla del generador pseudoaleatorio."
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
            "Propiedades estadísticas a verificar:\n"
            "1. **Media muestral**: $\\bar{e} = \\frac{1}{M} \\sum_{k=1}^M e_k \\approx 0$, "
            "lo que confirma la ausencia de sesgo sistemático en el proceso de captura.\n"
            "2. **Desviación estándar muestral**: $s_e = \\sqrt{\\frac{1}{M-1} \\sum_{k=1}^M (e_k - \\bar{e})^2} \\approx \\sigma = 0.02$.\n"
            "3. **Banda de confianza $\\pm 2\\sigma$**: Para una distribución Gaussiana, aproximadamente el $95.4\\%$ "
            "de las observaciones deben residir dentro del intervalo $[-\\text{2}\\sigma, +\\text{2}\\sigma]$."
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
            "stats = analizar_residuos(I_clean, I_obs_test, sigma=0.02)\n"
            "assert isinstance(stats, dict), \"La función debe retornar un diccionario.\"\n"
            "assert \"media\" in stats and \"desviacion_std\" in stats and \"porcentaje_2sigma\" in stats\n"
            "assert abs(stats[\"media\"]) < 0.01, f\"Media residual excesiva: {stats['media']}\"\n"
            "assert 0.01 < stats[\"desviacion_std\"] < 0.03, f\"Desviación residual anormal: {stats['desviacion_std']}\"\n"
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

    # Celda 18: Markdown - Lectura complementaria
    cells.append({
        "type": "markdown",
        "source": (
            "## 7. Lectura Complementaria: Datos Discretos de Conteo\n\n"
            "En este laboratorio hemos modelado el error mediante ruido Gaussiano continuo y homocedástico. "
            "Sin embargo, en entornos hospitalarios donde se registran conteos enteros de personas (por ejemplo, admisiones diarias), "
            "la variabilidad del proceso de notificación es intrínsecamente discreta y heterocedástica: la varianza de los datos "
            "depende directamente de la magnitud media esperada de los casos.\n\n"
            "Para los estudiantes interesados en profundizar en el modelado estadístico de conteo mediante "
            "procesos Poissonianos y verosimilitud de Poisson, se recomienda consultar el documento conceptual de apoyo:\n\n"
            "[Lectura Complementaria: Modelado de Ruido de Conteo y Distribución de Poisson](../../appendix/ruido_conteo_poisson.md)"
        )
    })

    return cells
