"""
Constructor del Cuaderno 2: Modelo SIR, Conservación de Masa y Estabilidad Numérica.
Produce celdas para las versiones 'estudiante' y 'solucion'.
"""

from typing import List, Dict, Any


def build_nb2_cells(is_solution: bool) -> List[Dict[str, Any]]:
    cells = []

    # Celda 1: Markdown - Introducción al modelo SIR
    cells.append({
        "type": "markdown",
        "source": (
            "# Laboratorio 2: Sistema Epidemiologico SIR: Balance de Masa y Regimenes de Estabilidad\n\n"
            "## 1. Introduccion y Formulacion Diferencial\n\n"
            "El modelo SIR compartimental, introducido por Kermack y McKendrick en 1927, describe la "
            "propagacion de una enfermedad infecciosa en una poblacion cerrada y homogeneamente mezclada. "
            "Las variables de estado representan fracciones poblacionales normalizadas:\n"
            "- $S(t)$: fraccion de individuos susceptibles.\n"
            "- $I(t)$: fraccion de individuos infectados y transmisores.\n"
            "- $R(t)$: fraccion de individuos recuperados con inmunidad o removidos.\n\n"
            "El sistema de ecuaciones diferenciales ordinarias no lineales es:\n\n"
            "$$\\frac{dS}{dt} = -\\beta S I$$\n"
            "$$\\frac{dI}{dt} = \\beta S I - \\gamma I$$\n"
            "$$\\frac{dR}{dt} = \\gamma I$$\n\n"
            "donde:\n"
            "- $\\beta > 0$ es la tasa efectiva de transmision por contacto.\n"
            "- $\\gamma > 0$ es la tasa media de recuperacion (o remocion), con tiempo medio infeccioso $1/\\gamma$.\n"
            "- El numero reproductivo basico es $R_0 = \\frac{\\beta}{\\gamma}$. "
            "Si $R_0 > 1$ y $S(0) \\approx 1$, la enfermedad experimenta un crecimiento inicial y alcanza un pico epidemico.\n\n"
            "### Invariante Fundamental: Conservacion de Masa\n\n"
            "Sumando las tres derivadas del sistema:\n\n"
            "$$\\frac{d}{dt}(S + I + R) = -\\beta S I + (\\beta S I - \\gamma I) + \\gamma I = 0$$\n\n"
            "Integrando temporalmente con condicion inicial $S(0) + I(0) + R(0) = 1$:\n\n"
            "$$S(t) + I(t) + R(t) = 1, \\quad \\forall t \\ge 0$$\n\n"
            "Esta ley de conservacion fisica debe ser preservada rigurosamente por cualquier esquema numerico admisible.\n\n"
            "## Objetivos de Aprendizaje\n\n"
            "1. Implementar la solucion continua de referencia de alta precision mediante Runge-Kutta 4(5) adaptativo (RK45).\n"
            "2. Implementar e integrar el sistema mediante el metodo de Euler explicito.\n"
            "3. Implementar el metodo de Euler implicito resolviendo analiticamente la ecuacion cuadratica desacoplada.\n"
            "4. Implementar el esquema Implicito-Explicito (IMEX) y demostrar analiticamente la conservacion de masa.\n"
            "5. Comparar los esquemas en tres regimenes: paso fino ($\\Delta t = 0.5$), paso moderado ($\\Delta t = 2.5$) "
            "donde Euler explicito sobreestima el pico epidemico en mas de un 13%, y paso grande ($\\Delta t = 5.0$)."
        )
    })

    # Celda 2: Código - Configuración de entorno
    cells.append({
        "type": "code",
        "source": (
            "# Celda de configuracion y librerias cientificas\n"
            "import os\n"
            "os.environ.setdefault(\"MPLCONFIGDIR\", \"/tmp/matplotlib\")\n"
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
            "print(\"Librerias cientificas configuradas correctamente.\")"
        )
    })

    # Celda 3: Markdown - Solución de referencia RK45
    cells.append({
        "type": "markdown",
        "source": (
            "## 2. Solucion Continua de Referencia (RK45)\n\n"
            "Para evaluar la fidelidad de los integradores con paso fijo, establecemos una trayectoria "
            "de referencia de alta precision mediante integracion Runge-Kutta adaptativa con control local de error "
            "(tolerancias $rtol = 10^{-10}$, $atol = 10^{-12}$)."
        )
    })

    # Celda 4: Código - Función sir_referencia_rk45
    if is_solution:
        code_rk45 = (
            "def sir_referencia_rk45(\n"
            "    S0: float,\n"
            "    I0: float,\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            "    T: float,\n"
            "    dt_eval: float = 0.005\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Genera la solucion continua de referencia mediante RK45 adaptativo.\n"
            "    \"\"\"\n"
            "    R0_init = 1.0 - S0 - I0\n"
            "    y0 = np.array([S0, I0, R0_init], dtype=np.float64)\n\n"
            "    def sir_ode(t: float, y: np.ndarray) -> np.ndarray:\n"
            "        s, i, _ = y\n"
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
        code_rk45 = (
            "def sir_referencia_rk45(\n"
            "    S0: float,\n"
            "    I0: float,\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            "    T: float,\n"
            "    dt_eval: float = 0.005\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Genera la solucion continua de referencia mediante RK45 adaptativo.\n"
            "    \"\"\"\n"
            "    # TU CODIGO AQUI\n"
            "    ..."
        )
    cells.append({"type": "code", "source": code_rk45})

    # Celda 5: Código - Pruebas unitarias sir_referencia_rk45
    cells.append({
        "type": "code",
        "source": (
            "# Pruebas de validacion para sir_referencia_rk45\n"
            "t_ref, S_ref, I_ref, R_ref = sir_referencia_rk45(0.99, 0.01, 0.4, 0.1, 50.0)\n"
            "assert np.isclose(S_ref[0], 0.99), \"S(0) no coincide con S0.\"\n"
            "assert np.isclose(I_ref[0], 0.01), \"I(0) no coincide con I0.\"\n"
            "assert np.isclose(R_ref[0], 0.00), \"R(0) no coincide con 1 - S0 - I0.\"\n"
            "err_masa_ref = np.max(np.abs(S_ref + I_ref + R_ref - 1.0))\n"
            "assert err_masa_ref < 1e-6, f\"Error de conservacion en referencia: {err_masa_ref:.2e}\"\n"
            "assert np.all(S_ref >= 0.0) and np.all(I_ref >= 0.0) and np.all(R_ref >= 0.0), \"Variables deben ser positivas.\"\n"
            "print(f\"Validacion de sir_referencia_rk45: EXITOSA (error de conservacion: {err_masa_ref:.2e}).\")"
        )
    })

    # Celda 6: Markdown - Euler Explícito
    cells.append({
        "type": "markdown",
        "source": (
            "## 3. Metodo de Euler Explicito\n\n"
            "Discretizando temporalmente con paso $\\Delta t$:\n\n"
            "$$S_{n+1} = S_n - \\Delta t \\beta S_n I_n$$\n"
            "$$I_{n+1} = I_n + \\Delta t (\\beta S_n I_n - \\gamma I_n)$$\n"
            "$$R_{n+1} = R_n + \\Delta t \\gamma I_n$$\n\n"
            "Sumando miembro a miembro:\n\n"
            "$$S_{n+1} + I_{n+1} + R_{n+1} = S_n - \\Delta t \\beta S_n I_n + I_n + \\Delta t \\beta S_n I_n - \\Delta t \\gamma I_n + R_n + \\Delta t \\gamma I_n = S_n + I_n + R_n = 1$$\n\n"
            "Aunque la suma se conserva exactamente en la aritmetica discreta, si $\\Delta t$ es moderado o grande, "
            "el esquema sufre de errores acumulados en la tasa de contagio: en la fase ascendente, la tasa de nuevos infectados "
            "se calcula con la poblacion susceptible del inicio del paso ($S_n$), sobreestimando fuertemente la transmision "
            "y falseando la magnitud y el tiempo del pico."
        )
    })

    # Celda 7: Código - Función sir_euler_explicito
    if is_solution:
        code_sir_expl = (
            "def sir_euler_explicito(\n"
            "    S0: float,\n"
            "    I0: float,\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            "    dt: float,\n"
            "    T: float\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Resuelve el sistema SIR mediante Euler explicito.\n"
            "    \"\"\"\n"
            "    n_steps = int(np.round(T / dt))\n"
            "    t = np.linspace(0.0, n_steps * dt, n_steps + 1)\n"
            "    S = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    I = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    R = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    S[0] = float(S0)\n"
            "    I[0] = float(I0)\n"
            "    R[0] = float(1.0 - S0 - I0)\n\n"
            "    for n in range(n_steps):\n"
            "        sn = S[n]\n"
            "        in_val = I[n]\n"
            "        rn = R[n]\n"
            "        contagio = dt * beta * sn * in_val\n"
            "        recuperacion = dt * gamma * in_val\n"
            "        S[n + 1] = sn - contagio\n"
            "        I[n + 1] = in_val + contagio - recuperacion\n"
            "        R[n + 1] = rn + recuperacion\n\n"
            "    return t, S, I, R"
        )
    else:
        code_sir_expl = (
            "def sir_euler_explicito(\n"
            "    S0: float,\n"
            "    I0: float,\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            "    dt: float,\n"
            "    T: float\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Resuelve el sistema SIR mediante Euler explicito.\n"
            "    \"\"\"\n"
            "    # TU CODIGO AQUI\n"
            "    ..."
        )
    cells.append({"type": "code", "source": code_sir_expl})

    # Celda 8: Código - Pruebas unitarias sir_euler_explicito
    cells.append({
        "type": "code",
        "source": (
            "# Pruebas de validacion para sir_euler_explicito\n"
            "t_e, S_e, I_e, R_e = sir_euler_explicito(0.99, 0.01, 0.4, 0.1, 0.5, 10.0)\n"
            "assert len(t_e) == 21, f\"Longitud esperada: 21, obtenida: {len(t_e)}\"\n"
            "err_masa_e = np.max(np.abs(S_e + I_e + R_e - 1.0))\n"
            "assert err_masa_e < 1e-12, f\"Fallo conservacion discreta: {err_masa_e:.2e}\"\n"
            "assert np.isclose(S_e[0], 0.99) and np.isclose(I_e[0], 0.01), \"Condiciones iniciales no coinciden.\"\n"
            "print(f\"Validacion de sir_euler_explicito: EXITOSA (error conservacion: {err_masa_e:.2e}).\")"
        )
    })

    # Celda 9: Markdown - Euler Implícito
    cells.append({
        "type": "markdown",
        "source": (
            "## 4. Metodo de Euler Implicito (Backward Euler)\n\n"
            "Evaluando las tasas de cambio en el instante $t_{n+1}$:\n\n"
            "$$S_{n+1} = S_n - \\Delta t \\beta S_{n+1} I_{n+1} \\implies S_{n+1} = \\frac{S_n}{1 + \\Delta t \\beta I_{n+1}}$$\n"
            "$$I_{n+1} = I_n + \\Delta t (\\beta S_{n+1} I_{n+1} - \\gamma I_{n+1})$$\n"
            "$$R_{n+1} = R_n + \\Delta t \\gamma I_{n+1}$$\n\n"
            "Sustituyendo $S_{n+1}$ en la ecuacion de $I_{n+1}$:\n\n"
            "$$I_{n+1} - I_n - \\Delta t \\left( \\beta \\frac{S_n}{1 + \\Delta t \\beta I_{n+1}} I_{n+1} - \\gamma I_{n+1} \\right) = 0$$\n\n"
            "Multiplicando por $(1 + \\Delta t \\beta I_{n+1})$ y reordenando como ecuacion cuadratica $A I_{n+1}^2 + B I_{n+1} + C = 0$:\n\n"
            "$$A = \\Delta t \\beta (1 + \\Delta t \\gamma) > 0$$\n"
            "$$B = 1 + \\Delta t \\gamma - \\Delta t \\beta (S_n + I_n)$$\n"
            "$$C = -I_n \\le 0$$\n\n"
            "Puesto que $A > 0$ y $C \\le 0$, el discriminante $B^2 - 4 A C = B^2 + 4 A I_n \\ge B^2 \\ge 0$. "
            "Existe exactamente una raiz no-negativa, la cual se calcula de forma numericamente estable como:\n\n"
            "- Si $B \\ge 0$: $I_{n+1} = \\frac{2 I_n}{B + \\sqrt{B^2 + 4 A I_n}}$\n"
            "- Si $B < 0$: $I_{n+1} = \\frac{-B + \\sqrt{B^2 + 4 A I_n}}{2 A}$\n\n"
            "Una vez obtenido $I_{n+1}$, se actualizan de inmediato:\n\n"
            "$$S_{n+1} = \\frac{S_n}{1 + \\Delta t \\beta I_{n+1}}, \\quad R_{n+1} = R_n + \\Delta t \\gamma I_{n+1}$$"
        )
    })

    # Celda 10: Código - Función sir_euler_implicito
    if is_solution:
        code_sir_impl = (
            "def sir_euler_implicito(\n"
            "    S0: float,\n"
            "    I0: float,\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            "    dt: float,\n"
            "    T: float\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Resuelve el sistema SIR mediante Euler implicito utilizando la raiz cuadratica analitica.\n"
            "    \"\"\"\n"
            "    n_steps = int(np.round(T / dt))\n"
            "    t = np.linspace(0.0, n_steps * dt, n_steps + 1)\n"
            "    S = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    I = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    R = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    S[0] = float(S0)\n"
            "    I[0] = float(I0)\n"
            "    R[0] = float(1.0 - S0 - I0)\n\n"
            "    for n in range(n_steps):\n"
            "        sn = S[n]\n"
            "        in_val = I[n]\n"
            "        rn = R[n]\n\n"
            "        A = dt * beta * (1.0 + dt * gamma)\n"
            "        B = 1.0 + dt * gamma - dt * beta * (sn + in_val)\n"
            "        disc = max(0.0, B * B + 4.0 * A * in_val)\n\n"
            "        if B >= 0.0:\n"
            "            i_next = 2.0 * in_val / (B + np.sqrt(disc))\n"
            "        else:\n"
            "            i_next = (-B + np.sqrt(disc)) / (2.0 * A)\n\n"
            "        s_next = sn / (1.0 + dt * beta * i_next)\n"
            "        r_next = rn + dt * gamma * i_next\n\n"
            "        S[n + 1] = s_next\n"
            "        I[n + 1] = i_next\n"
            "        R[n + 1] = r_next\n\n"
            "    return t, S, I, R"
        )
    else:
        code_sir_impl = (
            "def sir_euler_implicito(\n"
            "    S0: float,\n"
            "    I0: float,\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            "    dt: float,\n"
            "    T: float\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Resuelve el sistema SIR mediante Euler implicito utilizando la raiz cuadratica analitica.\n"
            "    \"\"\"\n"
            "    # TU CODIGO AQUI\n"
            "    ..."
        )
    cells.append({"type": "code", "source": code_sir_impl})

    # Celda 11: Código - Pruebas unitarias sir_euler_implicito
    cells.append({
        "type": "code",
        "source": (
            "# Pruebas de validacion para sir_euler_implicito\n"
            "t_i, S_i, I_i, R_i = sir_euler_implicito(0.99, 0.01, 0.4, 0.1, 0.5, 10.0)\n"
            "assert len(t_i) == 21, f\"Longitud esperada: 21, obtenida: {len(t_i)}\"\n"
            "assert np.all(S_i >= 0.0) and np.all(I_i >= 0.0) and np.all(R_i >= 0.0), \"No-negatividad violada.\"\n"
            "err_masa_i = np.max(np.abs(S_i + I_i + R_i - 1.0))\n"
            "assert err_masa_i < 1e-12, f\"Fallo conservacion discreta: {err_masa_i:.2e}\"\n"
            "print(f\"Validacion de sir_euler_implicito: EXITOSA (error conservacion: {err_masa_i:.2e}).\")"
        )
    })

    # Celda 12: Markdown - Esquema IMEX Conservativo
    cells.append({
        "type": "markdown",
        "source": (
            "## 5. Esquema Implicito-Explicito (IMEX) Conservativo\n\n"
            "Para combinar la velocidad computacional de los metodos explicitos con la estabilidad "
            "asintotica de los metodos implicitos, disenamos un esquema IMEX estructurado:\n\n"
            "$$S_{n+1} = S_n - \\Delta t \\beta S_n I_n$$\n"
            "$$I_{n+1} = \\frac{I_n + \\Delta t \\beta S_n I_n}{1 + \\Delta t \\gamma}$$\n"
            "$$R_{n+1} = R_n + \\Delta t \\gamma I_{n+1}$$\n\n"
            "### Demostracion de Conservacion Incondicional de Masa\n\n"
            "Sumando los tres estados en el paso $n+1$:\n\n"
            "$$S_{n+1} + I_{n+1} + R_{n+1} = (S_n - \\Delta t \\beta S_n I_n) + I_{n+1} + (R_n + \\Delta t \\gamma I_{n+1})$$\n"
            "$$= S_n - \\Delta t \\beta S_n I_n + R_n + I_{n+1} (1 + \\Delta t \\gamma)$$\n\n"
            "De la ecuacion de $I_{n+1}$, despejamos el numerador: $I_{n+1} (1 + \\Delta t \\gamma) = I_n + \\Delta t \\beta S_n I_n$. Sustituyendo:\n\n"
            "$$= S_n - \\Delta t \\beta S_n I_n + R_n + I_n + \\Delta t \\beta S_n I_n = S_n + I_n + R_n = 1$$\n\n"
            "La conservacion de masa es **exacta e incondicional** para cualquier tamano de paso $\\Delta t > 0$, "
            "y cada estado se calcula mediante operaciones aritmeticas elementales directas."
        )
    })

    # Celda 13: Código - Función sir_imex
    if is_solution:
        code_sir_imex = (
            "def sir_imex(\n"
            "    S0: float,\n"
            "    I0: float,\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            "    dt: float,\n"
            "    T: float\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Resuelve el sistema SIR mediante el esquema IMEX conservativo.\n"
            "    \"\"\"\n"
            "    n_steps = int(np.round(T / dt))\n"
            "    t = np.linspace(0.0, n_steps * dt, n_steps + 1)\n"
            "    S = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    I = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    R = np.zeros(n_steps + 1, dtype=np.float64)\n"
            "    S[0] = float(S0)\n"
            "    I[0] = float(I0)\n"
            "    R[0] = float(1.0 - S0 - I0)\n\n"
            "    for n in range(n_steps):\n"
            "        sn = S[n]\n"
            "        in_val = I[n]\n"
            "        rn = R[n]\n\n"
            "        contagio = dt * beta * sn * in_val\n"
            "        S[n + 1] = sn - contagio\n"
            "        I[n + 1] = (in_val + contagio) / (1.0 + dt * gamma)\n"
            "        R[n + 1] = rn + dt * gamma * I[n + 1]\n\n"
            "    return t, S, I, R"
        )
    else:
        code_sir_imex = (
            "def sir_imex(\n"
            "    S0: float,\n"
            "    I0: float,\n"
            "    beta: float,\n"
            "    gamma: float,\n"
            "    dt: float,\n"
            "    T: float\n"
            ") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:\n"
            "    \"\"\"\n"
            "    Resuelve el sistema SIR mediante el esquema IMEX conservativo.\n"
            "    \"\"\"\n"
            "    # TU CODIGO AQUI\n"
            "    ..."
        )
    cells.append({"type": "code", "source": code_sir_imex})

    # Celda 14: Código - Pruebas unitarias sir_imex
    cells.append({
        "type": "code",
        "source": (
            "# Pruebas de validacion para sir_imex\n"
            "t_m, S_m, I_m, R_m = sir_imex(0.99, 0.01, 0.4, 0.1, 0.5, 10.0)\n"
            "assert len(t_m) == 21, f\"Longitud esperada: 21, obtenida: {len(t_m)}\"\n"
            "err_masa_m = np.max(np.abs(S_m + I_m + R_m - 1.0))\n"
            "assert err_masa_m < 1e-12, f\"Fallo conservacion discreta: {err_masa_m:.2e}\"\n"
            "print(f\"Validacion de sir_imex: EXITOSA (error conservacion: {err_masa_m:.2e}).\")"
        )
    })

    # Celda 15: Markdown - Comparación en tres regímenes temporales
    cells.append({
        "type": "markdown",
        "source": (
            "## 6. Comparacion Pedagogica en Tres Regimenes Temporales\n\n"
            "Fijamos los parametros epidemiologicos de referencia:\n"
            "- Tasa de transmision: $\\beta = 0.4$\n"
            "- Tasa de recuperacion: $\\gamma = 0.1$\n"
            "- Numero reproductivo basico: $R_0 = 4.0$\n"
            "- Condiciones iniciales: $S(0) = 0.99$, $I(0) = 0.01$, $R(0) = 0.00$\n"
            "- Horizonte: $T = 50.0$ dias\n\n"
            "Comparamos los integradores bajo tres pasos temporales:\n"
            "1. **Paso Fino ($\\Delta t = 0.5$ dias)**: Alta fidelidad frente a la trayectoria continua.\n"
            "2. **Paso Moderado ($\\Delta t = 2.5$ dias)**: Regimen clave. Euler explicito sobreestima el pico "
            "epidemico en mas de un 13% y adelanta la ola epidemica, demostrando el sesgo intrinseco del metodo "
            "sin recurrir a pasos temporales absurdamente grandes.\n"
            "3. **Paso Grande ($\\Delta t = 5.0$ dias)**: Regimen de inestabilidad numerica severa."
        )
    })

    # Celda 16: Código - Ejecución y tabla de métricas del pico epidémico
    cells.append({
        "type": "code",
        "source": (
            "beta = 0.4\n"
            "gamma = 0.1\n"
            "S0 = 0.99\n"
            "I0 = 0.01\n"
            "T = 50.0\n\n"
            "# Solucion de referencia continua\n"
            "t_ref, S_ref, I_ref, R_ref = sir_referencia_rk45(S0, I0, beta, gamma, T, dt_eval=0.005)\n"
            "idx_peak_ref = np.argmax(I_ref)\n"
            "I_peak_ref = float(I_ref[idx_peak_ref])\n"
            "t_peak_ref = float(t_ref[idx_peak_ref])\n\n"
            "print(\"=\" * 85)\n"
            "print(f\"REFERENCIA CONTINUA (RK45): Pico I_max = {I_peak_ref:.4f} en t = {t_peak_ref:.2f} dias\")\n"
            "print(\"=\" * 85)\n"
            "print(f\"{'dt (dias)':>9s} | {'Metodo':>16s} | {'I_max':>8s} | {'Error I_max (%)':>16s} | {'t_pico':>8s} | {'Max |Sum-1|':>12s}\")\n"
            "print(\"-\" * 85)\n\n"
            "dt_regimes = [0.5, 2.5, 5.0]\n"
            "sim_data = {}\n\n"
            "for dt in dt_regimes:\n"
            "    sim_data[dt] = {}\n"
            "    t_exp, S_exp, I_exp, R_exp = sir_euler_explicito(S0, I0, beta, gamma, dt, T)\n"
            "    t_imp, S_imp, I_imp, R_imp = sir_euler_implicito(S0, I0, beta, gamma, dt, T)\n"
            "    t_imx, S_imx, I_imx, R_imx = sir_imex(S0, I0, beta, gamma, dt, T)\n\n"
            "    for name, t_arr, s_arr, i_arr, r_arr in [\n"
            "        (\"Euler Explicito\", t_exp, S_exp, I_exp, R_exp),\n"
            "        (\"Euler Implicito\", t_imp, S_imp, I_imp, R_imp),\n"
            "        (\"IMEX Conservativo\", t_imx, S_imx, I_imx, R_imx)\n"
            "    ]:\n"
            "        idx_p = np.argmax(i_arr)\n"
            "        i_pk = float(i_arr[idx_p])\n"
            "        t_pk = float(t_arr[idx_p])\n"
            "        err_rel = ((i_pk - I_peak_ref) / I_peak_ref) * 100.0\n"
            "        err_cons = float(np.max(np.abs(s_arr + i_arr + r_arr - 1.0)))\n"
            "        sim_data[dt][name] = (t_arr, i_arr, i_pk, t_pk, err_rel)\n"
            "        print(f\"{dt:9.2f} | {name:>16s} | {i_pk:8.4f} | {err_rel:+15.2f}% | {t_pk:8.1f} | {err_cons:12.2e}\")\n"
            "    print(\"-\" * 85)\n"
        )
    })

    # Celda 17: Código - Generación de figuras comparativas (3 paneles)
    cells.append({
        "type": "code",
        "source": (
            "fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)\n\n"
            "for idx, dt in enumerate(dt_regimes):\n"
            "    ax = axes[idx]\n"
            "    ax.plot(t_ref, I_ref, 'k-', label=\"Referencia RK45\", zorder=1)\n"
            "    for name, color, style in [\n"
            "        (\"Euler Explicito\", \"#d95f02\", \"o--\"),\n"
            "        (\"Euler Implicito\", \"#7570b3\", \"s--\"),\n"
            "        (\"IMEX Conservativo\", \"#1b9e77\", \"^--\")\n"
            "    ]:\n"
            "        t_arr, i_arr, i_pk, t_pk, err_rel = sim_data[dt][name]\n"
            "        ax.plot(t_arr, i_arr, style, color=color, markersize=4, label=f\"{name} ({err_rel:+.1f}%)\", alpha=0.85, zorder=2)\n"
            "    ax.set_title(f\"Regimen: dt = {dt:.1f} dias\")\n"
            "    ax.set_xlabel(\"Tiempo (dias)\")\n"
            "    if idx == 0:\n"
            "        ax.set_ylabel(\"Fraccion de Infectados I(t)\")\n"
            "    ax.legend(frameon=True, fontsize=9, loc=\"upper right\")\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    })

    # Celda 18: Markdown - Discusión y Conclusiones
    cells.append({
        "type": "markdown",
        "source": (
            "## 7. Conclusiones y Sintesis Epidemiologica\n\n"
            "1. **Defecto Sistematico de Euler Explicito en Pasos Moderados ($\\Delta t = 2.5$)**:\n"
            "   Incluso sin entrar en inestabilidades caoticas, a $\\Delta t = 2.5$ dias Euler explicito "
            "sobreestima el pico epidemico maximo en $+13.60\\%$ ($I_{\\max} = 0.4612$ frente a $0.4059$ de RK45) "
            "y desfasa la llegada del pico a $t = 22.5$ dias. Esto evidencia que un paso temporal moderado ya falsea "
            "completamente las proyecciones de camas hospitalarias en un modelo epidemiologico.\n\n"
            "2. **Comportamiento Amortiguador de los Metodos Implicito e IMEX**:\n"
            "   Euler implicito e IMEX subestiman levemente el pico en pasos moderados, actuando como disipadores "
            "numericos estables que garantizan la estabilidad asintotica sin inducir sobrepasos no fisicos.\n\n"
            "3. **Conservacion Estricta de Masa**:\n"
            "   Los tres esquemas discretos conservan la suma poblacional $S + I + R = 1.0$ al nivel de precision "
            "de maquina ($< 10^{-15}$). Sin embargo, la conservacion de masa por si sola no previene la distorsion "
            "dinamica de la ola epidemica."
        )
    })

    return cells
