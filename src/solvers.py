"""
Módulo de resolución numérica de ecuaciones diferenciales ordinarias (EDOs).

Implementa esquemas explícitos, implícitos e implícito-explícitos (IMEX) para:
1. La ecuación logística 1D: d x / dt = C * x * (1 - x).
2. El sistema epidemiológico SIR:
       dS/dt = - beta * S * I
       dI/dt =   beta * S * I - gamma * I
       dR/dt =   gamma * I

Universidad Nacional (UNA), Costa Rica.
Carrera de Ingeniería Matemática en Ciencia de Datos.
"""

from typing import Tuple, Optional
import numpy as np
from scipy.optimize import root


# ==============================================================================
# 1. MODELO 1D: ECUACIÓN LOGÍSTICA
# ==============================================================================

def solve_logistic_exact(
    x0: float,
    C: float,
    t: np.ndarray
) -> np.ndarray:
    """
    Solución analítica exacta de la ecuación logística dot{x} = C * x * (1 - x).

    Fórmula analítica cerrada:
        x(t) = x0 * exp(C * t) / (1 - x0 + x0 * exp(C * t))
             = 1 / (1 + ((1 - x0) / x0) * exp(-C * t))

    Parámetros
    ----------
    x0 : float
        Condición inicial x(0), típicamente en (0, 1).
    C : float
        Tasa intrínseca de crecimiento (C > 0).
    t : np.ndarray
        Vector 1D de tiempos de evaluación.

    Retorna
    -------
    np.ndarray
        Valores exactos de x(t).
    """
    t = np.asarray(t, dtype=np.float64)
    if x0 <= 0.0:
        return np.zeros_like(t)
    # Forma numéricamente estable usando exp(-C * t)
    c_ratio = (1.0 - x0) / x0
    return 1.0 / (1.0 + c_ratio * np.exp(-C * t))


def solve_logistic_explicit(
    x0: float,
    C: float,
    dt: float,
    T: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resuelve la ecuación logística mediante el método de Euler explícito:
        x_{n+1} = x_n + dt * C * x_n * (1 - x_n)

    Nota pedagógica de estabilidad:
    Para dt * C > 2, el método explícito sufre de oscilaciones espurias y
    para dt * C > 3 puede exhibir bifurcaciones caóticas y pérdida de positividad,
    un artefacto puramente numérico ajeno a la EDO continua.

    Parámetros
    ----------
    x0 : float
        Condición inicial x(0).
    C : float
        Parámetro de crecimiento.
    dt : float
        Tamaño del paso temporal (Delta t).
    T : float
        Tiempo final de integración.

    Retorna
    -------
    t : np.ndarray
        Vector 1D de tiempos [0, dt, 2*dt, ..., N*dt].
    x : np.ndarray
        Vector 1D con la solución numérica x(t).
    """
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.zeros(n_steps + 1, dtype=np.float64)
    x[0] = float(x0)

    for n in range(n_steps):
        xn = x[n]
        x[n + 1] = xn + dt * C * xn * (1.0 - xn)

    return t, x


def solve_logistic_implicit(
    x0: float,
    C: float,
    dt: float,
    T: float,
    use_root: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resuelve la ecuación logística mediante el método de Euler implícito:
        x_{n+1} = x_n + dt * C * x_{n+1} * (1 - x_{n+1})

    Reescribiendo como ecuación algebraica cuadrática en cada paso:
        dt * C * x_{n+1}^2 + (1 - dt * C) * x_{n+1} - x_n = 0

    Identificando a = dt * C, b = 1 - dt * C, c = -x_n:
    La raíz físicamente admisible (positiva para x_n >= 0) es:
        x_{n+1} = (-b + sqrt(b^2 + 4 * a * x_n)) / (2 * a)
    o de forma numéricamente estable sin cancelación sustractiva:
        x_{n+1} = 2 * x_n / (b + sqrt(b^2 + 4 * a * x_n))

    Parámetros
    ----------
    x0 : float
        Condición inicial x(0).
    C : float
        Parámetro de crecimiento.
    dt : float
        Tamaño del paso temporal (Delta t).
    T : float
        Tiempo final de integración.
    use_root : bool, opcional
        Si es True, resuelve la ecuación no lineal en cada paso con
        `scipy.optimize.root` en lugar de la fórmula cuadrática cerrada.
        Por defecto es False (usa fórmula cuadrática analítica exacta).

    Retorna
    -------
    t : np.ndarray
        Vector 1D de tiempos.
    x : np.ndarray
        Vector 1D con la solución numérica x(t).
    """
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.zeros(n_steps + 1, dtype=np.float64)
    x[0] = float(x0)

    a = dt * C
    b = 1.0 - dt * C

    for n in range(n_steps):
        xn = x[n]
        if use_root:
            # Alternativa pedagógica usando solver no lineal de SciPy
            def residual(x_next: np.ndarray) -> np.ndarray:
                return x_next - xn - dt * C * x_next * (1.0 - x_next)

            sol = root(residual, np.array([xn]), method="hybr")
            x[n + 1] = float(sol.x[0])
        else:
            # Solución analítica cuadrática exacta
            if a == 0.0:
                x[n + 1] = xn
            else:
                disc = b * b + 4.0 * a * xn
                # Raíz numéricamente estable (evita cancelación sustractiva según el signo de b)
                if b >= 0.0:
                    x[n + 1] = 2.0 * xn / (b + np.sqrt(np.maximum(0.0, disc)))
                else:
                    x[n + 1] = (-b + np.sqrt(np.maximum(0.0, disc))) / (2.0 * a)

    return t, x


def solve_logistic_imex(
    x0: float,
    C: float,
    dt: float,
    T: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resuelve la ecuación logística mediante un esquema Implícito-Explícito (IMEX)
    no estándar linealizado:
        (x_{n+1} - x_n) / dt = C * x_n - C * x_n * x_{n+1}

    Despejando analíticamente x_{n+1}:
        x_{n+1} * (1 + dt * C * x_n) = x_n * (1 + dt * C)
        x_{n+1} = x_n * (1 + dt * C) / (1 + dt * C * x_n)

    Propiedades matemáticas:
    1. Linealidad implícita: no requiere resolver sistemas no lineales iterativos.
    2. Positividad incondicional: si x_n >= 0, entonces x_{n+1} >= 0 para cualquier dt > 0.
    3. Conservación del atractor: el punto fijo x* = 1 se preserva algebraicamente.

    Parámetros
    ----------
    x0 : float
        Condición inicial x(0).
    C : float
        Parámetro de crecimiento.
    dt : float
        Tamaño del paso temporal (Delta t).
    T : float
        Tiempo final de integración.

    Retorna
    -------
    t : np.ndarray
        Vector 1D de tiempos.
    x : np.ndarray
        Vector 1D con la solución numérica x(t).
    """
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.zeros(n_steps + 1, dtype=np.float64)
    x[0] = float(x0)

    factor_num = 1.0 + dt * C
    for n in range(n_steps):
        xn = x[n]
        x[n + 1] = (xn * factor_num) / (1.0 + dt * C * xn)

    return t, x


# ==============================================================================
# 2. SISTEMA EPIDEMIOLÓGICO SIR
# ==============================================================================

def solve_sir_explicit(
    S0: float,
    I0: float,
    beta: float,
    gamma: float,
    dt: float,
    T: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resuelve el sistema epidemiológico SIR mediante el método de Euler explícito:
        S_{n+1} = S_n - dt * beta * S_n * I_n
        I_{n+1} = I_n + dt * (beta * S_n * I_n - gamma * I_n)
        R_{n+1} = R_n + dt * gamma * I_n

    Condición de masa total:
        R_0 = 1 - S_0 - I_0.
        S_n + I_n + R_n = 1.0 se conserva algebraicamente, pero si dt es excesivo
        S_n o I_n pueden volverse negativos o presentar oscilaciones numéricas inestables.

    Parámetros
    ----------
    S0 : float
        Fracción inicial de susceptibles S(0).
    I0 : float
        Fracción inicial de infectados I(0).
    beta : float
        Tasa de transmisión / contacto efectivo (beta > 0).
    gamma : float
        Tasa de recuperación (gamma > 0).
    dt : float
        Tamaño del paso temporal (Delta t).
    T : float
        Tiempo final de simulación.

    Retorna
    -------
    t : np.ndarray
        Vector 1D de tiempos.
    S : np.ndarray
        Vector 1D de la proporción de susceptibles.
    I : np.ndarray
        Vector 1D de la proporción de infectados.
    R : np.ndarray
        Vector 1D de la proporción de recuperados.
    """
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    S = np.zeros(n_steps + 1, dtype=np.float64)
    I = np.zeros(n_steps + 1, dtype=np.float64)
    R = np.zeros(n_steps + 1, dtype=np.float64)

    S[0] = float(S0)
    I[0] = float(I0)
    R[0] = float(1.0 - S0 - I0)

    for n in range(n_steps):
        sn = S[n]
        in_val = I[n]
        rn = R[n]

        infection = dt * beta * sn * in_val
        recovery = dt * gamma * in_val

        S[n + 1] = sn - infection
        I[n + 1] = in_val + infection - recovery
        R[n + 1] = rn + recovery

    return t, S, I, R


def solve_sir_implicit(
    S0: float,
    I0: float,
    beta: float,
    gamma: float,
    dt: float,
    T: float,
    use_root_solver: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resuelve el sistema epidemiológico SIR mediante el método de Euler implícito
    (Backward Euler).

    Ecuaciones algebraicas no lineales a satisfacer en cada paso n -> n+1:
        S_{n+1} - S_n + dt * beta * S_{n+1} * I_{n+1} = 0
        I_{n+1} - I_n - dt * (beta * S_{n+1} * I_{n+1} - gamma * I_{n+1}) = 0
        R_{n+1} - R_n - dt * gamma * I_{n+1} = 0

    Despejando S_{n+1} = S_n / (1 + dt * beta * I_{n+1}) y sustituyendo en la
    ecuación de I_{n+1}, se obtiene una ecuación cuadrática exacta A*I_{n+1}^2 + B*I_{n+1} - I_n = 0,
    donde:
        A = dt * beta * (1 + dt * gamma) > 0
        B = 1 + dt * gamma - dt * beta * (S_n + I_n)
        C = -I_n < 0

    Como A > 0 y C < 0, existe exactamente una única raíz física real estrictamente positiva,
    garantizando incondicionalmente no-negatividad y estabilidad sin oscilaciones espurias.

    Parámetros
    ----------
    S0 : float
        Fracción inicial de susceptibles S(0).
    I0 : float
        Fracción inicial de infectados I(0).
    beta : float
        Tasa de transmisión / contacto efectivo.
    gamma : float
        Tasa de recuperación.
    dt : float
        Tamaño del paso temporal (Delta t).
    T : float
        Tiempo final de simulación.
    use_root_solver : bool, opcional
        Si es True, utiliza `scipy.optimize.root` numérico en lugar de la raíz cuadrática exacta.

    Retorna
    -------
    t : np.ndarray
        Vector 1D de tiempos.
    S : np.ndarray
        Vector 1D de susceptibles.
    I : np.ndarray
        Vector 1D de infectados.
    R : np.ndarray
        Vector 1D de recuperados.
    """
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    S = np.zeros(n_steps + 1, dtype=np.float64)
    I = np.zeros(n_steps + 1, dtype=np.float64)
    R = np.zeros(n_steps + 1, dtype=np.float64)

    S[0] = float(S0)
    I[0] = float(I0)
    R[0] = float(1.0 - S0 - I0)

    if not use_root_solver:
        for n in range(n_steps):
            sn = S[n]
            in_val = I[n]
            rn = R[n]

            A = dt * beta * (1.0 + dt * gamma)
            B = 1.0 + dt * gamma - dt * beta * (sn + in_val)
            disc = max(0.0, B**2 + 4.0 * A * in_val)

            # Fórmula cuadrática con estabilidad numérica frente a cancelación sustractiva
            if B >= 0:
                i_next = 2.0 * in_val / (B + np.sqrt(disc))
            else:
                i_next = (-B + np.sqrt(disc)) / (2.0 * A)

            s_next = sn / (1.0 + dt * beta * i_next)
            r_next = rn + dt * gamma * i_next

            S[n + 1] = s_next
            I[n + 1] = i_next
            R[n + 1] = r_next
    else:
        # Modo con scipy.optimize.root con restricción de inicio
        y_prev = np.array([S[0], I[0], R[0]], dtype=np.float64)

        def residual(y: np.ndarray) -> np.ndarray:
            s, i, r = max(0.0, y[0]), max(0.0, y[1]), max(0.0, y[2])
            return np.array([
                y[0] - y_prev[0] + dt * beta * s * i,
                y[1] - y_prev[1] - dt * (beta * s * i - gamma * i),
                y[2] - y_prev[2] - dt * gamma * i
            ], dtype=np.float64)

        for n in range(n_steps):
            y_prev[0] = S[n]
            y_prev[1] = I[n]
            y_prev[2] = R[n]
            # Inicialización cercana para evitar raíces negativas
            guess = np.array([S[n], max(1e-6, I[n] * 0.95), R[n]], dtype=np.float64)
            sol = root(residual, guess, method="hybr")
            S[n + 1] = max(0.0, sol.x[0])
            I[n + 1] = max(0.0, sol.x[1])
            R[n + 1] = max(0.0, sol.x[2])

    return t, S, I, R


def solve_sir_imex(
    S0: float,
    I0: float,
    beta: float,
    gamma: float,
    dt: float,
    T: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Resuelve el sistema epidemiológico SIR mediante el esquema Implícito-Explícito (IMEX):
        S_{n+1} = S_n - dt * beta * S_n * I_n
        I_{n+1} = (I_n + dt * beta * S_n * I_n) / (1 + dt * gamma)
        R_{n+1} = R_n + dt * gamma * I_{n+1}

    Propiedad fundamental de conservación de masa:
    Sumando las tres expresiones paso a paso:
        S_{n+1} + I_{n+1} + R_{n+1}
          = (S_n - dt * beta * S_n * I_n) + I_{n+1} + (R_n + dt * gamma * I_{n+1})
          = S_n - dt * beta * S_n * I_n + R_n + I_{n+1} * (1 + dt * gamma)
    Como I_{n+1} * (1 + dt * gamma) = I_n + dt * beta * S_n * I_n:
          = S_n - dt * beta * S_n * I_n + R_n + I_n + dt * beta * S_n * I_n
          = S_n + I_n + R_n = 1.0.
    Por lo tanto, la masa total se conserva de forma EXACTA e incondicional en cada paso.

    Parámetros
    ----------
    S0 : float
        Fracción inicial de susceptibles S(0).
    I0 : float
        Fracción inicial de infectados I(0).
    beta : float
        Tasa de transmisión.
    gamma : float
        Tasa de recuperación.
    dt : float
        Tamaño del paso temporal (Delta t).
    T : float
        Tiempo final de simulación.

    Retorna
    -------
    t : np.ndarray
        Vector 1D de tiempos.
    S : np.ndarray
        Vector 1D de susceptibles.
    I : np.ndarray
        Vector 1D de infectados.
    R : np.ndarray
        Vector 1D de recuperados.
    """
    n_steps = int(np.round(T / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    S = np.zeros(n_steps + 1, dtype=np.float64)
    I = np.zeros(n_steps + 1, dtype=np.float64)
    R = np.zeros(n_steps + 1, dtype=np.float64)

    S[0] = float(S0)
    I[0] = float(I0)
    R[0] = float(1.0 - S0 - I0)

    denom_gamma = 1.0 + dt * gamma

    for n in range(n_steps):
        sn = S[n]
        in_val = I[n]
        rn = R[n]

        infection = dt * beta * sn * in_val
        s_next = sn - infection
        i_next = (in_val + infection) / denom_gamma
        r_next = rn + dt * gamma * i_next

        S[n + 1] = s_next
        I[n + 1] = i_next
        R[n + 1] = r_next

    return t, S, I, R
