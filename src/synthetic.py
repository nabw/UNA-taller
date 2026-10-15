"""
Módulo de generación de datos sintéticos y ruido para el modelo SIR.

Permite simular la trayectoria epidemiológica "verdadera" de referencia (Ground Truth)
con algoritmos de alta precisión (RK45) y perturbar las observaciones mediante
ruido Gaussiano y ruido Poisson para modelar incertidumbre observacional real.

Universidad Nacional (UNA), Costa Rica.
Carrera de Ingeniería Matemática en Ciencia de Datos.
"""

from typing import Tuple, Optional, Union
import numpy as np
from scipy.integrate import solve_ivp


def generate_ground_truth(
    beta: float = 0.4,
    gamma: float = 0.1,
    S0: float = 0.99,
    I0: float = 0.01,
    T: float = 50.0,
    dt_ref: float = 0.001
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera la trayectoria epidemiológica de referencia (Ground Truth) del modelo SIR
    mediante integración numérica adaptativa de alta precisión (Runge-Kutta RK45).

    El sistema dinámico continuo de referencia es:
        dS/dt = - beta * S * I
        dI/dt =   beta * S * I - gamma * I
        dR/dt =   gamma * I
    con R(0) = 1 - S(0) - I(0).

    Parámetros
    ----------
    beta : float, opcional
        Tasa de transmisión efectiva (por defecto 0.4).
    gamma : float, opcional
        Tasa de recuperación media (por defecto 0.1).
    S0 : float, opcional
        Fracción inicial de susceptibles (por defecto 0.99).
    I0 : float, opcional
        Fracción inicial de infectados (por defecto 0.01).
    T : float, opcional
        Tiempo final del horizonte de simulación en días (por defecto 50.0).
    dt_ref : float, opcional
        Paso temporal para la malla fina de evaluación (por defecto 0.001).

    Retorna
    -------
    t_ref : np.ndarray
        Vector 1D de tiempos de referencia de alta resolución.
    S_ref : np.ndarray
        Vector 1D de la proporción de susceptibles de referencia.
    I_ref : np.ndarray
        Vector 1D de la proporción de infectados de referencia.
    R_ref : np.ndarray
        Vector 1D de la proporción de recuperados de referencia.
    """
    R0 = 1.0 - S0 - I0
    y0 = np.array([S0, I0, R0], dtype=np.float64)

    def sir_ode(t: float, y: np.ndarray) -> np.ndarray:
        s, i, _ = y
        ds = -beta * s * i
        di = beta * s * i - gamma * i
        dr = gamma * i
        return np.array([ds, di, dr], dtype=np.float64)

    n_points = int(np.round(T / dt_ref)) + 1
    t_eval = np.linspace(0.0, T, n_points)

    sol = solve_ivp(
        sir_ode,
        t_span=(0.0, T),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12
    )

    t_ref = sol.t
    S_ref = sol.y[0]
    I_ref = sol.y[1]
    R_ref = sol.y[2]

    return t_ref, S_ref, I_ref, R_ref


def sample_observations(
    t_ref: np.ndarray,
    I_ref: np.ndarray,
    sample_times: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Submuestrea e interpola la curva continua de infectados I(t) en tiempos discretos
    de observación clínica (típicamente días enteros {0, 1, 2, ..., 50}).

    Parámetros
    ----------
    t_ref : np.ndarray
        Vector 1D de tiempos de la simulación de referencia.
    I_ref : np.ndarray
        Vector 1D de infectados de la simulación de referencia.
    sample_times : np.ndarray, opcional
        Instantes de tiempo discretos en los que se registran observaciones.
        Si es None, se utiliza por defecto np.arange(0.0, 51.0, 1.0).

    Retorna
    -------
    I_obs : np.ndarray
        Vector 1D con los valores de I interpolados en los tiempos de observación.
    """
    if sample_times is None:
        sample_times = np.arange(0.0, 51.0, 1.0)
    else:
        sample_times = np.asarray(sample_times, dtype=np.float64)

    t_ref = np.asarray(t_ref, dtype=np.float64)
    I_ref = np.asarray(I_ref, dtype=np.float64)

    # Interpolación lineal 1D sobre la malla densa de referencia
    I_obs = np.interp(sample_times, t_ref, I_ref)
    return I_obs


def add_gaussian_noise(
    I_obs: np.ndarray,
    sigma: float = 0.02,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Añade ruido aditivo Gaussiano homocedástico N(0, sigma^2) a las observaciones
    de infectados, asegurando la no-negatividad física mediante proyección truncada.

        I_{noisy} = max(0, I_{obs} + epsilon),   epsilon ~ N(0, sigma^2)

    Parámetros
    ----------
    I_obs : np.ndarray
        Vector 1D de observaciones limpias de infectados.
    sigma : float, opcional
        Desviación estándar del ruido Gaussiano (por defecto 0.02).
    seed : int, opcional
        Semilla para el generador pseudoaleatorio de NumPy (por defecto 42).

    Retorna
    -------
    np.ndarray
        Vector 1D de observaciones con ruido Gaussiano proyectadas a R_{>=0}.
    """
    I_obs = np.asarray(I_obs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=I_obs.shape)
    I_noisy = I_obs + noise
    return np.maximum(0.0, I_noisy)


def add_poisson_noise(
    I_obs: np.ndarray,
    N_pop: int = 10000,
    seed: Optional[int] = 42
) -> np.ndarray:
    """
    Añade ruido de conteo Poissoniano escalado a una población total N_pop.

    Este modelo estadístico refleja el proceso generativo discreto de notificación
    de casos clínicos en epidemiología:
        Casos ~ Poisson(N_pop * I_obs)
        I_{noisy} = Casos / N_pop

    Propiedades:
    - Varianza heterocedástica proporcional a la media: Var(I) = I_obs / N_pop.
    - Preserva naturalmente la no-negatividad de los casos.

    Parámetros
    ----------
    I_obs : np.ndarray
        Vector 1D de observaciones limpias (proporción de la población).
    N_pop : int, opcional
        Tamaño de la población total (por defecto 10000).
    seed : int, opcional
        Semilla para el generador pseudoaleatorio de NumPy (por defecto 42).

    Retorna
    -------
    np.ndarray
        Vector 1D de proporciones de infectados observadas con dispersión Poissoniana.
    """
    I_obs = np.asarray(I_obs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    expected_counts = np.maximum(0.0, N_pop * I_obs)
    counts = rng.poisson(lam=expected_counts)
    return counts.astype(np.float64) / float(N_pop)
