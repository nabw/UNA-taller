"""
Módulo de calibración de parámetros y resolución de problemas inversos en EDOs.

Implementa la estimación de parámetros (beta, gamma, R0) a partir de datos observacionales
de series temporales, el cálculo de paisajes de pérdida (loss surface), el rastreo
del algoritmo Nelder-Mead y el análisis del sesgo numérico de discretización.

Universidad Nacional (UNA), Costa Rica.
Carrera de Ingeniería Matemática en Ciencia de Datos.
"""

from typing import Dict, Any, Sequence, Optional, Union
import numpy as np
from scipy.optimize import minimize, OptimizeResult

from src.solvers import solve_sir_imex, solve_sir_explicit, solve_sir_implicit


# ==============================================================================
# 1. FUNCIÓN DE PÉRDIDA Y PAISAJE DE OPTIMIZACIÓN
# ==============================================================================

def loss_function(
    theta: Sequence[float],
    t_obs: np.ndarray,
    I_obs: np.ndarray,
    dt_sim: float,
    solver_name: str = "imex",
    S0: float = 0.99,
    I0: float = 0.01
) -> float:
    """
    Función de pérdida por mínimos cuadrados ordinarios (MSE no normalizado):
        L(theta) = sum_{k=1}^M (I_{sim}(t_k; theta) - I_{obs}(t_k))^2

    Parámetros
    ----------
    theta : Sequence[float]
        Vector de parámetros [beta, gamma].
    t_obs : np.ndarray
        Vector 1D de tiempos de observación [t_1, t_2, ..., t_M].
    I_obs : np.ndarray
        Vector 1D de observaciones de infectados en los tiempos t_obs.
    dt_sim : float
        Paso temporal Delta t utilizado por el solver hacia adelante.
    solver_name : str, opcional
        Nombre del esquema de discretización: "imex" (por defecto),
        "explicit" o "implicit".
    S0 : float, opcional
        Fracción inicial de susceptibles (por defecto 0.99).
    I0 : float, opcional
        Fracción inicial de infectados (por defecto 0.01).

    Retorna
    -------
    float
        Valor de la pérdida cuadrática L(theta). Si beta <= 0 o gamma <= 0,
        o si la simulación numérica diverge/inestabiliza, retorna 1e6 como penalización.
    """
    beta, gamma = float(theta[0]), float(theta[1])

    # Penalización dura por violación de positividad física
    PENALTY = 1e6
    if beta <= 0.0 or gamma <= 0.0:
        return PENALTY

    t_obs = np.asarray(t_obs, dtype=np.float64)
    I_obs = np.asarray(I_obs, dtype=np.float64)
    T_final = float(t_obs[-1])

    solver_key = solver_name.lower().strip()
    try:
        if solver_key == "imex":
            t_sim, _, I_sim, _ = solve_sir_imex(S0, I0, beta, gamma, dt=dt_sim, T=T_final)
        elif solver_key in ("explicit", "euler"):
            t_sim, _, I_sim, _ = solve_sir_explicit(S0, I0, beta, gamma, dt=dt_sim, T=T_final)
        elif solver_key == "implicit":
            t_sim, _, I_sim, _ = solve_sir_implicit(S0, I0, beta, gamma, dt=dt_sim, T=T_final)
        else:
            raise ValueError(
                f"Solver desconocido: '{solver_name}'. "
                "Opciones válidas: 'imex', 'explicit', 'implicit'."
            )
    except Exception:
        # En caso de fallo numérico durante la integración
        return PENALTY

    # Verificación de estabilidad y valores válidos
    if np.any(np.isnan(I_sim)) or np.any(np.isinf(I_sim)) or np.any(I_sim < -1e-3) or np.any(I_sim > 2.0):
        return PENALTY

    # Interpolar la solución numérica sobre los tiempos exactos de observación
    I_interp = np.interp(t_obs, t_sim, I_sim)

    # Suma de residuos al cuadrado
    residuals = I_interp - I_obs
    loss = float(np.sum(residuals ** 2))

    if np.isnan(loss) or np.isinf(loss):
        return PENALTY

    return loss


def compute_loss_grid(
    beta_vals: np.ndarray,
    gamma_vals: np.ndarray,
    t_obs: np.ndarray,
    I_obs: np.ndarray,
    dt_sim: float,
    solver_name: str = "imex",
    S0: float = 0.99,
    I0: float = 0.01
) -> np.ndarray:
    """
    Evalúa la función de pérdida sobre una rejilla bidimensional regular de parámetros
    (beta, gamma) para visualización de contornos y análisis de identifiabilidad.

    Parámetros
    ----------
    beta_vals : np.ndarray
        Vector 1D de valores de prueba para la tasa de transmisión beta.
    gamma_vals : np.ndarray
        Vector 1D de valores de prueba para la tasa de recuperación gamma.
    t_obs : np.ndarray
        Vector 1D de tiempos de observación.
    I_obs : np.ndarray
        Vector 1D de observaciones de infectados.
    dt_sim : float
        Paso temporal de integración numérica para el modelo directo.
    solver_name : str, opcional
        Esquema numérico a emplear ("imex", "explicit", "implicit").
    S0 : float, opcional
        Condición inicial S(0).
    I0 : float, opcional
        Condición inicial I(0).

    Retorna
    -------
    L_grid : np.ndarray
        Matriz 2D de tamaño (len(gamma_vals), len(beta_vals)) donde
        L_grid[i, j] = loss_function([beta_vals[j], gamma_vals[i]], ...).
        Esta disposición es compatible directa con np.meshgrid(beta_vals, gamma_vals)
        y plt.contourf(beta_vals, gamma_vals, L_grid).
    """
    beta_vals = np.asarray(beta_vals, dtype=np.float64)
    gamma_vals = np.asarray(gamma_vals, dtype=np.float64)

    n_gamma = len(gamma_vals)
    n_beta = len(beta_vals)
    L_grid = np.zeros((n_gamma, n_beta), dtype=np.float64)

    for i in range(n_gamma):
        g = gamma_vals[i]
        for j in range(n_beta):
            b = beta_vals[j]
            L_grid[i, j] = loss_function(
                theta=[b, g],
                t_obs=t_obs,
                I_obs=I_obs,
                dt_sim=dt_sim,
                solver_name=solver_name,
                S0=S0,
                I0=I0
            )

    return L_grid


# ==============================================================================
# 2. CALIBRACIÓN DE PARÁMETROS Y ALGORITMO NELDER-MEAD
# ==============================================================================

def calibrate_parameters(
    theta_init: Sequence[float],
    t_obs: np.ndarray,
    I_obs: np.ndarray,
    dt_sim: float,
    solver_name: str = "imex",
    track_simplex: bool = True,
    S0: float = 0.99,
    I0: float = 0.01,
    maxiter: int = 500,
    tol: float = 1e-6
) -> Dict[str, Any]:
    """
    Calibra los parámetros del modelo SIR (beta, gamma) mediante optimización
    libre de derivadas (Algoritmo Simplex de Nelder-Mead).

    Permite registrar la secuencia de vértices explorados por el simplex en cada
    iteración para proyectar pedagógicamente el descenso sobre la superficie de costo.

    Parámetros
    ----------
    theta_init : Sequence[float]
        Punto de partida [beta_0, gamma_0] para la optimización.
    t_obs : np.ndarray
        Vector 1D de tiempos de observación.
    I_obs : np.ndarray
        Vector 1D de observaciones clínicas de infectados.
    dt_sim : float
        Paso temporal Delta t con el que se ejecuta el solver numérico.
    solver_name : str, opcional
        Esquema numérico a calibrar ("imex", "explicit", "implicit").
    track_simplex : bool, opcional
        Si es True, registra la trayectoria de los puntos evaluados por el algoritmo.
    S0 : float, opcional
        Condición inicial S(0).
    I0 : float, opcional
        Condición inicial I(0).
    maxiter : int, opcional
        Número máximo de iteraciones permitidas en Nelder-Mead.
    tol : float, opcional
        Tolerancia de convergencia para x y f(x).

    Retorna
    -------
    Dict[str, Any]
        Diccionario con las siguientes claves:
        - 'theta_hat': np.ndarray con los parámetros óptimos [beta_hat, gamma_hat].
        - 'beta_hat': float con la estimación de beta.
        - 'gamma_hat': float con la estimación de gamma.
        - 'R0_hat': float con el número reproductivo básico estimado (beta_hat / gamma_hat).
        - 'loss_val': float con el valor mínimo de la función de pérdida.
        - 'success': bool indicando si el optimizador convergió exitosamente.
        - 'iterations': int con el número total de iteraciones realizadas.
        - 'history': np.ndarray 2D de forma (K, 2) con la trayectoria del optimizador.
        - 'result': objeto OptimizeResult devuelto por SciPy.
    """
    theta_init = np.asarray(theta_init, dtype=np.float64)

    # Historial de trayectoria del simplex
    history = [theta_init.copy()]

    def callback_tracker(xk: np.ndarray) -> None:
        history.append(np.array(xk, dtype=np.float64).copy())

    def objective(theta: np.ndarray) -> float:
        return loss_function(
            theta=theta,
            t_obs=t_obs,
            I_obs=I_obs,
            dt_sim=dt_sim,
            solver_name=solver_name,
            S0=S0,
            I0=I0
        )

    res: OptimizeResult = minimize(
        fun=objective,
        x0=theta_init,
        method="Nelder-Mead",
        callback=callback_tracker if track_simplex else None,
        options={
            "maxiter": maxiter,
            "xatol": tol,
            "fatol": tol,
            "disp": False
        }
    )

    theta_hat = np.asarray(res.x, dtype=np.float64)
    beta_hat = float(theta_hat[0])
    gamma_hat = float(theta_hat[1])
    r0_hat = float(beta_hat / gamma_hat) if gamma_hat > 0.0 else float(np.nan)

    # Si el punto final difiere del último punto registrado en history, agregarlo
    if track_simplex:
        if len(history) == 0 or not np.allclose(history[-1], theta_hat):
            history.append(theta_hat.copy())
        history_arr = np.array(history, dtype=np.float64)
    else:
        history_arr = np.empty((0, 2), dtype=np.float64)

    return {
        "theta_hat": theta_hat,
        "beta_hat": beta_hat,
        "gamma_hat": gamma_hat,
        "R0_hat": r0_hat,
        "loss_val": float(res.fun),
        "success": bool(res.success),
        "iterations": int(res.nit),
        "history": history_arr,
        "result": res
    }


# ==============================================================================
# 3. EXPERIMENTO DE SESGO DE DISCRETIZACIÓN
# ==============================================================================

def experiment_discretization_bias(
    dt_values: Sequence[float],
    t_obs: np.ndarray,
    I_obs: np.ndarray,
    theta_init: Sequence[float] = (0.2, 0.2),
    solver_name: str = "imex",
    S0: float = 0.99,
    I0: float = 0.01
) -> Dict[str, np.ndarray]:
    """
    Evalúa empíricamente el fenómeno de Sesgo de Discretización (Discretization Bias).

    Al calibrar parámetros continuos (beta, gamma) a través de un esquema numérico
    con paso temporal finito Delta t, el estimador resultante theta_hat(Delta t)
    se desvía del valor real theta* para compensar los errores locales de truncamiento
    del algoritmo.

    Parámetros
    ----------
    dt_values : Sequence[float]
        Colección de pasos temporales Delta t a evaluar (ej. [0.05, 0.1, 0.2, ..., 1.0]).
    t_obs : np.ndarray
        Vector 1D de tiempos de observación.
    I_obs : np.ndarray
        Vector 1D de observaciones clínicas de infectados.
    theta_init : Sequence[float], opcional
        Punto de partida para el optimizador (por defecto [0.2, 0.2]).
    solver_name : str, opcional
        Esquema numérico a evaluar (por defecto "imex").
    S0 : float, opcional
        Condición inicial S(0).
    I0 : float, opcional
        Condición inicial I(0).

    Retorna
    -------
    Dict[str, np.ndarray]
        Resumen estructurado con las siguientes claves (cada una un np.ndarray 1D):
        - 'dt': pasos temporales evaluados.
        - 'beta_hat': estimaciones de beta en función de dt.
        - 'gamma_hat': estimaciones de gamma en función de dt.
        - 'R0_hat': estimaciones de R0 en función de dt.
        - 'loss': pérdidas finales alcanzadas.
        - 'success': indicadores booleanos de convergencia.
    """
    dt_list = []
    beta_list = []
    gamma_list = []
    r0_list = []
    loss_list = []
    success_list = []

    for dt in dt_values:
        dt_val = float(dt)
        calib = calibrate_parameters(
            theta_init=theta_init,
            t_obs=t_obs,
            I_obs=I_obs,
            dt_sim=dt_val,
            solver_name=solver_name,
            track_simplex=False,
            S0=S0,
            I0=I0
        )

        dt_list.append(dt_val)
        beta_list.append(calib["beta_hat"])
        gamma_list.append(calib["gamma_hat"])
        r0_list.append(calib["R0_hat"])
        loss_list.append(calib["loss_val"])
        success_list.append(calib["success"])

    return {
        "dt": np.array(dt_list, dtype=np.float64),
        "beta_hat": np.array(beta_list, dtype=np.float64),
        "gamma_hat": np.array(gamma_list, dtype=np.float64),
        "R0_hat": np.array(r0_list, dtype=np.float64),
        "loss": np.array(loss_list, dtype=np.float64),
        "success": np.array(success_list, dtype=bool)
    }
