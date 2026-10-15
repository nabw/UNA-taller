"""
Módulo de visualización científica y pedagógica con Matplotlib.

Contiene funciones especializadas para ilustrar:
1. Comparación de esquemas numéricos para la ecuación logística.
2. Dinámica comparada del modelo epidemiológico SIR y conservación de masa.
3. Observaciones sintéticas con ruido observacional.
4. Superficies de costo 2D y trayectoria del algoritmo Nelder-Mead.
5. Efecto del sesgo de discretización numérica sobre los parámetros estimados.

Universidad Nacional (UNA), Costa Rica.
Carrera de Ingeniería Matemática en Ciencia de Datos.
"""

from typing import Tuple, Dict, Sequence, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ==============================================================================
# 1. VISUALIZACIÓN DE LA ECUACIÓN LOGÍSTICA
# ==============================================================================

def plot_logistic(
    t_exp: np.ndarray,
    x_exp: np.ndarray,
    t_imp: np.ndarray,
    x_imp: np.ndarray,
    t_imex: np.ndarray,
    x_imex: np.ndarray,
    t_exact: Optional[np.ndarray] = None,
    x_exact: Optional[np.ndarray] = None,
    dt: Optional[float] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Compara las trayectorias numéricas de la ecuación logística obtenidas mediante
    Euler Explícito, Euler Implícito y el esquema IMEX linealizado.

    Parámetros
    ----------
    t_exp : np.ndarray
        Vector de tiempos para Euler Explícito.
    x_exp : np.ndarray
        Solución con Euler Explícito.
    t_imp : np.ndarray
        Vector de tiempos para Euler Implícito.
    x_imp : np.ndarray
        Solución con Euler Implícito.
    t_imex : np.ndarray
        Vector de tiempos para IMEX.
    x_imex : np.ndarray
        Solución con IMEX.
    t_exact : np.ndarray, opcional
        Vector de tiempos de alta resolución para la solución exacta analítica.
    x_exact : np.ndarray, opcional
        Solución exacta evaluada en t_exact.
    dt : float, opcional
        Tamaño de paso utilizado, para incluirlo en el título descriptivo.

    Retorna
    -------
    fig : plt.Figure
        Objeto figura de Matplotlib.
    ax : plt.Axes
        Eje de Matplotlib con las curvas graficadas.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Curva exacta de referencia continua
    if t_exact is not None and x_exact is not None:
        ax.plot(
            t_exact, x_exact,
            color="black",
            linestyle="-",
            linewidth=2.0,
            label="Solución Exacta",
            zorder=2
        )

    # Esquema IMEX (verde, línea y marcadores sutiles)
    ax.plot(
        t_imex, x_imex,
        color="#2ca02c",
        linestyle="--",
        marker="^",
        markersize=5,
        linewidth=1.8,
        label="IMEX Linealizado",
        zorder=4
    )

    # Euler Implícito (azul)
    ax.plot(
        t_imp, x_imp,
        color="#1f77b4",
        linestyle="-.",
        marker="s",
        markersize=4.5,
        linewidth=1.8,
        label="Euler Implícito",
        zorder=3
    )

    # Euler Explícito (rojo con marcadores)
    ax.plot(
        t_exp, x_exp,
        color="#d62728",
        linestyle=":",
        marker="o",
        markersize=4.5,
        linewidth=1.6,
        label="Euler Explícito",
        zorder=5
    )

    # Línea asíntota del atractor x* = 1
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    title_str = "Dinámica de la Ecuación Logística: $\\dot{x} = C x (1 - x)$"
    if dt is not None:
        title_str += f" ($\\Delta t = {dt}$)"
    ax.set_title(title_str, fontsize=12, pad=12)

    ax.set_xlabel("Tiempo $t$", fontsize=11)
    ax.set_ylabel("Población normalizada $x(t)$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

    # Ajuste de límites verticales para evitar que oscilaciones explícitas arruinen la escala
    all_x = np.concatenate([x_imp, x_imex])
    if x_exact is not None:
        all_x = np.concatenate([all_x, x_exact])
    min_val = min(-0.1, np.min(all_x) - 0.1)
    max_val = max(1.2, min(3.0, np.max(x_exp) + 0.1) if not np.any(np.isnan(x_exp)) else 1.2)
    ax.set_ylim(min_val, max_val)

    fig.tight_layout()
    return fig, ax


# ==============================================================================
# 2. COMPARACIÓN DE ESQUEMAS SIR
# ==============================================================================

def plot_sir_comparison(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    title: str = "Comparación de Esquemas SIR"
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Genera un panel comparativo de 2x2 para el modelo SIR que contrasta:
    - Compartimento Susceptibles S(t)
    - Compartimento Infectados I(t)
    - Compartimento Recuperados R(t)
    - Verificación de la conservación de la población total: S(t) + I(t) + R(t) = 1

    Parámetros
    ----------
    results_dict : Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
        Diccionario con formato {nombre_esquema: (t, S, I, R)}.
        Ejemplo:
            {
                "Explícito": (t_exp, S_exp, I_exp, R_exp),
                "Implícito": (t_imp, S_imp, I_imp, R_imp),
                "IMEX": (t_imex, S_imex, I_imex, R_imex)
            }
    title : str, opcional
        Título principal de la figura.

    Retorna
    -------
    fig : plt.Figure
        Objeto figura de Matplotlib.
    axes : np.ndarray
        Matriz 2D de ejes (shape (2, 2)) de los subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    ax_S, ax_I = axes[0, 0], axes[0, 1]
    ax_R, ax_N = axes[1, 0], axes[1, 1]

    # Paleta de estilos y colores diferenciados para cada solver
    styles = {
        "Explícito": {"color": "#d62728", "linestyle": ":", "marker": "o", "markevery": 5},
        "Implícito": {"color": "#1f77b4", "linestyle": "-.", "marker": "s", "markevery": 5},
        "IMEX": {"color": "#2ca02c", "linestyle": "--", "marker": "^", "markevery": 5},
        "Referencia": {"color": "black", "linestyle": "-", "marker": None, "markevery": 1},
        "Ground Truth": {"color": "black", "linestyle": "-", "marker": None, "markevery": 1}
    }

    color_palette = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]

    for idx, (name, (t, S, I, R)) in enumerate(results_dict.items()):
        style = styles.get(name, {
            "color": color_palette[idx % len(color_palette)],
            "linestyle": "-",
            "marker": None,
            "markevery": 1
        })

        kw = {
            "label": name,
            "color": style["color"],
            "linestyle": style["linestyle"],
            "linewidth": 1.8
        }
        if style["marker"] is not None:
            kw["marker"] = style["marker"]
            kw["markevery"] = style["markevery"]
            kw["markersize"] = 4.5

        ax_S.plot(t, S, **kw)
        ax_I.plot(t, I, **kw)
        ax_R.plot(t, R, **kw)

        # Conservación de la masa total S + I + R
        total_pop = S + I + R
        ax_N.plot(t, total_pop, **kw)

    # Configuración de los paneles
    ax_S.set_title("Susceptibles $S(t)$", fontsize=11)
    ax_S.set_ylabel("Proporción $S(t)$", fontsize=10)
    ax_S.grid(True, linestyle=":", alpha=0.6)
    ax_S.legend(loc="best", fontsize=9)

    ax_I.set_title("Infectados $I(t)$", fontsize=11)
    ax_I.set_ylabel("Proporción $I(t)$", fontsize=10)
    ax_I.grid(True, linestyle=":", alpha=0.6)
    ax_I.legend(loc="best", fontsize=9)

    ax_R.set_title("Recuperados $R(t)$", fontsize=11)
    ax_R.set_xlabel("Tiempo $t$ [días]", fontsize=10)
    ax_R.set_ylabel("Proporción $R(t)$", fontsize=10)
    ax_R.grid(True, linestyle=":", alpha=0.6)
    ax_R.legend(loc="best", fontsize=9)

    ax_N.set_title("Conservación Total: $S(t) + I(t) + R(t)$", fontsize=11)
    ax_N.set_xlabel("Tiempo $t$ [días]", fontsize=10)
    ax_N.set_ylabel("Población Total", fontsize=10)
    ax_N.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_N.grid(True, linestyle=":", alpha=0.6)
    ax_N.legend(loc="best", fontsize=9)

    fig.suptitle(title, fontsize=13, y=0.98)
    fig.tight_layout()
    return fig, axes


# ==============================================================================
# 3. OBSERVACIONES SINTÉTICAS CON RUIDO
# ==============================================================================

def plot_synthetic_observations(
    t_obs: np.ndarray,
    I_clean: np.ndarray,
    I_gauss: np.ndarray,
    I_poiss: np.ndarray
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Grafica la curva real de infectados junto con las observaciones sintéticas
    perturbadas por ruido Gaussiano y Poisson.

    Parámetros
    ----------
    t_obs : np.ndarray
        Vector 1D de tiempos discretos de muestreo (días).
    I_clean : np.ndarray
        Vector 1D de infectados sin ruido (Ground Truth).
    I_gauss : np.ndarray
        Vector 1D de observaciones con ruido Gaussiano proyectado.
    I_poiss : np.ndarray
        Vector 1D de observaciones con ruido Poisson discreto escalado.

    Retorna
    -------
    fig : plt.Figure
        Objeto figura de Matplotlib.
    ax : plt.Axes
        Eje con las curvas y dispersiones graficadas.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Curva limpia continua
    ax.plot(
        t_obs, I_clean,
        color="black",
        linestyle="-",
        linewidth=2.2,
        label="Ground Truth (Limpia)",
        zorder=3
    )

    # Observaciones con ruido Gaussiano
    ax.scatter(
        t_obs, I_gauss,
        color="#1f77b4",
        edgecolors="navy",
        marker="o",
        s=40,
        alpha=0.75,
        label="Ruido Gaussiano $\\mathcal{N}(0, \\sigma^2)$",
        zorder=4
    )

    # Observaciones con ruido Poisson
    ax.scatter(
        t_obs, I_poiss,
        color="#d62728",
        edgecolors="darkred",
        marker="^",
        s=45,
        alpha=0.75,
        label="Ruido Poisson (Casos / $N_{pop}$)",
        zorder=5
    )

    ax.set_title("Datos Sintéticos de Prevalencia de Infectados $I(t)$", fontsize=12, pad=12)
    ax.set_xlabel("Tiempo $t$ [días]", fontsize=11)
    ax.set_ylabel("Proporción de Infectados $I(t)$", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

    fig.tight_layout()
    return fig, ax


# ==============================================================================
# 4. SUPERFICIE DE COSTO Y CONVERGENCIA NELDER-MEAD
# ==============================================================================

def plot_loss_surface(
    beta_vals: np.ndarray,
    gamma_vals: np.ndarray,
    L_grid: np.ndarray,
    theta_true: Optional[Sequence[float]] = None,
    theta_hat: Optional[Sequence[float]] = None,
    simplex_history: Optional[np.ndarray] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Visualiza el paisaje de pérdida L(beta, gamma) mediante curvas de nivel y
    proyecta la trayectoria de optimización del algoritmo Nelder-Mead.

    Parámetros
    ----------
    beta_vals : np.ndarray
        Vector 1D de valores de beta (eje X).
    gamma_vals : np.ndarray
        Vector 1D de valores de gamma (eje Y).
    L_grid : np.ndarray
        Matriz 2D de pérdidas con forma (len(gamma_vals), len(beta_vals)).
    theta_true : Sequence[float], opcional
        Parámetros verdaderos de referencia [beta*, gamma*].
    theta_hat : Sequence[float], opcional
        Parámetros calibrados [beta_hat, gamma_hat].
    simplex_history : np.ndarray, opcional
        Matriz 2D de forma (K, 2) con la secuencia de puntos evaluados por el optimizador.

    Retorna
    -------
    fig : plt.Figure
        Objeto figura de Matplotlib.
    ax : plt.Axes
        Eje con el contorno de pérdida y la trayectoria del simplex.
    """
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    B, G = np.meshgrid(beta_vals, gamma_vals)

    # Filtrar valores no positivos o excesivos para escala logarítmica adecuada
    valid_mask = (L_grid > 0.0) & (L_grid < 1e5)
    if np.any(valid_mask):
        vmin = float(np.min(L_grid[valid_mask]))
        vmax = float(np.percentile(L_grid[valid_mask], 92))
    else:
        vmin, vmax = 1e-4, 1.0

    vmin = max(vmin, 1e-6)
    if vmax <= vmin:
        vmax = vmin * 10.0

    levels = np.logspace(np.log10(vmin), np.log10(vmax), 28)

    # Mapa de contornos rellenos
    cf = ax.contourf(
        B, G, L_grid,
        levels=levels,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis_r",
        extend="both"
    )

    # Líneas de contorno sutiles
    ax.contour(
        B, G, L_grid,
        levels=levels[::3],
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        colors="white",
        linewidths=0.5,
        alpha=0.5
    )

    cbar = fig.colorbar(cf, ax=ax, pad=0.03)
    cbar.set_label("Pérdida $\\mathcal{L}(\\beta, \\gamma)$ [Escala Logarítmica]", fontsize=10)

    # Trayectoria de optimización Nelder-Mead
    if simplex_history is not None and len(simplex_history) > 0:
        hist = np.asarray(simplex_history)
        ax.plot(
            hist[:, 0], hist[:, 1],
            color="white",
            linestyle="--",
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            markerfacecolor="cyan",
            markeredgecolor="black",
            label="Trayectoria Nelder-Mead",
            zorder=6
        )
        # Punto inicial
        ax.scatter(
            hist[0, 0], hist[0, 1],
            color="cyan",
            edgecolors="black",
            marker="o",
            s=90,
            label="Punto Inicial $\\theta_0$",
            zorder=7
        )

    # Parámetro verdadero (estrella dorada)
    if theta_true is not None:
        ax.scatter(
            theta_true[0], theta_true[1],
            color="gold",
            edgecolors="black",
            marker="*",
            s=220,
            label="Verdadero $\\theta^* = (\\beta^*, \\gamma^*)$",
            zorder=8
        )

    # Parámetro óptimo calibrado (cruz roja)
    if theta_hat is not None:
        ax.scatter(
            theta_hat[0], theta_hat[1],
            color="#ff3333",
            edgecolors="black",
            marker="X",
            s=120,
            label="Calibrado $\\hat{\\theta} = (\\hat{\\beta}, \\hat{\\gamma})$",
            zorder=9
        )

    ax.set_title("Superficie de Pérdida Cuadrática y Convergencia de Nelder-Mead", fontsize=12, pad=12)
    ax.set_xlabel("Tasa de Transmisión $\\beta$", fontsize=11)
    ax.set_ylabel("Tasa de Recuperación $\\gamma$", fontsize=11)
    ax.legend(loc="upper left", frameon=True, facecolor="black", edgecolor="white", labelcolor="white", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.4, color="white")

    fig.tight_layout()
    return fig, ax


# ==============================================================================
# 5. SESGO DE DISCRETIZACIÓN NUMÉRICA
# ==============================================================================

def plot_discretization_bias(
    bias_results: Dict[str, np.ndarray],
    theta_true: Tuple[float, float] = (0.4, 0.1)
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Visualiza el sesgo de discretización mostrando cómo varían las estimaciones
    de beta_hat, gamma_hat y R0_hat al aumentar el tamaño del paso temporal Delta t.

    Parámetros
    ----------
    bias_results : Dict[str, np.ndarray]
        Diccionario con arreglos ['dt', 'beta_hat', 'gamma_hat', 'R0_hat'].
    theta_true : Tuple[float, float], opcional
        Valores reales de referencia (beta*, gamma*) (por defecto (0.4, 0.1)).

    Retorna
    -------
    fig : plt.Figure
        Objeto figura de Matplotlib.
    axes : np.ndarray
        Vector 1D de ejes (3 subplots).
    """
    dt_vals = bias_results["dt"]
    beta_hat = bias_results["beta_hat"]
    gamma_hat = bias_results["gamma_hat"]
    r0_hat = bias_results["R0_hat"]

    beta_true, gamma_true = theta_true
    r0_true = beta_true / gamma_true

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True)
    ax1, ax2, ax3 = axes[0], axes[1], axes[2]

    # Panel 1: beta_hat vs dt
    ax1.plot(dt_vals, beta_hat, marker="o", color="#1f77b4", linewidth=1.8, markersize=5, label="$\\hat{\\beta}(\\Delta t)$")
    ax1.axhline(beta_true, color="#d62728", linestyle="--", linewidth=1.5, label=f"Verdadero $\\beta^* = {beta_true}$")
    ax1.set_title("Estimación de $\\beta$", fontsize=11)
    ax1.set_xlabel("Paso temporal $\\Delta t$", fontsize=10)
    ax1.set_ylabel("Valor estimado $\\hat{\\beta}$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", fontsize=9)

    # Panel 2: gamma_hat vs dt
    ax2.plot(dt_vals, gamma_hat, marker="s", color="#2ca02c", linewidth=1.8, markersize=5, label="$\\hat{\\gamma}(\\Delta t)$")
    ax2.axhline(gamma_true, color="#d62728", linestyle="--", linewidth=1.5, label=f"Verdadero $\\gamma^* = {gamma_true}$")
    ax2.set_title("Estimación de $\\gamma$", fontsize=11)
    ax2.set_xlabel("Paso temporal $\\Delta t$", fontsize=10)
    ax2.set_ylabel("Valor estimado $\\hat{\\gamma}$", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", fontsize=9)

    # Panel 3: R0_hat vs dt
    ax3.plot(dt_vals, r0_hat, marker="^", color="#9467bd", linewidth=1.8, markersize=5, label="$\\hat{\\mathcal{R}}_0(\\Delta t)$")
    ax3.axhline(r0_true, color="#d62728", linestyle="--", linewidth=1.5, label=f"Verdadero $\\mathcal{{R}}_0^* = {r0_true:.1f}$")
    ax3.set_title("Número Reproductivo Básico $\\hat{\\mathcal{R}}_0$", fontsize=11)
    ax3.set_xlabel("Paso temporal $\\Delta t$", fontsize=10)
    ax3.set_ylabel("Valor estimado $\\hat{\\mathcal{R}}_0$", fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="best", fontsize=9)

    fig.suptitle("Efecto del Sesgo de Discretización sobre la Inferencia de Parámetros", fontsize=13, y=1.02)
    fig.tight_layout()
    return fig, axes
