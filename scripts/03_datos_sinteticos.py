#!/usr/bin/env python3
"""
Script 03: Generación de Datos Sintéticos y Modelado de Ruido Observacional.

Simula el proceso de recolección de datos epidemiológicos reales:
1. Generación de la verdad de campo clínica continua mediante RK45 (Ground Truth).
2. Muestreo en instantes temporales discretos (días 0, 1, 2, ..., 50).
3. Inyección de ruido observacional:
   - Ruido Gaussiano aditivo truncado: N(0, sigma^2), sigma = 0.02 (homocedástico).
   - Ruido Poissoniano escalado a población: Casos ~ Poisson(N_pop * I),
     con N_pop = 10,000 (heterocedástico, varianza proporcional a la media).
4. Análisis estadístico comparativo de las series observadas.

Universidad Nacional (UNA), Costa Rica.
Carrera de Ingeniería Matemática en Ciencia de Datos.
"""

import os
import sys
from pathlib import Path

# Configurar backend headless de matplotlib y directorio temporal de cache
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Asegurar importación de los módulos del paquete src
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.synthetic import (
    generate_ground_truth,
    sample_observations,
    add_gaussian_noise,
    add_poisson_noise,
)


def compute_statistics(y_clean: np.ndarray, y_noisy: np.ndarray) -> dict:
    """Calcula métricas estadísticas descriptivas y de error observacional."""
    residuals = y_noisy - y_clean
    return {
        "mean": float(np.mean(y_noisy)),
        "std": float(np.std(y_noisy)),
        "min": float(np.min(y_noisy)),
        "max": float(np.max(y_noisy)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "mae": float(np.mean(np.abs(residuals))),
        "res_std": float(np.std(residuals)),
        "res_mean": float(np.mean(residuals)),
    }


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Configuración de parámetros epidemiológicos y de muestreo
    # -------------------------------------------------------------------------
    beta = 0.4
    gamma = 0.1
    S0 = 0.99
    I0 = 0.01
    T = 50.0
    seed = 42

    sigma_gauss = 0.02
    N_pop = 10000

    print("=" * 85)
    print(" LABORATORIO 3: GENERACIÓN DE DATOS SINTÉTICOS Y RUIDO OBSERVACIONAL")
    print(f" Parámetros verdaderos: beta = {beta:.2f}, gamma = {gamma:.2f}, R0 = {beta/gamma:.2f}")
    print(f" Ruido Gaussiano: sigma = {sigma_gauss:.4f} (Homocedástico)")
    print(f" Ruido Poisson: N_pop = {N_pop:,} personas (Heterocedástico)")
    print("=" * 85)

    # 1. Verdad de campo continua (Ground Truth)
    t_ref, S_ref, I_ref, R_ref = generate_ground_truth(
        beta=beta, gamma=gamma, S0=S0, I0=I0, T=T, dt_ref=0.005
    )

    # 2. Muestreo discreto en días t_k = 0, 1, ..., 50
    t_obs = np.arange(0.0, T + 1.0, 1.0)
    I_clean = sample_observations(t_ref=t_ref, I_ref=I_ref, sample_times=t_obs)

    # 3. Aplicación de modelos de ruido
    I_gauss = add_gaussian_noise(I_clean, sigma=sigma_gauss, seed=seed)
    I_poiss = add_poisson_noise(I_clean, N_pop=N_pop, seed=seed)

    # 4. Resumen estadístico
    stats_clean = {
        "mean": float(np.mean(I_clean)),
        "std": float(np.std(I_clean)),
        "min": float(np.min(I_clean)),
        "max": float(np.max(I_clean)),
        "rmse": 0.0,
        "mae": 0.0,
        "res_std": 0.0,
        "res_mean": 0.0,
    }
    stats_gauss = compute_statistics(I_clean, I_gauss)
    stats_poiss = compute_statistics(I_clean, I_poiss)

    print("\nRESUMEN ESTADÍSTICO DE LAS SERIES DE INFECTADOS I(t):")
    print("-" * 85)
    header = f"{'Serie':<20} | {'Media':<9} | {'Desv.Est':<9} | {'Mínimo':<8} | {'Máximo':<8} | {'RMSE':<9} | {'MAE':<9}"
    print(header)
    print("-" * len(header))

    series_data = [
        ("Limpia (Ground Truth)", stats_clean),
        ("Ruido Gaussiano", stats_gauss),
        ("Ruido Poisson", stats_poiss),
    ]
    for name, s in series_data:
        print(
            f"{name:<20} | {s['mean']:<9.4f} | {s['std']:<9.4f} | "
            f"{s['min']:<8.4f} | {s['max']:<8.4f} | {s['rmse']:<9.4e} | {s['mae']:<9.4e}"
        )

    print("-" * 85)
    print(f"Sesgo medio residual Gaussiano: {stats_gauss['res_mean']:+.4e} (Esperado ~ 0)")
    print(f"Sesgo medio residual Poisson:   {stats_poiss['res_mean']:+.4e} (Esperado ~ 0)")
    print(f"Desv. estándar residual Gaussiano: {stats_gauss['res_std']:.4f} (Teórico = {sigma_gauss:.4f})")
    print(f"Total de casos notificados (Poisson): {int(np.sum(I_poiss * N_pop)):,} casos en {len(t_obs)} días")

    # -------------------------------------------------------------------------
    # 5. Visualización: Panel de 2 gráficos (Trayectorias y Residuos)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})

    # Panel superior: Curvas de Prevalencia
    ax1.plot(
        t_ref, I_ref,
        color="black",
        linestyle="-",
        linewidth=2.2,
        label="Ground Truth $I(t)$ (RK45 Continuo)",
        zorder=3
    )
    ax1.scatter(
        t_obs, I_gauss,
        color="#1f77b4",
        edgecolors="navy",
        marker="o",
        s=42,
        alpha=0.85,
        label=f"Ruido Gaussiano $\\mathcal{{N}}(0, \\sigma^2)$, $\\sigma = {sigma_gauss}$",
        zorder=4
    )
    ax1.scatter(
        t_obs, I_poiss,
        color="#d62728",
        edgecolors="darkred",
        marker="^",
        s=48,
        alpha=0.85,
        label=f"Ruido Poisson $\\mathrm{{Pois}}(N_{{pop}} I) / N_{{pop}}$, $N_{{pop}} = {N_pop:,}$",
        zorder=5
    )

    ax1.set_title("Simulación Epidemiológica: Datos Sintéticos de Infectados $I(t)$", fontsize=12, pad=10)
    ax1.set_ylabel("Proporción de Infectados $I(t)$", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=10)

    # Panel inferior: Residuos Observacionales
    res_gauss = I_gauss - I_clean
    res_poiss = I_poiss - I_clean

    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    # Bandas de 2 sigma para Gaussiano
    ax2.fill_between(
        t_obs, -2 * sigma_gauss, 2 * sigma_gauss,
        color="#1f77b4", alpha=0.12, label="Banda Gaussiana $\\pm 2\\sigma$"
    )

    ax2.stem(
        t_obs - 0.15, res_gauss,
        linefmt="C0-", markerfmt="C0o", basefmt="none",
        label="Residuos Gaussianos",
    )
    ax2.stem(
        t_obs + 0.15, res_poiss,
        linefmt="C3-", markerfmt="C3^", basefmt="none",
        label="Residuos Poisson",
    )

    ax2.set_title("Residuos Observacionales: $I_{\\mathrm{obs}}(t_k) - I_{\\mathrm{clean}}(t_k)$", fontsize=11)
    ax2.set_xlabel("Tiempo $t$ [días]", fontsize=11)
    ax2.set_ylabel("Error Residual", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9)

    fig.tight_layout()

    out_dir = BASE_DIR / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figura_03_datos_sinteticos.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 85)
    print(f"[OK] Gráfico exportado exitosamente a: {out_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
