#!/usr/bin/env python3
"""
Script 03: Generación de datos sintéticos con ruido observacional Gaussiano.

Simula el proceso de recolección de observaciones epidemiológicas:
1. Generación de la referencia continua mediante RK45 (Ground Truth).
2. Muestreo en instantes temporales discretos (días 0, 1, 2, ..., 50).
3. Inyección de ruido observacional Gaussiano N(0, sigma^2), sigma = 0.02,
   proyectado a valores no negativos.
4. Análisis estadístico de residuos observacionales.

Universidad Nacional (UNA), Costa Rica.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.synthetic import (
    generate_ground_truth,
    sample_observations,
    add_gaussian_noise,
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
    beta = 0.4
    gamma = 0.1
    S0 = 0.99
    I0 = 0.01
    T = 50.0
    seed = 42
    sigma_gauss = 0.02

    print("=" * 80)
    print("GENERACIÓN DE DATOS SINTÉTICOS Y RUIDO OBSERVACIONAL")
    print(f"Parámetros: beta = {beta:.2f}, gamma = {gamma:.2f}, R0 = {beta/gamma:.2f}")
    print(f"Ruido Gaussiano: sigma = {sigma_gauss:.4f}")
    print("=" * 80)

    # 1. Referencia continua
    t_ref, S_ref, I_ref, R_ref = generate_ground_truth(
        beta=beta, gamma=gamma, S0=S0, I0=I0, T=T, dt_ref=0.005
    )

    # 2. Muestreo discreto en días t_k = 0, 1, ..., 50
    t_obs = np.arange(0.0, T + 1.0, 1.0)
    I_clean = sample_observations(t_ref=t_ref, I_ref=I_ref, sample_times=t_obs)

    # 3. Ruido Gaussiano
    I_gauss = add_gaussian_noise(I_clean, sigma=sigma_gauss, seed=seed)

    # 4. Estadísticas
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

    print("\nResumen estadístico de las series de infectados I(t):")
    print("-" * 80)
    header = f"{'Serie':<25} | {'Media':<9} | {'Desv.Est':<9} | {'Mínimo':<8} | {'Máximo':<8} | {'RMSE':<9} | {'MAE':<9}"
    print(header)
    print("-" * len(header))

    series_data = [
        ("Referencia limpia", stats_clean),
        ("Observaciones con ruido", stats_gauss),
    ]
    for name, s in series_data:
        print(
            f"{name:<25} | {s['mean']:<9.4f} | {s['std']:<9.4f} | "
            f"{s['min']:<8.4f} | {s['max']:<8.4f} | {s['rmse']:<9.4e} | {s['mae']:<9.4e}"
        )

    print("-" * 80)
    print(f"Media de residuos: {stats_gauss['res_mean']:+.4e} (teórica = 0.0)")
    print(f"Desv. est. residuos: {stats_gauss['res_std']:.4f} (teórica = {sigma_gauss:.4f})")

    # 5. Visualización
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]}
    )

    # Panel superior: Curva y datos
    ax1.plot(
        t_ref, I_ref,
        color="black",
        linestyle="-",
        linewidth=2.0,
        label="Referencia continua $I(t)$",
        zorder=3
    )
    ax1.scatter(
        t_obs, I_gauss,
        color="#1f77b4",
        edgecolors="navy",
        marker="o",
        s=38,
        alpha=0.85,
        label=rf"Observaciones con ruido $\mathcal{{N}}(0, \sigma^2)$ ($\sigma={sigma_gauss}$)",
        zorder=4
    )

    ax1.set_title("Datos sintéticos de infectados $I(t)$", fontsize=12, pad=10)
    ax1.set_ylabel("Proporción de infectados $I(t)$", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    # Panel inferior: Residuos observacionales
    res_gauss = I_gauss - I_clean

    ax2.axhline(0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax2.fill_between(
        t_obs, -2 * sigma_gauss, 2 * sigma_gauss,
        color="#1f77b4", alpha=0.15, label=r"Banda $\pm 2\sigma$"
    )
    ax2.stem(
        t_obs, res_gauss,
        linefmt="b-", markerfmt="bo", basefmt=" "
    )

    ax2.set_title("Residuos observacionales: $I_{obs}(t) - I_{real}(t)$", fontsize=11, pad=8)
    ax2.set_xlabel("Tiempo $t$ [días]", fontsize=10)
    ax2.set_ylabel("Residuo", fontsize=10)
    ax2.set_ylim(-0.065, 0.065)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=9.5)

    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / "figura_03_datos_sinteticos.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura guardada en: {out_path}")


if __name__ == "__main__":
    main()
