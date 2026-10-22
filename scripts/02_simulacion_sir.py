#!/usr/bin/env python3
"""
Script 02: Simulación del problema directo SIR y conservación de masa.

Compara la integración numérica del sistema epidemiológico:
    dS/dt = - beta * S * I
    dI/dt =   beta * S * I - gamma * I
    dR/dt =   gamma * I

Esquemas numéricos evaluados:
1. Euler Explícito
2. Euler Implícito
3. Esquema IMEX (Implícito-Explícito Conservativo)
4. Solución de referencia continua (RK45 adaptativo)

Regímenes temporales evaluados:
- Paso fino: dt = 0.5 días
- Paso moderado: dt = 2.5 días (donde Euler explícito sobreestima marcadamente
  el peak epidémico y desfasa la onda epidémica)

Invariante física:
    S(t) + I(t) + R(t) = 1.0 (conservación de la población total)

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

from src.solvers import (
    solve_sir_explicit,
    solve_sir_implicit,
    solve_sir_imex,
)
from src.synthetic import generate_ground_truth


def main() -> None:
    beta = 0.4
    gamma = 0.1
    S0 = 0.99
    I0 = 0.01
    T = 50.0
    R0_basic = beta / gamma

    dt_fine = 0.5
    dt_moderate = 2.5

    print("=" * 85)
    print("SIMULACIÓN DEL MODELO SIR - MÉTODOS NUMÉRICOS Y ESTABILIDAD")
    print(f"Parámetros: beta = {beta:.2f}, gamma = {gamma:.2f}, R0 = {R0_basic:.2f}")
    print(f"Condiciones iniciales: S(0) = {S0:.2f}, I(0) = {I0:.2f}, R(0) = {1.0 - S0 - I0:.2f}, T = {T:.1f} días")
    print(f"Comparación: dt_fino = {dt_fine} días vs dt_moderado = {dt_moderate} días")
    print("=" * 85)

    # Referencia continua RK45
    t_ref, S_ref, I_ref, R_ref = generate_ground_truth(
        beta=beta, gamma=gamma, S0=S0, I0=I0, T=T, dt_ref=0.005
    )
    peak_idx_ref = int(np.argmax(I_ref))
    t_peak_ref = float(t_ref[peak_idx_ref])
    I_peak_ref = float(I_ref[peak_idx_ref])
    print(f"\n[Referencia RK45] Peak de infectados: I_max = {I_peak_ref:.4f} en t = {t_peak_ref:.2f} días")

    dt_experiments = [
        ("Paso fino", dt_fine),
        ("Paso moderado", dt_moderate),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex="col")

    solver_styles = {
        "Explícito": {"color": "#d62728", "ls": ":", "marker": "o", "zorder": 4},
        "Implícito": {"color": "#1f77b4", "ls": "-.", "marker": "s", "zorder": 3},
        "IMEX": {"color": "#2ca02c", "ls": "--", "marker": "^", "zorder": 5},
    }

    for row_idx, (case_name, dt) in enumerate(dt_experiments):
        print(f"\n" + "-" * 85)
        print(f"EVALUACIÓN PARA {case_name.upper()} (dt = {dt:.2f} días)")
        print("-" * 85)

        t_exp, S_exp, I_exp, R_exp = solve_sir_explicit(S0, I0, beta, gamma, dt, T)
        t_imp, S_imp, I_imp, R_imp = solve_sir_implicit(S0, I0, beta, gamma, dt, T)
        t_imex, S_imex, I_imex, R_imex = solve_sir_imex(S0, I0, beta, gamma, dt, T)

        results = {
            "Explícito": (t_exp, S_exp, I_exp, R_exp),
            "Implícito": (t_imp, S_imp, I_imp, R_imp),
            "IMEX": (t_imex, S_imex, I_imex, R_imex),
        }

        header = (
            f"{'Método':<12} | {'max |S+I+R - 1|':<17} | {'min(S)':<9} | {'min(I)':<9} | "
            f"{'I_max':<9} | {'t_peak':<8} | {'RMSE(I)':<10} | {'Comportamiento'}"
        )
        print(header)
        print("-" * len(header))

        for name, (t_m, S_m, I_m, R_m) in results.items():
            mass = S_m + I_m + R_m
            max_dev_mass = float(np.max(np.abs(mass - 1.0)))

            min_s = float(np.min(S_m))
            min_i = float(np.min(I_m))
            max_i = float(np.max(I_m))
            t_peak = float(t_m[int(np.argmax(I_m))])

            I_ref_interp = np.interp(t_m, t_ref, I_ref)
            rmse_i = float(np.sqrt(np.mean((I_m - I_ref_interp) ** 2)))

            rel_err_peak = (max_i - I_peak_ref) / I_peak_ref * 100.0

            if min_s < -1e-6 or min_i < -1e-6:
                diagnostico = "Viola no-negatividad"
            elif rel_err_peak > 10.0:
                diagnostico = f"Sobreestima peak (+{rel_err_peak:.1f}%)"
            elif max_dev_mass < 1e-12:
                diagnostico = "Conserva masa exacta"
            else:
                diagnostico = "Estable y acotado"

            print(
                f"{name:<12} | {max_dev_mass:<17.2e} | {min_s:<9.4f} | {min_i:<9.4f} | "
                f"{max_i:<9.4f} | {t_peak:<8.1f} | {rmse_i:<10.4e} | {diagnostico}"
            )

            style = solver_styles[name]
            markevery = max(1, len(t_m) // 15)

            axes[row_idx, 0].plot(
                t_m, S_m,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markevery=markevery,
                markersize=4.5,
                linewidth=1.6,
                label=name,
                zorder=style["zorder"]
            )
            axes[row_idx, 1].plot(
                t_m, I_m,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markevery=markevery,
                markersize=4.5,
                linewidth=1.6,
                label=name,
                zorder=style["zorder"]
            )
            axes[row_idx, 2].plot(
                t_m, R_m,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markevery=markevery,
                markersize=4.5,
                linewidth=1.6,
                label=name,
                zorder=style["zorder"]
            )
            axes[row_idx, 3].plot(
                t_m, mass,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markevery=markevery,
                markersize=4.5,
                linewidth=1.6,
                label=name,
                zorder=style["zorder"]
            )

        # Referencia continua
        axes[row_idx, 0].plot(t_ref, S_ref, "k-", linewidth=2.0, label="Referencia RK45", zorder=2)
        axes[row_idx, 1].plot(t_ref, I_ref, "k-", linewidth=2.0, label="Referencia RK45", zorder=2)
        axes[row_idx, 2].plot(t_ref, R_ref, "k-", linewidth=2.0, label="Referencia RK45", zorder=2)
        axes[row_idx, 3].axhline(1.0, color="k", linestyle="-", linewidth=1.5, alpha=0.7, label="Población total = 1")

        axes[row_idx, 0].set_ylabel(f"{case_name}\n(dt = {dt} d)\n\nProporción S(t)", fontsize=10)
        axes[row_idx, 1].set_ylabel("Proporción I(t)", fontsize=10)
        axes[row_idx, 2].set_ylabel("Proporción R(t)", fontsize=10)
        axes[row_idx, 3].set_ylabel("S(t) + I(t) + R(t)", fontsize=10)

        for col in range(4):
            axes[row_idx, col].grid(True, linestyle=":", alpha=0.6)
            if row_idx == 0:
                titles = [
                    "Susceptibles S(t)",
                    "Infectados I(t)",
                    "Recuperados R(t)",
                    "Conservación de Masa",
                ]
                axes[0, col].set_title(titles[col], fontsize=11, fontweight="bold", pad=8)

        axes[row_idx, 3].set_ylim(0.998, 1.002)

    for col in range(4):
        axes[1, col].set_xlabel("Tiempo t [días]", fontsize=10)
    axes[0, 1].legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        r"Comparación de esquemas numéricos en el modelo SIR ($\beta=0.4, \gamma=0.1$)",
        fontsize=13, y=0.99
    )
    fig.tight_layout()

    out_path = Path(__file__).resolve().parent / "figura_02_sir.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigura guardada en: {out_path}")


if __name__ == "__main__":
    main()
