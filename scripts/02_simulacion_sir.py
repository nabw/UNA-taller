#!/usr/bin/env python3
"""
Script 02: Simulación del Problema Directo SIR y Conservación de Masa.

Compara la integración numérica del sistema epidemiológico:
    dS/dt = - beta * S * I
    dI/dt =   beta * S * I - gamma * I
    dR/dt =   gamma * I

Esquemas numéricos evaluados:
1. Euler Explícito
2. Euler Implícito
3. Esquema IMEX (Implícito-Explícito Conservativo)
4. Solución de Referencia Continua (RK45 adaptativo)

Regímenes temporales evaluados:
- Paso moderado / estable: dt = 0.5
- Paso grande / crítico: dt = 4.0 (o dt = 3.0), donde el esquema explícito
  sufre distorsión severa del pico epidémico y oscilaciones espurias.

Invariante física evaluada:
    S(t) + I(t) + R(t) = 1.0  (Conservación estricta de la población total)

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

from src.solvers import (
    solve_sir_explicit,
    solve_sir_implicit,
    solve_sir_imex,
)
from src.synthetic import generate_ground_truth


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Parámetros epidemiológicos y temporales
    # -------------------------------------------------------------------------
    beta = 0.4
    gamma = 0.1
    S0 = 0.99
    I0 = 0.01
    T = 50.0
    R0_basic = beta / gamma  # R_0 = 4.0 (umbral epidémico supercrítico)

    dt_stable = 0.5
    dt_large = 6.0

    print("=" * 85)
    print(" LABORATORIO 2: SISTEMA EPIDEMIOLÓGICO SIR - PROBLEMA DIRECTO Y CONSERVACIÓN")
    print(" Modelo de Compartimentos / Grafos: [S] --(beta*S*I)--> [I] --(gamma*I)--> [R]")
    print(" Conservación de masa: d/dt(S + I + R) = 0 ==> S(t) + I(t) + R(t) = 1.0")
    print(f" Parámetros: beta = {beta:.2f}, gamma = {gamma:.2f}, R0 básico = {R0_basic:.2f}")
    print(f" Condiciones iniciales: S(0) = {S0:.2f}, I(0) = {I0:.2f}, R(0) = {1.0 - S0 - I0:.2f}, T = {T:.1f}")
    print(f" Evaluación: dt_estable = {dt_stable} vs dt_extremo = {dt_large}")
    print("=" * 85)

    # Verdad de campo continua mediante RK45 de alta precisión
    t_ref, S_ref, I_ref, R_ref = generate_ground_truth(
        beta=beta, gamma=gamma, S0=S0, I0=I0, T=T, dt_ref=0.005
    )
    peak_idx_ref = int(np.argmax(I_ref))
    t_peak_ref = float(t_ref[peak_idx_ref])
    I_peak_ref = float(I_ref[peak_idx_ref])
    print(f"\n[Referencia RK45] Pico epidémico exacto: I_max = {I_peak_ref:.4f} en t = {t_peak_ref:.2f} días")

    dt_experiments = [
        ("Paso Estable", dt_stable),
        ("Paso Grande / Crítico", dt_large),
    ]

    # Figura con 2 filas (para dt_stable y dt_large) y 4 columnas (S, I, R, Total S+I+R)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), sharex="col")

    solver_styles = {
        "Explícito": {"color": "#d62728", "ls": ":", "marker": "o", "zorder": 4},
        "Implícito": {"color": "#1f77b4", "ls": "-.", "marker": "s", "zorder": 3},
        "IMEX": {"color": "#2ca02c", "ls": "--", "marker": "^", "zorder": 5},
    }

    for row_idx, (case_name, dt) in enumerate(dt_experiments):
        print(f"\n" + "-" * 85)
        print(f" EVALUACIÓN PARA {case_name.upper()} (dt = {dt:.2f} días)")
        print("-" * 85)

        # Resolver numéricamente con los tres esquemas
        t_exp, S_exp, I_exp, R_exp = solve_sir_explicit(S0, I0, beta, gamma, dt, T)
        t_imp, S_imp, I_imp, R_imp = solve_sir_implicit(S0, I0, beta, gamma, dt, T)
        t_imex, S_imex, I_imex, R_imex = solve_sir_imex(S0, I0, beta, gamma, dt, T)

        results = {
            "Explícito": (t_exp, S_exp, I_exp, R_exp),
            "Implícito": (t_imp, S_imp, I_imp, R_imp),
            "IMEX": (t_imex, S_imex, I_imex, R_imex),
        }

        header = (
            f"{'Esquema':<12} | {'max |S+I+R - 1|':<17} | {'min(S)':<9} | {'min(I)':<9} | "
            f"{'I_max':<9} | {'t_pico':<8} | {'RMSE(I)':<10} | {'Propiedad Físico-Numérica'}"
        )
        print(header)
        print("-" * len(header))

        for name, (t_m, S_m, I_m, R_m) in results.items():
            # Invariante de masa total S + I + R
            mass = S_m + I_m + R_m
            max_dev_mass = float(np.max(np.abs(mass - 1.0)))

            # Valores extremos
            min_s = float(np.min(S_m))
            min_i = float(np.min(I_m))
            max_i = float(np.max(I_m))
            t_peak = float(t_m[int(np.argmax(I_m))])

            # Error RMSE contra la referencia continua interpolada
            I_ref_interp = np.interp(t_m, t_ref, I_ref)
            rmse_i = float(np.sqrt(np.mean((I_m - I_ref_interp) ** 2)))

            # Diagnóstico de propiedades
            if min_s < -1e-6 or min_i < -1e-6:
                diagnostico = "VIOLA no-negatividad"
            elif max_i > I_peak_ref * 1.15:
                diagnostico = "Sobreestimación severa del pico"
            elif max_dev_mass < 1e-12:
                diagnostico = "Conserva masa exacta al límite máquina"
            else:
                diagnostico = "Estable y acotado"

            print(
                f"{name:<12} | {max_dev_mass:<17.2e} | {min_s:<9.4f} | {min_i:<9.4f} | "
                f"{max_i:<9.4f} | {t_peak:<8.1f} | {rmse_i:<10.4e} | {diagnostico}"
            )

            style = solver_styles[name]
            markevery = max(1, len(t_m) // 15)

            # Graficar en la fila correspondiente
            axes[row_idx, 0].plot(
                t_m, S_m,
                color=style["color"],
                linestyle=style["ls"],
                marker=style["marker"],
                markevery=markevery,
                markersize=4.5,
                linewidth=1.7,
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
                linewidth=1.7,
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
                linewidth=1.7,
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
                linewidth=1.7,
                label=name,
                zorder=style["zorder"]
            )

        # Agregar curva continua de referencia en los paneles de S, I, R
        axes[row_idx, 0].plot(t_ref, S_ref, color="black", linestyle="-", linewidth=1.8, label="RK45 Ref", zorder=2)
        axes[row_idx, 1].plot(t_ref, I_ref, color="black", linestyle="-", linewidth=1.8, label="RK45 Ref", zorder=2)
        axes[row_idx, 2].plot(t_ref, R_ref, color="black", linestyle="-", linewidth=1.8, label="RK45 Ref", zorder=2)
        axes[row_idx, 3].axhline(1.0, color="black", linestyle="-", linewidth=1.2, label="Masa = 1.0", zorder=2)

        # Títulos y etiquetas de subplots
        row_title_suffix = f" ($\\Delta t = {dt}$ días)"
        axes[row_idx, 0].set_title(f"Susceptibles $S(t)${row_title_suffix}", fontsize=10)
        axes[row_idx, 1].set_title(f"Infectados $I(t)${row_title_suffix}", fontsize=10)
        axes[row_idx, 2].set_title(f"Recuperados $R(t)${row_title_suffix}", fontsize=10)
        axes[row_idx, 3].set_title(f"Masa Total $S+I+R${row_title_suffix}", fontsize=10)

        for col in range(4):
            axes[row_idx, col].grid(True, linestyle=":", alpha=0.6)
            axes[row_idx, col].set_ylabel("Proporción" if col < 3 else "Suma total", fontsize=9)
            if row_idx == 0:
                axes[row_idx, col].legend(fontsize=8, loc="best")

    for col in range(4):
        axes[1, col].set_xlabel("Tiempo $t$ [días]", fontsize=10)

    # Ajuste de escala para el panel de conservación
    axes[0, 3].set_ylim(0.999, 1.001)
    axes[1, 3].set_ylim(0.999, 1.001)

    fig.suptitle(
        "Dinámica del Modelo Epidemiológico SIR: Comparación de Esquemas y Conservación de Masa",
        fontsize=13,
        y=0.99
    )
    fig.tight_layout()

    out_dir = BASE_DIR / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figura_02_sir.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 85)
    print(f"[OK] Gráfico exportado exitosamente a: {out_path}")
    print("=" * 85)


if __name__ == "__main__":
    main()
