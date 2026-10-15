#!/usr/bin/env python3
"""
Script 01: Dinámica y Estabilidad Numérica en la Ecuación Logística 1D.

Compara el comportamiento numérico de:
1. Euler Explícito
2. Euler Implícito
3. Esquema IMEX Linealizado
4. Solución Analítica Exacta

Bajo dos regímenes de discretización temporal:
- Paso pequeño / asintótico: dt = 0.1 (estable, dt * C = 0.2 < 2.0)
- Paso crítico / grande: dt = 1.1 (inestable para explícito, dt * C = 2.2 > 2.0)

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
    solve_logistic_exact,
    solve_logistic_explicit,
    solve_logistic_implicit,
    solve_logistic_imex,
)


def compute_metrics(
    t_num: np.ndarray,
    x_num: np.ndarray,
    C: float,
    x0: float
) -> dict:
    """Calcula errores y propiedades numéricas frente a la solución exacta."""
    x_exact = solve_logistic_exact(x0=x0, C=C, t=t_num)
    errors = np.abs(x_num - x_exact)
    linf_error = float(np.max(errors))
    rmse_error = float(np.sqrt(np.mean(errors ** 2)))
    min_val = float(np.min(x_num))
    max_val = float(np.max(x_num))
    is_non_negative = bool(min_val >= -1e-12)
    # Detección de sobrepaso del atractor asintótico x* = 1
    overshoot = bool(max_val > 1.0 + 1e-6)

    # Detección de oscilaciones espurias (cambios de signo en derivadas discretas cerca del atractor)
    dx = np.diff(x_num)
    sign_changes = int(np.sum(dx[:-1] * dx[1:] < -1e-10))
    oscillatory = bool(sign_changes > 0 or overshoot)

    return {
        "linf": linf_error,
        "rmse": rmse_error,
        "min": min_val,
        "max": max_val,
        "non_negative": is_non_negative,
        "overshoot": overshoot,
        "oscillatory": oscillatory,
    }


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Configuración de parámetros
    # -------------------------------------------------------------------------
    C = 2.0
    x0 = 0.05
    T = 6.0
    dt_small = 0.1
    dt_large = 1.3

    print("=" * 80)
    print(" LABORATORIO 1: ECUACIÓN LOGÍSTICA 1D - COMPARACIÓN DE ESQUEMAS")
    print(" dot{x} = C * x * (1 - x)")
    print(f" Parámetros: C = {C:.2f}, x(0) = {x0:.2f}, T = {T:.2f}")
    print(f" Límite lineal de estabilidad para Euler Explícito: dt < 2 / C = {2.0 / C:.2f}")
    print(f" Evaluación: dt_pequeño = {dt_small} vs dt_grande = {dt_large}")
    print("=" * 80)

    # Malla temporal de alta resolución para la solución exacta continua
    t_fine = np.linspace(0.0, T, 1000)
    x_fine = solve_logistic_exact(x0=x0, C=C, t=t_fine)

    dt_cases = [
        ("Paso Pequeño (Estable)", dt_small),
        ("Paso Grande / Crítico", dt_large),
    ]

    # Preparar figura de dos paneles comparativos
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for idx, (label_case, dt) in enumerate(dt_cases):
        ax = axes[idx]
        c_dt = C * dt

        print(f"\n--- CASO {idx + 1}: {label_case} | dt = {dt:.2f} (C * dt = {c_dt:.2f}) ---")

        # Integración numérica con los 3 esquemas
        t_exp, x_exp = solve_logistic_explicit(x0=x0, C=C, dt=dt, T=T)
        t_imp, x_imp = solve_logistic_implicit(x0=x0, C=C, dt=dt, T=T)
        t_imex, x_imex = solve_logistic_imex(x0=x0, C=C, dt=dt, T=T)

        methods = [
            ("Euler Explícito", t_exp, x_exp, "#d62728", ":", "o"),
            ("Euler Implícito", t_imp, x_imp, "#1f77b4", "-.", "s"),
            ("IMEX Linealizado", t_imex, x_imex, "#2ca02c", "--", "^"),
        ]

        # Imprimir tabla de métricas en consola
        header = f"{'Método':<18} | {'Error L_inf':<12} | {'RMSE':<12} | {'min(x)':<10} | {'max(x)':<10} | {'No-negativo':<11} | {'Comportamiento'}"
        print(header)
        print("-" * len(header))

        for name, t_m, x_m, color, ls, marker in methods:
            m = compute_metrics(t_m, x_m, C=C, x0=x0)
            if m["oscillatory"]:
                comp = "Oscilaciones espurias / Rebasamiento"
            else:
                comp = "Monótono estable"

            print(
                f"{name:<18} | {m['linf']:<12.4e} | {m['rmse']:<12.4e} | "
                f"{m['min']:<10.4f} | {m['max']:<10.4f} | "
                f"{str(m['non_negative']):<11} | {comp}"
            )

            ax.plot(
                t_m, x_m,
                color=color,
                linestyle=ls,
                marker=marker,
                markersize=5,
                linewidth=1.7,
                label=name,
                zorder=4 if "IMEX" in name else 3
            )

        # Curva de referencia analítica continua
        ax.plot(
            t_fine, x_fine,
            color="black",
            linestyle="-",
            linewidth=2.0,
            label="Solución Exacta",
            zorder=2
        )

        # Asíntota del atractor x* = 1
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

        ax.set_title(f"{label_case}: $\\Delta t = {dt}$ ($C \\Delta t = {c_dt:.1f}$)", fontsize=11)
        ax.set_xlabel("Tiempo $t$", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Población normalizada $x(t)$", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9, loc="lower right")

    axes[0].set_ylim(-0.05, 1.25)
    fig.suptitle(
        "Ecuación Logística: Comparación de Estabilidad (Explícito vs Implícito vs IMEX)",
        fontsize=13,
        y=0.98
    )
    fig.tight_layout()

    # Guardar figura
    out_dir = BASE_DIR / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "figura_01_logistica.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 80)
    print(f"[OK] Gráfico exportado exitosamente a: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
