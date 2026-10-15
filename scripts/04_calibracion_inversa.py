#!/usr/bin/env python3
"""
Script 04: Calibración Inversa de Parámetros y Análisis de Sesgo de Discretización.

Resuelve el problema inverso en el modelo epidemiológico SIR:
1. Carga / genera observaciones clínicas sintéticas con ruido.
2. Evalúa y mapea el paisaje de la función de pérdida L(beta, gamma) en 2D.
3. Calibra [beta, gamma] mediante el algoritmo Nelder-Mead registrando la
   trayectoria del símplex sobre la superficie de costo.
4. Ejecuta un experimento sistemático variando el paso temporal Delta t in [0.05, ..., 1.5]
   para cuantificar el sesgo de discretización numérica (Discretization Bias).
5. Exporta las figuras científicas:
   - scripts/figura_04_superficie_perdida.png
   - scripts/figura_04_sesgo_discretizacion.png

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
)
from src.inverse import (
    compute_loss_grid,
    calibrate_parameters,
    experiment_discretization_bias,
)
from src.plotting import (
    plot_loss_surface,
    plot_discretization_bias,
)


def main() -> None:
    # -------------------------------------------------------------------------
    # 1. Configuración de parámetros verdaderos y datos observacionales
    # -------------------------------------------------------------------------
    beta_true = 0.4
    gamma_true = 0.1
    R0_true = beta_true / gamma_true
    S0 = 0.99
    I0 = 0.01
    T = 50.0
    sigma_noise = 0.02
    seed = 42

    print("=" * 85)
    print(" LABORATORIO 4: PROBLEMA INVERSO, CALIBRACIÓN Y SESGO DE DISCRETIZACIÓN")
    print(f" Parámetros verdaderos de referencia: beta* = {beta_true:.2f}, gamma* = {gamma_true:.2f}, R0* = {R0_true:.2f}")
    print(f" Condiciones iniciales: S(0) = {S0:.2f}, I(0) = {I0:.2f}, T = {T:.1f} días")
    print("=" * 85)

    # Generación de observaciones con ruido Gaussiano
    t_ref, S_ref, I_ref, R_ref = generate_ground_truth(
        beta=beta_true, gamma=gamma_true, S0=S0, I0=I0, T=T, dt_ref=0.005
    )
    t_obs = np.arange(0.0, T + 1.0, 1.0)
    I_clean = sample_observations(t_ref, I_ref, t_obs)
    I_noisy = add_gaussian_noise(I_clean, sigma=sigma_noise, seed=seed)

    # -------------------------------------------------------------------------
    # 2. Calibración inicial con Nelder-Mead (dt_sim = 0.10)
    # -------------------------------------------------------------------------
    theta_init = [0.2, 0.2]
    dt_base = 0.10
    solver_name = "imex"

    print(f"\n1. CALIBRACIÓN DE PARÁMETROS CON NELDER-MEAD (dt = {dt_base:.2f}, Solver = {solver_name.upper()}):")
    print(f"   Punto de partida theta_0: beta_0 = {theta_init[0]:.2f}, gamma_0 = {theta_init[1]:.2f}")

    calib_res = calibrate_parameters(
        theta_init=theta_init,
        t_obs=t_obs,
        I_obs=I_noisy,
        dt_sim=dt_base,
        solver_name=solver_name,
        track_simplex=True,
        S0=S0,
        I0=I0,
        maxiter=600,
        tol=1e-6
    )

    b_hat = calib_res["beta_hat"]
    g_hat = calib_res["gamma_hat"]
    r0_hat = calib_res["R0_hat"]
    err_b = abs(b_hat - beta_true) / beta_true * 100.0
    err_g = abs(g_hat - gamma_true) / gamma_true * 100.0
    err_r0 = abs(r0_hat - R0_true) / R0_true * 100.0

    print(f"   - Estado: {'Convergencia Exitosa' if calib_res['success'] else 'No Convergió'}")
    print(f"   - Iteraciones evaluadas: {calib_res['iterations']}")
    print(f"   - Pérdida final MSE: {calib_res['loss_val']:.6e}")
    print(f"   - beta_hat  = {b_hat:.4f} (Error: {err_b:.2f}%)")
    print(f"   - gamma_hat = {g_hat:.4f} (Error: {err_g:.2f}%)")
    print(f"   - R0_hat    = {r0_hat:.4f} (Error: {err_r0:.2f}%)")

    # -------------------------------------------------------------------------
    # 3. Mapeo de la superficie de pérdida 2D
    # -------------------------------------------------------------------------
    print("\n2. EVALUANDO REJILLA BIDIMENSIONAL DE PÉRDIDA L(beta, gamma)...")
    beta_vals = np.linspace(0.12, 0.75, 45)
    gamma_vals = np.linspace(0.02, 0.28, 45)

    L_grid = compute_loss_grid(
        beta_vals=beta_vals,
        gamma_vals=gamma_vals,
        t_obs=t_obs,
        I_obs=I_noisy,
        dt_sim=dt_base,
        solver_name=solver_name,
        S0=S0,
        I0=I0
    )

    out_dir = BASE_DIR / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Graficar y guardar superficie de pérdida
    fig_surf, ax_surf = plot_loss_surface(
        beta_vals=beta_vals,
        gamma_vals=gamma_vals,
        L_grid=L_grid,
        theta_true=[beta_true, gamma_true],
        theta_hat=calib_res["theta_hat"],
        simplex_history=calib_res["history"]
    )
    path_surf = out_dir / "figura_04_superficie_perdida.png"
    fig_surf.savefig(path_surf, dpi=300)
    plt.close(fig_surf)
    print(f"   [OK] Superficie de costo guardada en: {path_surf}")

    # -------------------------------------------------------------------------
    # 4. Experimento de Sesgo de Discretización Numérica
    # -------------------------------------------------------------------------
    dt_experiments = [0.05, 0.10, 0.25, 0.50, 0.80, 1.00, 1.50]
    print(f"\n3. EXPERIMENTO DE SESGO DE DISCRETIZACIÓN (Delta t in {dt_experiments}):")
    print("   Calibrando parámetros para cada tamaño de paso temporal...")

    bias_res = experiment_discretization_bias(
        dt_values=dt_experiments,
        t_obs=t_obs,
        I_obs=I_noisy,
        theta_init=theta_init,
        solver_name=solver_name,
        S0=S0,
        I0=I0
    )

    print("\nTABLA COMPARATIVA DE PARÁMETROS CALIBRADOS SEGÚN EL PASO TEMPORAL:")
    print("-" * 85)
    header = f"{'Delta t':<9} | {'beta_hat':<10} | {'gamma_hat':<10} | {'R0_hat':<10} | {'Error R0 (%)':<13} | {'Pérdida MSE':<12} | {'Convergencia'}"
    print(header)
    print("-" * len(header))

    for dt, b, g, r0, l_val, succ in zip(
        bias_res["dt"],
        bias_res["beta_hat"],
        bias_res["gamma_hat"],
        bias_res["R0_hat"],
        bias_res["loss"],
        bias_res["success"]
    ):
        err_rel_r0 = abs(r0 - R0_true) / R0_true * 100.0
        status_str = "Éxito" if succ else "Fallo"
        print(
            f"{dt:<9.2f} | {b:<10.4f} | {g:<10.4f} | {r0:<10.4f} | "
            f"{err_rel_r0:<13.2f} | {l_val:<12.5e} | {status_str}"
        )
    print("-" * 85)
    print("OBSERVACIÓN MATEMÁTICA FUNDAMENTAL:")
    print("A medida que Delta t aumenta de 0.05 a 1.50, el estimador beta_hat se sesga sistemáticamente")
    print(f"hacia arriba ({bias_res['beta_hat'][0]:.4f} -> {bias_res['beta_hat'][-1]:.4f}) para compensar")
    print(f"el error de truncamiento local del integrador numérico, desviando R0 de {bias_res['R0_hat'][0]:.2f} a {bias_res['R0_hat'][-1]:.2f}.")

    # Graficar y guardar sesgo de discretización
    fig_bias, axes_bias = plot_discretization_bias(
        bias_results=bias_res,
        theta_true=(beta_true, gamma_true)
    )
    path_bias = out_dir / "figura_04_sesgo_discretizacion.png"
    fig_bias.savefig(path_bias, dpi=300)
    plt.close(fig_bias)
    print(f"\n[OK] Gráfico de sesgo de discretización guardado en: {path_bias}")
    print("=" * 85)


if __name__ == "__main__":
    main()
