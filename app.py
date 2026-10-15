"""
Aplicación Interactiva Streamlit: Laboratorio de Discretización de Sistemas Dinámicos
y Problemas Inversos en Ciencia de Datos.

Módulos interactivos:
1. Ecuación Logística 1D (Estabilidad, bifurcaciones y esquema IMEX).
2. Problema Directo SIR (Esquemas explícito, implícito, IMEX y conservación de masa).
3. Generación de Datos Sintéticos (Modelado de ruido observacional Gaussiano y Poisson).
4. Problema Inverso y Sesgo Numérico (Superficie de pérdida, Nelder-Mead y Discretization Bias).

Universidad Nacional (UNA), Costa Rica.
Carrera de Ingeniería Matemática en Ciencia de Datos.
"""

import os
import sys
from pathlib import Path

# Configuración del entorno de Matplotlib para servidores headless
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Asegurar importación desde la raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.solvers import (
    solve_logistic_exact,
    solve_logistic_explicit,
    solve_logistic_implicit,
    solve_logistic_imex,
    solve_sir_explicit,
    solve_sir_implicit,
    solve_sir_imex,
)
from src.synthetic import (
    generate_ground_truth,
    sample_observations,
    add_gaussian_noise,
    add_poisson_noise,
)
from src.inverse import (
    compute_loss_grid,
    calibrate_parameters,
    experiment_discretization_bias,
)
from src.plotting import (
    plot_logistic,
    plot_sir_comparison,
    plot_synthetic_observations,
    plot_loss_surface,
    plot_discretization_bias,
)


# Configuración global de la página
st.set_page_config(
    page_title="Discretización & Problemas Inversos",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def module_logistic():
    st.header("1. Ecuación Logística 1D: Estabilidad Numérica y Esquema IMEX")
    st.markdown(
        r"""
        La ecuación logística continua modela el crecimiento poblacional saturado:
        $$\frac{dx}{dt} = C \, x \, (1 - x), \quad x(0) = x_0 \in (0, 1)$$
        Posee un atractor asintótico estable en $x^* = 1$.
        Sin embargo, al discretizar con un paso temporal $\Delta t$, el método de **Euler Explícito**
        se vuelve linealmente inestable cuando:
        $$\Delta t \cdot C > 2 \implies \Delta t > \frac{2}{C}$$
        provocando oscilaciones espurias, rebasamiento y pérdida de la no-negatividad física.
        El esquema **IMEX no estándar** linealiza el término no lineal:
        $$\frac{x_{n+1} - x_n}{\Delta t} = C \, x_n - C \, x_n \, x_{n+1} \implies x_{n+1} = \frac{x_n (1 + \Delta t \, C)}{1 + \Delta t \, C \, x_n}$$
        garantizando estabilidad y no-negatividad incondicional para cualquier $\Delta t > 0$.
        """
    )

    col_ctrl, col_info = st.columns([1, 2])

    with col_ctrl:
        st.subheader("Parámetros del Sistema")
        C = st.slider("Tasa intrínseca $C$", min_value=0.2, max_value=4.0, value=2.0, step=0.1)
        x0 = st.slider("Condición inicial $x_0$", min_value=0.01, max_value=0.40, value=0.05, step=0.01)
        dt = st.slider("Paso temporal $\\Delta t$", min_value=0.02, max_value=1.50, value=0.10, step=0.02)
        T = st.slider("Horizonte de tiempo $T$", min_value=2.0, max_value=12.0, value=6.0, step=0.5)

    c_dt = C * dt
    dt_crit = 2.0 / C

    with col_info:
        st.subheader("Diagnóstico de Estabilidad")
        m1, m2, m3 = st.columns(3)
        m1.metric("Producto $C \\cdot \\Delta t$", f"{c_dt:.2f}")
        m2.metric("Paso Crítico $\\Delta t_{crit}$", f"{dt_crit:.2f}")
        is_unstable = c_dt > 2.0
        m3.metric(
            "Régimen Explícito",
            "Inestable / Oscilatorio" if is_unstable else "Estable",
            delta="- Peligro" if is_unstable else "+ Seguro",
            delta_color="inverse" if is_unstable else "normal"
        )

        if is_unstable:
            st.warning(
                f"⚠️ **Régimen Inestable detectado:** $C \\Delta t = {c_dt:.2f} > 2.0$. "
                "Euler Explícito exhibirá rebasamiento del atractor y oscilaciones espurias. "
                "Observe cómo IMEX y Euler Implícito preservan la estabilidad asintótica."
            )
        else:
            st.success(
                f"✅ **Régimen Estable:** $C \\Delta t = {c_dt:.2f} \\le 2.0$. "
                "Los tres esquemas convergen suavemente a la solución analítica continua."
            )

    # Simulación numérica
    t_fine = np.linspace(0.0, T, 800)
    x_fine = solve_logistic_exact(x0=x0, C=C, t=t_fine)

    t_exp, x_exp = solve_logistic_explicit(x0=x0, C=C, dt=dt, T=T)
    t_imp, x_imp = solve_logistic_implicit(x0=x0, C=C, dt=dt, T=T)
    t_imex, x_imex = solve_logistic_imex(x0=x0, C=C, dt=dt, T=T)

    # Gráfico
    fig, ax = plot_logistic(
        t_exp=t_exp, x_exp=x_exp,
        t_imp=t_imp, x_imp=x_imp,
        t_imex=t_imex, x_imex=x_imex,
        t_exact=t_fine, x_exact=x_fine,
        dt=dt
    )
    st.pyplot(fig)
    plt.close(fig)

    # Tabla de métricas
    st.subheader("Métricas de Error y Propiedades Numéricas")
    x_exact_exp = solve_logistic_exact(x0, C, t_exp)
    x_exact_imp = solve_logistic_exact(x0, C, t_imp)
    x_exact_imex = solve_logistic_exact(x0, C, t_imex)

    data_metrics = [
        {
            "Método": "Euler Explícito",
            "Error L_inf": float(np.max(np.abs(x_exp - x_exact_exp))),
            "RMSE": float(np.sqrt(np.mean((x_exp - x_exact_exp) ** 2))),
            "Mínimo x(t)": float(np.min(x_exp)),
            "Máximo x(t)": float(np.max(x_exp)),
            "Preserva [0, 1]": "Sí" if (np.min(x_exp) >= 0 and np.max(x_exp) <= 1.0001) else "No (Violado)"
        },
        {
            "Método": "Euler Implícito",
            "Error L_inf": float(np.max(np.abs(x_imp - x_exact_imp))),
            "RMSE": float(np.sqrt(np.mean((x_imp - x_exact_imp) ** 2))),
            "Mínimo x(t)": float(np.min(x_imp)),
            "Máximo x(t)": float(np.max(x_imp)),
            "Preserva [0, 1]": "Sí" if (np.min(x_imp) >= 0 and np.max(x_imp) <= 1.0001) else "No (Violado)"
        },
        {
            "Método": "IMEX Linealizado",
            "Error L_inf": float(np.max(np.abs(x_imex - x_exact_imex))),
            "RMSE": float(np.sqrt(np.mean((x_imex - x_exact_imex) ** 2))),
            "Mínimo x(t)": float(np.min(x_imex)),
            "Máximo x(t)": float(np.max(x_imex)),
            "Preserva [0, 1]": "Sí" if (np.min(x_imex) >= 0 and np.max(x_imex) <= 1.0001) else "No (Violado)"
        },
    ]
    st.table(data_metrics)


def module_sir():
    st.header("2. Problema Directo SIR: Esquemas Numéricos e Invariante de Masa")
    st.markdown(
        r"""
        El modelo epidemiológico clásico se motiva mediante un sistema de compartimentos conectados:
        $$[S] \xrightarrow{\quad \beta \, S \, I \quad} [I] \xrightarrow{\quad \gamma \, I \quad} [R]$$
        Por balance de flujos (ley de acción de masas para la infección y tasa de recuperación constante):
        $$\frac{dS}{dt} = -\beta \, S \, I, \quad \frac{dI}{dt} = \beta \, S \, I - \gamma \, I, \quad \frac{dR}{dt} = \gamma \, I$$
        Propiedades físicas fundamentales:
        1. **Conservación de masa**: $\frac{d}{dt}(S + I + R) = -\beta SI + (\beta SI - \gamma I) + \gamma I = 0 \implies S(t) + I(t) + R(t) = 1.0$.
        2. **Positividad**: $S(t) \ge 0, \; I(t) \ge 0, \; R(t) \ge 0$.
        3. **Número Reproductivo Básico**: $\mathcal{R}_0 = \frac{\beta}{\gamma}$. Si $\mathcal{R}_0 > 1$, ocurre un brote epidémico.
        """
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Parámetros Epidemiológicos")
        beta = st.slider("Tasa de transmisión $\\beta$", min_value=0.05, max_value=1.0, value=0.40, step=0.05)
        gamma = st.slider("Tasa de recuperación $\\gamma$", min_value=0.02, max_value=0.5, value=0.10, step=0.02)
        S0 = st.slider("Susceptibles iniciales $S_0$", min_value=0.85, max_value=0.999, value=0.99, step=0.01)
        I0 = st.slider("Infectados iniciales $I_0$", min_value=0.001, max_value=0.05, value=0.01, step=0.005)
        dt = st.slider("Paso temporal $\\Delta t$ [días]", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        T = st.slider("Duración $T$ [días]", min_value=20.0, max_value=100.0, value=50.0, step=5.0)

    r0_basic = beta / gamma
    with col2:
        st.subheader("Indicadores Epidemiológicos y Físicos")
        c_r0, c_peak, c_inv = st.columns(3)
        c_r0.metric("Número Reproductivo $\\mathcal{R}_0$", f"{r0_basic:.2f}", delta="Epidémico" if r0_basic > 1.0 else "Contenido")

    # Solución de referencia continua RK45
    t_ref, S_ref, I_ref, R_ref = generate_ground_truth(beta, gamma, S0, I0, T, dt_ref=0.005)
    i_max_ref = float(np.max(I_ref))
    t_peak_ref = float(t_ref[int(np.argmax(I_ref))])
    c_peak.metric("Pico Teórico RK45", f"{i_max_ref:.2%}", f"Día {t_peak_ref:.1f}")

    # Simulaciones con los tres esquemas
    t_exp, S_exp, I_exp, R_exp = solve_sir_explicit(S0, I0, beta, gamma, dt, T)
    t_imp, S_imp, I_imp, R_imp = solve_sir_implicit(S0, I0, beta, gamma, dt, T)
    t_imex, S_imex, I_imex, R_imex = solve_sir_imex(S0, I0, beta, gamma, dt, T)

    results_dict = {
        "Referencia": (t_ref, S_ref, I_ref, R_ref),
        "Explícito": (t_exp, S_exp, I_exp, R_exp),
        "Implícito": (t_imp, S_imp, I_imp, R_imp),
        "IMEX": (t_imex, S_imex, I_imex, R_imex),
    }

    dev_exp = float(np.max(np.abs(S_exp + I_exp + R_exp - 1.0)))
    dev_imex = float(np.max(np.abs(S_imex + I_imex + R_imex - 1.0)))
    c_inv.metric("Error Masa IMEX", f"{dev_imex:.1e}", "Exacta")

    # Gráfico 2x2
    fig, axes = plot_sir_comparison(results_dict, title=f"Dinámica SIR Comparada ($\\Delta t = {dt}$ días)")
    st.pyplot(fig)
    plt.close(fig)

    # Alertas si explícito viola no-negatividad o sobreestima
    min_s_exp = float(np.min(S_exp))
    min_i_exp = float(np.min(I_exp))
    max_i_exp = float(np.max(I_exp))
    if min_s_exp < -1e-5 or min_i_exp < -1e-5:
        st.error(
            f"❌ **Euler Explícito violó la no-negatividad:** $\\min(S) = {min_s_exp:.4f}, \\min(I) = {min_i_exp:.4f}$. "
            "Poblaciones negativas carecen de sentido físico."
        )
    elif max_i_exp > i_max_ref * 1.15:
        st.warning(
            f"⚠️ **Distorsión del pico en Euler Explícito:** $I_{{max}} = {max_i_exp:.2%}$ vs "
            f"Referencia $I_{{max}} = {i_max_ref:.2%}$ (Sobreestimación de {abs(max_i_exp - i_max_ref)/i_max_ref * 100:.1f}%)."
        )


def module_synthetic():
    st.header("3. Generación de Datos Sintéticos y Ruido Observacional")
    st.markdown(
        r"""
        En aplicaciones clínicas reales no observamos trayectorias analíticas continuas, sino
        conteos discretos afectados por incertidumbre observacional y error de muestreo.
        Aquí simulamos:
        1. **Verdad de Campo (Ground Truth)**: Obtenida mediante integración RK45 de alta precisión.
        2. **Ruido Gaussiano Aditivo Proyectado**:
           $$I_{\text{gauss}}(t_k) = \max\left(0, \, I(t_k) + \varepsilon_k\right), \quad \varepsilon_k \sim \mathcal{N}(0, \sigma^2)$$
           Modela errores de medición homocedásticos (varianza uniforme constante).
        3. **Ruido Poissoniano Discreto Escalado**:
           $$C_k \sim \text{Poisson}\left(N_{\text{pop}} \, I(t_k)\right), \quad I_{\text{poiss}}(t_k) = \frac{C_k}{N_{\text{pop}}}$$
           Modela el proceso estocástico de notificación de casos con varianza heterocedástica:
           $\operatorname{Var}(I) = \frac{I}{N_{\text{pop}}}$ (mayor dispersión en el pico epidémico).
        """
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Controles de Ruido")
        sigma = st.slider("Desv. Est. Ruido Gaussiano $\\sigma$", min_value=0.005, max_value=0.06, value=0.02, step=0.005)
        N_pop = st.number_input("Población total $N_{pop}$ (Poisson)", min_value=500, max_value=100000, value=10000, step=500)
        seed = st.number_input("Semilla pseudoaleatoria", min_value=0, max_value=9999, value=42, step=1)
        T = st.slider("Horizonte temporal [días]", min_value=20.0, max_value=80.0, value=50.0, step=5.0)

    # Generación de datos
    t_ref, S_ref, I_ref, R_ref = generate_ground_truth(beta=0.4, gamma=0.1, S0=0.99, I0=0.01, T=T, dt_ref=0.005)
    t_obs = np.arange(0.0, T + 1.0, 1.0)
    I_clean = sample_observations(t_ref, I_ref, t_obs)
    I_gauss = add_gaussian_noise(I_clean, sigma=sigma, seed=seed)
    I_poiss = add_poisson_noise(I_clean, N_pop=int(N_pop), seed=seed)

    res_gauss = I_gauss - I_clean
    res_poiss = I_poiss - I_clean
    rmse_g = float(np.sqrt(np.mean(res_gauss ** 2)))
    rmse_p = float(np.sqrt(np.mean(res_poiss ** 2)))
    total_cases = int(np.sum(I_poiss * N_pop))

    with col2:
        st.subheader("Estadísticas Observacionales")
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE Gaussiano", f"{rmse_g:.4f}", f"$\\sigma = {sigma:.3f}$")
        m2.metric("RMSE Poisson", f"{rmse_p:.4f}", f"$N = {N_pop:,}$")
        m3.metric("Total Casos Notificados", f"{total_cases:,}", "Acumulados")

    # Gráfico
    fig, ax = plot_synthetic_observations(t_obs, I_clean, I_gauss, I_poiss)
    st.pyplot(fig)
    plt.close(fig)

    # Gráfico de residuos
    st.subheader("Análisis de Residuos Observacionales (Homocedasticidad vs Heterocedasticidad)")
    fig_res, ax_res = plt.subplots(figsize=(10, 3.5))
    ax_res.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax_res.fill_between(t_obs, -2 * sigma, 2 * sigma, color="#1f77b4", alpha=0.15, label="Banda Gaussiana $\\pm 2\\sigma$")
    ax_res.stem(t_obs - 0.15, res_gauss, linefmt="C0-", markerfmt="C0o", basefmt="none", label="Residuos Gauss")
    ax_res.stem(t_obs + 0.15, res_poiss, linefmt="C3-", markerfmt="C3^", basefmt="none", label="Residuos Poisson")
    ax_res.set_xlabel("Tiempo $t$ [días]")
    ax_res.set_ylabel("Residuo $I_{obs} - I_{clean}$")
    ax_res.grid(True, linestyle=":", alpha=0.5)
    ax_res.legend(loc="upper right", fontsize=9)
    fig_res.tight_layout()
    st.pyplot(fig_res)
    plt.close(fig_res)


def module_inverse():
    st.header("4. Problema Inverso, Superficie de Pérdida y Sesgo Numérico")
    st.markdown(
        r"""
        El problema inverso consiste en estimar el vector de parámetros $\theta = (\beta, \gamma)$
        a partir de observaciones ruidosas de infectados $\{t_k, I_{\text{obs}}(t_k)\}_{k=1}^M$.
        Definimos la función de pérdida cuadrática por mínimos cuadrados:
        $$\mathcal{L}(\theta) = \sum_{k=1}^M \left( I_{\text{sim}}(t_k; \, \theta, \Delta t) - I_{\text{obs}}(t_k) \right)^2$$
        
        ### Fenómeno Clave: Sesgo de Discretización (Discretization Bias)
        Al resolver el problema directo con un paso $\Delta t > 0$, el optimizador encuentra el estimador:
        $$\hat{\theta}(\Delta t) = \arg\min_{\theta} \mathcal{L}(\theta; \Delta t)$$
        El estimador $\hat{\theta}(\Delta t)$ **se desvía sistemáticamente** del parámetro continuo real $\theta^*$
        para que la solución numérica artificial coincida con los datos, compensando el error local de truncamiento
        del integrador numérico.
        """
    )

    # Generar observaciones sintéticas de referencia
    beta_true = 0.40
    gamma_true = 0.10
    r0_true = beta_true / gamma_true
    T = 50.0

    t_ref, S_ref, I_ref, R_ref = generate_ground_truth(beta_true, gamma_true, 0.99, 0.01, T)
    t_obs = np.arange(0.0, T + 1.0, 1.0)
    I_clean = sample_observations(t_ref, I_ref, t_obs)
    I_noisy = add_gaussian_noise(I_clean, sigma=0.02, seed=42)

    col_cfg1, col_cfg2 = st.columns([1, 1])
    with col_cfg1:
        st.subheader("Configuración de Calibración Nelder-Mead")
        beta_init = st.slider("Punto inicial $\\beta_0$", min_value=0.08, max_value=0.70, value=0.20, step=0.02)
        gamma_init = st.slider("Punto inicial $\\gamma_0$", min_value=0.03, max_value=0.30, value=0.20, step=0.01)
        dt_calib = st.select_slider(
            "Paso temporal de simulación $\\Delta t_{sim}$",
            options=[0.05, 0.10, 0.25, 0.50, 0.80, 1.00, 1.50],
            value=0.10
        )
        solver_name = st.selectbox("Esquema de simulación", ["imex", "explicit"], index=0)

    # Calibrar con los parámetros seleccionados
    calib = calibrate_parameters(
        theta_init=[beta_init, gamma_init],
        t_obs=t_obs,
        I_obs=I_noisy,
        dt_sim=dt_calib,
        solver_name=solver_name,
        track_simplex=True,
        maxiter=500
    )

    b_hat = calib["beta_hat"]
    g_hat = calib["gamma_hat"]
    r0_hat = calib["R0_hat"]

    with col_cfg2:
        st.subheader("Resultados de la Calibración")
        m1, m2, m3 = st.columns(3)
        m1.metric("$\\hat{\\beta}$ Estimado", f"{b_hat:.4f}", f"Real: {beta_true:.2f}")
        m2.metric("$\\hat{\\gamma}$ Estimado", f"{g_hat:.4f}", f"Real: {gamma_true:.2f}")
        m3.metric("$\\hat{\\mathcal{R}}_0$ Estimado", f"{r0_hat:.4f}", f"Real: {r0_true:.2f}")

        st.info(
            f"**Algoritmo Nelder-Mead:** Convergencia en **{calib['iterations']} iteraciones** con pérdida final "
            f"$\\mathcal{{L}} = {calib['loss_val']:.5e}$. Error relativo en $\\mathcal{{R}}_0$: **{abs(r0_hat - r0_true)/r0_true * 100:.2f}%**."
        )

    # Visualización de la Superficie de Pérdida
    st.subheader("1. Superficie de Pérdida 2D y Trayectoria del Símplex")
    with st.spinner("Evaluando paisaje de pérdida..."):
        beta_vals = np.linspace(0.12, 0.75, 38)
        gamma_vals = np.linspace(0.02, 0.28, 38)
        L_grid = compute_loss_grid(
            beta_vals=beta_vals,
            gamma_vals=gamma_vals,
            t_obs=t_obs,
            I_obs=I_noisy,
            dt_sim=dt_calib,
            solver_name=solver_name
        )

        fig_surf, ax_surf = plot_loss_surface(
            beta_vals=beta_vals,
            gamma_vals=gamma_vals,
            L_grid=L_grid,
            theta_true=[beta_true, gamma_true],
            theta_hat=calib["theta_hat"],
            simplex_history=calib["history"]
        )
        st.pyplot(fig_surf)
        plt.close(fig_surf)

    # Experimento de Sesgo de Discretización
    st.subheader("2. Demostración Sistemática del Sesgo de Discretización (Discretization Bias)")
    st.markdown(
        r"""
        Al recalibrar los parámetros para distintos pasos temporales $\Delta t \in [0.05, 1.50]$, observe cómo
        el estimador $\hat{\beta}(\Delta t)$ y el número reproductivo $\hat{\mathcal{R}}_0(\Delta t)$ crecen artificialmente
        a medida que la malla temporal se ensancha, a pesar de que los datos provienen del mismo sistema dinámico continuo.
        """
    )

    with st.spinner("Ejecutando calibraciones en función de Delta t..."):
        dt_list = [0.05, 0.10, 0.25, 0.50, 0.80, 1.00, 1.50]
        bias_res = experiment_discretization_bias(
            dt_values=dt_list,
            t_obs=t_obs,
            I_obs=I_noisy,
            theta_init=[0.2, 0.2],
            solver_name=solver_name
        )

        fig_bias, axes_bias = plot_discretization_bias(bias_res, theta_true=(beta_true, gamma_true))
        st.pyplot(fig_bias)
        plt.close(fig_bias)

    # Tabla resumen del sesgo
    table_rows = []
    for dt, b, g, r0, l_val in zip(
        bias_res["dt"],
        bias_res["beta_hat"],
        bias_res["gamma_hat"],
        bias_res["R0_hat"],
        bias_res["loss"]
    ):
        table_rows.append({
            "Delta t": f"{dt:.2f}",
            "beta_hat": f"{b:.4f}",
            "gamma_hat": f"{g:.4f}",
            "R0_hat": f"{r0:.4f}",
            "Error en R0 (%)": f"{abs(r0 - r0_true) / r0_true * 100:.2f}%",
            "Pérdida MSE": f"{l_val:.5e}"
        })
    st.table(table_rows)


def main():
    st.sidebar.title("🧬 Laboratorio Numérico")
    st.sidebar.markdown(
        """
        **Universidad Nacional (UNA)**
        *Ingeniería Matemática en Ciencia de Datos*
        """
    )
    st.sidebar.markdown("---")

    module = st.sidebar.radio(
        "Seleccione el Módulo:",
        [
            "1D: Ecuación Logística (IMEX vs Explícito vs Implícito)",
            "Problema Directo SIR",
            "Generación de Datos Sintéticos",
            "Problema Inverso y Sesgo Numérico",
        ],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Autores:** Cátedra de Computación Científica & Modelado Dinámico
        """
    )

    if module == "1D: Ecuación Logística (IMEX vs Explícito vs Implícito)":
        module_logistic()
    elif module == "Problema Directo SIR":
        module_sir()
    elif module == "Generación de Datos Sintéticos":
        module_synthetic()
    elif module == "Problema Inverso y Sesgo Numérico":
        module_inverse()


if __name__ == "__main__":
    main()
