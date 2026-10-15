"""
Paquete principal para el Laboratorio de Discretización de Sistemas Dinámicos y Problemas Inversos.
Universidad Nacional (UNA), Costa Rica.
Carrera de Ingeniería Matemática en Ciencia de Datos.
"""

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
    loss_function,
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

__version__ = "0.1.0"

__all__ = [
    # Solvers
    "solve_logistic_exact",
    "solve_logistic_explicit",
    "solve_logistic_implicit",
    "solve_logistic_imex",
    "solve_sir_explicit",
    "solve_sir_implicit",
    "solve_sir_imex",
    # Synthetic
    "generate_ground_truth",
    "sample_observations",
    "add_gaussian_noise",
    "add_poisson_noise",
    # Inverse
    "loss_function",
    "compute_loss_grid",
    "calibrate_parameters",
    "experiment_discretization_bias",
    # Plotting
    "plot_logistic",
    "plot_sir_comparison",
    "plot_synthetic_observations",
    "plot_loss_surface",
    "plot_discretization_bias",
]
