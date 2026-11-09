"""Staged k-scale implementation using notation close to the paper."""
from .data import read_and_parse_data, load_S_i
from .neighbors import CM, RG, D, build_KNN
from .merge import gen, k_merge, Generalization
from .anonymization import compute_C_tilde, nu_k_pick, generalize_Sigma, k_scale
from .metrics import calculate_metrics, plot_spatial_accuracy_cdf

__all__ = ["read_and_parse_data", "load_S_i", "CM", "RG", "D", "build_KNN",
           "gen", "k_merge", "Generalization", "compute_C_tilde", "nu_k_pick",
           "generalize_Sigma", "k_scale", "calculate_metrics", "plot_spatial_accuracy_cdf"]

from .attacks import simulate_attacks, plot_attack_heatmap, single_point_compatible_values
__all__ += ["simulate_attacks", "plot_attack_heatmap", "single_point_compatible_values"]
