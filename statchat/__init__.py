"""
statchat: Medical Statistical Approaches for Causal Inference

A basic package for medical statistical approaches focusing on causal inference
with control theoretic overlap in toy models that can be integrated and compared
with empirical data.
"""

__version__ = "0.1.0"

from .causal_inference import (
    estimate_treatment_effect,
    propensity_score_matching,
    inverse_probability_weighting,
)
from .control_theory import (
    control_theoretic_overlap,
    overlap_weights,
    compute_overlap_score,
)
from .toy_models import (
    generate_medical_data,
    generate_rct_data,
    generate_observational_data,
)
from .comparisons import (
    compare_estimates,
    compute_bias,
    compute_variance,
    compare_methods_simulation,
)

__all__ = [
    # Causal inference methods
    "estimate_treatment_effect",
    "propensity_score_matching",
    "inverse_probability_weighting",
    # Control theory methods
    "control_theoretic_overlap",
    "overlap_weights",
    "compute_overlap_score",
    # Toy model generators
    "generate_medical_data",
    "generate_rct_data",
    "generate_observational_data",
    # Comparison utilities
    "compare_estimates",
    "compute_bias",
    "compute_variance",
    "compare_methods_simulation",
]
