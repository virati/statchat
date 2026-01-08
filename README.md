# statchat

A basic package for medical statistical approaches for causal inference with control theoretic overlap in toy models.

## Overview

`statchat` provides tools for causal inference in medical statistical analysis, specifically focusing on control theoretic overlap methods. The package includes:

- **Causal Inference Methods**: Propensity score matching, inverse probability weighting, and basic treatment effect estimation
- **Control Theoretic Overlap**: Advanced weighting methods that balance precision and generalizability
- **Toy Model Generators**: Create synthetic medical data (RCT and observational) for testing and comparison
- **Comparison Utilities**: Tools to compare methods and evaluate performance

## Installation

```bash
pip install -e .
```

Or install with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import statchat
import numpy as np

# Generate synthetic medical data with confounding
data = statchat.generate_medical_data(
    n_samples=1000,
    treatment_effect=2.5,
    confounding_strength=0.6,
    random_state=42
)

# Estimate treatment effect using control theoretic overlap
result = statchat.control_theoretic_overlap(
    covariates=data['covariates'],
    treatment=data['treatment'],
    outcomes=data['outcomes']
)

print(f"Estimated ATE: {result['ate']:.3f}")
print(f"True ATE: {data['true_ate']:.3f}")
print(f"Overlap Score: {result['overlap_score']:.3f}")
```

## Features

### Causal Inference Methods

- `estimate_treatment_effect`: Simple difference in means estimator
- `propensity_score_matching`: Match treated and control units based on propensity scores
- `inverse_probability_weighting`: Weight units by inverse of treatment probability

### Control Theoretic Overlap

- `control_theoretic_overlap`: Estimate treatment effects with overlap weighting
- `overlap_weights`: Compute overlap weights for balancing treatment/control groups
- `compute_overlap_score`: Measure covariate overlap quality

### Toy Model Generators

- `generate_medical_data`: General medical data with configurable confounding
- `generate_rct_data`: Randomized controlled trial data (no confounding)
- `generate_observational_data`: Observational data with various confounding patterns

### Comparison Utilities

- `compare_estimates`: Compare estimates from different methods
- `compute_bias`: Calculate estimator bias
- `compute_variance`: Calculate estimator variance
- `compare_methods_simulation`: Run Monte Carlo simulations to compare methods

## Examples

See the `examples/` directory for detailed usage examples:

- `basic_usage.py`: Compare different causal inference methods
- `rct_vs_observational.py`: Contrast RCT and observational data
- `monte_carlo_simulation.py`: Run simulations to evaluate methods

Run an example:

```bash
python examples/basic_usage.py
```

## Testing

Run the test suite:

```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details.

## References

The control theoretic overlap approach is inspired by recent advances in causal inference that focus on balancing internal validity (precision) with external validity (generalizability) by weighting observations based on covariate overlap between treatment and control groups.
