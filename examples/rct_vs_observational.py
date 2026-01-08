"""
Example: Comparing RCT and Observational Data

This example demonstrates the difference between randomized controlled trial (RCT)
data and observational data, and shows how different methods perform on each.
"""

import numpy as np
import statchat


def analyze_data(data, data_type="RCT"):
    """Analyze data using multiple causal inference methods."""
    
    print(f"\n{data_type} Data Analysis")
    print("=" * 70)
    
    methods = {
        'Simple': statchat.estimate_treatment_effect(
            outcomes=data['outcomes'],
            treatment=data['treatment']
        ),
        'PSM': statchat.propensity_score_matching(
            covariates=data['covariates'],
            treatment=data['treatment'],
            outcomes=data['outcomes']
        ),
        'IPW': statchat.inverse_probability_weighting(
            covariates=data['covariates'],
            treatment=data['treatment'],
            outcomes=data['outcomes']
        ),
        'Overlap': statchat.control_theoretic_overlap(
            covariates=data['covariates'],
            treatment=data['treatment'],
            outcomes=data['outcomes']
        ),
    }
    
    print(f"\nTrue ATE: {data['true_ate']:.3f}")
    print(f"\n{'Method':<20} {'Estimate':>10} {'Bias':>10}")
    print("-" * 70)
    
    for method_name, result in methods.items():
        ate = result['ate']
        bias = ate - data['true_ate']
        print(f"{method_name:<20} {ate:>10.3f} {bias:>10.3f}")
    
    return methods


def main():
    """Main example comparing RCT and observational data."""
    
    print("=" * 70)
    print("StatsChat: RCT vs Observational Data Comparison")
    print("=" * 70)
    
    np.random.seed(42)
    true_effect = 2.0
    
    # Generate RCT data (no confounding)
    print("\n1. Generating RCT data (randomized treatment)...")
    rct_data = statchat.generate_rct_data(
        n_samples=1000,
        n_covariates=5,
        treatment_effect=true_effect,
        random_state=42
    )
    
    # Generate observational data (with confounding)
    print("2. Generating observational data (confounded treatment)...")
    obs_data = statchat.generate_observational_data(
        n_samples=1000,
        n_covariates=5,
        treatment_effect=true_effect,
        confounding_type="linear",
        random_state=43
    )
    
    # Analyze RCT data
    rct_methods = analyze_data(rct_data, "RCT")
    
    # Analyze observational data
    obs_methods = analyze_data(obs_data, "Observational")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("\nIn RCT data:")
    print("  - All methods should give similar (unbiased) estimates")
    print("  - Simple difference in means is sufficient")
    print("\nIn Observational data:")
    print("  - Simple difference is BIASED due to confounding")
    print("  - Adjustment methods (PSM, IPW, Overlap) reduce bias")
    print("  - Control theoretic overlap balances precision and robustness")
    print("=" * 70)


if __name__ == "__main__":
    main()
