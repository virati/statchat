"""
Example: Monte Carlo Simulation to Compare Methods

This example runs a Monte Carlo simulation to compare the performance of
different causal inference methods across multiple replications.
"""

import numpy as np
import statchat


def main():
    """Run Monte Carlo simulation comparing methods."""
    
    print("=" * 70)
    print("StatsChat: Monte Carlo Simulation Comparison")
    print("=" * 70)
    print()
    
    # Define methods to compare
    methods = {
        'Simple': lambda **kwargs: statchat.estimate_treatment_effect(
            outcomes=kwargs['outcomes'],
            treatment=kwargs['treatment']
        ),
        'PSM': statchat.propensity_score_matching,
        'IPW': statchat.inverse_probability_weighting,
        'Overlap': statchat.control_theoretic_overlap,
    }
    
    # Run simulation
    print("Running simulation with 100 replications...")
    print("(This may take a moment...)")
    print()
    
    performance = statchat.compare_methods_simulation(
        data_generator=statchat.generate_medical_data,
        methods=methods,
        n_simulations=100,
        n_samples=500,
        n_covariates=5,
        treatment_effect=2.0,
        confounding_strength=0.5,
        noise_std=1.0,
        random_state=42
    )
    
    # Display results
    print("=" * 70)
    print("Simulation Results")
    print("=" * 70)
    print()
    print(f"{'Method':<20} {'Bias':>10} {'Variance':>10} {'RMSE':>10} {'Success':>10}")
    print("-" * 70)
    
    for method_name, metrics in performance.items():
        print(f"{method_name:<20} "
              f"{metrics['bias']:>10.3f} "
              f"{metrics['variance']:>10.3f} "
              f"{metrics['rmse']:>10.3f} "
              f"{metrics['n_successful']:>10}")
    
    print()
    print("=" * 70)
    print("Interpretation:")
    print("-" * 70)
    print("- Bias: How far estimates are from the true effect on average")
    print("- Variance: How much estimates vary across replications")
    print("- RMSE: Root mean squared error (combines bias and variance)")
    print("- Success: Number of successful estimations out of 100")
    print()
    print("The overlap method typically has low bias and reasonable variance,")
    print("making it robust for causal inference in confounded settings.")
    print("=" * 70)


if __name__ == "__main__":
    main()
