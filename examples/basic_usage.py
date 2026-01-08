"""
Example: Comparing Causal Inference Methods on Toy Medical Data

This example demonstrates how to use the statchat package to:
1. Generate synthetic medical data with confounding
2. Apply different causal inference methods
3. Compare their performance using control theoretic overlap
"""

import numpy as np
import statchat


def main():
    """Main example function demonstrating package usage."""
    
    print("=" * 70)
    print("StatsChat: Medical Statistical Causal Inference Example")
    print("=" * 70)
    print()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Generate synthetic medical data with confounding
    print("1. Generating synthetic medical data...")
    print("-" * 70)
    
    data = statchat.generate_medical_data(
        n_samples=1000,
        n_covariates=5,
        treatment_effect=2.5,
        confounding_strength=0.6,
        noise_std=1.0,
        random_state=42
    )
    
    print(f"   - Number of samples: {len(data['treatment'])}")
    print(f"   - Number of covariates: {data['covariates'].shape[1]}")
    print(f"   - True treatment effect: {data['true_ate']:.2f}")
    print(f"   - Treated patients: {np.sum(data['treatment'])}")
    print(f"   - Control patients: {np.sum(~data['treatment'])}")
    print()
    
    # 2. Apply simple difference in means (naive estimator)
    print("2. Simple difference in means (naive, biased with confounding)...")
    print("-" * 70)
    
    naive_result = statchat.estimate_treatment_effect(
        outcomes=data['outcomes'],
        treatment=data['treatment'],
        method='simple'
    )
    
    print(f"   - Estimated ATE: {naive_result['ate']:.3f}")
    print(f"   - True ATE: {data['true_ate']:.3f}")
    print(f"   - Bias: {naive_result['ate'] - data['true_ate']:.3f}")
    print()
    
    # 3. Apply propensity score matching
    print("3. Propensity Score Matching...")
    print("-" * 70)
    
    psm_result = statchat.propensity_score_matching(
        covariates=data['covariates'],
        treatment=data['treatment'],
        outcomes=data['outcomes']
    )
    
    print(f"   - Estimated ATE: {psm_result['ate']:.3f}")
    print(f"   - True ATE: {data['true_ate']:.3f}")
    print(f"   - Bias: {psm_result['ate'] - data['true_ate']:.3f}")
    print(f"   - Number of matched pairs: {psm_result['n_matched']}")
    print()
    
    # 4. Apply inverse probability weighting
    print("4. Inverse Probability Weighting (IPW)...")
    print("-" * 70)
    
    ipw_result = statchat.inverse_probability_weighting(
        covariates=data['covariates'],
        treatment=data['treatment'],
        outcomes=data['outcomes']
    )
    
    print(f"   - Estimated ATE: {ipw_result['ate']:.3f}")
    print(f"   - True ATE: {data['true_ate']:.3f}")
    print(f"   - Bias: {ipw_result['ate'] - data['true_ate']:.3f}")
    print(f"   - Effective sample size: {ipw_result['effective_n']:.1f}")
    print()
    
    # 5. Apply control theoretic overlap weighting
    print("5. Control Theoretic Overlap Weighting...")
    print("-" * 70)
    
    overlap_result = statchat.control_theoretic_overlap(
        covariates=data['covariates'],
        treatment=data['treatment'],
        outcomes=data['outcomes'],
        overlap_threshold=0.1
    )
    
    print(f"   - Estimated ATE: {overlap_result['ate']:.3f}")
    print(f"   - True ATE: {data['true_ate']:.3f}")
    print(f"   - Bias: {overlap_result['ate'] - data['true_ate']:.3f}")
    print(f"   - Overlap score: {overlap_result['overlap_score']:.4f}")
    print(f"   - Samples in overlap region: {overlap_result['n_overlap']} / {overlap_result['n_total']}")
    print(f"   - Effective sample size: {overlap_result['effective_n']:.1f}")
    print()
    
    # 6. Compare all methods
    print("6. Method Comparison Summary...")
    print("-" * 70)
    
    methods = {
        'Naive (Difference in Means)': naive_result,
        'Propensity Score Matching': psm_result,
        'Inverse Probability Weighting': ipw_result,
        'Control Theoretic Overlap': overlap_result,
    }
    
    comparison = statchat.compare_estimates(methods, true_ate=data['true_ate'])
    
    print(f"{'Method':<35} {'Estimate':>10} {'Bias':>10}")
    print("-" * 70)
    for method_name, results in comparison.items():
        print(f"{method_name:<35} {results['ate']:>10.3f} {results['bias']:>10.3f}")
    
    print()
    print("=" * 70)
    print("Conclusion:")
    print("-" * 70)
    print("The control theoretic overlap method balances precision and")
    print("generalizability by focusing on the region with good covariate")
    print("overlap between treatment and control groups.")
    print("=" * 70)


if __name__ == "__main__":
    main()
