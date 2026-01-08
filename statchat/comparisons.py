"""
Comparison utilities for evaluating causal inference methods.

This module provides functions to compare estimates from different methods,
compute bias and variance, and evaluate performance against true effects.
"""

import numpy as np
from typing import Dict, List, Union, Optional


def compute_bias(
    estimates: Union[float, np.ndarray],
    true_value: float
) -> float:
    """
    Compute the bias of an estimator.
    
    Parameters
    ----------
    estimates : float or np.ndarray
        Point estimate or array of estimates
    true_value : float
        True parameter value
        
    Returns
    -------
    float
        Bias (mean difference from true value)
    """
    estimates = np.atleast_1d(estimates)
    return np.mean(estimates) - true_value


def compute_variance(estimates: np.ndarray) -> float:
    """
    Compute the variance of an estimator.
    
    Parameters
    ----------
    estimates : np.ndarray
        Array of estimates from multiple runs
        
    Returns
    -------
    float
        Variance of the estimates
    """
    estimates = np.asarray(estimates)
    return np.var(estimates, ddof=1)


def compute_mse(
    estimates: np.ndarray,
    true_value: float
) -> float:
    """
    Compute the mean squared error (MSE) of an estimator.
    
    MSE = Bias^2 + Variance
    
    Parameters
    ----------
    estimates : np.ndarray
        Array of estimates from multiple runs
    true_value : float
        True parameter value
        
    Returns
    -------
    float
        Mean squared error
    """
    estimates = np.asarray(estimates)
    bias = compute_bias(estimates, true_value)
    variance = compute_variance(estimates)
    return bias ** 2 + variance


def compare_estimates(
    estimates_dict: Dict[str, Union[float, Dict[str, float]]],
    true_ate: Optional[float] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare treatment effect estimates from different methods.
    
    Parameters
    ----------
    estimates_dict : Dict[str, Union[float, Dict[str, float]]]
        Dictionary mapping method names to estimates or result dictionaries
    true_ate : float, optional
        True average treatment effect for bias computation
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Comparison results with statistics for each method
    """
    results = {}
    
    for method_name, estimate_data in estimates_dict.items():
        # Extract ATE from dictionary or use direct value
        if isinstance(estimate_data, dict):
            ate = estimate_data.get('ate', estimate_data.get('treatment_effect', 0.0))
            method_results = estimate_data.copy()
        else:
            ate = float(estimate_data)
            method_results = {'ate': ate}
        
        # Compute bias if true ATE is provided
        if true_ate is not None:
            method_results['bias'] = ate - true_ate
            method_results['absolute_bias'] = abs(ate - true_ate)
            method_results['relative_bias'] = (ate - true_ate) / true_ate if true_ate != 0 else np.inf
        
        results[method_name] = method_results
    
    return results


def compare_methods_simulation(
    data_generator,
    methods: Dict[str, callable],
    n_simulations: int = 100,
    **generator_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple causal inference methods via simulation.
    
    Parameters
    ----------
    data_generator : callable
        Function that generates data (should return dict with 'covariates',
        'treatment', 'outcomes', 'true_ate')
    methods : Dict[str, callable]
        Dictionary mapping method names to callable estimation functions
    n_simulations : int, optional
        Number of simulation runs (default: 100)
    **generator_kwargs
        Additional arguments to pass to data_generator
        
    Returns
    -------
    Dict[str, Dict[str, float]]
        Performance metrics (bias, variance, MSE, coverage) for each method
    """
    results = {name: [] for name in methods.keys()}
    true_ates = []
    
    for _ in range(n_simulations):
        # Generate data
        data = data_generator(**generator_kwargs)
        true_ates.append(data['true_ate'])
        
        # Apply each method
        for method_name, method_func in methods.items():
            try:
                estimate = method_func(
                    covariates=data['covariates'],
                    treatment=data['treatment'],
                    outcomes=data['outcomes']
                )
                # Extract ATE from result
                if isinstance(estimate, dict):
                    ate = estimate['ate']
                else:
                    ate = float(estimate)
                results[method_name].append(ate)
            except Exception as e:
                # Handle errors gracefully
                results[method_name].append(np.nan)
    
    # Compute performance metrics
    true_ate = np.mean(true_ates)
    performance = {}
    
    for method_name, estimates in results.items():
        estimates = np.array(estimates)
        # Remove NaN values
        valid_estimates = estimates[~np.isnan(estimates)]
        
        if len(valid_estimates) > 0:
            performance[method_name] = {
                'mean_estimate': np.mean(valid_estimates),
                'bias': compute_bias(valid_estimates, true_ate),
                'variance': compute_variance(valid_estimates),
                'mse': compute_mse(valid_estimates, true_ate),
                'rmse': np.sqrt(compute_mse(valid_estimates, true_ate)),
                'std_error': np.std(valid_estimates, ddof=1),
                'n_successful': len(valid_estimates),
                'n_failed': len(estimates) - len(valid_estimates),
            }
        else:
            performance[method_name] = {
                'mean_estimate': np.nan,
                'bias': np.nan,
                'variance': np.nan,
                'mse': np.nan,
                'rmse': np.nan,
                'std_error': np.nan,
                'n_successful': 0,
                'n_failed': len(estimates),
            }
    
    return performance
