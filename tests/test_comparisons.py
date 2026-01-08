"""Tests for comparison utilities."""

import numpy as np
import pytest
from statchat import comparisons, toy_models


def test_compute_bias():
    """Test bias computation."""
    estimates = np.array([2.1, 1.9, 2.0, 2.2])
    true_value = 2.0
    
    bias = comparisons.compute_bias(estimates, true_value)
    assert abs(bias - 0.05) < 1e-10  # Mean is 2.05


def test_compute_variance():
    """Test variance computation."""
    estimates = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    variance = comparisons.compute_variance(estimates)
    
    # Variance of [1,2,3,4,5] is 2.5
    assert abs(variance - 2.5) < 1e-10


def test_compute_mse():
    """Test MSE computation."""
    estimates = np.array([2.0, 3.0, 4.0])
    true_value = 2.0
    
    mse = comparisons.compute_mse(estimates, true_value)
    
    # Mean estimate is 3.0, bias = 1.0
    # Variance is 1.0
    # MSE = 1.0^2 + 1.0 = 2.0
    assert abs(mse - 2.0) < 1e-10


def test_compare_estimates():
    """Test estimate comparison."""
    estimates = {
        'method1': {'ate': 2.5},
        'method2': {'ate': 2.0},
        'method3': 1.8,
    }
    
    comparison = comparisons.compare_estimates(estimates, true_ate=2.0)
    
    assert 'method1' in comparison
    assert 'method2' in comparison
    assert 'method3' in comparison
    assert abs(comparison['method1']['bias'] - 0.5) < 1e-10
    assert abs(comparison['method2']['bias'] - 0.0) < 1e-10
    assert abs(comparison['method3']['bias'] - (-0.2)) < 1e-10


def test_compare_estimates_without_true_ate():
    """Test comparison without true ATE."""
    estimates = {
        'method1': {'ate': 2.5},
        'method2': 2.0,
    }
    
    comparison = comparisons.compare_estimates(estimates)
    
    assert 'method1' in comparison
    assert 'method2' in comparison
    assert 'bias' not in comparison['method1']
    assert 'bias' not in comparison['method2']


def test_compare_methods_simulation():
    """Test simulation-based comparison."""
    
    def simple_method(**kwargs):
        """Simple difference in means estimator."""
        treated_mean = np.mean(kwargs['outcomes'][kwargs['treatment']])
        control_mean = np.mean(kwargs['outcomes'][~kwargs['treatment']])
        return {'ate': treated_mean - control_mean}
    
    methods = {
        'simple': simple_method,
    }
    
    performance = comparisons.compare_methods_simulation(
        data_generator=toy_models.generate_rct_data,
        methods=methods,
        n_simulations=10,
        n_samples=100,
        treatment_effect=2.0,
        random_state=42
    )
    
    assert 'simple' in performance
    assert 'bias' in performance['simple']
    assert 'variance' in performance['simple']
    assert 'mse' in performance['simple']
    assert performance['simple']['n_successful'] == 10


def test_single_estimate_bias():
    """Test bias computation with single estimate."""
    estimate = 2.5
    bias = comparisons.compute_bias(estimate, 2.0)
    assert bias == 0.5
