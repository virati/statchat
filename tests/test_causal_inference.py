"""Tests for causal inference methods."""

import numpy as np
import pytest
from statchat import causal_inference


def test_estimate_treatment_effect_simple():
    """Test simple treatment effect estimation."""
    outcomes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    treatment = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    
    result = causal_inference.estimate_treatment_effect(outcomes, treatment)
    
    assert 'ate' in result
    assert 'treated_mean' in result
    assert 'control_mean' in result
    assert result['treated_mean'] == 8.0
    assert result['control_mean'] == 3.0
    assert result['ate'] == 5.0


def test_propensity_score_matching():
    """Test propensity score matching."""
    np.random.seed(42)
    n = 100
    covariates = np.random.randn(n, 3)
    treatment = np.random.binomial(1, 0.5, n)
    outcomes = covariates.sum(axis=1) + treatment * 2.0
    
    result = causal_inference.propensity_score_matching(
        covariates, treatment, outcomes
    )
    
    assert 'ate' in result
    assert 'n_matched' in result
    assert result['n_matched'] > 0
    assert isinstance(result['ate'], float)


def test_inverse_probability_weighting():
    """Test inverse probability weighting."""
    np.random.seed(42)
    n = 100
    covariates = np.random.randn(n, 3)
    treatment = np.random.binomial(1, 0.5, n)
    outcomes = covariates.sum(axis=1) + treatment * 2.0
    
    result = causal_inference.inverse_probability_weighting(
        covariates, treatment, outcomes
    )
    
    assert 'ate' in result
    assert 'effective_n' in result
    assert result['effective_n'] > 0
    assert isinstance(result['ate'], float)


def test_treatment_effect_with_no_treated():
    """Test treatment effect when there are no treated units."""
    outcomes = np.array([1, 2, 3, 4, 5])
    treatment = np.array([0, 0, 0, 0, 0])
    
    result = causal_inference.estimate_treatment_effect(outcomes, treatment)
    
    assert result['treated_mean'] == 0.0
    assert result['n_treated'] == 0
    assert result['n_control'] == 5


def test_treatment_effect_with_no_control():
    """Test treatment effect when there are no control units."""
    outcomes = np.array([6, 7, 8, 9, 10])
    treatment = np.array([1, 1, 1, 1, 1])
    
    result = causal_inference.estimate_treatment_effect(outcomes, treatment)
    
    assert result['control_mean'] == 0.0
    assert result['n_treated'] == 5
    assert result['n_control'] == 0
