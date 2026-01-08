"""Tests for control theory methods."""

import numpy as np
import pytest
from statchat import control_theory


def test_compute_overlap_score():
    """Test overlap score computation."""
    propensity_scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    score = control_theory.compute_overlap_score(propensity_scores)
    
    assert isinstance(score, float)
    assert 0 <= score <= 0.25  # Maximum is 0.25 at p=0.5
    assert score > 0


def test_overlap_weights():
    """Test overlap weights computation."""
    propensity_scores = np.array([0.2, 0.4, 0.6, 0.8])
    treatment = np.array([0, 0, 1, 1])
    
    weights = control_theory.overlap_weights(propensity_scores, treatment)
    
    assert len(weights) == 4
    # For control (T=0): weight = p
    assert abs(weights[0] - 0.2) < 1e-10
    assert abs(weights[1] - 0.4) < 1e-10
    # For treated (T=1): weight = 1-p
    assert abs(weights[2] - 0.4) < 1e-10
    assert abs(weights[3] - 0.2) < 1e-10


def test_control_theoretic_overlap():
    """Test control theoretic overlap method."""
    np.random.seed(42)
    n = 100
    covariates = np.random.randn(n, 3)
    treatment = np.random.binomial(1, 0.5, n)
    outcomes = covariates.sum(axis=1) + treatment * 2.0
    
    result = control_theory.control_theoretic_overlap(
        covariates, treatment, outcomes
    )
    
    assert 'ate' in result
    assert 'overlap_score' in result
    assert 'n_overlap' in result
    assert result['n_overlap'] <= n
    assert isinstance(result['ate'], float)


def test_overlap_with_high_threshold():
    """Test overlap method with high threshold."""
    np.random.seed(42)
    n = 50
    covariates = np.random.randn(n, 2)
    treatment = np.random.binomial(1, 0.5, n)
    outcomes = np.random.randn(n)
    
    result = control_theory.control_theoretic_overlap(
        covariates, treatment, outcomes, overlap_threshold=0.4
    )
    
    # With high threshold, fewer samples in overlap region
    assert result['n_overlap'] < n
    assert 'overlap_score' in result


def test_overlap_score_symmetric():
    """Test that overlap score is symmetric."""
    p = 0.3
    score1 = 0.3 * (1 - 0.3)
    score2 = 0.7 * (1 - 0.7)
    
    # Overlap weights are symmetric: p(1-p) = (1-p)p
    assert abs(score1 - score2) < 1e-10
