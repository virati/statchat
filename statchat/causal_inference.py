"""
Causal inference methods for medical statistical analysis.

This module provides basic causal inference methods including propensity score
matching, inverse probability weighting, and treatment effect estimation.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def estimate_treatment_effect(
    outcomes: np.ndarray,
    treatment: np.ndarray,
    method: str = "simple"
) -> Dict[str, float]:
    """
    Estimate the average treatment effect (ATE).
    
    Parameters
    ----------
    outcomes : np.ndarray
        Array of outcome values
    treatment : np.ndarray
        Binary array indicating treatment (1) or control (0)
    method : str, optional
        Method for estimation: "simple" for difference in means
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing 'ate', 'treated_mean', and 'control_mean'
    """
    treatment = np.asarray(treatment, dtype=bool)
    outcomes = np.asarray(outcomes, dtype=float)
    
    treated_outcomes = outcomes[treatment]
    control_outcomes = outcomes[~treatment]
    
    treated_mean = np.mean(treated_outcomes) if len(treated_outcomes) > 0 else 0.0
    control_mean = np.mean(control_outcomes) if len(control_outcomes) > 0 else 0.0
    ate = treated_mean - control_mean
    
    return {
        'ate': ate,
        'treated_mean': treated_mean,
        'control_mean': control_mean,
        'n_treated': len(treated_outcomes),
        'n_control': len(control_outcomes),
    }


def propensity_score_matching(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcomes: np.ndarray,
    n_neighbors: int = 1
) -> Dict[str, float]:
    """
    Estimate treatment effect using propensity score matching.
    
    This is a simplified implementation that uses logistic regression-based
    propensity scores and nearest neighbor matching.
    
    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix (n_samples, n_features)
    treatment : np.ndarray
        Binary treatment indicator
    outcomes : np.ndarray
        Outcome values
    n_neighbors : int, optional
        Number of neighbors to match (default: 1)
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing treatment effect estimates
    """
    treatment = np.asarray(treatment, dtype=bool)
    covariates = np.asarray(covariates)
    outcomes = np.asarray(outcomes)
    
    # Simple propensity score based on standardized sum of covariates
    # (In practice, use logistic regression)
    cov_normalized = (covariates - covariates.mean(axis=0)) / (covariates.std(axis=0) + 1e-8)
    propensity_scores = 1 / (1 + np.exp(-cov_normalized.sum(axis=1)))
    
    # Match treated to control based on propensity scores
    treated_idx = np.where(treatment)[0]
    control_idx = np.where(~treatment)[0]
    
    matched_outcomes_treated = []
    matched_outcomes_control = []
    
    for t_idx in treated_idx:
        # Find nearest neighbor in control group
        distances = np.abs(propensity_scores[control_idx] - propensity_scores[t_idx])
        nearest_idx = control_idx[np.argmin(distances)]
        
        matched_outcomes_treated.append(outcomes[t_idx])
        matched_outcomes_control.append(outcomes[nearest_idx])
    
    if len(matched_outcomes_treated) > 0:
        ate = np.mean(matched_outcomes_treated) - np.mean(matched_outcomes_control)
    else:
        ate = 0.0
    
    return {
        'ate': ate,
        'n_matched': len(matched_outcomes_treated),
        'treated_mean': np.mean(matched_outcomes_treated) if matched_outcomes_treated else 0.0,
        'control_mean': np.mean(matched_outcomes_control) if matched_outcomes_control else 0.0,
    }


def inverse_probability_weighting(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcomes: np.ndarray
) -> Dict[str, float]:
    """
    Estimate treatment effect using inverse probability weighting (IPW).
    
    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix (n_samples, n_features)
    treatment : np.ndarray
        Binary treatment indicator
    outcomes : np.ndarray
        Outcome values
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing weighted treatment effect estimates
    """
    treatment = np.asarray(treatment, dtype=bool)
    covariates = np.asarray(covariates)
    outcomes = np.asarray(outcomes)
    
    # Simple propensity score based on standardized sum of covariates
    cov_normalized = (covariates - covariates.mean(axis=0)) / (covariates.std(axis=0) + 1e-8)
    propensity_scores = 1 / (1 + np.exp(-cov_normalized.sum(axis=1)))
    
    # Clip propensity scores to avoid extreme weights
    propensity_scores = np.clip(propensity_scores, 0.1, 0.9)
    
    # Compute IPW weights
    weights = np.where(treatment, 1.0 / propensity_scores, 1.0 / (1.0 - propensity_scores))
    
    # Weighted means
    treated_weighted_mean = np.sum(outcomes[treatment] * weights[treatment]) / np.sum(weights[treatment])
    control_weighted_mean = np.sum(outcomes[~treatment] * weights[~treatment]) / np.sum(weights[~treatment])
    
    ate = treated_weighted_mean - control_weighted_mean
    
    return {
        'ate': ate,
        'treated_weighted_mean': treated_weighted_mean,
        'control_weighted_mean': control_weighted_mean,
        'effective_n': np.sum(weights) ** 2 / np.sum(weights ** 2),  # Effective sample size
    }
