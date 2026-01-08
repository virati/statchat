"""
Control theoretic approaches to causal inference.

This module implements control theoretic overlap methods for causal inference,
which balance precision and generalizability in treatment effect estimation.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def compute_overlap_score(propensity_scores: np.ndarray) -> float:
    """
    Compute the overlap score for a set of propensity scores.
    
    The overlap score measures the degree of overlap in the covariate
    distributions between treatment and control groups.
    
    Parameters
    ----------
    propensity_scores : np.ndarray
        Array of propensity scores (probabilities between 0 and 1)
        
    Returns
    -------
    float
        Overlap score (higher values indicate better overlap)
    """
    propensity_scores = np.asarray(propensity_scores)
    # Overlap score based on the harmonic mean of p(1-p)
    overlap_weights = propensity_scores * (1 - propensity_scores)
    return np.mean(overlap_weights)


def overlap_weights(
    propensity_scores: np.ndarray,
    treatment: np.ndarray
) -> np.ndarray:
    """
    Compute overlap weights for causal inference.
    
    Overlap weights balance the sample to the overlap population, where
    treatment and control groups have similar covariate distributions.
    
    Parameters
    ----------
    propensity_scores : np.ndarray
        Array of propensity scores
    treatment : np.ndarray
        Binary treatment indicator
        
    Returns
    -------
    np.ndarray
        Array of overlap weights
    """
    propensity_scores = np.asarray(propensity_scores)
    treatment = np.asarray(treatment, dtype=bool)
    
    # Overlap weights: w = p(1-p) / [p*I(T=1) + (1-p)*I(T=0)]
    # For treated: w = 1 - p
    # For control: w = p
    weights = np.where(treatment, 1 - propensity_scores, propensity_scores)
    
    return weights


def control_theoretic_overlap(
    covariates: np.ndarray,
    treatment: np.ndarray,
    outcomes: np.ndarray,
    overlap_threshold: float = 0.1
) -> Dict[str, float]:
    """
    Estimate treatment effect using control theoretic overlap weighting.
    
    This method uses overlap weights to focus on the population with good
    covariate overlap, improving the precision and robustness of estimates.
    
    Parameters
    ----------
    covariates : np.ndarray
        Covariate matrix (n_samples, n_features)
    treatment : np.ndarray
        Binary treatment indicator
    outcomes : np.ndarray
        Outcome values
    overlap_threshold : float, optional
        Minimum propensity score overlap threshold (default: 0.1)
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing treatment effect estimates with overlap weighting
    """
    treatment = np.asarray(treatment, dtype=bool)
    covariates = np.asarray(covariates)
    outcomes = np.asarray(outcomes)
    
    # Estimate propensity scores (simplified)
    cov_normalized = (covariates - covariates.mean(axis=0)) / (covariates.std(axis=0) + 1e-8)
    propensity_scores = 1 / (1 + np.exp(-cov_normalized.sum(axis=1)))
    
    # Compute overlap weights
    weights = overlap_weights(propensity_scores, treatment)
    
    # Apply overlap threshold to exclude extreme propensity scores
    valid_idx = (propensity_scores >= overlap_threshold) & (propensity_scores <= 1 - overlap_threshold)
    
    if np.sum(valid_idx) == 0:
        return {
            'ate': 0.0,
            'overlap_score': 0.0,
            'n_overlap': 0,
            'treated_mean': 0.0,
            'control_mean': 0.0,
        }
    
    # Filter to overlap region
    treatment_overlap = treatment[valid_idx]
    outcomes_overlap = outcomes[valid_idx]
    weights_overlap = weights[valid_idx]
    propensity_scores_overlap = propensity_scores[valid_idx]
    
    # Compute weighted means in overlap region
    treated_mask = treatment_overlap
    control_mask = ~treatment_overlap
    
    if np.sum(weights_overlap[treated_mask]) > 0 and np.sum(weights_overlap[control_mask]) > 0:
        treated_mean = (
            np.sum(outcomes_overlap[treated_mask] * weights_overlap[treated_mask]) /
            np.sum(weights_overlap[treated_mask])
        )
        control_mean = (
            np.sum(outcomes_overlap[control_mask] * weights_overlap[control_mask]) /
            np.sum(weights_overlap[control_mask])
        )
        ate = treated_mean - control_mean
    else:
        treated_mean = 0.0
        control_mean = 0.0
        ate = 0.0
    
    overlap_score = compute_overlap_score(propensity_scores_overlap)
    
    return {
        'ate': ate,
        'treated_mean': treated_mean,
        'control_mean': control_mean,
        'overlap_score': overlap_score,
        'n_overlap': np.sum(valid_idx),
        'n_total': len(treatment),
        'effective_n': np.sum(weights_overlap) ** 2 / np.sum(weights_overlap ** 2),  # Kish's effective sample size
    }
