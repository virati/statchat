"""
Toy model generators for medical statistical analysis.

This module provides functions to generate synthetic medical data including
randomized controlled trial (RCT) data and observational data with various
confounding patterns.
"""

import numpy as np
from typing import Dict, Tuple, Optional


def generate_medical_data(
    n_samples: int = 1000,
    n_covariates: int = 5,
    treatment_effect: float = 2.0,
    confounding_strength: float = 0.5,
    noise_std: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic medical data with confounding.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate (default: 1000)
    n_covariates : int, optional
        Number of covariates (default: 5)
    treatment_effect : float, optional
        True average treatment effect (default: 2.0)
    confounding_strength : float, optional
        Strength of confounding (0 to 1, default: 0.5)
    noise_std : float, optional
        Standard deviation of outcome noise (default: 1.0)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing 'covariates', 'treatment', 'outcomes', and 'true_ate'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate covariates from normal distribution
    covariates = np.random.randn(n_samples, n_covariates)
    
    # Covariate sum for confounding
    covariate_sum = covariates.sum(axis=1)
    
    # Treatment assignment with confounding
    treatment_propensity = 1 / (1 + np.exp(-(confounding_strength * covariate_sum)))
    treatment = np.random.binomial(1, treatment_propensity)
    
    # Generate outcomes with treatment effect and confounding
    baseline_outcome = covariate_sum + np.random.randn(n_samples) * noise_std
    outcomes = baseline_outcome + treatment * treatment_effect
    
    return {
        'covariates': covariates,
        'treatment': treatment,
        'outcomes': outcomes,
        'propensity_scores': treatment_propensity,
        'true_ate': treatment_effect,
    }


def generate_rct_data(
    n_samples: int = 1000,
    n_covariates: int = 5,
    treatment_effect: float = 2.0,
    noise_std: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic randomized controlled trial (RCT) data.
    
    In RCT data, treatment assignment is random and independent of covariates.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate (default: 1000)
    n_covariates : int, optional
        Number of covariates (default: 5)
    treatment_effect : float, optional
        True average treatment effect (default: 2.0)
    noise_std : float, optional
        Standard deviation of outcome noise (default: 1.0)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing 'covariates', 'treatment', 'outcomes', and 'true_ate'
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate covariates
    covariates = np.random.randn(n_samples, n_covariates)
    
    # Random treatment assignment (no confounding)
    treatment = np.random.binomial(1, 0.5, size=n_samples)
    
    # Generate outcomes
    covariate_sum = covariates.sum(axis=1)
    baseline_outcome = covariate_sum + np.random.randn(n_samples) * noise_std
    outcomes = baseline_outcome + treatment * treatment_effect
    
    return {
        'covariates': covariates,
        'treatment': treatment,
        'outcomes': outcomes,
        'propensity_scores': np.full(n_samples, 0.5),
        'true_ate': treatment_effect,
    }


def generate_observational_data(
    n_samples: int = 1000,
    n_covariates: int = 5,
    treatment_effect: float = 2.0,
    confounding_type: str = "linear",
    noise_std: float = 1.0,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic observational data with various confounding patterns.
    
    Parameters
    ----------
    n_samples : int, optional
        Number of samples to generate (default: 1000)
    n_covariates : int, optional
        Number of covariates (default: 5)
    treatment_effect : float, optional
        True average treatment effect (default: 2.0)
    confounding_type : str, optional
        Type of confounding: "linear", "quadratic", or "threshold" (default: "linear")
    noise_std : float, optional
        Standard deviation of outcome noise (default: 1.0)
    random_state : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing observational data with specified confounding pattern
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Generate covariates
    covariates = np.random.randn(n_samples, n_covariates)
    covariate_sum = covariates.sum(axis=1)
    
    # Treatment assignment based on confounding type
    if confounding_type == "linear":
        logit = 0.5 * covariate_sum
    elif confounding_type == "quadratic":
        logit = 0.3 * covariate_sum + 0.1 * (covariate_sum ** 2)
    elif confounding_type == "threshold":
        logit = np.where(covariate_sum > 0, 2.0, -2.0)
    else:
        raise ValueError(f"Unknown confounding_type: {confounding_type}")
    
    treatment_propensity = 1 / (1 + np.exp(-logit))
    treatment = np.random.binomial(1, treatment_propensity)
    
    # Generate outcomes with confounding
    if confounding_type == "quadratic":
        baseline_outcome = covariate_sum + 0.5 * (covariate_sum ** 2)
    else:
        baseline_outcome = covariate_sum
    
    baseline_outcome += np.random.randn(n_samples) * noise_std
    outcomes = baseline_outcome + treatment * treatment_effect
    
    return {
        'covariates': covariates,
        'treatment': treatment,
        'outcomes': outcomes,
        'propensity_scores': treatment_propensity,
        'true_ate': treatment_effect,
        'confounding_type': confounding_type,
    }
