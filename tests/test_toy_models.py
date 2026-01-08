"""Tests for toy model generators."""

import numpy as np
import pytest
from statchat import toy_models


def test_generate_medical_data():
    """Test medical data generation."""
    data = toy_models.generate_medical_data(
        n_samples=100,
        n_covariates=5,
        treatment_effect=2.0,
        random_state=42
    )
    
    assert 'covariates' in data
    assert 'treatment' in data
    assert 'outcomes' in data
    assert 'true_ate' in data
    assert data['covariates'].shape == (100, 5)
    assert len(data['treatment']) == 100
    assert len(data['outcomes']) == 100
    assert data['true_ate'] == 2.0


def test_generate_rct_data():
    """Test RCT data generation."""
    data = toy_models.generate_rct_data(
        n_samples=200,
        n_covariates=3,
        treatment_effect=1.5,
        random_state=42
    )
    
    assert 'covariates' in data
    assert 'treatment' in data
    assert 'outcomes' in data
    assert 'true_ate' in data
    assert data['covariates'].shape == (200, 3)
    assert data['true_ate'] == 1.5
    
    # In RCT, propensity scores should be close to 0.5
    assert np.allclose(data['propensity_scores'], 0.5)


def test_generate_observational_data_linear():
    """Test observational data with linear confounding."""
    data = toy_models.generate_observational_data(
        n_samples=100,
        n_covariates=4,
        treatment_effect=3.0,
        confounding_type="linear",
        random_state=42
    )
    
    assert data['confounding_type'] == "linear"
    assert data['true_ate'] == 3.0
    assert len(data['treatment']) == 100


def test_generate_observational_data_quadratic():
    """Test observational data with quadratic confounding."""
    data = toy_models.generate_observational_data(
        n_samples=100,
        confounding_type="quadratic",
        random_state=42
    )
    
    assert data['confounding_type'] == "quadratic"


def test_generate_observational_data_threshold():
    """Test observational data with threshold confounding."""
    data = toy_models.generate_observational_data(
        n_samples=100,
        confounding_type="threshold",
        random_state=42
    )
    
    assert data['confounding_type'] == "threshold"


def test_observational_data_invalid_type():
    """Test that invalid confounding type raises error."""
    with pytest.raises(ValueError):
        toy_models.generate_observational_data(
            n_samples=100,
            confounding_type="invalid",
            random_state=42
        )


def test_data_reproducibility():
    """Test that random_state ensures reproducibility."""
    data1 = toy_models.generate_medical_data(n_samples=50, random_state=123)
    data2 = toy_models.generate_medical_data(n_samples=50, random_state=123)
    
    np.testing.assert_array_equal(data1['covariates'], data2['covariates'])
    np.testing.assert_array_equal(data1['treatment'], data2['treatment'])
    np.testing.assert_array_equal(data1['outcomes'], data2['outcomes'])
