import pandas as pd
import numpy as np
from coreml.regression.simple_linear_regression import SimpleLinearRegression
import pytest

def test_fit_with_simple_linear_relationship():
    """Test fit with perfect linear relationship y = 2x + 3"""
    x = np.array([1, 2, 3, 4, 5])
    y = 2 * x + 3
    data = pd.DataFrame({'x': x, 'y': y})
    model = SimpleLinearRegression(data, 'y')
    model.fit()
    assert model.fitted is True
    assert pytest.approx(model.b1, 0.001) == 2.0
    assert pytest.approx(model.b0, 0.001) == 3.0


def test_fit_with_noisy_linear_relationship():
    """Test fit with noisy linear relationship"""
    np.random.seed(42)
    x = np.array([1, 2, 3, 4, 5])
    y = 1.5 * x + 2 + np.random.normal(0, 0.1, size=len(x))
    data = pd.DataFrame({'x': x, 'y': y})
    model = SimpleLinearRegression(data, 'y')
    model.fit()
    assert model.fitted is True
    assert pytest.approx(model.b1, 0.1) == 1.5  # Allowing some tolerance
    assert pytest.approx(model.b0, 0.1) == 2.0   # Allowing some tolerance


def test_fit_with_specified_feature_column():
    """Test fit when feature column is specified"""
    x = np.array([10, 20, 30, 40, 50])
    y = 0.5 * x - 2
    data = pd.DataFrame({'feature': x, 'target': y})
    model = SimpleLinearRegression(data, 'target', feature_col='feature')
    model.fit()
    assert model.fitted ==True
    assert pytest.approx(model.b1, 0.001) == 0.5
    assert pytest.approx(model.b0, 0.001) == -2.0


def test_fit_raises_error_if_not_fitted_before_pred():
    """Test that pred raises error if called before fit"""
    x = np.array([1, 2, 3])
    y = np.array([2, 4, 6])
    data = pd.DataFrame({'x': x, 'y': y})
    model = SimpleLinearRegression(data, 'y')
    with pytest.raises(ValueError, match="This model has not been fitted yet"):
        model.pred()


def test_fit_with_single_point_raises_error():
    """Test that fit with single data point raises error (due to division by zero in variance)"""
    x = np.array([1])
    y = np.array([2])
    data = pd.DataFrame({'x': x, 'y': y})
    model = SimpleLinearRegression(data, 'y')
    with pytest.raises(ZeroDivisionError):
        model.fit()


