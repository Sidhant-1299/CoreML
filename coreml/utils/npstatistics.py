"""
Contains statistical funcitons implemented from scratch
in NumPy for vectorization and efficiency 
"""

import numpy as np


def mean(x: np.ndarray) -> float :
    """
    Parameters:
    -----------

    x = array-like

    Returns:
    ---------
    the mean of an array
    ∑x / n
    """

    return np.mean(x)

def covariance(x:np.ndarray, y:np.ndarray) -> float:
    """
    Returns the covariance of two equal sized arrays
    ∑(x-x̄)(y-ȳ)

    Parameters:
    x = array-like
    y = array-like
    """

    if len(x) != len(y):
        raise ValueError("To calculate covariance, the two arrays must be of the same length")

    return np.dot(x-np.mean(x),y - np.mean(y))/len(x)
    

def variance(x: np.ndarray) -> float:
    """
    Returns the variance ∑(x - x̄)²/n

    Parmeters:
    x = array-like
    """

    return np.sum((x-np.mean(x))**2)/len(x)

