"""
Contains statistical funcitons implemented from scratch
in NumPy for vectorization and efficiency 
"""

import numpy as np


def mean(x):
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

def covariance(x,y):
    """
    Returns the covariance of two equal sized arrays
    ∑(x-x̄)(y-ȳ)

    Parameters:
    x = array-like
    y = array-like
    """

    if len(x) == len(y):
        return np.dot(x*y)/len(x)
    
    return ValueError("To calculate covariance, the two arrays must be of the same lenght")

def variance(x):
    """
    Returns the variance ∑(x - x̄)²/n

    Parmeters:
    x = array-like
    """

    return np.sum(x-np.mean(x))/len(x)

