"""
Contains custom exceptions
for ML algorithms
"""


class NotFittedError(Exception):
    """
    Exception class to raise if a model is used before fitting.
    """
    pass