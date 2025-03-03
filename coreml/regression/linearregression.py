"""
Abstract Base class for Linear Regression Models
"""


from abc import abstractmethod


class LinearRegression:
    """
    Abstract class for simple linear regression 
    and multiple linear regression
    """

    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def pred(self): # return predicted y given X
        pass
    
