"""
Abstract Base class for Linear Regression Models
"""

import numpy as np
import pandas as pd

from abc import abstractmethod

class LinearRegression:
    """
    Abstract class for simple linear regression 
    and multiple linear regression
    """

    def __init__(self, data: pd.DataFrame, target_col: str):
        # Check data is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data is not a pandas DataFrame")

        # Check target column exists in the DataFrame
        if target_col not in data.columns:
            raise ValueError(f"Target column {target_col} not in dataframe")
        

        # Initialize y (target) and X (features) from the DataFrame
        self.y = data[target_col]
        self.x = data.drop(columns=[target_col])

        

    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def summary(self):
        pass

    @abstractmethod
    def pred(self): # return predicted y given X
        pass


         