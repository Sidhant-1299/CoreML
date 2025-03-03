import numpy as np

from typing import Tuple

from coreml.utils.npstatistics import mean,covariance,variance
from coreml.utils.exceptions import NotFittedError
from coreml.regression.linearregression import LinearRegression


class SimpleLinearRegression(LinearRegression):
    """
    Simple Linear Regression assuming a linear
    relationship with a single feature

    x is the independent variable 
    y is the dependent variable

    Calculates:
    -----------
    Coefficients β0 and β1
    
    Parameters:
    -----------
    x = array-like
    y = array-like

    Raises:
    -------
    ValueError
        If `X` and `y` have different lengths or if `X` or `y` is empty.
    """

    def __init__(self, x:np.ndarray, y:np.ndarray):

        self.x = x
        self.y = y

        if len(self.x) != len(self.y):
            raise ValueError("Array X and Y must be of the same length")
        
        self.x_bar = mean(x)
        self.y_bar = mean(y)
        self.b0 = None
        self.b1 = None
        self.fitted = False #Flag to check if model is fit

    def calculate_b1(self) -> float:
        """
        β1 = ∑(x-x̄)(y-ȳ)/∑(x - x̄)²
        """
        return covariance(self.x,self.y)/variance(self.x)
        

    def calculate_b0(self) -> float:
        """
        β0 = ȳ - β1 * x̄

        Assumes:
        --------
        β1 = self.b1 is priorly calculated
        """
        return self.y_bar - self.b1 * self.x_bar
    
    def fit(self) -> None:
        """
        Fits the simple linear regression model to the provided training data.

        This method calculates the slope (coefficient) and intercept of the regression line
        using the least squares method. The results are stored as instance attributes
        for use in prediction.

        Returns:
        --------
        Coefficients β0 and β1

        """

        self.b1 = self.calculate_b1() #b1 first becaue b0 depends on b1
        self.b0 = self.calculate_b0()
        self.fitted = True # Once these values have been calculated, model has been fit

    
    def model_fitted(self) -> None:
        if not self.fitted: #raise NotFittedError if model has not been fit yet
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using 'predict'."
            )
    
    def pred(self):
        """
        Prediction of dependent variable Y

        This method returns an array of predicted Y given the X variable
        
        Assumes:
        --------
        Model has already been fit

        Returns:
        --------
        array-like
        """

        self.model_fitted() #Raises NotFittedError if model has not been fit
        return self.b1*self.x + self.b0
        
    def calcluate_total_sum_of_squares(self) -> int:
        """
        Calculates the total sum of squares
        It is hte variability in Y before regression is performed
        TSS is the inherent variability in Y
        TSS = ∑(y-ȳ)²

        Returns:
        -------
        int
        """

        TSS = np.sum((self.y - self.y_bar)**2)
        return TSS

    def calculate_residual_sum_of_squares(self) -> int:
        """
        RSS is the  variability left after regression is performed
        RSS = ∑(y -  ŷ)²

        Assumes:
        --------
        Model has been fit
        Returns:
        --------
        int
        """

        self.model_fitted() #checks if model is already fit except raises NotFittedError
        RSS = np.sum(self.y - self.pred()**2) 
        return RSS
    
    def calculate_rsquare(self):
        """
        Measures the proportion of variability  in Y that can be explained by X
        R² = (TSS - RSS)/TSS
        TSS is the total variability in Y before regression is performed
        TSS is the inherent variability in Y
        RSS is the  variability left after regression is performed
        TSS - RSS is the variability that can be explained by the regression

        Returns:
        --------
        rsquare = (TSS - RSS)/TSS

        """
        TSS = self.calcluate_total_sum_of_squares()
        RSS = self.calculate_residual_sum_of_squares()
        return (1 - RSS/TSS)
        


    def calculate_residual_standard_err(self):
        """
        """

    def summary(self):
        pass





if __name__=="__main__":
    x = np.arange(2,10,1)
    y = 2* x
    s = SimpleLinearRegression(x,y)
    s.fit()
    print(x,y,s.pred(),s.b0,s.b1)