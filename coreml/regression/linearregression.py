from abc import abstractmethod
import numpy as np
from coreml.utils.npstatistics import mean,covariance,variance


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

    def __init__(self,x,y):

        self.x = x
        self.y = y

        if len(self.x) != len(self.y):
            raise ValueError("Array X and Y must be of the same length")
        
        self.x_bar = mean(x)
        self.y_bar = mean(y)
        self.b0 = None
        self.b1 = None
        self.fit = False #Flag to check if model is fit

    def calculate_b1(self):
        """
        β1 = ∑(x-x̄)(y-ȳ)/∑(x - x̄)²
        """
        return covariance(self.x,self.y)/variance(self.x)
        

    def calculate_b0(self):
        """
        β0 = ȳ - β1 * x̄

        Assumes:
        --------
        β1 = self.b1 is priorly calculated
        """
        return self.y_bar - self.b1 * self.x_bar
    
    def fit(self):
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

        return self.b0,self.b1
    
    def pred(self):
        ...
    
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
        TSS = np.sum((self.y - self.y_bar)**2) #Total sum of square
        


    def calculate_residual_standard_err(self):
        ...

    def summary(self):
        pass



class MultipleLinearRegression(LinearRegression):
    pass




if __name__=="__main__":
    print("hello world")