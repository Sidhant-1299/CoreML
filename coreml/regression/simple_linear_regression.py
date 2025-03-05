from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from coreml.utils.npstatistics import mean,covariance,variance
from coreml.regression.linear_regression import LinearRegression


class SimpleLinearRegression(LinearRegression):
    """
    Simple Linear Regression assuming a linear
    relationship with a single feature

    Calculates:
    -----------
    Coefficients β0 and β1
    
    Parameters:
    -----------
    data: a pandas dataframe
    target_col : column name of the dependent variable

    Raises:
    -------
    
    """

    def __init__(self, data: pd.DataFrame, target_col: str, feature_col = None) -> None:
        
        #initializes x and y from parent class
        super().__init__(data,target_col)

        self.y = self.y.to_numpy()

        #for multi dimensional data with a specified X column
        if feature_col:
            self.x = data[feature_col].to_numpy()
        
        else:
            #Check if X has only one feature
            try:
                self.x = self.x.to_numpy().reshape(self.y.shape)
            except:
                raise ValueError("Simple Linear regression does not support multiple X columns")
        
        #If X and Y have unequal rows raise ValueError
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("Column shapes are of different sizes")

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
        if not self.fitted: #raise ValueError if model has not been fit yet
            raise ValueError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using 'predict'."
            )
    
    def pred(self) -> np.ndarray:
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

        self.model_fitted() #Raises ValueError if model has not been fit
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

        self.model_fitted() #checks if model is already fit except raises ValueError
        RSS = np.sum((self.y - self.pred())**2) 
        return RSS
    
    def calculate_rsquare(self) -> int:
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
        


    def calculate_residual_standard_err(self) -> int:
        """
        RSE is the average amount the response will deviate from
        the true line
        RSE = √((∑(y -  ŷ)²)/(n-2))
        RSE = √(RSS/n-2)

        
        Returns:
        --------
        int
        """

        RSS = self.calculate_residual_sum_of_squares()
        return np.sqrt((RSS/(len(self.x) - 2)))

    def summary(self) -> pd.DataFrame:
        """
        A pandas DataFrame with summary statistics
        including R square and RSE

        Returns:
        --------

        (2 x 2) pandas dataframe
        """
        Rsquare = self.calculate_rsquare()
        RSE = self.calculate_residual_standard_err()
        Quantity = ['Residual Standard Error', 'R²','β0','β1']
        Value = [RSE, Rsquare,self.b0,self.b1]
        return pd.DataFrame({'Quantity':Quantity,'Value':Value})
    
    def plot(self, figsize = (10,8)) -> None:
        #Ensure model has been fit
        self.model_fitted()

        plt.figure(figsize=figsize)
        plt.title(fr"$y = {self.b1}*X {'-' if self.b0 < 0 else '+'} {abs(self.b0)}$")
        plt.plot(self.x, self.y, 'r',label='Y')
        plt.plot(self.x, self.pred(), 'b',label='X' )
        plt.grid(True)
        plt.legend()
        plt.show()






if __name__=="__main__":
    x = np.arange(2,10,1)

    y = 2* x - 1
    data = pd.DataFrame({'x':x,'y':y})
    s = SimpleLinearRegression(data, 'y')

    s.fit()
    print(s.calcluate_total_sum_of_squares())
    print(s.calculate_residual_sum_of_squares())
    print(s.summary())
    s.plot()