from coreml.regression.linear_regression import LinearRegression


class MultipleLinearRegression(LinearRegression):
    def __init__(self, data, target_col):
        super().__init__(data, target_col)
        