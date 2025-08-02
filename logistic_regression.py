import pandas as pd
import sklearn.linear_model
from sklearn.linear_model import LogisticRegression

def fit_model(train_X: pd.DataFrame, train_y: pd.Series) -> sklearn.linear_model:
    """Fits a LogisticRegression regression model to the training data.

        Args:
            train_X: A DataFrame containing the input features
            train_y: A Series containing the target label and its associated column

        Returns: A Logistic Regression model fit to the data

        """
    regression_model = LogisticRegression(random_state = 1, max_iter = 1500)
    regression_model.fit(train_X, train_y)
    return regression_model