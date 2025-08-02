import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def fit_model(train_X: pd.DataFrame, train_y: pd.Series) -> DecisionTreeClassifier:
    """Fits a DecisionTreeClassifier classification model to the training data.

    Args:
        train_X: A DataFrame containing the input features
        train_y: A Series containing the target label and its associated column

    Returns: A DecisionTreeClassifier model fit to the data

    """
    classifier = DecisionTreeClassifier(criterion="gini", random_state=1)
    classifier.fit(train_X, train_y)
    return classifier


