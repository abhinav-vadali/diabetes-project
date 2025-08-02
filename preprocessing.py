import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_data(path_name: str) -> pd.DataFrame:
    """Loads data from a CSV file into a DataFrame

    Args:
        path_name: A string

    Returns: A DataFrame containing the data

    """
    return pd.read_csv(path_name)


def isolate_target(data: pd.DataFrame, target: str) -> pd.DataFrame + pd.Series:
    """Splits the DataFrame into a Series containing the target variable and a DataFrame containing the rest of the data

    Args:
        data: A DataFrame of the data being used
        target: A string representing the name of the variable being predicted

    Returns:

    """
    y = data[target]
    X = data.drop(columns=target)
    return X, y


def split_data(X_scaled: pd.DataFrame, y: pd.Series) -> pd.DataFrame + pd.DataFrame + pd.Series + pd.Series:
    """Splits the data into training data and testing data

    Args:
        X_scaled: A DataFrame containing the input variables
        y: A Series containing the data of the target variable

    Returns: Two DataFrames with training and testing input data and two Series with training and testing output data

    """
    train_X, test_X, train_y, test_y = train_test_split(X_scaled, y, random_state=1)
    return train_X, test_X, train_y, test_y


def encode_categorical(data: pd.DataFrame) -> pd.DataFrame:
    """Encodes categorical data using one-hot encoding

    Args:
        data: The DataFrame with categorial and numerical data

    Returns: The same DataFrame with categorical data encoded into numerical data through one-hot encoding

    """
    columns = data.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False)
    columns_encoded = encoder.fit_transform(data[columns])
    columns_as_df = pd.DataFrame(columns_encoded, columns=encoder.get_feature_names_out(columns), index=data.index)
    data_encoded = pd.concat([data, columns_as_df], axis=1).drop(columns, axis=1)
    return data_encoded


def scale_data(features: list) -> pd.DataFrame:
    """Uses a Standard Scaler to scale extremes in the data

    Args:
        features: a list of the features of the data (column names)

    Returns: A DataFrame containing the data after being scaled

    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled
