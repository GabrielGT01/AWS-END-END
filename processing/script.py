# | filename: script.py
# | code-line-numbers: true

import os
import tarfile
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def preprocess(base_directory):
    """Load the supplied data, split it and transform it."""
    df = _read_data_from_input_csv_files(base_directory)

    # Target transformer for admission (binary classification)
    target_transformer = ColumnTransformer(
        transformers=[("admission", OrdinalEncoder(), ["admission"])],
        remainder='passthrough'
    )

    # Numeric transformer pipeline
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="mean"),
        StandardScaler(),
    )

    # Categorical transformer pipeline
    categorical_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(drop='first', sparse=False),
    )

    # Define categorical columns correctly
    cat_columns = ['gender', 'major', 'race', 'work_industry']
    
    # Define numeric columns explicitly
    num_columns = ['gpa', 'gmat', 'work_exp']

    # Combined feature transformer
    features_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, num_columns),
            ("categorical", categorical_transformer, cat_columns),
        ],
    )

    # Split data 
    df_train, df_validation, df_test = _split_data(df)

    # Save baselines
    _save_train_baseline(base_directory, df_train)
    _save_test_baseline(base_directory, df_test)

    # Transform targets
    y_train = target_transformer.fit_transform(
        df_train[['admission']]
    )
    y_validation = target_transformer.transform(
        df_validation[['admission']]
    )
    y_test = target_transformer.transform(
        df_test[['admission']]
    )

    # Drop the target column 
    df_train = df_train.drop("admission", axis=1)
    df_validation = df_validation.drop("admission", axis=1)
    df_test = df_test.drop("admission", axis=1)

    # Transform features
    X_train = features_transformer.fit_transform(df_train)
    X_validation = features_transformer.transform(df_validation)
    X_test = features_transformer.transform(df_test)

    # Save the processed data
    _save_splits(
        base_directory,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    )

    # Save the transformers
    _save_model(base_directory, target_transformer, features_transformer)

def _read_data_from_input_csv_files(base_directory):
    """Read the data from the input CSV files."""
    input_directory = Path(base_directory) / "input"
    files = list(input_directory.glob("*.csv"))

    if len(files) == 0:
        message = f"There are no CSV files in {input_directory.as_posix()}/"
        raise ValueError(message)

    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)
    return df.sample(frac=1, random_state=42)




def _split_data(df):
    """Split the data into train, validation, and test."""
    # Step 1: Split the data into train (70%) and temp (30%) with stratification
    df_train, temp = train_test_split(df, test_size=0.3, stratify=df['admission'], random_state=42)

    # Step 2: Split the temp data equally into validation (15%) and test (15%) with stratification
    df_validation, df_test = train_test_split(temp, test_size=0.5, stratify=temp['admission'], random_state=42)


    return df_train, df_validation, df_test


def _save_train_baseline(base_directory, df_train):
    """Save the untransformed training data to disk.

    We will need the training data to compute a baseline durning monitoring to
    determine the quality of the data that the model receives
    when deployed and the target wont be supplied by the users.
    """
    baseline_path = Path(base_directory) / "train-baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)

    df = df_train.copy().dropna()

    # To compute the data quality baseline, we don't need the
    # target variable, so we'll drop it from the dataframe.
    df = df.drop("admission", axis=1)

    df.to_csv(baseline_path / "train-baseline.csv", header=True, index=False)


def _save_test_baseline(base_directory, df_test):
    """Save the untransformed test data to disk.

    We will need the test data to compute a baseline to
    determine the quality of the model predictions when deployed.
    """
    baseline_path = Path(base_directory) / "test-baseline"
    baseline_path.mkdir(parents=True, exist_ok=True)

    df = df_test.copy().dropna()

    # We'll use the test baseline to generate predictions later,
    # and we can't have a header line because the model won't be
    # able to make a prediction for it.
    df.to_csv(baseline_path / "test-baseline.csv", header=False, index=False)


def _save_splits(
    base_directory,
    X_train,  # noqa: N803
    y_train,
    X_validation,  # noqa: N803
    y_validation,
    X_test,  # noqa: N803
    y_test,
):
    """Save data splits to disk.

    This function concatenates the transformed features
    and the target variable, and saves each one of the split
    sets to disk.
    """
    train = np.concatenate((X_train, y_train), axis=1)
    validation = np.concatenate((X_validation, y_validation), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv",
        header=False,
        index=False,
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


def _save_model(base_directory, target_transformer, features_transformer):
    """Save the Scikit-Learn transformation pipelines.

    This function creates a model.tar.gz file that
    contains the two transformation pipelines we built
    to transform the data.
    """
    #create a temporary folder to store, that disappears after the model is moved
    with tempfile.TemporaryDirectory() as directory:
        joblib.dump(target_transformer, Path(directory) / "target.joblib")
        joblib.dump(features_transformer, Path(directory) / "features.joblib")
        
        #create a model folder in aws directory 
        model_path = Path(base_directory) / "model"
        model_path.mkdir(parents=True, exist_ok=True)

        with tarfile.open(f"{(model_path / 'model.tar.gz').as_posix()}", "w:gz") as tar:
             #copy the model from the temp folder to the model path
            tar.add(Path(directory) / "target.joblib", arcname="target.joblib")
            tar.add(
                Path(directory) / "features.joblib", arcname="features.joblib",
            )


if __name__ == "__main__":
    preprocess(base_directory="/opt/ml/processing")
