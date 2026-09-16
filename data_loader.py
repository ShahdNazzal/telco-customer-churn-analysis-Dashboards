"""
Data loading utilities for the Telco Customer Churn project.

Handles downloading the dataset from Kaggle and loading it into
a pandas DataFrame.
"""

import os

import kagglehub
import pandas as pd


def download_dataset() -> str:
    """
    Download the Telco Customer Churn dataset from Kaggle.

    Returns
    -------
    str
        Local path to the folder containing the downloaded dataset.
    """
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    print(f"Dataset downloaded to: {path}")
    return path


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Load the Telco Customer Churn CSV into a DataFrame.

    Parameters
    ----------
    dataset_path : str
        Folder returned by `download_dataset`.

    Returns
    -------
    pd.DataFrame
        Raw, unprocessed customer churn dataset.
    """
    csv_path = os.path.join(dataset_path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df
