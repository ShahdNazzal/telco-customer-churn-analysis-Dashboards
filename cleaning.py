"""
Data cleaning and feature engineering for the Telco Customer Churn dataset.
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw Telco Customer Churn dataset.

    - Fixes blank `TotalCharges` values (new customers with tenure = 0)
    - Drops the non-predictive `customerID` column
    - Encodes `Churn` as binary (Yes=1, No=0)
    - Removes duplicate rows

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as loaded from the Kaggle CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    df = df.copy()

    df["TotalCharges"] = df["TotalCharges"].replace(" ", "0")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])

    df = df.drop(columns=["customerID"])

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    before = df.shape[0]
    df = df.drop_duplicates()
    removed = before - df.shape[0]
    if removed:
        print(f"Removed {removed} duplicate rows")

    print(f"Cleaned dataset shape: {df.shape}")
    print(f"Churn rate: {round(df['Churn'].mean() * 100, 1)}%")

    return df


def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a categorical `Tenure_Group` column, binned from `tenure`.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing a `tenure` column.

    Returns
    -------
    pd.DataFrame
        Dataset with an added `Tenure_Group` column.
    """
    df = df.copy()
    df["Tenure_Group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )
    return df
