"""
Exploratory data analysis helpers for the Telco Customer Churn dataset.

Each function returns a summary table (as a pandas object) so results
can be printed, logged, or fed into reports/notebooks.
"""

import pandas as pd


def churn_distribution(df: pd.DataFrame) -> pd.Series:
    """Return churn count and percentage distribution."""
    return df["Churn"].value_counts(normalize=True).mul(100).round(2)


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Return the churn rate (%) grouped by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Column to group by (e.g. "Contract", "InternetService").
    """
    return (df.groupby(column)["Churn"].mean() * 100).round(2)


def churn_rate_by_tenure_group(df: pd.DataFrame) -> pd.Series:
    """Return churn rate (%) grouped by Tenure_Group."""
    return (df.groupby("Tenure_Group", observed=True)["Churn"].mean() * 100).round(2)


def churn_rate_by_monthly_charges(df: pd.DataFrame) -> pd.Series:
    """Return churn rate (%) grouped by quartile of MonthlyCharges."""
    groups = pd.qcut(
        df["MonthlyCharges"], q=4, labels=["Low", "Medium", "High", "Very High"]
    )
    return (df.groupby(groups, observed=True)["Churn"].mean() * 100).round(2)
