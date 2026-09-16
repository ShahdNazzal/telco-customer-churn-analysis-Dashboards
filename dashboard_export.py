"""
Builds the final dataset used by the Power BI dashboard and exports it to CSV.
"""

import pandas as pd

DASHBOARD_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "Tenure_Group",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
    "Churn", "Predicted_Churn", "Churn_Probability", "Cluster",
]


def build_dashboard_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order the columns needed for the Power BI dashboard."""
    return df[DASHBOARD_COLUMNS].copy()


def export_dashboard_csv(
    df: pd.DataFrame, output_path: str = "data/telco_churn_dashboard.csv"
) -> str:
    """
    Export the final dashboard-ready dataset to CSV.

    Returns
    -------
    str
        Path to the exported file.
    """
    df.to_csv(output_path, index=False)
    print(f"Dashboard dataset exported to: {output_path}")
    return output_path
