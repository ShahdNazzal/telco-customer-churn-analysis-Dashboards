"""
End-to-end pipeline for the Telco Customer Churn Analysis project.

Run with:
    python main.py
"""

from src.churn_model import run_churn_pipeline
from src.cleaning import add_tenure_group, clean_data
from src.dashboard_export import build_dashboard_dataset, export_dashboard_csv
from src.data_loader import download_dataset, load_dataset
from src.segmentation import segment_customers

FINAL_THRESHOLD = 0.30
N_CLUSTERS = 3


def main():
    # 1. Load
    dataset_path = download_dataset()
    df = load_dataset(dataset_path)

    # 2. Clean
    df = clean_data(df)

    # 3. Segment customers (K-Means)
    df = segment_customers(df, n_clusters=N_CLUSTERS)

    # 4. Predict churn (Logistic Regression + Random Forest comparison)
    results = run_churn_pipeline(df, final_threshold=FINAL_THRESHOLD)
    df = results["scored_dataset"]

    # 5. Add tenure grouping for the dashboard
    df = add_tenure_group(df)

    # 6. Build final dashboard dataset and export
    dashboard_df = build_dashboard_dataset(df)
    export_dashboard_csv(dashboard_df)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
