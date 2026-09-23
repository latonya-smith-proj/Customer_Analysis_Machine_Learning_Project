"""Clean and feature-engineer the customer personality dataset.

Ported from the original R `Codes.R` data-management section.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "customer_personality.csv"


def load_and_clean_data(input_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    # Drop missing values
    df = df.dropna()

    # Merge rare/inconsistent marital-status categories into "Single"
    df["Marital_Status"] = df["Marital_Status"].replace(
        {"Alone": "Single", "Absurd": "Single", "YOLO": "Single"}
    )

    # Recode education levels
    df["Education"] = df["Education"].replace(
        {
            "2n Cycle": "Technical School",
            "Basic": "High School",
            "Graduation": "College Graduate",
        }
    )

    # Age from birth year; drop implausible ages (3 outliers > 100 in the original data)
    df["age"] = 2023 - df["Year_Birth"]
    df = df[df["age"] < 100]

    # Consolidated campaign-acceptance target: any of 5 campaigns or the final response
    campaign_cols = [
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5", "Response",
    ]
    df["Accepted_Cmp"] = (df[campaign_cols].sum(axis=1) > 0).map({True: "Yes", False: "No"})

    # Parse enrollment date, then compute tenure in years
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], format="%d-%m-%Y")
    df["duration"] = (pd.Timestamp.now() - df["Dt_Customer"]).dt.days / 365.25

    # Combine Kidhome + Teenhome into a single "children" count
    df["children"] = df["Teenhome"] + df["Kidhome"]

    # Drop redundant / non-predictive columns
    drop_cols = [
        "ID", "Year_Birth", "Kidhome", "Teenhome", "Dt_Customer", "Recency",
        "AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
        "Z_CostContact", "Z_Revenue", "Response",
    ]
    df = df.drop(columns=drop_cols)

    # Drop income outliers
    df = df[df["Income"] < 200_000]

    # Categorical dtypes for modeling convenience
    for col in ["Education", "Marital_Status", "Complain", "Accepted_Cmp"]:
        df[col] = df[col].astype("category")

    return df.reset_index(drop=True)


def add_highest_purchase_source(df: pd.DataFrame) -> pd.DataFrame:
    """Flag each customer's dominant purchase channel (web/catalog/store)."""
    df = df.copy()

    def _source(row):
        web, cat, store = row["NumWebPurchases"], row["NumCatalogPurchases"], row["NumStorePurchases"]
        if web >= cat and web >= store:
            return "web"
        if cat > web and cat >= store:
            return "catalog"
        if store > web and store > cat:
            return "store"
        return "equal"

    df["HighestPurchaseSource"] = df.apply(_source, axis=1)
    return df


if __name__ == "__main__":
    cleaned = load_and_clean_data()
    print(f"Cleaned dataset: {cleaned.shape[0]} rows, {cleaned.shape[1]} columns")
    print(cleaned.head())
    out_path = DATA_DIR / "managed_data.csv"
    cleaned.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
