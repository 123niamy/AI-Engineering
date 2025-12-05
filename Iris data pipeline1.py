# Iris data pipeline workflow
"""
All-in-one data engineering pipeline on the Iris dataset.

Stages:
1) Data ingestion
2) Data storage (raw)
3) Preprocessing (missing values, scaling, encoding)
4) Data integration (join with metadata)
5) Data quality and validation
6) Data governance and security (metadata logging, masking)
7) Data serving (curated CSV, summary, visualizations)
"""

import os
from datetime import datetime
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Configuration
# -----------------------------
class Config:
    # Output names only; the environment decides where artifacts are stored
    RAW_CSV = "iris_raw.csv"
    CURATED_CSV = "iris_curated.csv"
    SUMMARY_CSV = "iris_summary.csv"
    QUALITY_JSON = "iris_quality_report.json"
    GOVERNANCE_JSON = "iris_governance_metadata.json"
    PLOTS_DIR = "plots"


# -----------------------------
# Stage 1: Data Ingestion
# -----------------------------
def ingest_iris() -> pd.DataFrame:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df


# -----------------------------
# Stage 2: Data Storage (Raw)
# -----------------------------
def store_raw(df: pd.DataFrame, filename: str) -> None:
    df.to_csv(filename, index=False)


# -----------------------------
# Stage 3: Preprocessing
# - simulate missing values
# - impute
# - scale features
# - encode categorical
# -----------------------------
def simulate_missing(df: pd.DataFrame, frac: float = 0.05, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df_missing = df.copy()
    numeric_cols = df_missing.select_dtypes(include=np.number).columns.tolist()
    for col in numeric_cols:
        idx = df_missing.index[rng.choice(df_missing.index.size, size=int(frac * df_missing.index.size), replace=False)]
        df_missing.loc[idx, col] = np.nan
    return df_missing


def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean())
    return df_imputed


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df_scaled = df.copy()
    numeric_cols = df_scaled.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
    return df_scaled


def encode_species(df: pd.DataFrame) -> pd.DataFrame:
    df_encoded = df.copy()
    df_encoded["species_code"] = df_encoded["species"].astype("category").cat.codes
    return df_encoded


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df_missing = simulate_missing(df)
    df_imputed = impute_mean(df_missing)
    df_scaled = scale_features(df_imputed)
    df_encoded = encode_species(df_scaled)
    return df_encoded


# -----------------------------
# Stage 4: Data Integration
# - join with a synthetic metadata table
# -----------------------------
def integrate_metadata(df: pd.DataFrame) -> pd.DataFrame:
    species_meta = pd.DataFrame({
        "species": ["setosa", "versicolor", "virginica"],
        "description": ["Small flower", "Medium flower", "Large flower"]
    })
    return pd.merge(df, species_meta, on="species", how="left")


# -----------------------------
# Stage 5: Data Quality and Validation
# -----------------------------
def quality_validation(df: pd.DataFrame) -> dict:
    report = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "missing_values": {col: int(df[col].isna().sum()) for col in df.columns},
        "duplicates": int(df.duplicated().sum()),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    return report


def save_quality_report(report: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(report, f, indent=2)


# -----------------------------
# Stage 6: Data Governance and Security
# - schema and lineage-like metadata
# - timestamp, version, column catalog
# - masking sensitive text field
# -----------------------------
def governance_metadata(df: pd.DataFrame, stage: str) -> dict:
    meta = {
        "pipeline_stage": stage,
        "timestamp": datetime.now().isoformat(),
        "row_count": int(df.shape[0]),
        "schema": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "columns": list(df.columns),
        "provenance": {
            "source": "sklearn.datasets.load_iris",
            "transformations": [
                "simulate_missing(5%)",
                "mean_imputation",
                "standard_scaler",
                "species_encoding",
                "metadata_integration"
            ]
        }
    }
    return meta


def mask_sensitive(df: pd.DataFrame, column: str, masked_column: str = "description_masked") -> pd.DataFrame:
    df_masked = df.copy()
    if column in df_masked.columns:
        df_masked[masked_column] = df_masked[column].astype(str).apply(lambda x: "*" * len(x))
    return df_masked


def save_governance_metadata(meta: dict, filename: str) -> None:
    with open(filename, "w") as f:
        json.dump(meta, f, indent=2)


# -----------------------------
# Stage 7: Data Serving
# - save curated dataset
# - summary statistics
# - visualizations
# -----------------------------
def serve_curated(df: pd.DataFrame, curated_csv: str, summary_csv: str) -> None:
    df.to_csv(curated_csv, index=False)
    summary = df.describe(include="all")
    summary.to_csv(summary_csv)


def ensure_plots_dir(dirname: str) -> None:
    if not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)


def plot_feature_distributions(df: pd.DataFrame, plots_dir: str) -> None:
    ensure_plots_dir(plots_dir)
    plt.style.use("seaborn-v0_8")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    species_values = df["species"].unique()

    for col in numeric_cols:
        fig = plt.figure(figsize=(6, 4))
        for sp in species_values:
            sns.histplot(
                df[df["species"] == sp][col],
                label=sp,
                kde=True,
                stat="density",
                common_norm=False
            )
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"hist_{col.replace(' ', '_')}.png"))
        plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, plots_dir: str) -> None:
    ensure_plots_dir(plots_dir)
    # Use only numeric columns for correlation
    corr = df.select_dtypes(include=np.number).corr()
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
    plt.close(fig)


# -----------------------------
# Pipeline Runner
# -----------------------------
def run_pipeline():
    # 1) Ingest
    df = ingest_iris()

    # 2) Store raw
    store_raw(df, Config.RAW_CSV)

    # 3) Preprocess
    df_pre = preprocess(df)

    # 4) Integrate metadata
    df_int = integrate_metadata(df_pre)

    # 5) Quality and validation
    quality = quality_validation(df_int)
    save_quality_report(quality, Config.QUALITY_JSON)

    # 6) Governance and security
    gov_meta = governance_metadata(df_int, stage="curated")
    save_governance_metadata(gov_meta, Config.GOVERNANCE_JSON)
    df_sec = mask_sensitive(df_int, column="description")

    # 7) Serving
    serve_curated(df_sec, Config.CURATED_CSV, Config.SUMMARY_CSV)
    plot_feature_distributions(df_sec, Config.PLOTS_DIR)
    plot_correlation_heatmap(df_sec, Config.PLOTS_DIR)

    print("Completed: Iris data pipeline from ingestion to serving.")

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    run_pipeline()
