"""Combineer cryptoprijzen en Google Trends voor ETH-machine-learning.

Deze script laadt:
- crypto_prices_clean.csv
- google_trends_clean.csv

Vervolgens worden de datasets:
- opgeschoond
- samengevoegd op datum
- gesorteerd
- aangevuld met eenvoudige target- en returnfeatures

Het resultaat wordt weggeschreven naar eth_ml_dataset.csv.
"""

import numpy as np
import pandas as pd

CRYPTO_CSV = "crypto_prices_clean.csv"
GOOGLE_CSV = "google_trends_clean.csv"
OUTPUT_CSV = "eth_ml_dataset.csv"


def load_crypto_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.loc[:, ["date", "BTC", "VIX", "ETH"]]
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df


def load_google_trends(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    return df


def merge_datasets(crypto: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(
        crypto,
        trends,
        how="inner",
        on="date",
        validate="one_to_one",
    )
    merged = merged.sort_values("date").reset_index(drop=True)

    # Eenvoudige targetvariabelen voor ETH-modellering
    merged["ETH_pct_change"] = merged["ETH"].pct_change()
    merged["ETH_log_return"] = np.log(merged["ETH"] / merged["ETH"].shift(1))
    merged["ETH_next_day"] = merged["ETH"].shift(-1)
    merged["ETH_return_next_day"] = merged["ETH_next_day"] / merged["ETH"] - 1

    merged = merged.dropna(subset=["ETH_next_day"]).reset_index(drop=True)
    return merged


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    df.to_csv(output_path, index=False)
    print(f"Saved merged dataset to: {output_path}")


if __name__ == "__main__":
    crypto_df = load_crypto_prices(CRYPTO_CSV)
    trends_df = load_google_trends(GOOGLE_CSV)

    print(f"Loaded crypto prices: {crypto_df.shape}")
    print(f"Loaded Google Trends: {trends_df.shape}")

    merged_df = merge_datasets(crypto_df, trends_df)
    print(f"Merged dataset shape: {merged_df.shape}")
    print("Columns:", merged_df.columns.tolist())
    save_dataset(merged_df, OUTPUT_CSV)
