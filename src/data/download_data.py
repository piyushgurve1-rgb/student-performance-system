"""
Dataset acquisition script for the Student Performance and Dropout Risk Analysis System.
Downloads and verifies the UCI 'Predict Students' Dropout and Academic Success' dataset (ID: 697).
"""

import os
import io
import json
import hashlib
import zipfile
import urllib.request
from datetime import datetime

UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
)
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
RAW_CSV_PATH = os.path.join(RAW_DATA_DIR, "data.csv")
INFO_JSON_PATH = os.path.join(RAW_DATA_DIR, "dataset_info.json")


def compute_sha256(filepath: str) -> str:
    """Calculate the SHA-256 checksum of a file for integrity verification."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_uci_dataset(dest_csv: str = RAW_CSV_PATH, info_path: str = INFO_JSON_PATH) -> str:
    """
    Downloads the official UCI dataset archive, extracts data.csv,
    verifies dimensions, and writes an attribution metadata file.
    """
    os.makedirs(os.path.dirname(dest_csv), exist_ok=True)

    if os.path.exists(dest_csv):
        print(f"[INFO] Raw dataset already exists at: {dest_csv}")
    else:
        print(f"[INFO] Downloading official dataset archive from UCI:\n  {UCI_DATASET_URL}")
        req = urllib.request.Request(
            UCI_DATASET_URL,
            headers={"User-Agent": "StudentPerformanceProject/1.0"}
        )
        with urllib.request.urlopen(req) as response:
            zip_bytes = response.read()

        print("[INFO] Extracting 'data.csv' from zip archive...")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            if "data.csv" not in z.namelist():
                raise FileNotFoundError(
                    f"'data.csv' not found inside UCI zip archive! Found: {z.namelist()}"
                )
            z.extract("data.csv", path=os.path.dirname(dest_csv))
        print(f"[SUCCESS] Extracted raw dataset to: {dest_csv}")

    # Integrity verification
    sha256 = compute_sha256(dest_csv)

    # Basic inspection of rows and columns
    import pandas as pd
    df = pd.read_csv(dest_csv, sep=";")
    rows, cols = df.shape
    print(f"[VERIFY] Loaded raw dataset: {rows} rows, {cols} columns.")

    metadata = {
        "dataset_name": "Predict Students' Dropout and Academic Success",
        "uci_id": 697,
        "citation": (
            "Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2021). "
            "Predict Students' Dropout and Academic Success. "
            "Data in Brief / UCI Machine Learning Repository. DOI: 10.1016/j.dib.2021.107530"
        ),
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "source_url": UCI_DATASET_URL,
        "download_timestamp": datetime.utcnow().isoformat() + "Z",
        "sha256_checksum": sha256,
        "rows": rows,
        "columns": cols,
        "delimiter": ";",
        "target_column": "Target",
        "target_distribution": df["Target"].value_counts().to_dict(),
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SUCCESS] Dataset metadata saved to: {info_path}")

    return dest_csv


if __name__ == "__main__":
    download_uci_dataset()
