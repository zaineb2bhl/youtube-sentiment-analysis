"""Phase 2 - Téléchargement du dataset Reddit Sentiment Analysis."""
from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/Himanshu-1703/"
    "reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
)
ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "reddit.csv"
LABELS = {-1: "négatif", 0: "neutre", 1: "positif"}


def download(url: str = DATA_URL, dest: Path = RAW_PATH, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"[download] Fichier déjà présent : {dest}")
        return dest
    print(f"[download] Téléchargement depuis {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"[download] Sauvegardé dans {dest} ({len(resp.content) / 1024:.0f} Ko)")
    return dest


def print_stats(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"\nColonnes : {list(df.columns)}")
    print(f"Nombre de commentaires : {len(df)}")
    print(f"Valeurs manquantes :\n{df.isna().sum()}")
    label_col = "category" if "category" in df.columns else "label"
    counts = df[label_col].value_counts().sort_index()
    print("\nDistribution des labels :")
    for lbl, n in counts.items():
        print(f"  {int(lbl):>2} ({LABELS.get(int(lbl), '?'):8}) : {n:6d}  ({n / len(df):.1%})")
    return df


if __name__ == "__main__":
    print_stats(download())
