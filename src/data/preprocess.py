"""Phase 2 - Nettoyage, analyse exploratoire et split train/test reproductible."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.utils.text_cleaning import clean_text  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "reddit.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42
TEST_SIZE = 0.2
LABEL_NAMES = {-1: "négatif", 0: "neutre", 1: "positif"}
COLORS = {-1: "#d64545", 0: "#8a8f98", 1: "#2e9e5b"}


def load_and_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns={"clean_comment": "text", "category": "label"})
    n0 = len(df)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)
    print(f"[clean] {n0} -> {len(df)} lignes (NaN, vides et doublons supprimés)")
    return df[["text", "label"]].reset_index(drop=True)


def eda(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    df = df.assign(n_words=df["text"].str.split().str.len())
    counts = df["label"].value_counts().sort_index()
    print("\n[eda] Distribution des classes :")
    print(counts.rename(LABEL_NAMES).to_string())
    print(f"\n[eda] Ratio max/min : {counts.max() / counts.min():.2f}")
    print("\n[eda] Longueur (mots) par classe :")
    print(df.groupby("label")["n_words"].describe()[["mean", "50%", "max"]].round(1).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([LABEL_NAMES[l] for l in counts.index], counts.values,
                color=[COLORS[l] for l in counts.index])
    axes[0].set_title("Distribution des classes")
    axes[0].set_ylabel("Nombre de commentaires")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v}", ha="center", va="bottom")
    for lbl in sorted(df["label"].unique()):
        axes[1].hist(df.loc[df["label"] == lbl, "n_words"].clip(upper=150), bins=50,
                     alpha=0.55, label=LABEL_NAMES[lbl], color=COLORS[lbl])
    axes[1].set_title("Longueur des commentaires (mots, tronquée à 150)")
    axes[1].set_xlabel("Nombre de mots")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "eda_distribution.png", dpi=120)
    plt.close(fig)
    print(f"[eda] Figure sauvegardée : {REPORTS_DIR / 'eda_distribution.png'}")


def split_and_save(df: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["label"],
                                   random_state=RANDOM_STATE)
    df.to_csv(PROCESSED_DIR / "reddit_clean.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    print(f"\n[split] train={len(train)}  test={len(test)}  (stratifié, random_state={RANDOM_STATE})")


if __name__ == "__main__":
    data = load_and_clean()
    eda(data)
    split_and_save(data)
