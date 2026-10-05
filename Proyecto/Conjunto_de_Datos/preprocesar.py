"""Une y preprocesa CONDA y Jigsaw en un solo conjunto con esquema comun.

Uso (desde cualquier carpeta):
    pip install pandas numpy
    python Proyecto/Conjunto_de_Datos/preprocesar.py

Salidas en Proyecto/Conjunto_de_Datos/Unificado/:
    dataset_unificado.csv.gz   conjunto unido y limpio
    reporte_calidad.md         conteos de cada paso de limpieza

Las decisiones (mapeo de etiquetas, que se elimina y por que) estan en DECISIONES.md.
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "Unificado"
SEED = 42
VALID_FRAC = 0.25  # CONDA oficial: 26,921 train / 8,974 valid (~75/25)

CONDA_TOXIC = {"E", "I"}  # Explicit / Implicit toxicity. A (Action) y O (Other) = no toxico
JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

log = []  # lineas del reporte


def note(msg):
    print(msg)
    log.append(msg)


def clean_text(s: pd.Series) -> pd.Series:
    """Normaliza sin alterar el contenido: NFC, quita [SEPA], controla espacios. Conserva mayusculas."""
    s = s.map(lambda x: unicodedata.normalize("NFC", x))
    s = s.str.replace(r"\s*\[SEPA\]\s*", " ", regex=True)
    s = s.str.replace(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", regex=True)  # caracteres de control (no \n \t)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def load_conda() -> pd.DataFrame:
    d = BASE / "CONDA" / "datos"
    parts = []
    for split in ["train", "valid"]:  # CONDA_test.csv no trae intentClass: no sirve para entrenar/evaluar
        df = pd.read_csv(d / f"CONDA_{split}.csv")
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    note(f"CONDA: {len(df)} filas leidas (train+valid; test excluido por no tener etiquetas)")

    before = len(df)
    df = df.dropna(subset=["utterance"]).copy()
    note(f"CONDA: -{before - len(df)} sin texto")

    df["text"] = clean_text(df["utterance"].astype(str))
    before = len(df)
    df = df[df["text"] != ""].copy()
    note(f"CONDA: -{before - len(df)} vacios tras limpiar")

    out = pd.DataFrame(
        {
            "uid": "conda_" + df["Id"].astype(str),
            "source": "conda",
            "split": df["split"],
            "group_id": "conda_match_" + df["matchId"].astype(str),
            "text": df["text"],
            "is_toxic": df["intentClass"].isin(CONDA_TOXIC).astype(int),
            "conda_intent": df["intentClass"],
        }
    )
    for c in JIGSAW_LABELS:
        out[f"jigsaw_{c}"] = np.nan
    return out


def load_jigsaw() -> pd.DataFrame:
    df = pd.read_csv(BASE / "Jigsaw_Toxic_Comments" / "jigsaw_train.csv.gz")
    note(f"Jigsaw: {len(df)} filas leidas")

    df["text"] = clean_text(df["comment_text"].astype(str))
    before = len(df)
    df = df[df["text"] != ""].copy()
    note(f"Jigsaw: -{before - len(df)} vacios tras limpiar")

    any_label = df[JIGSAW_LABELS].sum(axis=1) > 0
    note(f"Jigsaw: {int((any_label & (df['toxic'] == 0)).sum())} con alguna etiqueta tóxica pero toxic=0 (se marcan is_toxic=1)")

    out = pd.DataFrame(
        {
            "uid": "jigsaw_" + df["id"].astype(str),
            "source": "jigsaw",
            "group_id": "jigsaw_" + df["id"].astype(str),
            "text": df["text"],
            "is_toxic": any_label.astype(int),
            "conda_intent": np.nan,
        }
    )
    for c in JIGSAW_LABELS:
        out[f"jigsaw_{c}"] = df[c].astype(int)

    # Jigsaw solo trae train: se crea un valid estratificado con la misma proporcion que CONDA.
    rng = np.random.default_rng(SEED)
    out["split"] = "train"
    for _, idx in out.groupby("is_toxic").groups.items():
        idx = np.array(list(idx))
        pick = rng.choice(idx, size=int(len(idx) * VALID_FRAC), replace=False)
        out.loc[pick, "split"] = "valid"
    return out


def main():
    OUT.mkdir(exist_ok=True)
    conda, jigsaw = load_conda(), load_jigsaw()
    df = pd.concat([conda, jigsaw], ignore_index=True)

    # Marcas de calidad (no se borra nada por duplicado: ver DECISIONES.md)
    key = df["text"].str.lower()
    df["is_dup_text"] = key.duplicated(keep=False).astype(int)
    lab_n = df.groupby(key)["is_toxic"].transform("nunique")
    df["is_conflict_text"] = (lab_n > 1).astype(int)
    df["n_chars"] = df["text"].str.len()
    df["n_words"] = df["text"].str.split().str.len()

    cols = ["uid", "source", "split", "group_id", "text", "is_toxic", "conda_intent",
            *[f"jigsaw_{c}" for c in JIGSAW_LABELS], "is_dup_text", "is_conflict_text", "n_chars", "n_words"]
    df = df[cols]
    df.to_csv(OUT / "dataset_unificado.csv.gz", index=False, compression="gzip")

    note("")
    note(f"UNIFICADO: {len(df)} filas")
    note("\nFilas por fuente y split:\n" + df.groupby(["source", "split"]).size().unstack().to_string())
    note("\n% toxico por fuente y split:\n" + (df.groupby(["source", "split"])["is_toxic"].mean() * 100).round(2).unstack().to_string())
    note("\nLongitud (caracteres) por fuente:\n" + df.groupby("source")["n_chars"].describe().round(1).to_string())
    note("\nLongitud (palabras) por fuente:\n" + df.groupby("source")["n_words"].describe().round(1).to_string())
    note(f"\nTextos repetidos (is_dup_text=1): {int(df['is_dup_text'].sum())} filas")
    note(f"Textos repetidos con etiqueta contradictoria (is_conflict_text=1): {int(df['is_conflict_text'].sum())} filas")
    cross = df.groupby(key)["source"].nunique()
    note(f"Textos presentes en ambas fuentes: {int((cross > 1).sum())}")
    leaks = df[df["source"] == "conda"].groupby(key)["split"].nunique()
    note(f"CONDA: textos que aparecen en train y valid a la vez: {int((leaks > 1).sum())}")

    (OUT / "reporte_calidad.md").write_text("# Reporte de calidad\n\n```\n" + "\n".join(log) + "\n```\n", encoding="utf-8")


if __name__ == "__main__":
    main()
