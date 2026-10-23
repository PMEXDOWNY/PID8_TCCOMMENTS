"""Utilidades compartidas por los scripts de modelado: datos, lexicon, splits y metricas.

Todos los scripts de Proyecto/Modelado corren desde la raiz del repo:
    pip install pandas numpy scikit-learn matplotlib
    python Proyecto/Modelado/01_baseline_y_lineales.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

RAIZ = Path(__file__).resolve().parents[2]
DATOS = RAIZ / "Proyecto" / "Conjunto_de_Datos"
SALIDA = Path(__file__).resolve().parent / "Resultados"
SEED = 42
TEST_FRAC_CONDA = 0.20  # fraccion de PARTIDAS (matchId) que se aparta como test en CONDA

# Una palabra es una secuencia de letras/numeros/guion bajo/apostrofe, en minusculas.
TOKEN = re.compile(r"[a-z0-9_']+")


def cargar_lexicon_t() -> set:
    """Lexicon T de CONDA: groserias y jerga ofensiva (columna T de lexicon_refined_1209.csv)."""
    lex = pd.read_csv(DATOS / "CONDA" / "recursos" / "lexicon_refined_1209.csv")
    return set(lex["T"].dropna().astype(str).str.lower())


def cargar_datos() -> pd.DataFrame:
    """Dataset unificado + n_lex (cuantas palabras del lexicon T trae cada mensaje) y los dos grupos de interes.

    grupo_casual   : tiene alguna palabra del lexicon T y NO es toxico (is_toxic=0). Debe dejarse pasar.
    grupo_implicita: es toxico (is_toxic=1) y NO tiene ninguna palabra del lexicon T. Debe detectarse.
    """
    df = pd.read_csv(DATOS / "Unificado" / "dataset_unificado.csv.gz", low_memory=False)
    T = cargar_lexicon_t()
    df["n_lex"] = df["text"].map(lambda t: sum(w in T for w in TOKEN.findall(t.lower())))
    df["grupo_casual"] = ((df["n_lex"] > 0) & (df["is_toxic"] == 0)).astype(int)
    df["grupo_implicita"] = ((df["n_lex"] == 0) & (df["is_toxic"] == 1)).astype(int)
    return df


def split_por_fuente(df: pd.DataFrame, fuente: str):
    """Devuelve (train, test) de una fuente.

    CONDA : se juntan train+valid oficiales y se re-parte POR PARTIDA (group_id=matchId), porque el split
            oficial mezcla partidas entre train y valid (ver DECISIONES.md, seccion 4).
    Jigsaw: se usa el valid estratificado que ya se hizo en el preprocesamiento.
    """
    d = df[df["source"] == fuente].reset_index(drop=True)
    if fuente == "conda":
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC_CONDA, random_state=SEED)
        i_tr, i_te = next(gss.split(d, groups=d["group_id"]))
        return d.iloc[i_tr].reset_index(drop=True), d.iloc[i_te].reset_index(drop=True)
    return d[d["split"] == "train"].reset_index(drop=True), d[d["split"] == "valid"].reset_index(drop=True)


def split_dev(train: pd.DataFrame, fuente: str):
    """Parte un dev (20%) dentro de train para elegir hiperparametros. Por partida en CONDA, estratificado en Jigsaw."""
    if fuente == "conda":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        return next(gss.split(train, groups=train["group_id"]))
    idx = np.arange(len(train))
    i_tr, i_dev = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=train["is_toxic"])
    return i_tr, i_dev


def metricas(y, pred, score) -> dict:
    """Metricas de la clase toxica. Nunca se reporta exactitud sola: con 74% de clase O cualquier cosa da alto."""
    return {
        "exactitud": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "f1_toxico": f1_score(y, pred, pos_label=1),
        "precision_toxico": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall_toxico": recall_score(y, pred, pos_label=1),
        "auc_pr": average_precision_score(y, score),
    }


def tasas_grupos(test: pd.DataFrame, pred) -> dict:
    """Las dos tasas del articulo, sobre el test.

    groseria_casual_bloqueada     : % de mensajes con groseria pero NO toxicos que el modelo marca como toxicos
                                    (menor es mejor: es lo que un filtro de palabras hace mal).
    toxicidad_implicita_detectada : % de mensajes toxicos SIN groseria que el modelo marca como toxicos
                                    (mayor es mejor: es lo que un filtro de palabras no puede ver).
    """
    pred = np.asarray(pred)
    casual = test["grupo_casual"].to_numpy() == 1
    impl = test["grupo_implicita"].to_numpy() == 1
    return {
        "groseria_casual_bloqueada": float(pred[casual].mean()) if casual.any() else np.nan,
        "toxicidad_implicita_detectada": float(pred[impl].mean()) if impl.any() else np.nan,
        "n_casual": int(casual.sum()),
        "n_implicita": int(impl.sum()),
    }
