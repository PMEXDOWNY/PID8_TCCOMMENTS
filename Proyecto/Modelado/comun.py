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


def crear_vectorizadores(max_palabras=200_000, max_chars=150_000):
    """Los dos TF-IDF del proyecto: palabras (1-2gram) y caracteres (2-4gram, dentro de palabras), sin ajustar."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    palabras = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_palabras, sublinear_tf=True, dtype=np.float32)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, max_features=max_chars,
                            sublinear_tf=True, dtype=np.float32)
    return palabras, chars


def vectorizar_varios(train, tests, max_palabras=200_000, max_chars=150_000):
    """TF-IDF de palabras + caracteres. Se ajusta solo con train y se aplica a cada DataFrame de `tests`."""
    from scipy.sparse import hstack
    palabras, chars = crear_vectorizadores(max_palabras, max_chars)
    X_tr = hstack([palabras.fit_transform(train["text"]), chars.fit_transform(train["text"])]).tocsr()
    return X_tr, [hstack([palabras.transform(t["text"]), chars.transform(t["text"])]).tocsr() for t in tests]


def vectorizar(train, test, **max_rasgos):
    X_tr, (X_te,) = vectorizar_varios(train, [test], **max_rasgos)
    return X_tr, X_te


CLASES_CONDA = ["O", "E", "A", "I"]  # orden de las etiquetas 0..3 en los experimentos de 4 clases


def contextos_conda(k: int) -> pd.Series:
    """uid -> los k mensajes anteriores de la misma conversacion (conversationId), del mas viejo al mas nuevo, unidos con ' | '.

    Se arma con los tres archivos originales: el de test no trae etiqueta pero si el texto, y una conversacion puede
    repartirse entre archivos. El orden es (chatTime, Id); dentro de cada conversacion el Id crece con chatTime.
    Como el split train/test se hace por partida, el contexto de un mensaje nunca sale de la otra mitad.
    """
    d = DATOS / "CONDA" / "datos"
    a = pd.concat([pd.read_csv(d / f"CONDA_{s}.csv", usecols=["Id", "conversationId", "chatTime", "utterance"])
                   for s in ("train", "valid", "test")], ignore_index=True)
    a["utterance"] = a["utterance"].fillna("").astype(str).str.strip()
    a = a.sort_values(["conversationId", "chatTime", "Id"]).reset_index(drop=True)
    textos, conv = a["utterance"].tolist(), a["conversationId"].tolist()
    ctx, inicio = [], 0
    for i in range(len(a)):
        if i > 0 and conv[i] != conv[i - 1]:
            inicio = i
        ctx.append(" | ".join(t for t in textos[max(inicio, i - k):i] if t))
    return pd.Series(ctx, index="conda_" + a["Id"].astype(str))


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
