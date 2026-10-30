"""Entrena y GUARDA los modelos de CONDA que usa la demo (11_demo.py).

Uso (desde la raiz del repo):
    python Proyecto/Modelado/10_entrenar_demo.py

Guarda en Proyecto/Modelado/Demo/modelos/ (esa carpeta no se sube a git: los BERT pesan ~220 MB cada uno):
    lineales.joblib     TF-IDF (palabras y caracteres) + regresion logistica + SVM lineal, entrenados con todo el train de CONDA
    bert_binario.pt     BERT toxico / no toxico (mismo entrenamiento que 03_bert.py: 3 epocas, semilla 42, 80% de train)
    bert_4clases.pt     BERT con las 4 clases O, E, A, I (mismo entrenamiento que 04_cuatro_clases.py)
Es reanudable: si un archivo ya existe, se salta. Tarda unos 12 minutos con GPU (casi todo en los dos BERT).
Los modelos usan el mismo split y los mismos hiperparametros que las tablas de RESULTADOS.md: la demo muestra los mismos
modelos que se evaluaron, no unos distintos.
"""
import time
from pathlib import Path

import joblib
import torch
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from comun import CLASES_CONDA, SEED, cargar_datos, crear_vectorizadores, split_dev, split_por_fuente

DEMO = Path(__file__).resolve().parent / "Demo" / "modelos"
MAX_LEN, EPOCAS = 64, 3
C_LOGISTICA, C_SVM = 3.0, 0.3  # los C que salieron del dev en CONDA (RESULTADOS.md, seccion 1)


def lineales(train):
    t0 = time.time()
    palabras, chars = crear_vectorizadores()
    X = hstack([palabras.fit_transform(train["text"]), chars.fit_transform(train["text"])]).tocsr()
    y = train["is_toxic"].to_numpy()
    log = LogisticRegression(C=C_LOGISTICA, solver="liblinear", class_weight="balanced", random_state=SEED).fit(X, y)
    svm = LinearSVC(C=C_SVM, class_weight="balanced", random_state=SEED).fit(X, y)
    joblib.dump({"palabras": palabras, "chars": chars, "logistica": log, "svm": svm}, DEMO / "lineales.joblib", compress=3)
    print(f"   lineales guardados ({time.time() - t0:.0f} s)", flush=True)


def bert(train, y, n_clases, archivo):
    import bert_util as bu
    t0 = time.time()
    tok = bu.tokenizador()
    i_tr, i_dev = split_dev(train, "conda")
    items = bu.codificar(tok, train["text"], MAX_LEN)
    modelo, f1_dev = bu.entrenar([items[i] for i in i_tr], y[i_tr], [items[i] for i in i_dev], y[i_dev], n_clases,
                                 tok.pad_token_id, EPOCAS, SEED)
    # en media precision: la mitad de tamano en disco y no cambia las predicciones de forma apreciable
    torch.save({"estado": {k: v.half().cpu() for k, v in modelo.state_dict().items()}, "n_clases": n_clases,
                "max_len": MAX_LEN, "f1_dev": f1_dev}, DEMO / archivo)
    del modelo
    bu.liberar()
    print(f"   {archivo} guardado, F1 dev={f1_dev:.4f} ({time.time() - t0:.0f} s)", flush=True)


def main():
    DEMO.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    train, _ = split_por_fuente(df, "conda")
    print(f"CONDA train: {len(train)} mensajes", flush=True)
    if not (DEMO / "lineales.joblib").exists():
        lineales(train)
    if not (DEMO / "bert_binario.pt").exists():
        bert(train, train["is_toxic"].to_numpy(), 2, "bert_binario.pt")
    if not (DEMO / "bert_4clases.pt").exists():
        y4 = train["conda_intent"].map({c: i for i, c in enumerate(CLASES_CONDA)}).astype(int).to_numpy()
        bert(train, y4, 4, "bert_4clases.pt")
    print("listo:", sorted(p.name for p in DEMO.iterdir()))


if __name__ == "__main__":
    main()
