"""Experimento de contexto en CONDA: Les ayudan al modelo los mensajes anteriores de la misma conversacion?

Uso (desde la raiz del repo):
    python Proyecto/Modelado/05_contexto.py                 # lineales y BERT (9 corridas de BERT, ~1 hora con GPU)
    python Proyecto/Modelado/05_contexto.py --sin-bert
    python Proyecto/Modelado/05_contexto.py --solo-bert
    python Proyecto/Modelado/05_contexto.py --solo-tablas
    python Proyecto/Modelado/05_contexto.py --rapido        # prueba de humo con pocos datos; no toca los resultados

Variantes: k = 0 (sin contexto), 1 y 3 mensajes anteriores de la misma conversacion (conversationId), del mas viejo al
mas nuevo. En BERT el contexto va como primer segmento (CLS contexto SEP mensaje SEP) y el mensaje nunca se recorta;
en los lineales se agrega un TF-IDF de palabras del contexto, aparte del TF-IDF del mensaje.
Mismo split por partida y mismo dev que los demas scripts. Binario (is_toxic). Solo CONDA: en Jigsaw no hay hilo.
Se reportan las metricas sobre todo el test y sobre los mensajes que SI tienen contexto (los primeros de una conversacion
no tienen, y para ellos la variante con contexto es igual a la de sin contexto).

Salidas en Proyecto/Modelado/Resultados/: resultados_contexto.csv y resultados_contexto.md.
"""
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from comun import SALIDA, SEED, cargar_datos, contextos_conda, metricas, split_dev, split_por_fuente, tasas_grupos, vectorizar

KS = [0, 1, 3]
SEMILLAS = [SEED, SEED + 1, SEED + 2]
MAX_LEN, EPOCAS = 128, 3
C_GRID = [0.3, 1.0, 3.0]
NOMBRES = {"logistica": "Regresion logistica", "svm": "SVM lineal", "bert": "BERT"}
RAPIDO = "--rapido" in sys.argv


def con_contexto(test, pred, score):
    """Metricas solo sobre los mensajes que tienen al menos un mensaje previo."""
    m = (test["ctx1"] != "").to_numpy()  # criterio fijo: tiene al menos un mensaje previo en su conversacion
    f1 = f1_score(test["is_toxic"][m], np.asarray(pred)[m], pos_label=1)
    from sklearn.metrics import average_precision_score
    return {"f1_toxico_con_ctx": f1, "auc_pr_con_ctx": average_precision_score(test["is_toxic"][m], np.asarray(score)[m]), "n_con_ctx": int(m.sum())}


def fila(modelo, k, semilla, test, pred, score, **extra):
    return {"modelo": modelo, "k": k, "semilla": semilla, **extra, **metricas(test["is_toxic"], pred, score),
            **tasas_grupos(test, pred), **con_contexto(test, pred, score)}


def lineales(train, test, k):
    X_tr, X_te = vectorizar(train, test)
    if k > 0:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True, dtype=np.float32)
        X_tr = hstack([X_tr, v.fit_transform(train["ctx"])]).tocsr()
        X_te = hstack([X_te, v.transform(test["ctx"])]).tocsr()
    y = train["is_toxic"].to_numpy()
    i_tr, i_dev = split_dev(train, "conda")
    filas = []
    for modelo in ["logistica", "svm"]:
        mejor = (-1.0, None)
        for C in C_GRID:
            clf = _crear(modelo, C).fit(X_tr[i_tr], y[i_tr])
            f1 = f1_score(y[i_dev], clf.predict(X_tr[i_dev]), pos_label=1)
            if f1 > mejor[0]:
                mejor = (f1, C)
        clf = _crear(modelo, mejor[1]).fit(X_tr, y)
        print(f"    k={k} {modelo}: C={mejor[1]}", flush=True)
        filas.append(fila(modelo, k, np.nan, test, clf.predict(X_te), clf.decision_function(X_te), C=mejor[1]))
    return filas


def _crear(modelo, C):
    if modelo == "logistica":
        return LogisticRegression(C=C, solver="liblinear", class_weight="balanced", random_state=SEED)
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED)


def bert(train, test, k, semilla):
    """Una corrida de BERT (una variante de contexto, una semilla); devuelve la fila de resultados."""
    import bert_util as bu
    tok = bu.tokenizador()
    i_tr, i_dev = split_dev(train, "conda")
    items = bu.codificar(tok, train["text"], MAX_LEN, train["ctx"] if k else None)
    items_te = bu.codificar(tok, test["text"], MAX_LEN, test["ctx"] if k else None)
    y = train["is_toxic"].to_numpy()
    t0 = time.time()
    print(f"   BERT k={k}, semilla {semilla}", flush=True)
    m, f1_dev = bu.entrenar([items[i] for i in i_tr], y[i_tr], [items[i] for i in i_dev], y[i_dev], 2, tok.pad_token_id, 1 if RAPIDO else EPOCAS, semilla)
    lg = bu.logits(m, items_te, tok.pad_token_id)
    margen = lg[:, 1] - lg[:, 0]
    f = fila("bert", k, semilla, test, (margen > 0).astype(int), margen, f1_dev=f1_dev)
    print(f"      -> F1 toxico={f['f1_toxico']:.4f}  con contexto={f['f1_toxico_con_ctx']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    del m
    bu.liberar()
    return f


def tabla_md(res):
    cols = ["f1_toxico", "recall_toxico", "auc_pr", "groseria_casual_bloqueada", "toxicidad_implicita_detectada", "f1_toxico_con_ctx"]
    enc = ["Modelo", "Mensajes previos (k)", "F1 toxico", "Recall toxico", "AUC-PR", "Groseria casual bloqueada", "Toxicidad implicita detectada", "F1 toxico solo con contexto"]
    lineas = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for (modelo, k), g in res.groupby(["modelo", "k"], sort=False):
        if len(g) > 1:
            celdas = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}" for c in cols]
            nombre = f"{NOMBRES[modelo]} ({len(g)} semillas)"
        else:
            celdas = [f"{g[c].iloc[0]:.3f}" for c in cols]
            nombre = NOMBRES[modelo]
        lineas.append("| " + " | ".join([nombre, str(k)] + celdas) + " |")
    n = int(res["n_con_ctx"].iloc[0])
    nota = f"\nMensajes del test con al menos un mensaje previo: {n} de {int(res['n_test'].iloc[0]) if 'n_test' in res else '?'}."
    # diferencia pareada por semilla frente a k=0 (solo BERT)
    b = res[res.modelo == "bert"]
    if not b.empty and (b.k == 0).any():
        base = b[b.k == 0].set_index("semilla")["f1_toxico"]
        for k in sorted(set(b.k) - {0}):
            d = b[b.k == k].set_index("semilla")["f1_toxico"] - base
            nota += f"\nBERT, F1 toxico con k={k} menos k=0, por semilla: " + ", ".join(f"{v:+.3f}" for v in d.dropna()) + f" (media {d.mean():+.3f})."
    return "\n".join(lineas) + "\n" + nota


def main():
    SALIDA.mkdir(exist_ok=True)
    csv = SALIDA / "resultados_contexto.csv"
    if "--solo-tablas" in sys.argv:
        (SALIDA / "resultados_contexto.md").write_text(tabla_md(pd.read_csv(csv)) + "\n", encoding="utf-8")
        return
    df = cargar_datos()
    train, test = split_por_fuente(df, "conda")
    if RAPIDO:
        train, test = train.sample(600, random_state=SEED).reset_index(drop=True), test.sample(400, random_state=SEED).reset_index(drop=True)
    for k in (1, 3):
        c = contextos_conda(k)
        train[f"ctx{k}"], test[f"ctx{k}"] = train["uid"].map(c).fillna(""), test["uid"].map(c).fillna("")
    print(f"CONDA contexto: train={len(train)} test={len(test)}; con >=1 mensaje previo en test: {(test['ctx1'] != '').mean():.1%}", flush=True)
    previas = pd.read_csv(csv).to_dict("records") if (csv.exists() and not RAPIDO) else []
    llave = lambda f: (f["modelo"], int(f["k"]), -1 if f["semilla"] != f["semilla"] else int(f["semilla"]))

    def guardar(nuevas):
        """Inserta o reemplaza las filas nuevas y escribe el CSV y la tabla (se llama tras cada corrida)."""
        for f in nuevas:
            f["n_test"] = len(test)
        claves = {llave(f) for f in nuevas}
        res = pd.DataFrame([f for f in previas if llave(f) not in claves] + nuevas)
        previas[:] = res.to_dict("records")
        res["_o"] = res["modelo"].map({"logistica": 0, "svm": 1, "bert": 2})
        res = res.sort_values(["_o", "k", "semilla"]).drop(columns="_o")
        if not RAPIDO:
            res.to_csv(csv, index=False)
            (SALIDA / "resultados_contexto.md").write_text(tabla_md(res) + "\n", encoding="utf-8")
        return res

    res = None
    ctx = lambda k: (train.assign(ctx=train[f"ctx{k}"] if k else ""), test.assign(ctx=test[f"ctx{k}"] if k else ""))
    if "--solo-bert" not in sys.argv:
        for k in KS:
            res = guardar(lineales(*ctx(k), k))
    if "--sin-bert" not in sys.argv:
        semillas = [int(s) for s in sys.argv[sys.argv.index("--semillas") + 1:] if s.isdigit()] if "--semillas" in sys.argv else SEMILLAS
        hechas = {llave(f) for f in previas}
        for semilla in semillas[:1] if RAPIDO else semillas:  # semilla afuera: si se corta, quedan completas las primeras semillas
            for k in KS:
                if ("bert", k, semilla) in hechas:
                    print(f"   BERT k={k}, semilla {semilla}: ya estaba, se salta", flush=True)
                    continue
                res = guardar([bert(*ctx(k), k, semilla)])  # se guarda tras cada corrida
    print("\n" + tabla_md(res if res is not None else pd.read_csv(csv)))


if __name__ == "__main__":
    main()
