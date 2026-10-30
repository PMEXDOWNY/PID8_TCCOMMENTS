"""CONDA con las 4 clases de intencion (O, E, A, I), para compararse con la Tabla 5 de Weld et al. 2021.

Uso (desde la raiz del repo):
    python Proyecto/Modelado/04_cuatro_clases.py              # lineales y BERT
    python Proyecto/Modelado/04_cuatro_clases.py --sin-bert   # solo lineales (minutos)
    python Proyecto/Modelado/04_cuatro_clases.py --solo-bert  # BERT, 3 semillas (~15 min con GPU)
    python Proyecto/Modelado/04_cuatro_clases.py --solo-tablas

Mismo split por partida, mismo dev y mismas metricas que los demas scripts. Ademas del F1 por clase, la prediccion de 4
clases se convierte a binaria (toxico = E o I) para sacar las mismas metricas y las dos tasas que los modelos binarios.

Salidas en Proyecto/Modelado/Resultados/: resultados_4clases.csv y resultados_4clases.md.
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from comun import CLASES_CONDA, SALIDA, SEED, cargar_datos, metricas, split_dev, split_por_fuente, tasas_grupos, vectorizar

C_GRID = [0.3, 1.0, 3.0]
SEMILLAS = [SEED, SEED + 1, SEED + 2]
MAX_LEN, EPOCAS = 64, 3
IDX = {c: i for i, c in enumerate(CLASES_CONDA)}
TOXICAS = [IDX["E"], IDX["I"]]
NOMBRES = {"logistica": "Regresion logistica", "svm": "SVM lineal", "bert": "BERT"}
# Weld et al. 2021, Tabla 5, Joint BERT (otro split: 60/20/20 por conversacion). Exactitud por mensaje = UCA.
WELD = {"f1_O": 0.954, "f1_E": 0.872, "f1_A": 0.800, "f1_I": 0.768, "exactitud": 0.921}


def metricas4(test, pred, prob):
    """F1 por clase, F1 macro, exactitud y las metricas binarias de 'toxico = E o I'. prob puede ser None."""
    y = test["y4"].to_numpy()
    f1c = f1_score(y, pred, average=None, labels=range(4))
    out = {f"f1_{c}": f1c[i] for c, i in IDX.items()}
    out.update(f1_macro4=f1_score(y, pred, average="macro"), exactitud=accuracy_score(y, pred))
    pred_t = np.isin(pred, TOXICAS).astype(int)
    score = prob[:, TOXICAS].sum(1) if prob is not None else pred_t
    b = metricas(test["is_toxic"], pred_t, score)
    out.update({f"bin_{k}": v for k, v in b.items() if k in ("f1_toxico", "precision_toxico", "recall_toxico", "auc_pr")})
    if prob is None:  # el SVM no da probabilidades: un AUC-PR con predicciones duras no es un AUC-PR
        out["bin_auc_pr"] = np.nan
    out.update(tasas_grupos(test, pred_t))
    return out


def lineales(train, test):
    X_tr, X_te = vectorizar(train, test)
    y = train["y4"].to_numpy()
    i_tr, i_dev = split_dev(train, "conda")
    filas = []
    for modelo in ["logistica", "svm"]:
        mejor = (-1.0, None)
        for C in C_GRID:
            clf = _crear(modelo, C).fit(X_tr[i_tr], y[i_tr])
            f1 = f1_score(y[i_dev], clf.predict(X_tr[i_dev]), average="macro")
            print(f"    {modelo:9s} C={C:<4} F1 macro dev={f1:.4f}", flush=True)
            if f1 > mejor[0]:
                mejor = (f1, C)
        clf = _crear(modelo, mejor[1]).fit(X_tr, y)
        prob = clf.predict_proba(X_te) if modelo == "logistica" else None
        filas.append({"modelo": modelo, "semilla": np.nan, "C": mejor[1], **metricas4(test, clf.predict(X_te), prob)})
    return filas


def _crear(modelo, C):
    if modelo == "logistica":  # uno contra resto, igual que liblinear hacia con mas de 2 clases
        return OneVsRestClassifier(LogisticRegression(C=C, solver="liblinear", class_weight="balanced", random_state=SEED))
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED)


def bert(train, test):
    import torch
    from scipy.special import softmax
    import bert_util as bu
    tok = bu.tokenizador()
    i_tr, i_dev = split_dev(train, "conda")
    items = bu.codificar(tok, train["text"], MAX_LEN)
    items_te = bu.codificar(tok, test["text"], MAX_LEN)
    y = train["y4"].to_numpy()
    filas = []
    for semilla in SEMILLAS:
        t0 = time.time()
        print(f"   BERT 4 clases, semilla {semilla}", flush=True)
        m, f1_dev = bu.entrenar([items[i] for i in i_tr], y[i_tr], [items[i] for i in i_dev], y[i_dev], 4, tok.pad_token_id, EPOCAS, semilla)
        lg = bu.logits(m, items_te, tok.pad_token_id)
        filas.append({"modelo": "bert", "semilla": semilla, "C": np.nan, "f1_dev": f1_dev, **metricas4(test, lg.argmax(1), softmax(lg, axis=1))})
        print(f"      -> F1 macro={filas[-1]['f1_macro4']:.4f}  E={filas[-1]['f1_E']:.3f} I={filas[-1]['f1_I']:.3f} ({time.time() - t0:.0f} s)", flush=True)
        del m
        bu.liberar()
    return filas


def tabla_md(res):
    cols = ["f1_O", "f1_E", "f1_A", "f1_I", "f1_macro4", "exactitud", "bin_f1_toxico", "groseria_casual_bloqueada", "toxicidad_implicita_detectada"]
    enc = ["Modelo", "F1 O", "F1 E", "F1 A", "F1 I", "F1 macro (4 clases)", "Exactitud", "F1 toxico (E o I)", "Groseria casual bloqueada", "Toxicidad implicita detectada"]
    lineas = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for modelo, g in res.groupby("modelo", sort=False):
        if len(g) > 1:
            celdas = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}" for c in cols]
            nombre = f"{NOMBRES[modelo]} (media ± desv. de {len(g)} semillas)"
        else:
            celdas = [f"{g[c].iloc[0]:.3f}" for c in cols]
            nombre = NOMBRES[modelo]
        lineas.append("| " + " | ".join([nombre] + celdas) + " |")
    w = [f"{WELD[c]:.3f}" for c in ["f1_O", "f1_E", "f1_A", "f1_I"]] + ["", f"{WELD['exactitud']:.3f}", "", "", ""]
    lineas.append("| Weld et al. 2021, Joint BERT (otro split) | " + " | ".join(w) + " |")
    return "\n".join(lineas)


def main():
    SALIDA.mkdir(exist_ok=True)
    csv = SALIDA / "resultados_4clases.csv"
    if "--solo-tablas" in sys.argv:
        (SALIDA / "resultados_4clases.md").write_text(tabla_md(pd.read_csv(csv)) + "\n", encoding="utf-8")
        return
    df = cargar_datos()
    df["y4"] = df["conda_intent"].map(IDX)
    train, test = split_por_fuente(df, "conda")
    assert train["y4"].notna().all() and test["y4"].notna().all()
    train["y4"], test["y4"] = train["y4"].astype(int), test["y4"].astype(int)
    print(f"CONDA 4 clases: train={len(train)} test={len(test)}  test por clase: {test['y4'].value_counts().sort_index().to_dict()}", flush=True)
    previas = pd.read_csv(csv).to_dict("records") if csv.exists() else []
    nuevas = []
    if "--solo-bert" not in sys.argv:
        nuevas += lineales(train, test)
    if "--sin-bert" not in sys.argv:
        nuevas += bert(train, test)
    claves = {(f["modelo"], f["semilla"] if f["semilla"] == f["semilla"] else -1) for f in nuevas}
    conserva = [f for f in previas if (f["modelo"], f["semilla"] if f["semilla"] == f["semilla"] else -1) not in claves]
    res = pd.DataFrame(conserva + nuevas)
    res["_o"] = res["modelo"].map({"logistica": 0, "svm": 1, "bert": 2})
    res = res.sort_values(["_o", "semilla"]).drop(columns="_o")
    res.to_csv(csv, index=False)
    (SALIDA / "resultados_4clases.md").write_text(tabla_md(res) + "\n", encoding="utf-8")
    print("\n" + tabla_md(res))


if __name__ == "__main__":
    main()
