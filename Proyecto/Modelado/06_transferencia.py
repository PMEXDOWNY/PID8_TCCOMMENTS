"""Transferencia entre fuentes: entrenar en CONDA, en Jigsaw o en las dos juntas, y evaluar en cada test (como Antypas 2023).

Uso (desde la raiz del repo):
    python Proyecto/Modelado/06_transferencia.py                # lineales y BERT (BERT ~1 hora con GPU)
    python Proyecto/Modelado/06_transferencia.py --sin-bert     # lineales (~30 min, casi todo Jigsaw y el conjunto)
    python Proyecto/Modelado/06_transferencia.py --solo-bert
    python Proyecto/Modelado/06_transferencia.py --solo-tablas
    python Proyecto/Modelado/06_transferencia.py --rapido       # prueba de humo con pocos datos; no toca los resultados

Tres entrenamientos (conda, jigsaw, conjunto) x dos tests (conda, jigsaw). La diagonal (entrenar y evaluar en la misma
fuente) es el modelo por fuente; el conjunto responde si conviene un solo modelo para las dos. Los tests son los mismos
de los demas scripts. C se elige en un dev dentro de train (el del conjunto junta los dos dev). BERT: una sola semilla
(42), longitud maxima 128 en todos los casos, 3 epocas si se entrena solo en CONDA y 2 si entra Jigsaw.
Las etiquetas de las dos fuentes no significan lo mismo (is_toxic de CONDA = E o I; de Jigsaw = cualquiera de 6 banderas),
asi que fuera de la diagonal el umbral de decision no esta calibrado: el AUC-PR no depende del umbral y es la cifra
mas justa para comparar; la prevalencia de toxicos en cada test es la referencia de un modelo al azar.

Salidas en Proyecto/Modelado/Resultados/: resultados_transferencia.csv y resultados_transferencia.md.
"""
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from comun import SALIDA, SEED, cargar_datos, metricas, split_dev, split_por_fuente, tasas_grupos, vectorizar_varios

C_GRID = [0.3, 1.0, 3.0]
MAX_LEN = 128
EPOCAS = {"conda": 3, "jigsaw": 2, "conjunto": 2}
ENTRENOS = ["conda", "jigsaw", "conjunto"]
NOMBRES = {"logistica": "Regresion logistica", "svm": "SVM lineal", "bert": "BERT"}
RAPIDO = "--rapido" in sys.argv


def preparar():
    df = cargar_datos()
    tr, te, dev = {}, {}, {}
    for f in ("conda", "jigsaw"):
        tr[f], te[f] = split_por_fuente(df, f)
        if RAPIDO:
            tr[f], te[f] = tr[f].sample(600, random_state=SEED).reset_index(drop=True), te[f].sample(400, random_state=SEED).reset_index(drop=True)
        dev[f] = split_dev(tr[f], f)
    tr["conjunto"] = pd.concat([tr["conda"], tr["jigsaw"]], ignore_index=True)
    n = len(tr["conda"])
    i_tr_c, i_dev_c = dev["conda"]
    i_tr_j, i_dev_j = dev["jigsaw"]
    dev["conjunto"] = (np.concatenate([i_tr_c, i_tr_j + n]), np.concatenate([i_dev_c, i_dev_j + n]))
    return tr, te, dev


def _crear(modelo, C):
    if modelo == "logistica":
        return LogisticRegression(C=C, solver="liblinear", class_weight="balanced", random_state=SEED)
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED)


def fila(modelo, entrenado, evaluado, test, pred, score, **extra):
    return {"modelo": modelo, "entrenado_en": entrenado, "evaluado_en": evaluado, "n_test": len(test),
            "prevalencia": float(test["is_toxic"].mean()), **extra,
            **metricas(test["is_toxic"], pred, score), **tasas_grupos(test, pred)}


def lineales(tr, te, dev):
    filas = []
    for ent in ENTRENOS:
        t0 = time.time()
        X_tr, (X_c, X_j) = vectorizar_varios(tr[ent], [te["conda"], te["jigsaw"]])
        y = tr[ent]["is_toxic"].to_numpy()
        i_tr, i_dev = dev[ent]
        for modelo in ["logistica", "svm"]:
            mejor = (-1.0, None)
            for C in C_GRID:
                clf = _crear(modelo, C).fit(X_tr[i_tr], y[i_tr])
                f1 = f1_score(y[i_dev], clf.predict(X_tr[i_dev]), pos_label=1)
                if f1 > mejor[0]:
                    mejor = (f1, C)
            clf = _crear(modelo, mejor[1]).fit(X_tr, y)
            print(f"    {modelo:9s} entrenado en {ent:8s}: C={mejor[1]}", flush=True)
            for ev, X in (("conda", X_c), ("jigsaw", X_j)):
                filas.append(fila(modelo, ent, ev, te[ev], clf.predict(X), clf.decision_function(X), C=mejor[1]))
        print(f"   ({ent}: {time.time() - t0:.0f} s)", flush=True)
    return filas


def bert(tr, te, dev, semilla=SEED):
    import bert_util as bu
    tok = bu.tokenizador()
    items_te = {f: bu.codificar(tok, te[f]["text"], MAX_LEN) for f in te}
    filas = []
    csv = SALIDA / "resultados_transferencia.csv"
    hechos = set(pd.read_csv(csv).query("modelo == 'bert'")["entrenado_en"]) if (csv.exists() and not RAPIDO) else set()
    for ent in ENTRENOS:
        if ent in hechos:
            print(f"   BERT entrenado en {ent}: ya estaba, se salta", flush=True)
            continue
        t0 = time.time()
        print(f"   BERT entrenado en {ent}, semilla {semilla}", flush=True)
        items = bu.codificar(tok, tr[ent]["text"], MAX_LEN)
        y = tr[ent]["is_toxic"].to_numpy()
        i_tr, i_dev = dev[ent]
        m, f1_dev = bu.entrenar([items[i] for i in i_tr], y[i_tr], [items[i] for i in i_dev], y[i_dev], 2, tok.pad_token_id,
                                1 if RAPIDO else EPOCAS[ent], semilla)
        for ev in ("conda", "jigsaw"):
            lg = bu.logits(m, items_te[ev], tok.pad_token_id)
            margen = lg[:, 1] - lg[:, 0]
            filas.append(fila("bert", ent, ev, te[ev], (margen > 0).astype(int), margen, semilla=semilla, f1_dev=f1_dev))
            print(f"      -> evaluado en {ev}: F1 toxico={filas[-1]['f1_toxico']:.4f}  AUC-PR={filas[-1]['auc_pr']:.4f}", flush=True)
        print(f"   ({ent}: {time.time() - t0:.0f} s)", flush=True)
        del m
        bu.liberar()
        yield filas[-2:]


def tabla_md(res):
    cols = ["f1_toxico", "recall_toxico", "auc_pr", "groseria_casual_bloqueada", "toxicidad_implicita_detectada"]
    enc = ["Modelo", "Entrenado en", "Evaluado en", "Prevalencia de toxicos", "F1 toxico", "Recall toxico", "AUC-PR", "Groseria casual bloqueada", "Toxicidad implicita detectada"]
    lineas = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for _, r in res.iterrows():
        diag = " (misma fuente)" if r["entrenado_en"] == r["evaluado_en"] else ""
        lineas.append("| " + " | ".join([NOMBRES[r["modelo"]], r["entrenado_en"], r["evaluado_en"] + diag, f"{r['prevalencia']:.3f}"] + [f"{r[c]:.3f}" for c in cols]) + " |")
    return "\n".join(lineas)


def guardar(filas_nuevas, csv):
    previas = pd.read_csv(csv).to_dict("records") if (csv.exists() and not RAPIDO) else []
    llave = lambda f: (f["modelo"], f["entrenado_en"], f["evaluado_en"])
    nuevas = {llave(f) for f in filas_nuevas}
    res = pd.DataFrame([f for f in previas if llave(f) not in nuevas] + filas_nuevas)
    res["_m"] = res["modelo"].map({"logistica": 0, "svm": 1, "bert": 2})
    res["_e"] = res["entrenado_en"].map({e: i for i, e in enumerate(ENTRENOS)})
    res = res.sort_values(["_m", "_e", "evaluado_en"]).drop(columns=["_m", "_e"])
    if not RAPIDO:
        res.to_csv(csv, index=False)
        (SALIDA / "resultados_transferencia.md").write_text(tabla_md(res) + "\n", encoding="utf-8")
    return res


def main():
    SALIDA.mkdir(exist_ok=True)
    csv = SALIDA / "resultados_transferencia.csv"
    if "--solo-tablas" in sys.argv:
        (SALIDA / "resultados_transferencia.md").write_text(tabla_md(pd.read_csv(csv)) + "\n", encoding="utf-8")
        return
    tr, te, dev = preparar()
    print("tamanos:", {k: len(v) for k, v in tr.items()}, {k: len(v) for k, v in te.items()}, flush=True)
    res = None
    if "--solo-bert" not in sys.argv:
        res = guardar(lineales(tr, te, dev), csv)
    if "--sin-bert" not in sys.argv:
        for filas in bert(tr, te, dev):
            res = guardar(filas, csv)  # se guarda tras cada entrenamiento por si se corta
    print("\n" + tabla_md(res))


if __name__ == "__main__":
    main()
