"""Mensajes no vistos (guia del 6 de noviembre, corrida 2): la tabla principal de CONDA, solo sobre los mensajes del test
cuyo texto NO aparece en train.

Uso (desde la raiz del repo):
    python Proyecto/Modelado/14_no_vistos.py              # entrena los 6 modelos (BiGRU y BERT con 3 semillas)
    python Proyecto/Modelado/14_no_vistos.py --solo-tablas

Por que: en CONDA muchos mensajes se repiten literalmente (gg, lol, ez...). Si el mismo texto esta en train y en test, el
modelo "lo recuerda" y el resultado se infla. Aqui se repite la tabla de la seccion 2 de RESULTADOS.md (misma particion por
partida, mismos modelos, mismos hiperparametros) pero se mide solo sobre los mensajes cuyo texto, en minusculas y sin espacios
sobrantes, no esta en train: 4,917 de los 7,569 del test.

Los modelos se reentrenan en esta corrida (no se guardan los de las tablas anteriores) y las predicciones del test se guardan
en Resultados/predicciones_no_vistos.npz, de modo que el script se puede reanudar y las tablas se rehacen sin reentrenar.
Las cifras de "todo el test" salen de estos mismos modelos y deben coincidir con la seccion 2 salvo por pequenas diferencias
de reentrenamiento.

Salidas en Proyecto/Modelado/Resultados/:
    resultados_no_vistos.csv   una fila por modelo, semilla y subconjunto (todo el test / no vistos)
    resultados_no_vistos.md    tabla comparativa
"""
import importlib
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import bert_util as bu
from comun import SALIDA, SEED, cargar_datos, metricas, split_dev, split_por_fuente, tasas_grupos, vectorizar

bigru = importlib.import_module("02_bigru")
CACHE = SALIDA / "predicciones_no_vistos.npz"
ORDEN = ["Filtro de palabras", "Regresión logística", "SVM lineal", "XGBoost", "BiGRU", "BERT"]


def normalizar(s):
    return s.str.strip().str.lower()


def cargar_cache():
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=False)
        return {k: z[k] for k in z.files}
    return {}


def guardar_cache(c):
    np.savez_compressed(CACHE, **c)


def entrenar_todos(train, test):
    """Devuelve {clave: puntaje en test}; la prediccion es puntaje > 0 (margen) salvo XGBoost (prob. > 0.5, se guarda prob - 0.5)."""
    c = cargar_cache()
    y = train["is_toxic"].to_numpy()
    i_tr, i_dev = split_dev(train, "conda")

    c["Filtro de palabras|0"] = test["n_lex"].to_numpy().astype(np.float32) - 0.5  # > 0 equivale a tener alguna palabra
    if "Regresión logística|0" not in c or "SVM lineal|0" not in c:
        X_tr, X_te = vectorizar(train, test)
        c["Regresión logística|0"] = LogisticRegression(C=3.0, solver="liblinear", class_weight="balanced", random_state=SEED).fit(X_tr, y).decision_function(X_te)
        c["SVM lineal|0"] = LinearSVC(C=0.3, class_weight="balanced", random_state=SEED).fit(X_tr, y).decision_function(X_te)
        guardar_cache(c)
        print("   lineales listos", flush=True)
    if "XGBoost|0" not in c:
        t0 = time.time()
        X_tr, X_te = vectorizar(train, test)
        clf = xgb.XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.1, colsample_bytree=0.5, subsample=0.8, tree_method="hist",
                                device="cpu", scale_pos_weight=(y[i_tr] == 0).sum() / (y[i_tr] == 1).sum(), eval_metric="aucpr",
                                early_stopping_rounds=30, random_state=SEED, n_jobs=-1)
        clf.fit(X_tr[i_tr], y[i_tr], eval_set=[(X_tr[i_dev], y[i_dev])], verbose=False)
        c["XGBoost|0"] = clf.predict_proba(X_te)[:, 1] - 0.5
        guardar_cache(c)
        print(f"   XGBoost listo ({time.time() - t0:.0f} s)", flush=True)

    pendientes = [s for s in bigru.SEEDS if f"BiGRU|{s}" not in c]
    if pendientes:
        vocab = bigru.construir_vocab(train["text"].iloc[i_tr])
        X, L = bigru.codificar(train["text"], vocab, bigru.MAX_LEN["conda"])
        X_te, L_te = bigru.codificar(test["text"], vocab, bigru.MAX_LEN["conda"])
        for s in pendientes:
            t0 = time.time()
            m, _ = bigru.entrenar(X[i_tr], L[i_tr], y[i_tr], X[i_dev], L[i_dev], y[i_dev], len(vocab) + 2, s)
            c[f"BiGRU|{s}"] = bigru.puntajes(m, X_te, L_te)
            guardar_cache(c)
            print(f"   BiGRU semilla {s} lista ({time.time() - t0:.0f} s)", flush=True)

    pendientes = [s for s in (SEED, SEED + 1, SEED + 2) if f"BERT|{s}" not in c]
    if pendientes:
        tok = bu.tokenizador()
        items = bu.codificar(tok, train["text"], 64)
        items_te = bu.codificar(tok, test["text"], 64)
        for s in pendientes:
            t0 = time.time()
            m, _ = bu.entrenar([items[i] for i in i_tr], y[i_tr], [items[i] for i in i_dev], y[i_dev], 2, tok.pad_token_id, 3, s)
            lg = bu.logits(m, items_te, tok.pad_token_id)
            c[f"BERT|{s}"] = lg[:, 1] - lg[:, 0]
            guardar_cache(c)
            print(f"   BERT semilla {s} lista ({time.time() - t0:.0f} s)", flush=True)
            del m
            bu.liberar()
    return c


def tabla(test, nuevo, cache):
    filas = []
    for clave, score in cache.items():
        modelo, semilla = clave.split("|")
        pred = (score > 0).astype(int)
        for sub, mask in [("todo el test", np.ones(len(test), bool)), ("no vistos", nuevo)]:
            d = test[mask].reset_index(drop=True)
            filas.append({"modelo": modelo, "semilla": int(semilla), "subconjunto": sub, "n": len(d),
                          **metricas(d["is_toxic"], pred[mask], score[mask]), **tasas_grupos(d, pred[mask])})
    return pd.DataFrame(filas)


def celda(g, col):
    return f"{g[col].mean():.3f} ± {g[col].std(ddof=1):.3f}" if len(g) > 1 else f"{g[col].iloc[0]:.3f}"


def tabla_md(res):
    enc = ["Modelo", "F1 tóxico (todo el test)", "F1 tóxico (no vistos)", "AUC-PR (no vistos)",
           "Grosería casual bloqueada (no vistos)", "Toxicidad implícita detectada (no vistos)"]
    lineas = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for m in ORDEN:
        g_todo = res[(res["modelo"] == m) & (res["subconjunto"] == "todo el test")]
        g_nv = res[(res["modelo"] == m) & (res["subconjunto"] == "no vistos")]
        if g_nv.empty:
            continue
        nombre = m + (f" (media ± desv., {len(g_nv)} semillas)" if len(g_nv) > 1 else "")
        lineas.append("| " + " | ".join([nombre, celda(g_todo, "f1_toxico"), celda(g_nv, "f1_toxico"), celda(g_nv, "auc_pr"),
                                         celda(g_nv, "groseria_casual_bloqueada"), celda(g_nv, "toxicidad_implicita_detectada")]) + " |")
    return "\n".join(lineas)


def main():
    SALIDA.mkdir(exist_ok=True)
    df = cargar_datos()
    train, test = split_por_fuente(df, "conda")
    nuevo = (~normalizar(test["text"]).isin(set(normalizar(train["text"])))).to_numpy()
    nv = test[nuevo]
    print(f"test: {len(test)} mensajes; no vistos (texto ausente de train): {int(nuevo.sum())} ({nuevo.mean():.1%}) | "
          f"casual={int(nv['grupo_casual'].sum())} implicita={int(nv['grupo_implicita'].sum())} | % toxico no vistos={nv['is_toxic'].mean():.1%} "
          f"(todo el test {test['is_toxic'].mean():.1%})", flush=True)
    cache = cargar_cache() if "--solo-tablas" in sys.argv else entrenar_todos(train, test)
    res = tabla(test, nuevo, cache)
    res.to_csv(SALIDA / "resultados_no_vistos.csv", index=False)
    md = tabla_md(res)
    (SALIDA / "resultados_no_vistos.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
