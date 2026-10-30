"""Entrega del 28 de octubre: filtro de palabras, regresion logistica y SVM lineal, por fuente.

Uso (desde la raiz del repo):
    python Proyecto/Modelado/01_baseline_y_lineales.py

Que hace:
  1. Baseline: marca como toxico todo mensaje con al menos una palabra del lexicon T (filtro de palabras prohibidas).
  2. Regresion logistica y SVM lineal sobre TF-IDF de palabras + caracteres. Un modelo POR FUENTE (CONDA y Jigsaw).
  3. Split por partida (group_id) en CONDA y valid estratificado en Jigsaw. C se elige en un dev dentro de train.
  4. Metricas por fuente y las dos tasas (groseria casual bloqueada, toxicidad implicita detectada).

Salidas en Proyecto/Modelado/Resultados/:
    resultados_lineales.csv / .md    tabla por modelo y por fuente
    tasas_lineales.png               F1 toxico y las dos tasas
"""
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from comun import SALIDA, SEED, cargar_datos, metricas, split_dev, split_por_fuente, tasas_grupos, vectorizar

C_GRID = [0.3, 1.0, 3.0]
NOMBRES = {"filtro_palabras": "Filtro de palabras (lexicon T)", "logistica": "Regresion logistica", "svm": "SVM lineal"}


def crear(modelo, C):
    if modelo == "logistica":
        return LogisticRegression(C=C, solver="liblinear", class_weight="balanced", random_state=SEED)
    return LinearSVC(C=C, class_weight="balanced", random_state=SEED)


def puntaje(clf, X):
    return clf.decision_function(X)  # logistica y SVM: margen; sirve para AUC-PR


def elegir_C(modelo, X_tr, y_tr, i_tr, i_dev):
    """C con mejor F1 de la clase toxica en el dev (dentro de train)."""
    mejor = (-1.0, None)
    for C in C_GRID:
        clf = crear(modelo, C).fit(X_tr[i_tr], y_tr[i_tr])
        f1 = f1_score(y_tr[i_dev], clf.predict(X_tr[i_dev]), pos_label=1)
        print(f"    {modelo:9s} C={C:<4} F1 toxico dev={f1:.4f}")
        if f1 > mejor[0]:
            mejor = (f1, C)
    return mejor[1]


def fila(fuente, modelo, C, test, pred, score, n_train):
    return {"fuente": fuente, "modelo": modelo, "C": C, "n_train": n_train, "n_test": len(test),
            **metricas(test["is_toxic"], pred, score), **tasas_grupos(test, pred)}


def tabla_md(res):
    cols = ["fuente", "modelo", "f1_macro", "f1_toxico", "recall_toxico", "auc_pr",
            "groseria_casual_bloqueada", "toxicidad_implicita_detectada"]
    encabezado = ["Fuente", "Modelo", "F1 macro", "F1 toxico", "Recall toxico", "AUC-PR",
                  "Groseria casual bloqueada", "Toxicidad implicita detectada"]
    lineas = ["| " + " | ".join(encabezado) + " |", "|" + "---|" * len(encabezado)]
    for _, r in res.iterrows():
        vals = [r["fuente"], NOMBRES[r["modelo"]]] + [f"{r[c]:.3f}" for c in cols[2:]]
        lineas.append("| " + " | ".join(vals) + " |")
    return "\n".join(lineas)


def grafica(res):
    fuentes = ["conda", "jigsaw"]
    modelos = ["filtro_palabras", "logistica", "svm"]
    metr = [("f1_toxico", "F1 de la clase toxica"),
            ("groseria_casual_bloqueada", "Groseria casual bloqueada\n(menor es mejor)"),
            ("toxicidad_implicita_detectada", "Toxicidad implicita detectada\n(mayor es mejor)")]
    fig, ejes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    ancho = 0.25
    for ax, (col, titulo) in zip(ejes, metr):
        for k, m in enumerate(modelos):
            vals = [res[(res.fuente == f) & (res.modelo == m)][col].iloc[0] for f in fuentes]
            barras = ax.bar(np.arange(len(fuentes)) + (k - 1) * ancho, vals, ancho, label=NOMBRES[m])
            ax.bar_label(barras, fmt="%.2f", fontsize=8)
        ax.set_xticks(range(len(fuentes)), ["CONDA", "Jigsaw"])
        ax.set_title(titulo, fontsize=10)
        ax.set_ylim(0, 1.08)
    ejes[0].set_ylabel("Valor en test")
    h, l = ejes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(SALIDA / "tasas_lineales.png", dpi=150)
    plt.close(fig)


def main():
    SALIDA.mkdir(exist_ok=True)
    if "--solo-tablas" in sys.argv:  # regenera tabla y grafica desde el CSV, sin reentrenar (Jigsaw tarda ~20 min)
        res = pd.read_csv(SALIDA / "resultados_lineales.csv")
        (SALIDA / "resultados_lineales.md").write_text(tabla_md(res) + "\n", encoding="utf-8")
        grafica(res)
        return
    df = cargar_datos()
    filas = []
    for fuente in ["conda", "jigsaw"]:
        t0 = time.time()
        train, test = split_por_fuente(df, fuente)
        print(f"\n== {fuente}: train={len(train)} test={len(test)} "
              f"(% toxico train={train['is_toxic'].mean():.1%} test={test['is_toxic'].mean():.1%}; "
              f"casual={int(test['grupo_casual'].sum())} implicita={int(test['grupo_implicita'].sum())} en test)")
        if fuente == "conda":
            solape = set(train["group_id"]) & set(test["group_id"])
            assert not solape, "hay partidas en train y test"

        # 1. Filtro de palabras: toxico si trae al menos una palabra del lexicon T
        pred = (test["n_lex"] > 0).astype(int)
        filas.append(fila(fuente, "filtro_palabras", np.nan, test, pred, test["n_lex"], len(train)))

        # 2. Modelos lineales
        X_tr, X_te = vectorizar(train, test)
        y_tr = train["is_toxic"].to_numpy()
        i_tr, i_dev = split_dev(train, fuente)
        for modelo in ["logistica", "svm"]:
            C = elegir_C(modelo, X_tr, y_tr, i_tr, i_dev)
            clf = crear(modelo, C).fit(X_tr, y_tr)
            filas.append(fila(fuente, modelo, C, test, clf.predict(X_te), puntaje(clf, X_te), len(train)))
        print(f"   ({time.time() - t0:.0f} s)")

    res = pd.DataFrame(filas)
    res.to_csv(SALIDA / "resultados_lineales.csv", index=False)
    md = tabla_md(res)
    (SALIDA / "resultados_lineales.md").write_text(md + "\n", encoding="utf-8")
    grafica(res)
    print("\n" + md)


if __name__ == "__main__":
    main()
