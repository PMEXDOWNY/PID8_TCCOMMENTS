"""XGBoost sobre los mismos rasgos TF-IDF (palabras y caracteres), por fuente.

Uso (desde la raiz del repo):
    pip install xgboost
    python Proyecto/Modelado/07_xgboost.py              # CONDA y Jigsaw
    python Proyecto/Modelado/07_xgboost.py --fuentes jigsaw   # solo una fuente (conserva la fila de la otra)
    python Proyecto/Modelado/07_xgboost.py --solo-tablas
    python Proyecto/Modelado/07_xgboost.py --rapido     # prueba de humo con pocos datos; no toca los resultados

Mismos splits, dev, metricas y grupos que los demas scripts. Como el BiGRU y BERT, se entrena con el 80% de train y el
20% restante (dev) decide cuantos arboles dejar (parada temprana por AUC-PR del dev); el test no se toca.
Hiperparametros fijos, sin ajuste: 600 arboles como maximo, profundidad 6, tasa de aprendizaje 0.1, 50% de rasgos por
arbol, peso de la clase toxica balanceado.

Salidas en Proyecto/Modelado/Resultados/: resultados_xgboost.csv y resultados_xgboost.md.
"""
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb

from comun import SALIDA, SEED, cargar_datos, metricas, split_dev, split_por_fuente, tasas_grupos, vectorizar

RAPIDO = "--rapido" in sys.argv
# Con los 350,000 rasgos completos, XGBoost no termino en Jigsaw despues de mas de 1 hora (comentarios largos, ~97,000 filas);
# ahi se usan 50,000 (30,000 de palabras y 20,000 de caracteres). CONDA usa los rasgos completos.
RASGOS = {"conda": {}, "jigsaw": {"max_palabras": 30_000, "max_chars": 20_000}}


def tabla_md(res):
    cols = ["f1_macro", "f1_toxico", "recall_toxico", "auc_pr", "groseria_casual_bloqueada", "toxicidad_implicita_detectada"]
    enc = ["Fuente", "Modelo", "F1 macro", "F1 toxico", "Recall toxico", "AUC-PR", "Groseria casual bloqueada", "Toxicidad implicita detectada"]
    lineas = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for _, r in res.iterrows():
        lineas.append("| " + " | ".join([r["fuente"], "XGBoost"] + [f"{r[c]:.3f}" for c in cols]) + " |")
    return "\n".join(lineas)


def main():
    SALIDA.mkdir(exist_ok=True)
    csv = SALIDA / "resultados_xgboost.csv"
    if "--solo-tablas" in sys.argv:
        (SALIDA / "resultados_xgboost.md").write_text(tabla_md(pd.read_csv(csv)) + "\n", encoding="utf-8")
        return
    df = cargar_datos()
    fuentes = sys.argv[sys.argv.index("--fuentes") + 1:] if "--fuentes" in sys.argv else ["conda", "jigsaw"]
    filas = [f for f in pd.read_csv(csv).to_dict("records") if f["fuente"] not in fuentes] if (csv.exists() and not RAPIDO) else []
    for fuente in fuentes:
        t0 = time.time()
        train, test = split_por_fuente(df, fuente)
        if RAPIDO:
            train, test = train.sample(800, random_state=SEED).reset_index(drop=True), test.sample(400, random_state=SEED).reset_index(drop=True)
        X_tr, X_te = vectorizar(train, test, **RASGOS[fuente])
        y = train["is_toxic"].to_numpy()
        i_tr, i_dev = split_dev(train, fuente)
        clf = xgb.XGBClassifier(n_estimators=40 if RAPIDO else 600, max_depth=6, learning_rate=0.1, colsample_bytree=0.5,
                                subsample=0.8, tree_method="hist", device="cpu", scale_pos_weight=(y[i_tr] == 0).sum() / (y[i_tr] == 1).sum(),
                                eval_metric="aucpr", early_stopping_rounds=30, random_state=SEED, n_jobs=-1)
        clf.fit(X_tr[i_tr], y[i_tr], eval_set=[(X_tr[i_dev], y[i_dev])], verbose=False)
        prob = clf.predict_proba(X_te)[:, 1]
        pred = (prob > 0.5).astype(int)
        filas.append({"fuente": fuente, "modelo": "xgboost", "n_arboles": int(clf.best_iteration) + 1, "n_train": len(i_tr), "n_test": len(test),
                      **metricas(test["is_toxic"], pred, prob), **tasas_grupos(test, pred)})
        print(f"   {fuente}: arboles={filas[-1]['n_arboles']}  F1 toxico={filas[-1]['f1_toxico']:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if not RAPIDO:  # se guarda tras cada fuente por si se corta
            parcial = pd.DataFrame(filas)
            parcial.to_csv(csv, index=False)
            (SALIDA / "resultados_xgboost.md").write_text(tabla_md(parcial) + "\n", encoding="utf-8")
    res = pd.DataFrame(filas)
    if not RAPIDO:
        res.to_csv(csv, index=False)
        (SALIDA / "resultados_xgboost.md").write_text(tabla_md(res) + "\n", encoding="utf-8")
    print("\n" + tabla_md(res))


if __name__ == "__main__":
    main()
