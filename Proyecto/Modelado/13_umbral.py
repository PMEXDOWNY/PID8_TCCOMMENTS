"""Curva de umbral (guia del 6 de noviembre, corrida 1): groseria casual bloqueada contra toxicidad implicita detectada.

Uso (desde la raiz del repo):
    python Proyecto/Modelado/13_umbral.py

Que hace (solo CONDA):
  1. Regresion logistica (C=3.0, como en 01_baseline_y_lineales.py) y BERT binario (semilla 42, el mismo de la demo),
     los dos entrenados con el 80% de train. El 20% restante (dev) NO se uso para ajustar los pesos, asi que sirve
     para elegir el umbral sin tocar el test.
  2. Se mueve el umbral de decision sobre la probabilidad de toxico (sigmoide del margen) y se calculan, en dev y en test,
     el F1 toxico y las dos tasas.
  3. El umbral se ELIGE en el dev con tres reglas (por defecto 0.5; el que maximiza F1; el mas bajo que deja la groseria
     casual bloqueada en 0.25 o menos) y se REPORTA lo que da en test, una sola vez.

Detalles que conviene tener presentes:
  - La logistica de esta corrida se entrena con el 80% de train (no con todo, como en la tabla principal), porque necesita
    un dev limpio. Por eso su fila "umbral 0.5" difiere un poco de la seccion 2 de RESULTADOS.md.
  - BERT es una sola semilla (42); la tabla principal promedia tres. Las probabilidades de los dos modelos no estan
    calibradas (se entrenaron con peso de clase balanceado): el umbral sirve para ordenar, no como probabilidad real.

Salidas en Proyecto/Modelado/Resultados/:
    resultados_umbral.csv   la curva completa (dev y test, cada 0.01)
    resultados_umbral.md    tabla de las tres reglas por modelo
    fig_umbral.png          curva y F1 contra umbral
"""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import bert_util as bu
from comun import SALIDA, SEED, cargar_datos, split_dev, split_por_fuente, vectorizar_varios

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEMO = SALIDA.parent / "Demo" / "modelos"
UMBRALES = np.round(np.arange(0.01, 1.0, 0.01), 2)
MAX_CASUAL = 0.25


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def puntajes_logistica(train, i_tr, i_dev, test):
    tr, dev = train.iloc[i_tr], train.iloc[i_dev]
    X_tr, (X_dev, X_te) = vectorizar_varios(tr, [dev, test])
    clf = LogisticRegression(C=3.0, solver="liblinear", class_weight="balanced", random_state=SEED)
    clf.fit(X_tr, tr["is_toxic"].to_numpy())
    return sigmoide(clf.decision_function(X_dev)), sigmoide(clf.decision_function(X_te))


def puntajes_bert(dev, test):
    from transformers import AutoModelForSequenceClassification
    guardado = torch.load(DEMO / "bert_binario.pt", map_location="cpu")
    modelo = AutoModelForSequenceClassification.from_pretrained(bu.MODELO, num_labels=guardado["n_clases"])
    modelo.load_state_dict(guardado["estado"])
    modelo.to(bu.DEV)
    tok = bu.tokenizador()
    salida = []
    for d in (dev, test):
        items = bu.codificar(tok, d["text"], guardado["max_len"])
        lg = bu.logits(modelo, items, tok.pad_token_id)
        salida.append(sigmoide(lg[:, 1] - lg[:, 0]))
    del modelo
    bu.liberar()
    return salida[0], salida[1]


def metricas_umbral(d, prob, t):
    pred = (prob > t).astype(int)
    casual = d["grupo_casual"].to_numpy() == 1
    impl = d["grupo_implicita"].to_numpy() == 1
    return {"f1_toxico": f1_score(d["is_toxic"], pred, pos_label=1),
            "casual_bloqueada": float(pred[casual].mean()), "implicita_detectada": float(pred[impl].mean())}


def curva(d, prob):
    return pd.DataFrame([{"umbral": t, **metricas_umbral(d, prob, t)} for t in UMBRALES])


def main():
    print(f"dispositivo: {bu.DEV}", flush=True)
    df = cargar_datos()
    train, test = split_por_fuente(df, "conda")
    i_tr, i_dev = split_dev(train, "conda")
    dev = train.iloc[i_dev].reset_index(drop=True)
    print(f"CONDA: train={len(i_tr)} dev={len(dev)} test={len(test)}  |  dev: casual={int(dev['grupo_casual'].sum())} "
          f"implicita={int(dev['grupo_implicita'].sum())}  |  test: casual={int(test['grupo_casual'].sum())} "
          f"implicita={int(test['grupo_implicita'].sum())}", flush=True)

    probs = {"Regresión logística": puntajes_logistica(train, i_tr, i_dev, test),
             "BERT (semilla 42)": puntajes_bert(dev, test)}

    filas, curvas = [], []
    for nombre, (p_dev, p_te) in probs.items():
        c_dev, c_te = curva(dev, p_dev), curva(test, p_te)
        completa = c_dev.merge(c_te, on="umbral", suffixes=("_dev", "_test"))
        completa.insert(0, "modelo", nombre)
        curvas.append(completa)
        # reglas: se eligen SOLO con el dev
        t_f1 = float(c_dev.loc[c_dev["f1_toxico"].idxmax(), "umbral"])
        ok = c_dev[c_dev["casual_bloqueada"] <= MAX_CASUAL]
        t_casual = float(ok["umbral"].min())
        for regla, t in [("umbral por defecto (0.50)", 0.50), ("umbral que maximiza F1 en dev", t_f1),
                         (f"umbral más bajo con casual ≤ {MAX_CASUAL:.2f} en dev", t_casual)]:
            fila_dev = c_dev[c_dev["umbral"] == t].iloc[0]
            fila_te = c_te[c_te["umbral"] == t].iloc[0]
            filas.append({"modelo": nombre, "regla": regla, "umbral": t,
                          "f1_dev": fila_dev["f1_toxico"], "casual_dev": fila_dev["casual_bloqueada"],
                          "implicita_dev": fila_dev["implicita_detectada"],
                          "f1_test": fila_te["f1_toxico"], "casual_test": fila_te["casual_bloqueada"],
                          "implicita_test": fila_te["implicita_detectada"]})
    res = pd.DataFrame(filas)
    pd.concat(curvas).to_csv(SALIDA / "resultados_umbral.csv", index=False)

    lineas = ["| Modelo | Regla para elegir el umbral (en dev) | Umbral | F1 tóxico | Grosería casual bloqueada | Toxicidad implícita detectada |",
              "|---|---|---|---|---|---|"]
    for r in res.itertuples():
        lineas.append(f"| {r.modelo} | {r.regla} | {r.umbral:.2f} | {r.f1_test:.3f} | {r.casual_test:.3f} | {r.implicita_test:.3f} |")
    md = "\n".join(lineas)
    (SALIDA / "resultados_umbral.md").write_text(md + "\n", encoding="utf-8")
    print("\nResultados en TEST (umbral elegido en dev):\n" + md)
    print("\nEn dev, con esas mismas reglas:")
    print(res[["modelo", "umbral", "f1_dev", "casual_dev", "implicita_dev"]].round(3).to_string(index=False))

    # figura: curva de las dos tasas y F1 contra umbral (test)
    naranja, cafe = "#EE8A12", "#7B3C14"
    colores = {"Regresión logística": naranja, "BERT (semilla 42)": cafe}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for completa in curvas:
        nombre = completa["modelo"].iloc[0]
        a1.plot(completa["casual_bloqueada_test"], completa["implicita_detectada_test"], color=colores[nombre], lw=2.2, label=nombre)
        a2.plot(completa["umbral"], completa["f1_toxico_test"], color=colores[nombre], lw=2.2, label=nombre)
        for regla, marca in [("umbral por defecto (0.50)", "o"), (f"umbral más bajo con casual ≤ {MAX_CASUAL:.2f} en dev", "*")]:
            r = res[(res["modelo"] == nombre) & (res["regla"] == regla)].iloc[0]
            a1.plot(r["casual_test"], r["implicita_test"], marker=marca, color=colores[nombre], ms=15 if marca == "*" else 9,
                    mec="black", ls="none")
            a2.axvline(r["umbral"], color=colores[nombre], ls=":" if marca == "o" else "--", lw=1.2)
    a1.axvline(MAX_CASUAL, color="gray", ls="--", lw=1)
    a1.text(MAX_CASUAL + 0.01, 0.04, "casual ≤ 0.25", color="gray", fontsize=9)
    a1.set_xlabel("Grosería casual bloqueada (menor es mejor)")
    a1.set_ylabel("Toxicidad implícita detectada (mayor es mejor)")
    a1.set_title("Curva de umbral en CONDA (test)\n● umbral 0.50   ★ umbral elegido en dev", fontsize=11)
    a1.set_xlim(0, 1)
    a1.set_ylim(0, 1.02)
    a1.legend(loc="lower right")
    a2.set_xlabel("Umbral de decisión")
    a2.set_ylabel("F1 de la clase tóxica (test)")
    a2.set_title("F1 tóxico según el umbral\n(líneas: 0.50 punteada, elegido en dev rayada)", fontsize=11)
    a2.set_ylim(0, 1)
    for a in (a1, a2):
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_umbral.png", dpi=150)
    plt.close(fig)
    print("\nguardado: fig_umbral.png, resultados_umbral.csv/.md")


if __name__ == "__main__":
    main()
