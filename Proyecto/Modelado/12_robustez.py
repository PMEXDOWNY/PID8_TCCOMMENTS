"""Robustez a la ortografia alterada en CONDA: cuanto detecta cada modelo cuando se disfrazan las groserias, y que tanto ayuda arreglarlo.

Uso (desde la raiz del repo):
    python Proyecto/Modelado/12_robustez.py                 # todo (~3 min: el BERT con aumento YA VIENE ENTRENADO y se reutiliza)
    python Proyecto/Modelado/12_robustez.py --sin-bert      # filtro y modelos lineales (~3 min)
    python Proyecto/Modelado/12_robustez.py --solo-tablas   # regenera tablas y figura desde el CSV

Usa Demo/modelos/bert_binario.pt (el BERT binario SIN aumento, entrenado igual que el de 03_bert.py) y Demo/modelos/bert_robusto.pt (el BERT
con aumento). Los dos YA VIENEN ENTRENADOS en el repo (Git LFS). Solo si faltan: bert_binario.pt lo crea 10_entrenar_demo.py y
bert_robusto.pt lo entrena este mismo script (~7 min extra con GPU).

Que se mide (todo sobre el test de CONDA, el mismo split por partida de los demas scripts):
  - "test limpio": F1 de la clase toxica y las dos tasas, con el texto sin tocar. Sirve para ver que arreglar la ortografia no empeora lo normal.
  - "deteccion": de los mensajes toxicos CON groserias del lexicon T, que % detecta el modelo cuando se alteran esas groserias (perturbar.py).
  - "falsos positivos": de 2,000 mensajes NO toxicos, a los que se les altera una palabra al azar, que % marca como toxicos.
Familias de ortografia alterada: leet, espacios, repeticion, tecla, sin_vocales, fonetica, homoglifos y mixto (las conoce el entrenamiento
aumentado y el normalizador) y "creativa" (simbolos raros y puntuacion incrustada: NADIE la ve, mide el limite real).

Doce metodos: filtro de palabras, logistica, SVM y BERT, cada uno solo, con normalizacion del texto (normalizar.py) y con
aumento de datos (copias alteradas de ~30% de los mensajes de entrenamiento, de toxicos y no toxicos). Una sola semilla.

Salidas en Resultados/: resultados_robustez.csv, resultados_robustez.md y fig_robustez.png.
"""
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC

from comun import SALIDA, SEED, TOKEN, cargar_datos, cargar_lexicon_t, crear_vectorizadores, split_dev, split_por_fuente, tasas_grupos
from normalizar import Normalizador, construir_vocabulario as vocabulario
from perturbar import FAMILIAS, aumentar, perturbar_texto

MODELOS = Path(__file__).resolve().parent / "Demo" / "modelos"
N_BENIGNOS = 2000
MAX_LEN, EPOCAS = 64, 3
SIN_BERT = "--sin-bert" in sys.argv
ORDEN = ["Filtro de palabras", "Filtro + normalización", "Logística", "Logística + normalización", "Logística + aumento",
         "SVM", "SVM + normalización", "SVM + aumento", "BERT", "BERT + normalización", "BERT con aumento",
         "BERT con aumento + normalización"]
COLS = ["limpio"] + FAMILIAS


def construir_pruebas(test, lexicon):
    explicitos = test[(test["is_toxic"] == 1) & (test["n_lex"] > 0)]["text"].tolist()
    benignos = test[test["is_toxic"] == 0].sample(N_BENIGNOS, random_state=SEED)["text"].tolist()
    pruebas = {"limpio": {"toxicos": explicitos, "benignos": benignos}}
    for k, fam in enumerate(FAMILIAS):
        rng = np.random.default_rng(SEED + 1 + k)
        tox = [t for t, n in (perturbar_texto(t, fam, rng, lexicon) for t in explicitos) if n]
        ben = [t for t, n in (perturbar_texto(t, fam, rng, None) for t in benignos) if n]
        pruebas[fam] = {"toxicos": tox, "benignos": ben}
    return pruebas


def lineal(textos, y, modelo, C, normalizar=None):
    p, c = crear_vectorizadores()
    base = [normalizar(t) for t in textos] if normalizar else textos
    X = hstack([p.fit_transform(base), c.fit_transform(base)]).tocsr()
    clf = (LogisticRegression(C=C, solver="liblinear", class_weight="balanced", random_state=SEED) if modelo == "logistica"
           else LinearSVC(C=C, class_weight="balanced", random_state=SEED)).fit(X, y)

    def predecir(ts):
        ts = [normalizar(t) for t in ts] if normalizar else list(ts)
        return clf.predict(hstack([p.transform(ts), c.transform(ts)]).tocsr())
    return predecir


def cargar_bert(ruta):
    import torch
    import bert_util as bu
    from transformers import AutoModelForSequenceClassification
    guardado = torch.load(ruta, map_location="cpu")
    m = AutoModelForSequenceClassification.from_pretrained(bu.MODELO, num_labels=2)
    m.load_state_dict(guardado["estado"])
    return m.to(bu.DEV).eval()


def entrenar_bert_aumentado(train, lexicon):
    import torch
    import bert_util as bu
    tok = bu.tokenizador()
    i_tr, i_dev = split_dev(train, "conda")
    y = train["is_toxic"].to_numpy()
    textos = train["text"].iloc[i_tr].tolist()
    nuevos, y_nuevos = aumentar(textos, y[i_tr], lexicon, np.random.default_rng(SEED + 99), 0.3)
    print(f"   BERT con aumento: {len(textos)} mensajes + {len(nuevos)} copias alteradas", flush=True)
    items_tr = bu.codificar(tok, textos + nuevos, MAX_LEN)
    items_dev = bu.codificar(tok, train["text"].iloc[i_dev].tolist(), MAX_LEN)   # el dev NO se aumenta
    m, f1_dev = bu.entrenar(items_tr, np.concatenate([y[i_tr], y_nuevos]), items_dev, y[i_dev], 2, tok.pad_token_id, EPOCAS, SEED)
    torch.save({"estado": {k: v.half().cpu() for k, v in m.state_dict().items()}, "n_clases": 2, "max_len": MAX_LEN, "f1_dev": f1_dev},
               MODELOS / "bert_robusto.pt")
    del m
    bu.liberar()
    print(f"   bert_robusto.pt guardado, F1 dev={f1_dev:.4f}", flush=True)


def predictor_bert(ruta, normalizar=None):
    import bert_util as bu
    tok, modelo = bu.tokenizador(), cargar_bert(ruta)

    def predecir(ts):
        ts = [normalizar(t) for t in ts] if normalizar else list(ts)
        lg = bu.logits(modelo, bu.codificar(tok, ts, MAX_LEN), tok.pad_token_id)
        return ((lg[:, 1] - lg[:, 0]) > 0).astype(int)
    return predecir


def tablas_md(res):
    def pivote(medida, fmt="{:.2f}"):
        d = res[res["medida"] == medida].pivot(index="metodo", columns="familia", values="valor").reindex([m for m in ORDEN if m in set(res["metodo"])])
        cols = [c for c in COLS if c in d.columns]
        lin = ["| Método | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
        for m, r in d.iterrows():
            lin.append(f"| {m} | " + " | ".join(fmt.format(r[c]) for c in cols) + " |")
        return "\n".join(lin)
    limpio = res[res["familia"] == "-"].pivot(index="metodo", columns="medida", values="valor").reindex([m for m in ORDEN if m in set(res["metodo"])])
    a = ["| Método | F1 tóxico | Grosería casual bloqueada | Toxicidad implícita detectada |", "|---|---|---|---|"]
    for m, r in limpio.iterrows():
        a.append(f"| {m} | {r['f1_toxico']:.3f} | {r['groseria_casual_bloqueada']:.3f} | {r['toxicidad_implicita_detectada']:.3f} |")
    n = res[(res["medida"] == "deteccion_explicitos") & (res["metodo"] == res["metodo"].iloc[0])].set_index("familia")["n"]
    nf = res[(res["medida"] == "falsos_positivos_benignos") & (res["metodo"] == res["metodo"].iloc[0])].set_index("familia")["n"]
    return ("### Test limpio (sin alterar)\n\n" + "\n".join(a) + "\n\n### Detección de mensajes tóxicos con groserías alteradas (mayor es mejor)\n\n" + pivote("deteccion_explicitos") +
            "\n\nMensajes por familia: " + ", ".join(f"{k} {int(v)}" for k, v in n.items()) +
            "\n\n### Falsos positivos: mensajes NO tóxicos con una palabra alterada que se marcan como tóxicos (menor es mejor)\n\n" + pivote("falsos_positivos_benignos") +
            "\n\nMensajes por familia: " + ", ".join(f"{k} {int(v)}" for k, v in nf.items()))


def figura(res):
    metodos = [m for m in ORDEN if m in set(res["metodo"])]
    fig, ejes = plt.subplots(1, 2, figsize=(15, 0.5 * len(metodos) + 2.6))
    for ax, medida, titulo, cmap in [(ejes[0], "deteccion_explicitos", "Detección de tóxicos con groserías alteradas\n(mayor es mejor)", "Blues"),
                                     (ejes[1], "falsos_positivos_benignos", "Falsos positivos en mensajes normales alterados\n(menor es mejor)", "Reds")]:
        d = res[res["medida"] == medida].pivot(index="metodo", columns="familia", values="valor").reindex(metodos)[COLS]
        ax.imshow(d.values, vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.6 else "#222222")
        ax.set_xticks(range(len(COLS)), COLS, rotation=35, ha="right", fontsize=8.5)
        ax.set_yticks(range(len(metodos)), metodos if ax is ejes[0] else [""] * len(metodos), fontsize=9)
        ax.set_title(titulo, fontsize=10)
        ax.axvline(len(COLS) - 1.5, color="#333333", lw=1.2)   # "creativa": la que nadie ve
    fig.text(0.5, 0.01, "La columna 'creativa' (a la derecha de la línea) usa disfraces que ni el aumento ni el normalizador conocen: mide el límite real.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(SALIDA / "fig_robustez.png", dpi=150)
    plt.close(fig)


def main():
    SALIDA.mkdir(exist_ok=True)
    csv = SALIDA / "resultados_robustez.csv"
    if "--solo-tablas" in sys.argv:
        res = pd.read_csv(csv)
        (SALIDA / "resultados_robustez.md").write_text(tablas_md(res) + "\n", encoding="utf-8")
        figura(res)
        return
    t0 = time.time()
    df = cargar_datos()
    train, test = split_por_fuente(df, "conda")
    lexicon = cargar_lexicon_t()
    norm_base = Normalizador(vocabulario(df, train), lexicon)
    cache = {}
    norm = lambda t: cache.setdefault(t, norm_base(t))
    pruebas = construir_pruebas(test, lexicon)
    print("conjuntos de prueba (toxicos / benignos):", {k: (len(v["toxicos"]), len(v["benignos"])) for k, v in pruebas.items()}, flush=True)
    y_tr = train["is_toxic"].to_numpy()
    nuevos, y_nuevos = aumentar(train["text"].tolist(), y_tr, lexicon, np.random.default_rng(SEED + 7), 0.3)
    textos_aum, y_aum = train["text"].tolist() + nuevos, np.concatenate([y_tr, y_nuevos])
    print(f"entrenamiento lineal: {len(train)} mensajes + {len(nuevos)} copias alteradas ({time.time() - t0:.0f} s)", flush=True)

    filtro = lambda ts: np.array([any(w in lexicon for w in TOKEN.findall(t.lower())) for t in ts], dtype=int)
    metodos = {"Filtro de palabras": filtro, "Filtro + normalización": lambda ts: filtro([norm(t) for t in ts])}
    for nombre, tipo, C in [("Logística", "logistica", 3.0), ("SVM", "svm", 0.3)]:
        metodos[nombre] = lineal(train["text"].tolist(), y_tr, tipo, C)
        metodos[f"{nombre} + normalización"] = lineal(train["text"].tolist(), y_tr, tipo, C, norm)
        metodos[f"{nombre} + aumento"] = lineal(textos_aum, y_aum, tipo, C)
    print(f"modelos lineales listos ({time.time() - t0:.0f} s)", flush=True)
    if not SIN_BERT:
        if not (MODELOS / "bert_robusto.pt").exists():
            entrenar_bert_aumentado(train, lexicon)
        metodos["BERT"] = predictor_bert(MODELOS / "bert_binario.pt")
        metodos["BERT + normalización"] = predictor_bert(MODELOS / "bert_binario.pt", norm)
        metodos["BERT con aumento"] = predictor_bert(MODELOS / "bert_robusto.pt")
        metodos["BERT con aumento + normalización"] = predictor_bert(MODELOS / "bert_robusto.pt", norm)

    y = test["is_toxic"].to_numpy()
    filas = []
    for nombre, pred in metodos.items():
        p = pred(test["text"].tolist())
        g = tasas_grupos(test, p)
        f1_limpio = f1_score(y, p, pos_label=1)
        for medida, v in [("f1_toxico", f1_limpio), ("precision_toxico", precision_score(y, p, pos_label=1)),
                          ("recall_toxico", recall_score(y, p, pos_label=1)), ("groseria_casual_bloqueada", g["groseria_casual_bloqueada"]),
                          ("toxicidad_implicita_detectada", g["toxicidad_implicita_detectada"])]:
            filas.append({"metodo": nombre, "medida": medida, "familia": "-", "valor": v, "n": len(y)})
        for fam, d in pruebas.items():
            filas.append({"metodo": nombre, "medida": "deteccion_explicitos", "familia": fam, "valor": float(pred(d["toxicos"]).mean()), "n": len(d["toxicos"])})
            filas.append({"metodo": nombre, "medida": "falsos_positivos_benignos", "familia": fam, "valor": float(pred(d["benignos"]).mean()), "n": len(d["benignos"])})
        print(f"   {nombre}: F1 limpio={f1_limpio:.3f} ({time.time() - t0:.0f} s)", flush=True)
        pd.DataFrame(filas).to_csv(csv, index=False)   # se guarda tras cada metodo
    res = pd.DataFrame(filas)
    (SALIDA / "resultados_robustez.md").write_text(tablas_md(res) + "\n", encoding="utf-8")
    figura(res)
    print("\n" + tablas_md(res))


if __name__ == "__main__":
    main()
