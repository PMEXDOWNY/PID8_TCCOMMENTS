"""Figuras y tabla consolidada de todo el modelado, leyendo los CSV de Resultados/.

Uso (desde la raiz del repo):
    python Proyecto/Modelado/08_figuras.py

Salidas en Proyecto/Modelado/Resultados/:
    fig_modelos.png         F1 toxico y las dos tasas, todos los modelos, por fuente (barras de error = desv. entre semillas)
    fig_4clases.png         F1 por clase de CONDA (O, E, A, I) contra la Tabla 5 de Weld et al.
    fig_contexto.png        F1 toxico en CONDA segun cuantos mensajes previos se le dan al modelo
    fig_transferencia.png   AUC-PR entrenando en una fuente y evaluando en la otra (y con las dos juntas)
    resultados_todos.md     tabla consolidada por modelo y fuente
Las figuras que dependen de un CSV que no existe se omiten.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comun import SALIDA

# Paleta apta para daltonismo (Okabe-Ito)
COLOR = {"filtro_palabras": "#999999", "logistica": "#0072B2", "svm": "#E69F00", "xgboost": "#009E73", "bigru": "#CC79A7", "bert": "#D55E00"}
NOMBRE = {"filtro_palabras": "Filtro de palabras", "logistica": "Regresion logistica", "svm": "SVM lineal", "xgboost": "XGBoost", "bigru": "BiGRU", "bert": "BERT"}
METRICAS = [("f1_toxico", "F1 de la clase toxica\n(mayor es mejor)"),
            ("groseria_casual_bloqueada", "Groseria casual bloqueada\n(menor es mejor)"),
            ("toxicidad_implicita_detectada", "Toxicidad implicita detectada\n(mayor es mejor)")]
WELD = {"O": 0.954, "E": 0.872, "A": 0.800, "I": 0.768}


def leer(nombre):
    p = SALIDA / nombre
    return pd.read_csv(p) if p.exists() else None


def todos():
    """Una fila por (fuente, modelo, semilla) con las columnas comunes."""
    partes = []
    for archivo in ["resultados_lineales.csv", "resultados_xgboost.csv", "resultados_bigru.csv", "resultados_bert.csv"]:
        d = leer(archivo)
        if d is not None:
            partes.append(d)
    d = pd.concat(partes, ignore_index=True)
    cols = ["fuente", "modelo"] + [c for c, _ in METRICAS] + ["f1_macro", "recall_toxico", "auc_pr"]
    return d[cols]


def resumen(d, col):
    g = d.groupby(["fuente", "modelo"], sort=False)[col]
    return g.mean(), g.std(ddof=1).fillna(0.0)


def fig_modelos(d):
    modelos = [m for m in NOMBRE if m in set(d["modelo"])]
    fuentes = ["conda", "jigsaw"]
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    ancho = 0.8 / len(modelos)
    for ax, (col, titulo) in zip(ejes, METRICAS):
        media, sd = resumen(d, col)
        for k, m in enumerate(modelos):
            vals = [media.get((f, m), np.nan) for f in fuentes]
            err = [sd.get((f, m), 0.0) for f in fuentes]
            x = np.arange(len(fuentes)) + (k - (len(modelos) - 1) / 2) * ancho
            barras = ax.bar(x, vals, ancho, yerr=err, capsize=2, color=COLOR[m], label=NOMBRE[m], edgecolor="white", linewidth=0.5)
            ax.bar_label(barras, fmt="%.2f", fontsize=6.5, padding=2)
        ax.set_xticks(range(len(fuentes)), ["CONDA", "Jigsaw"])
        ax.set_title(titulo, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.spines[["top", "right"]].set_visible(False)
    ejes[0].set_ylabel("Valor en test")
    h, l = ejes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=len(modelos), fontsize=9, frameon=False)
    fig.text(0.5, 0.005, "BiGRU y BERT en CONDA: media de 3 semillas; BERT en Jigsaw: 1 semilla. El grupo casual de Jigsaw no mide lo mismo que el de CONDA (ver RESULTADOS.md).",
             ha="center", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(SALIDA / "fig_modelos.png", dpi=150)
    plt.close(fig)


def fig_4clases():
    d = leer("resultados_4clases.csv")
    if d is None:
        return
    clases = ["O", "E", "A", "I"]
    modelos = [m for m in ["logistica", "svm", "bert"] if m in set(d["modelo"])]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ancho = 0.8 / (len(modelos) + 1)
    for k, m in enumerate(modelos):
        g = d[d["modelo"] == m]
        vals = [g[f"f1_{c}"].mean() for c in clases]
        err = [g[f"f1_{c}"].std(ddof=1) if len(g) > 1 else 0 for c in clases]
        barras = ax.bar(np.arange(4) + (k - len(modelos) / 2) * ancho + ancho / 2, vals, ancho, yerr=err, capsize=2, color=COLOR[m], label=NOMBRE[m])
        ax.bar_label(barras, fmt="%.2f", fontsize=7, padding=2)
    barras = ax.bar(np.arange(4) + (len(modelos) / 2) * ancho + ancho / 2, [WELD[c] for c in clases], ancho, color="white", edgecolor="#333333", hatch="//", label="Weld et al. 2021 (Joint BERT, otro split)")
    ax.bar_label(barras, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(range(4), ["O (otro)", "E (explicita)", "A (accion)", "I (implicita)"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 por clase en test")
    ax.set_title("CONDA, 4 clases de intencion", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_4clases.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_contexto():
    d = leer("resultados_contexto.csv")
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for m in ["logistica", "svm", "bert"]:
        g = d[d["modelo"] == m]
        if g.empty:
            continue
        ks = sorted(g["k"].unique())
        media = [g[g["k"] == k]["f1_toxico"].mean() for k in ks]
        err = [g[g["k"] == k]["f1_toxico"].std(ddof=1) if (g["k"] == k).sum() > 1 else 0 for k in ks]
        ax.errorbar(ks, media, yerr=err, marker="o", capsize=3, color=COLOR[m], label=NOMBRE[m])
    ax.set_xticks([0, 1, 3], ["sin contexto", "1 mensaje previo", "3 mensajes previos"])
    ax.set_ylim(0.80, 0.90)
    ax.set_ylabel("F1 de la clase toxica en test (CONDA)")
    ax.set_title("Efecto del contexto de la conversacion", fontsize=11)
    ax.text(0.5, -0.17, "BERT: media de 3 semillas, barras = desviacion estandar. Los modelos lineales tienen una sola corrida (sin barras).",
            transform=ax.transAxes, ha="center", fontsize=7.5, color="#555555")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_contexto.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_transferencia():
    d = leer("resultados_transferencia.csv")
    if d is None:
        return
    modelos = [m for m in ["logistica", "svm", "bert"] if m in set(d["modelo"])]
    ent = ["conda", "jigsaw", "conjunto"]
    ev = ["conda", "jigsaw"]
    fig, ejes = plt.subplots(1, len(modelos), figsize=(3.6 * len(modelos), 3.4), squeeze=False)
    for ax, m in zip(ejes[0], modelos):
        M = np.full((3, 2), np.nan)
        for i, e in enumerate(ent):
            for j, v in enumerate(ev):
                s = d[(d["modelo"] == m) & (d["entrenado_en"] == e) & (d["evaluado_en"] == v)]
                if len(s):
                    M[i, j] = s["auc_pr"].iloc[0]
        ax.imshow(M, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        for i in range(3):
            for j in range(2):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=11, color="white" if M[i, j] > 0.55 else "#222222")
        ax.set_xticks(range(2), ["CONDA", "Jigsaw"])
        ax.set_yticks(range(3), ["CONDA", "Jigsaw", "Las dos"])
        ax.set_xlabel("Evaluado en")
        ax.set_title(NOMBRE[m], fontsize=10)
    ejes[0][0].set_ylabel("Entrenado en")
    fig.suptitle("AUC-PR al entrenar en una fuente y evaluar en la otra", fontsize=11)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_transferencia.png", dpi=150)
    plt.close(fig)


def tabla_todos(d):
    cols = [("f1_macro", "F1 macro"), ("f1_toxico", "F1 toxico"), ("recall_toxico", "Recall toxico"), ("auc_pr", "AUC-PR"),
            ("groseria_casual_bloqueada", "Groseria casual bloqueada"), ("toxicidad_implicita_detectada", "Toxicidad implicita detectada")]
    lineas = ["| Fuente | Modelo | " + " | ".join(n for _, n in cols) + " |", "|" + "---|" * (2 + len(cols))]
    for f in ["conda", "jigsaw"]:
        for m in NOMBRE:
            g = d[(d["fuente"] == f) & (d["modelo"] == m)]
            if g.empty:
                continue
            if len(g) > 1:
                celdas = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}" for c, _ in cols]
                nombre = f"{NOMBRE[m]} ({len(g)} semillas)"
            else:
                celdas = [f"{g[c].iloc[0]:.3f}" for c, _ in cols]
                nombre = NOMBRE[m] + (" (1 semilla)" if m == "bert" else "")
            lineas.append("| " + " | ".join([f.upper() if f == "conda" else "Jigsaw", nombre] + celdas) + " |")
    (SALIDA / "resultados_todos.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")


def main():
    d = todos()
    fig_modelos(d)
    fig_4clases()
    fig_contexto()
    fig_transferencia()
    tabla_todos(d)


if __name__ == "__main__":
    main()
