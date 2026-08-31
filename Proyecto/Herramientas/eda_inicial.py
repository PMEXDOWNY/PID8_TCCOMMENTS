#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EDA inicial sobre los datos tal como se descargaron (v0).

Genera las figuras de Proyecto/Documentacion/figuras/ que se comentan en
Proyecto/Documentacion/02_eda_inicial.md, e imprime en pantalla los numeros
en que se basa cada una.

Este script trabaja sobre los archivos v0, sin limpiar nada. La limpieza
del texto (v1) les toca a ustedes; cuando la hagan, vuelvan a correr este
script sobre sus archivos limpios y comparen las figuras.

Uso (desde la raiz del repositorio):
    python3 Proyecto/Herramientas/eda_inicial.py

Requiere pandas y matplotlib:
    python3 -m pip install --user pandas matplotlib
"""

import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:
    sys.exit(f"Falta una libreria ({exc.name}). "
             "Instalar con: python3 -m pip install --user pandas matplotlib")

# Rutas relativas a este archivo, para que funcione desde cualquier carpeta.
AQUI = Path(__file__).resolve().parent
DATOS = AQUI.parent / "Conjunto_de_Datos"
FIGURAS = AQUI.parent / "Documentacion" / "figuras"

CONDA_TRAIN = DATOS / "CONDA" / "datos" / "CONDA_train.csv"
JIGSAW_TRAIN = DATOS / "Jigsaw_Toxic_Comments" / "jigsaw_train.csv.gz"

# Orden fijo de las clases de CONDA (de mas a menos frecuente en train).
CLASES_CONDA = ["O", "E", "A", "I"]
NOMBRES_CONDA = {
    "O": "O (otro)",
    "E": "E (toxico explicito)",
    "A": "A (accion)",
    "I": "I (toxico implicito)",
}
ETIQUETAS_JIGSAW = ["toxic", "obscene", "insult", "severe_toxic",
                    "identity_hate", "threat"]

# Un solo tono para las barras: aqui el color no distingue series, solo
# pinta la magnitud, asi que no hace falta paleta.
AZUL = "#3b6fb6"
GRIS_TEXTO = "#3a3a3a"


def preparar_estilo():
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "text.color": GRIS_TEXTO,
        "axes.labelcolor": GRIS_TEXTO,
        "xtick.color": GRIS_TEXTO,
        "ytick.color": GRIS_TEXTO,
        "axes.edgecolor": "#c8c8c8",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e6e6",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    })


def guardar(fig, nombre):
    ruta = FIGURAS / nombre
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"  guardada: {ruta.relative_to(AQUI.parent.parent)}")


def figura_1_clases_conda(conda):
    conteo = conda["intentClass"].value_counts().reindex(CLASES_CONDA)
    total = conteo.sum()
    print("\nFigura 1. Balance de clases en CONDA (train):")
    for clase, n in conteo.items():
        print(f"  {NOMBRES_CONDA[clase]:<24} {n:>7,}  ({100 * n / total:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.bar([NOMBRES_CONDA[c] for c in CLASES_CONDA], conteo.values,
                    color=AZUL, width=0.6)
    for barra, n in zip(barras, conteo.values):
        ax.annotate(f"{n:,}\n({100 * n / total:.1f}%)",
                    (barra.get_x() + barra.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=9, color=GRIS_TEXTO)
    ax.set_ylabel("Enunciados")
    ax.set_ylim(0, conteo.max() * 1.18)
    ax.set_title("CONDA train: balance de clases (intentClass)")
    guardar(fig, "figura_01_conda_balance_clases.png")


def figura_2_longitud_conda(conda):
    largos = conda["utterance"].astype(str).str.len()
    print("\nFigura 2. Longitud del mensaje (caracteres) por clase, CONDA train:")
    datos, etiquetas = [], []
    for clase in CLASES_CONDA:
        sub = largos[conda["intentClass"] == clase]
        datos.append(sub.values)
        etiquetas.append(NOMBRES_CONDA[clase])
        print(f"  {NOMBRES_CONDA[clase]:<24} mediana={sub.median():>4.0f}  "
              f"media={sub.mean():>5.1f}  p95={sub.quantile(0.95):>5.0f}  "
              f"max={sub.max():>4}")
    print(f"  {'todas las clases':<24} mediana={largos.median():>4.0f}  "
          f"media={largos.mean():>5.1f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    partes = ax.violinplot(datos, showmedians=True, showextrema=False)
    for cuerpo in partes["bodies"]:
        cuerpo.set_facecolor(AZUL)
        cuerpo.set_alpha(0.6)
    partes["cmedians"].set_color(GRIS_TEXTO)
    ax.set_xticks(range(1, len(etiquetas) + 1))
    ax.set_xticklabels(etiquetas)
    ax.set_ylabel("Longitud del mensaje (caracteres)")
    # La cola larga aplasta los violines; el eje logaritmico deja ver
    # donde vive la mayoria de los mensajes.
    ax.set_yscale("log")
    ax.set_title("CONDA train: longitud del mensaje por clase (eje log)")
    guardar(fig, "figura_02_conda_longitud_por_clase.png")


def figura_3_etiquetas_jigsaw(jigsaw):
    conteo = jigsaw[ETIQUETAS_JIGSAW].sum().sort_values(ascending=True)
    total = len(jigsaw)
    print("\nFigura 3. Positivos por etiqueta en Jigsaw (train):")
    for etiqueta, n in conteo.sort_values(ascending=False).items():
        print(f"  {etiqueta:<15} {int(n):>7,}  ({100 * n / total:.1f}%)")

    fig, ax = plt.subplots(figsize=(7, 4))
    barras = ax.barh(conteo.index, conteo.values, color=AZUL, height=0.6)
    for barra, n in zip(barras, conteo.values):
        ax.annotate(f" {int(n):,} ({100 * n / total:.1f}%)",
                    (n, barra.get_y() + barra.get_height() / 2),
                    va="center", fontsize=9, color=GRIS_TEXTO)
    ax.set_xlabel("Comentarios positivos (de 159,571)")
    ax.set_xlim(0, conteo.max() * 1.35)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    ax.set_title("Jigsaw train: comentarios positivos por etiqueta")
    guardar(fig, "figura_03_jigsaw_conteo_etiquetas.png")


def figura_4_limpios_vs_toxicos(jigsaw):
    alguna = (jigsaw[ETIQUETAS_JIGSAW].sum(axis=1) > 0)
    n_toxicos = int(alguna.sum())
    n_limpios = len(jigsaw) - n_toxicos
    total = len(jigsaw)
    print("\nFigura 4. Limpios vs con alguna etiqueta, Jigsaw (train):")
    print(f"  limpios            {n_limpios:>8,}  ({100 * n_limpios / total:.1f}%)")
    print(f"  con alguna etiqueta{n_toxicos:>8,}  ({100 * n_toxicos / total:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    categorias = ["Limpio\n(sin etiquetas)", "Con alguna etiqueta\nde toxicidad"]
    valores = [n_limpios, n_toxicos]
    barras = ax.bar(categorias, valores, color=AZUL, width=0.5)
    for barra, n in zip(barras, valores):
        ax.annotate(f"{n:,}\n({100 * n / total:.1f}%)",
                    (barra.get_x() + barra.get_width() / 2, n),
                    ha="center", va="bottom", fontsize=9, color=GRIS_TEXTO)
    ax.set_ylabel("Comentarios")
    ax.set_ylim(0, max(valores) * 1.22)
    ax.set_title("Jigsaw train: comentarios limpios vs con toxicidad")
    guardar(fig, "figura_04_jigsaw_limpios_vs_toxicos.png")


def figura_5_coocurrencia_jigsaw(jigsaw):
    etiquetas = jigsaw[ETIQUETAS_JIGSAW].astype(int)
    matriz = etiquetas.T.dot(etiquetas)
    print("\nFigura 5. Co-ocurrencia de etiquetas en Jigsaw (train):")
    print(matriz.to_string())

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    im = ax.imshow(matriz.values, cmap="Blues")
    n_etq = len(ETIQUETAS_JIGSAW)
    ax.set_xticks(range(n_etq))
    ax.set_yticks(range(n_etq))
    ax.set_xticklabels(ETIQUETAS_JIGSAW, rotation=45, ha="right")
    ax.set_yticklabels(ETIQUETAS_JIGSAW)
    ax.grid(False)
    umbral = matriz.values.max() * 0.55
    for i in range(n_etq):
        for j in range(n_etq):
            valor = matriz.values[i, j]
            ax.text(j, i, f"{valor:,}", ha="center", va="center", fontsize=8,
                    color="white" if valor > umbral else GRIS_TEXTO)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Comentarios en comun")
    ax.set_title("Jigsaw train: co-ocurrencia de etiquetas\n"
                 "(diagonal = total de cada etiqueta)")
    guardar(fig, "figura_05_jigsaw_coocurrencia.png")


def main():
    for ruta in (CONDA_TRAIN, JIGSAW_TRAIN):
        if not ruta.exists():
            sys.exit(f"No se encontro {ruta}. Revisen que los datasets esten "
                     "descargados donde indica su FUENTE.md.")
    FIGURAS.mkdir(parents=True, exist_ok=True)
    preparar_estilo()

    print(f"Leyendo {CONDA_TRAIN.name} y {JIGSAW_TRAIN.name}...")
    conda = pd.read_csv(CONDA_TRAIN)
    jigsaw = pd.read_csv(JIGSAW_TRAIN)
    print(f"CONDA train : {len(conda):,} filas")
    print(f"Jigsaw train: {len(jigsaw):,} filas")

    figura_1_clases_conda(conda)
    figura_2_longitud_conda(conda)
    figura_3_etiquetas_jigsaw(jigsaw)
    figura_4_limpios_vs_toxicos(jigsaw)
    figura_5_coocurrencia_jigsaw(jigsaw)

    print("\nListo. Las figuras quedaron en Proyecto/Documentacion/figuras/.")
    print("La lectura de cada figura esta en Proyecto/Documentacion/02_eda_inicial.md.")


if __name__ == "__main__":
    main()
