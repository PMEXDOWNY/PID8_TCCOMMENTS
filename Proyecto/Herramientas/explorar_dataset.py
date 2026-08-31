#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explorador rápido de datasets (material de apoyo del curso).

Carga un dataset tabular (CSV, TSV, parquet, o .gz de los anteriores) e imprime
un perfil básico: filas, columnas, balance de clases, longitud de los textos y
una muestra aleatoria.

Uso:
    python3 explorar_dataset.py ARCHIVO [--texto COL] [--etiqueta COL] [--muestra N]

Ejemplos:
    python3 explorar_dataset.py ../Conjunto_de_Datos/CONDA/datos/CONDA_train.csv \
        --texto utterance --etiqueta intentClass
    python3 explorar_dataset.py ../Conjunto_de_Datos/Jigsaw_Toxic_Comments/jigsaw_train.csv.gz \
        --texto comment_text --etiqueta toxic

Si no se indican --texto/--etiqueta, el script intenta adivinarlos con nombres
comunes (utterance, comment_text, text / intentClass, toxic, label...).
"""

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("Falta pandas. Instalar con: python3 -m pip install --user pandas")

CANDIDATOS_TEXTO = ["utterance", "comment_text", "text", "texto", "message", "chat", "key"]
CANDIDATOS_ETIQUETA = ["intentClass", "toxic", "label", "etiqueta", "class", "target"]


def cargar(ruta: Path) -> pd.DataFrame:
    """Carga CSV/TSV/parquet, con o sin compresión .gz (pandas la detecta sola)."""
    nombre = ruta.name.lower()
    if nombre.endswith((".parquet", ".pq")):
        return pd.read_parquet(ruta)
    if ".tsv" in nombre:
        return pd.read_csv(ruta, sep="\t")
    return pd.read_csv(ruta)  # cubre .csv y .csv.gz


def adivinar(columnas, candidatos):
    por_minusculas = {c.lower(): c for c in columnas}
    for cand in candidatos:
        if cand.lower() in por_minusculas:
            return por_minusculas[cand.lower()]
    return None


def main():
    ap = argparse.ArgumentParser(description="Perfil rápido de un dataset tabular.")
    ap.add_argument("archivo", type=Path, help="ruta al CSV/TSV/parquet (.gz permitido)")
    ap.add_argument("--texto", help="columna con el texto")
    ap.add_argument("--etiqueta", help="columna con la etiqueta/clase")
    ap.add_argument("--muestra", type=int, default=5, help="tamaño de la muestra aleatoria (def. 5)")
    ap.add_argument("--semilla", type=int, default=42, help="semilla de la muestra (def. 42)")
    args = ap.parse_args()

    if not args.archivo.exists():
        sys.exit(f"No existe el archivo: {args.archivo}")

    df = cargar(args.archivo)

    print("=" * 70)
    print(f"Archivo : {args.archivo}")
    print(f"Filas   : {len(df):,}")
    print(f"Columnas: {len(df.columns)} -> {list(df.columns)}")

    print("-" * 70)
    print("Tipos y valores faltantes:")
    for col in df.columns:
        faltan = df[col].isna().sum()
        print(f"  {col:<20} {str(df[col].dtype):<10} faltantes={faltan:,}")

    col_etq = args.etiqueta or adivinar(df.columns, CANDIDATOS_ETIQUETA)
    if col_etq and col_etq in df.columns:
        print("-" * 70)
        print(f"Balance de clases ({col_etq}):")
        conteo = df[col_etq].value_counts(dropna=False)
        for valor, n in conteo.items():
            print(f"  {str(valor):<15} {n:>8,}  ({100 * n / len(df):5.1f}%)")
    else:
        print("-" * 70)
        print("(No se encontró columna de etiqueta; usar --etiqueta COL)")

    col_txt = args.texto or adivinar(df.columns, CANDIDATOS_TEXTO)
    if col_txt and col_txt in df.columns:
        largos = df[col_txt].astype(str).str.len()
        print("-" * 70)
        print(f"Longitud del texto ({col_txt}), en caracteres:")
        print(f"  media={largos.mean():.1f}  mediana={largos.median():.0f}  "
              f"min={largos.min()}  max={largos.max()}")
        print("-" * 70)
        print(f"Muestra aleatoria ({args.muestra} filas):")
        muestra = df.sample(min(args.muestra, len(df)), random_state=args.semilla)
        for _, fila in muestra.iterrows():
            texto = str(fila[col_txt])[:100].replace("\n", " ")
            etq = f" [{fila[col_etq]}]" if col_etq and col_etq in df.columns else ""
            print(f"  -{etq} {texto}")
    else:
        print("-" * 70)
        print("(No se encontró columna de texto; usar --texto COL)")
    print("=" * 70)


if __name__ == "__main__":
    main()
