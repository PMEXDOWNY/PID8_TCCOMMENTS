"""Analisis exploratorio del conjunto unificado (hito del 7 de octubre).

Uso:
    pip install pandas numpy matplotlib
    python Proyecto/Conjunto_de_Datos/preprocesar.py     # si aun no existe el unificado
    python Proyecto/Conjunto_de_Datos/exploratorio.py

Salidas en Proyecto/Conjunto_de_Datos/Exploratorio/:
    figuras/*.png        graficas
    estadisticas.md      tablas con los numeros que usan las graficas
"""
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "Exploratorio"
FIG = OUT / "figuras"
FIG.mkdir(parents=True, exist_ok=True)

JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
COLORS = {"conda": "#2a9d8f", "jigsaw": "#e76f51"}
NAMES = {"conda": "CONDA (chat de Dota 2)", "jigsaw": "Jigsaw (comentarios de Wikipedia)"}
INTENT_NAMES = {"O": "O\nOtro", "E": "E\nExplícita", "A": "A\nAcción", "I": "I\nImplícita"}

plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

df = pd.read_csv(BASE / "Unificado" / "dataset_unificado.csv.gz", low_memory=False)
conda, jigsaw = df[df.source == "conda"], df[df.source == "jigsaw"]
md = ["# Estadísticas del análisis exploratorio\n", "Generado por `exploratorio.py`. No editar a mano.\n"]


def table(frame: pd.DataFrame) -> str:
    return "```\n" + frame.to_string() + "\n```\n"


def bar_labels(ax, bars, vals, fmt="{:.1f}%"):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=9)


# ---------- 1. Desbalance de clases ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [4, 6, 3]})

ic = conda["conda_intent"].value_counts().reindex(["O", "E", "A", "I"])
pct = ic / ic.sum() * 100
bars = axes[0].bar([INTENT_NAMES[k] for k in ic.index], pct, color=COLORS["conda"])
bar_labels(axes[0], bars, pct)
axes[0].set(title=f"CONDA: clases de intención (n={ic.sum():,})", ylabel="% de mensajes", ylim=(0, 85))

jp = jigsaw[[f"jigsaw_{c}" for c in JIGSAW_LABELS]].mean() * 100
bars = axes[1].bar([c.replace("_", "\n") for c in JIGSAW_LABELS], jp.values, color=COLORS["jigsaw"])
bar_labels(axes[1], bars, jp.values, "{:.2f}%")
axes[1].set(title=f"Jigsaw: 6 banderas (n={len(jigsaw):,})", ylabel="% de comentarios con la bandera = 1", ylim=(0, 12))

tox = df.groupby("source")["is_toxic"].mean() * 100
bars = axes[2].bar([NAMES[s].split(" (")[0] for s in tox.index], tox.values, color=[COLORS[s] for s in tox.index])
bar_labels(axes[2], bars, tox.values)
axes[2].set(title="Etiqueta común is_toxic", ylabel="% tóxico", ylim=(0, 25))
fig.suptitle("Desbalance de clases", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "01_desbalance_clases.png", bbox_inches="tight")
plt.close(fig)

imb = pd.DataFrame({"n": ic, "%": pct.round(2)})
md += ["## 1. Desbalance de clases\n", "**CONDA, intentClass**\n", table(imb)]
jt = pd.DataFrame({"n_positivos": jigsaw[[f"jigsaw_{c}" for c in JIGSAW_LABELS]].sum().astype(int).values, "%": jp.round(2).values},
                  index=JIGSAW_LABELS)
md += ["**Jigsaw, banderas**\n", table(jt)]
ratio = {s: (df[df.source == s].is_toxic == 0).sum() / (df[df.source == s].is_toxic == 1).sum() for s in ["conda", "jigsaw"]}
md += [f"Razón no tóxico : tóxico en CONDA = {ratio['conda']:.1f} : 1; en Jigsaw = {ratio['jigsaw']:.1f} : 1.\n"]
rare = jigsaw["jigsaw_threat"].sum()
md += [f"La clase más rara de Jigsaw es `threat`: {int(rare)} ejemplos ({rare / len(jigsaw) * 100:.2f}%).\n"]

# ---------- 2. Longitud ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
bins = np.logspace(0, np.log10(df.n_words.max()), 50)
for s in ["conda", "jigsaw"]:
    d = df[df.source == s]
    axes[0].hist(d.n_words, bins=bins, alpha=0.65, color=COLORS[s], label=NAMES[s], weights=np.ones(len(d)) / len(d) * 100)
axes[0].set(xscale="log", xlabel="palabras por texto (escala log)", ylabel="% de textos", title="Distribución de longitud en palabras")
axes[0].legend(fontsize=8)

data = [df[df.source == s].n_chars for s in ["conda", "jigsaw"]]
bp = axes[1].boxplot(data, tick_labels=["CONDA", "Jigsaw"], showfliers=False, patch_artist=True, widths=0.5)
for patch, s in zip(bp["boxes"], ["conda", "jigsaw"]):
    patch.set_facecolor(COLORS[s])
axes[1].set(yscale="log", ylabel="caracteres (escala log)", title="Longitud en caracteres (sin atípicos)")

pos = np.arange(2)
for i, s in enumerate(["conda", "jigsaw"]):
    d = df[df.source == s]
    med = d.groupby("is_toxic")["n_words"].median()
    b = axes[2].bar(pos + (i - 0.5) * 0.35, [med[0], med[1]], 0.35, color=COLORS[s], label=NAMES[s].split(" (")[0])
    for x, v in zip(pos + (i - 0.5) * 0.35, [med[0], med[1]]):
        axes[2].text(x, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
axes[2].set(xticks=pos, xticklabels=["no tóxico", "tóxico"], ylabel="mediana de palabras", title="Longitud según is_toxic")
axes[2].legend(fontsize=8)
fig.suptitle("Longitud de los textos: chat de juego vs. comentarios completos", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG / "02_longitud.png", bbox_inches="tight")
plt.close(fig)

lens = df.groupby("source")[["n_chars", "n_words"]].describe(percentiles=[0.5, 0.9, 0.99]).round(1)
md += ["## 2. Longitud\n", table(df.groupby("source")["n_words"].describe(percentiles=[0.5, 0.9, 0.99]).round(1)),
       "Caracteres:\n", table(df.groupby("source")["n_chars"].describe(percentiles=[0.5, 0.9, 0.99]).round(1))]
by_tox = df.groupby(["source", "is_toxic"])["n_words"].agg(["size", "median", "mean"]).round(1)
md += ["Palabras por fuente y etiqueta:\n", table(by_tox)]
one = (conda.n_words == 1).mean() * 100
md += [f"Mensajes de CONDA de una sola palabra: {one:.1f}%. Comentarios de Jigsaw de una sola palabra: {(jigsaw.n_words == 1).mean() * 100:.2f}%.\n",
       f"Jigsaw truncado a 5000 caracteres: {(jigsaw.n_chars >= 5000).sum()} comentarios en el tope.\n"]

# ---------- 3. Co-ocurrencia de banderas en Jigsaw ----------
L = jigsaw[[f"jigsaw_{c}" for c in JIGSAW_LABELS]].astype(int).values
co = L.T @ L
cond = co / np.diag(co)[:, None] * 100  # P(columna | fila)
fig, ax = plt.subplots(figsize=(6.6, 5.4))
im = ax.imshow(cond, cmap="Oranges", vmin=0, vmax=100)
ax.set_xticks(range(6), JIGSAW_LABELS, rotation=40, ha="right")
ax.set_yticks(range(6), JIGSAW_LABELS)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{cond[i, j]:.0f}", ha="center", va="center", color="white" if cond[i, j] > 60 else "black", fontsize=9)
ax.set(title="Jigsaw: si el comentario tiene la bandera de la FILA,\n¿qué % tiene también la de la COLUMNA?")
fig.colorbar(im, ax=ax, label="%")
fig.tight_layout()
fig.savefig(FIG / "03_jigsaw_coocurrencia.png", bbox_inches="tight")
plt.close(fig)
md += ["## 3. Co-ocurrencia de banderas en Jigsaw (% condicional, fila → columna)\n", table(pd.DataFrame(cond, index=JIGSAW_LABELS, columns=JIGSAW_LABELS).round(1))]
multi = (L.sum(axis=1) > 1).sum()
md += [f"Comentarios con 2 o más banderas: {multi} ({multi / len(jigsaw) * 100:.2f}% del total; {multi / (L.sum(axis=1) > 0).sum() * 100:.1f}% de los tóxicos).\n"]

# ---------- 4. Mensajes más frecuentes de CONDA ----------
txt = conda.assign(k=conda.text.str.lower())
top = txt.groupby("k").agg(n=("k", "size"), pct_tox=("is_toxic", "mean")).sort_values("n", ascending=False).head(15)
top["pct_tox"] *= 100
fig, ax = plt.subplots(figsize=(7.5, 5))
y = np.arange(len(top))[::-1]
ax.barh(y, top.n, color=[plt.cm.RdYlGn_r(v / 100) for v in top.pct_tox])
ax.set_yticks(y, top.index)
for yi, n, p in zip(y, top.n, top.pct_tox):
    ax.text(n, yi, f" {n}  ({p:.0f}% tóxico)", va="center", fontsize=8)
ax.set(xlabel="apariciones", title="CONDA: 15 mensajes más repetidos\n(color = % etiquetado tóxico)", xlim=(0, top.n.max() * 1.35))
fig.tight_layout()
fig.savefig(FIG / "04_conda_mensajes_frecuentes.png", bbox_inches="tight")
plt.close(fig)
md += ["## 4. Mensajes más repetidos en CONDA\n", table(top.round(1))]
rep = txt.groupby("k").size()
md += [f"Textos distintos en CONDA: {len(rep):,} de {len(txt):,} mensajes. Los 15 más repetidos suman {top.n.sum()} mensajes ({top.n.sum() / len(txt) * 100:.1f}%).\n"]

# ---------- 5. Vocabulario asociado a toxicidad ----------
tok = re.compile(r"[a-z][a-z']+")


def log_odds(d: pd.DataFrame, min_count: int = 30, k: int = 15):
    ct, cn = Counter(), Counter()
    for t, y_ in zip(d.text.str.lower(), d.is_toxic):
        (ct if y_ else cn).update(set(tok.findall(t)))
    nt, nn = (d.is_toxic == 1).sum(), (d.is_toxic == 0).sum()
    rows = []
    for w in set(ct) | set(cn):
        a, b = ct[w], cn[w]
        if a + b >= min_count:
            rows.append((w, a, b, np.log(((a + 0.5) / nt) / ((b + 0.5) / nn))))
    r = pd.DataFrame(rows, columns=["palabra", "en_toxicos", "en_no_toxicos", "log_odds"])
    return r.sort_values("log_odds", ascending=False).head(k).round(2), r.sort_values("log_odds").head(k).round(2)


md += ["## 5. Vocabulario por clase (log-odds, palabras con ≥30 textos)\n",
       "Cuenta en cuántos textos aparece cada palabra. Un log-odds alto significa que la palabra aparece mucho más en los tóxicos que en los no tóxicos.\n"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for ax, s in zip(axes, ["conda", "jigsaw"]):
    hi, lo = log_odds(df[df.source == s])
    md += [f"**{NAMES[s]}: más asociadas a tóxico**\n", table(hi.set_index("palabra")),
           f"**{NAMES[s]}: más asociadas a no tóxico**\n", table(lo.set_index("palabra"))]
    ax.barh(hi.palabra[::-1], hi.log_odds[::-1], color=COLORS[s])
    ax.set(title=f"{s.upper()}: palabras más ligadas a tóxico", xlabel="log-odds (tóxico vs no tóxico)")
fig.tight_layout()
fig.savefig(FIG / "05_vocabulario_toxico.png", bbox_inches="tight")
plt.close(fig)

# ---------- 6. Solapamiento y calidad ----------
q = df.groupby("source").agg(filas=("uid", "size"), textos_repetidos=("is_dup_text", "sum"), etiqueta_conflictiva=("is_conflict_text", "sum"))
q["% repetidos"] = (q.textos_repetidos / q.filas * 100).round(1)
md += ["## 6. Repetidos y contradicciones\n", table(q)]

# ---------- 7. Splits ----------
sp = df.groupby(["source", "split"])["is_toxic"].agg(["size", "mean"]).round(4)
md += ["## 7. Distribución por split\n", table(sp)]

(OUT / "estadisticas.md").write_text("\n".join(md), encoding="utf-8")
print("OK ->", OUT)
