"""BiGRU con embeddings entrenados desde cero, por fuente (entrega del 4 de noviembre).

Uso (desde la raiz del repo):
    pip install torch pandas numpy scikit-learn
    python Proyecto/Modelado/02_bigru.py

Mismos splits, metricas y grupos que 01_baseline_y_lineales.py, para que las tablas sean comparables.
Diferencias con los lineales:
  - La red se entrena con el 80% de train y usa el 20% restante (dev) para parar a tiempo; los lineales se
    reajustan con todo train. El test no se toca hasta el final.
  - Se repite con 3 semillas (el split no cambia, solo la inicializacion y el orden de los lotes).

Salidas en Proyecto/Modelado/Resultados/:
    resultados_bigru.csv   una fila por fuente y semilla
    resultados_bigru.md    media y desviacion estandar por fuente
"""
import copy
import re
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from comun import SALIDA, SEED, cargar_datos, metricas, split_dev, split_por_fuente, tasas_grupos

SEEDS = [SEED, SEED + 1, SEED + 2]
MAX_LEN = {"conda": 40, "jigsaw": 200}  # tokens; los mensajes de CONDA son cortos, los de Jigsaw se truncan
MAX_VOCAB = 50_000
MAX_EPOCHS, PACIENCIA, BATCH, LR = 10, 2, 256, 2e-3
TOKENS = re.compile(r"[a-z0-9_']+|[^\sa-z0-9_']")  # palabras y signos sueltos (conserva emoticonos y "?")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenizar(t):
    return TOKENS.findall(t.lower())


def construir_vocab(textos):
    cuenta = Counter(w for t in textos for w in tokenizar(t))
    comunes = [w for w, c in cuenta.most_common(MAX_VOCAB) if c >= 2]
    return {w: i + 2 for i, w in enumerate(comunes)}  # 0 = relleno, 1 = desconocida


def codificar(textos, vocab, largo):
    X = np.zeros((len(textos), largo), dtype=np.int64)
    L = np.ones(len(textos), dtype=np.int64)
    for i, t in enumerate(textos):
        ids = [vocab.get(w, 1) for w in tokenizar(t)][:largo] or [1]
        X[i, : len(ids)] = ids
        L[i] = len(ids)
    return torch.from_numpy(X), torch.from_numpy(L)


class BiGRU(nn.Module):
    def __init__(self, V, emb=128, hid=128, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.gru = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(4 * hid, 1)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        paq = pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.gru(paq)
        o, _ = pad_packed_sequence(o, batch_first=True, total_length=x.size(1))
        mask = (x != 0).unsqueeze(-1)
        maximo = o.masked_fill(~mask, -1e9).max(1).values
        media = (o * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.out(self.drop(torch.cat([maximo, media], 1))).squeeze(-1)


@torch.no_grad()
def puntajes(modelo, X, L):
    modelo.eval()
    sal = [modelo(X[i:i + 1024].to(DEV), L[i:i + 1024]).cpu() for i in range(0, len(X), 1024)]
    return torch.cat(sal).numpy()


def entrenar(X_tr, L_tr, y_tr, X_dev, L_dev, y_dev, V, semilla):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = BiGRU(V).to(DEV)
    pos_w = torch.tensor((y_tr == 0).sum() / (y_tr == 1).sum(), dtype=torch.float32, device=DEV)  # equivale a "balanced"
    perdida = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-2)
    y_t = torch.from_numpy(y_tr.astype(np.float32))
    mejor, mejor_estado, sin_mejora = -1.0, None, 0
    for ep in range(1, MAX_EPOCHS + 1):
        modelo.train()
        orden = torch.randperm(len(X_tr))
        for i in range(0, len(orden), BATCH):
            b = orden[i:i + BATCH]
            opt.zero_grad()
            perdida(modelo(X_tr[b].to(DEV), L_tr[b]), y_t[b].to(DEV)).backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            opt.step()
        f1 = f1_score(y_dev, (puntajes(modelo, X_dev, L_dev) > 0).astype(int), pos_label=1)
        print(f"      epoca {ep:2d}  F1 toxico dev={f1:.4f}")
        if f1 > mejor:
            mejor, mejor_estado, sin_mejora = f1, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= PACIENCIA:
                break
    modelo.load_state_dict(mejor_estado)
    return modelo, mejor


def tabla_md(res):
    cols = ["f1_macro", "f1_toxico", "recall_toxico", "auc_pr", "groseria_casual_bloqueada", "toxicidad_implicita_detectada"]
    enc = ["Fuente", "Modelo", "F1 macro", "F1 toxico", "Recall toxico", "AUC-PR", "Groseria casual bloqueada", "Toxicidad implicita detectada"]
    lineas = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for fuente, g in res.groupby("fuente", sort=False):
        celdas = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}" for c in cols]
        lineas.append("| " + " | ".join([fuente, f"BiGRU (media ± desv. de {len(g)} semillas)"] + celdas) + " |")
    return "\n".join(lineas)


def main():
    SALIDA.mkdir(exist_ok=True)
    if "--solo-tablas" in sys.argv:
        res = pd.read_csv(SALIDA / "resultados_bigru.csv")
        (SALIDA / "resultados_bigru.md").write_text(tabla_md(res) + "\n", encoding="utf-8")
        return
    print(f"dispositivo: {DEV}")
    df = cargar_datos()
    filas = []
    for fuente in ["conda", "jigsaw"]:
        train, test = split_por_fuente(df, fuente)
        i_tr, i_dev = split_dev(train, fuente)
        vocab = construir_vocab(train["text"].iloc[i_tr])
        print(f"\n== {fuente}: train={len(i_tr)} dev={len(i_dev)} test={len(test)} vocab={len(vocab)}")
        X, L = codificar(train["text"], vocab, MAX_LEN[fuente])
        X_te, L_te = codificar(test["text"], vocab, MAX_LEN[fuente])
        y = train["is_toxic"].to_numpy()
        for semilla in SEEDS:
            t0 = time.time()
            print(f"   semilla {semilla}")
            modelo, f1_dev = entrenar(X[i_tr], L[i_tr], y[i_tr], X[i_dev], L[i_dev], y[i_dev], len(vocab) + 2, semilla)
            score = puntajes(modelo, X_te, L_te)
            pred = (score > 0).astype(int)
            filas.append({"fuente": fuente, "modelo": "bigru", "semilla": semilla, "n_train": len(i_tr),
                          "n_test": len(test), "f1_dev": f1_dev, **metricas(test["is_toxic"], pred, score),
                          **tasas_grupos(test, pred)})
            print(f"      -> test F1 toxico={filas[-1]['f1_toxico']:.4f} ({time.time() - t0:.0f} s)")
    res = pd.DataFrame(filas)
    res.to_csv(SALIDA / "resultados_bigru.csv", index=False)
    md = tabla_md(res)
    (SALIDA / "resultados_bigru.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()
