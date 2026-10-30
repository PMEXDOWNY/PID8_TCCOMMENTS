"""BERT afinado (bert-base-uncased) por fuente (entrega del 4 de noviembre).

Uso (desde la raiz del repo):
    pip install torch transformers pandas numpy scikit-learn
    python Proyecto/Modelado/03_bert.py                              # CONDA con 3 semillas y Jigsaw con 1
    python Proyecto/Modelado/03_bert.py --fuentes conda --semillas 42
    python Proyecto/Modelado/03_bert.py --solo-tablas                # regenera el .md desde el CSV

Mismos splits, dev, metricas y grupos que 01_baseline_y_lineales.py y 02_bigru.py. Igual que el BiGRU, se
entrena con el 80% de train y el 20% restante (dev) sirve para elegir la epoca; el test no se toca hasta el final.

Salidas en Proyecto/Modelado/Resultados/:
    resultados_bert.csv   una fila por fuente y semilla
    resultados_bert.md    media y desviacion estandar por fuente
"""
import argparse
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from comun import SALIDA, SEED, cargar_datos, metricas, split_dev, split_por_fuente, tasas_grupos

MODELO = "bert-base-uncased"
MAX_LEN = {"conda": 64, "jigsaw": 128}  # tokens de WordPiece; los comentarios largos de Jigsaw se truncan
EPOCAS = {"conda": 3, "jigsaw": 2}
SEMILLAS = {"conda": [SEED, SEED + 1, SEED + 2], "jigsaw": [SEED]}  # Jigsaw tarda mucho mas por semilla
BATCH, LR, WARMUP = 32, 2e-5, 0.1
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def codificar(tok, textos, largo):
    return tok(list(textos), truncation=True, max_length=largo)["input_ids"]


def lote(ids, idx, pad):
    """Rellena solo hasta el largo maximo del lote (mucho mas rapido que rellenar siempre a MAX_LEN)."""
    seqs = [ids[i] for i in idx]
    m = max(len(s) for s in seqs)
    x = torch.full((len(seqs), m), pad, dtype=torch.long)
    for k, s in enumerate(seqs):
        x[k, : len(s)] = torch.tensor(s)
    return x.to(DEV), (x != pad).long().to(DEV)


@torch.no_grad()
def puntajes(modelo, ids, pad):
    """Margen entre las dos clases (logit tóxico - logit no tóxico); > 0 equivale a predecir tóxico."""
    modelo.eval()
    orden = np.argsort([len(s) for s in ids])  # agrupa longitudes parecidas
    sal = np.zeros(len(ids), dtype=np.float32)
    for i in range(0, len(orden), 256):
        b = orden[i:i + 256]
        x, m = lote(ids, b, pad)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            lg = modelo(input_ids=x, attention_mask=m).logits.float()
        sal[b] = (lg[:, 1] - lg[:, 0]).cpu().numpy()
    return sal


def entrenar(ids_tr, y_tr, ids_dev, y_dev, pad, epocas, semilla):
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO, num_labels=2).to(DEV)
    pos_w = float((y_tr == 0).sum() / (y_tr == 1).sum())  # equivale a class_weight="balanced"
    perdida = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_w], device=DEV))
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-2)
    pasos = epocas * int(np.ceil(len(ids_tr) / BATCH))
    plan = get_linear_schedule_with_warmup(opt, int(WARMUP * pasos), pasos)
    escala = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    y_t = torch.from_numpy(y_tr.astype(np.int64))
    mejor, mejor_estado = -1.0, None
    for ep in range(1, epocas + 1):
        modelo.train()
        orden = np.random.permutation(len(ids_tr))
        t0 = time.time()
        for i in range(0, len(orden), BATCH):
            b = orden[i:i + BATCH]
            x, m = lote(ids_tr, b, pad)
            opt.zero_grad()
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                lg = modelo(input_ids=x, attention_mask=m).logits
            escala.scale(perdida(lg.float(), y_t[b].to(DEV))).backward()
            escala.unscale_(opt)
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escala.step(opt)
            escala.update()
            plan.step()
        f1 = f1_score(y_dev, (puntajes(modelo, ids_dev, pad) > 0).astype(int), pos_label=1)
        print(f"      epoca {ep}  F1 toxico dev={f1:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if f1 > mejor:
            mejor, mejor_estado = f1, copy.deepcopy({k: v.cpu() for k, v in modelo.state_dict().items()})
    modelo.load_state_dict(mejor_estado)
    return modelo, mejor


def tabla_md(res):
    cols = ["f1_macro", "f1_toxico", "recall_toxico", "auc_pr", "groseria_casual_bloqueada", "toxicidad_implicita_detectada"]
    enc = ["Fuente", "Modelo", "F1 macro", "F1 toxico", "Recall toxico", "AUC-PR", "Groseria casual bloqueada", "Toxicidad implicita detectada"]
    lineas = ["| " + " | ".join(enc) + " |", "|" + "---|" * len(enc)]
    for fuente, g in res.groupby("fuente", sort=False):
        if len(g) > 1:
            celdas = [f"{g[c].mean():.3f} ± {g[c].std(ddof=1):.3f}" for c in cols]
            nombre = f"BERT (media ± desv. de {len(g)} semillas)"
        else:
            celdas = [f"{g[c].iloc[0]:.3f}" for c in cols]
            nombre = "BERT (1 semilla)"
        lineas.append("| " + " | ".join([fuente, nombre] + celdas) + " |")
    return "\n".join(lineas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuentes", nargs="+", default=["conda", "jigsaw"])
    ap.add_argument("--semillas", nargs="+", type=int, default=None, help="reemplaza las semillas por defecto de todas las fuentes")
    ap.add_argument("--solo-tablas", action="store_true")
    a = ap.parse_args()
    SALIDA.mkdir(exist_ok=True)
    csv = SALIDA / "resultados_bert.csv"
    if a.solo_tablas:
        res = pd.read_csv(csv)
        (SALIDA / "resultados_bert.md").write_text(tabla_md(res) + "\n", encoding="utf-8")
        return
    print(f"dispositivo: {DEV}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODELO)
    df = cargar_datos()
    filas = pd.read_csv(csv).to_dict("records") if csv.exists() else []  # conserva lo ya corrido
    for fuente in a.fuentes:
        train, test = split_por_fuente(df, fuente)
        i_tr, i_dev = split_dev(train, fuente)
        print(f"\n== {fuente}: train={len(i_tr)} dev={len(i_dev)} test={len(test)}", flush=True)
        ids = codificar(tok, train["text"], MAX_LEN[fuente])
        ids_te = codificar(tok, test["text"], MAX_LEN[fuente])
        y = train["is_toxic"].to_numpy()
        for semilla in (a.semillas or SEMILLAS[fuente]):
            t0 = time.time()
            print(f"   semilla {semilla}", flush=True)
            modelo, f1_dev = entrenar([ids[i] for i in i_tr], y[i_tr], [ids[i] for i in i_dev], y[i_dev],
                                      tok.pad_token_id, EPOCAS[fuente], semilla)
            score = puntajes(modelo, ids_te, tok.pad_token_id)
            pred = (score > 0).astype(int)
            filas = [f for f in filas if not (f["fuente"] == fuente and f["semilla"] == semilla)]
            filas.append({"fuente": fuente, "modelo": "bert", "semilla": semilla, "n_train": len(i_tr),
                          "n_test": len(test), "f1_dev": f1_dev, **metricas(test["is_toxic"], pred, score),
                          **tasas_grupos(test, pred)})
            print(f"      -> test F1 toxico={filas[-1]['f1_toxico']:.4f} ({time.time() - t0:.0f} s)", flush=True)
            res = pd.DataFrame(filas)
            res.to_csv(csv, index=False)  # se guarda tras cada semilla por si se corta
            (SALIDA / "resultados_bert.md").write_text(tabla_md(res) + "\n", encoding="utf-8")
            del modelo
            torch.cuda.empty_cache()
    print("\n" + tabla_md(pd.DataFrame(filas)))


if __name__ == "__main__":
    main()
