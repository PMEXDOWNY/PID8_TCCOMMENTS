"""Utilidades de BERT compartidas por 04_cuatro_clases.py, 05_contexto.py y 06_transferencia.py.

03_bert.py (el BERT binario por fuente, ya corrido) tiene su propia copia; esta version agrega dos cosas:
contexto (un segundo segmento con los mensajes anteriores) y un numero de clases distinto de 2.
Mismos hiperparametros que 03_bert.py: AdamW lr 2e-5, lote de 32, calentamiento 10%, fp16, peso de clase balanceado.
"""
import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

MODELO = "bert-base-uncased"
BATCH, LR, WARMUP = 32, 2e-5, 0.1
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEV.type == "cuda":
    # Con lotes de largo variable (relleno dinamico) PyTorch reserva bloques sueltos hasta ocupar toda la GPU y no deja
    # espacio para el espacio de trabajo de cuBLAS: eso produjo "CUDA error: out of memory" y CUBLAS_STATUS_EXECUTION_FAILED
    # en una GPU de 6 GB. Con el limite, el asignador libera su cache antes de llegar al tope.
    torch.cuda.set_per_process_memory_fraction(0.85)


def tokenizador():
    return AutoTokenizer.from_pretrained(MODELO)


def codificar(tok, textos, largo, contextos=None):
    """Lista de (ids, tipos). Sin contexto: [CLS] texto [SEP]. Con contexto: [CLS] contexto [SEP] texto [SEP].

    El texto del mensaje nunca se recorta por culpa del contexto; si no cabe todo, se recorta el contexto por el
    principio (se conserva lo mas reciente). Los mensajes sin contexto (primeros de su conversacion) quedan como sin contexto.
    """
    cls, sep = tok.cls_token_id, tok.sep_token_id
    objetivo = tok(list(textos), add_special_tokens=False, truncation=True, max_length=largo - 3)["input_ids"]
    cx = tok(list(contextos), add_special_tokens=False)["input_ids"] if contextos is not None else [[]] * len(objetivo)
    salida = []
    for t, c in zip(objetivo, cx):
        hueco = largo - 3 - len(t)
        c = c[-hueco:] if (hueco > 0 and c) else []
        if c:
            salida.append(([cls] + c + [sep] + t + [sep], [0] * (len(c) + 2) + [1] * (len(t) + 1)))
        else:
            salida.append(([cls] + t + [sep], [0] * (len(t) + 2)))
    return salida


def lote(items, idx, pad):
    """Rellena solo hasta el largo maximo del lote."""
    seqs = [items[i] for i in idx]
    m = max(len(s[0]) for s in seqs)
    m = -(-m // 8) * 8  # multiplo de 8: los productos de matrices en fp16 fallaron en esta GPU con largos impares (cuBLAS); lo extra es relleno enmascarado
    x = torch.full((len(seqs), m), pad, dtype=torch.long)
    t = torch.zeros((len(seqs), m), dtype=torch.long)
    for k, (ids, tipos) in enumerate(seqs):
        x[k, : len(ids)] = torch.tensor(ids)
        t[k, : len(tipos)] = torch.tensor(tipos)
    return x.to(DEV), (x != pad).long().to(DEV), t.to(DEV)


@torch.no_grad()
def logits(modelo, items, pad):
    """Logits (n, n_clases). En binario, margen toxico = logits[:, 1] - logits[:, 0]."""
    modelo.eval()
    orden = np.argsort([len(s[0]) for s in items])
    sal = None
    for i in range(0, len(orden), 256):
        b = orden[i:i + 256]
        x, m, t = lote(items, b, pad)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
            lg = modelo(input_ids=x, attention_mask=m, token_type_ids=t).logits.float().cpu().numpy()
        if sal is None:
            sal = np.zeros((len(items), lg.shape[1]), dtype=np.float32)
        sal[b] = lg
    return sal


def _f1_dev(y, lg):
    if lg.shape[1] == 2:
        return f1_score(y, (lg[:, 1] - lg[:, 0] > 0).astype(int), pos_label=1)
    return f1_score(y, lg.argmax(1), average="macro")


def entrenar(items_tr, y_tr, items_dev, y_dev, n_clases, pad, epocas, semilla):
    """Afina BERT y devuelve (modelo, mejor F1 en dev). Se conserva la mejor epoca segun dev:
    F1 de la clase toxica si son 2 clases, F1 macro si son mas."""
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODELO, num_labels=n_clases).to(DEV)
    cuentas = np.bincount(y_tr, minlength=n_clases)
    pesos = len(y_tr) / (n_clases * np.maximum(cuentas, 1))  # equivale a class_weight="balanced"
    perdida = nn.CrossEntropyLoss(weight=torch.tensor(pesos, dtype=torch.float32, device=DEV))
    opt = torch.optim.AdamW(modelo.parameters(), lr=LR, weight_decay=1e-2)
    pasos = epocas * int(np.ceil(len(items_tr) / BATCH))
    plan = get_linear_schedule_with_warmup(opt, int(WARMUP * pasos), pasos)
    escala = torch.amp.GradScaler(enabled=DEV.type == "cuda")
    y_t = torch.from_numpy(np.asarray(y_tr, dtype=np.int64))
    mejor, mejor_estado = -1.0, None
    for ep in range(1, epocas + 1):
        modelo.train()
        orden = np.random.permutation(len(items_tr))
        t0 = time.time()
        for i in range(0, len(orden), BATCH):
            b = orden[i:i + BATCH]
            for intento in range(4):
                # La GPU (6 GB) la comparte Windows con otras aplicaciones; si una toma memoria de video justo en este
                # lote, el error "out of memory" es recuperable: se libera cache, se espera y se repite el mismo lote.
                try:
                    x, m, t = lote(items_tr, b, pad)
                    opt.zero_grad()
                    with torch.autocast(DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                        lg = modelo(input_ids=x, attention_mask=m, token_type_ids=t).logits
                    escala.scale(perdida(lg.float(), y_t[b].to(DEV))).backward()
                    break
                except RuntimeError as e:
                    if intento == 3 or "out of memory" not in str(e).lower():
                        raise
                    print(f"      aviso: sin memoria en la GPU, se repite el lote ({intento + 1}/3)", flush=True)
                    x = m = t = lg = None
                    opt.zero_grad(set_to_none=True)
                    liberar()
                    time.sleep(5)
            escala.unscale_(opt)
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            escala.step(opt)
            escala.update()
            plan.step()
        f1 = _f1_dev(np.asarray(y_dev), logits(modelo, items_dev, pad))
        print(f"      epoca {ep}  F1 dev={f1:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        if f1 > mejor:
            mejor, mejor_estado = f1, copy.deepcopy({k: v.cpu() for k, v in modelo.state_dict().items()})
    modelo.load_state_dict(mejor_estado)
    return modelo, mejor


def liberar():
    """Devuelve la memoria de la GPU. Antes hay que hacer `del modelo` en quien llama: borrar la variable aqui no libera
    la referencia de quien llama, y el modelo anterior se quedaba en la GPU (provoco un error de memoria en 06)."""
    import gc
    gc.collect()
    torch.cuda.empty_cache()
