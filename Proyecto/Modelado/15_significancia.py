"""Significancia estadistica de las diferencias entre modelos en CONDA (articulo de investigacion, seccion V-A).

Uso (desde la raiz del repo; necesita Resultados/predicciones_no_vistos.npz, que crea 14_no_vistos.py):
    python Proyecto/Modelado/15_significancia.py

Dos pruebas, las dos sobre las MISMAS predicciones del test de CONDA (7,569 mensajes de 384 partidas aprox.):
  1. Bootstrap por bloques de partidas, pareado: se remuestrean partidas completas (no mensajes sueltos, porque los mensajes de una
     partida no son independientes) y en cada remuestreo se calcula la diferencia de F1 tox entre dos modelos. Con 5,000
     remuestreos se reporta la diferencia media, su intervalo de 95% (percentiles) y un valor p bilateral aproximado. Para
     BiGRU y BERT el F1 de cada remuestreo es el promedio de sus 3 semillas, igual que en la tabla principal.
  2. Prueba de McNemar exacta (binomial) sobre los aciertos de cada mensaje, con la semilla 42 de BiGRU y BERT: compara la tasa de
     errores de dos modelos con los mismos datos. Cuenta los mensajes que uno acierta y el otro falla.

Salidas en Proyecto/Modelado/Resultados/: resultados_significancia.csv y resultados_significancia.md
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from comun import SALIDA, SEED, cargar_datos, split_por_fuente

B = 5000
MODELOS = ["Filtro de palabras", "Regresión logística", "SVM lineal", "XGBoost", "BiGRU", "BERT"]
PARES = [("BERT", "Regresión logística"), ("BERT", "SVM lineal"), ("BERT", "XGBoost"), ("BERT", "BiGRU"),
         ("XGBoost", "Regresión logística"), ("Regresión logística", "SVM lineal"), ("Regresión logística", "BiGRU"),
         ("Regresión logística", "Filtro de palabras")]


def main():
    df = cargar_datos()
    train, test = split_por_fuente(df, "conda")
    y = test["is_toxic"].to_numpy()
    z = np.load(SALIDA / "predicciones_no_vistos.npz")
    pred = {}  # modelo -> lista de vectores de prediccion (una por semilla)
    for k in z.files:
        m, _ = k.split("|")
        pred.setdefault(m, []).append((z[k] > 0).astype(int))
    grupos, idx = np.unique(test["group_id"].to_numpy(), return_inverse=True)
    n_g = len(grupos)
    print(f"test: {len(test)} mensajes en {n_g} partidas", flush=True)

    # por partida y por corrida: tp, fp, fn
    def conteos(p):
        tp = np.bincount(idx, weights=((p == 1) & (y == 1)), minlength=n_g)
        fp = np.bincount(idx, weights=((p == 1) & (y == 0)), minlength=n_g)
        fn = np.bincount(idx, weights=((p == 0) & (y == 1)), minlength=n_g)
        return tp, fp, fn

    c = {m: [conteos(p) for p in ps] for m, ps in pred.items()}
    rng = np.random.default_rng(SEED)
    pesos = rng.multinomial(n_g, np.full(n_g, 1.0 / n_g), size=B).astype(float)  # B x n_g: cuantas veces entra cada partida

    def f1_boot(m):
        f = []
        for tp, fp, fn in c[m]:
            a, b_, d = pesos @ tp, pesos @ fp, pesos @ fn
            f.append(2 * a / np.maximum(2 * a + b_ + d, 1e-12))
        return np.mean(f, axis=0)  # promedio de semillas, por remuestreo

    F = {m: f1_boot(m) for m in MODELOS}
    puntual = {m: np.mean([2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()) for tp, fp, fn in c[m]]) for m in MODELOS}

    filas = []
    for a, b_ in PARES:
        d = F[a] - F[b_]
        lo, hi = np.percentile(d, [2.5, 97.5])
        p_boot = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        # McNemar con la semilla 42 (la primera) de cada modelo
        ok_a = pred[a][0] == y
        ok_b = pred[b_][0] == y
        solo_a, solo_b = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
        mc = binomtest(solo_a, solo_a + solo_b, 0.5).pvalue if solo_a + solo_b else 1.0
        filas.append({"modelo_A": a, "modelo_B": b_, "f1_A": puntual[a], "f1_B": puntual[b_], "delta_f1": puntual[a] - puntual[b_],
                      "ic95_bajo": lo, "ic95_alto": hi, "p_bootstrap": p_boot, "solo_A_acierta": solo_a, "solo_B_acierta": solo_b,
                      "p_mcnemar": mc})
    res = pd.DataFrame(filas)
    res.to_csv(SALIDA / "resultados_significancia.csv", index=False)

    def fp(p):
        return "< 0.001" if p < 0.001 else f"{p:.3f}"

    lineas = ["| Comparación (A contra B) | ΔF1 tóxico (A − B) | IC 95% (bootstrap por partidas) | p (bootstrap) | Solo A acierta / solo B acierta | p (McNemar exacto) |",
              "|---|---|---|---|---|---|"]
    for r in res.itertuples():
        lineas.append(f"| {r.modelo_A} contra {r.modelo_B} | {r.delta_f1:+.3f} | [{r.ic95_bajo:+.3f}, {r.ic95_alto:+.3f}] | {fp(r.p_bootstrap)} | "
                      f"{r.solo_A_acierta} / {r.solo_B_acierta} | {fp(r.p_mcnemar)} |")
    ic = {m: np.percentile(F[m], [2.5, 97.5]) for m in MODELOS}
    lineas.append("")
    lineas.append("IC 95% del F1 tóxico de cada modelo (bootstrap por partidas): " + "; ".join(
        f"{m} {puntual[m]:.3f} [{ic[m][0]:.3f}, {ic[m][1]:.3f}]" for m in MODELOS))
    md = "\n".join(lineas)
    (SALIDA / "resultados_significancia.md").write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
