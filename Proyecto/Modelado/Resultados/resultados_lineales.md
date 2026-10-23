| Fuente | Modelo | F1 macro | F1 toxico | Recall toxico | AUC-PR | Groseria casual bloqueada | Toxicidad implicita detectada |
|---|---|---|---|---|---|---|---|
| conda | Filtro de palabras (lexicon T) | 0.789 | 0.650 | 0.548 | 0.545 | 1.000 | 0.000 |
| conda | Regresion logistica | 0.904 | 0.848 | 0.882 | 0.924 | 0.373 | 0.784 |
| conda | SVM lineal | 0.904 | 0.848 | 0.881 | 0.921 | 0.354 | 0.787 |
| jigsaw | Filtro de palabras (lexicon T) | 0.740 | 0.553 | 0.758 | 0.428 | 1.000 | 0.000 |
| jigsaw | Regresion logistica | 0.891 | 0.805 | 0.846 | 0.905 | 0.102 | 0.616 |
| jigsaw | SVM lineal | 0.897 | 0.815 | 0.810 | 0.899 | 0.082 | 0.536 |
