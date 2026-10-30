| Fuente | Modelo | F1 macro | F1 toxico | Recall toxico | AUC-PR | Groseria casual bloqueada | Toxicidad implicita detectada |
|---|---|---|---|---|---|---|---|
| CONDA | Filtro de palabras | 0.789 | 0.650 | 0.548 | 0.545 | 1.000 | 0.000 |
| CONDA | Regresion logistica | 0.904 | 0.848 | 0.882 | 0.924 | 0.373 | 0.784 |
| CONDA | SVM lineal | 0.904 | 0.848 | 0.881 | 0.921 | 0.354 | 0.787 |
| CONDA | XGBoost | 0.911 | 0.858 | 0.857 | 0.921 | 0.354 | 0.732 |
| CONDA | BiGRU (3 semillas) | 0.891 ± 0.006 | 0.827 ± 0.008 | 0.828 ± 0.018 | 0.900 ± 0.005 | 0.344 ± 0.033 | 0.712 ± 0.025 |
| CONDA | BERT (3 semillas) | 0.915 ± 0.004 | 0.865 ± 0.006 | 0.865 ± 0.011 | 0.910 ± 0.018 | 0.379 ± 0.027 | 0.739 ± 0.022 |
| Jigsaw | Filtro de palabras | 0.740 | 0.553 | 0.758 | 0.428 | 1.000 | 0.000 |
| Jigsaw | Regresion logistica | 0.891 | 0.805 | 0.846 | 0.905 | 0.102 | 0.616 |
| Jigsaw | SVM lineal | 0.897 | 0.815 | 0.810 | 0.899 | 0.082 | 0.536 |
| Jigsaw | XGBoost | 0.882 | 0.790 | 0.828 | 0.886 | 0.096 | 0.588 |
| Jigsaw | BiGRU (3 semillas) | 0.890 ± 0.003 | 0.803 ± 0.005 | 0.834 ± 0.007 | 0.894 ± 0.002 | 0.094 ± 0.002 | 0.573 ± 0.025 |
| Jigsaw | BERT (1 semilla) | 0.911 | 0.841 | 0.885 | 0.932 | 0.105 | 0.700 |
