| Modelo | Entrenado en | Evaluado en | Prevalencia de toxicos | F1 toxico | Recall toxico | AUC-PR | Groseria casual bloqueada | Toxicidad implicita detectada |
|---|---|---|---|---|---|---|---|---|
| Regresion logistica | conda | conda (misma fuente) | 0.203 | 0.848 | 0.882 | 0.924 | 0.373 | 0.784 |
| Regresion logistica | conda | jigsaw | 0.102 | 0.283 | 0.831 | 0.542 | 0.751 | 0.571 |
| Regresion logistica | jigsaw | conda | 0.203 | 0.586 | 0.542 | 0.633 | 0.675 | 0.173 |
| Regresion logistica | jigsaw | jigsaw (misma fuente) | 0.102 | 0.805 | 0.846 | 0.905 | 0.102 | 0.616 |
| Regresion logistica | conjunto | conda | 0.203 | 0.816 | 0.891 | 0.916 | 0.575 | 0.791 |
| Regresion logistica | conjunto | jigsaw | 0.102 | 0.796 | 0.825 | 0.896 | 0.101 | 0.590 |
| SVM lineal | conda | conda (misma fuente) | 0.203 | 0.848 | 0.881 | 0.921 | 0.354 | 0.787 |
| SVM lineal | conda | jigsaw | 0.102 | 0.290 | 0.819 | 0.526 | 0.721 | 0.555 |
| SVM lineal | jigsaw | conda | 0.203 | 0.587 | 0.517 | 0.638 | 0.642 | 0.147 |
| SVM lineal | jigsaw | jigsaw (misma fuente) | 0.102 | 0.815 | 0.810 | 0.899 | 0.082 | 0.536 |
| SVM lineal | conjunto | conda | 0.203 | 0.820 | 0.893 | 0.914 | 0.557 | 0.799 |
| SVM lineal | conjunto | jigsaw | 0.102 | 0.795 | 0.817 | 0.892 | 0.100 | 0.580 |
| BERT | conda | conda (misma fuente) | 0.203 | 0.866 | 0.872 | 0.921 | 0.425 | 0.751 |
| BERT | conda | jigsaw | 0.102 | 0.637 | 0.774 | 0.779 | 0.242 | 0.437 |
| BERT | jigsaw | conda | 0.203 | 0.583 | 0.540 | 0.664 | 0.689 | 0.177 |
| BERT | jigsaw | jigsaw (misma fuente) | 0.102 | 0.844 | 0.886 | 0.932 | 0.098 | 0.706 |
| BERT | conjunto | conda | 0.203 | 0.856 | 0.875 | 0.924 | 0.439 | 0.751 |
| BERT | conjunto | jigsaw | 0.102 | 0.841 | 0.874 | 0.931 | 0.091 | 0.682 |
