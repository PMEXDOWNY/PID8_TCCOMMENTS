| Modelo | Mensajes previos (k) | F1 toxico | Recall toxico | AUC-PR | Groseria casual bloqueada | Toxicidad implicita detectada | F1 toxico solo con contexto |
|---|---|---|---|---|---|---|---|
| Regresion logistica | 0 | 0.848 | 0.882 | 0.924 | 0.373 | 0.784 | 0.843 |
| Regresion logistica | 1 | 0.843 | 0.872 | 0.923 | 0.373 | 0.776 | 0.835 |
| Regresion logistica | 3 | 0.854 | 0.872 | 0.924 | 0.349 | 0.770 | 0.850 |
| SVM lineal | 0 | 0.848 | 0.881 | 0.921 | 0.354 | 0.787 | 0.842 |
| SVM lineal | 1 | 0.842 | 0.860 | 0.916 | 0.354 | 0.760 | 0.833 |
| SVM lineal | 3 | 0.846 | 0.852 | 0.918 | 0.321 | 0.737 | 0.837 |
| BERT (3 semillas) | 0 | 0.862 ± 0.005 | 0.874 ± 0.014 | 0.916 ± 0.008 | 0.407 ± 0.022 | 0.753 ± 0.030 | 0.858 ± 0.005 |
| BERT (3 semillas) | 1 | 0.859 ± 0.005 | 0.856 ± 0.020 | 0.913 ± 0.001 | 0.368 ± 0.022 | 0.726 ± 0.040 | 0.855 ± 0.005 |
| BERT (3 semillas) | 3 | 0.856 ± 0.004 | 0.878 ± 0.005 | 0.921 ± 0.003 | 0.395 ± 0.017 | 0.763 ± 0.014 | 0.854 ± 0.004 |

Mensajes del test con al menos un mensaje previo: 5565 de 7569.
BERT, F1 toxico con k=1 menos k=0, por semilla: -0.008, -0.001, +0.002 (media -0.002).
BERT, F1 toxico con k=3 menos k=0, por semilla: -0.006, -0.004, -0.006 (media -0.005).
