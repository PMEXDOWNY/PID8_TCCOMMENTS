| Modelo | F1 tóxico (todo el test) | F1 tóxico (no vistos) | AUC-PR (no vistos) | Grosería casual bloqueada (no vistos) | Toxicidad implícita detectada (no vistos) |
|---|---|---|---|---|---|
| Filtro de palabras | 0.650 | 0.691 | 0.604 | 1.000 | 0.000 |
| Regresión logística | 0.848 | 0.825 | 0.905 | 0.428 | 0.710 |
| SVM lineal | 0.848 | 0.824 | 0.901 | 0.405 | 0.714 |
| XGBoost | 0.858 | 0.833 | 0.898 | 0.410 | 0.639 |
| BiGRU (media ± desv., 3 semillas) | 0.827 ± 0.008 | 0.796 ± 0.008 | 0.871 ± 0.006 | 0.395 ± 0.042 | 0.609 ± 0.032 |
| BERT (media ± desv., 3 semillas) | 0.860 ± 0.002 | 0.835 ± 0.002 | 0.892 ± 0.011 | 0.461 ± 0.044 | 0.660 ± 0.047 |
