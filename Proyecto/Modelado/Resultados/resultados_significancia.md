| Comparación (A contra B) | ΔF1 tóxico (A − B) | IC 95% (bootstrap por partidas) | p (bootstrap) | Solo A acierta / solo B acierta | p (McNemar exacto) |
|---|---|---|---|---|---|
| BERT contra Regresión logística | +0.012 | [+0.000, +0.023] | 0.044 | 197 / 144 | 0.005 |
| BERT contra SVM lineal | +0.012 | [+0.001, +0.023] | 0.029 | 197 / 143 | 0.004 |
| BERT contra XGBoost | +0.002 | [-0.009, +0.014] | 0.696 | 156 / 151 | 0.819 |
| BERT contra BiGRU | +0.033 | [+0.024, +0.041] | < 0.001 | 267 / 148 | < 0.001 |
| XGBoost contra Regresión logística | +0.009 | [-0.003, +0.021] | 0.129 | 183 / 135 | 0.008 |
| Regresión logística contra SVM lineal | +0.000 | [-0.005, +0.006] | 0.849 | 37 / 36 | 1.000 |
| Regresión logística contra BiGRU | +0.021 | [+0.010, +0.032] | < 0.001 | 243 / 177 | 0.001 |
| Regresión logística contra Filtro de palabras | +0.198 | [+0.169, +0.230] | < 0.001 | 678 / 256 | < 0.001 |

IC 95% del F1 tóxico de cada modelo (bootstrap por partidas): Filtro de palabras 0.650 [0.622, 0.676]; Regresión logística 0.848 [0.832, 0.864]; SVM lineal 0.848 [0.832, 0.863]; XGBoost 0.858 [0.840, 0.874]; BiGRU 0.827 [0.811, 0.843]; BERT 0.860 [0.844, 0.875]
