| Modelo | Regla para elegir el umbral (en dev) | Umbral | F1 tóxico | Grosería casual bloqueada | Toxicidad implícita detectada |
|---|---|---|---|---|---|
| Regresión logística | umbral por defecto (0.50) | 0.50 | 0.850 | 0.330 | 0.781 |
| Regresión logística | umbral que maximiza F1 en dev | 0.68 | 0.858 | 0.264 | 0.721 |
| Regresión logística | umbral más bajo con casual ≤ 0.25 en dev | 0.70 | 0.859 | 0.255 | 0.709 |
| BERT (semilla 42) | umbral por defecto (0.50) | 0.50 | 0.860 | 0.439 | 0.745 |
| BERT (semilla 42) | umbral que maximiza F1 en dev | 0.71 | 0.868 | 0.382 | 0.719 |
| BERT (semilla 42) | umbral más bajo con casual ≤ 0.25 en dev | 0.96 | 0.851 | 0.255 | 0.636 |
