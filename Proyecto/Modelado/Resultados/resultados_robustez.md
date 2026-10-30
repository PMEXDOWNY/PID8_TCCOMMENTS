### Test limpio (sin alterar)

| Método | F1 tóxico | Grosería casual bloqueada | Toxicidad implícita detectada |
|---|---|---|---|
| Filtro de palabras | 0.650 | 1.000 | 0.000 |
| Filtro + normalización | 0.655 | 1.000 | 0.029 |
| Logística | 0.848 | 0.373 | 0.784 |
| Logística + normalización | 0.849 | 0.373 | 0.794 |
| Logística + aumento | 0.846 | 0.349 | 0.784 |
| SVM | 0.848 | 0.354 | 0.787 |
| SVM + normalización | 0.850 | 0.358 | 0.796 |
| SVM + aumento | 0.845 | 0.340 | 0.778 |
| BERT | 0.860 | 0.439 | 0.745 |
| BERT + normalización | 0.859 | 0.434 | 0.750 |
| BERT con aumento | 0.846 | 0.410 | 0.747 |
| BERT con aumento + normalización | 0.849 | 0.410 | 0.758 |

### Detección de mensajes tóxicos con groserías alteradas (mayor es mejor)

| Método | limpio | leet | espacios | repeticion | tecla | sin_vocales | fonetica | homoglifos | mixto | creativa |
|---|---|---|---|---|---|---|---|---|---|---|
| Filtro de palabras | 1.00 | 0.19 | 0.06 | 0.09 | 0.28 | 0.20 | 0.27 | 0.39 | 0.19 | 0.27 |
| Filtro + normalización | 1.00 | 0.98 | 0.99 | 0.94 | 0.73 | 0.53 | 0.99 | 1.00 | 0.87 | 0.61 |
| Logística | 0.96 | 0.54 | 0.26 | 0.88 | 0.68 | 0.56 | 0.76 | 0.55 | 0.57 | 0.62 |
| Logística + normalización | 0.96 | 0.94 | 0.95 | 0.94 | 0.84 | 0.74 | 0.95 | 0.97 | 0.88 | 0.75 |
| Logística + aumento | 0.95 | 0.80 | 0.66 | 0.91 | 0.79 | 0.87 | 0.90 | 0.88 | 0.82 | 0.71 |
| SVM | 0.96 | 0.51 | 0.24 | 0.88 | 0.67 | 0.56 | 0.76 | 0.52 | 0.56 | 0.60 |
| SVM + normalización | 0.96 | 0.94 | 0.95 | 0.94 | 0.84 | 0.75 | 0.95 | 0.97 | 0.88 | 0.74 |
| SVM + aumento | 0.95 | 0.81 | 0.69 | 0.91 | 0.80 | 0.88 | 0.90 | 0.89 | 0.83 | 0.69 |
| BERT | 0.97 | 0.40 | 0.19 | 0.78 | 0.60 | 0.47 | 0.70 | 0.59 | 0.52 | 0.39 |
| BERT + normalización | 0.97 | 0.95 | 0.96 | 0.94 | 0.82 | 0.69 | 0.96 | 0.98 | 0.88 | 0.65 |
| BERT con aumento | 0.97 | 0.65 | 0.54 | 0.87 | 0.73 | 0.85 | 0.83 | 0.89 | 0.77 | 0.55 |
| BERT con aumento + normalización | 0.97 | 0.96 | 0.97 | 0.96 | 0.89 | 0.96 | 0.95 | 0.98 | 0.94 | 0.75 |

Mensajes por familia: limpio 842, leet 712, espacios 839, repeticion 839, tecla 776, sin_vocales 801, fonetica 466, homoglifos 805, mixto 839, creativa 801

### Falsos positivos: mensajes NO tóxicos con una palabra alterada que se marcan como tóxicos (menor es mejor)

| Método | limpio | leet | espacios | repeticion | tecla | sin_vocales | fonetica | homoglifos | mixto | creativa |
|---|---|---|---|---|---|---|---|---|---|---|
| Filtro de palabras | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.04 | 0.03 | 0.04 |
| Filtro + normalización | 0.04 | 0.05 | 0.05 | 0.06 | 0.05 | 0.07 | 0.04 | 0.05 | 0.05 | 0.05 |
| Logística | 0.05 | 0.06 | 0.05 | 0.07 | 0.06 | 0.06 | 0.07 | 0.05 | 0.06 | 0.06 |
| Logística + normalización | 0.05 | 0.07 | 0.07 | 0.07 | 0.06 | 0.07 | 0.08 | 0.07 | 0.07 | 0.06 |
| Logística + aumento | 0.05 | 0.05 | 0.08 | 0.06 | 0.07 | 0.06 | 0.06 | 0.05 | 0.07 | 0.07 |
| SVM | 0.05 | 0.06 | 0.05 | 0.07 | 0.05 | 0.05 | 0.07 | 0.05 | 0.06 | 0.06 |
| SVM + normalización | 0.05 | 0.07 | 0.07 | 0.07 | 0.06 | 0.07 | 0.09 | 0.07 | 0.07 | 0.06 |
| SVM + aumento | 0.05 | 0.05 | 0.08 | 0.06 | 0.06 | 0.06 | 0.06 | 0.05 | 0.08 | 0.06 |
| BERT | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 | 0.04 | 0.06 | 0.04 | 0.04 | 0.03 |
| BERT + normalización | 0.04 | 0.06 | 0.06 | 0.06 | 0.05 | 0.07 | 0.07 | 0.06 | 0.06 | 0.04 |
| BERT con aumento | 0.05 | 0.09 | 0.11 | 0.07 | 0.06 | 0.07 | 0.08 | 0.10 | 0.07 | 0.06 |
| BERT con aumento + normalización | 0.05 | 0.07 | 0.07 | 0.08 | 0.07 | 0.09 | 0.08 | 0.07 | 0.07 | 0.06 |

Mensajes por familia: limpio 2000, leet 1256, espacios 1295, repeticion 1309, tecla 1287, sin_vocales 1243, fonetica 418, homoglifos 1271, mixto 1304, creativa 1309
