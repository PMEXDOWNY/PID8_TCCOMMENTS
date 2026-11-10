# PID8 TCCOMMENTS

En Proyecto/Conjunto_de_Datos hay dos datasets reales. CONDA (Universidad de Sídney, ACL-IJCNLP 2021), 26,921 mensajes de chat de videojuego con la columna intentClass, que trae 4 categorías (O, E, A, I) y no es simplemente toxico/no toxico, hay que revisar el paper o la ficha de la fuente para saber que representa cada una antes de usarla. También trae un lexicon de jerga y palabras del juego en recursos/lexicon_refined_1209.csv. Y Jigsaw Toxic Comment (Google/Conversation AI), 159,571 comentarios de Wikipedia con 6 etiquetas binarias (toxic, severe_toxic, obscene, threat, insult, identity_hate).

Como la propuesta habla de trabajar con ambos, el primer problema real es que no comparten el mismo esquema de etiquetas: uno tiene 4 categorías de intención y el otro 6 banderas binarias independientes. Antes de poder comparar o combinar resultados entre los dos hay que decidir cómo hacerlos compatibles, por ejemplo definiendo qué categorías de CONDA cuentan como "tóxico" para poder cruzarlas contra las etiquetas de Jigsaw, y documentar esa decisión con su justificación.

También hay que limpiar cada uno por su lado. CONDA tiene 7 mensajes sin texto, 8 sin ID de jugador y 834 filas sin las columnas de slots, hay que decidir qué hacer con esos huecos. Los mensajes de CONDA son muy cortos, de chat en vivo (cosas como "wow!" o "wtf"), mientras que los de Jigsaw son comentarios completos de varias oraciones, así que conviene explorar y graficar esa diferencia de longitud entre los dos antes de meter cualquier texto a un modelo.

Revisen también qué tan desbalanceadas están las clases en cada dataset (en CONDA la categoría O domina con cerca del 74% de los mensajes, en Jigsaw menos del 10% de los comentarios están marcados como tóxicos) y grafiquen esa distribución, porque varios de los artículos de su propia matriz tratan justo el problema de desbalance de clases en este tipo de datos.

Por último, revisen cuántas de las fuentes que tienen en la matriz ya tienen su PDF correspondiente en Fuentes, la cobertura ya va muy bien pero no está completa.

30 Septiembre -> Unir ambos conjuntos de datos, revisarlos/ pre-procesarlos.

07 Octubre -> Apartado exploratorio del conjunto de datos.

## 19 de octubre

La unión de los dos datasets y el exploratorio que subieron a main están bien. DECISIONES.md explica cómo definieron is_toxic y por qué marcaron los duplicados en vez de borrarlos, y en el exploratorio ya encontraron lo importante: que el split oficial de CONDA mezcla partidas entre train y valid, que la longitud de los mensajes va al revés en cada fuente y que es trivial adivinar de qué fuente viene un mensaje. Faltan tres cosas de forma. FUENTE.md en Conjunto_de_Datos con URL, institución y fecha de descarga de cada dataset. El paper original de CONDA (Weld et al. 2021, ACL-IJCNLP, arXiv 2106.06213), que no está ni en la matriz ni en Fuentes y es el que define las clases O, E, A, I que están usando. Y completar la matriz, van en 50 de 60.

Ahora toca el modelado. Aquí va punto por punto.

**Objetivo:** detectar mensajes tóxicos en el chat del juego sin bloquear las groserías que la gente dice de forma casual, que es lo que pusieron en su pre-propuesta. Para medirlo, los datos ya traen dos grupos: en CONDA hay 1,070 mensajes con alguna palabra del lexicon T (groserías) etiquetados como no tóxicos, esa es la grosería casual; y hay 3,182 mensajes tóxicos sin ninguna palabra del lexicon, casi todos de la clase I, esa es la toxicidad implícita. Un buen modelo deja pasar al primer grupo y atrapa al segundo. Las tres preguntas del artículo: cuánto mejora cada modelo sobre el filtro de palabras en esos dos grupos; si darle al modelo los mensajes anteriores de la misma partida ayuda (ToxBuster de Yang 2023 dice que sí con BERT, Pavlopoulos 2020 dice que no); y si conviene un modelo por fuente o uno entrenado con las dos juntas, probando también entrenar en una y evaluar en la otra como hace Antypas 2023.

**Modelos a usar:** de más simple a más pesado, todos de su matriz. Baseline: marcar como tóxico cualquier mensaje que tenga una palabra del lexicon T, que es lo que hace un filtro de palabras prohibidas (Noever 2018 encontró que las groserías son el rasgo más predictivo, Davidson 2017 muestra que ese filtro confunde ofensivo con tóxico). Regresión logística y SVM lineal sobre TF-IDF de palabras y de caracteres (van Aken 2018, Chakrabarty 2019, Abusaqer 2025). XGBoost o CatBoost sobre esos mismos rasgos (Salminen 2020, Abusaqer 2025). Una red BiGRU o LSTM con embeddings (van Aken 2018, Roy 2024, Mohammad 2025). Y BERT o RoBERTa afinado en Colab (Yang 2023, Saleh 2021, Abusaqer 2025), que es el que debería mejorar la clase I.

**Validación:** en CONDA separen train y test por partida usando group_id, nunca con el split oficial, porque ya vieron que mezcla partidas. En Jigsaw usen el valid estratificado que ya hicieron.

**Métricas:** siempre por fuente, nunca un solo número para las dos juntas, y nunca exactitud sola, porque con 74% de clase O cualquier cosa da alto. F1 macro, F1 y recall de la clase tóxica, AUC-PR, y las dos tasas: qué porcentaje de la grosería casual bloquea y qué porcentaje de la toxicidad implícita detecta. En CONDA reporten también las 4 clases para compararse con la Tabla 5 del paper de Weld (Joint BERT: E 0.87, I 0.77, A 0.80, O 0.95).

**Lo que deben esperar:** con TF-IDF y logística, cerca de 0.84 de F1 tóxico en CONDA y 0.81 en Jigsaw. El filtro de palabras cerca de 0.65 y 0.55, y bloquea casi toda la grosería casual sin detectar nada de lo implícito. Si les sale muy distinto, revisen antes de seguir.

**Entrega:** todo el modelado va en Proyecto/Modelado, con scripts que corran desde la raíz del repo, una tabla de resultados por modelo y por fuente, las figuras y un RESULTADOS.md donde expliquen qué salió y por qué. Esa tabla es la sección de resultados del artículo. Además documenten todo, con sus gráficas, en el PDF o en el documento donde estén llevando el artículo, porque esa es la evidencia que revisamos.

28 Octubre -> FUENTE.md, paper de CONDA en matriz y Fuentes, filtro de palabras, logística y SVM por fuente con split por partida, tabla con las dos tasas (grosería casual bloqueada, toxicidad implícita detectada).

04 Noviembre -> BiGRU o LSTM, BERT o RoBERTa al menos en CONDA, experimento de contexto y de transferencia entre fuentes, matriz completa con sus PDFs, todo en Proyecto/Modelado/RESULTADOS.md y documentado con gráficas en PDF o en el documento del artículo.

## 6 de noviembre

Revise lo del 28 de octubre y del 4 de noviembre y esta completo y muy bien hecho: FUENTE.md, paper de CONDA, matriz 60 de 60 con sus PDF, los 12 scripts, RESULTADOS.md con las tres preguntas respondidas y sus limitaciones, el PDF, la demo y el experimento de robustez. Reproduje los lineales y la tabla sale igual, y el split por partida no tiene fugas. Su BERT con 0.865 de F1 toxico y las 4 clases con I en 0.73 estan en el rango de la literatura (Weld 2021 reporta I 0.77 con otro split y etiquetas por palabra), y lo del contexto coincide con Pavlopoulos 2020.

Probe si se podia subir mas: umbral elegido en dev, rasgos del lexicon por slot, ensamble de logistica y SVM, y dos BERT preentrenados en toxicidad (toxic-bert y hateBERT). Ninguno sale de la variacion entre semillas. El techo lo pone el dato (mensajes de dos a cinco palabras), no el modelo, asi que no hace falta RoBERTa. Lo que queda son dos corridas chicas que les dan tablas para el articulo, y los dos articulos.

**Objetivo:** el mismo del 19 de octubre. Detectar toxicidad en el chat sin bloquear la groseria casual.

**Variable objetivo:** is_toxic, sin cambios.

**Variables de entrada:** las mismas. TF-IDF de palabras y caracteres para los lineales, texto para BERT.

**Modelos:** los que ya tienen. No agreguen ninguno.

**Corridas que faltan:**
1. Curva de umbral. Con la logistica y el BERT de CONDA, mover el umbral de decision y graficar groseria casual bloqueada contra toxicidad implicita detectada. El umbral se elige en el dev, nunca en el test. Reporten en una tabla el umbral por defecto y el que deja la groseria casual bloqueada en 0.25 o menos. Esta curva es el corazon de su propuesta.
2. Mensajes no vistos. Repetir la tabla principal de CONDA (seccion 2 de RESULTADOS.md) solo sobre los mensajes del test cuyo texto no aparece en train, que son 4,917 de 7,569. Es su limitacion 1 y se cierra con una tabla.

**Validacion y metricas:** las mismas. Split por partida en CONDA, F1 macro, F1 toxico, AUC-PR y las dos tasas.

**Lo que deben esperar:** en la curva, cada 10 puntos menos de groseria casual bloqueada cuestan cerca de 7 puntos de toxicidad implicita; con la logistica y casual en 0.25 o menos la implicita queda cerca de 0.71 y el F1 toxico no se mueve. En los mensajes no vistos la logistica baja de 0.848 a cerca de 0.825 de F1 toxico, la groseria casual bloqueada sube a cerca de 0.43 y la implicita baja a cerca de 0.71. Si les sale parecido, las conclusiones de RESULTADOS.md se mantienen y asi lo dicen en el articulo.

**Pendientes del repo:** agreguen las dos tablas y la figura de la curva a RESULTADOS.md y regeneren el PDF con 09_reporte_pdf.py. Nada mas.

**Articulo de investigacion:** `Plantilla_Articulo_Propuesta` tal cual, sin cambiar secciones. RESULTADOS.md ya es la seccion de resultados; el resto se arma asi:

| Seccion | Que va |
|---|---|
| Titulo | Ejemplo: "Deteccion de toxicidad en chat de videojuegos sin bloquear la groseria casual: filtro de palabras, modelos lineales y BERT en CONDA y Jigsaw" |
| Resumen | Dos fuentes, seis modelos, F1 toxico 0.865 en CONDA y 0.844 en Jigsaw, filtro de palabras 0.65 y 0.55, el contexto no ayuda, la transferencia entre fuentes cae a 0.28 y 0.59 |
| I. Introduccion | Tres contribuciones: medir por separado la groseria casual y la toxicidad implicita; probar si el contexto de la partida ayuda; probar si conviene un modelo por fuente o uno conjunto |
| II. Trabajos relacionados | Weld 2021, Yang 2023, Pavlopoulos 2020, Antypas 2023, van Aken 2018, Davidson 2017 y los demas de la matriz. Cierre: ninguno mide la groseria casual como tasa aparte |
| III. Metodologia | Datos, DECISIONES.md, definicion de los dos grupos con el lexicon, rasgos, modelos, normalizador |
| IV. Diseno experimental | Split por partida, dev, semillas, umbral elegido en dev, hardware |
| V. Resultados | Tablas de RESULTADOS.md mas la curva de umbral y la tabla de mensajes no vistos |
| VI. Discusion | Comparacion con Weld en las 4 clases, por que el contexto no ayuda aqui, la caida de la transferencia, el techo del dato y las limitaciones de la seccion 9 |
| VII. Conclusiones | Con los numeros. Futuro: mensajes de otros juegos, personas reales burlando el filtro |
| Referencias | IEEE; CONDA y Jigsaw con URL y fecha de descarga de FUENTE.md |

**Articulo de revision:** `Plantilla_Articulo_Revision` tal cual. La matriz si cambia: agreguen estas columnas para las 60 filas: ano; tipo de fuente; dominio (videojuegos, redes sociales, comentarios, otro); idioma; conjunto de datos; familia de modelo (lexico, lineal, arboles o boosting, red recurrente, transformer, LLM); usa contexto (si o no); metrica principal y valor. Registren el metodo de busqueda.

| Seccion | Que va |
|---|---|
| I. Introduccion | PR1: que modelos se usan para detectar toxicidad en texto entre 2017 y 2026 y que resultado reportan. PR2: en que dominios y conjuntos de datos se evaluan. PR3: como tratan la groseria casual, el contexto y la ortografia alterada |
| II. Metodo | Bases consultadas, cadena de busqueda, fecha, criterios de inclusion y exclusion, numeros que cuadren con las 60 filas |
| III. Panorama | Figura por ano y familia de modelo; tabla por dominio y conjunto de datos. Sale de las columnas nuevas |
| IV. Problematicas | Ofensivo contra toxico, desbalance, mensajes cortos de chat, cambio de dominio, disfraces de ortografia |
| V. Soluciones | Por familia: lexicos, lineales con TF-IDF, boosting, recurrentes, transformers, LLM |
| VI. Analisis | Que se repite (BERT gana por poco, los lexicos fallan en lo implicito) y cuando dos trabajos no son comparables |
| VII. Vacios | Pocos miden la groseria casual aparte; casi ninguno evalua en chat de videojuegos; el contexto da resultados contrarios. De aqui sale el articulo de investigacion |
| VIII. Conclusiones | Responder PR1 a PR3 y limitaciones |

20 Noviembre -> todo junto: corridas 1 y 2 en RESULTADOS.md y el PDF, matriz con las columnas nuevas y los dos articulos completos en sus plantillas. Las corridas son de un dia; dediquen el tiempo a los articulos.
