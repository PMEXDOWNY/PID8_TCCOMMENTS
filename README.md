# PID8 TCCOMMENTS

En Proyecto/Conjunto_de_Datos hay dos datasets reales. CONDA (Universidad de Sídney, ACL-IJCNLP 2021), 26,921 mensajes de chat de videojuego con la columna intentClass, que trae 4 categorías (O, E, A, I) y no es simplemente toxico/no toxico, hay que revisar el paper o la ficha de la fuente para saber que representa cada una antes de usarla. También trae un lexicon de jerga y palabras del juego en recursos/lexicon_refined_1209.csv. Y Jigsaw Toxic Comment (Google/Conversation AI), 159,571 comentarios de Wikipedia con 6 etiquetas binarias (toxic, severe_toxic, obscene, threat, insult, identity_hate).

Como la propuesta habla de trabajar con ambos, el primer problema real es que no comparten el mismo esquema de etiquetas: uno tiene 4 categorías de intención y el otro 6 banderas binarias independientes. Antes de poder comparar o combinar resultados entre los dos hay que decidir cómo hacerlos compatibles, por ejemplo definiendo qué categorías de CONDA cuentan como "tóxico" para poder cruzarlas contra las etiquetas de Jigsaw, y documentar esa decisión con su justificación.

También hay que limpiar cada uno por su lado. CONDA tiene 7 mensajes sin texto, 8 sin ID de jugador y 834 filas sin las columnas de slots, hay que decidir qué hacer con esos huecos. Los mensajes de CONDA son muy cortos, de chat en vivo (cosas como "wow!" o "wtf"), mientras que los de Jigsaw son comentarios completos de varias oraciones, así que conviene explorar y graficar esa diferencia de longitud entre los dos antes de meter cualquier texto a un modelo.

Revisen también qué tan desbalanceadas están las clases en cada dataset (en CONDA la categoría O domina con cerca del 74% de los mensajes, en Jigsaw menos del 10% de los comentarios están marcados como tóxicos) y grafiquen esa distribución, porque varios de los artículos de su propia matriz tratan justo el problema de desbalance de clases en este tipo de datos.

Por último, revisen cuántas de las fuentes que tienen en la matriz ya tienen su PDF correspondiente en Fuentes, la cobertura ya va muy bien pero no está completa.

30 Septiembre -> Unir ambos conjuntos de datos, revisarlos/ pre-procesarlos.

07 Octubre -> Apartado exploratorio del conjunto de datos.

## 19 de octubre

La unión y el exploratorio que quedaron en main están bien. DECISIONES.md explica la regla de is_toxic y deja marcados los duplicados en vez de borrarlos, y el exploratorio ya detectó lo importante: el split oficial de CONDA mezcla partidas entre train y valid, la longitud va en sentido contrario en cada fuente y la fuente se adivina trivialmente. Faltan tres cosas de forma. No hay FUENTE.md en Conjunto_de_Datos con la URL, la institución y la fecha de descarga de CONDA y de Jigsaw (Kaggle). El paper de CONDA (Weld et al. 2021, ACL-IJCNLP, arXiv 2106.06213) no está ni en la matriz ni en Fuentes, y es el que define las clases que están usando y trae los resultados base en su Tabla 5. Y la matriz va en 50 de 60 filas.

Ahora toca el modelado y el objetivo sale de su propia pre-propuesta: detectar mensajes tóxicos en el chat de juego sin bloquear groserías que se usan de forma casual. Eso se mide, no se declara. En CONDA hay 1,070 mensajes con una palabra del slot T que están etiquetados como no tóxicos (grosería casual) y 3,182 tóxicos que no tienen ninguna palabra del lexicon (toxicidad implícita, casi toda la clase I). Una lista de palabras bloquea casi el 100% del primer grupo y no detecta nada del segundo, y ese es el baseline contra el que se compara todo lo demás. Las tres preguntas del artículo son: cuánto mejora un modelo sobre la lista de palabras en esos dos grupos, si el contexto de la conversación ayuda (ToxBuster de Yang 2023 dice que sí con BERT, Pavlopoulos 2020 dice que no), y si conviene un modelo por fuente o uno unificado, probando también entrenar en una fuente y evaluar en la otra como hace Antypas 2023.

Los modelos son los de su matriz. Baseline con el lexicon T de CONDA, que es lo que Noever 2018 encontró como el rasgo más predictivo y lo que Davidson 2017 muestra que confunde ofensivo con odio. Después regresión logística y SVM lineal sobre TF-IDF de palabras y caracteres (van Aken 2018, Chakrabarty 2019, Abusaqer 2025), XGBoost o CatBoost sobre los mismos rasgos (Salminen 2020, Abusaqer 2025), un BiGRU o LSTM con embeddings (van Aken 2018, Roy 2024, Mohammad 2025) y BERT o RoBERTa afinado en Colab (Yang 2023, Saleh 2021, Abusaqer 2025), que es el que debería mover la clase I. En CONDA el split es siempre por partida usando group_id, nunca el oficial, y en Jigsaw usan el valid estratificado que ya hicieron. Métricas: F1 macro, F1 y recall de la clase tóxica y AUC-PR, siempre por fuente y nunca exactitud sola, más las dos tasas de los subgrupos de arriba. En CONDA reporten también las 4 clases para compararse con la Tabla 5 de Weld (Joint BERT: E 0.87, I 0.77, A 0.80, O 0.95). Para que tengan referencia, con TF-IDF y logística deberían andar cerca de 0.84 de F1 tóxico en CONDA y 0.81 en Jigsaw, con la lista de palabras en 0.65 y 0.55, y entrenar en una fuente y probar en la otra cae mucho.

Todo el modelado va en Proyecto/Modelado, con scripts que corran desde la raíz del repo, la tabla de resultados por modelo y por fuente, las figuras, y un RESULTADOS.md donde expliquen qué salió y por qué. Esa tabla es la sección de resultados del artículo de investigación.

26 Octubre -> FUENTE.md, paper de CONDA en matriz y Fuentes, lista de palabras, logística y SVM por fuente con split por partida, tabla con las tasas de grosería casual y toxicidad implícita.

02 Noviembre -> BiGRU o LSTM, BERT o RoBERTa al menos en CONDA, experimento de contexto y de transferencia entre fuentes, matriz completa con sus PDFs, todo en Proyecto/Modelado/RESULTADOS.md.
