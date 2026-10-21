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
