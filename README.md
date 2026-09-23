# PID8 TCCOMMENTS

En Proyecto/Conjunto_de_Datos hay dos datasets reales. CONDA (Universidad de Sídney, ACL-IJCNLP 2021), 26,921 mensajes de chat de videojuego con la columna intentClass, que trae 4 categorías (O, E, A, I) y no es simplemente toxico/no toxico, hay que revisar el paper o la ficha de la fuente para saber que representa cada una antes de usarla. También trae un lexicon de jerga y palabras del juego en recursos/lexicon_refined_1209.csv. Y Jigsaw Toxic Comment (Google/Conversation AI), 159,571 comentarios de Wikipedia con 6 etiquetas binarias (toxic, severe_toxic, obscene, threat, insult, identity_hate).

Como la propuesta habla de trabajar con ambos, el primer problema real es que no comparten el mismo esquema de etiquetas: uno tiene 4 categorías de intención y el otro 6 banderas binarias independientes. Antes de poder comparar o combinar resultados entre los dos hay que decidir cómo hacerlos compatibles, por ejemplo definiendo qué categorías de CONDA cuentan como "tóxico" para poder cruzarlas contra las etiquetas de Jigsaw, y documentar esa decisión con su justificación.

También hay que limpiar cada uno por su lado. CONDA tiene 7 mensajes sin texto, 8 sin ID de jugador y 834 filas sin las columnas de slots, hay que decidir qué hacer con esos huecos. Los mensajes de CONDA son muy cortos, de chat en vivo (cosas como "wow!" o "wtf"), mientras que los de Jigsaw son comentarios completos de varias oraciones, así que conviene explorar y graficar esa diferencia de longitud entre los dos antes de meter cualquier texto a un modelo.

Revisen también qué tan desbalanceadas están las clases en cada dataset (en CONDA la categoría O domina con cerca del 74% de los mensajes, en Jigsaw menos del 10% de los comentarios están marcados como tóxicos) y grafiquen esa distribución, porque varios de los artículos de su propia matriz tratan justo el problema de desbalance de clases en este tipo de datos.

Por último, revisen cuántas de las fuentes que tienen en la matriz ya tienen su PDF correspondiente en Fuentes, la cobertura ya va muy bien pero no está completa.
