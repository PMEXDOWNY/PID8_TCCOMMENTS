# Fichas comparativas de datasets: toxicidad en chats de videojuegos

Material de apoyo del curso para que elijan el conjunto de datos del proyecto.
La pregunta de investigación del equipo es la detección de toxicidad en chats
de videojuegos; estas fichas resumen tres opciones para que la decisión sea de
ustedes y quede argumentada.

## Resumen rápido

| Dataset | En el repo | Dominio | Filas | Etiquetas | Idioma |
|---|---|---|---|---|---|
| CONDA | sí, `Conjunto_de_Datos/CONDA/` | chat in-game de Dota 2 | ~45 mil enunciados | intención y tokens | inglés |
| Dota 2 game chats (GosuAI) | no, requiere token de Kaggle | chat in-game de Dota 2 | ~14 millones de mensajes | ninguna | mezcla, mayoría inglés |
| Jigsaw Toxic Comments | sí, `Conjunto_de_Datos/Jigsaw_Toxic_Comments/` | comentarios de Wikipedia | ~160 mil comentarios | 6 binarias | inglés |

## 1) CONDA (CONtextual Dual-Annotated dataset)

Fuente: https://github.com/usydnlp/CONDA, publicado con un artículo de
ACL-IJCNLP 2021 (arXiv:2106.06213).

Son 44,869 enunciados (26,921 train, 8,974 valid, 8,974 test sin anotar) de
1,911 partidas de Dota 2, unos 4 MB en CSV. El texto está en inglés con jerga
de gaming ("gg", "ez", "stfu", "report X"). La anotación tiene dos niveles:
`intentClass` por enunciado (E tóxico explícito, I implícito, A acción, O
otro; en train 74% O, 13% E, 6% A, 6% I) y slots por token (T palabra tóxica,
D término de Dota, S objetivo, C separador, O otro). Incluye `matchId` y
`conversationId` para reconstruir el contexto de la conversación.

A favor: es exactamente el dominio de la pregunta, chat in-game real anotado
por humanos; distingue toxicidad explícita de implícita; cabe en cualquier
laptop y tiene un benchmark publicado en CodaLab contra el cual compararse.

En contra: solo inglés y solo Dota 2, lo que limita la generalización; el
split de test no trae etiquetas, así que tendrían que evaluar con `valid` o
re-particionar `train`; los mensajes son muy cortos y con mucho ruido
ortográfico.

Descarga exacta (ya está hecha en `Proyecto/Conjunto_de_Datos/CONDA/datos/`):

```bash
git clone https://github.com/usydnlp/CONDA.git
# archivos: CONDA/data/CONDA_{train,valid,test}.csv
```

## 2) Dota 2 game chats (GosuAI)

Fuente: https://www.kaggle.com/datasets/romovpa/gosuai-dota-2-game-chats

Alrededor de 14 millones de mensajes de chat de más de un millón de partidas
de Dota 2, unos 600 MB descomprimido. No suban el archivo completo al
repositorio porque supera el límite de 90 MB del curso. El idioma es una
mezcla con mayoría de inglés, también ruso y otros. No trae ninguna etiqueta:
es texto crudo, útil para exploración a gran escala, para pre-entrenamiento no
supervisado, o para etiquetar una muestra propia (por ejemplo aplicando el
lexicón de CONDA o con anotación manual del equipo).

A favor: volumen enorme del dominio exacto, bueno para embeddings del dominio
y estadísticas de jerga.

En contra: sin etiquetas no sirve por sí solo para entrenamiento supervisado;
requiere cuenta y token de Kaggle y es pesado para laptops modestas.

Descarga exacta (requiere token de Kaggle):

1. Crear cuenta en Kaggle; en el perfil, entrar a Settings, luego API y
   Create New Token, lo que descarga `kaggle.json`.
2. ```bash
   python3 -m pip install --user kaggle
   mkdir -p ~/.kaggle && mv ~/Descargas/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   kaggle datasets download -d romovpa/gosuai-dota-2-game-chats -p Proyecto/Conjunto_de_Datos/Dota2_GosuAI/
   unzip Proyecto/Conjunto_de_Datos/Dota2_GosuAI/gosuai-dota-2-game-chats.zip -d Proyecto/Conjunto_de_Datos/Dota2_GosuAI/
   ```
3. Documentar en un `FUENTE.md` la URL y la fecha de descarga.

## 3) Jigsaw Toxic Comment Classification Challenge

Fuente primaria: la competencia de Kaggle organizada por Jigsaw y Conversation
AI (Google), licencia CC0:
https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge

El train tiene 159,571 comentarios en inglés de páginas de discusión de
Wikipedia (66 MB en CSV, 26 MB comprimido). Trae seis etiquetas binarias que
pueden combinarse: `toxic` (9.6%), `obscene` (5.3%), `insult` (4.9%),
`severe_toxic` (1.0%), `identity_hate` (0.9%) y `threat` (0.3%); el 89.8% de
los comentarios está limpio.

A favor: es el benchmark estándar de toxicidad, con muchísima literatura para
comparar; su taxonomía distingue tipos de toxicidad (amenaza, insulto, odio
identitario); sirve para transferencia, por ejemplo pre-entrenar aquí y
ajustar con CONDA.

En contra: no es dominio gaming; son comentarios largos y relativamente
formales de Wikipedia, con vocabulario y longitud muy distintos del chat
in-game, así que hay riesgo de cambio de dominio. Además está muy
desbalanceado.

Descarga exacta. Espejo directo sin credenciales (ya descargado en
`Proyecto/Conjunto_de_Datos/Jigsaw_Toxic_Comments/`):

```bash
curl -L "https://huggingface.co/datasets/tasksource/jigsaw_toxicity/resolve/main/train.csv" -o jigsaw_train.csv
```

Alternativa oficial vía Kaggle, que incluye también el test con etiquetas
liberadas (requiere token de Kaggle y aceptar las reglas de la competencia en
la página):

```bash
kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
```

## Cómo decidir

La elección del dataset principal es de ustedes y debe quedar justificada en
la propuesta. Preguntas que ayudan a decidir: cuál coincide con el dominio de
su pregunta de investigación, cuál tiene etiquetas utilizables con los
recursos que tienen, contra qué literatura podrían comparar resultados, y qué
combinación tiene sentido (por ejemplo, un dataset principal y otro como
punto de comparación o para transferencia). Las figuras de
`Proyecto/Documentacion/02_eda_inicial.md` les dan evidencia concreta para
ese argumento.

Para explorar cualquier CSV:

```bash
python3 Proyecto/Herramientas/explorar_dataset.py <archivo> --texto <col_texto> --etiqueta <col_etiqueta>
```
