# Fichas comparativas de datasets — toxicidad en chats de videojuegos

Material de apoyo del curso para elegir el conjunto de datos del proyecto.
Pregunta de investigación del equipo: **detección de toxicidad en chats de videojuegos**.

## Resumen rápido

| Dataset | ¿Ya está en el repo? | Dominio | Filas | Etiquetas | Idioma |
|---|---|---|---|---|---|
| CONDA | ✅ `Conjunto_de_Datos/CONDA/` | Chat in-game de Dota 2 | ~45K enunciados | Sí (intención + tokens) | Inglés |
| Dota 2 game chats (GosuAI) | ❌ requiere token de Kaggle | Chat in-game de Dota 2 | ~14M mensajes | **No** (sin etiquetar) | Multi (mayoría inglés) |
| Jigsaw Toxic Comments | ✅ `Conjunto_de_Datos/Jigsaw_Toxic_Comments/` | Comentarios de Wikipedia | ~160K comentarios | Sí (6 binarias) | Inglés |

---

## 1) CONDA — CONtextual Dual-Annotated dataset

- **Fuente:** https://github.com/usydnlp/CONDA — paper ACL-IJCNLP 2021 (arXiv:2106.06213).
- **Tamaño:** 44,869 enunciados (26,921 train / 8,974 valid / 8,974 test sin anotar) de 1,911 partidas de Dota 2. ~4 MB en CSV.
- **Idioma:** inglés con jerga de gaming ("gg", "ez", "stfu", "report X").
- **Etiquetas:** doble nivel: `intentClass` por enunciado (E=tóxico explícito, I=implícito, A=acción, O=otro; 74% O / 13% E / 6% A / 6% I en train) y slots por token (T tóxico, D término de Dota, S objetivo, C separador, O otro). Incluye `matchId`/`conversationId` para modelar contexto.
- **Pros para el proyecto:**
  - Es EXACTAMENTE el dominio de la pregunta: chat in-game real, etiquetado por humanos.
  - Distingue toxicidad explícita vs. implícita y permite usar el contexto de la conversación.
  - Tamaño manejable en cualquier laptop; con benchmark publicado para comparar resultados (CodaLab).
- **Contras:**
  - Solo inglés y solo Dota 2 (generalización limitada a otros juegos/idiomas).
  - El split de test no trae etiquetas (evaluar con `valid` o re-particionar `train`).
  - Mensajes muy cortos y con mucho ruido ortográfico.
- **Descarga exacta:**
  ```bash
  git clone https://github.com/usydnlp/CONDA.git
  # archivos: CONDA/data/CONDA_{train,valid,test}.csv
  ```
  (Ya descargado en `Proyecto/Conjunto_de_Datos/CONDA/datos/`.)

---

## 2) Dota 2 game chats (GosuAI) — Kaggle `romovpa/gosuai-dota-2-game-chats`

- **Fuente:** https://www.kaggle.com/datasets/romovpa/gosuai-dota-2-game-chats
- **Tamaño:** ~14 millones de mensajes de chat de más de 1 millón de partidas de Dota 2 (~600 MB descomprimido). ¡No subir el archivo completo al repo; supera el límite de 90 MB!
- **Idioma:** mezcla (inglés mayoritario, también ruso y otros).
- **Etiquetas:** **ninguna** — es texto crudo. Útil para análisis exploratorio a gran escala, pre-entrenamiento no supervisado, o para etiquetar una muestra propia (p. ej. aplicando el lexicón de CONDA o anotación manual del equipo).
- **Pros para el proyecto:**
  - Volumen enorme del dominio exacto (chat in-game).
  - Bueno para embeddings/modelos de lenguaje del dominio y estadísticas de jerga.
- **Contras:**
  - Sin etiquetas: no sirve solo para entrenamiento supervisado.
  - Requiere cuenta y token de Kaggle; pesado para laptops modestas.
- **Descarga exacta (requiere token de Kaggle):**
  1. Crear cuenta en Kaggle → perfil → *Settings* → *API* → *Create New Token* (descarga `kaggle.json`).
  2. ```bash
     python3 -m pip install --user kaggle
     mkdir -p ~/.kaggle && mv ~/Descargas/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
     kaggle datasets download -d romovpa/gosuai-dota-2-game-chats -p Proyecto/Conjunto_de_Datos/Dota2_GosuAI/
     unzip Proyecto/Conjunto_de_Datos/Dota2_GosuAI/gosuai-dota-2-game-chats.zip -d Proyecto/Conjunto_de_Datos/Dota2_GosuAI/
     ```
  3. Documentar en un `FUENTE.md` la URL y la fecha de descarga.

---

## 3) Jigsaw Toxic Comment Classification Challenge

- **Fuente primaria:** https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge (Jigsaw / Conversation AI, Google). Licencia CC0.
- **Tamaño:** 159,571 comentarios en el train (66 MB CSV; 26 MB comprimido).
- **Idioma:** inglés (comentarios de páginas de discusión de Wikipedia).
- **Etiquetas:** 6 binarias multi-etiqueta: `toxic` (9.6%), `obscene` (5.3%), `insult` (4.9%), `severe_toxic` (1.0%), `identity_hate` (0.9%), `threat` (0.3%); 89.8% de comentarios limpios.
- **Pros para el proyecto:**
  - Benchmark estándar de toxicidad con muchísima literatura para comparar.
  - Taxonomía fina de toxicidad (amenaza, insulto, odio identitario…).
  - Útil para transferencia: pre-entrenar aquí y ajustar con CONDA.
- **Contras:**
  - **No es dominio gaming**: comentarios largos y formales de Wikipedia; el vocabulario y la longitud difieren mucho del chat in-game (riesgo de cambio de dominio).
  - Muy desbalanceado.
- **Descarga exacta:**
  - Espejo directo sin credenciales (ya descargado en `Proyecto/Conjunto_de_Datos/Jigsaw_Toxic_Comments/`):
    ```bash
    curl -L "https://huggingface.co/datasets/tasksource/jigsaw_toxicity/resolve/main/train.csv" -o jigsaw_train.csv
    ```
  - Alternativa oficial vía Kaggle (incluye también test con etiquetas liberadas):
    ```bash
    kaggle competitions download -c jigsaw-toxic-comment-classification-challenge
    ```
    (requiere token de Kaggle y aceptar las reglas de la competencia en la web).

---

## Recomendación para el equipo

- **Dataset principal:** CONDA — coincide con la pregunta de investigación y ya está etiquetado.
- **Complemento comparativo / transferencia:** Jigsaw — permite comparar contra un benchmark clásico y estudiar el cambio de dominio Wikipedia → gaming.
- **Opcional (si hay tiempo y disco):** GosuAI — para análisis a gran escala o para construir un conjunto etiquetado propio.

Para explorar cualquier CSV: `python3 Proyecto/Herramientas/explorar_dataset.py <archivo> --texto <col_texto> --etiqueta <col_etiqueta>`.
