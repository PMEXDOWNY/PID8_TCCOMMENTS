# FUENTE — Jigsaw Toxic Comment Classification (train)

- **Nombre completo:** Toxic Comment Classification Challenge — conjunto de entrenamiento
- **Origen primario:** competencia de Kaggle organizada por Jigsaw / Conversation AI (Google):
  https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
- **Espejo usado para la descarga (directo, sin credenciales):**
  https://huggingface.co/datasets/tasksource/jigsaw_toxicity (archivo `train.csv`)
- **Fecha de descarga:** 2026-08-31
- **Método de descarga:**
  ```bash
  curl -L "https://huggingface.co/datasets/tasksource/jigsaw_toxicity/resolve/main/train.csv" -o jigsaw_train.csv
  ```
  El archivo se guardó comprimido como `jigsaw_train.csv.gz` (26 MB; 66 MB descomprimido).
  Para descomprimir: `gunzip -k jigsaw_train.csv.gz`.
- **Licencia:** CC0 (los textos provienen de comentarios de páginas de discusión de Wikipedia).

## Verificación

El espejo corresponde al `train.csv` original de la competencia: 159,571 filas y
las columnas `id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate`.

## Cita sugerida

> Jigsaw / Conversation AI. (2018). *Toxic Comment Classification Challenge.* Kaggle.
> https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
