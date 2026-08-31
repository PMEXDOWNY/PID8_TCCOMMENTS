# Jigsaw Toxic Comments — comentarios de Wikipedia con etiquetas de toxicidad

Material de apoyo del curso. Dataset real de referencia (benchmark) para
clasificación multi-etiqueta de toxicidad. Ver `FUENTE.md` para origen y descarga.

## Archivo

- `jigsaw_train.csv.gz` (26 MB comprimido; 66 MB descomprimido) → `gunzip -k jigsaw_train.csv.gz`

## Perfil

- **Filas:** 159,571 comentarios (inglés, páginas de discusión de Wikipedia).
- **Columnas:** `id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate` (etiquetas binarias 0/1, multi-etiqueta).
- **Longitud del texto:** media 394 caracteres, mediana 205, máximo 5,000 — textos mucho más largos que un chat de videojuego.

## Distribución de etiquetas

| Etiqueta | Positivos | % |
|---|---|---|
| toxic | 15,294 | 9.6% |
| obscene | 8,449 | 5.3% |
| insult | 7,877 | 4.9% |
| severe_toxic | 1,595 | 1.0% |
| identity_hate | 1,405 | 0.9% |
| threat | 478 | 0.3% |
| sin ninguna etiqueta (limpio) | 143,346 | 89.8% |

Fuertemente desbalanceado; las etiquetas se traslapan (un comentario puede ser
`toxic` + `obscene` + `insult` a la vez).

## Nota para el proyecto

Este dataset **no es de videojuegos**: sirve como benchmark comparativo, para
pre-entrenar/transferir modelos, o para discutir el cambio de dominio
(comentarios largos de Wikipedia vs. mensajes cortos de chat in-game como CONDA).

## Exploración rápida

```bash
gunzip -k jigsaw_train.csv.gz
python3 ../../Herramientas/explorar_dataset.py jigsaw_train.csv --texto comment_text --etiqueta toxic
```
