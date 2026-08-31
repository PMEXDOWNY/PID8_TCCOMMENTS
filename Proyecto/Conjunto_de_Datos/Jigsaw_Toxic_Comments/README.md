# Jigsaw Toxic Comments: comentarios de Wikipedia con etiquetas de toxicidad

Material de apoyo del curso. Dataset real de referencia (benchmark) para
clasificación multi-etiqueta de toxicidad. El origen y la descarga están en
`FUENTE.md`.

## Archivo

`jigsaw_train.csv.gz` (26 MB comprimido, 66 MB descomprimido). Para
descomprimir sin borrar el original: `gunzip -k jigsaw_train.csv.gz`.

## Perfil

- Filas: 159,571 comentarios en inglés de páginas de discusión de Wikipedia.
- Columnas: `id, comment_text, toxic, severe_toxic, obscene, threat, insult,
  identity_hate`. Las seis etiquetas son binarias (0/1) y pueden combinarse
  en un mismo comentario.
- Longitud del texto: media 394 caracteres, mediana 205, máximo 5,000.
  Textos mucho más largos que un chat de videojuego.

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

Está fuertemente desbalanceado y las etiquetas se traslapan: un comentario
puede ser a la vez `toxic`, `obscene` e `insult`.

## Nota para el proyecto

Este dataset no es de videojuegos. Sirve como benchmark comparativo, para
pre-entrenar o transferir modelos, o para discutir el cambio de dominio entre
comentarios largos de Wikipedia y mensajes cortos de chat in-game como los de
CONDA.

## Exploración rápida

```bash
gunzip -k jigsaw_train.csv.gz
python3 ../../Herramientas/explorar_dataset.py jigsaw_train.csv --texto comment_text --etiqueta toxic
```

Las figuras de la exploración inicial están en
`Proyecto/Documentacion/02_eda_inicial.md`.
