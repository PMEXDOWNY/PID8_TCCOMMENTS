# CONDA: toxicidad en chat de partidas de Dota 2

Material de apoyo del curso. Dataset real de chat dentro del juego (in-game)
de 1,911 partidas completas de Dota 2, anotado a nivel de enunciado
(intención) y a nivel de token (slots). El origen y la cita están en
`FUENTE.md`.

## Archivos

| Archivo | Filas | Tamaño | Nota |
|---|---|---|---|
| `datos/CONDA_train.csv` | 26,921 | 2.7 MB | anotado |
| `datos/CONDA_valid.csv` | 8,974 | 934 KB | anotado |
| `datos/CONDA_test.csv` | 8,974 | 473 KB | sin anotar; las etiquetas se evalúan en la competencia de CodaLab |
| `recursos/lexicon_refined_1209.csv` | 1,209 términos | 35 KB | lexicón usado para la anotación a nivel de palabra |

En total son 44,869 enunciados provenientes de unas 12 mil conversaciones.

## Columnas

`Id, matchId, conversationId, utterance, chatTime, playerSlot, playerId, intentClass, slotClasses, slotTokens`

- `utterance`: el mensaje de chat (inglés, con jerga de gaming).
- `intentClass`: etiqueta a nivel de enunciado, solo en train y valid.
  E es tóxico explícito, I tóxico implícito, A acción (por ejemplo pedir
  reportes) y O otro, no tóxico.
- `slotClasses` y `slotTokens`: anotación por token. T es palabra tóxica, D
  término de Dota, S pronombre u objetivo, C carácter separador, O otro. El
  marcador `[SEPA]` separa mensajes consecutivos del mismo jugador.
- `matchId` y `conversationId` permiten reconstruir el contexto de la
  conversación, que es la característica distintiva de este dataset.

## Distribución de `intentClass`

| Clase | train | valid | % (train) |
|---|---|---|---|
| O (otro) | 19,982 | 6,629 | 74.2% |
| E (tóxico explícito) | 3,528 | 1,183 | 13.1% |
| A (acción) | 1,719 | 580 | 6.4% |
| I (tóxico implícito) | 1,692 | 582 | 6.3% |

Las clases están desbalanceadas; tomen eso en cuenta al elegir métricas y al
entrenar.

## Longitud de los textos

Mensajes muy cortos, típicos del chat en vivo: media de unos 18 caracteres,
mediana 11, máximo 408. Muy distinto de datasets de comentarios largos como
Jigsaw.

## Ejemplos del train

- E: "stfu trash", "sry um fucking drunk"
- I: "gg [SEPA] ez", "ez" (sarcasmo o burla al terminar la partida)
- A: "report doom, ty", "commend all TA"
- O: "one by one ROFL", "LOL EVERYONE MID PLS"

## Exploración rápida

```bash
python3 ../../Herramientas/explorar_dataset.py datos/CONDA_train.csv --texto utterance --etiqueta intentClass
```

Las figuras de la exploración inicial están en
`Proyecto/Documentacion/02_eda_inicial.md`.
