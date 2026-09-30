# Decisiones de unión y preprocesamiento (hito 30 de septiembre)

Todo se reproduce con `python Proyecto/Conjunto_de_Datos/preprocesar.py`.
Salida: `Unificado/dataset_unificado.csv.gz` (195,458 filas) y `Unificado/reporte_calidad.md`.

## 1. Etiqueta común: `is_toxic`

CONDA (`intentClass`) y Jigsaw (6 banderas) no comparten esquema. Se define una etiqueta binaria común.

| Fuente | Regla | `is_toxic` = 1 |
|---|---|---|
| CONDA | `E` (Explicit toxicity) o `I` (Implicit toxicity) | 19.5% |
| CONDA | `A` (Action) y `O` (Other) → 0 | |
| Jigsaw | Cualquiera de las 6 banderas = 1 | 10.2% |

**Justificación.** Las definiciones salen de Weld et al. (2021), sección 3.3 de CONDA:
- `E`: la intención es insultar o humillar a otros.
- `I`: toxicidad oculta que se infiere por el contexto (por ejemplo, sarcasmo).
- `A`: acción del juego (reportar, pausar, salir) que no es `E` ni `I`.
- `O`: lo demás, incluidas groserías o emociones **no dirigidas a otros** (por ejemplo, "kill the fucking helicopter").

Que `O` incluya groserías sin objetivo es justo el caso del proyecto: no bloquear palabras que en un juego se usan de forma casual.

**Decisión sobre Jigsaw.** Se usa "cualquier bandera" y no solo `toxic`, porque 931 comentarios tienen una bandera (obscene, insult, etc.) sin `toxic`. `severe_toxic` siempre implica `toxic`.

**Cómo cambiarla.** El archivo conserva `conda_intent` y las 6 columnas `jigsaw_*`, así que la regla se puede reasignar sin repetir la limpieza.

**Límite.** Jigsaw no distingue toxicidad implícita ni contexto de juego, y CONDA sí. `is_toxic` es comparable en el sentido grueso, pero no equivalente.

## 2. Limpieza por fuente

**CONDA**
- **Se excluye `CONDA_test.csv`**: no trae `intentClass`, `slotClasses` ni `slotTokens`. Quedan 35,895 filas etiquetadas (train 26,921 + valid 8,974).
- **Se eliminan 8 filas sin texto** (7 en train, 1 en valid). Sin texto no hay nada que clasificar.
- **Se conservan las 8 sin `playerId` y las 834+268 sin slots.** `playerId` no se usa en el texto, y los slots no entran al conjunto unificado.
- **Se quita el marcador `[SEPA]`** (9,297 filas), que une mensajes consecutivos de un mismo jugador: `easiest [SEPA] game` → `easiest game`.

**Jigsaw**
- Sin nulos ni IDs repetidos. Los saltos de línea (59% de los comentarios) y los espacios múltiples pasan a un solo espacio.

**Ambas**
- Normalización Unicode NFC.
- Se quitan caracteres de control.
- **Se conservan mayúsculas, URLs, IPs y emoticonos.** Las mayúsculas y los emoticonos pueden ser señal de toxicidad. Si se quieren quitar, es decisión del modelado.

## 3. Duplicados: se marcan, no se borran

- `is_dup_text`: 13,864 filas con texto repetido (sin distinguir mayúsculas).
- `is_conflict_text`: 5,286 filas cuyo texto aparece con etiquetas `is_toxic` distintas.

No se borran porque son mensajes reales del chat ("gg", "lol", "ez" se repiten miles de veces) y su etiqueta depende del contexto. El caso típico es `ez`: 395 de sus 399 apariciones son `I`, y casi todos los `gg` son `O`. Borrarlos cambiaría la distribución real. Para modelar se puede filtrar por estas columnas.

## 4. Splits

- **CONDA:** se usa el split oficial (train/valid).
- **Jigsaw:** solo trae train, así que se crea un `valid` aleatorio del 25%, estratificado por `is_toxic`, con semilla 42. Es la misma proporción que CONDA.
- **No hay test.** El de CONDA no tiene etiquetas. Falta decidir si se aparta uno propio.

### Advertencia: fuga de datos en CONDA

El split oficial no se hizo por partida ni por conversación:
- 1,772 de las 1,778 partidas de valid también están en train.
- 3,540 conversaciones tienen mensajes en ambos.
- 799 textos exactos aparecen en train y valid.

Las métricas sobre valid saldrán optimistas. `group_id` (`matchId` en CONDA) sirve para re-hacer el split por grupos si se necesita una evaluación honesta.

## 5. Esquema del archivo unificado

| Columna | Descripción |
|---|---|
| `uid` | `conda_<Id>` o `jigsaw_<id>` |
| `source`, `split` | `conda`/`jigsaw`; `train`/`valid` |
| `group_id` | `matchId` (CONDA) o el propio id (Jigsaw), para splits por grupo |
| `text` | texto limpio |
| `is_toxic` | etiqueta común (0/1) |
| `conda_intent` | `O`/`E`/`A`/`I`, vacío en Jigsaw |
| `jigsaw_*` | 6 banderas originales, vacías en CONDA |
| `is_dup_text`, `is_conflict_text` | marcas de calidad |
| `n_chars`, `n_words` | longitud |

## Pendiente para el 7 de octubre

Gráficas de longitud (mediana de 10 caracteres en CONDA contra 204 en Jigsaw) y de desbalance de clases. Los datos para ambas ya están en el archivo unificado.
