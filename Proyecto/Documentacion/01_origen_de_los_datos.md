# Bitácora de datos, entrada 01: origen de los datos (v0)

Esta carpeta es la bitácora del conjunto de datos del proyecto. La idea es
simple: cada vez que los datos cambien de forma (se descargan, se limpian, se
recortan, se re-etiquetan) se agrega una entrada que diga qué se hizo, por qué
y cómo reproducirlo. La versión v0 son los archivos tal como se descargaron,
sin tocar. Todo lo que venga después (v1 en adelante) es trabajo de ustedes y
se documenta aquí mismo, en este formato.

## Qué hay descargado (v0)

En el repositorio ya están dos datasets reales, cada uno con su `FUENTE.md`
que registra URL, institución, fecha y método de descarga. Eso es lo que los
hace citables: cualquiera puede ir a la fuente, bajar los mismos archivos y
verificar que las cifras coinciden.

### CONDA (chat de partidas de Dota 2)

Está en `Proyecto/Conjunto_de_Datos/CONDA/`. Es un dataset académico publicado
junto con un artículo de ACL-IJCNLP 2021 por el grupo de NLP de la University
of Sydney; el repositorio oficial es público en GitHub y la cita completa está
en su `FUENTE.md`. Contiene 44,869 enunciados de chat dentro del juego,
provenientes de 1,911 partidas de Dota 2, repartidos así:

| Archivo | Filas | Anotado |
|---|---|---|
| `datos/CONDA_train.csv` | 26,921 | sí |
| `datos/CONDA_valid.csv` | 8,974 | sí |
| `datos/CONDA_test.csv` | 8,974 | no (las etiquetas se reservan para la competencia de CodaLab) |

Cada enunciado trae una etiqueta de intención (`intentClass`: E tóxico
explícito, I tóxico implícito, A acción, O otro) y una anotación por token
(`slotClasses`). También incluye `matchId` y `conversationId`, que permiten
reconstruir la conversación completa, y en `recursos/` viene el lexicón de
1,209 términos que usaron los autores para la anotación a nivel de palabra.
Que el test venga sin etiquetas no es un error de descarga: es una decisión de
los autores, y tiene consecuencias para ustedes (ver el cierre de este
documento).

### Jigsaw Toxic Comments (comentarios de Wikipedia)

Está en `Proyecto/Conjunto_de_Datos/Jigsaw_Toxic_Comments/`. Proviene de la
competencia de Kaggle organizada en 2018 por Jigsaw y Conversation AI
(Google), con licencia CC0. Es el punto de referencia clásico en detección de
toxicidad y por eso vale la pena tenerlo a la mano aunque no sea de
videojuegos: casi cualquier artículo que lean lo menciona o compara contra él.

El archivo `jigsaw_train.csv.gz` contiene 159,571 comentarios de páginas de
discusión de Wikipedia, cada uno con seis etiquetas binarias que pueden
combinarse (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`,
`identity_hate`). Su `FUENTE.md` documenta el espejo exacto desde el que se
descargó y cómo verificar que corresponde al original de Kaggle.

### Tercer dataset, opcional: Dota 2 game chats (GosuAI)

No está descargado y no es obligatorio. En Kaggle
(`romovpa/gosuai-dota-2-game-chats`) hay unos 14 millones de mensajes de chat
de más de un millón de partidas de Dota 2, sin ninguna etiqueta. Puede servir
para estadísticas de jerga a gran escala o para construir una muestra
etiquetada por ustedes, pero requiere cuenta y token de Kaggle y pesa
alrededor de 600 MB descomprimido, así que no cabe en el repositorio (el
límite del curso es 90 MB por archivo). Si deciden usarlo, descárguenlo fuera
del repo o solo una muestra, y documenten la descarga en un `FUENTE.md` como
los otros dos. Las instrucciones paso a paso están en
`Proyecto/Herramientas/DATASETS.md`.

## Por qué estos datos cuentan como reales

La regla del curso pide datos reales, no simulados. Aquí se cumple porque los
tres provienen de personas escribiendo en plataformas reales: jugadores de
Dota 2 en los dos primeros casos, editores de Wikipedia en el tercero. Nadie
generó estos textos para el curso, y en los dos datasets descargados las
etiquetas fueron asignadas por anotadores humanos siguiendo un protocolo
publicado. La evidencia de todo esto queda en los `FUENTE.md` y en las citas
que contienen; cuando escriban la propuesta y los artículos, esas citas son
las que van en la bibliografía.

## Estado de versiones

| Versión | Qué es | Dónde está |
|---|---|---|
| v0 | Archivos tal como se descargaron de la fuente, sin modificar | `Proyecto/Conjunto_de_Datos/` |
| v1 | Limpieza y preparación del texto | pendiente, les toca a ustedes |

Una regla práctica: no sobrescriban los archivos v0. Cuando generen la v1,
guárdenla en archivos nuevos (por ejemplo una subcarpeta `v1/`) para poder
comparar siempre contra el original.

## Lo que sigue y les toca a ustedes

- Elegir el dataset principal del proyecto y justificar la elección por
  escrito. La justificación debe salir de criterios (dominio, etiquetas,
  tamaño, idioma, qué tanto se parece a su pregunta de investigación), no de
  cuál archivo es más cómodo.
- Definir la limpieza de texto (v1): qué hacen con mayúsculas, repeticiones
  tipo "gggggg", enlaces, mensajes vacíos, el marcador `[SEPA]` de CONDA, y
  qué hacen con el split de test de CONDA que viene sin etiquetas. Documenten
  cada decisión como una entrada nueva de esta bitácora, en este mismo
  formato.
- Llenar la matriz de revisión (`Articulos/Revision/`) con los cinco PDFs que
  ya tienen en `Articulos/Revision/Fuentes/`. La matriz alimenta directamente
  la propuesta.
- Aterrizar la propuesta a un solo objetivo: detección de toxicidad. Los
  datos traen varios idiomas mezclados y es tentador abrir un segundo frente
  de detección de idioma; resistan la tentación y trátenlo, si acaso, como un
  paso de limpieza, no como objetivo.
