# PID8 TCCOMMENTS: detección de toxicidad en chat de videojuegos

**Pregunta del proyecto:** ¿se puede detectar la toxicidad en el chat de un juego **sin bloquear las groserías casuales** ("wtf", "holy shit nice ult") y **detectando la toxicidad que no usa groserías** ("maybe stop playing, you are hurting the team")? Un filtro de palabras prohibidas hace lo contrario: bloquea todo lo que trae una grosería y no ve lo demás.

Datos: **CONDA** (chat de Dota 2, 35,887 mensajes etiquetados) y **Jigsaw** (comentarios de Wikipedia, 159,571). Modelado y documento de resultados: Carlos Saavedra.

## Dónde está cada cosa

```
ENTREGA.md                          este archivo
README.md                           instrucciones del curso (no se modifica)
ESTRUCTURA.md                       estructura estándar del curso
requirements.txt                    versiones de las librerías usadas

Articulos/
  Revision/
    Matriz_de_Revision.xlsx         matriz de literatura: 60 artículos, sin repetidos
    Fuentes/                        los 60 PDFs, numerados como la matriz (01_... a 60_...)
      INDICE.md                     número, autores, título y archivo de cada uno
  Investigacion/Fuentes/            (vacío)

Proyecto/
  Propuesta/pre_propuesta.md
  Conjunto_de_Datos/
    FUENTE.md                       de dónde salieron los datos (URL, institución, quién y cuándo)
    DECISIONES.md                   cómo se unieron CONDA y Jigsaw y cómo se definió is_toxic
    preprocesar.py, exploratorio.py
    CONDA/, Jigsaw_Toxic_Comments/  datos originales sin modificar
    Unificado/                      dataset unificado y reporte de calidad
    Exploratorio/                   análisis exploratorio con figuras
  Modelado/
    RESULTADOS.md                   resultados completos, interpretación y limitaciones
    01_ ... 09_*.py, 12_robustez.py, 13_umbral.py, 14_no_vistos.py   experimentos, figuras y PDF (ver abajo)
    10_entrenar_demo.py, 11_demo.py demo interactiva
    comun.py, bert_util.py          funciones compartidas
    normalizar.py, perturbar.py     normalizador de ortografía alterada y generador de pruebas
    Resultados/                     CSV, tablas, figuras y Reporte_de_resultados.pdf
    Demo/LEEME.md                   cómo usar la demo

Recursos/                           (vacío)
```

## Resultados principales

Detalle, tablas y figuras en `Proyecto/Modelado/RESULTADOS.md` (y su versión con gráficas en `Proyecto/Modelado/Resultados/Reporte_de_resultados.pdf`).

1. **Todos los modelos superan al filtro de palabras.** En CONDA, F1 de la clase tóxica: filtro 0.650, regresión logística 0.848, BERT 0.865.
2. **Pero los modelos siguen bloqueando entre un tercio y dos quintos de la grosería casual** (0.34 a 0.38; el filtro bloquea el 100%), y BERT **no** es el mejor en detectar toxicidad implícita (0.74 contra 0.78 de los lineales).
3. **El contexto de la conversación no ayuda** (BERT, 3 semillas: 0.862, 0.859 y 0.856 con 0, 1 y 3 mensajes previos).
4. **Cambiar de fuente cuesta mucho** (CONDA a Jigsaw: de 0.87 a 0.64 con BERT y de 0.85 a 0.28 con la logística), y un modelo con las dos fuentes juntas no mejora a los modelos por fuente.
5. **Las 4 clases de CONDA:** la clase implícita (I) es la más difícil para todos (F1 de 0.73 a 0.74); BERT no mejora a los lineales.
6. **Los insultos con letras cambiadas (`1diot`, `f u c k`) engañan a casi todos los modelos** (BERT pasa de detectar el 97% a entre 19% y 78%). Normalizar el texto antes de los modelos lo corrige en gran parte sin reentrenar, y combinarlo con entrenamiento aumentado llega a 89% a 98% en los disfraces conocidos y a 75% en uno que no se vio, a cambio de 1 a 5 puntos más de falsos positivos.
7. **Se puede bloquear menos grosería casual moviendo el umbral, casi sin perder F1 con la logística** (CONDA, umbral elegido en el dev): de 0.50 a 0.70 la casual bloqueada baja de 0.330 a 0.255 y la implícita detectada de 0.781 a 0.709, con el F1 tóxico en 0.850 a 0.859. Con BERT cuesta más (umbral 0.96, implícita 0.636).
8. **Solo con mensajes cuyo texto no está en train** (4,917 de 7,569 en CONDA), todos los modelos pierden de 2 a 3 puntos de F1 tóxico (logística 0.825, BERT 0.835), pero el orden entre modelos y las conclusiones no cambian.

Limitaciones principales: una sola partición, una sola semilla en BERT sobre Jigsaw, en la transferencia y en la curva de umbral, sin ajuste de hiperparámetros ni de umbral en las tablas principales, y las pruebas de ortografía alterada son sintéticas (sección 11 de `RESULTADOS.md`).

## Cómo reproducir (opcional)

**No hace falta correr nada para ver los resultados ni para usar la demo:** las cifras, tablas y figuras ya están guardadas en `Proyecto/Modelado/Resultados/` y los modelos de la demo **ya vienen entrenados** en el repo (con Git LFS). Los pasos de abajo son solo para quien quiera repetir los experimentos.

Desde la raíz del repo, con `pip install -r requirements.txt`:

| Paso | Comando | Tiempo aprox. |
|---|---|---|
| Datos unificados (ya incluidos) | `python Proyecto/Conjunto_de_Datos/preprocesar.py` | minutos |
| Filtro, logística y SVM | `python Proyecto/Modelado/01_baseline_y_lineales.py` | 21 min |
| BiGRU | `python Proyecto/Modelado/02_bigru.py` | 35 min con GPU |
| BERT por fuente | `python Proyecto/Modelado/03_bert.py` | 45 min con GPU |
| 4 clases de CONDA | `python Proyecto/Modelado/04_cuatro_clases.py` | 15 min con GPU |
| Contexto | `python Proyecto/Modelado/05_contexto.py` | 1 hora con GPU |
| Transferencia entre fuentes | `python Proyecto/Modelado/06_transferencia.py` | 2 horas con GPU |
| XGBoost | `python Proyecto/Modelado/07_xgboost.py` | 50 min (CPU) |
| Robustez a ortografía alterada | `python Proyecto/Modelado/12_robustez.py` | 25 min con GPU |
| Curva de umbral (CONDA) | `python Proyecto/Modelado/13_umbral.py` | menos de 1 min con GPU |
| Mensajes no vistos (CONDA) | `python Proyecto/Modelado/14_no_vistos.py` | 20 min con GPU |
| Figuras | `python Proyecto/Modelado/08_figuras.py` | segundos |
| PDF con gráficas | `python Proyecto/Modelado/09_reporte_pdf.py` | segundos |

Los experimentos con BERT usan GPU si hay; sin ella tardan mucho más. Los scripts 05 y 06 se pueden reanudar: guardan tras cada corrida y saltan las ya hechas.

## Demo interactiva

```
python Proyecto/Modelado/11_demo.py             # abre http://127.0.0.1:8765
```

Los modelos **ya están entrenados** (`Proyecto/Modelado/Demo/modelos/`, con Git LFS, unos 630 MB): no hay que entrenar nada. Si alguien quiere reentrenarlos, es opcional: `python Proyecto/Modelado/10_entrenar_demo.py` (~12 min con GPU).

Escribes un mensaje (en inglés) y ves el veredicto del filtro de palabras, la regresión logística, el SVM y los BERT. Una casilla permite normalizar la ortografía (deshacer `1diot`, `f u c k`) antes de los modelos y ver la diferencia. Más detalles y advertencias en `Proyecto/Modelado/Demo/LEEME.md`, que incluye **7 capturas reales de la demo funcionando** (`Proyecto/Modelado/Demo/capturas/`): un mensaje normal, grosería casual, tóxico explícito, tóxico implícito, un insulto disfrazado sin y con normalización, y el caso límite.

## Revisión de literatura

`Articulos/Revision/Matriz_de_Revision.xlsx` tiene 60 artículos, cada uno con autores, DOI o URL, problema, método, resultados con números y responsable de lectura. Los 60 PDFs están en `Articulos/Revision/Fuentes/`, con el número de la matriz al inicio del nombre.
