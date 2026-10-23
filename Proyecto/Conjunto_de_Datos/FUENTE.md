# Fuente de los datasets

Procedencia de los dos conjuntos de datos que se usan en el proyecto. Los datos originales se conservan sin modificar en `CONDA/` y `Jigsaw_Toxic_Comments/`; toda la limpieza se hace en `preprocesar.py` y queda en `Unificado/`.

> **Fecha de descarga.** El repo solo permite comprobar cuándo entraron los archivos a GitHub: el **31 de agosto de 2026**, en el commit `bae25ed` ("Material de apoyo técnico del curso", Santiago Yustre Bautista). La fecha en que se descargaron de su sitio original no consta en el repo. **Quien los descargó debe completar los campos marcados `POR COMPLETAR`.**

## CONDA

| Campo | Dato |
|---|---|
| Nombre | CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection |
| Institución | University of Sydney, NLP Group |
| Autores | Henry Weld, Guanghao Huang, Jean Lee, Tongshu Zhang, Kunze Wang, Xinghong Guo, Siqu Long, Josiah Poon, Soyeon Caren Han |
| Publicación | Findings of ACL-IJCNLP 2021 |
| Paper | https://arxiv.org/abs/2106.06213 · https://aclanthology.org/2021.findings-acl.213/ |
| Datos | https://github.com/usydnlp/CONDA |
| Contenido | Chat de partidas de Dota 2, con etiqueta de intención por mensaje (`intentClass`: O, E, A, I) y etiquetas de tokens (`slotClasses`) |
| Descargado de su sitio original | POR COMPLETAR (fecha y quién) |
| Entró al repo | 2026-08-31, commit `bae25ed` |

Archivos y huellas SHA-256 (para comprobar que no cambiaron):

| Archivo | Filas | SHA-256 |
|---|---|---|
| `CONDA/datos/CONDA_train.csv` | 26,921 | `fe7b18d19deae8c5a465163247dc929baea643f381e7f28596fec0b1559bddcd` |
| `CONDA/datos/CONDA_valid.csv` | 8,974 | `4ed5b3eee87131014cdd373fe0303bc57c246cf10c5b864f08bd1ef8306d4949` |
| `CONDA/datos/CONDA_test.csv` | 8,974 (sin `intentClass`) | `487eb6bc8336c5d5bea1ebad593f59b9f157209d987537572d99930ab01518af` |
| `CONDA/recursos/lexicon_refined_1209.csv` | 1,627 (columnas C, D, P, S, T) | `fb085bbfc48a4799967df44fae2bf4db73fc75a940665e684bb79f71a7a2347f` |

El lexicon agrupa vocabulario del juego por tipo. El proyecto usa la columna **T** (groserías y jerga ofensiva) para el filtro de palabras. Las definiciones de las cuatro clases salen de la sección 3.3 del paper (ver `DECISIONES.md`).

## Jigsaw Toxic Comment Classification

| Campo | Dato |
|---|---|
| Nombre | Toxic Comment Classification Challenge |
| Institución | Jigsaw y Conversation AI (Google/Alphabet), publicado en Kaggle |
| Datos | https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge |
| Contenido | Comentarios de páginas de discusión de Wikipedia, con 6 etiquetas binarias (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) |
| Descargado de su sitio original | POR COMPLETAR (fecha y quién) |
| Entró al repo | 2026-08-31, commit `bae25ed` |

| Archivo | Filas | SHA-256 |
|---|---|---|
| `Jigsaw_Toxic_Comments/jigsaw_train.csv.gz` | 159,571 | `75bff46f2739bd697ce61260f93d9f752644f52549e0bcfa4e54d654fbce43aa` |

Solo se usa el archivo `train` de la competencia, porque es el único con etiquetas. `Unificado/` le crea un `valid` propio (ver `DECISIONES.md`, sección 4).

## Reproducir el conjunto unificado

```
python Proyecto/Conjunto_de_Datos/preprocesar.py
```
