# Estructura estándar del proyecto

Organización unificada para todos los proyectos del curso.

```
Articulos/
├── Revision/            → Material para el ARTÍCULO DE REVIEW
│   ├── Fuentes/         → PDFs de los artículos encontrados
│   └── (la matriz de revisión .xlsx/.csv va en este nivel)
└── Investigacion/       → Material para el ARTÍCULO DE INVESTIGACIÓN
    └── Fuentes/
Recursos/                → Archivos, notas y referencias para entender mejor el tema
Proyecto/
├── Propuesta/           → Pre-propuesta y propuesta
└── Conjunto_de_Datos/   → Conjunto de datos del proyecto
```

## Reglas del curso

- **El conjunto de datos debe ser REAL, no simulado.** Incluyan en
  `Proyecto/Conjunto_de_Datos/` un archivo (por ejemplo `FUENTE.md`) indicando
  de dónde se obtuvo (URL, institución, fecha de descarga).
- La **matriz de revisión** se trabaja en `Articulos/Revision/` (Excel o CSV).
- **Fase actual:** revisión de literatura — búsqueda de artículos y llenado de
  la matriz de revisión. Con eso se genera después la propuesta con base en el
  conjunto de datos encontrado.
- No cambien los nombres de estas carpetas; agreguen dentro lo que necesiten.
