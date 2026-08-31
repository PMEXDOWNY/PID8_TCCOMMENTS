# Estructura estándar del proyecto

Organización unificada para todos los proyectos del curso.

```
Articulos/
├── Revision/            material para el artículo de review
│   ├── Fuentes/         PDFs de los artículos encontrados
│   └── (la matriz de revisión .xlsx o .csv va en este nivel)
└── Investigacion/       material para el artículo de investigación
    └── Fuentes/
Recursos/                archivos, notas y referencias para entender el tema
Proyecto/
├── Propuesta/           pre-propuesta y propuesta
├── Conjunto_de_Datos/   conjunto de datos del proyecto
├── Documentacion/       bitácora de los datos por versiones, con sus figuras
└── Herramientas/        scripts y fichas de apoyo
```

## Reglas del curso

- El conjunto de datos debe ser real, no simulado. Incluyan en
  `Proyecto/Conjunto_de_Datos/` un archivo (por ejemplo `FUENTE.md`) que
  indique de dónde se obtuvo: URL, institución y fecha de descarga.
- La matriz de revisión se trabaja en `Articulos/Revision/`, en Excel o CSV.
- Fase actual: revisión de literatura, es decir, búsqueda de artículos y
  llenado de la matriz de revisión. Con eso se genera después la propuesta
  con base en el conjunto de datos encontrado.
- No cambien los nombres de estas carpetas; agreguen dentro lo que necesiten.
