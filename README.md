# Proyecto: Motor de Detección de Idioma y Filtrado de Lenguaje Ofensivo

**Autor:** Victor
**Estado:** Definición inicial (v0.1)
**Fecha:** Agosto 2026

---

El objetivo es crear un motor de texto que detecte el texto escrito para determinar si un mensaje es apropiado y puede pasar o si debe ser bloqueado automáticamente. El reto es que el motor pueda entender contexto para que no banee palabras que se usarían de manera casual en un videojuego.

---

## 1. Resumen ejecutivo

Construir un motor de moderación de texto que (a) determine si un texto está escrito en inglés y (b) detecte y filtre lenguaje ofensivo en ese idioma, apoyado en una base de datos pública de palabras prohibidas **depurada, enriquecida y normalizada** por nosotros.

El valor del proyecto no está en el filtro en sí —eso ya existe en muchas librerías— sino en la **calidad de la base de datos**: las listas públicas disponibles son planas, sin severidad, sin excepciones y con muchos falsos positivos. El entregable diferenciador es una base de datos curada con metadatos, más un motor que resiste ofuscaciones (leetspeak, separadores, caracteres repetidos).

---

## 2. Objetivos

### Objetivo general
Desarrollar un sistema que reciba texto libre y devuelva un veredicto de moderación estructurado, accionable y explicable.

### Objetivos específicos
1. Consolidar 2–3 bases de datos públicas de palabras prohibidas en inglés en un único dataset normalizado.
2. Enriquecer ese dataset con severidad, categoría, variantes y excepciones.
3. Implementar detección de idioma para confirmar que el texto es inglés antes de aplicar el filtro.
4. Implementar un motor de coincidencia resistente a ofuscación.
5. Exponer el motor como API y como demo web pública.
6. Medir precisión y recall contra un set de prueba propio.

### Fuera de alcance (v1)
- Idiomas distintos al inglés (la arquitectura queda preparada, pero no se implementa).
- Detección de toxicidad semántica o acoso sin palabras prohibidas explícitas (requiere modelos de ML, no listas).
- Moderación de imágenes, audio o video.
- Panel de administración multiusuario con roles.

---

## 3. Caso de uso elegido

**Decisión:** el sistema se entrega como **servicio de moderación embebible** — un endpoint HTTP más un snippet JavaScript de una línea que cualquier formulario web (comentarios, chat, registro de usuarios, reseñas) puede consumir.

**Justificación:**
- Es agnóstico al producto final: no obliga a construir un chat o una red social completa para demostrar valor.
- La demo pública (una sola página donde el usuario escribe y ve el análisis en tiempo real) es visualmente contundente y se arma rápido.
- Reutiliza el stack ya conocido: HTML de un solo archivo, Netlify, Firestore.
- Permite vender/reutilizar el mismo motor en varios proyectos posteriores.

**Escenario demo:** caja de comentarios de un sitio en inglés. El usuario escribe, el motor responde en menos de 50 ms con: idioma detectado, nivel de riesgo, palabras encontradas y texto censurado sugerido.

---

## 4. Arquitectura

```
┌──────────────────────────────────────────────────────────┐
│  CLIENTE (widget JS / demo HTML / integración de terceros)│
└───────────────────────┬──────────────────────────────────┘
                        │ POST /moderate  { text, options }
                        ▼
┌──────────────────────────────────────────────────────────┐
│  API (Netlify Function)                                   │
│                                                           │
│  1. Sanitización de entrada (límite de longitud, unicode) │
│  2. Detección de idioma  ──►  ¿es inglés? ¿confianza?     │
│  3. Normalización / des-ofuscación                        │
│  4. Coincidencia contra diccionario (trie compilado)      │
│  5. Aplicación de excepciones (whitelist)                  │
│  6. Scoring y veredicto                                   │
└───────────────────────┬──────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────┐
│  DATOS                                                    │
│  • Firestore: diccionario curado (fuente de verdad)       │
│  • JSON compilado en build: copia inmutable para runtime  │
│  • Firestore: log de detecciones (para retroalimentación) │
└──────────────────────────────────────────────────────────┘
```

**Nota de diseño:** el diccionario vive en Firestore para poder editarlo, pero el runtime **no** consulta Firestore en cada petición. Se compila a un JSON estático en cada despliegue. Esto evita latencia y costo por lectura.

---

## 5. Componente 1 — La base de datos mejorada

### 5.1 Fuentes públicas candidatas

| Fuente | Qué aporta | Consideración |
|---|---|---|
| `dsojevic/profanity-list` (GitHub) | Lista en JSON con severidad, etiquetas y excepciones ya modeladas | La mejor base estructural de partida |
| `mmathys/profanity` (Hugging Face) | ~1600 términos con calificación de severidad promediada por evaluadores humanos y categorías temáticas | Excelente para poblar el campo de severidad |
| LDNOOBW / LDNOOBW_V2 | Volumen alto y muchas variantes creativas de evasión | Requiere depuración fuerte; alta tasa de ruido |
| `MauriceButler/badwords` | Lista corta, conservadora, derivada de un listado de Google | Útil como núcleo de alta confianza |

**Criterio de selección definitivo:** tomar `dsojevic/profanity-list` como esquema y núcleo, cruzarlo con `mmathys/profanity` para severidad, y usar LDNOOBW solo como fuente de variantes/ofuscaciones (nunca como fuente primaria de términos).

> Verificar la licencia de cada fuente antes de integrarla y documentarla en `LICENSES.md`. Algunas listas en Hugging Face exigen aceptar condiciones de uso.

### 5.2 Qué significa "mejorar" la base de datos

Este es el corazón del proyecto. Las mejoras concretas:

1. **Deduplicación semántica**: unificar variantes ortográficas de un mismo lema bajo un registro único con lista de variantes.
2. **Severidad (1–3)**: leve / fuerte / severo. Permite que cada cliente decida su propio umbral.
3. **Categorización**: sexual, escatológico, insulto genérico, discriminatorio/odio, violencia, blasfemia. Un cliente puede bloquear odio pero permitir groserías leves.
4. **Excepciones (anti-Scunthorpe)**: el problema clásico — la ciudad inglesa *Scunthorpe* contiene una secuencia ofensiva y quedaba bloqueada por filtros ingenuos. Cada término lleva una lista de palabras legítimas que lo contienen (`class`, `assessment`, `bass`, `pass`...).
5. **Flag de coincidencia parcial**: booleano que indica si el término puede detectarse dentro de otra palabra o solo como palabra completa.
6. **Variantes de ofuscación**: generadas por regla, no a mano.
7. **Metadatos de origen**: de qué lista(s) vino cada término y si fue revisado manualmente.

### 5.3 Esquema propuesto

```json
{
  "id": "term_0417",
  "lemma": "example",
  "language": "en",
  "severity": 2,
  "categories": ["insult", "generic"],
  "matchMode": "whole_word",
  "variants": ["exampl3", "ex4mple", "e-x-a-m-p-l-e"],
  "exceptions": ["counterexample", "exemplary"],
  "sources": ["dsojevic", "mmathys"],
  "reviewed": true,
  "notes": "Alta frecuencia de uso no ofensivo en contexto técnico",
  "updatedAt": "2026-08-14T00:00:00Z"
}
```

### 5.4 Proceso de construcción

1. Descargar las fuentes en crudo a `/data/raw/`.
2. Script de ingesta (Node.js) → normaliza cada fuente al esquema anterior.
3. Merge con estrategia de prioridad por fuente y registro de conflictos.
4. Revisión manual asistida: los términos con conflicto de severidad o sin excepciones definidas se marcan para revisión.
5. Exportar a `dictionary.compiled.json` + carga a Firestore.

---

## 6. Componente 2 — El motor de detección

### 6.1 Detección de idioma

Aunque v1 solo filtra inglés, la detección de idioma cumple dos funciones reales:

- **Evitar falsos positivos cruzados**: hay palabras inocuas en español, italiano o neerlandés que coinciden con términos ofensivos en inglés. Si el texto no es inglés, no se aplica el diccionario inglés.
- **Preparar la extensibilidad**: cuando se agregue otro idioma, el enrutador ya existe.

**Enfoque:** modelo de n-gramas (trigramas) sobre perfiles de idioma, vía librería ligera del lado servidor (`franc`, `tinyld` o `cld3` según peso y precisión medidos).

**Limitación conocida y cómo se maneja:** la identificación de idioma es poco confiable en textos muy cortos (menos de ~20 caracteres). Regla de negocio: si el texto es más corto que el umbral o la confianza es baja, se **asume inglés y se aplica el filtro** (falla del lado seguro), pero se marca `languageConfidence: "low"` en la respuesta.

### 6.2 Normalización y des-ofuscación

Pipeline aplicado antes de la coincidencia:

1. Minúsculas y normalización Unicode (NFKD).
2. Mapeo de homoglifos y leetspeak: `@→a`, `4→a`, `3→e`, `1/!→i`, `0→o`, `$→s`, `+→t`, y caracteres cirílicos/griegos visualmente idénticos.
3. Colapso de caracteres repetidos: `baaaaad → bad`.
4. Eliminación de separadores intercalados: `b.a.d`, `b a d`, `b-a-d`.
5. Se conserva el **mapa de posiciones** entre texto normalizado y original, para poder censurar el texto original con precisión.

> Cada paso de normalización sube el recall y baja la precisión. Los pasos 3 y 4 deben ser configurables (`aggressive: true/false`).

### 6.3 Coincidencia

- Diccionario compilado a un **trie / autómata Aho-Corasick** para búsqueda en una sola pasada, O(n) sobre el texto.
- Alternativa simple para v1: expresión regular compilada por severidad. Suficiente hasta ~5000 términos.
- Post-filtro de excepciones: toda coincidencia se contrasta contra la whitelist del término antes de confirmarse.

### 6.4 Salida (contrato de la API)

```json
{
  "language": { "detected": "en", "confidence": 0.94, "inScope": true },
  "verdict": "flagged",
  "score": 68,
  "matches": [
    {
      "termId": "term_0417",
      "matched": "ex4mple",
      "position": [12, 19],
      "severity": 2,
      "categories": ["insult"],
      "obfuscated": true
    }
  ],
  "censored": "This is an **** comment",
  "processingMs": 7
}
```

`verdict` ∈ `clean` | `flagged` | `blocked`, según los umbrales configurados por el cliente.

---

## 7. Stack técnico

| Capa | Tecnología | Motivo |
|---|---|---|
| Frontend / demo | HTML de un solo archivo + JS vanilla | Stack ya dominado, cero build, despliegue inmediato |
| Hosting | Netlify | Ya en uso, deploy por Git |
| API | Netlify Functions (Node.js) | Serverless, sin infraestructura que mantener |
| Datos (edición) | Firebase / Firestore | Ya en uso, panel de edición rápido |
| Datos (runtime) | JSON compilado en el bundle | Latencia mínima, sin costo por lectura |
| Cache cliente | IndexedDB | Modo offline del widget si se requiere |
| Scripts de ingesta | Node.js | Mismo lenguaje en todo el proyecto |
| Pruebas | Vitest o Node test runner | Set de casos dorados versionado |

---

## 8. Fases y entregables

### Fase 1 — Base de datos (la fase larga)
- Descarga y auditoría de fuentes públicas, revisión de licencias.
- Script de ingesta y normalización al esquema propio.
- Merge, deduplicación y resolución de conflictos.
- Revisión manual de los términos de severidad alta y de los que requieren excepciones.
- **Entregable:** `dictionary.compiled.json` documentado + `LICENSES.md`.

### Fase 2 — Motor
- Módulo de normalización con pruebas unitarias.
- Integración del detector de idioma.
- Motor de coincidencia + aplicación de excepciones.
- **Entregable:** módulo JS independiente, sin dependencias del frontend.

### Fase 3 — API y demo
- Netlify Function con el contrato definido en 6.4.
- Página demo de un solo archivo: campo de texto, resultado en vivo, resaltado de coincidencias, controles de severidad.
- Snippet embebible.
- **Entregable:** URL pública funcionando.

### Fase 4 — Evaluación y ajuste
- Construcción del set de prueba etiquetado.
- Medición, ajuste de umbrales, corrección de falsos positivos detectados.
- **Entregable:** reporte de métricas.

### Fase 5 (opcional) — Panel de administración
- CRUD del diccionario sobre Firestore, revisión de detecciones registradas.

---

## 9. Métricas de éxito

| Métrica | Meta v1 | Cómo se mide |
|---|---|---|
| Precisión (precision) | ≥ 0.95 | Set de prueba etiquetado manualmente |
| Cobertura (recall) | ≥ 0.90 | Mismo set, incluyendo casos ofuscados |
| Falsos positivos en texto limpio | ≤ 1% | 500 comentarios inocuos reales en inglés |
| Detección correcta de idioma | ≥ 0.95 en textos > 40 caracteres | Muestras multilingües etiquetadas |
| Latencia p95 | < 50 ms para textos de hasta 500 caracteres | Medición en la Function |
| Tamaño del bundle del diccionario | < 500 KB comprimido | Build |

**Set de prueba:** mínimo 300 casos — 100 limpios, 100 con términos directos, 100 ofuscados. Versionado junto al código. Es el activo que permite cambiar el motor sin romper nada.

---

## 10. Riesgos

| Riesgo | Impacto | Mitigación |
|---|---|---|
| Falsos positivos tipo Scunthorpe | Alto — destruye la confianza en el filtro | Whitelist de excepciones obligatoria por término + coincidencia por palabra completa por defecto |
| Listas públicas con ruido y términos no ofensivos | Medio | Revisión manual de la lista final; ninguna fuente entra sin depurar |
| Licencias incompatibles con uso comercial | Alto | Auditoría de licencias en Fase 1, antes de escribir código |
| Ofuscaciones nuevas que el filtro no cubre | Medio | Log de detecciones + revisión periódica; las listas de bloqueo siempre van un paso atrás |
| Detección de idioma errónea en textos cortos | Bajo | Regla de fallar del lado seguro (asumir inglés) |
| Sobre-normalización agresiva | Medio | Modo agresivo configurable, desactivado por defecto |

---

## 11. Consideraciones éticas y legales

- Un filtro de palabras es una herramienta contundente, no un sistema de moderación completo: bloquea vocabulario, no intención. Debe presentarse como tal ante cualquier cliente.
- El sistema **sugiere**, no censura de forma autónoma. La decisión final (bloquear, revisar, permitir) es del integrador.
- El diccionario incluye términos discriminatorios y de odio. El repositorio debe advertirlo y el acceso al contenido crudo debe estar limitado.
- Registro de detecciones: guardar únicamente el veredicto y los IDs de términos, no el texto completo del usuario, salvo consentimiento explícito del integrador.
- La percepción de qué es ofensivo varía por cultura, región y contexto. Por eso la severidad y las categorías son configurables en lugar de una decisión única impuesta.

---

## 12. Decisiones pendientes

- [ ] Nombre del proyecto y dominio.
- [ ] ¿El diccionario final será público (repo abierto) o privado?
- [ ] ¿La API será gratuita/demostrativa o con llave de acceso y límite de uso?
- [ ] Umbrales por defecto de `flagged` y `blocked`.
- [ ] ¿Se registra el texto original en el log o solo el veredicto?
- [ ] Selección final de la librería de detección de idioma tras medir peso y precisión.

---

## 13. Referencias

- dsojevic/profanity-list — https://github.com/dsojevic/profanity-list
- mmathys/profanity (Hugging Face) — https://huggingface.co/datasets/mmathys/profanity
- LDNOOBW_V2 (Hugging Face) — https://huggingface.co/datasets/PeterGraebner/LDNOOBW_V2
- MauriceButler/badwords — https://github.com/MauriceButler/badwords
- censor-text/profanity-list — https://github.com/censor-text/profanity-list
