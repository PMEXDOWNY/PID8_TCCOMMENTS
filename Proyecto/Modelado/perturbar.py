"""Generador de ortografia alterada: sirve para PROBAR la robustez de los modelos y para AUMENTAR los datos de entrenamiento.

Familias que el entrenamiento aumentado y el normalizador conocen (COBERTURA):
  leet        a->4/@, e->3, i->1/!, o->0, s->5/$, t->7 ...        "idiot"  -> "1d10t"
  espacios    letras separadas por espacio, punto, guion...       "fuck"   -> "f u c k", "f.u.c.k"
  repeticion  una letra repetida varias veces                     "fuck"   -> "fuuuck"
  tecla       errores de tecleo: cambio de lugar, falta, vecina   "fuck"   -> "fcuk", "fuk", "fyck"
  sin_vocales se quitan las vocales                               "fuck"   -> "fck"
  fonetica    ph por f, c/k, z/s, kk                              "fuck"   -> "phuck", "fukk"
  homoglifos  letras parecidas de otros alfabetos                 "idiot"  -> "іdіоt" (cirilico)
  mixto       una familia al azar por palabra

Familia que NADIE ve (se usa solo para medir el limite):
  creativa    dos simbolos raros / signos de puntuacion en la palabra    "fucking" -> "f#cki/ng"

Todas las funciones reciben una palabra en minusculas y un generador de numeros al azar, y devuelven la palabra
alterada o None si esa familia no se puede aplicar a esa palabra.
"""
import re

import numpy as np

PALABRA = re.compile(r"[A-Za-z']+")
LEET = {"a": ["4", "@"], "e": ["3"], "i": ["1", "!"], "o": ["0"], "s": ["5", "$"], "t": ["7"], "l": ["1"], "b": ["8"]}
HOMOGLIFOS = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}
TECLADO = {"q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh", "u": "yij", "i": "uok", "o": "ipl", "p": "ol",
           "a": "qsz", "s": "awdx", "d": "sefc", "f": "drgv", "g": "fthb", "h": "gyjn", "j": "hukm", "k": "jil", "l": "ko",
           "z": "asx", "x": "zsdc", "c": "xdfv", "v": "cfgb", "b": "vghn", "n": "bhjm", "m": "njk"}
SIMBOLOS_RAROS = "#%&+=~^¤¶§¿¬"
VOCALES = set("aeiou")
COBERTURA = ["leet", "espacios", "repeticion", "tecla", "sin_vocales", "fonetica", "homoglifos"]
FAMILIAS = COBERTURA + ["mixto", "creativa"]


def leet(w, rng):
    pos = [i for i, c in enumerate(w) if c in LEET]
    if not pos:
        return None
    n = int(rng.integers(1, len(pos) + 1))
    elegidas = set(rng.choice(pos, size=n, replace=False).tolist())
    return "".join(str(rng.choice(LEET[c])) if i in elegidas else c for i, c in enumerate(w))


def espacios(w, rng):
    if len(w) < 3 or not w.isalpha():
        return None
    return str(rng.choice([" ", ".", "-", "_", "*"])).join(w)


def repeticion(w, rng):
    if len(w) < 3:
        return None
    i = int(rng.integers(1, len(w)))
    return w[:i] + w[i] * int(rng.integers(2, 5)) + w[i + 1:]


def tecla(w, rng):
    if len(w) < 4:
        return None
    i = int(rng.integers(1, len(w) - 1))
    op = int(rng.integers(0, 3))
    if op == 0:   # dos letras vecinas cambiadas de lugar
        return w[:i] + w[i + 1] + w[i] + w[i + 2:] if i + 2 <= len(w) else None
    if op == 1:   # falta una letra
        return w[:i] + w[i + 1:]
    vecinas = TECLADO.get(w[i])   # una letra vecina del teclado
    return w[:i] + str(rng.choice(list(vecinas))) + w[i + 1:] if vecinas else None


def sin_vocales(w, rng):
    if len(w) < 4:
        return None
    r = w[0] + "".join(c for c in w[1:] if c not in VOCALES)
    return r if r != w and len(r) >= 2 else None


def fonetica(w, rng):
    opciones = []
    if w.startswith("f"):
        opciones.append("ph" + w[1:])
    if "ck" in w:
        opciones += [w.replace("ck", "kk", 1), w.replace("ck", "cc", 1)]
    if "c" in w:
        opciones.append(w.replace("c", "k", 1))
    if "k" in w:
        opciones.append(w.replace("k", "c", 1))
    if w.endswith("s"):
        opciones.append(w[:-1] + "z")
    opciones = [o for o in opciones if o != w]
    return str(rng.choice(opciones)) if opciones else None


def homoglifos(w, rng):
    pos = [i for i, c in enumerate(w) if c in HOMOGLIFOS]
    if not pos:
        return None
    n = int(rng.integers(1, len(pos) + 1))
    elegidas = set(rng.choice(pos, size=n, replace=False).tolist())
    return "".join(HOMOGLIFOS[c] if i in elegidas else c for i, c in enumerate(w))


def creativa(w, rng):
    """Dos alteraciones por palabra (una si es corta): un simbolo raro en lugar de una letra, o puntuacion incrustada.
    Con dos alteraciones queda a mas de una letra de la palabra original, asi que el normalizador (que corrige a una letra
    de distancia) ya no la alcanza: es la familia que mide el limite."""
    if len(w) < 4:
        return None
    n_alt = 2 if len(w) >= 6 else 1
    pos = sorted(rng.choice(np.arange(1, len(w) - 1), size=min(n_alt, len(w) - 2), replace=False).tolist(), reverse=True)
    for i in pos:
        if rng.random() < 0.5:
            w = w[:i] + str(rng.choice(list(SIMBOLOS_RAROS))) + w[i + 1:]
        else:
            w = w[:i] + str(rng.choice([",", "/", "'", ";", ":"])) + w[i:]
    return w


def mixto(w, rng):
    for f in rng.permutation(COBERTURA):   # una familia al azar que si se pueda aplicar
        r = FUNCIONES[str(f)](w, rng)
        if r:
            return r
    return None


FUNCIONES = {"leet": leet, "espacios": espacios, "repeticion": repeticion, "tecla": tecla, "sin_vocales": sin_vocales,
             "fonetica": fonetica, "homoglifos": homoglifos, "creativa": creativa, "mixto": mixto}


def perturbar_texto(texto, familia, rng, palabras_objetivo=None):
    """Altera las palabras del texto. Si se da `palabras_objetivo` (conjunto, minusculas) solo se tocan esas; si no, una
    palabra al azar de 4+ letras. Devuelve (texto_nuevo, cuantas_palabras_cambiadas)."""
    f = FUNCIONES[familia]
    encontradas = list(PALABRA.finditer(texto))
    if palabras_objetivo is not None:
        candidatas = [m for m in encontradas if m.group().lower() in palabras_objetivo]
    else:
        candidatas = [m for m in encontradas if len(m.group()) >= 4]
        if candidatas:
            candidatas = [candidatas[int(rng.integers(0, len(candidatas)))]]
    nuevo, cambios, ultimo = [], 0, 0
    for m in candidatas:
        alterada = f(m.group().lower(), rng)
        if alterada and alterada != m.group().lower():
            nuevo.append(texto[ultimo:m.start()] + alterada)
            ultimo, cambios = m.end(), cambios + 1
    nuevo.append(texto[ultimo:])
    return "".join(nuevo), cambios


def aumentar(textos, etiquetas, lexicon, rng, probabilidad=0.3):
    """Copias alteradas de una parte de los mensajes de entrenamiento, con LA MISMA etiqueta que el original.
    Se altera tanto lo toxico como lo no toxico: si solo se alterara lo toxico, el modelo aprenderia que 'letras raras' = tóxico.
    Solo usa las familias de cobertura (nunca la creativa)."""
    nuevos, y_nuevos = [], []
    for t, y in zip(textos, etiquetas):
        if rng.random() > probabilidad:
            continue
        tiene = any(m.group().lower() in lexicon for m in PALABRA.finditer(t))
        familia = str(rng.choice(COBERTURA + ["mixto"]))
        alterado, n = perturbar_texto(t, familia, rng, lexicon if tiene else None)
        if n:
            nuevos.append(alterado)
            y_nuevos.append(y)
    return nuevos, np.array(y_nuevos, dtype=int)
