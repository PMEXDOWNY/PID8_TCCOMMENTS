"""Normalizacion de texto contra la ortografia alterada.

Uso:
    from normalizar import Normalizador
    norm = Normalizador(vocabulario, lexicon_t)        # vocabulario: dict palabra -> frecuencia
    norm("you are an 1diot and f u c k you")           # -> "you are an idiot and fuck you"

Regla de seguridad (la que evita estropear mensajes normales): una palabra SOLO se corrige si el modelo no la conoce
(no esta en el vocabulario ni en el lexicon T). "1v1", "gg" o "w0w" frecuentes en el chat no se tocan, y los numeros
("9000") tampoco. Correcciones, en este orden:
  1. letras parecidas de otros alfabetos y letras de ancho completo (la "a" cirilica, "ｆｕｃｋ") -> letras normales;
  2. cadenas de letras sueltas ("f u c k", "f.u.c.k", "f-u-c-k") -> se juntan si la palabra resultante es conocida;
  3. leetspeak (1->i/l, 0->o, 3->e, 4->a, 5->s, 7->t, @->a, $->s, !->i ...) -> si el resultado es conocido;
  4. letras repetidas ("fuuuck") -> se reduce a 1 o 2 letras si el resultado es conocido;
  5. variantes foneticas ("phuck", "fukk", "fucc") -> ph->f, kk/cc->ck, k<->c, z<->s si el resultado es conocido;
  6. sin vocales ("fck", "sht", "btch") -> la palabra del lexicon T con esas consonantes;
  7. comodin ("f*ck") y errores de una letra o dos letras cambiadas de lugar ("idiiot", "fcuk") respecto de una palabra
     del lexicon T: solo palabras de 4+ letras que empiecen con la misma letra (asi "mint" no se vuelve "bint").
NO cubre a proposito las sustituciones "creativas" (simbolos raros como "f#ck", puntuacion incrustada, etc.), para poder
medir hasta donde llega la normalizacion.
"""
import re
import unicodedata
from collections import Counter
from itertools import product

LEET = {"0": ["o"], "1": ["i", "l"], "3": ["e"], "4": ["a"], "5": ["s"], "7": ["t"], "8": ["b"],
        "@": ["a"], "$": ["s"], "!": ["i"], "|": ["i", "l"]}
HOMOGLIFOS = str.maketrans({"а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i", "ѕ": "s",
                            "ј": "j", "ԁ": "d", "һ": "h", "ɡ": "g", "ⅼ": "l", "ο": "o", "ν": "v", "α": "a", "ε": "e"})
FONETICAS = [("ph", "f"), ("kk", "ck"), ("cc", "ck"), ("kc", "ck"), ("zz", "ss"), ("z", "s"), ("s", "z"), ("k", "c"), ("c", "k")]
CARACTERES = r"a-z0-9@$!|*'"
BORDES = re.compile(rf"^([^{CARACTERES}]*)(.*?)([^{CARACTERES}]*)$", re.S)
CADENA_PUNTOS = re.compile(r"(?:[a-z0-9@$!|][.\-_*]){2,}[a-z0-9@$!|]")
VOCALES = set("aeiou")


def limpiar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.translate(HOMOGLIFOS))
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()


def un_error(a: str, b: str) -> bool:
    """a y b difieren en exactamente una sustitucion, insercion o eliminacion de una letra."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    corta, larga = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(corta) and corta[i] == larga[i]:
        i += 1
    return corta[i:] == larga[i + 1:]


def transposicion(a: str, b: str) -> bool:
    """a y b son iguales salvo por dos letras vecinas cambiadas de lugar ("fcuk" / "fuck")."""
    if len(a) != len(b) or a == b:
        return False
    d = [i for i in range(len(a)) if a[i] != b[i]]
    return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]


def esqueleto(w: str) -> str:
    return "".join(c for c in w if c not in VOCALES)


class Normalizador:
    def __init__(self, vocabulario, lexicon):
        """vocabulario: dict palabra -> frecuencia (o un conjunto). Entre varias correcciones posibles se elige la
        palabra mas frecuente."""
        cuenta = dict(vocabulario) if isinstance(vocabulario, dict) else {w: 1 for w in vocabulario}
        self.lexicon = {w.lower() for w in lexicon}
        self.conocidas = set(cuenta) | self.lexicon
        self.lex_largas = sorted((w for w in self.lexicon if len(w) >= 4 and w.isalpha()), key=lambda w: -cuenta.get(w, 0))
        self.por_esqueleto = {}
        for w in self.lex_largas:
            if len(esqueleto(w)) >= 3:
                self.por_esqueleto.setdefault(esqueleto(w), w)
        self.cuenta = cuenta
        self.canon = self._canonicas()

    def _canonicas(self):
        """El lexicon T ya trae variantes disfrazadas ("fck", "sht", "f4ck", "phuck", "n00b"). Para el filtro eso es util, pero
        un modelo las ve como palabras distintas de "fuck". Cada variante se lleva a la palabra mas frecuente del lexicon
        que la explica (leetspeak, sin vocales, dos letras cambiadas de lugar o variante fonetica)."""
        canon = {}
        for w in self.lexicon:
            if len(w) < 3:
                continue
            cands = [self._leet(w) if re.search(r"[0-9@$!|]", w) else None, self._sin_vocales_lex(w)]
            cands += [p for p in self.lex_largas if p[0] == w[0] and len(w) >= 4 and transposicion(w, p)]
            for a, b in FONETICAS[:5]:     # ph->f, kk/cc/kc->ck, zz->ss
                if a in w:
                    cands.append(w.replace(a, b, 1))
            disfraz_claro = bool(re.search(r"[0-9@$!|]", w)) or not (set(w) & VOCALES)   # numeros/simbolos o sin vocales
            cands = [c for c in cands if c and c != w and c in self.lexicon
                     and (disfraz_claro or self.cuenta.get(c, 0) > self.cuenta.get(w, 0))]
            if cands:
                canon[w] = max(cands, key=lambda c: self.cuenta.get(c, 0))
        return canon

    def _sin_vocales_lex(self, w):
        if len(w) >= 3 and w.isalpha() and not (set(w) & VOCALES):
            return self.por_esqueleto.get(w)
        return None

    # ---- reglas de una palabra (w ya en minusculas, sin signos en los bordes) ----
    def _leet(self, w, primera=False):
        pos = [i for i, c in enumerate(w) if c in LEET]
        if not pos or len(pos) > 6 or not re.search(r"[a-z]", w):   # sin letras es un numero: no se toca
            return None
        for comb in product(*[LEET[w[i]][:1] if primera else LEET[w[i]] for i in pos]):
            cand = list(w)
            for i, c in zip(pos, comb):
                cand[i] = c
            cand = "".join(cand)
            if primera or cand in self.conocidas:
                return cand
        return None

    def _repeticiones(self, w):
        if not re.search(r"(.)\1\1", w) or not re.search(r"[a-z]", w):
            return None
        for n in (1, 2):
            cand = re.sub(r"(.)\1{2,}", lambda m: m.group(1) * n, w)
            if cand in self.conocidas:
                return cand
        return None

    def _fonetica(self, w):
        for a, b in FONETICAS:
            i = w.find(a)
            while i != -1:
                cand = w[:i] + b + w[i + len(a):]
                if cand in self.conocidas:
                    return cand
                i = w.find(a, i + 1)
        return None

    def _sin_vocales(self, w):
        if len(w) >= 3 and w.isalpha() and not (set(w) & VOCALES):
            return self.por_esqueleto.get(w)
        return None

    def _comodin(self, w):
        if "*" not in w or len(w) < 4:
            return None
        patron = re.compile("^" + re.escape(w).replace(r"\*", ".") + "$")
        return next((p for p in self.lex_largas if patron.match(p)), None)

    def _cercana(self, w):
        """Una letra de diferencia o dos letras vecinas cambiadas, respecto de una palabra del lexicon T."""
        if len(w) < 4:
            return None
        for p in self.lex_largas:   # misma primera letra: asi "mint" no se vuelve "bint" ni "fude" "nude"
            if p[0] == w[0] and (transposicion(w, p) or (len(w) >= 5 and un_error(w, p))):
                return p
        return None

    def palabra(self, w):
        if w in self.canon:
            return self.canon[w]
        if len(w) < 3 or w in self.conocidas or w.rstrip("!|$@") in self.conocidas:
            return w
        for regla in (self._comodin, self._leet, self._repeticiones, self._fonetica, self._sin_vocales):
            r = regla(w)
            if r:
                return r
        base = self._leet(w, primera=True) or w   # leet y otro disfraz a la vez
        for r in (self._repeticiones(base), self._fonetica(base), self._cercana(w), self._cercana(base)):
            if r:
                return r
        return w

    def __call__(self, texto: str) -> str:
        tokens = []
        for tok in limpiar(texto).split():
            ini, nucleo, fin = BORDES.match(tok).groups()
            if CADENA_PUNTOS.fullmatch(nucleo):
                nucleo = re.sub(r"[.\-_*]", "", nucleo)
            tokens.append([ini, nucleo, fin])
        # cadenas de 3 o mas letras sueltas ("f u c k"): se juntan solo si la palabra resultante es conocida
        i, salida = 0, []
        while i < len(tokens):
            j = i
            while j < len(tokens) and len(tokens[j][1]) == 1 and tokens[j][1].isalnum() and not tokens[j][0] and not tokens[j][2]:
                j += 1
            if j - i >= 3:
                junta = "".join(t[1] for t in tokens[i:j])
                corregida = self.palabra(junta)
                if corregida in self.conocidas:
                    salida.append(corregida)
                    i = j
                    continue
            ini, nucleo, fin = tokens[i]
            salida.append(ini + self.palabra(nucleo) + fin)
            i += 1
        return " ".join(salida)


def construir_vocabulario(df, train):
    """Palabras 'conocidas' para el normalizador: CONDA train (2+ veces) y el train de Jigsaw (3+ veces, ingles general).
    Devuelve un dict palabra -> frecuencia. `df` es el dataset unificado y `train` el train de CONDA."""
    from comun import TOKEN
    c = Counter(w for t in train["text"] for w in TOKEN.findall(t.lower()))
    vocab = {w: n for w, n in c.items() if n >= 2}
    jig = Counter(w for t in df[(df["source"] == "jigsaw") & (df["split"] == "train")]["text"] for w in TOKEN.findall(t.lower()))
    for w, n in jig.items():
        if n >= 3:
            vocab[w] = vocab.get(w, 0) + n
    return vocab
