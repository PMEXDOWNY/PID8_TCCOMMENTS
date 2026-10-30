"""Demo local: escribe un mensaje y mira que dice cada modelo.

Uso (desde la raiz del repo):
    python Proyecto/Modelado/10_entrenar_demo.py     # una vez: entrena y guarda los modelos (~12 min con GPU)
    python Proyecto/Modelado/11_demo.py              # abre http://127.0.0.1:8765 en el navegador
    python Proyecto/Modelado/11_demo.py --puerto 9000 --sin-bert     # sin BERT (arranca en segundos)

Solo escucha en tu computadora (127.0.0.1): nadie en la red puede entrar. No usa librerias nuevas.
Los modelos son los de CONDA (chat de Dota 2 en ingles), los mismos que se evaluaron en RESULTADOS.md:
  - filtro de palabras: marca toxico si el mensaje trae una palabra del lexicon T;
  - regresion logistica y SVM lineal sobre TF-IDF (muestran las palabras que mas pesaron);
  - BERT binario (toxico / no toxico) y BERT de 4 clases (O, E, A, I con su probabilidad).
Las probabilidades NO estan calibradas: sirven para comparar mensajes, no para leerlas como porcentajes reales.
"""
import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import numpy as np

from comun import CLASES_CONDA, TOKEN, cargar_lexicon_t
from normalizar import Normalizador, construir_vocabulario

DEMO = Path(__file__).resolve().parent / "Demo" / "modelos"
MAX_CARACTERES = 600


class Modelos:
    def __init__(self, con_bert=True):
        self.cerrojo = threading.Lock()  # un mensaje a la vez: la GPU y los modelos no estan pensados para hilos
        self.lexicon = cargar_lexicon_t()
        self.norm = self._cargar_normalizador()
        self.lineales = joblib.load(DEMO / "lineales.joblib") if (DEMO / "lineales.joblib").exists() else None
        self.nombres_palabras = self.lineales["palabras"].get_feature_names_out() if self.lineales else None
        self.nombres_chars = self.lineales["chars"].get_feature_names_out() if self.lineales else None
        self.berts, self.tok, self.dev = {}, None, None
        if con_bert:
            self._cargar_berts()
        self.disponibles = (["filtro"] + (["logistica", "svm"] if self.lineales else []) + sorted(self.berts))

    def _cargar_normalizador(self):
        """Deshace disfraces como 1diot, f u c k o fuuuck antes de pasar el texto a los modelos (ver normalizar.py)."""
        ruta = DEMO / "vocabulario.joblib"
        if ruta.exists():
            vocab = joblib.load(ruta)
        else:
            from comun import cargar_datos, split_por_fuente
            print("   armando el vocabulario del normalizador (solo la primera vez)...", flush=True)
            df = cargar_datos()
            vocab = construir_vocabulario(df, split_por_fuente(df, "conda")[0])
            joblib.dump(vocab, ruta, compress=3)
        return Normalizador(vocab, self.lexicon)

    def _cargar_berts(self):
        archivos = {"bert_binario": "bert_binario.pt", "bert_4clases": "bert_4clases.pt", "bert_robusto": "bert_robusto.pt"}
        existentes = {k: DEMO / v for k, v in archivos.items() if (DEMO / v).exists()}
        if not existentes:
            return
        import torch
        import bert_util as bu
        from transformers import AutoModelForSequenceClassification
        self.torch, self.dev, self.tok = torch, bu.DEV, bu.tokenizador()
        for nombre, ruta in existentes.items():
            guardado = torch.load(ruta, map_location="cpu")
            m = AutoModelForSequenceClassification.from_pretrained(bu.MODELO, num_labels=guardado["n_clases"])
            m.load_state_dict(guardado["estado"])
            self.berts[nombre] = (m.to(self.dev).eval(), guardado["max_len"])
            print(f"   {nombre} cargado ({self.dev})", flush=True)

    # ---- cada modelo ----
    def _filtro(self, texto):
        palabras = [w for w in TOKEN.findall(texto.lower()) if w in self.lexicon]
        return {"toxico": bool(palabras), "palabras": sorted(set(palabras))}

    def _aportes(self, X_bloque, coef, nombres, tope=4):
        """Rasgos (palabras o fragmentos de letras) que mas empujan hacia toxico (+) o hacia no toxico (-)."""
        fila = X_bloque.tocoo()
        pesos = [(nombres[j].replace(" ", "_"), float(v * coef[j])) for j, v in zip(fila.col, fila.data) if coef[j] != 0]  # "_" = limite de palabra
        suben = sorted([p for p in pesos if p[1] > 0], key=lambda p: -p[1])[:tope]
        bajan = sorted([p for p in pesos if p[1] < 0], key=lambda p: p[1])[:tope]
        return {"suben": suben, "bajan": bajan}

    def _lineal(self, texto):
        from scipy.sparse import hstack
        L = self.lineales
        Xp, Xc = L["palabras"].transform([texto]), L["chars"].transform([texto])
        X = hstack([Xp, Xc]).tocsr()
        nv = Xp.shape[1]
        log, svm = L["logistica"], L["svm"]
        p = float(log.predict_proba(X)[0, 1])
        margen = float(svm.decision_function(X)[0])
        def aportes(m):
            return {**self._aportes(Xp, m.coef_[0][:nv], self.nombres_palabras),
                    "letras": self._aportes(Xc, m.coef_[0][nv:], self.nombres_chars, tope=5)}
        return ({"toxico": p > 0.5, "probabilidad": p, "aportes": aportes(log)},
                {"toxico": margen > 0, "margen": margen, "aportes": aportes(svm)})

    def _logits(self, nombre, texto):
        modelo, max_len = self.berts[nombre]
        enc = self.tok(texto, truncation=True, max_length=max_len, return_tensors="pt").to(self.dev)
        with self.torch.no_grad(), self.torch.autocast(self.dev.type, dtype=self.torch.float16, enabled=self.dev.type == "cuda"):
            return modelo(**enc).logits.float().cpu().numpy()[0]

    def predecir(self, texto, normalizar=True):
        original = texto
        if normalizar:
            texto = self.norm(texto)
        cambio = texto != " ".join(original.lower().split())
        out = {"texto": original, "texto_normalizado": texto if cambio else None, "filtro": self._filtro(texto)}
        with self.cerrojo:
            if self.lineales:
                out["logistica"], out["svm"] = self._lineal(texto)
            if "bert_binario" in self.berts:
                lg = self._logits("bert_binario", texto)
                p = float(np.exp(lg[1]) / np.exp(lg).sum())
                out["bert_binario"] = {"toxico": p > 0.5, "probabilidad": p}
            if "bert_robusto" in self.berts:
                lg = self._logits("bert_robusto", texto)
                p = float(np.exp(lg[1]) / np.exp(lg).sum())
                out["bert_robusto"] = {"toxico": p > 0.5, "probabilidad": p}
            if "bert_4clases" in self.berts:
                lg = self._logits("bert_4clases", texto)
                pr = np.exp(lg - lg.max()) / np.exp(lg - lg.max()).sum()
                probs = {c: float(pr[i]) for i, c in enumerate(CLASES_CONDA)}
                out["bert_4clases"] = {"clase": CLASES_CONDA[int(pr.argmax())], "probabilidades": probs,
                                       "prob_toxico": probs["E"] + probs["I"], "toxico": probs["E"] + probs["I"] > 0.5}
        return out


PAGINA = r"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demo: detector de toxicidad en chat</title>
<style>
:root{--fondo:#f6f7f9;--tarjeta:#fff;--texto:#1c2430;--suave:#5b6675;--borde:#dde2e8;--acento:#0b5cad;
--toxico:#b3261e;--toxico-bg:#fdecea;--limpio:#1b6e3c;--limpio-bg:#e7f5ec;--barra:#e6eaf0}
@media (prefers-color-scheme:dark){:root{--fondo:#12161c;--tarjeta:#1b212a;--texto:#e7ebf0;--suave:#9aa5b4;--borde:#2c3541;
--acento:#6ab0ff;--toxico:#ff8a80;--toxico-bg:#3a1f1d;--limpio:#7bd89a;--limpio-bg:#1c3326;--barra:#2c3541}}
*{box-sizing:border-box}body{margin:0;background:var(--fondo);color:var(--texto);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 60px}h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--suave);margin:0 0 20px}textarea{width:100%;min-height:92px;padding:12px;border:1px solid var(--borde);border-radius:10px;
background:var(--tarjeta);color:var(--texto);font:inherit;resize:vertical}textarea:focus{outline:2px solid var(--acento)}
.fila{display:flex;gap:10px;align-items:center;margin:10px 0;flex-wrap:wrap}
button{font:inherit;border-radius:8px;border:1px solid var(--borde);background:var(--tarjeta);color:var(--texto);padding:8px 14px;cursor:pointer}
button.principal{background:var(--acento);border-color:var(--acento);color:#fff;font-weight:600}
button:hover{filter:brightness(.96)}.ej{font-size:13px;padding:5px 10px}
.grupo{margin:14px 0 4px;color:var(--suave);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
#cuenta{margin-left:auto;color:var(--suave);font-size:13px}
.rejilla{display:grid;grid-template-columns:repeat(auto-fit,minmax(285px,1fr));gap:12px;margin-top:18px}
.card{background:var(--tarjeta);border:1px solid var(--borde);border-radius:12px;padding:14px}
.card h3{margin:0 0 2px;font-size:15px}.card .que{color:var(--suave);font-size:12.5px;margin:0 0 10px}
.chip{display:inline-block;padding:3px 10px;border-radius:99px;font-weight:600;font-size:13px}
.tox{background:var(--toxico-bg);color:var(--toxico)}.lim{background:var(--limpio-bg);color:var(--limpio)}
.barra{height:8px;background:var(--barra);border-radius:6px;margin:10px 0 4px;overflow:hidden}.barra i{display:block;height:100%;background:var(--acento)}
.num{color:var(--suave);font-size:13px}.pal{margin:8px 0 0;font-size:13px}.pal b{font-weight:600}
.tag{display:inline-block;background:var(--barra);border-radius:6px;padding:1px 7px;margin:2px 3px 2px 0;font-size:12.5px}
.cuatro .fila4{display:grid;grid-template-columns:22px 1fr 52px;gap:8px;align-items:center;font-size:13px;margin:3px 0}
.cuatro .barra{margin:0}.nota{margin-top:22px;color:var(--suave);font-size:13px;border-top:1px solid var(--borde);padding-top:14px}
.error{color:var(--toxico);margin-top:12px}.notanorm{margin-top:10px;color:var(--suave);font-size:13px}
</style></head><body><main>
<h1>Detector de toxicidad en chat de juegos</h1>
<p class="sub">Escribe un mensaje (en inglés, como en el chat de Dota 2) y mira qué dice cada modelo del proyecto.</p>
<textarea id="texto" maxlength="600" placeholder="Ej.: gg wp, wtf that was close lol"></textarea>
<div class="fila"><button class="principal" id="ir">Analizar</button><button id="borrar">Borrar</button>
<label title="Deshace disfraces como 1diot, f u c k o fuuuck antes de pasar el texto a los modelos" style="font-size:14px"><input type="checkbox" id="norm" checked> Normalizar ortografía</label>
<span id="cuenta">0 / 600</span></div>
<div class="grupo">Ejemplos para probar</div><div id="ejemplos"></div>
<div id="normalizado" class="notanorm"></div><div id="error" class="error"></div><div id="resultado" class="rejilla"></div>
<p class="nota"><b>Cómo leerlo.</b> Los modelos se entrenaron con chat de Dota 2 en inglés (dataset CONDA): en otro idioma o en otro juego no son
confiables. Las probabilidades no están calibradas; sirven para comparar mensajes, no como porcentajes reales de que alguien sea tóxico.
Un modelo puede equivocarse, y precisamente eso es lo que mide el proyecto: cuánta grosería casual bloquea y cuánta toxicidad implícita detecta.
Clases de CONDA: <b>O</b> otro (no tóxico), <b>E</b> tóxico explícito, <b>A</b> acción en el juego, <b>I</b> tóxico implícito.</p>
</main><script>
const EJ={"Mensajes normales":["gg wp","nice play, thanks for the help","push mid, i have tp"],
"Grosería casual (no debería bloquearse)":["wtf that was so close lol","holy shit nice ult","fuck yeah we won"],
"Tóxico explícito":["you are a worthless noob, uninstall the game","shut the fuck up idiot","report this trash"],
"Tóxico implícito (sin groserías)":["maybe stop playing, you are hurting the team","do you even have hands","we lose because of you again"],
"Insultos con ortografía alterada":["you are an 1diot","you are stup1d","f u c k you","fuuuck you","you are a phucking noob","you are a f,uck/ing idiot"]};
const $=id=>document.getElementById(id), texto=$("texto");
for(const [g,l] of Object.entries(EJ)){const t=document.createElement("div");t.className="grupo";t.textContent=g;$("ejemplos").append(t);
 for(const e of l){const b=document.createElement("button");b.className="ej";b.textContent=e;b.onclick=()=>{texto.value=e;actualizar();analizar()};$("ejemplos").append(b," ")}}
function actualizar(){$("cuenta").textContent=texto.value.length+" / 600"}texto.oninput=actualizar;
$("borrar").onclick=()=>{texto.value="";actualizar();$("resultado").replaceChildren();$("error").textContent=""};
$("ir").onclick=analizar;texto.onkeydown=e=>{if(e.key==="Enter"&&(e.ctrlKey||e.metaKey))analizar()};
const chip=t=>{const s=document.createElement("span");s.className="chip "+(t?"tox":"lim");s.textContent=t?"Tóxico":"No tóxico";return s};
function tarjeta(titulo,que){const c=document.createElement("div");c.className="card";const h=document.createElement("h3");h.textContent=titulo;
 const p=document.createElement("p");p.className="que";p.textContent=que;c.append(h,p);return c}
function barra(v){const b=document.createElement("div");b.className="barra";const i=document.createElement("i");i.style.width=Math.round(v*100)+"%";b.append(i);return b}
function num(t){const d=document.createElement("div");d.className="num";d.textContent=t;return d}
function tags(titulo,lista,signo){const d=document.createElement("div");d.className="pal";const b=document.createElement("b");b.textContent=titulo+" ";d.append(b);
 if(!lista.length){d.append("—");return d}for(const [w,v] of lista){const s=document.createElement("span");s.className="tag";s.textContent=w+" ("+(signo+Math.abs(v).toFixed(2))+")";d.append(s)}return d}
function simples(titulo,lista){const d=document.createElement("div");d.className="pal";const b=document.createElement("b");b.textContent=titulo+" ";d.append(b);
 if(!lista.length){d.append("ninguna");return d}for(const w of lista){const s=document.createElement("span");s.className="tag";s.textContent=w;d.append(s)}return d}
function lineal(titulo,que,r,esLog){const c=tarjeta(titulo,que);c.append(chip(r.toxico));
 if(esLog){c.append(barra(r.probabilidad),num("probabilidad de tóxico: "+(r.probabilidad*100).toFixed(1)+"%"))}
 else{c.append(num("margen: "+r.margen.toFixed(2)+" (mayor que 0 = tóxico)"))}
 c.append(tags("Palabras que empujan a tóxico:",r.aportes.suben,"+"),tags("Palabras que empujan a no tóxico:",r.aportes.bajan,"−"));
 c.append(tags("Fragmentos de letras a tóxico (_ = límite de palabra):",r.aportes.letras.suben,"+"));return c}
async function analizar(){const t=texto.value.trim();$("error").textContent="";if(!t){$("error").textContent="Escribe un mensaje primero.";return}
 $("ir").disabled=true;$("ir").textContent="Analizando…";
 try{const r=await fetch("/api/predecir",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({texto:t,normalizar:$("norm").checked})});
  const d=await r.json();if(!r.ok)throw new Error(d.error||"error");const R=$("resultado");R.replaceChildren();$("normalizado").textContent=d.texto_normalizado?("Texto después de normalizar: "+d.texto_normalizado):"";
  let c=tarjeta("Filtro de palabras","Tóxico si trae una palabra del lexicon T (lo que hace un filtro de palabras prohibidas).");c.append(chip(d.filtro.toxico));
  c.append(simples("Palabras detectadas:",d.filtro.palabras));R.append(c);
  if(d.logistica)R.append(lineal("Regresión logística","TF-IDF de palabras y letras; suma un peso por palabra.",d.logistica,true));
  if(d.svm)R.append(lineal("SVM lineal","Igual que la logística, pero decide por un margen.",d.svm,false));
  if(d.bert_binario){c=tarjeta("BERT (tóxico / no tóxico)","Lee la frase completa; afinado con CONDA.");c.append(chip(d.bert_binario.toxico),barra(d.bert_binario.probabilidad),
   num("probabilidad de tóxico: "+(d.bert_binario.probabilidad*100).toFixed(1)+"%"));R.append(c)}
  if(d.bert_robusto){c=tarjeta("BERT con aumento de datos","Entrenado también con mensajes de ortografía alterada.");c.append(chip(d.bert_robusto.toxico),barra(d.bert_robusto.probabilidad),
   num("probabilidad de tóxico: "+(d.bert_robusto.probabilidad*100).toFixed(1)+"%"));R.append(c)}
  if(d.bert_4clases){const q=d.bert_4clases;c=tarjeta("BERT de 4 clases","Elige la intención: O, E, A o I.");c.classList.add("cuatro");
   const d1=document.createElement("div");d1.append(chip(q.toxico),"  clase más probable: ");const b=document.createElement("b");b.textContent=q.clase;d1.append(b);c.append(d1);
   for(const k of ["O","E","A","I"]){const f=document.createElement("div");f.className="fila4";const a=document.createElement("b");a.textContent=k;
    f.append(a,barra(q.probabilidades[k]));const n=document.createElement("span");n.className="num";n.textContent=(q.probabilidades[k]*100).toFixed(1)+"%";f.append(n);c.append(f)}
   c.append(num("tóxico = E o I: "+(q.prob_toxico*100).toFixed(1)+"%"));R.append(c)}
 }catch(e){$("error").textContent="No se pudo analizar: "+e.message}
 $("ir").disabled=false;$("ir").textContent="Analizar"}
// enlace con el mensaje: http://127.0.0.1:8765/?texto=gg+wp&normalizar=0  (analiza al abrir; sirve para compartir un ejemplo)
const q=new URLSearchParams(location.search);
if(q.get("texto")){texto.value=q.get("texto");actualizar();if(q.get("normalizar")==="0")$("norm").checked=false;analizar()}
</script></body></html>"""


def crear_servidor(modelos, puerto):
    class Manejador(BaseHTTPRequestHandler):
        def _enviar(self, codigo, tipo, cuerpo):
            datos = cuerpo.encode("utf-8") if isinstance(cuerpo, str) else cuerpo
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(datos)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(datos)

        def do_GET(self):
            ruta = self.path.split("?")[0]   # el mensaje puede venir en la direccion (?texto=...)
            if ruta in ("/", "/index.html"):
                self._enviar(200, "text/html; charset=utf-8", PAGINA)
            elif ruta == "/api/estado":
                self._enviar(200, "application/json", json.dumps({"modelos": modelos.disponibles}))
            else:
                self._enviar(404, "text/plain; charset=utf-8", "No encontrado")

        def do_POST(self):
            if self.path != "/api/predecir":
                return self._enviar(404, "text/plain; charset=utf-8", "No encontrado")
            try:
                largo = int(self.headers.get("Content-Length", 0))
                datos = json.loads(self.rfile.read(min(largo, 10_000)))
                texto = str(datos["texto"]).strip()
                if not texto:
                    raise ValueError("el mensaje esta vacio")
                self._enviar(200, "application/json", json.dumps(
                    modelos.predecir(texto[:MAX_CARACTERES], bool(datos.get("normalizar", True))), ensure_ascii=False))
            except Exception as e:  # la demo nunca debe tumbar el servidor por un mensaje raro
                self._enviar(400, "application/json", json.dumps({"error": str(e)}))

        def log_message(self, *args):  # sin ruido en la terminal
            pass

    return ThreadingHTTPServer(("127.0.0.1", puerto), Manejador)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", type=int, default=8765)
    ap.add_argument("--sin-bert", action="store_true", help="no carga los BERT (arranca en segundos)")
    ap.add_argument("--no-abrir", action="store_true", help="no abre el navegador solo")
    a = ap.parse_args()
    if not (DEMO / "lineales.joblib").exists():
        print("Faltan los modelos. Corre antes: python Proyecto/Modelado/10_entrenar_demo.py")
    print("cargando modelos...", flush=True)
    modelos = Modelos(con_bert=not a.sin_bert)
    print("modelos listos:", ", ".join(modelos.disponibles), flush=True)
    servidor = crear_servidor(modelos, a.puerto)
    url = f"http://127.0.0.1:{a.puerto}"
    print(f"Demo en {url}   (Ctrl+C para cerrar)", flush=True)
    if not a.no_abrir:
        webbrowser.open(url)
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        print("\ncerrado")


if __name__ == "__main__":
    main()
