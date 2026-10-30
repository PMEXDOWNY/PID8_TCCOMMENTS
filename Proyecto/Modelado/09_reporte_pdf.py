"""Convierte Proyecto/Modelado/RESULTADOS.md en un PDF con sus tablas y figuras.

Uso (desde la raiz del repo):
    pip install reportlab pillow
    python Proyecto/Modelado/09_reporte_pdf.py

Salida: Proyecto/Modelado/Resultados/Reporte_de_resultados.pdf
El PDF se genera siempre desde RESULTADOS.md, asi que el texto y las cifras no pueden diferir entre los dos.
Soporta lo que usa RESULTADOS.md: titulos (#, ##, ###), parrafos, listas con - y 1., tablas, imagenes ![](ruta), **negrita**, *cursiva* y `codigo`.
Necesita la fuente Arial de Windows para acentos y simbolos (si no esta, usa Helvetica y pierde acentos).
"""
import re
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

AQUI = Path(__file__).resolve().parent
ENTRADA = AQUI / "RESULTADOS.md"
SALIDA = AQUI / "Resultados" / "Reporte_de_resultados.pdf"
FUENTES = Path("C:/Windows/Fonts")


def registrar_fuentes():
    try:
        pdfmetrics.registerFont(TTFont("Arial", str(FUENTES / "arial.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(FUENTES / "arialbd.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-Italic", str(FUENTES / "ariali.ttf")))
        pdfmetrics.registerFont(TTFont("Arial-BoldItalic", str(FUENTES / "arialbi.ttf")))
        pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")
        pdfmetrics.registerFont(TTFont("Consolas", str(FUENTES / "consola.ttf")))
        return "Arial", "Consolas"
    except Exception:
        return "Helvetica", "Courier"


BASE, MONO = registrar_fuentes()
ESTILOS = {
    "h1": ParagraphStyle("h1", fontName=BASE + ("-Bold" if BASE == "Arial" else "-Bold"), fontSize=19, leading=23, spaceAfter=10, textColor=colors.HexColor("#1a1a1a")),
    "h2": ParagraphStyle("h2", fontName=BASE + "-Bold", fontSize=14, leading=18, spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#0b3d6b")),
    "h3": ParagraphStyle("h3", fontName=BASE + "-Bold", fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#0b3d6b")),
    "p": ParagraphStyle("p", fontName=BASE, fontSize=9.6, leading=13.4, spaceAfter=5, alignment=TA_LEFT),
    "li": ParagraphStyle("li", fontName=BASE, fontSize=9.6, leading=13.2, spaceAfter=2),
    "celda": ParagraphStyle("celda", fontName=BASE, fontSize=7.3, leading=9.2),
    "celda_enc": ParagraphStyle("celda_enc", fontName=BASE + "-Bold", fontSize=7.3, leading=9.2, textColor=colors.white),
    "nota": ParagraphStyle("nota", fontName=BASE + "-Italic", fontSize=8, leading=10.5, textColor=colors.HexColor("#555555"), spaceAfter=6),
}


def inline(t):
    """Markdown en linea -> marcado de reportlab."""
    t = t.replace("∈", "en")  # la fuente Arial no trae el simbolo de pertenencia
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", lambda m: f'<font face="{MONO}" size="8.6">{m.group(1)}</font>', t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", t)  # enlaces: solo el texto
    return t


def tabla(filas, ancho):
    enc, cuerpo = filas[0], filas[1:]
    n = len(enc)
    datos = [[Paragraph(inline(c), ESTILOS["celda_enc"]) for c in enc]]
    datos += [[Paragraph(inline(c), ESTILOS["celda"]) for c in (f + [""] * n)[:n]] for f in cuerpo]
    # la primera columna (o dos si son textos cortos) recibe mas ancho
    pesos = [max(len(re.sub(r"[*`]", "", f[i])) if i < len(f) else 0 for f in filas) for i in range(n)]
    pesos = [min(max(p, 16, int(1.8 * len(enc[i]))), 60) for i, p in enumerate(pesos)]  # minimo 16 y el ancho del encabezado para que una palabra corta como CONDA no se parta
    tot = sum(pesos)
    anchos = [ancho * p / tot for p in pesos]
    t = Table(datos, colWidths=anchos, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3d6b")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c5ccd6")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return t


def imagen(ruta, ancho_max):
    p = (AQUI / ruta).resolve()
    if not p.exists():
        return Paragraph(f"[falta la figura {ruta}]", ESTILOS["nota"])
    w, h = PILImage.open(p).size
    ancho = min(ancho_max, ancho_max)
    return Image(str(p), width=ancho, height=ancho * h / w)


def construir():
    ancho = A4[0] - 3.2 * cm
    lineas = ENTRADA.read_text(encoding="utf-8").splitlines()
    flujo, i = [], 0
    while i < len(lineas):
        l = lineas[i].rstrip()
        if not l.strip():
            i += 1
        elif l.startswith("#"):
            nivel = len(l) - len(l.lstrip("#"))
            flujo.append(Paragraph(inline(l.lstrip("# ").strip()), ESTILOS[f"h{min(nivel, 3)}"]))
            i += 1
        elif l.startswith("|"):
            filas = []
            while i < len(lineas) and lineas[i].startswith("|"):
                celdas = [c.strip() for c in lineas[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) or c == "---" for c in celdas):
                    filas.append(celdas)
                i += 1
            flujo += [tabla(filas, ancho), Spacer(1, 7)]
        elif l.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", l)
            flujo += [imagen(m.group(2), ancho), Paragraph(inline(m.group(1)), ESTILOS["nota"])] if m else []
            i += 1
        elif re.match(r"^(\s*)([-*]|\d+\.)\s+", l):
            items, ordenada = [], bool(re.match(r"^\s*\d+\.", l))
            while i < len(lineas) and re.match(r"^(\s*)([-*]|\d+\.)\s+", lineas[i]):
                sangria = len(lineas[i]) - len(lineas[i].lstrip())
                texto = re.sub(r"^\s*([-*]|\d+\.)\s+", "", lineas[i]).strip()
                i += 1
                while i < len(lineas) and lineas[i].strip() and not re.match(r"^\s*([-*]|\d+\.|\|)", lineas[i]) and not lineas[i].startswith(("#", "![")):
                    texto += " " + lineas[i].strip()
                    i += 1
                items.append((sangria, texto))
            principales = [t for s, t in items if s < 2]
            hijos = [t for s, t in items if s >= 2]
            fl = []
            for s, t in items:
                fl.append(ListItem(Paragraph(inline(t), ESTILOS["li"]), leftIndent=(14 if s < 2 else 30), bulletColor=colors.HexColor("#555555")))
            flujo.append(ListFlowable(fl, bulletType="1" if ordenada and not hijos else "bullet", start=1 if ordenada and not hijos else None,
                                      bulletFontName=BASE, bulletFontSize=8, leftIndent=14))
            flujo.append(Spacer(1, 4))
        else:
            texto = l.strip()
            i += 1
            while i < len(lineas) and lineas[i].strip() and not lineas[i].startswith(("#", "|", "![", "- ", "* ")) and not re.match(r"^\d+\.\s", lineas[i]):
                texto += " " + lineas[i].strip()
                i += 1
            flujo.append(Paragraph(inline(texto), ESTILOS["p"]))
    return flujo


def pie(canvas, doc):
    canvas.saveState()
    canvas.setFont(BASE, 7.5)
    canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(1.6 * cm, 1.0 * cm, "PID8 TCCOMMENTS - Resultados del modelado")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Pagina {doc.page}")
    canvas.restoreState()


def main():
    SALIDA.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(SALIDA), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.6 * cm, bottomMargin=1.7 * cm,
                            title="Resultados del modelado - PID8 TCCOMMENTS", author="Carlos Saavedra")
    doc.build(construir(), onFirstPage=pie, onLaterPages=pie)
    print("PDF:", SALIDA)


if __name__ == "__main__":
    main()
