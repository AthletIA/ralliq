#!/usr/bin/env python3
"""
Instagram Carousel Generator
Dr. Andre Montenegro — Proteína & Termogênese
"""

from PIL import Image, ImageDraw, ImageFont
import os
import re
import math

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

W, H = 1080, 1080
PAD = 72          # horizontal padding
TOP_PAD = 85
BOT_ZONE = 110    # height reserved for branding bar

# Colors
BG          = (8,   8,   8)
BG_COVER    = (6,   6,   6)
WHITE       = (255, 255, 255)
GOLD        = (212, 175,  55)
GOLD_DIM    = (160, 130,  35)
DIMGRAY     = (155, 155, 155)
DARKGOLD    = ( 45,  37,   8)

# Font paths
_F = "/usr/share/fonts/truetype/open-sans/"
F_XB  = _F + "OpenSans-ExtraBold.ttf"
F_B   = _F + "OpenSans-Bold.ttf"
F_SB  = _F + "OpenSans-Semibold.ttf"
F_R   = _F + "OpenSans-Regular.ttf"
F_L   = _F + "OpenSans-Light.ttf"

TOTAL = 10   # total slides


# ──────────────────────────────────────────────
# RICH TEXT ENGINE
# ──────────────────────────────────────────────
# Markup:
#   **text**   → white extrabold
#   [g]text[/g] → gold extrabold
#   [r]text[/r] → white regular (lighter weight)
#   plain text  → white extrabold (default for content slides)

def _parse(text):
    """Return list of (tag, content) segments."""
    pattern = r'(\*\*.*?\*\*|\[g\].*?\[/g\]|\[r\].*?\[/r\])'
    parts = re.split(pattern, text)
    out = []
    for p in parts:
        if not p:
            continue
        if p.startswith('**') and p.endswith('**'):
            out.append(('wb', p[2:-2]))
        elif p.startswith('[g]') and p.endswith('[/g]'):
            out.append(('gb', p[3:-4]))
        elif p.startswith('[r]') and p.endswith('[/r]'):
            out.append(('wr', p[3:-4]))
        else:
            out.append(('wb', p))
    return out


def _resolve(tag, size):
    font_map = {
        'wb': (ImageFont.truetype(F_XB, size), WHITE),
        'gb': (ImageFont.truetype(F_XB, size), GOLD),
        'wr': (ImageFont.truetype(F_R,  size), DIMGRAY),
    }
    return font_map.get(tag, (ImageFont.truetype(F_XB, size), WHITE))


def render_rich(draw, text, x, y, max_w, size, spacing=1.38):
    """
    Render markup text with word-wrap.
    Returns y position after last line.
    """
    segs = _parse(text)

    # flatten to (tag, word_or_space) tokens
    tokens = []
    for tag, content in segs:
        words = content.split(' ')
        for i, w in enumerate(words):
            if w:
                tokens.append((tag, w))
            if i < len(words) - 1:
                tokens.append((tag, ' '))

    # compute reference line height from XB font
    ref_font = ImageFont.truetype(F_XB, size)
    ref_h = ref_font.getbbox('Ágy')[3] - ref_font.getbbox('Ágy')[1]
    lh = int(ref_h * spacing)

    # build lines with word-wrap
    lines = []
    cur_line, cur_w = [], 0

    for tag, tok in tokens:
        font, _ = _resolve(tag, size)
        tw = draw.textlength(tok, font=font)

        if tok == ' ':
            if cur_line:
                cur_line.append((tag, tok, tw))
                cur_w += tw
        elif cur_w + tw > max_w and cur_line:
            # strip trailing space
            while cur_line and cur_line[-1][1] == ' ':
                cur_line.pop()
            lines.append(cur_line)
            cur_line = [(tag, tok, tw)]
            cur_w = tw
        else:
            cur_line.append((tag, tok, tw))
            cur_w += tw

    if cur_line:
        while cur_line and cur_line[-1][1] == ' ':
            cur_line.pop()
        lines.append(cur_line)

    # draw
    for line in lines:
        cx = x
        for tag, tok, tw in line:
            font, color = _resolve(tag, size)
            draw.text((cx, y), tok, font=font, fill=color)
            cx += draw.textlength(tok, font=font)
        y += lh

    return y


# ──────────────────────────────────────────────
# BACKGROUNDS
# ──────────────────────────────────────────────

def _diag_lines(draw, color, spacing=55, width=1):
    for i in range(-H, W + H, spacing):
        draw.line([(i, 0), (i + H, H)], fill=color, width=width)


def bg_content(img):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG)
    _diag_lines(draw, (22, 18, 5), spacing=58, width=1)
    return draw


def bg_cover(img):
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, H], fill=BG_COVER)
    # radial diagonal burst from bottom-left corner
    for angle in range(0, 90, 3):
        rad = math.radians(angle)
        ex = int(W * 1.6 * math.cos(rad))
        ey = H - int(H * 1.6 * math.sin(rad))
        alpha = max(10, 35 - angle // 3)
        c = (alpha * 2, alpha, 0)
        draw.line([(0, H), (ex, ey)], fill=c, width=1)
    # corner accent line
    draw.line([(0, H - 2), (W, H - 2)], fill=DARKGOLD, width=2)
    return draw


# ──────────────────────────────────────────────
# BRANDING BAR
# ──────────────────────────────────────────────

def branding(draw, arrow=True):
    y_bar = H - BOT_ZONE
    # top rule
    draw.line([(PAD, y_bar), (W - PAD, y_bar)], fill=GOLD_DIM, width=1)

    yb = y_bar + 18
    # brand name
    bf = ImageFont.truetype(F_B, 22)
    draw.text((PAD, yb), "DR.ANDREMONTENEGRO", font=bf, fill=GOLD)
    # subtitle
    sf = ImageFont.truetype(F_L, 16)
    draw.text((PAD, yb + 28), "Médico  •  Performance & Emagrecimento  •  CRM/SC 19662",
              font=sf, fill=(110, 110, 110))

    if arrow:
        bx, by = W - 180, yb + 2
        draw.rounded_rectangle([bx, by, bx + 130, by + 50],
                               radius=25, outline=GOLD_DIM, width=2)
        # draw arrow manually: shaft + arrowhead
        ax, ay = bx + 38, by + 25
        draw.line([(ax, ay), (ax + 50, ay)], fill=GOLD, width=3)
        draw.polygon([(ax + 42, ay - 9), (ax + 58, ay), (ax + 42, ay + 9)], fill=GOLD)


def slide_num(draw, n):
    f = ImageFont.truetype(F_L, 18)
    txt = f"{n}/{TOTAL}"
    tw = draw.textlength(txt, font=f)
    draw.text((W - PAD - tw, TOP_PAD - 5), txt, font=f, fill=(65, 65, 65))


# ──────────────────────────────────────────────
# SLIDE FACTORIES
# ──────────────────────────────────────────────

def slide_cover():
    img = Image.new('RGB', (W, H))
    draw = bg_cover(img)
    slide_num(draw, 1)

    y = 95
    # category chip
    cf = ImageFont.truetype(F_SB, 19)
    draw.text((PAD, y), "TERMOGÊNESE  /  PROTEÍNA", font=cf, fill=GOLD)
    y += 38
    draw.line([(PAD, y), (PAD + 220, y)], fill=GOLD, width=2)
    y += 42

    # giant headline
    xl = ImageFont.truetype(F_XB, 88)
    lrg = ImageFont.truetype(F_XB, 80)

    draw.text((PAD, y), "COMER MAIS",       font=xl,  fill=WHITE);  y += 100
    draw.text((PAD, y), "PROTEÍNA",          font=xl,  fill=GOLD);   y += 100
    draw.text((PAD, y), 'NÃO "ACELERA"',    font=lrg, fill=WHITE);  y += 92
    draw.text((PAD, y), "SEU METABOLISMO.", font=lrg, fill=WHITE);  y += 90

    # subtitle
    sf = ImageFont.truetype(F_R, 29)
    draw.text((PAD, y), "O efeito existe. Mas foi exagerado na internet.", font=sf, fill=DIMGRAY)

    branding(draw, arrow=False)
    return img


def _content(n, body, label=None, note=None, body_size=50):
    img = Image.new('RGB', (W, H))
    draw = bg_content(img)
    slide_num(draw, n)

    y = TOP_PAD
    max_w = W - PAD * 2

    if label:
        lf = ImageFont.truetype(F_SB, 17)
        draw.text((PAD, y), label.upper(), font=lf, fill=GOLD)
        y += 30
        draw.line([(PAD, y), (PAD + 140, y)], fill=GOLD_DIM, width=1)
        y += 28
    else:
        y += 20

    y = render_rich(draw, body, PAD, y, max_w, size=body_size)

    if note:
        y += 22
        render_rich(draw, note, PAD, y, max_w, size=26)

    branding(draw)
    return img


def slide_cta():
    img = Image.new('RGB', (W, H))
    draw = bg_cover(img)
    slide_num(draw, TOTAL)

    y = 160
    max_w = W - PAD * 2

    tf = ImageFont.truetype(F_R, 33)
    draw.text((PAD, y), "Para receber conteúdo como este", font=tf, fill=DIMGRAY)
    y += 48

    bf = ImageFont.truetype(F_XB, 55)
    draw.text((PAD, y), "diretamente no", font=bf, fill=WHITE)
    y += 65
    draw.text((PAD, y), "seu Instagram:",  font=bf, fill=WHITE)
    y += 90

    draw.line([(PAD, y), (PAD + 180, y)], fill=GOLD, width=2)
    y += 38

    # bullet items
    items = [
        ("Siga",            "@dr.andremontenegro"),
        ("Ative",           "as notificações de posts"),
        ("Acesse",          "o link na bio para mais conteúdo"),
    ]
    lf  = ImageFont.truetype(F_R,  30)
    lbf = ImageFont.truetype(F_B,  30)
    dot = ImageFont.truetype(F_XB, 30)
    for verb, rest in items:
        draw.text((PAD,      y), "•",    font=dot, fill=GOLD)
        draw.text((PAD + 28, y), verb,   font=lbf, fill=WHITE)
        vw = draw.textlength(verb + " ", font=lbf)
        draw.text((PAD + 28 + vw, y), rest, font=lf, fill=DIMGRAY)
        y += 52

    # signature
    y += 20
    sig_f = ImageFont.truetype(F_L, 26)
    sig_b = ImageFont.truetype(F_SB, 26)
    draw.text((PAD, y), "Dr. Andre Montenegro",    font=sig_b, fill=WHITE)
    draw.text((PAD, y + 32), "Médico | Performance & Emagrecimento", font=sig_f, fill=DIMGRAY)

    branding(draw, arrow=False)
    return img


# ──────────────────────────────────────────────
# SLIDE CONTENT DEFINITIONS
# ──────────────────────────────────────────────

SLIDES = [
    # 1 – cover (special)
    None,

    # 2 – energy balance + ETA
    dict(
        n=2, label="Equilíbrio Energético",
        body=(
            "[r]A perda de gordura se resume à diferença entre[/r] "
            "**energia ingerida e energia gasta.** "
            "[r]Um componente do lado do gasto é a energia necessária para digerir e metabolizar os alimentos —[/r] "
            "**o Efeito Térmico dos Alimentos (ETA),** "
            "[r]também chamado de Termogênese Induzida pela Dieta (TID).[/r]"
        ),
        body_size=46,
    ),

    # 3 – protein DIT question
    dict(
        n=3, label="Proteína & Gasto Energético",
        body=(
            "[r]Entre os macronutrientes, a proteína é, de longe,[/r] "
            "**a fonte de energia menos eficiente:** "
            "[r]cerca de[/r] **25% da sua energia disponível** "
            "[r]é gasta apenas para metabolizá-la. Isso levanta a questão: podemos usar uma ingestão mais alta de proteína para[/r] "
            "**aumentar o gasto energético total** "
            "[r]a ponto de fazer diferença real na perda de gordura?[/r]"
        ),
        body_size=44,
    ),

    # 4 – meta-analysis result
    dict(
        n=4, label="O que a Ciência Diz",
        body=(
            "[r]Uma meta-análise de[/r] **52 ensaios clínicos randomizados** "
            "[r]investigou exatamente isso. Os resultados mostram que, mantendo as calorias constantes,[/r] "
            "**dietas hiperproteicas aumentam sim o ETA e o gasto energético diário total** "
            "[r]em comparação às dietas com menos proteína.[/r]"
        ),
        body_size=46,
    ),

    # 5 – but practical?
    dict(
        n=5, label="Relevância Clínica",
        body=(
            "**Mas esse efeito é clinicamente relevante?** "
            "[r]O ETA representa apenas[/r] **5–15% do gasto energético total.** "
            "[r]Além disso, as diferenças observadas foram impulsionadas por estudos em que o grupo hiperproteico consumia uma proporção de proteína[/r] "
            "**pouco realista para o cotidiano.**"
        ),
        body_size=47,
    ),

    # 6 – satiation is the real driver
    dict(
        n=6, label="Mecanismo Real",
        body=(
            "[r]Isso não significa que proteína não ajuda no emagrecimento —[/r] "
            "**dietas hiperproteicas são reconhecidamente superiores** "
            "[r]para perda de gordura sustentada. Porém, isso se deve muito mais ao[/r] "
            "[g]efeito de saciedade[/g] "
            "[r]do que ao pequeno aumento no gasto energético.[/r]"
        ),
        body_size=46,
    ),

    # 7 – bigger picture: muscle
    dict(
        n=7, label="O Quadro Completo",
        body=(
            "[r]Em um contexto mais amplo, o papel da proteína na perda de gordura é apenas um motivo[/r] "
            "**menor** [r]para priorizá-la. O motivo central é[/r] "
            "**a construção e manutenção de massa muscular —** "
            "[r]essencial para saúde, longevidade e performance física.[/r]"
        ),
        body_size=47,
    ),

    # 8 – don't burn protein for fuel
    dict(
        n=8, label="Uso Correto da Proteína",
        body=(
            "[r]Embora o metabolismo da proteína consuma mais energia que outros macronutrientes,[/r] "
            "**não queremos metabolizar proteína como combustível.** "
            "[r]Queremos que ela vá para[/r] "
            "**construção e reparação muscular,** "
            "[r]sendo catabolizado apenas o excedente.[/r]"
        ),
        body_size=46,
    ),

    # 9 – conclusion
    dict(
        n=9, label="Conclusão",
        body=(
            "**Priorize proteína na sua dieta.** "
            "[r]Existem inúmeras razões excelentes para isso.[/r] "
            "[r]O pequeno aumento no gasto energético[/r] "
            "**não é a principal delas.**"
        ),
        body_size=54,
    ),

    # 10 – CTA (special)
    None,
]


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    slides = []

    # Slide 1 — cover
    s1 = slide_cover()
    path = os.path.join(OUTPUT_DIR, "slide_01_cover.png")
    s1.save(path, quality=95)
    slides.append(path)
    print(f"✓ Slide 1 salvo")

    # Slides 2-9 — content
    for i, cfg in enumerate(SLIDES[1:-1], start=2):
        s = _content(**cfg)
        fname = f"slide_{i:02d}.png"
        path = os.path.join(OUTPUT_DIR, fname)
        s.save(path, quality=95)
        slides.append(path)
        print(f"✓ Slide {i} salvo")

    # Slide 10 — CTA
    s10 = slide_cta()
    path = os.path.join(OUTPUT_DIR, "slide_10_cta.png")
    s10.save(path, quality=95)
    slides.append(path)
    print(f"✓ Slide 10 salvo")

    print(f"\n{len(slides)} slides gerados em: {OUTPUT_DIR}")
    return slides


if __name__ == "__main__":
    main()
