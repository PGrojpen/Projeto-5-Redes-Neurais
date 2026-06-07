# -*- coding: utf-8 -*-
"""Gera o relatório de progresso da Etapa 2 em PDF (reportlab)."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                HRFlowable, ListFlowable, ListItem, PageBreak)

OUT = Path(__file__).resolve().parent / "Etapa2_Relatorio.pdf"

# ---------- paleta ----------
INK   = colors.HexColor("#1a2332")
ACC   = colors.HexColor("#2b6cb0")   # azul
ACC2  = colors.HexColor("#2f855a")   # verde (sucesso)
WARN  = colors.HexColor("#c05621")   # laranja (falha/limite)
LIGHT = colors.HexColor("#eef2f7")
GRID  = colors.HexColor("#cbd5e0")

# ---------- estilos ----------
ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica", fontSize=10,
                      leading=14.5, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=6)
small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, textColor=colors.HexColor("#4a5568"))
h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=15,
                    textColor=ACC, spaceBefore=14, spaceAfter=2)
h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5,
                    textColor=INK, spaceBefore=10, spaceAfter=3)
title = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=24,
                       textColor=colors.white, alignment=TA_LEFT, leading=27)
subtitle = ParagraphStyle("subtitle", parent=ss["Normal"], fontName="Helvetica", fontSize=12,
                          textColor=colors.HexColor("#cfe0f0"), alignment=TA_LEFT, leading=16)
cellh = ParagraphStyle("cellh", parent=body, fontName="Helvetica-Bold", fontSize=9, textColor=colors.white, alignment=TA_CENTER, spaceAfter=0)
cell  = ParagraphStyle("cell", parent=body, fontSize=9, alignment=TA_CENTER, spaceAfter=0, leading=12)
cellL = ParagraphStyle("cellL", parent=cell, alignment=TA_LEFT)

story = []

def heading(txt, style=h1):
    story.append(Paragraph(txt, style))
    if style is h1:
        story.append(HRFlowable(width="100%", thickness=1.2, color=ACC, spaceBefore=2, spaceAfter=7))

def para(txt, style=body):
    story.append(Paragraph(txt, style))

def bullets(items, color=INK):
    flow = []
    for it in items:
        flow.append(ListItem(Paragraph(it, body), leftIndent=6, value="•"))
    story.append(ListFlowable(flow, bulletType="bullet", start="•", leftIndent=12,
                              bulletColor=color, bulletFontSize=9))

def table(data, col_widths, header_bg=ACC, lefts=None):
    """data: lista de linhas (strings). Primeira linha = cabeçalho."""
    lefts = lefts or []
    rows = []
    for r, row in enumerate(data):
        cells = []
        for cidx, v in enumerate(row):
            if r == 0:
                cells.append(Paragraph(str(v), cellh))
            else:
                stylep = cellL if cidx in lefts else cell
                cells.append(Paragraph(str(v), stylep))
        rows.append(cells)
    t = Table(rows, colWidths=col_widths, repeatRows=1)
    st = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]
    for r in range(1, len(data)):
        if r % 2 == 0:
            st.append(("BACKGROUND", (0, r), (-1, r), LIGHT))
    t.setStyle(TableStyle(st))
    story.append(t)
    story.append(Spacer(1, 8))

# ============================================================== CAPA
def header_footer(canvas, doc):
    canvas.saveState()
    # faixa de título só na primeira página é desenhada via flowables; aqui só rodapé
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#7a8aa0"))
    canvas.drawString(2*cm, 1.1*cm, "Etapa 2 — Detecção de Faces Sintéticas")
    canvas.drawRightString(A4[0]-2*cm, 1.1*cm, "Página %d" % doc.page)
    canvas.setStrokeColor(GRID)
    canvas.line(2*cm, 1.4*cm, A4[0]-2*cm, 1.4*cm)
    canvas.restoreState()

# Banner de título (tabela colorida ocupando a largura)
banner = Table([[Paragraph("Etapa 2 — Detecção de Faces Sintéticas", title)],
                [Paragraph("Relatório de Progresso: da Análise Espectral à Busca de Hiperparâmetros", subtitle)]],
               colWidths=[17*cm])
banner.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, -1), INK),
    ("TOPPADDING", (0, 0), (-1, 0), 18),
    ("BOTTOMPADDING", (0, -1), (-1, -1), 16),
    ("LEFTPADDING", (0, 0), (-1, -1), 16),
    ("RIGHTPADDING", (0, 0), (-1, -1), 16),
    ("TOPPADDING", (0, 1), (-1, 1), 2),
]))
story.append(banner)
story.append(Spacer(1, 4))
story.append(Paragraph("Classificação binária real vs. sintético (rostos) e generalização cross-generator &nbsp;|&nbsp; "
                       "ResNet-50 e EfficientNet-B2 &nbsp;|&nbsp; Data: 07/06/2026", small))
story.append(Spacer(1, 10))

# ============================================================== 1. SUMÁRIO
heading("1. Sumário Executivo")
para("Esta etapa treina redes profundas para distinguir rostos reais de rostos gerados por IA. "
     "A descoberta central é que o problema fácil e o problema difícil são diferentes: classificar imagens do "
     "<b>mesmo</b> gerador de treino (StyleGAN) é trivial — a ResNet-50 atinge <b>AUC 0,997</b> no teste in-distribution — "
     "mas generalizar para imagens de <b>outros</b> geradores (o conjunto ArtiFact) despenca para <b>AUC 0,63</b>. "
     "Esse abismo é o foco de toda a investigação.")
para("<b>O que funcionou:</b> augmentation aleatória de degradação (<b>jpeg+noise</b>) elevou a AUC cross-generator "
     "de <b>0,632</b> (raw) para <b>0,687</b> — um ganho real e bem medido (5 seeds, desvio 0,006).")
para("<b>O que não funcionou:</b> a busca de hiperparâmetros do otimizador (learning rate, dropout, weight decay) "
     "<b>não trouxe ganho</b>: nenhuma das 20 configurações superou o default, e o efeito dos HP (~0,010) foi menor "
     "que o ruído entre seeds (~0,023).")
para("<b>Por quê (mecanismo):</b> a análise espectral mostrou que o StyleGAN deixa uma <i>assinatura de frequência</i> "
     "que a CNN aprende trivialmente. O ganho de generalização vem de <b>destruir essa assinatura na imagem</b> "
     "(augmentation), não de afinar o otimizador. Esse achado define a direção dos próximos passos.")

# ============================================================== 2. DADOS
heading("2. Bases de Dados")
para("O projeto usa <b>duas</b> bases com papéis distintos — é essa separação que permite medir generalização honesta "
     "(treinar num gerador, testar em outros).")
story.append(Paragraph("2.1 &nbsp; 140k Real and Fake Faces — treino / validação / teste (in-distribution)", h2))
para("Base do Kaggle com <b>70.000 imagens reais</b> (rostos do CelebA) e <b>70.000 sintéticas</b> (geradas por "
     "<b>StyleGAN</b>), já balanceada. Substituiu a CelebA Align da Etapa 1. É a única fonte de treino — portanto o "
     "modelo só vê <i>um</i> gerador durante o aprendizado.")
story.append(Paragraph("2.2 &nbsp; ArtiFact (faces) — validação / teste cross-generator", h2))
para("Base do Kaggle com imagens de <b>múltiplos geradores</b> (StyleGAN, ProGAN, Stable Diffusion, DALL-E, entre outros). "
     "Filtramos apenas <b>rostos humanos</b>. Os rótulos vêm da fonte: reais = <i>ffhq, celebahq, metfaces</i>; "
     "falsas = qualquer outro gerador de faces. Cada imagem guarda a fonte num <i>manifest</i>, o que permite até excluir "
     "a própria família StyleGAN para um teste puramente cross-generator. É o nosso termômetro de generalização: "
     "<b>fixo e balanceado</b> em todos os experimentos, avaliado por <b>AUC</b> (robusta a desvio de calibração entre geradores).")

# ============================================================== 3. PIPELINE
heading("3. Pipeline de Investigação (01b a 01f)")
para("A série <b>01</b> é exploratória: roda em regime barato (ResNet-50, 10 épocas, 5% do 140k) para isolar, com "
     "rapidez e múltiplas seeds, o que de fato move a métrica cross-generator antes de treinar os modelos finais.")

story.append(Paragraph("3.1 &nbsp; 01b — Análise Espectral: a digital do StyleGAN", h2))
para("Via FFT, comparou o espectro médio de imagens reais, falsas (StyleGAN) e falsas comprimidas. Conclusão: o StyleGAN "
     "deixa <b>resíduos de alta frequência</b> que as CNNs aprendem <i>trivialmente</i> — exatamente o atalho que explica "
     "o AUC 0,99 instantâneo na fonte. A varredura de qualidade JPEG (q=100 a 50) revelou uma <b>curva em U</b>: existe um "
     "ponto ótimo de compressão onde o espectro do fake mais se aproxima do real (mínimo da distância espectral).")

story.append(Paragraph("3.2 &nbsp; 01c — Grad-CAM: o que o modelo olha", h2))
para("Mapas de ativação mostram que o modelo nem sempre foca em semântica facial (olhos, pele); o espectro 2D confirma "
     "o <b>grid 8x8 da compressão JPEG</b> como pontos brilhantes. Corrobora a tese: parte da decisão se apoia em "
     "artefatos, não em conteúdo — daí a má generalização.")

story.append(Paragraph("3.3 &nbsp; 01d — Sweep de pré-processamento fixo (cross-generator)", h2))
para("Varredura de 5 degradações aplicadas de forma <b>fixa</b> ao dataset. Todas superam o raw (0,632), confirmando que "
     "atenuar a digital ajuda. Melhor método isolado: <b>JPEG q=70</b>.")
table(
    [["Método", "Melhor valor", "AUC ArtiFact", "Desvio"],
     ["jpeg", "q = 70", "0,678", "0,021"],
     ["downscale", "fator 1,25", "0,669", "0,006"],
     ["median", "tam. 3", "0,663", "0,012"],
     ["noise", "sigma 0,03", "0,662", "0,017"],
     ["blur", "raio 1,0", "0,656", "0,002"]],
    col_widths=[4.2*cm, 4.2*cm, 4.3*cm, 4.3*cm], lefts=[0])
para("Baseline sem pré-processamento (raw): <b>0,632</b>.", small)

story.append(Paragraph("3.4 &nbsp; 01e — Grid de augmentation aleatória (fase vencedora)", h2))
para("Em vez de degradação fixa, aqui a degradação é <b>aleatória por imagem e por época</b> (com prob. p=0,6, sorteia "
     "método e intensidade de um pool). É mais forte porque não deixa resíduo consistente para o modelo memorizar. "
     "Testou as 15 combinações de {jpeg, blur, downscale, noise}; as 3 melhores foram desempatadas com <b>5 seeds</b>.")
table(
    [["Pool (aleatório)", "AUC ArtiFact", "Desvio", "min", "max"],
     ["jpeg + noise  (vencedor)", "0,687", "0,006", "0,681", "0,697"],
     ["jpeg + blur + noise", "0,684", "0,015", "0,661", "0,703"],
     ["blur + downscale", "0,675", "0,010", "0,669", "0,693"]],
    col_widths=[6.0*cm, 3.0*cm, 2.7*cm, 2.65*cm, 2.65*cm], header_bg=ACC2, lefts=[0])
para("Baselines: raw <b>0,632</b> | jpeg fixo q70 <b>0,678</b>. O <b>jpeg+noise</b> venceu por combinar a maior média "
     "com a <b>menor variância</b> — sinal robusto, não sorte.", small)

story.append(Paragraph("3.5 &nbsp; 01f — Busca de hiperparâmetros (resultado negativo)", h2))
para("Fixada a augmentation vencedora, fizemos random search de lr, dropout e weight decay (20 configs x 2 seeds = 40 runs). "
     "O resultado é um <b>platô</b>: nenhuma config superou o default, e a análise de variância mostra por quê.")
table(
    [["Métrica de diagnóstico", "Valor", "Leitura"],
     ["Efeito dos HP (desvio entre configs)", "0,010", "minúsculo"],
     ["Ruído de seed (desvio intra-config)", "0,023", "maior que o efeito"],
     ["Configs acima do default (0,687)", "0 / 20", "nenhuma"],
     ["corr(learning rate, AUC cross-gen)", "-0,30", "único sinal: lr alto piora"],
     ["corr(dropout, AUC cross-gen)", "-0,01", "inerte"],
     ["val_auc_140k de todas as configs", "0,99 a 1,00", "fonte saturada: métrica interna cega"]],
    col_widths=[7.6*cm, 2.6*cm, 6.8*cm], header_bg=WARN, lefts=[0, 2])
para("Mecanismo: como a fonte (StyleGAN) é resolvida por qualquer config (val ~0,99), o único hiperparâmetro com efeito é "
     "o learning rate — e só porque lr alto afasta mais a rede das features pré-treinadas do ImageNet, destruindo o "
     "conhecimento geral que ajuda a generalizar. Dropout e weight decay (acoplado ao Adam) não tocam nesse eixo.", small)

# ============================================================== 4. MODELOS
heading("4. Modelos Treinados (03 / 04)")
para("Os notebooks finais treinam os modelos completos. O número que importa é o contraste entre as duas colunas: "
     "desempenho excelente na fonte, fraco fora dela.")
table(
    [["Modelo", "AUC in-distribution (teste 140k)", "AUC cross-generator (ArtiFact)"],
     ["ResNet-50 (raw)", "0,9975", "0,6316"],
     ["EfficientNet-B2", "pendente (não treinado)", "pendente"]],
    col_widths=[4.6*cm, 6.2*cm, 6.2*cm], lefts=[0])
para("Observação importante: o modelo ResNet-50 salvo em disco ainda é a versão <b>raw</b> (sem a augmentation vencedora). "
     "A integração do jpeg+noise foi feita no código dos notebooks 03/04, mas <b>ainda não foi re-treinada</b>; a "
     "EfficientNet-B2 ainda não foi treinada. Logo, os modelos finais ainda não incorporam o principal ganho descoberto.", small)

# ============================================================== 5. DESCOBERTAS
heading("5. Descobertas: Sucessos e Falhas")
story.append(Paragraph("Sucessos", ParagraphStyle("sok", parent=h2, textColor=ACC2)))
bullets([
    "<b>Diagnóstico do problema real:</b> separar in-distribution (trivial, 0,997) de cross-generator (difícil, 0,63) — o gap é a métrica que importa.",
    "<b>Explicação mecanicista:</b> análise espectral (01b) + Grad-CAM (01c) mostram que o modelo se apoia na digital de frequência do StyleGAN.",
    "<b>Protocolo de avaliação rigoroso:</b> ArtiFact fixo e balanceado, métrica AUC, múltiplas seeds para separar sinal de ruído, salvamento incremental.",
    "<b>Augmentation que funciona:</b> jpeg+noise elevou o cross-generator de 0,632 para 0,687 (+0,055), bem medido (5 seeds, desvio 0,006).",
    "<b>Engenharia sólida:</b> aug_utils.py importável (permite num_workers>0 no Windows) e split disjunto dev/test do ArtiFact (avaliação sem vazamento).",
], color=ACC2)
story.append(Paragraph("Falhas e limitações", ParagraphStyle("swarn", parent=h2, textColor=WARN)))
bullets([
    "<b>Ajuste de hiperparâmetros não ajudou:</b> 0/20 acima do default; efeito (0,010) menor que o ruído de seed (0,023). Único achado: evitar learning rate alto (> 3e-4).",
    "<b>Teto cross-generator ainda baixo:</b> ~0,69 contra 0,997 in-distribution. Os modelos ainda generalizam mal para geradores não vistos.",
    "<b>Métrica interna cega para seleção:</b> a fonte satura (val ~0,99 para tudo), então não dá para selecionar HP por ela.",
    "<b>Ganho ainda não consolidado nos modelos finais:</b> ResNet-50 salvo é raw; EfficientNet-B2 não treinada. A augmentation vencedora precisa ser re-treinada.",
], color=WARN)

# ============================================================== 6. PRÓXIMOS PASSOS
heading("6. Próximos Passos")
para("A direção é clara: o ganho vem de <b>atacar a imagem</b> (impedir o aprendizado da digital do StyleGAN), não de afinar o otimizador.")
bullets([
    "<b>Atacar a imagem mais forte:</b> levar os destruidores de digital (blur, downscale — fortes no sweep/grid) para dentro do pool aleatório, possivelmente encadeados, com intensidade guiada pela curva em U da 01b.",
    "<b>Preservar features pré-treinadas:</b> congelar camadas iniciais / learning rate discriminativo, para a rede não esquecer o conhecimento geral do ImageNet (atacando o mecanismo que o lr baixo revelou).",
    "<b>Early-stopping no sinal cross-generator:</b> escolher a época pelo ArtiFact-dev (agora disjunto do teste), já que a métrica in-distribution é cega.",
    "<b>Consolidar:</b> re-treinar ResNet-50 e treinar EfficientNet-B2 com a receita vencedora e reportar o número final na metade de teste do ArtiFact.",
])

story.append(Spacer(1, 6))
story.append(HRFlowable(width="100%", thickness=0.6, color=GRID))
story.append(Paragraph("Relatório gerado a partir dos resultados salvos em <i>artifacts/</i> "
                       "(sweep, aug_grid, hp_search) e das métricas dos modelos em <i>artifacts/models/</i>.", small))

doc = SimpleDocTemplate(str(OUT), pagesize=A4,
                        leftMargin=2*cm, rightMargin=2*cm, topMargin=1.6*cm, bottomMargin=1.8*cm,
                        title="Etapa 2 — Relatório de Progresso", author="Projeto 5 — Redes Neurais")
doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
print("PDF gerado:", OUT)
