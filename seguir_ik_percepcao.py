#!/usr/bin/env python3
"""seguir_ik_percepcao.py — VARIANTE com PERCEPÇÃO (P1): segue a PESSOA, não só o rosto.

Cópia do seguir_ik_web.py (que fica intacto como reserva) + a camada percepcao.py: o
ALVO do servo passa a ser o ROSTO; se a cabeça sai do quadro (você senta/levanta/cobre),
vira o CORPO (cabeça estimada acima dos ombros) → o braço mira PRA CIMA e te reencontra,
em vez de perder. Só cai na fuga/varredura quando a pessoa some de vez. Resto do app
(servo/IK/pescoço/altura/gestos/comportamentos/web) inalterado. Ver PLANO_CRIATURA (P1).

----- (doc original do seguir_ik_web abaixo) -----
seguir_ik_web.py — VARIANTE WEB do seguir_ik (cópia lado-a-lado).

IDÊNTICO ao seguir_ik.py (que fica INTACTO como reserva), porém também:
  • sobe um SERVIDOR WEB (servidor_web.py) numa thread daemon → painel no navegador
    e no celular (vídeo MJPEG + estado/comandos por websocket);
  • a cada frame PUBLICA o frame anotado + o estado público (engine_estado.ESTADO);
  • no topo do laço DRENA a fila de comandos da web e aplica via aplicar_comando(),
    na PRÓPRIA thread do engine (sem corrida). A janela cv2/teclado continua
    funcionando em paralelo (reserva). O braço só é comandado por este laço.

Mesma funcionalidade do 10_seguir_ik.py, mas separada em módulos (refactor
lado-a-lado; o 10 segue intacto como referência até esta versão ser validada):

    mira_ik.py         modelo Pinocchio + IK (geometria, resolver_ik, sinal_altura)
    controle_braco.py  loop MIT (gravidade) + estado `est` + motores
    ui_hud.py          cores, painel, toast, tela sem braço
    diario.py          log JSONL + Tee do terminal
    seguir_ik.py       (este) orquestração: laço principal + teclas + servo/altura

Arquitetura: SEMPRE MIT (nunca troca de modo). PAN pela base, TILT pelo punho,
ALTURA acompanha sua altura (sobe/desce devagar). Ver DOCUMENTACAO seção 13.

TECLAS (modo normal): TAB abre o painel de AJUSTES · ESPACO trava a home · k calibra ·
t segue · f flutua · z marca sentado · u fuga · g head-tilt · c recentra · n salva ·
x/y sinais · i esconde · ESC sai.
TECLAS (painel AJUSTES, modal): w/s navega · -/= (ou [/]) ajusta · ENTER alterna · n
salva · TAB/ESC fecha. Todos os parâmetros de feel ficam aqui (ganho, zona, limite,
previsão, pescoço, altura…) — ajustáveis ao vivo e salvos no 'n'.
"""

import json
import os
import random
import sys
import threading
import time
import webbrowser
from collections import deque

import numpy as np
import cv2

ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)
from reBotArm_control_py.actuator import RobotArm  # noqa: E402

from mira_ik import _LO, _HI, geometria, resolver_ik, sinal_altura_de  # noqa: E402
from controle_braco import (  # noqa: E402
    est, controlador, motor_pronto, status_motores, KP, KD,
)
from ui_hud import (  # noqa: E402
    COR_TXT, COR_TIT, COR_OK, COR_AVISO, COR_ERRO, COR_VAL, COR_DIM, TOAST_DUR,
    painel, desenha_toast, tela_sem_braco,
)
from diario import Diario, Tee  # noqa: E402
from autonomia_viva import Autonomia  # noqa: E402  (fuga + varredura; original intacto)
from gestos import aplicar_gesto, GESTO_TIPOS, HT_DESCE  # noqa: E402
from vida import respirar, Curiosidade  # noqa: E402
from engine_estado import ESTADO  # noqa: E402
import servidor_web  # noqa: E402

import camera  # noqa: E402
from detector import DetectorFaces  # noqa: E402
from percepcao import Percepcao  # noqa: E402  (rosto YuNet + corpo Pose -> ALVO)
from rastreador import RastreadorAlvo  # noqa: E402

CAMERA_PULSO = "C920"

# Servo visual.
KP_SERVO = 0.08            # ganho proporcional (manso; ajuste fino com [ / ])
DEADZONE_PX = 25           # raio central onde NÃO se mexe (mata o tremor parado); o/p ajusta
MAX_STEP_DEG = 1.2         # passo máx de mira/frame
LIMITE_DEG = 20.0          # envelope do PAN (abre com '=' se quiser)
PREVISAO_MS = 60.0
FOV_H, FOV_V = 70.0, 43.0  # C920: com IK o olhar é a rotação ÓPTICA real → FOV ≈ correto

IK_FLIP_DEG = 15.0          # salto de junta acima disso = virada → desfaz a mira
SLEW_MAX = np.radians(8.0)  # taxa máx de mudança do alvo por frame (suaviza trancos)
MARGEM_LIMITE_DEG = 4.0     # nunca comandar a junta encostada no batente

# Acompanhamento de ALTURA (só vertical): se o tilt se MANTÉM, sobe/desce devagar.
ALTURA_GANHO = 0.15         # m/s por rad de tilt sustentado (LENTO, sutil)
ALTURA_ZONA_DEG = 4.0       # só mexe a altura se |tilt| passar disso (evita creep)
ALTURA_MAX_DEFAULT = 0.12   # alcance vertical inicial (m, ± deste valor); ajustável c/ v/b
LIMITE_TILT_DEG = 30.0      # tilt modesto (a ALTURA cobre o vertical) → evita pose extrema
FALHAS_MAX = 8              # IK falhando + que isso seguidas = PRESO → recua p/ destravar

# "PESCOÇO": o pan PEQUENO é feito pelo PUNHO (joint5, ~18 px/deg — validado no
# lab_pescoco); o GRANDE pela BASE (joint1, via IK). Cascata: divide o pan total em
# neck (punho, capado em NECK_MAX) + base (o resto). Faixa ajustável ao vivo (9/0).
NECK_MAX_DEG = 10.0         # quanto o punho paneia antes da base entrar
PESCOCO_J5 = 4             # índice do joint5 (punho)
NECK_RELAX = 0.6           # /s: quão rápido o punho "desenrola" pro 0 e a base assume
                           #   (a cabeca endireita devagar; a base compensa o giro)

# COMPORTAMENTOS ("vida"): curiosidade (gesto sozinho quando parado), respirar
# (micro-movimento ocioso) e a varredura (em autonomia_viva.py). Defaults ajustáveis.
PARADO_S = 5.0             # s parado+centralizado p/ disparar curiosidade
COOLDOWN_S = 9.0          # s entre reações automáticas
VEL_PARADO = 80.0         # px/s: abaixo disso o alvo é "parado"
RESPIRAR_AMP = 1.0        # intensidade do micro-movimento ocioso (0 = estátua)

# Auto-calibração (tecla 'k'): cutuca a MIRA ±DELTA e mede o deslocamento do rosto.
DELTA_CAL_DEG = 8.0
CAL_SETTLE_S = 0.8
CAL_MEAS_S = 0.4

JANELA_W, JANELA_H = 1600, 900
DUR_REPOUSO = 3.5
DUR_ACORDAR = 3.0
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_ik")
# Config (home + repouso + calibração + ajustes + config por gesto), salva com 'n'.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_seguir_ik.json")


# Config PADRÃO de cada gesto (amp graus, vel/hold s, curioso=entra na curiosidade).
# Cada TIPO guarda a SUA config — editar um não mexe nos outros.
GESTO_DEF = {
    "single":    {"amp": 25.0, "vel": 0.30, "hold": 0.9,  "volta": 0.5, "segue": True,  "curioso": True},
    "swing":     {"amp": 22.0, "vel": 0.25, "hold": 0.4,  "volta": 0.5, "segue": True,  "curioso": True},
    "sim":       {"amp": 12.0, "vel": 0.22, "hold": 0.15, "volta": 0.4, "segue": False, "curioso": True},
    "nao":       {"amp": 14.0, "vel": 0.22, "hold": 0.15, "volta": 0.4, "segue": False, "curioso": False},
    "feliz":     {"amp": 10.0, "vel": 0.18, "hold": 0.10, "volta": 0.4, "segue": False, "curioso": True},
    "dancar":    {"amp": 12.0, "vel": 0.22, "hold": 0.10, "volta": 0.4, "segue": False, "curioso": False},
    "espreitar": {"amp": 25.0, "vel": 0.50, "hold": 0.70, "volta": 0.7, "segue": True,  "curioso": True},
}


def par_default():
    """Parâmetros ajustáveis (todos vão pro painel de AJUSTES e pro config). Em
    unidades de exibição já úteis — graus, ms, px — exceto alt_max (metros)."""
    return {
        "ganho": KP_SERVO, "zona": DEADZONE_PX, "limite": LIMITE_DEG,
        "limite_tilt": LIMITE_TILT_DEG, "previsao": PREVISAO_MS, "max_step": MAX_STEP_DEG,
        "neck_max": NECK_MAX_DEG, "neck_relax": NECK_RELAX, "sinal_neck": -1,
        "altura_on": True, "alt_max": ALTURA_MAX_DEFAULT,
        "altura_ganho": ALTURA_GANHO, "altura_zona": ALTURA_ZONA_DEG,
        "gesto_tipo": "single",
        "gestos": {t: dict(d) for t, d in GESTO_DEF.items()},   # config POR gesto
        # comportamentos ("vida")
        "procurar_on": True, "curioso_on": True, "respirar_on": True,
        "respirar_amp": RESPIRAR_AMP, "parado_s": PARADO_S, "cooldown_s": COOLDOWN_S,
        "vel_parado": VEL_PARADO,
        # percepção/gestos (toggles dinâmicos)
        "seguir_corpo": True, "gestos_on": True, "corpo_conf": 0.7, "reid_on": False,
    }


# Painel de AJUSTES (navegável). Cada item aponta uma chave de `par`:
# (grupo, chave, rótulo, passo, mínimo, máximo, formato).
AJUSTES_SPEC = [
    ("TRACKING", "ganho",        "ganho",            0.02, 0.0,   1.0,  "f2"),
    ("TRACKING", "zona",         "zona morta",          3,   0,   120,  "px"),
    ("TRACKING", "limite",       "limite pan",        5.0, 5.0, 120.0,  "deg"),
    ("TRACKING", "limite_tilt",  "limite tilt",       5.0, 5.0,  60.0,  "deg"),
    ("TRACKING", "previsao",     "previsao",         10.0, 0.0, 250.0,  "ms"),
    ("TRACKING", "max_step",     "passo max",         0.2, 0.2,   5.0,  "degf1"),
    ("PESCOCO",  "neck_max",     "faixa punho",       2.0, 0.0,  40.0,  "deg"),
    ("PESCOCO",  "neck_relax",   "desenrolar",        0.1, 0.0,   3.0,  "f1ps"),
    ("PESCOCO",  "sinal_neck",   "sinal punho",         0,  -1,     1,  "sinal"),
    ("ALTURA",   "altura_on",    "acompanha altura",    0,   0,     1,  "bool"),
    ("ALTURA",   "alt_max",      "alcance",          0.02, 0.0,  0.30,  "cm"),
    ("ALTURA",   "altura_ganho", "ganho subida",     0.05, 0.0,   1.0,  "f2"),
    ("ALTURA",   "altura_zona",  "tilt min p/ subir", 1.0, 0.0,  20.0,  "deg"),
    ("GESTOS",   "gesto_tipo",   "tipo (1-7 toca)",     0,   0,     0,  "tipo"),
    ("GESTOS",   "g_amp",        "amplitude",         5.0, 5.0,  90.0,  "deg"),
    ("GESTOS",   "g_vel",        "velocidade",       0.05, 0.10,  1.50, "f2s"),
    ("GESTOS",   "g_hold",       "hold",              0.1, 0.0,   4.0,  "f1s"),
    ("GESTOS",   "g_volta",      "velocidade volta",  0.05, 0.05,  3.0, "f2s"),
    ("GESTOS",   "g_segue",      "segue ao gesticular", 0,   0,     1,  "bool"),
    ("GESTOS",   "g_curioso",    "usa na curiosidade",  0,   0,     1,  "bool"),
    ("COMPORTAMENTOS", "procurar_on",  "procurar (fuga+varredura)", 0, 0, 1, "bool"),
    ("COMPORTAMENTOS", "curioso_on",   "curiosidade",         0,   0,    1,  "bool"),
    ("COMPORTAMENTOS", "parado_s",     "parado p/ reagir",  0.5, 1.0, 20.0,  "f1s"),
    ("COMPORTAMENTOS", "cooldown_s",   "intervalo",         1.0, 2.0, 40.0,  "f1s"),
    ("COMPORTAMENTOS", "vel_parado",   "limiar 'parado'",    10,  20,  300,  "px"),
    ("COMPORTAMENTOS", "respirar_on",  "respirar",            0,   0,    1,  "bool"),
    ("COMPORTAMENTOS", "respirar_amp", "intensidade respiro", 0.2, 0.0, 4.0, "f1"),
    ("COMPORTAMENTOS", "seguir_corpo", "seguir pelo corpo",     0,   0,    1,  "bool"),
    ("COMPORTAMENTOS", "gestos_on",    "gestos ligados",        0,   0,    1,  "bool"),
    ("COMPORTAMENTOS", "corpo_conf",   "confianca corpo",     0.05, 0.30, 0.95, "f2"),
    ("COMPORTAMENTOS", "reid_on",      "re-ID (nao trocar)",    0,   0,    1,  "bool"),
]


def _par_get(par, chave):
    """Lê um valor do painel. Chaves 'g_*' são POR gesto (do tipo selecionado)."""
    if chave.startswith("g_"):
        return par["gestos"][par["gesto_tipo"]][chave[2:]]
    return par[chave]


def _par_set(par, chave, v):
    """Grava um valor do painel. Chaves 'g_*' vão pro gesto do tipo selecionado."""
    if chave.startswith("g_"):
        par["gestos"][par["gesto_tipo"]][chave[2:]] = v
    else:
        par[chave] = v


def _fmt_val(v, fmt):
    """Formata um valor de `par` para exibição no painel."""
    return {
        "bool": lambda: "ON" if v else "off",
        "sinal": lambda: f"{int(v):+d}",
        "px": lambda: f"{int(v)} px",
        "ms": lambda: f"{v:.0f} ms",
        "deg": lambda: f"{v:.0f} deg",
        "degf1": lambda: f"{v:.1f} deg",
        "cm": lambda: f"{v * 100:.0f} cm",
        "f1ps": lambda: f"{v:.1f}/s",
        "f2": lambda: f"{v:.2f}",
        "f2s": lambda: f"{v:.2f}s",
        "f1s": lambda: f"{v:.1f}s",
        "tipo": lambda: str(v),
    }.get(fmt, lambda: f"{v:.2f}")()


def linhas_ajustes(par, sel):
    """Linhas (txt, cor) do painel de AJUSTES, agrupadas, item selecionado destacado."""
    linhas = [("== AJUSTES ==  w/s move · -/= ajusta · ENTER alterna · n salva · TAB fecha",
               COR_TIT)]
    grupo = None
    for i, item in enumerate(AJUSTES_SPEC):
        g, chave, lbl, fmt = item[0], item[1], item[2], item[6]
        if g != grupo:
            linhas.append((f" {g}", COR_DIM))
            grupo = g
        marca = ">" if i == sel else " "
        cor = COR_OK if i == sel else COR_VAL
        linhas.append((f" {marca} {lbl:<19}{_fmt_val(_par_get(par, chave), fmt)}", cor))
    return linhas


def ajustar_item(par, sel, d):
    """Aplica +/- d*passo ao item selecionado (ou alterna bool/sinal/tipo)."""
    _g, chave, _lbl, passo, mn, mx, fmt = AJUSTES_SPEC[sel]
    cur = _par_get(par, chave)
    if fmt == "bool":
        _par_set(par, chave, not cur)
    elif fmt == "sinal":
        _par_set(par, chave, -int(cur))
    elif fmt == "tipo":                      # cicla os tipos de gesto
        i = GESTO_TIPOS.index(cur) if cur in GESTO_TIPOS else 0
        _par_set(par, chave, GESTO_TIPOS[(i + (1 if d > 0 else -1)) % len(GESTO_TIPOS)])
    else:
        v = min(mx, max(mn, cur + d * passo))
        _par_set(par, chave, int(round(v)) if fmt == "px" else round(float(v), 4))


def salvar_config_ik(repouso, home, sinal_pan, sinal_tilt, radpx_x, radpx_y, par):
    """Salva home + repouso (sentado) + calibração + TODOS os ajustes (dict `par`)."""
    data = {"repouso": [float(x) for x in repouso], "home": [float(x) for x in home],
            "sinal_pan": int(sinal_pan), "sinal_tilt": int(sinal_tilt),
            "radpx_x": float(radpx_x), "radpx_y": float(radpx_y),
            "par": dict(par)}
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def carregar_config_ik():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH) as f:
        return json.load(f)


def par_de_cfg(cfg):
    """Lê o `par` salvo (defaults p/ chaves novas). Compat com config antigo (chaves planas)."""
    par = par_default()
    if isinstance(cfg.get("par"), dict):
        salvos = dict(cfg["par"])
        salvos.pop("gestos", None)          # 'gestos' é mesclado à parte (garante todos os tipos)
        par.update(salvos)
        gsalvos = cfg["par"].get("gestos", {})
        par["gestos"] = {t: {**GESTO_DEF[t], **gsalvos.get(t, {})} for t in GESTO_TIPOS}
    else:                                   # config antigo: chaves planas → mapeia
        mapa = {"ganho": "kp_servo", "zona": "deadzone_px", "limite": "limite_deg",
                "previsao": "previsao_ms", "alt_max": "alt_max", "altura_on": "altura_on",
                "neck_max": "neck_max", "sinal_neck": "sinal_neck", "neck_relax": "neck_relax"}
        for k, old in mapa.items():
            if old in cfg:
                par[k] = cfg[old]
    return par


def _criar_reid(log):
    """Cria o re-ID (ArcFace) sob demanda. Import aqui p/ não pesar o startup quando off."""
    try:
        log("re-ID: carregando ArcFace... (~2s)", "info")
        from reid import ReID
        r = ReID(dispositivo="ovcpu")
        log("re-ID: pronto", "ok")
        return r
    except Exception as e:
        log(f"re-ID indisponivel: {e}", "erro")
        return None


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "ik_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    diario = Diario(log_path)
    sys.stdout = Tee(sys.__stdout__, diario.stdout)   # captura todo o terminal no log
    print(f"--- log desta sessao: {log_path} ---")

    perc = Percepcao(com_corpo=True)     # percepção: rosto (YuNet) + corpo (Pose) -> ALVO
    detector = perc.detector             # reusa o mesmo YuNet na calibração
    reid = None                          # re-ID (ArcFace): criado sob demanda (1ª vez ~2s)
    reid_prev = False                    # p/ detectar a borda "ligou re-ID" → trava na hora
    t_perc0 = time.time()                # base de tempo p/ o pose (modo VIDEO)
    rastreador = RastreadorAlvo(t_pred_ms=PREVISAO_MS)
    try:
        arm = RobotArm()
        arm.connect()
        pos0 = np.asarray(arm.get_positions(request=True), dtype=float)
    except Exception as e:
        print("!!! Nao consegui falar com o braco:", e)
        diario.evento("erro_conexao", msg=str(e))
        diario.close()
        sys.stdout = sys.__stdout__
        tela_sem_braco()
        return
    print("--- conectado ---")
    n = arm.num_joints
    # Ganhos FIRMES = MIT de fábrica (juntas grandes ~120, punho ~18); seguram o peso.
    est["kp_hold"] = np.array([j.kp for j in arm._joints], dtype=float)
    est["kd_hold"] = np.array([j.kd for j in arm._joints], dtype=float)
    print(f"--- kp_hold (firme): {np.round(est['kp_hold'],0)} ---")
    est["q_target"] = pos0.copy()
    est["home"] = pos0.copy()
    est["repouso"] = pos0.copy()      # pose inicial = repouso (comece sentado)

    # Parâmetros de feel (ajustáveis ao vivo pelo painel de AJUSTES, salvos no 'n').
    par = par_default()
    sinal_pan, sinal_tilt = -1, +1     # padrão validado no 1o teste de hardware (calibra com k)
    max_step = np.radians(par["max_step"])
    margem = np.radians(MARGEM_LIMITE_DEG)
    modo_ajuste = False                # 'TAB' abre o painel navegável de ajustes
    sel = 0                            # item selecionado no painel

    fase = "posicionar"        # "posicionar" (flutuando) | "seguir" (IK)
    tracking = False
    calibrado = False          # 'k' mede sinal+escala reais (essencial p/ não divergir)
    base_pan, base_tilt = 0.0, 0.0     # mira (rad) relativa à home (0 = encarando)
    altura = 0.0               # deslocamento vertical da câmera (m) — acompanha sua altura
    sinal_altura = 1.0         # sentido (deduzido da geometria ao travar/acordar)
    n_falhas = 0               # IK seguidas falhando (p/ destravar quando preso)
    alt_teto_cima = 0.5        # teto adaptativo da altura p/ CIMA (m): aprende se o IK falha
    alt_teto_baixo = 0.5       # teto adaptativo p/ BAIXO (m)
    t_prev = time.time()       # p/ o dt do integrador lento da altura
    auto_estado = "seguindo"   # seguindo | perseguindo | varrendo | ocioso
    auto_seta = 0.0            # direção da saída (p/ HUD)
    autonomia = Autonomia(max_step)
    curiosidade = Curiosidade()  # dispara um gesto sozinho quando você fica parado
    gesto_t0 = None            # gesto em andamento (None = nenhum); 'g'/1..7 disparam
    gesto_segue = False        # gesto atual mantém o tracking ativo (não congela o servo)?
    gesto_dir = 1              # alterna o lado a cada disparo (single)
    gesto_tipo = "single"      # tipo do gesto em andamento
    gesto_params = {}          # snapshot dos params (amp/sobe/segura/dir) do disparo
    q_ref = None               # seed da IK SEM o pescoço (não contamina o warm-start)
    neck = 0.0                 # pan atual do punho (rad), p/ HUD/log
    pan_base = 0.0             # parte do pan que a BASE assume (lenta; desenrola o punho)
    geom = None
    ik_ok, ik_iters, ik_ms = True, 0, 0.0
    fault_ativo = False        # algum motor falhou (LED vermelho) → congela e avisa
    frame_idx = 0
    mostra_overlay = True
    toast = [None]
    eventos = deque(maxlen=80)     # log de alto nível ("cérebro") p/ o terminal da web
    _ev = [0]
    pedir_parar = [False]          # ESC/Stop na web → pousa e encerra
    prev_auto = "seguindo"         # p/ logar transições de estado (perseguindo/varrendo…)

    def log_evento(txt, kind="info"):
        _ev[0] += 1
        eventos.append({"id": _ev[0], "t": time.strftime("%H:%M:%S"), "txt": txt, "kind": kind})
        diario.evento("log", txt=txt, kind=kind)

    def aviso(texto, cor=COR_OK):
        toast[0] = (texto, cor, time.time())
        kind = ("ok" if cor == COR_OK else "erro" if cor == COR_ERRO
                else "aviso" if cor == COR_AVISO else "info")
        log_evento(texto, kind)
        print(f"--- {texto} ---")

    def salvar_tudo():
        """Salva home + repouso + calibração + TODOS os ajustes (par). Usado pelo 'n'."""
        if calibrado and est.get("home") is not None:
            salvar_config_ik(est["repouso"], est["home"], sinal_pan, sinal_tilt,
                             radpx_x, radpx_y, par)
            aviso("Config salva! Proxima vez ele acorda e segue sozinho.", COR_OK)
        else:
            aviso("Trave a home (ESPACO) e calibre (k) antes de salvar", COR_AVISO)

    def tocar_gesto(tipo, amp, vel, hold, volta=HT_DESCE, segue=False):
        """Dispara um gesto (tipo + params). 'volta' = tempo de VOLTA (descida).
        'segue' = mantém o tracking ativo durante o gesto (não congela o servo)."""
        nonlocal gesto_t0, gesto_dir, gesto_tipo, gesto_params, gesto_segue
        gesto_dir = -gesto_dir          # alterna o lado (single)
        gesto_tipo = tipo
        gesto_segue = bool(segue)
        gesto_params = {"amp": amp, "sobe": vel, "segura": hold, "desce": volta,
                        "dir": gesto_dir}
        gesto_t0 = time.time()

    def tocar_gesto_tipo(tipo):
        """Seleciona um tipo e dispara com a config DELE (cada gesto tem a sua)."""
        if not par.get("gestos_on", True):       # toggle dinâmico: gestos desligados
            return
        gp = par["gestos"][tipo]
        par["gesto_tipo"] = tipo
        log_evento(f"Gesto: {tipo}", "info")
        tocar_gesto(tipo, gp["amp"], gp["vel"], gp["hold"], gp.get("volta", HT_DESCE),
                    gp.get("segue", False))

    def aplicar_comando(cmd):
        """Aplica um comando vindo da WEB. Roda na thread do loop (seguro). No MVP só
        ações SEM risco (tracking/ajustes/gestos); calibrar/flutuar ficam no teclado."""
        nonlocal tracking, base_pan, base_tilt, altura
        c = cmd.get("cmd")
        if c == "tracking":
            if fase == "seguir" and calibrado:
                tracking = (not tracking) if cmd.get("val") is None else bool(cmd["val"])
                aviso("Tracking ON" if tracking else "Tracking OFF",
                      COR_OK if tracking else COR_DIM)
        elif c == "set_par":                      # ajuste global (ganho, zona, etc.)
            chave, val = cmd.get("chave"), cmd.get("val")
            if chave in par and not isinstance(par[chave], dict):
                par[chave] = val
        elif c == "set_gesto":                    # ajuste de UM gesto (amp/vel/hold/curioso)
            tipo, chave, val = cmd.get("tipo"), cmd.get("chave"), cmd.get("val")
            if tipo in par["gestos"] and chave in par["gestos"][tipo]:
                par["gestos"][tipo][chave] = val
        elif c == "nudge":                        # -/+ no item da spec (reusa a logica do cv2)
            sel = cmd.get("sel")
            if isinstance(sel, int) and 0 <= sel < len(AJUSTES_SPEC):
                ajustar_item(par, sel, 1 if cmd.get("d", 1) >= 0 else -1)
        elif c == "gesto":                        # toca um gesto pelo tipo
            if fase == "seguir" and cmd.get("tipo") in GESTO_TIPOS:
                tocar_gesto_tipo(cmd["tipo"])
        elif c == "recentra":
            base_pan = base_tilt = 0.0
            altura = 0.0
            aviso("Olhar recentrado", COR_VAL)
        elif c == "salvar":
            salvar_tudo()
        elif c == "encara":                       # ESPACO: trava a home e segue
            if fase == "posicionar":
                entrar_em_seguir()
        elif c == "flutua":                       # f: volta a flutuar
            if fase == "seguir":
                voltar_a_flutuar()
        elif c == "calibrar":                     # k: auto-calibração
            if fase == "seguir" and not fault_ativo:
                tracking = False
                calibrar()
            else:
                aviso("Trave a home primeiro (ESPACO)", COR_AVISO)
        elif c == "sentado":                      # z: marca a pose de repouso
            if fase == "posicionar":
                est["repouso"] = np.asarray(arm.get_positions(), dtype=float).copy()
                aviso("Pose 'sentado' (repouso) marcada", COR_OK)
        elif c == "procurar":                     # u: fuga+varredura on/off
            par["procurar_on"] = not par["procurar_on"]
            autonomia.reset()
            aviso("Procurar ON" if par["procurar_on"] else "Procurar OFF",
                  COR_OK if par["procurar_on"] else COR_DIM)
        elif c == "curioso":                      # m: curiosidade on/off
            par["curioso_on"] = not par["curioso_on"]
            curiosidade.reset()
            aviso("Curiosidade ON" if par["curioso_on"] else "Curiosidade OFF",
                  COR_OK if par["curioso_on"] else COR_DIM)
        elif c == "parar":                        # ESC/Stop: pousa e encerra
            pedir_parar[0] = True

    diario.config(modelo="ponto_fixo", sinais=[sinal_pan, sinal_tilt], ganho=par["ganho"],
                  deadzone_px=par["zona"], limite_deg=par["limite"], previsao_ms=par["previsao"],
                  max_step_deg=par["max_step"], ik_flip_deg=IK_FLIP_DEG,
                  margem_limite_deg=MARGEM_LIMITE_DEG, kp_hold=KP,
                  limites_junta_deg=[[round(float(np.degrees(_LO[i])), 1),
                                      round(float(np.degrees(_HI[i])), 1)] for i in range(n)])

    cap = None
    try:
        cap, idx = camera.abrir_camera(CAMERA_PULSO)
        print(f"--- câmera {CAMERA_PULSO} (idx {idx}) ---")
        arm.enable()
        if not motor_pronto(arm):
            print("!!! Braco sem comunicacao (sem energia?).")
            if cap is not None:
                cap.release()
            tela_sem_braco()
            return
        # SEMPRE MIT (como o 08): entra em MIT UMA vez (com o braço sentado = carga
        # baixa, seguro) e o loop único roda do início ao fim. Nunca troca de modo.
        est["q_target"] = np.asarray(arm.get_positions(), dtype=float).copy()
        est["livre"] = False
        est["tracking"] = False
        arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
        arm.start_control_loop(controlador)

        janela = "Camera Follow - IK (PERCEPCAO: segue a pessoa)"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(janela, JANELA_W, JANELA_H)

        # Sobe o servidor web (thread daemon). O braço só é comandado por ESTE laço;
        # a web apenas LÊ (estado/vídeo) e EMPILHA comandos na fila.
        servidor_web.iniciar(spec=AJUSTES_SPEC, porta=8000)
        print("--- painel web: http://localhost:8000  (no celular: http://IP-DO-PC:8000) ---")
        threading.Timer(1.3, lambda: webbrowser.open("http://localhost:8000")).start()

        # radpx (rad de mira por pixel) a partir do FOV — fallback; a calibração mede.
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        radpx_x = np.radians(FOV_H) / w
        radpx_y = np.radians(FOV_V) / h

        def entrar_em_seguir():
            """Trava a home e deriva a geometria. (Sem troca de modo: já em MIT.)"""
            nonlocal fase, geom, base_pan, base_tilt, altura, sinal_altura, gesto_t0
            nonlocal q_ref, pan_base
            gesto_t0 = None
            est["home"] = arm.get_positions().copy()
            geom = geometria(est["home"])
            base_pan = base_tilt = 0.0
            altura = 0.0
            pan_base = 0.0
            sinal_altura = sinal_altura_de(geom)
            q_ref = est["home"].copy()
            autonomia.reset()
            curiosidade.reset()
            est["q_target"] = est["home"].copy()
            est["livre"] = False
            fase = "seguir"
            p0, R0, c, r, opt = geom
            print(f"--- HOME travada. pivô={np.round(c,3)} raio={r:.3f} ---")
            diario.evento("travou_home",
                          home_deg=[round(float(np.degrees(x)), 2) for x in est["home"]],
                          p0=[round(float(v), 3) for v in p0],
                          pivo=[round(float(v), 3) for v in c], raio=round(r, 3),
                          eixo_optico=[round(float(v), 2) for v in opt])
            aviso("Home travada - modo SEGUIR (IK). 't' liga o tracking.", COR_OK)

        def voltar_a_flutuar(msg="Flutuando - mova o braco com a mao. ESPACO encara."):
            """Volta a flutuar (segue a mão). INSTANTÂNEO: já em MIT, só liga o flag."""
            nonlocal fase, tracking, gesto_t0
            est["tracking"] = False
            est["q_target"] = np.asarray(arm.get_positions(), dtype=float).copy()
            est["integral"] = None
            est["livre"] = True
            tracking = False
            gesto_t0 = None
            autonomia.reset()
            curiosidade.reset()
            fase = "posicionar"
            diario.evento("voltou_flutuar",
                          q_deg=[round(float(np.degrees(x)), 1) for x in arm.get_positions()])
            aviso(msg, COR_AVISO)

        def ramp_repouso():
            """Pouso suave: leva o alvo ao repouso devagar e SEGURA até assentar."""
            if fase != "seguir":
                return
            est["tracking"] = False          # o integral sustenta durante a descida
            ini = est["q_target"].copy()
            dest = est["repouso"]
            t0 = time.time()
            while True:
                frac = (time.time() - t0) / DUR_REPOUSO
                if frac >= 1.0:
                    break
                s = frac * frac * (3.0 - 2.0 * frac)
                est["q_target"][:] = ini + (dest - ini) * s
                ret2, fr = cap.read()
                if ret2:
                    desenha_toast(fr, "RETORNANDO AO REPOUSO...", COR_TIT)
                    cv2.imshow(janela, fr)
                    cv2.waitKey(1)
            est["q_target"][:] = dest
            t1 = time.time()
            while time.time() - t1 < 2.5:
                if np.max(np.abs(np.asarray(arm.get_positions()) - dest)) < np.radians(1.0):
                    break
                ret2, fr = cap.read()
                if ret2:
                    desenha_toast(fr, "ASSENTANDO...", COR_TIT)
                    cv2.imshow(janela, fr); cv2.waitKey(1)
            time.sleep(0.3)

        # ── Auto-calibração (malha aberta): cutuca a mira e mede o rosto ──────────
        def ir_para_mira(pan_d, tilt_d, texto):
            """Leva a mira até (pan_d, tilt_d) graus, comandando IK, e assenta."""
            nonlocal base_pan, base_tilt
            alvo_p, alvo_t = np.radians(pan_d), np.radians(tilt_d)
            t0 = time.time()
            while True:
                base_pan += float(np.clip(alvo_p - base_pan, -max_step, max_step))
                base_tilt += float(np.clip(alvo_t - base_tilt, -max_step, max_step))
                q, ok, _, _ = resolver_ik(geom, base_pan, base_tilt, est["q_target"])
                if ok and np.degrees(np.max(np.abs(q - est["q_target"]))) <= IK_FLIP_DEG:
                    np.clip(q, _LO + margem, _HI - margem, out=q)
                    est["q_target"][:] = q
                ret2, fr = cap.read()
                if ret2:
                    desenha_toast(fr, texto, COR_TIT)
                    cv2.imshow(janela, fr); cv2.waitKey(1)
                perto = abs(base_pan - alvo_p) < 1e-3 and abs(base_tilt - alvo_t) < 1e-3
                if (perto and time.time() - t0 > CAL_SETTLE_S) or time.time() - t0 > 5.0:
                    break

        def medir_face(texto):
            """Posição média do rosto (relativa ao centro) por CAL_MEAS_S s."""
            t0 = time.time(); xs = []; ys = []
            while time.time() - t0 < CAL_MEAS_S:
                ret2, fr = cap.read()
                if not ret2:
                    continue
                hh, ww = fr.shape[:2]
                faces = detector.detectar(fr, escala=0.5)
                if faces:
                    f = max(faces, key=lambda f: f.area)
                    xs.append(f.centro_olhos[0] - ww // 2)
                    ys.append(f.centro_olhos[1] - hh // 2)
                desenha_toast(fr, texto, COR_TIT)
                cv2.imshow(janela, fr); cv2.waitKey(1)
            return (float(np.mean(xs)), float(np.mean(ys))) if xs else None

        def calibrar():
            """Mede sinal + escala (px/grau) reais cutucando a mira ±DELTA."""
            nonlocal radpx_x, radpx_y, sinal_pan, sinal_tilt, calibrado
            est["tracking"] = False          # integral ajuda a mira a chegar nos nudges
            D, txt = DELTA_CAL_DEG, "CALIBRANDO... fique parado"
            ir_para_mira(-D, 0, txt); mp_ = medir_face(txt)
            ir_para_mira(+D, 0, txt); pp_ = medir_face(txt)
            ir_para_mira(0, -D, txt); mt_ = medir_face(txt)
            ir_para_mira(0, +D, txt); pt_ = medir_face(txt)
            ir_para_mira(0, 0, txt)
            if not all((mp_, pp_, mt_, pt_)):
                aviso("Calibracao falhou - rosto visivel e parado?", COR_ERRO)
                return
            dfx = pp_[0] - mp_[0]                 # deslocamento do rosto p/ +2D de pan
            dfy = pt_[1] - mt_[1]
            ppd_x, ppd_y = abs(dfx) / (2 * D), abs(dfy) / (2 * D)
            sinal_pan = -1 if dfx > 0 else +1     # centra com o oposto do deslocamento
            sinal_tilt = -1 if dfy > 0 else +1
            radpx_x = np.radians(1.0) / max(ppd_x, 0.1)
            radpx_y = np.radians(1.0) / max(ppd_y, 0.1)
            calibrado = True
            diario.evento("calibrado", sinais=[sinal_pan, sinal_tilt],
                          ppd=[round(ppd_x, 2), round(ppd_y, 2)])
            cor = COR_OK if (ppd_x > 1.0 and ppd_y > 1.0) else COR_AVISO
            aviso(f"Calibrado! pan {ppd_x:.1f} tilt {ppd_y:.1f} px/deg  "
                  f"sinais {sinal_pan:+d}/{sinal_tilt:+d}", cor)

        def acordar(cfg):
            """Com config salva: aplica calibração/ajustes e ACORDA suave (rampa MIT)
            da pose atual até a home, parando na pose (você liga o seguir com 't')."""
            nonlocal fase, geom, base_pan, base_tilt, tracking, calibrado, q_ref, pan_base
            nonlocal sinal_pan, sinal_tilt, radpx_x, radpx_y, altura, sinal_altura
            home = np.asarray(cfg["home"], dtype=float)
            est["home"] = home.copy()
            est["repouso"] = np.asarray(cfg["repouso"], dtype=float)
            sinal_pan, sinal_tilt = int(cfg["sinal_pan"]), int(cfg["sinal_tilt"])
            radpx_x, radpx_y = float(cfg["radpx_x"]), float(cfg["radpx_y"])
            par.update(par_de_cfg(cfg))      # todos os ajustes (compat config antigo)
            geom = geometria(home)
            base_pan = base_tilt = 0.0
            altura = 0.0
            pan_base = 0.0
            sinal_altura = sinal_altura_de(geom)
            q_ref = home.copy()
            est["tracking"] = False          # integral sustenta durante a subida
            ini = np.asarray(arm.get_positions(), dtype=float).copy()
            diario.evento("acordar_ini",
                          ini_deg=[round(float(np.degrees(x)), 1) for x in ini],
                          home_deg=[round(float(np.degrees(x)), 1) for x in home],
                          delta_deg=[round(float(np.degrees(home[i] - ini[i])), 1)
                                     for i in range(n)])
            t0 = time.time()
            while True:                       # rampa suave (o loop MIT segue o alvo)
                frac = (time.time() - t0) / DUR_ACORDAR
                if frac >= 1.0:
                    break
                s = frac * frac * (3.0 - 2.0 * frac)
                est["q_target"][:] = ini + (home - ini) * s
                ret2, fr = cap.read()
                if ret2:
                    desenha_toast(fr, "ACORDANDO...", COR_TIT)
                    cv2.imshow(janela, fr); cv2.waitKey(1)
                diario.frame(fase="acordando", trk=False,
                             qd=[round(float(np.degrees(x)), 1) for x in est["q_target"]],
                             qr=[round(float(np.degrees(x)), 1) for x in arm.get_positions()])
            est["q_target"][:] = home
            calibrado = True
            fase = "seguir"
            tracking = True      # acorda JÁ seguindo (padrão); 't' pausa se quiser
            diario.evento("acordou",
                          home_deg=[round(float(np.degrees(x)), 2) for x in home],
                          sinais=[sinal_pan, sinal_tilt],
                          ppd=[round(np.radians(1) / radpx_x, 1),
                               round(np.radians(1) / radpx_y, 1)])
            aviso("Acordei e estou te seguindo. (t pausa | k recalibra | n salva | ESC)",
                  COR_OK)

        # Decide o início: com config -> acorda na pose; sem -> flutua p/ posar.
        cfg_ik = carregar_config_ik()
        if cfg_ik is not None:
            acordar(cfg_ik)
        else:
            voltar_a_flutuar("1o uso: flutue; 'z' marca o sentado; ESPACO encara; "
                             "'k' calibra; 'n' salva.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            # ---- comandos vindos da WEB (aplicados AQUI, na thread do engine) ----
            while True:
                cmd_web = ESTADO.proximo_comando()
                if cmd_web is None:
                    break
                try:
                    aplicar_comando(cmd_web)
                except Exception as e:
                    diario.evento("erro_comando_web", cmd=cmd_web, msg=str(e))
            if pedir_parar[0]:                       # ESC/Stop pela web → pousa e encerra
                log_evento("Encerrando — pousando no repouso", "aviso")
                ramp_repouso()
                break

            # ---- RE-ID (opcional): trava em VOCÊ e não troca quando outra pessoa entra ----
            if par["reid_on"]:
                if reid is None:
                    reid = _criar_reid(log_evento)        # 1ª vez carrega o ArcFace (~2s)
                if reid is not None:
                    reid.submeter(frame)
                    if not reid_prev:                     # acabou de ligar → trava em quem está
                        reid.travar()
                        log_evento("re-ID: travado na pessoa atual", "ok")
            reid_prev = par["reid_on"]

            # ---- PERCEPÇÃO: o ALVO é o rosto; se a cabeça sai, vira o CORPO (cabeça
            # estimada acima dos ombros) → não perde a pessoa quando senta/levanta/cobre. ----
            est_p = perc.processa(frame, int((time.time() - t_perc0) * 1000),
                                  usar_corpo=par["seguir_corpo"], conf=par["corpo_conf"],
                                  reid=reid if par["reid_on"] else None)
            alvo_face = est_p["rosto"]              # p/ desenhar a caixa do rosto
            fonte_alvo = est_p["fonte"]             # "rosto" | "corpo" | None
            ponto_cru = est_p["alvo"]               # ALVO (rosto -> corpo -> None)

            frame_idx += 1
            agora = time.time()
            dt_frame = min(agora - t_prev, 0.1)   # dt real do loop (p/ a altura)
            t_prev = agora
            rastreador.t_pred_ms = par["previsao"]
            _, prev = rastreador.update(ponto_cru, time.time())

            pos = arm.get_positions()
            status = status_motores(arm)
            # Detecção de falha (LED vermelho): código de erro (>1) em algum motor.
            if not fault_ativo and fase == "seguir" and any(s > 1 for s in status):
                fault_ativo = True
                tracking = False
                diario.evento("FALHA_MOTOR", status=status,
                              qr_deg=[round(float(np.degrees(x)), 1) for x in pos])
                aviso("FALHA NO MOTOR (LED vermelho). Congelado - ESC p/ sair.", COR_ERRO)

            # Liga/desliga o "modo tracking" do loop de controle (zera o integral só
            # quando está seguindo de fato; parado/segurando, o integral sustenta).
            est["tracking"] = (fase == "seguir" and tracking and not fault_ativo)

            max_step = np.radians(par["max_step"])     # passo máx/frame (ajustável)
            lim = np.radians(par["limite"])            # envelope do PAN (amplo, via base)
            lim_tilt = np.radians(par["limite_tilt"])  # envelope do TILT (modesto)
            erro = None
            jump = None
            prev_pan, prev_tilt, prev_altura = base_pan, base_tilt, altura  # p/ DESFAZER

            # ---- SERVO VISUAL → mira (pan, tilt) ----
            # Durante um gesto o pan/tilt normalmente CONGELA; mas gestos com "segue"=ON
            # (ex.: espreitar, single/swing) mantêm o tracking ativo por baixo do gesto.
            rastreia = (gesto_t0 is None) or gesto_segue
            if (fase == "seguir" and tracking and not fault_ativo and rastreia
                    and ponto_cru is not None and prev is not None):
                dx, dy = prev[0] - cx, prev[1] - cy
                erro = (dx, dy)
                if abs(dx) >= par["zona"]:
                    des = sinal_pan * par["ganho"] * dx * radpx_x
                    base_pan = np.clip(base_pan + np.clip(des, -max_step, max_step), -lim, lim)
                if abs(dy) >= par["zona"]:
                    des = sinal_tilt * par["ganho"] * dy * radpx_y
                    base_tilt = np.clip(base_tilt + np.clip(des, -max_step, max_step),
                                        -lim_tilt, lim_tilt)
                # ALTURA (cascata lenta): se o tilt PERSISTE além da zona, sobe/desce a
                # câmera devagar pra re-nivelar o olhar (você na altura dos olhos).
                if par["altura_on"] and abs(base_tilt) > np.radians(par["altura_zona"]):
                    altura += par["altura_ganho"] * sinal_altura * base_tilt * dt_frame
                    lim_c = min(par["alt_max"], alt_teto_cima)    # teto = min(config, alcançável)
                    lim_b = min(par["alt_max"], alt_teto_baixo)
                    altura = float(np.clip(altura, -lim_b, lim_c))

            # ---- FUGA/perseguição: se o rosto SUMIU, vai pro lado que você saiu e
            # espera; se está presente, devolve a mira sem mexer (o servo comanda).
            # (Não persegue durante um head-tilt — o roll some com o rosto na imagem.) ----
            autonomia.varredura_on = par["procurar_on"]
            if (fase == "seguir" and tracking and not fault_ativo and par["procurar_on"]
                    and rastreia):
                base_pan, base_tilt, auto_estado, auto_seta = autonomia.update(
                    ponto_cru, prev, cx, cy, base_pan, base_tilt,
                    sinal_pan, sinal_tilt, radpx_x, radpx_y, lim, lim_tilt)
            else:
                auto_estado = "seguindo"

            if auto_estado != prev_auto and rastreia:            # loga as transições
                _nm = {"perseguindo": "Perdi você — perseguindo o canto",
                       "varrendo": "Procurando ao redor (varredura)",
                       "ocioso": "Ocioso — esperando você voltar",
                       "seguindo": "Te encontrei — seguindo"}
                log_evento(_nm.get(auto_estado, auto_estado),
                           "ok" if auto_estado == "seguindo" else "aviso")
            prev_auto = auto_estado

            # ---- VIDA: curiosidade (gesto sozinho parado) + respirar (overlay ocioso) ----
            idle_pan = idle_tilt = 0.0
            if fase == "seguir" and tracking and not fault_ativo and gesto_t0 is None:
                centrado = (erro is not None and abs(erro[0]) < par["zona"]
                            and abs(erro[1]) < par["zona"])
                parado = float(np.hypot(*rastreador.velocidade())) < par["vel_parado"]
                pronto = (auto_estado == "seguindo" and ponto_cru is not None
                          and centrado and parado and par["curioso_on"])
                if curiosidade.update(agora, pronto, par["parado_s"], par["cooldown_s"]):
                    curiosos = [t for t in GESTO_TIPOS if par["gestos"][t]["curioso"]]
                    if curiosos:
                        log_evento("Curiosidade (você ficou parado)", "aviso")
                        tocar_gesto_tipo(random.choice(curiosos))
                # respira só quando seguindo+presente (e o gesto não acabou de disparar)
                if par["respirar_on"] and auto_estado == "seguindo" and gesto_t0 is None:
                    rp, rt = respirar(agora, par["respirar_amp"])
                    idle_pan, idle_tilt = np.radians(rp), np.radians(rt)

            # ---- GESTO (não-bloqueante). Corpo todo (roll/dtilt/dpan/dreach via pose IK)
            # ou só a cabeça (dq = offset DIRETO numa junta do punho). ----
            roll = dtilt = dpan = dreach = 0.0
            dq = {}
            if gesto_t0 is not None:
                off = aplicar_gesto(gesto_tipo, time.time() - gesto_t0, gesto_params)
                if off is None:
                    gesto_t0 = None
                else:
                    roll = off.get("roll", 0.0)
                    dtilt = off.get("dtilt", 0.0)
                    dpan = off.get("dpan", 0.0)
                    dreach = off.get("dreach", 0.0)
                    dq = off.get("dq", {})

            # ---- PESCOÇO (cascata com "desenrolar") ----
            # O PUNHO (joint5) faz o pan RÁPIDO; a BASE (joint1) "desenrola" o punho de
            # volta ao 0 e assume o giro (a cabeca endireita, o corpo compensa). Se o
            # punho satura (±neck_max), a base assume o excedente NA HORA. Camera pan =
            # pan_base + neck = base_pan sempre (rosto fica centralizado o tempo todo).
            pan_base += par["neck_relax"] * (base_pan - pan_base) * dt_frame   # base assume devagar
            neck = float(np.clip(base_pan - pan_base,
                                 -np.radians(par["neck_max"]), np.radians(par["neck_max"])))
            pan_base = base_pan - neck     # punho saturou → base assume o resto na hora
            base_ik = pan_base

            # ---- IK: mira (base_ik via base) → 6 juntas; o punho entra POR CIMA ----
            if fase == "seguir" and not fault_ativo:
                q_prev_cmd = np.asarray(est["q_target"], dtype=float).copy()   # p/ o slew
                q_ik, ok, ik_iters, ik_ms = resolver_ik(geom, base_ik + dpan + idle_pan,
                                                        base_tilt + dtilt + idle_tilt, q_ref,
                                                        altura, roll, dreach)
                if ok:
                    jump = float(np.degrees(np.max(np.abs(q_ik - q_ref))))
                if ok and jump <= IK_FLIP_DEG:
                    np.clip(q_ik, _LO + margem, _HI - margem, out=q_ik)
                    q_ref[:] = q_ik                          # seed limpo p/ o próximo frame
                    q_ik[PESCOCO_J5] = float(np.clip(        # pescoço: offset no punho
                        q_ik[PESCOCO_J5] + par["sinal_neck"] * neck,
                        _LO[PESCOCO_J5] + margem, _HI[PESCOCO_J5] - margem))
                    for idx, val in dq.items():              # gesto "só cabeça": junta direta
                        q_ik[idx] = float(np.clip(q_ik[idx] + val,
                                                  _LO[idx] + margem, _HI[idx] - margem))
                    est["q_target"][:] = q_ik                # comando = IK + pescoço + gesto
                    ik_ok = True
                    n_falhas = 0
                    # relaxa o teto adaptativo DEVAGAR (re-testa subir mais), só se NÃO no limite
                    if altura < 0.7 * alt_teto_cima:
                        alt_teto_cima = min(0.5, alt_teto_cima + 0.004)
                    if -altura < 0.7 * alt_teto_baixo:
                        alt_teto_baixo = min(0.5, alt_teto_baixo + 0.004)
                else:
                    ik_ok = False
                    n_falhas += 1
                    # RECUPERA baixando SÓ a altura (mantendo pan/tilt + seed LOCAL q_ref):
                    # sem ir pro HOME, sem solavanco. E APRENDE o teto alcançável.
                    passo_alt = 0.01 if altura >= 0 else -0.01   # recolhe 1cm rumo a 0
                    alt_r, achou = altura, None
                    for _ in range(25):
                        alt_r -= passo_alt
                        if (altura >= 0 and alt_r <= 0) or (altura < 0 and alt_r >= 0):
                            alt_r = 0.0
                        qx, okx, _, _ = resolver_ik(geom, base_ik + dpan + idle_pan,
                                                    base_tilt + dtilt + idle_tilt, q_ref,
                                                    alt_r, roll, dreach)
                        if okx and np.degrees(np.max(np.abs(qx - q_ref))) <= IK_FLIP_DEG:
                            achou = (qx, alt_r); break
                        if alt_r == 0.0:
                            break
                    if achou is not None:                    # a ALTURA era a culpada → recolhe suave
                        qx, alt_r = achou
                        if altura >= 0:
                            alt_teto_cima = max(0.0, abs(alt_r))
                        else:
                            alt_teto_baixo = max(0.0, abs(alt_r))
                        altura = alt_r
                        np.clip(qx, _LO + margem, _HI - margem, out=qx)
                        q_ref[:] = qx
                        qx[PESCOCO_J5] = float(np.clip(
                            qx[PESCOCO_J5] + par["sinal_neck"] * neck,
                            _LO[PESCOCO_J5] + margem, _HI[PESCOCO_J5] - margem))
                        for idx, val in dq.items():
                            qx[idx] = float(np.clip(qx[idx] + val,
                                                    _LO[idx] + margem, _HI[idx] - margem))
                        est["q_target"][:] = qx
                        n_falhas = 0
                        diario.evento("ik_teto_altura", alt_cm=round(alt_r * 100, 1),
                                      teto_cima_cm=round(alt_teto_cima * 100, 1))
                    else:                                    # não era a altura → desfaz a mira (suave)
                        base_pan, base_tilt, altura = prev_pan, prev_tilt, prev_altura
                        if n_falhas > FALHAS_MAX:            # fallback raro: recua ao centro (seed LOCAL)
                            base_pan, base_tilt = prev_pan * 0.9, prev_tilt * 0.9
                            pan_base *= 0.9
                            altura = prev_altura * 0.9
                            q2, ok2, _, _ = resolver_ik(geom, pan_base, base_tilt, q_ref, altura)
                            if ok2:
                                np.clip(q2, _LO + margem, _HI - margem, out=q2)
                                passo = np.radians(3.0)
                                q_ref[:] += np.clip(q2 - q_ref, -passo, passo)
                                est["q_target"][:] = q_ref
                                if np.degrees(np.max(np.abs(q2 - q_ref))) < 2.0:
                                    n_falhas = 0
                            diario.evento("ik_destravando",
                                          pan=round(float(np.degrees(base_pan)), 1),
                                          altura=round(altura, 3))

                # SLEW: limita a taxa de mudança do alvo por frame → mata trancos (no
                # tracking normal a mira já é lenta; só "corta" os saltos grandes).
                d = np.clip(np.asarray(est["q_target"], dtype=float) - q_prev_cmd,
                            -SLEW_MAX, SLEW_MAX)
                est["q_target"][:] = q_prev_cmd + d

            # ---- telemetria por frame ----
            diario.frame(i=frame_idx, fase=fase, trk=bool(tracking and not fault_ativo),
                         face=([int(ponto_cru[0]), int(ponto_cru[1])] if ponto_cru else None),
                         prev=([int(prev[0]), int(prev[1])] if prev is not None else None),
                         err=([int(erro[0]), int(erro[1])] if erro else None),
                         pan=round(float(np.degrees(base_pan)), 2),
                         tilt=round(float(np.degrees(base_tilt)), 2),
                         altura=round(altura, 3), n_falhas=n_falhas, auto=auto_estado,
                         neck=round(float(np.degrees(neck)), 1),
                         base_ik=round(float(np.degrees(base_ik)), 1),
                         qd=[round(float(np.degrees(x)), 1) for x in est["q_target"]],
                         qr=[round(float(np.degrees(x)), 1) for x in pos],
                         ik=[bool(ik_ok), int(ik_iters), round(float(ik_ms), 2)],
                         jump=(round(jump, 1) if jump is not None else None),
                         st=status, sg=[sinal_pan, sinal_tilt],
                         fonte=fonte_alvo, npes=est_p["n_rostos"],
                         reid=(None if not par["reid_on"] or reid is None else
                               [bool(reid.alvo_presente), round(reid.alvo_sim, 2)]),
                         par=[round(par["ganho"], 2), par["limite"], par["previsao"]])

            # ---- estado público p/ a web (websocket) ----
            tt = toast[0]                                   # (texto, cor, t0) ou None
            toast_pub = None
            if tt is not None and (agora - tt[2]) < TOAST_DUR:   # só enquanto fresco
                kind = ("ok" if tt[1] == COR_OK else "erro" if tt[1] == COR_ERRO
                        else "aviso" if tt[1] == COR_AVISO else "info")
                toast_pub = {"txt": tt[0], "t": round(tt[2], 3), "kind": kind}
            ESTADO.publicar_estado({
                "fase": fase, "tracking": bool(tracking and not fault_ativo),
                "toast": toast_pub,
                "calibrado": calibrado, "fault": fault_ativo,
                "tem_rosto": est_p["tem_rosto"], "fonte_alvo": fonte_alvo,
                "n_pessoas": est_p["n_rostos"], "dist": est_p["dist"],
                "postura": est_p["postura"],
                "reid": (None if not par["reid_on"] or reid is None else
                         {"presente": reid.alvo_presente, "sim": round(reid.alvo_sim, 2)}),
                "erro": [int(erro[0]), int(erro[1])] if erro else None,
                "pan": round(float(np.degrees(base_pan)), 1),
                "tilt": round(float(np.degrees(base_tilt)), 1),
                "altura_cm": round(altura * 100, 1),
                "ik_ok": bool(ik_ok), "ik_ms": round(float(ik_ms), 1),
                "auto": auto_estado, "fuga_on": par["procurar_on"],
                "gesto_tipo": par["gesto_tipo"], "gesto_ativo": gesto_t0 is not None,
                "sinais": [int(sinal_pan), int(sinal_tilt)],
                "par": {k: v for k, v in par.items() if k != "gestos"},
                "gestos": par["gestos"], "tipos": GESTO_TIPOS,
                "eventos": list(eventos),
            })

            # ---- Desenhos ----
            if mostra_overlay:
                if est_p["_lms"] is not None:           # corpo (pontos leves)
                    for p in est_p["_lms"]:
                        if p.visibility >= 0.5:
                            cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 200, 120), -1)
                if alvo_face is not None:
                    cv2.rectangle(frame, (alvo_face.x, alvo_face.y),
                                  (alvo_face.x + alvo_face.w, alvo_face.y + alvo_face.h),
                                  (0, 230, 0), 2)
                cv2.rectangle(frame, (cx - par["zona"], cy - par["zona"]),
                              (cx + par["zona"], cy + par["zona"]), (0, 200, 200), 1)
                cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
                cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)
                if prev is not None:                    # ALVO: laranja se vier do CORPO
                    cor = ((0, 180, 255) if fonte_alvo == "corpo"
                           else COR_ERRO if (fase == "seguir" and tracking) else COR_AVISO)
                    cv2.line(frame, (cx, cy), prev, cor, 2)
                    cv2.circle(frame, prev, 7, cor, 2)

                if fault_ativo:
                    modo = ("!! FALHA NO MOTOR - congelado. ESC para sair !!", COR_ERRO)
                elif fase == "posicionar":
                    modo = ("POSICIONAR (flutuando) - 'z' marca sentado | ESPACO encara",
                            COR_AVISO)
                elif tracking:
                    modo = ("SEGUIR - TRACKING ON (IK)", COR_OK)
                elif not calibrado:
                    modo = ("SEGUIR - CALIBRE com 'k' (fique parado)", COR_AVISO)
                else:
                    modo = ("SEGUIR - parado (t = tracking)", COR_DIM)
                ik_txt = (f"IK: {'ok ' if ik_ok else 'revertido'} "
                          f"iters={ik_iters} {ik_ms:.2f}ms") if fase == "seguir" else "IK: --"
                cal = ("calibrado: OK" if calibrado else "calibrado: NAO (tecle k)",
                       COR_OK if calibrado else COR_ERRO)
                gp_hud = par["gestos"][par["gesto_tipo"]]
                tipos_txt = " ".join(f"{i+1}:{t[:4]}" for i, t in enumerate(GESTO_TIPOS))
                if modo_ajuste:
                    linhas = linhas_ajustes(par, sel)
                else:
                    linhas = [
                        ("== CAMERA FOLLOW — IK (modular) ==", COR_TIT),
                        modo,
                        cal,
                        (f"mira: pan={np.degrees(base_pan):+.1f} tilt={np.degrees(base_tilt):+.1f}deg  "
                         f"altura={altura*100:+.0f}/{par['alt_max']*100:.0f}cm"
                         f"{'' if par['altura_on'] else ' OFF'}", COR_VAL),
                        (ik_txt, COR_OK if ik_ok else COR_AVISO),
                        (f"erro: {('%+d,%+d px' % erro) if erro else '--'}", COR_TXT),
                        (f"ALVO: {(fonte_alvo or 'perdido').upper()}  pessoas: {est_p['n_rostos']}"
                         f"  dist: {est_p['dist'] or '?'}  postura: {est_p['postura'] or '?'}"
                         + ("" if not par["reid_on"] or reid is None else
                            f"  re-ID: {'VOCE' if reid.alvo_presente else 'sumiu'} {reid.alvo_sim:.2f}"),
                         (0, 180, 255) if fonte_alvo == "corpo"
                         else COR_OK if fonte_alvo == "rosto" else COR_DIM),
                        (f"ganho={par['ganho']:.2f}  zona={par['zona']}px  "
                         f"limite={par['limite']:.0f}deg  prev={par['previsao']:.0f}ms   (TAB ajusta)",
                         COR_VAL),
                        (f"sinais x/y: {sinal_pan:+d}/{sinal_tilt:+d}   "
                         f"ESCALA: {np.radians(1)/radpx_x:.1f} / {np.radians(1)/radpx_y:.1f} px/deg",
                         COR_VAL),
                        (f"procurar(u):{'ON' if par['procurar_on'] else 'off'} "
                         f"curioso(m):{'ON' if par['curioso_on'] else 'off'} "
                         f"respira:{'ON' if par['respirar_on'] else 'off'}  "
                         f"estado: {auto_estado.upper()}"
                         + ((' -->' if auto_seta > 0 else ' <--') if auto_estado != "seguindo" else ""),
                         COR_AVISO if auto_estado in ("perseguindo", "varrendo") else COR_DIM),
                        (f"pescoco: +/-{par['neck_max']:.0f}deg  punho={np.degrees(neck):+.0f}  "
                         f"base={np.degrees(base_ik):+.0f}  sinal={par['sinal_neck']:+d}  "
                         f"desenrola={par['neck_relax']:.1f}", COR_VAL),
                        (f"gesto(g): {par['gesto_tipo']} amp={gp_hud['amp']:.0f} "
                         f"vel={gp_hud['vel']:.2f}s hold={gp_hud['hold']:.1f}s"
                         + ("  [tocando]" if gesto_t0 is not None else "")
                         + f"   1-7: {tipos_txt}",
                         COR_AVISO if gesto_t0 is not None else COR_VAL),
                        ("TAB ajustes | ESPACO encara | k calibra | t pausa | u procurar | "
                         "m curioso | g/1-7 gesto | c recentra | n salva | f flutua | ESC",
                         COR_DIM),
                    ]
                # web: frame com overlays geométricos, SEM o painel de texto (vira HTML)
                ESTADO.publicar_frame(frame.copy())
                painel(frame, 8, 8, linhas, escala=0.5)
            else:
                ESTADO.publicar_frame(frame.copy())   # overlays escondidos: frame cru

            if toast[0] is not None and time.time() - toast[0][2] < TOAST_DUR:
                desenha_toast(frame, toast[0][0], toast[0][1])

            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                diario.evento("tecla", cod=int(k), ch=(chr(k) if 32 <= k < 127 else ""))
            if k == 255:
                pass
            elif modo_ajuste:                # ====== modo AJUSTE (modal) ======
                if k in (9, 27):             # TAB ou ESC fecha o painel
                    modo_ajuste = False
                elif k == ord("w"):
                    sel = (sel - 1) % len(AJUSTES_SPEC)
                elif k == ord("s"):
                    sel = (sel + 1) % len(AJUSTES_SPEC)
                elif k in (ord("="), ord("]"), 13):   # +  (ENTER alterna bool/sinal)
                    ajustar_item(par, sel, +1)
                elif k in (ord("-"), ord("[")):       # -
                    ajustar_item(par, sel, -1)
                elif k == ord("g"):                   # toca o gesto SEM sair do painel
                    if fase == "seguir":
                        tocar_gesto_tipo(par["gesto_tipo"])
                elif k == ord("n"):
                    salvar_tudo()
            elif k in (ord("q"), 27):        # ====== modo NORMAL (ações) ======
                ramp_repouso()
                break
            elif k == 9:                     # TAB abre o painel de AJUSTES
                modo_ajuste = True
                sel = 0
                aviso("AJUSTES: w/s move, -/= ajusta, ENTER alterna, n salva, TAB fecha",
                      COR_VAL)
            elif k == 32:                    # ESPACO: trava home e segue
                if fase == "posicionar":
                    entrar_em_seguir()
            elif k == ord("f"):              # volta a flutuar
                if fase == "seguir":
                    voltar_a_flutuar()
            elif k == ord("k"):              # auto-calibração (fique parado)
                if fase == "seguir" and not fault_ativo:
                    tracking = False
                    calibrar()
                else:
                    aviso("Trave a home primeiro (ESPACO)", COR_AVISO)
            elif k == ord("t"):
                if fase == "seguir":
                    if not calibrado:
                        aviso("Calibre antes: tecle 'k' (fique parado)", COR_AVISO)
                    else:
                        tracking = not tracking
                        aviso("Tracking ON" if tracking else "Tracking OFF",
                              COR_OK if tracking else COR_DIM)
                else:
                    aviso("Trave a home primeiro (ESPACO)", COR_AVISO)
            elif k == ord("x"):
                sinal_pan = -sinal_pan
            elif k == ord("y"):
                sinal_tilt = -sinal_tilt
            elif k == ord("c"):              # recentra o olhar na home (e zera a altura)
                base_pan = base_tilt = 0.0
                altura = 0.0
            elif k == ord("u"):              # procurar (fuga + varredura) on/off
                par["procurar_on"] = not par["procurar_on"]
                autonomia.reset()
                aviso("Procurar ON (persegue + varre quando some)" if par["procurar_on"]
                      else "Procurar OFF", COR_OK if par["procurar_on"] else COR_DIM)
            elif k == ord("m"):              # curiosidade on/off (gesto sozinho parado)
                par["curioso_on"] = not par["curioso_on"]
                curiosidade.reset()
                aviso("Curiosidade ON" if par["curioso_on"] else "Curiosidade OFF",
                      COR_OK if par["curioso_on"] else COR_DIM)
            elif k == ord("g"):              # toca o gesto do tipo selecionado (config dele)
                if fase == "seguir":
                    tocar_gesto_tipo(par["gesto_tipo"])
            elif 49 <= k <= 48 + len(GESTO_TIPOS):   # '1'..'7' tocam cada gesto direto
                if fase == "seguir":
                    tocar_gesto_tipo(GESTO_TIPOS[k - 49])
                else:
                    aviso("Trave a home primeiro (ESPACO)", COR_AVISO)
            elif k == ord("n"):              # salva config (acorda sozinho na proxima)
                salvar_tudo()
            elif k == ord("z"):              # marca a pose 'sentado' (repouso) limpa
                if fase == "posicionar":
                    est["repouso"] = np.asarray(arm.get_positions(), dtype=float).copy()
                    aviso("Pose 'sentado' (repouso) marcada", COR_OK)
                else:
                    aviso("Marque o repouso flutuando ('f' volta a flutuar)", COR_AVISO)
            elif k == ord("i"):
                mostra_overlay = not mostra_overlay

    finally:
        print("\n--- encerrando: desligando torque (APOIE o braço) ---")
        for fn in (arm.stop_control_loop, arm.disable, arm.disconnect):
            try:
                fn()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("--- encerrado ---")
        diario.evento("fim")
        diario.close()
        sys.stdout = sys.__stdout__
        print(f"--- log salvo: {log_path} ---")


if __name__ == "__main__":
    main()
