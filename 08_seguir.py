#!/usr/bin/env python3
"""
Fase 6c - CAMERA FOLLOW: a garra encara o rosto (laço fechado).

Percepção: YuNet -> rastreador (One Euro + Kalman) -> erro em pixels.
Controle: lei PROPORCIONAL ancorada na posição real da junta (à prova de
windup) -> empurra joint5 (pan) e joint4 (tilt) para centralizar o rosto.

  alvo_junta = posicao_atual_da_junta + ganho * erro_angular

Como a câmera gira junto com a junta (~1:1), com ganho <= 1 isso converge
suave e NUNCA passa do ponto (nada de "quicar"/windup).

CALIBRAÇÃO AUTOMÁTICA (tecla k): o programa cutuca cada junta alguns graus,
mede pra que lado o rosto se move na imagem, e deduz sozinho o SINAL e a
ESCALA real (pixels por grau). Sem adivinhação.

Base de hold = MIT + compensação de gravidade (estável, do 07).

SEGURANCA: tracking começa OFF ('t' liga); clamp +/-LIMITE_DEG da neutra;
passo por frame limitado; deadzone; segura sem rosto; desliga torque ao sair.

TECLAS:
  ESPACO ... (flutuando) captura pose neutra e trava
  f ........ volta a flutuar
  k ........ CALIBRAR sinais e escala (fique parado, rosto visível)
  t ........ liga/desliga o TRACKING
  x / y .... inverte manualmente o sinal de pan / tilt
  [ / ] .... ganho - / +
  , / . .... previsão (ms) - / +
  c ........ recentra o olhar na pose neutra
  ESC / q .. sair (desliga torque)
"""

import json
import os
import sys
import time

import numpy as np
import cv2

# Caminho do repo de controle do braço (Seeed). Configurável por variável de
# ambiente REBOT_ARM_REPO para quem clonar o projeto em outra máquina.
ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)
import pinocchio as pin  # noqa: E402
from reBotArm_control_py.actuator import RobotArm  # noqa: E402
from reBotArm_control_py.dynamics import compute_generalized_gravity  # noqa: E402
from reBotArm_control_py.kinematics import load_robot_model  # noqa: E402

import camera  # noqa: E402
from detector import DetectorFaces  # noqa: E402
from rastreador import RastreadorAlvo  # noqa: E402

CAMERA_PULSO = "C920"

# Hold (gravidade + PD + integral gentil), igual ao 07.
KP, KD, KI = 8.0, 2.0, 3.0
VEL_THR, W_THR = 0.04, 0.08
EE_FRAME = "end_link"

PAN, TILT = 4, 3            # joint5, joint4

# Escala inicial (chute por FOV da C920); a calibração substitui pela medida.
FOV_H, FOV_V = 70.0, 43.0

KP_SERVO = 0.20            # ganho proporcional (fração do erro angular)
DEADZONE_PX = 12
MAX_STEP_DEG = 1.5         # passo máx/frame: alto o bastante p/ a lei desacelerar sozinha
LIMITE_DEG = 40.0          # excursão máxima de cada junta (pan/tilt) a partir da neutra
JANELA_W, JANELA_H = 1600, 900   # tamanho inicial da janela de vídeo

# Gesto "head tilt" (inclinar a cabeça, tipo cachorro curioso) no joint6 (roll).
HEAD_TILT_JOINT = 5        # joint6 (índice 5) — rola a "cabeça"/câmera para o lado
HEAD_TILT_DEG = 25.0       # amplitude da inclinação
HT_SOBE, HT_SEGURA, HT_DESCE = 0.30, 0.9, 0.5   # tempos (s): inclina rápido, segura, volta
PREVISAO_MS = 60.0

# Calibração automática.
DELTA_CAL_DEG = 5.0        # quanto cutuca cada junta ao calibrar
SETTLE_S = 0.6             # espera o braço chegar
MEAS_S = 0.3              # janela de medição do rosto

# Movimentos suaves (acordar / dormir).
DUR_ACORDAR = 3.0          # s para subir à pose neutra
DUR_REPOUSO = 3.5          # s para voltar ao repouso (mais lento/suave)

# Arquivo de config (pose neutra + calibração), salvo com a tecla 'n'.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_seguir.json")
# Arquivo de gestos (presets de head-tilt). Portátil/compartilhável.
GESTOS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "gestos.json")

_model = load_robot_model()
_data = _model.createData()
_ee_id = _model.getFrameId(EE_FRAME)

est = {"q_target": None, "home": None, "livre": True, "integral": None,
       "tracking": False}


def controlador(arm, dt):
    """500 Hz: gravidade + segura/segue o alvo (estável, do 07)."""
    if dt <= 0:
        dt = 0.002
    q = arm.get_positions()
    qd = arm.get_velocities()
    tau_g = compute_generalized_gravity(q=q)

    if est["integral"] is None:
        est["integral"] = np.zeros_like(q)
    integ = est["integral"]
    qt = est["q_target"]

    if est["tracking"]:
        # Durante o tracking o LAÇO VISUAL é o integrador. Zera o integral das
        # juntas: com qt em movimento ele acumularia e causaria bounce.
        integ[:] = 0.0
    else:
        # Hold / float / repouso: integral gentil corrige o erro estático.
        integ += (qt - q) * KI * dt
        np.clip(integ, -0.5, 0.5, out=integ)
        if est["livre"]:
            pin.computeJointJacobians(_model, _data, q)
            pin.updateFramePlacements(_model, _data)
            J = pin.getFrameJacobian(_model, _data, _ee_id, pin.ReferenceFrame.WORLD)
            v = J @ qd
            if np.linalg.norm(v[:3]) > VEL_THR or np.linalg.norm(v[3:]) > W_THR:
                qt[:] = q
                integ *= 0.9

    n = arm.num_joints
    arm.mit(pos=qt, vel=np.zeros(n),
            kp=np.full(n, KP), kd=np.full(n, KD), tau=tau_g + integ)


def _mostrar(cap, janela, texto):
    """Lê um frame e desenha um aviso grande nele (usado na calibração)."""
    if janela is None:
        time.sleep(0.01)
        return
    ret, frame = cap.read()
    if ret:
        desenha_toast(frame, texto, COR_TIT)
        cv2.imshow(janela, frame)
        cv2.waitKey(1)


def medir_erro(cap, detector, dur=MEAS_S, janela=None, texto=""):
    """Média do erro (dx, dy) do rosto em relação ao centro, por 'dur' s."""
    t0 = time.time()
    xs, ys = [], []
    while time.time() - t0 < dur:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        faces = detector.detectar(frame, escala=0.5)
        if faces:
            f = max(faces, key=lambda f: f.area)
            p = f.centro_olhos
            xs.append(p[0] - w // 2)
            ys.append(p[1] - h // 2)
        if janela is not None:
            desenha_toast(frame, texto, COR_TIT)
            cv2.imshow(janela, frame)
            cv2.waitKey(1)
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def calibrar_eixo(cap, detector, eixo, comp, janela=None, texto="CALIBRANDO..."):
    """Cutuca a junta 'eixo' em +/-DELTA e mede como o rosto (componente
    'comp': 0=x, 1=y) se desloca. Devolve (sinal, pixels_por_grau) ou None."""
    home, qt = est["home"], est["q_target"]
    delta = np.radians(DELTA_CAL_DEG)

    def esperar(dur):
        t0 = time.time()
        while time.time() - t0 < dur:
            _mostrar(cap, janela, texto)

    qt[eixo] = home[eixo] - delta
    esperar(SETTLE_S)
    menos = medir_erro(cap, detector, janela=janela, texto=texto)

    qt[eixo] = home[eixo] + delta
    esperar(SETTLE_S)
    mais = medir_erro(cap, detector, janela=janela, texto=texto)

    qt[eixo] = home[eixo]
    esperar(SETTLE_S)

    if menos is None or mais is None:
        return None
    d = mais[comp] - menos[comp]          # deslocamento do rosto por +2*delta
    if abs(d) < 8:                         # variação pequena demais -> não confio
        return None
    ppd = abs(d) / (2 * DELTA_CAL_DEG)     # pixels por grau (escala real)
    sinal = -1 if d > 0 else +1            # junta+ moveu rosto +d -> centra com o oposto
    return sinal, ppd


def mover_suave(arm, cap, janela, destino, msg, dur, segura=0.0):
    """Move o braço da posição atual até 'destino' em 'dur' segundos, com
    aceleração e desaceleração suaves (smoothstep) - "acordar"/"dormir".

    'segura' (s): após chegar, mantém o alvo no destino por mais esse tempo,
    com o TORQUE LIGADO, para o braço ALCANÇAR de fato a posição antes de
    desligar (senão ele despenca o último trecho ao perder a energia)."""
    est["livre"] = False
    est["tracking"] = False   # deixa o integral corrigir o erro estático no destino
    qt = est["q_target"]
    inicio = qt.copy()
    destino = np.asarray(destino, dtype=float)
    print(f"--- {msg} ---")

    def _frame():
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, msg, (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(janela, frame)
                cv2.waitKey(1)
                return
        time.sleep(0.02)

    t0 = time.time()
    while True:
        frac = (time.time() - t0) / dur
        if frac >= 1.0:
            break
        s = frac * frac * (3.0 - 2.0 * frac)   # smoothstep: ease-in-out
        qt[:] = inicio + (destino - inicio) * s
        _frame()
    qt[:] = destino
    t1 = time.time()
    while time.time() - t1 < segura:           # segura até o braço chegar de fato
        _frame()


def salvar_config(repouso, neutra, sinal_pan, sinal_tilt, radpx_x, radpx_y,
                  kp_servo, deadzone_px, limite_deg, vida):
    """Salva pose de repouso, pose neutra, calibração e parâmetros de feel."""
    data = {
        "repouso": [float(x) for x in repouso],
        "neutra": [float(x) for x in neutra],
        "sinal_pan": int(sinal_pan),
        "sinal_tilt": int(sinal_tilt),
        "radpx_x": float(radpx_x),
        "radpx_y": float(radpx_y),
        "kp_servo": float(kp_servo),
        "deadzone_px": int(deadzone_px),
        "limite_deg": float(limite_deg),
        "vida": float(vida),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def carregar_config():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH) as f:
        return json.load(f)


def carregar_gestos():
    """Carrega os presets de gestos (slots '1'..'9'). Dict vazio se não houver."""
    if not os.path.exists(GESTOS_PATH):
        return {}
    with open(GESTOS_PATH) as f:
        return json.load(f)


def salvar_gestos(presets):
    with open(GESTOS_PATH, "w") as f:
        json.dump(presets, f, indent=2)


def micro_movimento(t):
    """Offset suave e orgânico (em GRAUS) para dar 'vida' à cabeça.

    Soma de senos com períodos incomensuráveis (não múltiplos entre si), então
    o padrão não se repete de forma óbvia -> parece respiração/espreita natural,
    não um movimento mecânico. Amplitude pequena de propósito (cabe na zona morta).
    """
    pan = 0.6 * np.sin(2 * np.pi * t / 9.0) + 0.25 * np.sin(2 * np.pi * t / 5.3 + 1.0)
    tilt = 0.5 * np.sin(2 * np.pi * t / 4.7) + 0.2 * np.sin(2 * np.pi * t / 7.9 + 2.0)
    return pan, tilt   # graus (escalados depois por 'vida')


def perfil_head_tilt(e, t_sobe, t_segura, t_desce):
    """Fração 0..1 do gesto de inclinar a cabeça ao longo do tempo 'e' (s),
    ou None quando termina. Inclina rápido (smoothstep), segura, e volta suave."""
    dur = t_sobe + t_segura + t_desce
    if e >= dur:
        return None
    if e < t_sobe:                          # subida rápida
        s = e / t_sobe
        return s * s * (3 - 2 * s)
    if e < t_sobe + t_segura:               # segura inclinado (pensando...)
        return 1.0
    s = (e - t_sobe - t_segura) / t_desce   # volta
    return 1.0 - s * s * (3 - 2 * s)


def perfil_swing(e, t_sobe, t_segura, t_desce):
    """Gesto 'swing': vai pra um lado e direto pro inverso, SEM parar no centro.
    Retorna fração de -1..+1 (o sinal já alterna), ou None ao terminar."""
    t1 = t_sobe                  # 0 -> +1
    t2 = t1 + t_segura           # segura +1
    t3 = t2 + 2 * t_sobe         # +1 -> -1 (passa pelo centro)
    t4 = t3 + t_segura           # segura -1
    t5 = t4 + t_desce            # -1 -> 0
    if e >= t5:
        return None
    if e < t1:
        s = e / t_sobe
        return s * s * (3 - 2 * s)
    if e < t2:
        return 1.0
    if e < t3:
        s = (e - t2) / (2 * t_sobe)
        return 1.0 - 2 * (s * s * (3 - 2 * s))
    if e < t4:
        return -1.0
    s = (e - t4) / t_desce
    return -1.0 + (s * s * (3 - 2 * s))


# ----------------------------------------------------------------------------
# Interface visual (HUD, toasts, ajuda)
# ----------------------------------------------------------------------------
COR_TXT = (210, 210, 210)      # texto comum
COR_TIT = (0, 215, 255)        # títulos (âmbar)
COR_OK = (100, 235, 140)       # sucesso (verde)
COR_AVISO = (60, 175, 255)     # aviso (laranja)
COR_ERRO = (80, 80, 245)       # erro (vermelho)
COR_VAL = (235, 225, 130)      # valores/ajustes (ciano claro)
COR_GESTO = (255, 180, 90)     # head-tilt / gestos (azul claro)
COR_DIM = (150, 150, 150)      # dicas discretas
TOAST_DUR = 3.0                # segundos que um aviso fica na tela


def painel(frame, x0, y0, linhas, escala=0.5, alpha=0.66, borda=(85, 85, 85)):
    """Desenha um painel translúcido com borda e linhas coloridas.
    'linhas' = lista de str ou (str, cor_bgr). Devolve (x1, y1)."""
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    th = int(round(28 * (escala / 0.5)))
    larg = 0
    for it in linhas:
        txt = it[0] if isinstance(it, (tuple, list)) else it
        (wt, _), _ = cv2.getTextSize(txt, fonte, escala, 1)
        larg = max(larg, wt)
    x1, y1 = x0 + larg + 24, y0 + th * len(linhas) + 14
    ovl = frame.copy()
    cv2.rectangle(ovl, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.addWeighted(ovl, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), borda, 1)
    y = y0 + th
    for it in linhas:
        txt, cor = it if isinstance(it, (tuple, list)) else (it, COR_TXT)
        cv2.putText(frame, txt, (x0 + 12, y), fonte, escala, cor, 1, cv2.LINE_AA)
        y += th
    return x1, y1


def desenha_toast(frame, texto, cor):
    """Aviso grande, centralizado no topo, com borda colorida pelo tipo."""
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala = 0.75
    (wt, ht), _ = cv2.getTextSize(texto, fonte, escala, 2)
    w = frame.shape[1]
    x0, x1 = (w - wt) // 2 - 20, (w + wt) // 2 + 20
    y0, y1 = 10, 10 + ht + 26
    ovl = frame.copy()
    cv2.rectangle(ovl, (x0, y0), (x1, y1), (15, 15, 15), -1)
    cv2.addWeighted(ovl, 0.78, frame, 0.22, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), cor, 2)
    cv2.putText(frame, texto, (x0 + 20, y0 + ht + 10), fonte, escala, cor, 2, cv2.LINE_AA)


# Páginas de ajuda (tecla 'a' alterna). Cada linha é (texto, cor).
AJUDA_PAGINAS = [
    [("AJUDA  1/4 — TECLAS PRINCIPAIS        [a] proxima   [i] fecha", COR_TIT),
     ("", COR_TXT),
     ("ESPACO   travar a pose neutra (saindo do modo flutuar)", COR_TXT),
     ("t        liga / desliga o SEGUIR (tracking)", COR_TXT),
     ("f        FLUTUAR: mover o braco com a mao para reposicionar", COR_TXT),
     ("c        recentrar o olhar na pose neutra", COR_TXT),
     ("k        CALIBRAR sinal+escala (fique parado, rosto visivel)", COR_TXT),
     ("n        SALVAR config (pose neutra + calibracao + ajustes)", COR_TXT),
     ("r        REINICIAR a aplicacao (recarrega tudo do zero)", COR_TXT),
     ("i        esconder / mostrar todos os elementos da tela", COR_TXT),
     ("ESC / q  sair com o braco voltando suave ao repouso", COR_TXT)],

    [("AJUDA  2/4 — AJUSTES DO TRACKING        [a] proxima", COR_TIT),
     ("", COR_TXT),
     ("[ / ]    GANHO       - / +", COR_VAL),
     ("o / p    ZONA MORTA  - / +", COR_VAL),
     ("- / =    LIMITE      - / +", COR_VAL),
     ("v / b    VIDA        - / +", COR_VAL),
     (", / .    PREVISAO    - / +", COR_VAL),
     ("x / y    inverte o sinal de pan / tilt", COR_VAL),
     ("", COR_TXT),
     ("HEAD-TILT (gestos):", COR_GESTO),
     ("h        inclina a cabeca (alterna o lado)", COR_GESTO),
     ("j / l    inclina forcando esquerda / direita", COR_GESTO),
     ("g        swing (vai a um lado e direto ao outro)", COR_GESTO),
     ("9/0  7/8  4/5   angulo / velocidade / hold", COR_GESTO),
     ("s + 1/2/3   salvar gesto no slot     1/2/3   tocar o gesto", COR_GESTO)],

    [("AJUDA  3/4 — CONCEITOS                   [a] proxima", COR_TIT),
     ("", COR_TXT),
     ("GANHO: forca da correcao. Alto = rapido (pode 'quicar').", COR_TXT),
     ("       Baixo = suave e calmo, porem mais lento.", COR_DIM),
     ("ZONA MORTA: raio no centro onde o braco NAO se mexe.", COR_TXT),
     ("       Maior = cabeca calma (ignora micro-movimentos).", COR_DIM),
     ("LIMITE: o quanto pan/tilt podem girar a partir da neutra.", COR_TXT),
     ("VIDA: micro-movimento ocioso ('respirar'). 0 = estatua.", COR_TXT),
     ("PREVISAO: mira X ms a frente p/ compensar a latencia.", COR_TXT),
     ("CALIBRAR (k): mede sozinho o SINAL e a ESCALA (px por grau)", COR_TXT),
     ("       da sua camera+braco. Substitui qualquer 'chute'.", COR_DIM),
     ("SINAIS: o sentido de cada eixo. x/y invertem se errado.", COR_TXT)],

    [("AJUDA  4/4 — TUTORIAL: gravar um gesto    [a] fecha", COR_TIT),
     ("", COR_TXT),
     ("1) Deixe o tracking normal (ex.: ganho 0.15) e tecle 'n'.", COR_TXT),
     ("2) Ajuste o ganho DO GESTO (ex.: ] ate 0.5); zona/limite", COR_TXT),
     ("   /vida tambem, se quiser.", COR_DIM),
     ("3) Escolha o tipo (h ou g) e esculpa o movimento:", COR_TXT),
     ("   angulo 9/0,  velocidade 7/8,  hold 4/5.", COR_DIM),
     ("4) Salve: tecle 's' depois '1' (ou 2/3).", COR_TXT),
     ("   O preset guarda o tilt + esse contexto de controle.", COR_DIM),
     ("5) Volte ao normal: tecle 'r' (recarrega a config).", COR_TXT),
     ("6) Toque: aperte '1'. Durante o gesto o ganho vira 0.5", COR_TXT),
     ("   e volta sozinho ao normal quando termina.", COR_DIM)],
]


def main():
    detector = DetectorFaces()
    rastreador = RastreadorAlvo(t_pred_ms=PREVISAO_MS)
    arm = RobotArm()
    arm.connect()
    print("--- conectado ---")
    n = arm.num_joints
    est["q_target"] = arm.get_positions(request=True).copy()
    est["home"] = est["q_target"].copy()
    est["repouso"] = est["q_target"].copy()   # pose do início = repouso (comece "sentado")

    kp_servo = KP_SERVO
    deadzone_px = DEADZONE_PX
    limite_deg = LIMITE_DEG
    vida = 1.0                 # escala do micro-movimento ocioso (0 = desligado)
    sinal_pan, sinal_tilt = -1, +1
    previsao_ms = PREVISAO_MS
    tracking = False
    calibrado = False
    # "intenção" do olhar (sem a vida por cima) — a vida é um overlay no qt.
    base_pan = est["home"][PAN]
    base_tilt = est["home"][TILT]
    # Gesto head-tilt (não-bloqueante): t0=início (ou None), dir alterna o lado.
    gesto_t0 = None
    gesto_dir = 1
    gesto_tipo = "single"      # "single" (um lado) ou "swing" (vai-e-vem)
    mostra_overlay = True       # 'i' esconde/mostra todos os elementos na tela
    presets = carregar_gestos()  # slots de gestos salvos ('1'..'3')
    salvando = False             # modo "salvar gesto": após 's', o número escolhe o slot
    gesto_override = {}          # parâmetros de controle a forçar DURANTE o gesto atual
    toast = None                 # (texto, cor, t0) - aviso transitório na tela
    ajuda_pagina = 0             # 0 = sem ajuda; 1..4 = páginas

    def aviso(texto, cor=COR_OK):
        nonlocal toast
        toast = (texto, cor, time.time())
        print(f"--- {texto} ---")
    # Parâmetros do head-tilt, ajustáveis ao vivo (teclas numéricas).
    head_amp = HEAD_TILT_DEG   # amplitude (graus)
    ht_sobe = HT_SOBE          # tempo da inclinada (menor = mais rápido)
    ht_segura = HT_SEGURA      # tempo segurando inclinado
    # rad por pixel (chute inicial por FOV; calibração substitui).
    radpx_x = np.radians(FOV_H) / 1280.0
    radpx_y = np.radians(FOV_V) / 720.0

    cap = None
    try:
        cap, idx = camera.abrir_camera(CAMERA_PULSO)
        print(f"--- câmera {CAMERA_PULSO} (idx {idx}) ---")
        arm.enable()
        arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
        arm.start_control_loop(controlador)

        janela = "Camera Follow - Fase 6c (a garra te encara)"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(janela, JANELA_W, JANELA_H)
        max_step = np.radians(MAX_STEP_DEG)

        # --- Auto-start: se houver config, ACORDA na pose neutra e já segue ---
        cfg = carregar_config()
        if cfg is not None:
            # neutra relativa ao repouso salvo, aplicada ao repouso atual
            # (robusto a pequenas variações da pose inicial entre execuções).
            delta = np.array(cfg["neutra"]) - np.array(cfg["repouso"])
            neutra_alvo = est["repouso"] + delta
            mover_suave(arm, cap, janela, neutra_alvo, "ACORDANDO...", DUR_ACORDAR)
            est["home"] = neutra_alvo.copy()
            est["livre"] = False
            base_pan, base_tilt = est["home"][PAN], est["home"][TILT]
            sinal_pan = cfg["sinal_pan"]
            sinal_tilt = cfg["sinal_tilt"]
            radpx_x = cfg["radpx_x"]
            radpx_y = cfg["radpx_y"]
            kp_servo = cfg.get("kp_servo", KP_SERVO)
            deadzone_px = cfg.get("deadzone_px", DEADZONE_PX)
            limite_deg = cfg.get("limite_deg", LIMITE_DEG)
            vida = cfg.get("vida", 1.0)
            calibrado = True
            tracking = True
            aviso("Acordou - seguindo voce. Tecle 'a' para ajuda.", COR_OK)
        else:
            aviso("Sem config. ESPACO trava, k calibra, t segue, n salva. ('a'=ajuda)",
                  COR_AVISO)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            faces = detector.detectar(frame, escala=0.5)
            alvo_face = max(faces, key=lambda f: f.area) if faces else None
            ponto_cru = alvo_face.centro_olhos if alvo_face else None

            # Parâmetros EFETIVOS: durante um gesto com overrides, valem os do
            # gesto; senão, valem os normais (ao vivo). Reverte sozinho ao fim
            # do gesto, sem salvar/restaurar manual.
            ov = gesto_override if (gesto_t0 is not None and gesto_override) else {}
            kpv = ov.get("ganho", kp_servo)
            dzv = ov.get("zona_morta", deadzone_px)
            limv = ov.get("limite", limite_deg)
            vidav = ov.get("vida", vida)
            prevv = ov.get("prev", previsao_ms)

            rastreador.t_pred_ms = prevv
            _, prev = rastreador.update(ponto_cru, time.time())

            qt, home = est["q_target"], est["home"]
            est["tracking"] = tracking and not est["livre"]
            lim = np.radians(limv)
            pos = arm.get_positions()   # leitura única por loop (tracking + debug)

            # Vida (micro-movimento ocioso), só quando travado. É um OVERLAY:
            # a 'base' guarda a intenção do olhar; a vida vai por cima do qt.
            if not est["livre"]:
                mp, mt = micro_movimento(time.time())
                idle_pan = np.radians(mp * vidav)
                idle_tilt = np.radians(mt * vidav)
            else:
                idle_pan = idle_tilt = 0.0

            # ---- LEI DE CONTROLE proporcional (ancorada na posição real) ----
            erro = None
            if tracking and not est["livre"] and ponto_cru is not None and prev is not None:
                dx, dy = prev[0] - cx, prev[1] - cy
                erro = (dx, dy)
                # Cada eixo só se move FORA da zona morta (dentro, congela).
                # Ancoramos em (posição real - vida): assim o micro-movimento
                # NÃO acumula na base (sem windup). 'base' = intenção do olhar.
                if abs(dx) >= dzv:
                    des = (pos[PAN] - idle_pan) + sinal_pan * kpv * dx * radpx_x
                    des = np.clip(des, home[PAN] - lim, home[PAN] + lim)
                    base_pan += np.clip(des - base_pan, -max_step, max_step)
                if abs(dy) >= dzv:
                    des = (pos[TILT] - idle_tilt) + sinal_tilt * kpv * dy * radpx_y
                    des = np.clip(des, home[TILT] - lim, home[TILT] + lim)
                    base_tilt += np.clip(des - base_tilt, -max_step, max_step)

            # Alvo final = intenção (base) + vida (overlay). Só compõe quando
            # travado; no float, quem manda no qt é o controlador (segue a mão).
            if not est["livre"]:
                qt[PAN] = np.clip(base_pan + idle_pan, home[PAN] - lim, home[PAN] + lim)
                qt[TILT] = np.clip(base_tilt + idle_tilt, home[TILT] - lim, home[TILT] + lim)

            # Gesto head-tilt no joint6 (roll), não-bloqueante: inclina a cabeça
            # enquanto continua te observando (pan/tilt seguem normalmente).
            if gesto_t0 is not None and not est["livre"]:
                e = time.time() - gesto_t0
                if gesto_tipo == "swing":
                    p = perfil_swing(e, ht_sobe, ht_segura, HT_DESCE)
                else:
                    p = perfil_head_tilt(e, ht_sobe, ht_segura, HT_DESCE)
                if p is None:
                    qt[HEAD_TILT_JOINT] = home[HEAD_TILT_JOINT]
                    gesto_t0 = None
                else:
                    qt[HEAD_TILT_JOINT] = (home[HEAD_TILT_JOINT]
                                           + gesto_dir * np.radians(head_amp) * p)

            # ---- Desenhos ----
            if mostra_overlay:
                # Retângulo na face detectada (some quando não há rosto).
                if alvo_face is not None:
                    cv2.rectangle(frame, (alvo_face.x, alvo_face.y),
                                  (alvo_face.x + alvo_face.w, alvo_face.y + alvo_face.h),
                                  (0, 230, 0), 2)
                    cv2.putText(frame, f"{alvo_face.score:.2f}",
                                (alvo_face.x, alvo_face.y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 230, 0), 1, cv2.LINE_AA)
                # Zona morta + mira central.
                cv2.rectangle(frame, (cx - dzv, cy - dzv), (cx + dzv, cy + dzv),
                              (0, 200, 200), 1)
                cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
                cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)
                if prev is not None:
                    cor = COR_ERRO if est["tracking"] else COR_AVISO
                    cv2.line(frame, (cx, cy), prev, cor, 2)
                    cv2.circle(frame, prev, 7, cor, 2)

                if ajuda_pagina > 0:
                    painel(frame, 18, 18, AJUDA_PAGINAS[ajuda_pagina - 1],
                           escala=0.5, alpha=0.82)
                else:
                    if est["livre"]:
                        modo = ("FLUTUANDO  (ESPACO trava)", COR_AVISO)
                    elif tracking:
                        modo = ("TRACKING ON", COR_OK)
                    else:
                        modo = ("PARADO  (t = seguir)", COR_DIM)
                    cal = ("calibrado: OK", COR_OK) if calibrado else \
                        ("calibrado: NAO (tecle k)", COR_ERRO)
                    if salvando:
                        gestos_linha = ("GESTOS  >>> SALVAR: tecle 1 / 2 / 3 <<<", COR_AVISO)
                    else:
                        slots = "  ".join(
                            (f"{s}:{presets[s]['tipo'][:4]}" if s in presets else f"{s}:-")
                            for s in ("1", "2", "3"))
                        gestos_linha = (f"GESTOS  {slots}     s=salvar  1/2/3=tocar", COR_GESTO)
                    linhas = [
                        ("== CAMERA FOLLOW ==", COR_TIT),
                        modo,
                        cal,
                        (f"erro: {('%+d,%+d px' % erro) if erro else '--'}"
                         + ("   [GESTO override]" if ov else ""), COR_TXT),
                        (f"ganho[/]={kpv:.2f}  zona(o/p)={dzv}px  limite(-/=)={limv:.0f}deg  "
                         f"vida(v/b)={vidav:.1f}  prev(,/.)={prevv:.0f}ms", COR_VAL),
                        (f"sinais x/y: pan={sinal_pan:+d} tilt={sinal_tilt:+d}   "
                         f"escala: {1/radpx_x*np.radians(1):.1f}/{1/radpx_y*np.radians(1):.1f} px/deg",
                         COR_VAL),
                        (f"HEAD-TILT {gesto_tipo}:  ang(9/0)={head_amp:.0f}  vel(7/8)={ht_sobe:.2f}s  "
                         f"hold(4/5)={ht_segura:.1f}s    j6={np.degrees(pos[HEAD_TILT_JOINT]):+.0f}deg",
                         COR_GESTO),
                        ("h inclina   j/l fixa lado   g swing", COR_GESTO),
                        gestos_linha,
                        ("[a] AJUDA e legendas completas      [i] esconde tudo", COR_DIM),
                    ]
                    painel(frame, 8, 8, linhas, escala=0.5)

            # Toast (aviso) - sempre visível, mesmo com overlays escondidos.
            if toast is not None and time.time() - toast[2] < TOAST_DUR:
                desenha_toast(frame, toast[0], toast[1])

            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                mover_suave(arm, cap, janela, est["repouso"],
                            "RETORNANDO AO REPOUSO...", DUR_REPOUSO, segura=2.0)
                break
            elif k == ord("r"):
                # Reinício COMPLETO: volta ao repouso, encerra tudo e RE-EXECUTA
                # o script do zero (recarrega o código mais novo, reabre a janela).
                print("--- REINICIANDO a aplicação (recarrega o código)... ---")
                mover_suave(arm, cap, janela, est["repouso"],
                            "REINICIANDO...", DUR_REPOUSO, segura=2.0)
                for fn in (arm.stop_control_loop, arm.disable, arm.disconnect):
                    try:
                        fn()
                    except Exception:
                        pass
                cap.release()
                cv2.destroyAllWindows()
                time.sleep(0.5)   # deixa a porta serial e a câmera liberarem
                os.execv(sys.executable, [sys.executable] + sys.argv)
            elif k == ord("a"):              # alterna as páginas de ajuda
                ajuda_pagina = (ajuda_pagina + 1) % (len(AJUDA_PAGINAS) + 1)
            elif k == 32:  # ESPACO trava
                est["home"] = arm.get_positions().copy()
                est["q_target"][:] = est["home"]
                est["livre"] = False
                base_pan, base_tilt = est["home"][PAN], est["home"][TILT]
                aviso("Travado na pose neutra", COR_OK)
            elif k == ord("f"):
                est["livre"] = True
                tracking = False
                aviso("Flutuando - mova o braco com a mao", COR_AVISO)
            elif k == ord("k"):
                if est["livre"]:
                    aviso("Trave primeiro (ESPACO)", COR_AVISO)
                else:
                    tracking = False
                    rp = calibrar_eixo(cap, detector, PAN, 0, janela,
                                       "CALIBRANDO PAN... fique parado")
                    rt = calibrar_eixo(cap, detector, TILT, 1, janela,
                                       "CALIBRANDO TILT... fique parado")
                    if rp and rt:
                        sinal_pan, ppd_x = rp
                        sinal_tilt, ppd_y = rt
                        radpx_x = np.radians(1.0) / ppd_x
                        radpx_y = np.radians(1.0) / ppd_y
                        calibrado = True
                        aviso(f"Calibrado! pan {ppd_x:.1f} | tilt {ppd_y:.1f} px/deg", COR_OK)
                    else:
                        aviso("Calibracao falhou - rosto visivel e parado?", COR_ERRO)
            elif k == ord("n"):
                salvar_config(est["repouso"], arm.get_positions(),
                              sinal_pan, sinal_tilt, radpx_x, radpx_y,
                              kp_servo, deadzone_px, limite_deg, vida)
                aviso("Config salva (pose + calibracao + ajustes)", COR_OK)
            elif k == ord("t"):
                if est["livre"]:
                    aviso("Trave primeiro (ESPACO)", COR_AVISO)
                else:
                    tracking = not tracking
                    aviso("Tracking ON" if tracking else "Tracking OFF",
                          COR_OK if tracking else COR_DIM)
            elif k == ord("x"):
                sinal_pan = -sinal_pan
            elif k == ord("y"):
                sinal_tilt = -sinal_tilt
            elif k == ord("]"):
                kp_servo = min(1.0, kp_servo + 0.05)
            elif k == ord("["):
                kp_servo = max(0.0, kp_servo - 0.05)
            elif k == ord("p"):
                deadzone_px = min(120, deadzone_px + 3)
            elif k == ord("o"):
                deadzone_px = max(0, deadzone_px - 3)
            elif k == ord("="):
                limite_deg = min(80.0, limite_deg + 5.0)
            elif k == ord("-"):
                limite_deg = max(5.0, limite_deg - 5.0)
            elif k == ord("."):
                previsao_ms = min(250.0, previsao_ms + 10.0)
            elif k == ord(","):
                previsao_ms = max(0.0, previsao_ms - 10.0)
            elif k == ord("c"):
                base_pan, base_tilt = home[PAN], home[TILT]
                qt[PAN], qt[TILT] = home[PAN], home[TILT]
            elif k == ord("b"):
                vida = min(3.0, vida + 0.2)
            elif k == ord("v"):
                vida = max(0.0, vida - 0.2)
            elif k == ord("h"):              # head-tilt: alterna o lado
                if not est["livre"]:
                    gesto_t0 = time.time(); gesto_tipo = "single"; gesto_dir = -gesto_dir
                    gesto_override = {}
            elif k == ord("j"):              # head-tilt: força um lado
                if not est["livre"]:
                    gesto_t0 = time.time(); gesto_tipo = "single"; gesto_dir = -1
                    gesto_override = {}
            elif k == ord("l"):              # head-tilt: força o outro lado
                if not est["livre"]:
                    gesto_t0 = time.time(); gesto_tipo = "single"; gesto_dir = +1
                    gesto_override = {}
            elif k == ord("g"):              # swing: vai-e-vem sem parar no centro
                if not est["livre"]:
                    gesto_t0 = time.time(); gesto_tipo = "swing"; gesto_dir = +1
                    gesto_override = {}
            elif k == ord("i"):              # esconde/mostra todos os overlays
                mostra_overlay = not mostra_overlay
            elif k == ord("s"):              # entra no modo "salvar gesto"
                salvando = True
                aviso("SALVAR gesto: tecle 1, 2 ou 3", COR_AVISO)
            elif k in (ord("1"), ord("2"), ord("3")):
                slot = chr(k)
                if salvando:                 # salva o gesto + contexto de controle
                    presets[slot] = {"tipo": gesto_tipo, "amp": head_amp,
                                     "sobe": ht_sobe, "segura": ht_segura,
                                     "ganho": kp_servo, "zona_morta": deadzone_px,
                                     "limite": limite_deg, "vida": vida,
                                     "prev": previsao_ms}
                    salvar_gestos(presets)
                    salvando = False
                    aviso(f"Gesto {slot} salvo ({gesto_tipo})", COR_OK)
                elif slot in presets and not est["livre"]:   # toca o gesto do slot
                    pr = presets[slot]
                    gesto_tipo = pr["tipo"]
                    head_amp, ht_sobe, ht_segura = pr["amp"], pr["sobe"], pr["segura"]
                    # overrides de controle do gesto (só as chaves presentes).
                    gesto_override = {key: pr[key] for key in
                                      ("ganho", "zona_morta", "limite", "vida", "prev")
                                      if key in pr}
                    gesto_dir = +1 if gesto_tipo == "swing" else -gesto_dir
                    gesto_t0 = time.time()
                    aviso(f"Tocando gesto {slot} ({gesto_tipo})", COR_GESTO)
                elif slot not in presets:
                    aviso(f"Slot {slot} vazio - use s + {slot} para salvar", COR_AVISO)
            # Ajuste do head-tilt (teclas numéricas):
            elif k == ord("0"):
                head_amp = min(110.0, head_amp + 5.0)
            elif k == ord("9"):
                head_amp = max(5.0, head_amp - 5.0)
            elif k == ord("8"):
                ht_sobe = max(0.10, ht_sobe - 0.05)   # mais rápido
            elif k == ord("7"):
                ht_sobe = min(1.50, ht_sobe + 0.05)   # mais lento
            elif k == ord("5"):
                ht_segura = min(4.0, ht_segura + 0.1)
            elif k == ord("4"):
                ht_segura = max(0.0, ht_segura - 0.1)

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


if __name__ == "__main__":
    main()
