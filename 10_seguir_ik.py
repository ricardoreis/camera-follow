#!/usr/bin/env python3
"""10_seguir_ik.py — ETAPA 1: o braço TODO via IK ("pescoço fixo").

Diferença para o 08: lá o olhar usa só 2 juntas do punho. Aqui o olhar vira uma
ORIENTAÇÃO-alvo da câmera e a CINEMÁTICA INVERSA (IK) distribui o movimento pelas 6
juntas — a joint1 (base) sozinha dá ±160° de pan. Modelo "pescoço fixo" (pivô/órbita):
a câmera orbita um ponto fixo (o "pescoço"), sempre mirando pra fora.

  R_alvo = Rz(pan)·Ry(tilt)·R0                       (gira a mira)
  p_alvo = c + r·(Rz(pan)·Ry(tilt)·eixo_óptico)       (orbita o pivô c, raio r)

Arquitetura (ver DOCUMENTACAO seção 13):
  - SEMPRE em modo MIT + compensação de gravidade (igual ao 08). NUNCA troca de modo
    (a troca POS_VEL↔MIT é bloqueante ~0,5s na lib e fazia o braço despencar no 'f' —
    validado no lab_modo.py). Um único loop de 500 Hz roda do início ao fim.
  - FASE POSICIONAR (livre=True): você flutua o braço com a mão até "encarando" e
    trava com ESPACO (captura a HOME, deriva a geometria).
  - FASE SEGUIR (livre=False): a cada frame o servo visual produz (pan, tilt) → IK
    (warm-start) → q_target das 6 juntas; o loop MIT segura/segue esse alvo.

SEGURANÇA (Etapa 1): tracking começa OFF ('t' liga); envelope pequeno (±LIM, ajustável);
se a IK não convergir, SEGURA a última pose; sinais ajustáveis (x/y) — com o envelope
pequeno, sinal errado só vai até o limite e para (sem disparada); ESC volta suave ao
repouso. COMECE com o braço sentado (a pose inicial vira o repouso).

Ainda NÃO tem (vem na Etapa 2): gestos/head-tilt, autonomia, auto-calibração, log CSV.

TECLAS:
  ESPACO  trava a HOME e entra no modo SEGUIR (POS_VEL + IK)
  f       volta a FLUTUAR (re-posicionar com a mão)
  t       liga/desliga o tracking
  x / y   inverte o sinal de pan / tilt
  [ / ]   ganho - / +        - / =  envelope (limite) - / +
  , / .   previsão (ms) - / +     c  recentra o olhar na home
  i       esconde/mostra overlays
  ESC / q sair (braço volta suave ao repouso)
"""

import json
import os
import sys
import time

import numpy as np
import cv2

ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)
import pinocchio as pin  # noqa: E402
from reBotArm_control_py.actuator import RobotArm  # noqa: E402
from reBotArm_control_py.dynamics import compute_generalized_gravity  # noqa: E402
from reBotArm_control_py.kinematics import (  # noqa: E402
    load_robot_model, compute_fk, get_end_effector_frame_id,
)
from reBotArm_control_py.kinematics.inverse_kinematics import (  # noqa: E402
    solve_ik, pos_rot_to_se3, IKParams,
)

import camera  # noqa: E402
from detector import DetectorFaces  # noqa: E402
from rastreador import RastreadorAlvo  # noqa: E402

CAMERA_PULSO = "C920"

# Ganhos do hold MIT. FLUTUAR usa kp MOLE (pra mover com a mão). SEGURAR/SEGUIR usa
# kp FIRME — os ganhos de fábrica do MIT (juntas grandes ~120, punho ~18), senão as
# juntas que carregam o peso cedem sob gravidade e o tilt fica "elástico" (bounce).
KP, KD, KI = 8.0, 2.0, 3.0          # MOLE (flutuar)
VEL_THR, W_THR = 0.04, 0.08

# Servo visual.
KP_SERVO = 0.08            # ganho proporcional (manso; ajuste fino com [ / ])
DEADZONE_PX = 25           # raio central onde NÃO se mexe (mata o tremor parado); o/p ajusta
MAX_STEP_DEG = 1.2         # passo máx de mira/frame (pan agora é 1:1 com joint1 → pode subir)
LIMITE_DEG = 20.0          # envelope pequeno (suavidade primeiro; abre com '=' se quiser)
PREVISAO_MS = 60.0
FOV_H, FOV_V = 70.0, 43.0  # C920: com IK o olhar é a rotação ÓPTICA real → FOV ≈ correto

# Qual coluna de R0 (eixo do end_link no mundo, na home) é o eixo óptico da câmera.
EIXO_OPTICO_COL = 0        # chute = X; confirma-se observando se a órbita faz sentido
# Eixo do CORPO (coluna de R0) usado para o TILT (pitch do punho). body-Y é suave para
# a IK nesta montagem (validado em simulação); o pan vem da base (joint1), não daqui.
TILT_BODY_COL = 1          # 1 = eixo Y do end_link

# IK em tempo real: warm-start + alvo perto → poucas iterações (ver 09_ik_lab.py).
IK_RT = IKParams(max_iter=200, tolerance=1e-3, step_size=0.5, damping=0.01)

# Segurança / suavidade (lições dos testes):
#  - A mira anda devagar (MAX_STEP_DEG) e o hold MIT (kp/kd) limita a velocidade.
#  - Se a IK "virar" (salto grande = singularidade), DESFAZEMOS o passo da mira
#    (em vez de congelar), então nunca há disparada com a câmera morta.
IK_FLIP_DEG = 15.0          # salto de junta acima disso = virada → desfaz a mira
MARGEM_LIMITE_DEG = 4.0     # nunca comandar a junta encostada no batente

# Auto-calibração (tecla 'k'): cutuca a MIRA ±DELTA e mede o deslocamento do rosto.
DELTA_CAL_DEG = 8.0         # quanto cutuca a mira ao calibrar (malha aberta)
CAL_SETTLE_S = 0.8         # espera o braço chegar à mira pedida
CAL_MEAS_S = 0.4           # janela de medição do rosto

JANELA_W, JANELA_H = 1600, 900
DUR_REPOUSO = 3.5
DUR_ACORDAR = 3.0          # s da rampa suave (POS_VEL) até a home ao acordar
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_ik")
# Config (home + repouso + calibração + ajustes), salva com 'n'. Específica do
# SEU braço/câmera → fica no .gitignore.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_seguir_ik.json")

_model = load_robot_model()
_data = _model.createData()
_ee_id = get_end_effector_frame_id(_model)
# Limites de junta (rad), com infinitos saneados — usados para o clamp com margem.
_LO = np.array([x if np.isfinite(x) else -np.pi for x in _model.lowerPositionLimit])
_HI = np.array([x if np.isfinite(x) else np.pi for x in _model.upperPositionLimit])

# Estado partilhado com o loop de controle de 500 Hz.
#  livre    = True: segue a mão (flutuar) | False: segura firme o q_target
#  tracking = True durante o rastreamento → ZERA o integral (o laço visual é o
#             integrador; evita windup/atraso), igual ao 08.
est = {"q_target": None, "home": None, "repouso": None,
       "livre": True, "integral": None, "tracking": False,
       "kp_hold": None, "kd_hold": None}   # ganhos FIRMES (preenchidos após conectar)


# ───────────────────────── loop de controle único (500 Hz) ────────────────────
# SEMPRE em MIT (como o 08). NUNCA troca de modo — era a troca POS_VEL↔MIT (que é
# bloqueante ~0,5s na lib) que deixava os motores sem comando e fazia o braço
# DESPENCAR no 'f' (e dava o giro no acordar). Validado no lab_modo.py.

def controlador(arm, dt):
    """Hold por gravidade (MIT). Se 'livre', segue a mão; se 'tracking', zera o
    integral (o laço visual da IK é o integrador). Igual em espírito ao 07/08."""
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
        integ[:] = 0.0          # laço visual é o integrador → sem windup
    else:
        integ += (qt - q) * KI * dt
        np.clip(integ, -0.5, 0.5, out=integ)
        if est["livre"]:        # detecta a mão movendo o braço → acompanha
            pin.computeJointJacobians(_model, _data, q)
            pin.updateFramePlacements(_model, _data)
            J = pin.getFrameJacobian(_model, _data, _ee_id, pin.ReferenceFrame.WORLD)
            v = J @ qd
            if np.linalg.norm(v[:3]) > VEL_THR or np.linalg.norm(v[3:]) > W_THR:
                qt[:] = q
                integ *= 0.9
    n = arm.num_joints
    if est["livre"]:                       # flutuar com a mão → mole
        kp_arr, kd_arr = np.full(n, KP), np.full(n, KD)
    else:                                  # segurar / seguir → firme (não cede c/ gravidade)
        kp_arr, kd_arr = est["kp_hold"], est["kd_hold"]
    arm.mit(pos=qt, vel=np.zeros(n), kp=kp_arr, kd=kd_arr, tau=tau_g + integ)


# ───────────────────────────── geometria / IK ─────────────────────────────────

def salvar_config_ik(repouso, home, sinal_pan, sinal_tilt, radpx_x, radpx_y,
                     kp_servo, deadzone_px, limite_deg, previsao_ms):
    """Salva home + repouso (sentado) + calibração + ajustes de feel."""
    data = {"repouso": [float(x) for x in repouso], "home": [float(x) for x in home],
            "sinal_pan": int(sinal_pan), "sinal_tilt": int(sinal_tilt),
            "radpx_x": float(radpx_x), "radpx_y": float(radpx_y),
            "kp_servo": float(kp_servo), "deadzone_px": int(deadzone_px),
            "limite_deg": float(limite_deg), "previsao_ms": float(previsao_ms)}
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def carregar_config_ik():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH) as f:
        return json.load(f)


def geometria(q_home):
    """Deriva (p0, R0, pivô c, raio r, eixo óptico) da pose-home, via FK."""
    p0, R0, _ = compute_fk(_model, q_home)
    opt = R0[:, EIXO_OPTICO_COL].copy()
    r = float(np.linalg.norm(p0))
    c = p0 - r * opt
    return p0, R0, c, r, opt


def resolver_ik(geom, pan, tilt, q_seed):
    """Mira (pan, tilt) em rad → pose-alvo SE(3), resolvendo a IK do q_seed.

    PAN nasce da BASE (joint1): giramos a pose-alvo INTEIRA (posição E orientação) em
    torno do Z do mundo na base → a IK resolve com o joint1 (suave, amplo, sem virar).
    TILT é um pitch no eixo do CORPO (body-Y, punho), pós-multiplicado em R0:

        R_alvo = Rz_world(pan) · R0 · Ry_body(tilt)
        p_alvo = Rz_world(pan) · p0

    (parametrizar pan/tilt nos eixos do MUNDO travava a IK quando Z_link≈vertical).
    Devolve (q, ok, iters, ms)."""
    p0, R0, c, r, opt = geom
    Rz = pin.AngleAxis(pan, np.array([0.0, 0.0, 1.0])).matrix()   # giro na base (mundo Z)
    eixo_tilt = R0[:, TILT_BODY_COL]                              # eixo Y do corpo
    Ry = pin.AngleAxis(tilt, eixo_tilt).matrix()
    R = Rz @ Ry @ R0
    p = Rz @ p0
    alvo = pos_rot_to_se3(p, R)
    t0 = time.perf_counter()
    res = solve_ik(_model, _data, _ee_id, alvo, q_seed.copy(), IK_RT)
    ms = (time.perf_counter() - t0) * 1000.0
    return res.q, res.success, res.iterations, ms


# ──────────────────────────────── UI ──────────────────────────────────────────

COR_TXT = (210, 210, 210)
COR_TIT = (0, 215, 255)
COR_OK = (100, 235, 140)
COR_AVISO = (60, 175, 255)
COR_ERRO = (80, 80, 245)
COR_VAL = (235, 225, 130)
COR_DIM = (150, 150, 150)
TOAST_DUR = 3.0


def painel(frame, x0, y0, linhas, escala=0.5, alpha=0.66):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    th = int(round(28 * (escala / 0.5)))
    larg = max(cv2.getTextSize(it[0] if isinstance(it, (tuple, list)) else it,
                               fonte, escala, 1)[0][0] for it in linhas)
    x1, y1 = x0 + larg + 24, y0 + th * len(linhas) + 14
    ovl = frame.copy()
    cv2.rectangle(ovl, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.addWeighted(ovl, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (85, 85, 85), 1)
    y = y0 + th
    for it in linhas:
        txt, cor = it if isinstance(it, (tuple, list)) else (it, COR_TXT)
        cv2.putText(frame, txt, (x0 + 12, y), fonte, escala, cor, 1, cv2.LINE_AA)
        y += th
    return x1, y1


def desenha_toast(frame, texto, cor):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    (wt, ht), _ = cv2.getTextSize(texto, fonte, 0.75, 2)
    w = frame.shape[1]
    x0, x1 = (w - wt) // 2 - 20, (w + wt) // 2 + 20
    ovl = frame.copy()
    cv2.rectangle(ovl, (x0, 10), (x1, 36 + ht), (15, 15, 15), -1)
    cv2.addWeighted(ovl, 0.78, frame, 0.22, 0, dst=frame)
    cv2.rectangle(frame, (x0, 10), (x1, 36 + ht), cor, 2)
    cv2.putText(frame, texto, (x0 + 20, 20 + ht), fonte, 0.75, cor, 2, cv2.LINE_AA)


def tela_sem_braco():
    img = np.full((300, 940, 3), 25, np.uint8)
    for i, (txt, cor) in enumerate([
            ("SEM COMUNICACAO COM O BRACO B601-DM", COR_ERRO),
            ("Verifique se esta LIGADO e CONECTADO (USB / MotorBridge).", COR_TXT),
            ("Ligue/conecte e rode de novo. (tecle algo p/ sair)", COR_DIM)]):
        cv2.putText(img, txt, (28, 70 + i * 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, cor, 2, cv2.LINE_AA)
    cv2.imshow("Camera Follow IK", img)
    cv2.waitKey(8000)
    cv2.destroyAllWindows()


def motor_pronto(arm):
    try:
        for jc in arm._joints:
            st = arm._motor_map[jc.name].get_state()
            if st is None or getattr(st, "status_code", None) != 1:
                return False
        return True
    except Exception:
        return False


def status_motores(arm):
    """Código de status de cada motor (1 = habilitado/ok; >1 = erro/falha;
    0 = desabilitado; -1 sem leitura). Para detectar a falha (LED vermelho)."""
    out = []
    for jc in arm._joints:
        try:
            st = arm._motor_map[jc.name].get_state()
            out.append(int(getattr(st, "status_code", -1)) if st is not None else -1)
        except Exception:
            out.append(-1)
    return out


# ───────────────────────── log detalhado (JSONL) ──────────────────────────────

class Diario:
    """Log estruturado (uma linha JSON por registro). Captura config, eventos
    (teclas, modos, falhas), saída de terminal e telemetria por frame. Pensado
    para ser LIDO depois (inclusive pelo assistente) e reconstruir a sessão."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.t0 = time.time()
        self.path = path

    def _w(self, obj):
        obj["t"] = round(time.time() - self.t0, 3)
        try:
            self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.f.flush()      # flush imediato: se travar/cair, o log sobrevive
        except Exception:
            pass

    def config(self, **kv):
        self._w({"tipo": "config", **kv})

    def evento(self, ev, **kv):
        self._w({"tipo": "evento", "ev": ev, **kv})

    def frame(self, **kv):
        self._w({"tipo": "frame", **kv})

    def stdout(self, linha):
        self._w({"tipo": "stdout", "linha": linha})

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class Tee:
    """Espelha o stdout: escreve no terminal E manda cada linha para um callback
    (para gravar no log toda 'saída de terminal', inclusive a da lib do braço)."""

    def __init__(self, orig, cb):
        self.orig, self.cb, self._buf = orig, cb, ""

    def write(self, s):
        self.orig.write(s)
        self._buf += s
        while "\n" in self._buf:
            linha, self._buf = self._buf.split("\n", 1)
            if linha.strip():
                try:
                    self.cb(linha)
                except Exception:
                    pass

    def flush(self):
        self.orig.flush()


# ──────────────────────────────── main ────────────────────────────────────────

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "ik_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
    diario = Diario(log_path)
    sys.stdout = Tee(sys.__stdout__, diario.stdout)   # captura todo o terminal no log
    print(f"--- log desta sessao: {log_path} ---")

    detector = DetectorFaces()
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

    # Parâmetros de feel (ajustáveis ao vivo).
    kp_servo, deadzone_px, limite_deg, previsao_ms = KP_SERVO, DEADZONE_PX, LIMITE_DEG, PREVISAO_MS
    sinal_pan, sinal_tilt = -1, +1     # padrão validado no 1o teste de hardware
    max_step = np.radians(MAX_STEP_DEG)
    margem = np.radians(MARGEM_LIMITE_DEG)

    fase = "posicionar"        # "posicionar" (MIT/float) | "seguir" (POS_VEL/IK)
    tracking = False
    calibrado = False          # 'k' mede sinal+escala reais (essencial p/ não divergir)
    base_pan, base_tilt = 0.0, 0.0     # mira (rad) relativa à home (0 = encarando)
    geom = None
    ik_ok, ik_iters, ik_ms = True, 0, 0.0
    fault_ativo = False        # algum motor falhou (LED vermelho) → congela e avisa
    frame_idx = 0
    mostra_overlay = True
    toast = [None]

    def aviso(texto, cor=COR_OK):
        toast[0] = (texto, cor, time.time())
        print(f"--- {texto} ---")
        diario.evento("aviso", txt=texto)

    diario.config(modelo="ponto_fixo", sinais=[sinal_pan, sinal_tilt], ganho=kp_servo,
                  deadzone_px=deadzone_px, limite_deg=limite_deg, previsao_ms=previsao_ms,
                  max_step_deg=MAX_STEP_DEG, ik_flip_deg=IK_FLIP_DEG,
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
        # baixa, seguro) e o loop único roda do início ao fim. Nunca troca de modo →
        # 'f' instantâneo, sem despencar; acordar sobe reto.
        est["q_target"] = np.asarray(arm.get_positions(), dtype=float).copy()
        est["livre"] = False
        est["tracking"] = False
        arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
        arm.start_control_loop(controlador)

        janela = "Camera Follow - Etapa 1 (braco todo via IK)"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(janela, JANELA_W, JANELA_H)

        # radpx (rad de mira por pixel) a partir do FOV — fallback; a calibração mede.
        ret, frame = cap.read()
        h, w = frame.shape[:2]
        radpx_x = np.radians(FOV_H) / w
        radpx_y = np.radians(FOV_V) / h

        def entrar_em_seguir():
            """Trava a home e deriva a geometria. (Sem troca de modo: já em MIT.)"""
            nonlocal fase, geom, base_pan, base_tilt
            est["home"] = arm.get_positions().copy()
            geom = geometria(est["home"])
            base_pan = base_tilt = 0.0
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
            """Volta a flutuar (segue a mão). INSTANTÂNEO: já estamos em MIT, então é
            só ligar o flag — sem troca de modo, sem vão, SEM DESPENCAR (validado no lab)."""
            nonlocal fase, tracking
            est["tracking"] = False
            est["q_target"] = np.asarray(arm.get_positions(), dtype=float).copy()
            est["integral"] = None
            est["livre"] = True
            tracking = False
            fase = "posicionar"
            diario.evento("voltou_flutuar",
                          q_deg=[round(float(np.degrees(x)), 1) for x in arm.get_positions()])
            aviso(msg, COR_AVISO)

        def ramp_repouso():
            """Pouso suave: leva o alvo ao repouso devagar (POS_VEL já limita a vel.)."""
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
            # SEGURA até CHEGAR de fato (senão o torque corta antes e ele "cai" o
            # último trecho). Espera assentar dentro de ~1° ou estoura o tempo.
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
            """Com config salva: aplica calibração/ajustes e ACORDA suave (POS_VEL,
            sem tranco) da pose atual até a home, já ligando o tracking."""
            nonlocal fase, geom, base_pan, base_tilt, tracking, calibrado
            nonlocal sinal_pan, sinal_tilt, radpx_x, radpx_y
            nonlocal kp_servo, deadzone_px, limite_deg, previsao_ms
            home = np.asarray(cfg["home"], dtype=float)
            est["home"] = home.copy()
            est["repouso"] = np.asarray(cfg["repouso"], dtype=float)
            sinal_pan, sinal_tilt = int(cfg["sinal_pan"]), int(cfg["sinal_tilt"])
            radpx_x, radpx_y = float(cfg["radpx_x"]), float(cfg["radpx_y"])
            kp_servo = float(cfg.get("kp_servo", KP_SERVO))
            deadzone_px = int(cfg.get("deadzone_px", DEADZONE_PX))
            limite_deg = float(cfg.get("limite_deg", LIMITE_DEG))
            previsao_ms = float(cfg.get("previsao_ms", PREVISAO_MS))
            geom = geometria(home)
            base_pan = base_tilt = 0.0
            est["tracking"] = False          # integral sustenta durante a subida
            ini = np.asarray(arm.get_positions(), dtype=float).copy()
            # Log do ponto de partida vs destino (p/ ver o que cada junta percorre).
            diario.evento("acordar_ini",
                          ini_deg=[round(float(np.degrees(x)), 1) for x in ini],
                          home_deg=[round(float(np.degrees(x)), 1) for x in home],
                          delta_deg=[round(float(np.degrees(home[i] - ini[i])), 1)
                                     for i in range(n)])
            t0 = time.time()
            while True:                       # rampa suave (POS_VEL já está rodando)
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
            tracking = False     # acorda NA pose configurada; você liga o seguir com 't'
            diario.evento("acordou",
                          home_deg=[round(float(np.degrees(x)), 2) for x in home],
                          sinais=[sinal_pan, sinal_tilt],
                          ppd=[round(np.radians(1) / radpx_x, 1),
                               round(np.radians(1) / radpx_y, 1)])
            aviso("Acordei na sua pose. Tecle 't' para te seguir. (k recalibra | n salva | ESC)",
                  COR_OK)

        # Decide o início: com config -> acorda e já segue; sem -> flutua p/ posar.
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

            faces = detector.detectar(frame, escala=0.5)
            alvo_face = max(faces, key=lambda f: f.area) if faces else None
            ponto_cru = alvo_face.centro_olhos if alvo_face else None

            frame_idx += 1
            rastreador.t_pred_ms = previsao_ms
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

            lim = np.radians(limite_deg)
            erro = None
            jump = None
            prev_pan, prev_tilt = base_pan, base_tilt   # p/ DESFAZER se a IK não acompanhar

            # ---- SERVO VISUAL → mira (pan, tilt) ----
            if (fase == "seguir" and tracking and not fault_ativo
                    and ponto_cru is not None and prev is not None):
                dx, dy = prev[0] - cx, prev[1] - cy
                erro = (dx, dy)
                if abs(dx) >= deadzone_px:
                    des = sinal_pan * kp_servo * dx * radpx_x
                    base_pan = np.clip(base_pan + np.clip(des, -max_step, max_step), -lim, lim)
                if abs(dy) >= deadzone_px:
                    des = sinal_tilt * kp_servo * dy * radpx_y
                    base_tilt = np.clip(base_tilt + np.clip(des, -max_step, max_step), -lim, lim)

            # ---- IK: mira → orientação (ponto fixo) → 6 juntas ----
            # A mira só AVANÇA se a IK conseguir acompanhar. Se a solução "virar"
            # (salto grande = singularidade) ou não convergir, DESFAZEMOS o passo da
            # mira (sem deadlock). A velocidade real é limitada pelo POS_VEL (vlim).
            if fase == "seguir" and not fault_ativo:
                q_ik, ok, ik_iters, ik_ms = resolver_ik(geom, base_pan, base_tilt, est["q_target"])
                if ok:
                    jump = float(np.degrees(np.max(np.abs(q_ik - est["q_target"]))))
                if ok and jump <= IK_FLIP_DEG:
                    np.clip(q_ik, _LO + margem, _HI - margem, out=q_ik)
                    est["q_target"][:] = q_ik
                    ik_ok = True
                else:                        # não convergiu OU virada → desfaz a mira
                    ik_ok = False
                    base_pan, base_tilt = prev_pan, prev_tilt
                    diario.evento("ik_revertido", ok=bool(ok),
                                  jump_deg=(round(jump, 1) if jump is not None else None),
                                  pan=round(float(np.degrees(prev_pan)), 1),
                                  tilt=round(float(np.degrees(prev_tilt)), 1))

            # ---- telemetria por frame (o que eu leio depois para "ver" a sessão) ----
            diario.frame(i=frame_idx, fase=fase, trk=bool(tracking and not fault_ativo),
                         face=([int(ponto_cru[0]), int(ponto_cru[1])] if ponto_cru else None),
                         prev=([int(prev[0]), int(prev[1])] if prev is not None else None),
                         err=([int(erro[0]), int(erro[1])] if erro else None),
                         pan=round(float(np.degrees(base_pan)), 2),
                         tilt=round(float(np.degrees(base_tilt)), 2),
                         qd=[round(float(np.degrees(x)), 1) for x in est["q_target"]],
                         qr=[round(float(np.degrees(x)), 1) for x in pos],
                         ik=[bool(ik_ok), int(ik_iters), round(float(ik_ms), 2)],
                         jump=(round(jump, 1) if jump is not None else None),
                         st=status, sg=[sinal_pan, sinal_tilt],
                         par=[round(kp_servo, 2), limite_deg, previsao_ms])

            # ---- Desenhos ----
            if mostra_overlay:
                if alvo_face is not None:
                    cv2.rectangle(frame, (alvo_face.x, alvo_face.y),
                                  (alvo_face.x + alvo_face.w, alvo_face.y + alvo_face.h),
                                  (0, 230, 0), 2)
                cv2.rectangle(frame, (cx - deadzone_px, cy - deadzone_px),
                              (cx + deadzone_px, cy + deadzone_px), (0, 200, 200), 1)
                cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
                cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)
                if prev is not None:
                    cor = COR_ERRO if (fase == "seguir" and tracking) else COR_AVISO
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
                linhas = [
                    ("== CAMERA FOLLOW — IK (Etapa 1) ==", COR_TIT),
                    modo,
                    cal,
                    (f"mira: pan={np.degrees(base_pan):+.1f}  tilt={np.degrees(base_tilt):+.1f} deg "
                     f"(envelope +/-{limite_deg:.0f})", COR_VAL),
                    (ik_txt, COR_OK if ik_ok else COR_AVISO),
                    (f"erro: {('%+d,%+d px' % erro) if erro else '--'}", COR_TXT),
                    (f"ganho[/]={kp_servo:.2f}  zona(o/p)={deadzone_px}px  "
                     f"limite(-/=)={limite_deg:.0f}deg  prev(,/.)={previsao_ms:.0f}ms",
                     COR_VAL),
                    (f"sinais x/y: {sinal_pan:+d}/{sinal_tilt:+d}   "
                     f"ESCALA: {np.radians(1)/radpx_x:.1f} / {np.radians(1)/radpx_y:.1f} px/deg",
                     COR_VAL),
                    ("ESPACO encara | k calibra | n salva | t pausa | o/p zona | x/y sinais | ESC",
                     COR_DIM),
                ]
                painel(frame, 8, 8, linhas, escala=0.5)

            if toast[0] is not None and time.time() - toast[0][2] < TOAST_DUR:
                desenha_toast(frame, toast[0][0], toast[0][1])

            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                diario.evento("tecla", cod=int(k), ch=(chr(k) if 32 <= k < 127 else ""))
            if k in (ord("q"), 27):
                ramp_repouso()
                break
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
            elif k == ord("]"):
                kp_servo = min(1.0, round(kp_servo + 0.02, 2))
            elif k == ord("["):
                kp_servo = max(0.0, round(kp_servo - 0.02, 2))
            elif k == ord("p"):
                deadzone_px = min(120, deadzone_px + 3)
            elif k == ord("o"):
                deadzone_px = max(0, deadzone_px - 3)
            elif k == ord("="):
                limite_deg = min(120.0, limite_deg + 5.0)
            elif k == ord("-"):
                limite_deg = max(5.0, limite_deg - 5.0)
            elif k == ord("."):
                previsao_ms = min(250.0, previsao_ms + 10.0)
            elif k == ord(","):
                previsao_ms = max(0.0, previsao_ms - 10.0)
            elif k == ord("c"):              # recentra o olhar na home
                base_pan = base_tilt = 0.0
            elif k == ord("n"):              # salva config (acorda sozinho na proxima)
                if calibrado and est.get("home") is not None:
                    salvar_config_ik(est["repouso"], est["home"], sinal_pan, sinal_tilt,
                                     radpx_x, radpx_y, kp_servo, deadzone_px,
                                     limite_deg, previsao_ms)
                    aviso("Config salva! Proxima vez ele acorda e segue sozinho.", COR_OK)
                else:
                    aviso("Trave a home (ESPACO) e calibre (k) antes de salvar", COR_AVISO)
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
