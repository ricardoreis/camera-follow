#!/usr/bin/env python3
"""seguir_ik.py — braço TODO via IK ("pescoço fixo"), versão MODULAR.

Mesma funcionalidade do 10_seguir_ik.py, mas separada em módulos (refactor
lado-a-lado; o 10 segue intacto como referência até esta versão ser validada):

    mira_ik.py         modelo Pinocchio + IK (geometria, resolver_ik, sinal_altura)
    controle_braco.py  loop MIT (gravidade) + estado `est` + motores
    ui_hud.py          cores, painel, toast, tela sem braço
    diario.py          log JSONL + Tee do terminal
    seguir_ik.py       (este) orquestração: laço principal + teclas + servo/altura

Arquitetura: SEMPRE MIT (nunca troca de modo). PAN pela base, TILT pelo punho,
ALTURA acompanha sua altura (sobe/desce devagar). Ver DOCUMENTACAO seção 13.

TECLAS: ESPACO trava a home · k calibra · t segue · f flutua · z marca sentado ·
n salva · h altura on/off · v/b alcance de altura · o/p zona · -/= envelope ·
[/] ganho · ,/. previsão · x/y sinais · c recentra · i esconde · ESC sai.
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
from autonomia import Autonomia  # noqa: E402
from gestos import perfil_head_tilt, HEAD_TILT_DEG  # noqa: E402

import camera  # noqa: E402
from detector import DetectorFaces  # noqa: E402
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

# Auto-calibração (tecla 'k'): cutuca a MIRA ±DELTA e mede o deslocamento do rosto.
DELTA_CAL_DEG = 8.0
CAL_SETTLE_S = 0.8
CAL_MEAS_S = 0.4

JANELA_W, JANELA_H = 1600, 900
DUR_REPOUSO = 3.5
DUR_ACORDAR = 3.0
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_ik")
# Config (home + repouso + calibração + ajustes), salva com 'n' — gitignored.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_seguir_ik.json")


def salvar_config_ik(repouso, home, sinal_pan, sinal_tilt, radpx_x, radpx_y,
                     kp_servo, deadzone_px, limite_deg, previsao_ms,
                     alt_max, altura_on, neck_max, sinal_neck, neck_relax):
    """Salva home + repouso (sentado) + calibração + ajustes de feel + altura + pescoço."""
    data = {"repouso": [float(x) for x in repouso], "home": [float(x) for x in home],
            "sinal_pan": int(sinal_pan), "sinal_tilt": int(sinal_tilt),
            "radpx_x": float(radpx_x), "radpx_y": float(radpx_y),
            "kp_servo": float(kp_servo), "deadzone_px": int(deadzone_px),
            "limite_deg": float(limite_deg), "previsao_ms": float(previsao_ms),
            "alt_max": float(alt_max), "altura_on": bool(altura_on),
            "neck_max": float(neck_max), "sinal_neck": int(sinal_neck),
            "neck_relax": float(neck_relax)}
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def carregar_config_ik():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH) as f:
        return json.load(f)


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

    fase = "posicionar"        # "posicionar" (flutuando) | "seguir" (IK)
    tracking = False
    calibrado = False          # 'k' mede sinal+escala reais (essencial p/ não divergir)
    base_pan, base_tilt = 0.0, 0.0     # mira (rad) relativa à home (0 = encarando)
    altura = 0.0               # deslocamento vertical da câmera (m) — acompanha sua altura
    altura_on = True           # 'h' liga/desliga o acompanhamento de altura
    alt_max = ALTURA_MAX_DEFAULT  # alcance vertical (± m), ajustável ao vivo (v/b)
    sinal_altura = 1.0         # sentido (deduzido da geometria ao travar/acordar)
    n_falhas = 0               # IK seguidas falhando (p/ destravar quando preso)
    t_prev = time.time()       # p/ o dt do integrador lento da altura
    perseguicao_on = True      # 'u': quando o rosto some, vai pro lado que voce saiu
    auto_estado = "seguindo"   # seguindo | perseguindo | esperando
    auto_seta = 0.0            # direção da saída (p/ HUD)
    autonomia = Autonomia(max_step)
    gesto_t0 = None            # 'g': head-tilt em andamento (None = nenhum)
    gesto_dir = 1              # alterna o lado a cada disparo
    neck_max = NECK_MAX_DEG    # faixa do pescoço (punho) em graus; ajustável (9/0)
    sinal_neck = -1            # sentido do joint5 (lab: oposto da base); flip com 'j'
    neck_relax = NECK_RELAX    # velocidade do "desenrolar" (/s); ajustável (w/e), salva no n
    q_ref = None               # seed da IK SEM o pescoço (não contamina o warm-start)
    neck = 0.0                 # pan atual do punho (rad), p/ HUD/log
    pan_base = 0.0             # parte do pan que a BASE assume (lenta; desenrola o punho)
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
        # baixa, seguro) e o loop único roda do início ao fim. Nunca troca de modo.
        est["q_target"] = np.asarray(arm.get_positions(), dtype=float).copy()
        est["livre"] = False
        est["tracking"] = False
        arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
        arm.start_control_loop(controlador)

        janela = "Camera Follow - IK (modular)"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(janela, JANELA_W, JANELA_H)

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
            nonlocal kp_servo, deadzone_px, limite_deg, previsao_ms, alt_max, altura_on
            nonlocal neck_max, sinal_neck, neck_relax
            home = np.asarray(cfg["home"], dtype=float)
            est["home"] = home.copy()
            est["repouso"] = np.asarray(cfg["repouso"], dtype=float)
            sinal_pan, sinal_tilt = int(cfg["sinal_pan"]), int(cfg["sinal_tilt"])
            radpx_x, radpx_y = float(cfg["radpx_x"]), float(cfg["radpx_y"])
            kp_servo = float(cfg.get("kp_servo", KP_SERVO))
            deadzone_px = int(cfg.get("deadzone_px", DEADZONE_PX))
            limite_deg = float(cfg.get("limite_deg", LIMITE_DEG))
            previsao_ms = float(cfg.get("previsao_ms", PREVISAO_MS))
            alt_max = float(cfg.get("alt_max", ALTURA_MAX_DEFAULT))
            altura_on = bool(cfg.get("altura_on", True))
            neck_max = float(cfg.get("neck_max", NECK_MAX_DEG))
            sinal_neck = int(cfg.get("sinal_neck", -1))
            neck_relax = float(cfg.get("neck_relax", NECK_RELAX))
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

            faces = detector.detectar(frame, escala=0.5)
            alvo_face = max(faces, key=lambda f: f.area) if faces else None
            ponto_cru = alvo_face.centro_olhos if alvo_face else None

            frame_idx += 1
            agora = time.time()
            dt_frame = min(agora - t_prev, 0.1)   # dt real do loop (p/ a altura)
            t_prev = agora
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

            lim = np.radians(limite_deg)               # envelope do PAN (amplo, via base)
            lim_tilt = np.radians(LIMITE_TILT_DEG)     # envelope do TILT (modesto)
            erro = None
            jump = None
            prev_pan, prev_tilt, prev_altura = base_pan, base_tilt, altura  # p/ DESFAZER

            # ---- SERVO VISUAL → mira (pan, tilt) ----
            # Durante um head-tilt (gesto_t0) o pan/tilt CONGELA: a câmera rola sem
            # perseguir o alvo (que sai do lugar por causa do roll na imagem).
            if (fase == "seguir" and tracking and not fault_ativo and gesto_t0 is None
                    and ponto_cru is not None and prev is not None):
                dx, dy = prev[0] - cx, prev[1] - cy
                erro = (dx, dy)
                if abs(dx) >= deadzone_px:
                    des = sinal_pan * kp_servo * dx * radpx_x
                    base_pan = np.clip(base_pan + np.clip(des, -max_step, max_step), -lim, lim)
                if abs(dy) >= deadzone_px:
                    des = sinal_tilt * kp_servo * dy * radpx_y
                    base_tilt = np.clip(base_tilt + np.clip(des, -max_step, max_step),
                                        -lim_tilt, lim_tilt)
                # ALTURA (cascata lenta): se o tilt PERSISTE além da zona, sobe/desce a
                # câmera devagar pra re-nivelar o olhar (você na altura dos olhos).
                if altura_on and abs(base_tilt) > np.radians(ALTURA_ZONA_DEG):
                    altura += ALTURA_GANHO * sinal_altura * base_tilt * dt_frame
                    altura = float(np.clip(altura, -alt_max, alt_max))

            # ---- FUGA/perseguição: se o rosto SUMIU, vai pro lado que você saiu e
            # espera; se está presente, devolve a mira sem mexer (o servo comanda).
            # (Não persegue durante um head-tilt — o roll some com o rosto na imagem.) ----
            if (fase == "seguir" and tracking and not fault_ativo and perseguicao_on
                    and gesto_t0 is None):
                base_pan, base_tilt, auto_estado, auto_seta = autonomia.update(
                    ponto_cru, prev, cx, cy, base_pan, base_tilt,
                    sinal_pan, sinal_tilt, radpx_x, radpx_y, lim, lim_tilt)
            else:
                auto_estado = "seguindo"

            # ---- HEAD-TILT: roll no eixo óptico (não-bloqueante) ----
            roll = 0.0
            if gesto_t0 is not None:
                p_ht = perfil_head_tilt(time.time() - gesto_t0)
                if p_ht is None:
                    gesto_t0 = None
                else:
                    roll = gesto_dir * np.radians(HEAD_TILT_DEG) * p_ht

            # ---- PESCOÇO (cascata com "desenrolar") ----
            # O PUNHO (joint5) faz o pan RÁPIDO; a BASE (joint1) "desenrola" o punho de
            # volta ao 0 e assume o giro (a cabeca endireita, o corpo compensa). Se o
            # punho satura (±neck_max), a base assume o excedente NA HORA. Camera pan =
            # pan_base + neck = base_pan sempre (rosto fica centralizado o tempo todo).
            pan_base += neck_relax * (base_pan - pan_base) * dt_frame   # base assume devagar
            neck = float(np.clip(base_pan - pan_base, -np.radians(neck_max), np.radians(neck_max)))
            pan_base = base_pan - neck     # punho saturou → base assume o resto na hora
            base_ik = pan_base

            # ---- IK: mira (base_ik via base) → 6 juntas; o punho entra POR CIMA ----
            if fase == "seguir" and not fault_ativo:
                q_ik, ok, ik_iters, ik_ms = resolver_ik(geom, base_ik, base_tilt,
                                                        q_ref, altura, roll)
                if ok:
                    jump = float(np.degrees(np.max(np.abs(q_ik - q_ref))))
                if ok and jump <= IK_FLIP_DEG:
                    np.clip(q_ik, _LO + margem, _HI - margem, out=q_ik)
                    q_ref[:] = q_ik                          # seed limpo p/ o próximo frame
                    q_ik[PESCOCO_J5] = float(np.clip(        # pescoço: offset no punho
                        q_ik[PESCOCO_J5] + sinal_neck * neck,
                        _LO[PESCOCO_J5] + margem, _HI[PESCOCO_J5] - margem))
                    est["q_target"][:] = q_ik                # comando = IK + pescoço
                    ik_ok = True
                    n_falhas = 0
                else:
                    ik_ok = False
                    n_falhas += 1
                    if n_falhas <= FALHAS_MAX:    # falha isolada → SEGURA (desfaz o passo)
                        base_pan, base_tilt, altura = prev_pan, prev_tilt, prev_altura
                    else:                          # PRESO → recua a mira/altura e caminha
                        base_pan, base_tilt = prev_pan * 0.9, prev_tilt * 0.9   # rumo ao centro
                        altura = prev_altura * 0.9
                        pan_base *= 0.9
                        q2, ok2, _, _ = resolver_ik(geom, pan_base, base_tilt,
                                                    est["home"], altura)
                        if ok2:
                            np.clip(q2, _LO + margem, _HI - margem, out=q2)
                            passo = np.radians(3.0)
                            q_ref[:] += np.clip(q2 - q_ref, -passo, passo)
                            est["q_target"][:] = q_ref
                            if np.degrees(np.max(np.abs(q2 - q_ref))) < 2.0:
                                n_falhas = 0       # destravou
                        diario.evento("ik_destravando",
                                      pan=round(float(np.degrees(base_pan)), 1),
                                      tilt=round(float(np.degrees(base_tilt)), 1),
                                      altura=round(altura, 3))

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
                    ("== CAMERA FOLLOW — IK (modular) ==", COR_TIT),
                    modo,
                    cal,
                    (f"mira: pan={np.degrees(base_pan):+.1f} tilt={np.degrees(base_tilt):+.1f}deg  "
                     f"altura(h)={altura*100:+.0f}/{alt_max*100:.0f}cm (v/b)"
                     f"{'' if altura_on else ' OFF'}", COR_VAL),
                    (ik_txt, COR_OK if ik_ok else COR_AVISO),
                    (f"erro: {('%+d,%+d px' % erro) if erro else '--'}", COR_TXT),
                    (f"ganho[/]={kp_servo:.2f}  zona(o/p)={deadzone_px}px  "
                     f"limite(-/=)={limite_deg:.0f}deg  prev(,/.)={previsao_ms:.0f}ms",
                     COR_VAL),
                    (f"sinais x/y: {sinal_pan:+d}/{sinal_tilt:+d}   "
                     f"ESCALA: {np.radians(1)/radpx_x:.1f} / {np.radians(1)/radpx_y:.1f} px/deg",
                     COR_VAL),
                    (f"fuga(u): {'ON ' if perseguicao_on else 'off'} estado: {auto_estado.upper()}"
                     + ((' -->' if auto_seta > 0 else ' <--') if auto_estado != "seguindo" else ""),
                     COR_AVISO if auto_estado == "perseguindo" else COR_DIM),
                    (f"pescoco(9/0): +/-{neck_max:.0f}deg  punho={np.degrees(neck):+.0f}  "
                     f"base={np.degrees(base_ik):+.0f}  sinal(j)={sinal_neck:+d}  "
                     f"desenrolar(w/e)={neck_relax:.1f}", COR_VAL),
                    ("ESPACO encara | k calibra | n salva | t pausa | u fuga | g head-tilt | "
                     "9/0 pescoco | w/e desenrola | j sinal | h altura | v/b alcance | o/p zona | "
                     "-/= envelope | x/y sinais | f flutua | ESC", COR_DIM),
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
            elif k == ord("c"):              # recentra o olhar na home (e zera a altura)
                base_pan = base_tilt = 0.0
                altura = 0.0
            elif k == ord("h"):              # liga/desliga o acompanhamento de altura
                altura_on = not altura_on
                aviso("Altura ON (segue voce p/ cima/baixo)" if altura_on
                      else "Altura OFF (so inclina)", COR_OK if altura_on else COR_DIM)
            elif k == ord("b"):              # + alcance de altura
                alt_max = min(0.30, round(alt_max + 0.02, 2))
                aviso(f"Alcance de altura: +/-{alt_max*100:.0f} cm", COR_VAL)
            elif k == ord("v"):              # - alcance de altura
                alt_max = max(0.0, round(alt_max - 0.02, 2))
                altura = float(np.clip(altura, -alt_max, alt_max))
                aviso(f"Alcance de altura: +/-{alt_max*100:.0f} cm", COR_VAL)
            elif k == ord("u"):              # fuga/perseguição on/off
                perseguicao_on = not perseguicao_on
                autonomia.reset()
                aviso("Perseguicao ON (vai pro lado que voce sumiu)" if perseguicao_on
                      else "Perseguicao OFF", COR_OK if perseguicao_on else COR_DIM)
            elif k == ord("g"):              # head-tilt (inclina a cabeca, alterna o lado)
                if fase == "seguir":
                    gesto_t0 = time.time()
                    gesto_dir = -gesto_dir
            elif k == ord("0"):              # + faixa do pescoço (punho)
                neck_max = min(40.0, round(neck_max + 2.0, 1))
                aviso(f"Pescoco (punho) ate +/-{neck_max:.0f} deg", COR_VAL)
            elif k == ord("9"):              # - faixa do pescoço
                neck_max = max(0.0, round(neck_max - 2.0, 1))
                aviso(f"Pescoco (punho) ate +/-{neck_max:.0f} deg", COR_VAL)
            elif k == ord("j"):              # flip do sinal do pescoço (se virar errado)
                sinal_neck = -sinal_neck
                aviso(f"Sinal do pescoco (punho): {sinal_neck:+d}", COR_VAL)
            elif k == ord("e"):              # + rápido o desenrolar (cabeça endireita logo)
                neck_relax = min(3.0, round(neck_relax + 0.1, 1))
                aviso(f"Desenrolar do pescoco: {neck_relax:.1f}/s", COR_VAL)
            elif k == ord("w"):              # - rápido o desenrolar (mais devagar)
                neck_relax = max(0.0, round(neck_relax - 0.1, 1))
                aviso(f"Desenrolar do pescoco: {neck_relax:.1f}/s", COR_VAL)
            elif k == ord("n"):              # salva config (acorda sozinho na proxima)
                if calibrado and est.get("home") is not None:
                    salvar_config_ik(est["repouso"], est["home"], sinal_pan, sinal_tilt,
                                     radpx_x, radpx_y, kp_servo, deadzone_px,
                                     limite_deg, previsao_ms, alt_max, altura_on,
                                     neck_max, sinal_neck, neck_relax)
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
