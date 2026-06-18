#!/usr/bin/env python3
"""lab_pose.py — LAB 1 (CORPO): MediaPipe Pose Landmarker (33 pontos) em CPU.

Roda no venv dos labs:  .venv-labs/bin/python lab_pose.py

Ao iniciar, ACORDA o braço (sobe suave até a home salva) e deixa FLUTUANDO, pra você
posicionar o braço/câmera com as mãos e escolher o melhor enquadramento (igual ao
seguir_ik). Depois faz a detecção de pose ao vivo (esqueleto + fps + postura/distância).

Teclas: ESPAÇO trava (segura firme a pose atual) · f volta a flutuar · ESC pousa e sai.
"""

import json
import os
import sys
import time

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Caminho do repo de controle do braço (Seeed) — igual ao seguir_ik.
ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)
from reBotArm_control_py.actuator import RobotArm  # noqa: E402
from controle_braco import est, controlador, motor_pronto, KP, KD  # noqa: E402
import camera
from lab_bench import Medidor, hud, baixar_modelo, Registro

CAMERA = "C920"
MODELO_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
CONEXOES = vision.PoseLandmarksConnections.POSE_LANDMARKS
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_seguir_ik.json")
DUR_ACORDAR, DUR_REPOUSO = 3.0, 3.5

NARIZ, OMB_E, OMB_D = 0, 11, 12
OLHO_E, OLHO_D, OREL_E, OREL_D = 2, 5, 7, 8
QUAD_E, QUAD_D, JOE_E, JOE_D = 23, 24, 25, 26
VIS_MIN = 0.5


def vis(lms, i):
    return lms[i].visibility >= VIS_MIN


def desenha(frame, lms, w, h):
    pts = [(int(p.x * w), int(p.y * h)) for p in lms]
    for c in CONEXOES:
        if vis(lms, c.start) and vis(lms, c.end):
            cv2.line(frame, pts[c.start], pts[c.end], (0, 230, 120), 2)
    for i, p in enumerate(lms):
        if p.visibility >= VIS_MIN:
            cv2.circle(frame, pts[i], 3, (255, 200, 0), -1)


def analisa(lms, w, h):
    info = {"pontos": sum(1 for p in lms if p.visibility >= VIS_MIN)}
    if vis(lms, OMB_E) and vis(lms, OMB_D):
        larg = float(np.hypot((lms[OMB_E].x - lms[OMB_D].x) * w,
                              (lms[OMB_E].y - lms[OMB_D].y) * h))
        info["ombros_px"] = larg
        info["dist"] = "PERTO" if larg > w * 0.28 else "LONGE" if larg < w * 0.16 else "media"
        if vis(lms, NARIZ):
            cx = (lms[OMB_E].x + lms[OMB_D].x) / 2
            info["virado"] = ("esq" if lms[NARIZ].x < cx - 0.04
                              else "dir" if lms[NARIZ].x > cx + 0.04 else "frente")
    for quad, joe in ((QUAD_E, JOE_E), (QUAD_D, JOE_D)):
        if vis(lms, quad) and vis(lms, joe):
            info["postura"] = "EM PE" if (lms[joe].y - lms[quad].y) > 0.22 else "SENTADO"
            break
    else:
        info["postura"] = "? (pernas fora do quadro)"
    # ---- orientação da cabeça (APROX. — a precisa, em graus, vem do Face Landmarker) ----
    if all(vis(lms, i) for i in (OLHO_E, OLHO_D, OREL_E, OREL_D)):
        escala = abs(lms[OREL_E].x - lms[OREL_D].x) + 1e-3   # ~largura da cabeça (normaliza)
        eye_y = (lms[OLHO_E].y + lms[OLHO_D].y) / 2
        ear_y = (lms[OREL_E].y + lms[OREL_D].y) / 2
        pitch = (eye_y - ear_y) / escala        # olhos ABAIXO das orelhas → olhando p/ baixo
        info["olhar"] = "BAIXO" if pitch > 0.15 else "CIMA" if pitch < -0.18 else "frente"
        info["pitch"] = round(float(pitch), 2)
        # roll: inclinação da linha dos olhos (cabeça pro ombro)
        info["roll"] = round(float(np.degrees(np.arctan2(
            lms[OLHO_D].y - lms[OLHO_E].y, lms[OLHO_D].x - lms[OLHO_E].x))), 0)
    return info


def main():
    # ---- câmera ----
    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- câmera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem câmera pelo nome, tentando idx 0:", e)
        cap = cv2.VideoCapture(0)

    # ---- braço: conecta + loop MIT (SEMPRE MIT, como o seguir_ik) ----
    arm = RobotArm()
    arm.connect()
    pos0 = np.asarray(arm.get_positions(request=True), dtype=float)
    n = arm.num_joints
    est["kp_hold"] = np.array([j.kp for j in arm._joints], dtype=float)   # firme (fábrica)
    est["kd_hold"] = np.array([j.kd for j in arm._joints], dtype=float)
    est["q_target"] = pos0.copy()
    est["repouso"] = pos0.copy()      # pose inicial = repouso (comece "sentado")
    est["livre"] = False
    est["tracking"] = False
    arm.enable()
    if not motor_pronto(arm):
        print("!!! Braço sem comunicação (sem energia?). Saindo.")
        cap.release()
        return
    arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
    arm.start_control_loop(controlador)

    janela = "Lab 1 - Pose (MediaPipe)  [ESPACO trava | f flutua | ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, 1280, 760)

    def ramp(destino, msg, dur, segura=0.0):
        """Move o alvo até 'destino' em 'dur' s (smoothstep), torque firme; mostra frames."""
        est["livre"] = False
        est["tracking"] = False
        ini = est["q_target"].copy()
        destino = np.asarray(destino, dtype=float)
        t0 = time.time()
        while True:
            frac = (time.time() - t0) / dur
            if frac >= 1.0:
                break
            s = frac * frac * (3.0 - 2.0 * frac)
            est["q_target"][:] = ini + (destino - ini) * s
            ok, fr = cap.read()
            if ok:
                cv2.putText(fr, msg, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(janela, fr)
                cv2.waitKey(1)
        est["q_target"][:] = destino
        t1 = time.time()
        while time.time() - t1 < segura:
            ok, fr = cap.read()
            if ok:
                cv2.putText(fr, "ASSENTANDO...", (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(janela, fr); cv2.waitKey(1)

    def flutuar():
        est["tracking"] = False
        est["q_target"] = np.asarray(arm.get_positions(), dtype=float).copy()
        est["integral"] = None
        est["livre"] = True

    def travar():
        est["q_target"] = np.asarray(arm.get_positions(), dtype=float).copy()
        est["integral"] = None
        est["livre"] = False

    # ---- ACORDA: sobe suave até a home salva (ou fica na pose atual se não houver) ----
    home = pos0
    if os.path.exists(CONFIG_PATH):
        try:
            home = np.asarray(json.load(open(CONFIG_PATH))["home"], dtype=float)
        except Exception:
            pass
    ramp(home, "ACORDANDO...", DUR_ACORDAR)
    flutuar()
    print("--- FLUTUANDO: posicione o braço/câmera com as mãos. "
          "ESPACO trava | f flutua | ESC sai ---")

    # ---- modelo de pose ----
    modelo = baixar_modelo(MODELO_URL, "pose_landmarker_lite.task")
    opt = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=modelo),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(opt)

    med = Medidor()
    reg = Registro("pose")
    reg.linha(tipo="inicio", lab="pose", camera=CAMERA, modelo=os.path.basename(modelo))
    t0 = time.time()
    t_log = 0.0          # throttle do snapshot periódico
    sig_prev = None      # p/ logar transições (corpo/postura/distância)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            med.frame()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with med.estagio("pose"):
                res = landmarker.detect_for_video(mp_img, int((time.time() - t0) * 1000))

            modo = "FLUTUANDO (posicione com a mao)" if est["livre"] else "TRAVADO"
            linhas = [(f"FPS {med.fps():.0f}   pose {med.media_ms('pose'):.0f} ms   {w}x{h}",
                       (0, 255, 180)),
                      (f"{modo}   [ESPACO trava | f flutua | ESC pousa+sai]",
                       (120, 200, 255) if est["livre"] else (120, 235, 120))]
            info = {}
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                desenha(frame, lms, w, h)
                info = analisa(lms, w, h)
                linhas += [
                    (f"CORPO detectado  ({info['pontos']}/33 pts)", (120, 235, 120)),
                    (f"postura: {info.get('postura', '?')}", (235, 225, 130)),
                    (f"distancia: {info.get('dist', '?')}  (ombros {info.get('ombros_px', 0):.0f}px)",
                     (235, 225, 130)),
                    (f"olhar: {info.get('olhar', '?')}  virado: {info.get('virado', '?')}  "
                     f"roll: {info.get('roll', 0):.0f}deg", (235, 225, 130)),
                ]
            else:
                linhas.append(("nenhum corpo no quadro", (120, 120, 245)))

            # ---- log JSONL: snapshot a ~2 Hz + sempre que muda um sinal-chave ----
            corpo = bool(res.pose_landmarks)
            sig = (corpo, info.get("postura"), info.get("dist"),
                   info.get("olhar"), info.get("virado"))
            agora = time.time()
            if sig != sig_prev or agora - t_log > 0.5:
                reg.linha(fps=round(med.fps(), 1), pose_ms=round(med.media_ms("pose"), 1),
                          corpo=corpo, livre=bool(est["livre"]), **info)
                t_log, sig_prev = agora, sig

            hud(frame, linhas)
            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:                       # ESC: pousa e sai
                break
            elif k == 32:                     # ESPACO: trava (segura firme)
                travar()
            elif k == ord("f"):               # f: volta a flutuar
                flutuar()
    finally:
        print("--- pousando no repouso (APOIE o braço) ---")
        try:
            ramp(est["repouso"], "RETORNANDO AO REPOUSO...", DUR_REPOUSO, segura=2.0)
        except Exception:
            pass
        for fn in (arm.stop_control_loop, arm.disable, arm.disconnect):
            try:
                fn()
            except Exception:
                pass
        cap.release()
        cv2.destroyAllWindows()
        reg.fim(resumo=med.resumo())
        print("--- resumo:", med.resumo(), "---")


if __name__ == "__main__":
    main()
