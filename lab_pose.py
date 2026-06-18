#!/usr/bin/env python3
"""lab_pose.py — LAB 1 (CORPO): MediaPipe Pose Landmarker (33 pontos) em CPU.

Roda no venv dos labs:  .venv-labs/bin/python lab_pose.py

Desenha o esqueleto, mede fps/latência e deriva sinais de alto nível p/ o robô:
  • presença do corpo + nº de pontos confiáveis
  • PERTO/LONGE (largura dos ombros em px)
  • EM PÉ / SENTADO (heurística com quadril/joelho)
  • lado p/ onde o tronco está virado

Usa o Tasks API novo do MediaPipe (o legado mp.solutions saiu na 0.10.x). O modelo
.task baixa sozinho na 1ª vez (models_labs/, gitignored). Teclas: ESC sai.
"""

import time

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import camera
from lab_bench import Medidor, hud, baixar_modelo

CAMERA = "C920"
MODELO_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
              "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
CONEXOES = vision.PoseLandmarksConnections.POSE_LANDMARKS

NARIZ, OMB_E, OMB_D = 0, 11, 12
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
    return info


def main():
    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- câmera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem câmera pelo nome, tentando idx 0:", e)
        cap = cv2.VideoCapture(0)

    modelo = baixar_modelo(MODELO_URL, "pose_landmarker_lite.task")
    opt = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=modelo),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(opt)

    med = Medidor()
    janela = "Lab 1 - Pose (MediaPipe, CPU)  [ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, 1280, 760)
    t0 = time.time()

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

        linhas = [(f"FPS {med.fps():.0f}   pose {med.media_ms('pose'):.0f} ms   {w}x{h}",
                   (0, 255, 180))]
        if res.pose_landmarks:
            lms = res.pose_landmarks[0]
            desenha(frame, lms, w, h)
            info = analisa(lms, w, h)
            linhas += [
                (f"CORPO detectado  ({info['pontos']}/33 pts)", (120, 235, 120)),
                (f"postura: {info.get('postura', '?')}", (235, 225, 130)),
                (f"distancia: {info.get('dist', '?')}  (ombros {info.get('ombros_px', 0):.0f}px)",
                 (235, 225, 130)),
                (f"virado p/: {info.get('virado', '?')}", (235, 225, 130)),
            ]
        else:
            linhas.append(("nenhum corpo no quadro", (120, 120, 245)))

        hud(frame, linhas)
        cv2.imshow(janela, frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("--- resumo:", med.resumo(), "---")


if __name__ == "__main__":
    main()
