#!/usr/bin/env python3
"""lab_emocao.py — LAB 3 (EXPRESSÃO + EMOÇÃO + pose da cabeça).

Dois modelos:
  • MediaPipe FACE LANDMARKER (478 pontos = Face Mesh) → blendshapes (52 coeficientes:
    sorriso/boca aberta/sobrancelha/piscar) e a MATRIZ de pose da cabeça (pitch/yaw/roll
    em graus, a versão PRECISA do "olhando p/ cima/baixo"). Real-time em CPU.
  • HSEmotion (ONNX) → emoção ROTULADA (Anger/Happiness/Surprise/Neutral/…), ~12ms.

Acorda + flutua o braço (lab_braco) p/ enquadrar. Teclas: ESPACO trava · f flutua · ESC sai.
Rodar:  .venv-labs/bin/python lab_emocao.py
"""

import urllib.request  # noqa: F401  (corrige o downloader do hsemotion-onnx)
import time

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

import camera
from lab_bench import Medidor, hud, baixar_modelo, Registro
from lab_braco import Braco

CAMERA = "C920"
MODELO_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/latest/face_landmarker.task")


def blend_dict(cat_list):
    return {c.category_name: c.score for c in cat_list}


def expressao(bs):
    """Deriva uma expressão dos blendshapes (0..1)."""
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    jaw = bs.get("jawOpen", 0)
    brow_up = bs.get("browInnerUp", 0)
    frown = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
    if smile > 0.4:
        return "SORRINDO :)", smile
    if jaw > 0.4 and brow_up > 0.3:
        return "SURPRESO :O", jaw
    if frown > 0.4:
        return "SERIO / BRAVO >:(", frown
    if blink > 0.6:
        return "olhos fechados", blink
    return "neutro", 0.0


def head_pose(m):
    """pitch/yaw/roll (graus) da matriz 4x4 de transformação facial do MediaPipe."""
    R = np.asarray(m)[:3, :3]
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    pitch = np.degrees(np.arctan2(-R[2, 0], sy))
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return float(pitch), float(yaw), float(roll)


def caixa(lms, w, h, margem=0.08):
    xs = [p.x for p in lms]; ys = [p.y for p in lms]
    x1 = int(max((min(xs) - margem) * w, 0)); x2 = int(min((max(xs) + margem) * w, w))
    y1 = int(max((min(ys) - margem) * h, 0)); y2 = int(min((max(ys) + margem) * h, h))
    return x1, y1, x2, y2


def main():
    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- câmera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem câmera pelo nome, tentando idx 0:", e)
        cap = cv2.VideoCapture(0)
    janela = "Lab 3 - Emocao/Expressao (FaceMesh + HSEmotion)  [ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, 1280, 760)

    braco = Braco(cap, janela)
    braco.iniciar()

    modelo = baixar_modelo(MODELO_URL, "face_landmarker.task")
    fl = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=modelo),
        running_mode=vision.RunningMode.VIDEO, num_faces=2,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    print("--- carregando HSEmotion (ONNX) ---")
    emo_rec = HSEmotionRecognizer(model_name="enet_b0_8_best_afew")

    med = Medidor()
    reg = Registro("emocao")
    reg.linha(tipo="inicio", lab="emocao", modelos=["face_landmarker", "hsemotion"])
    t0 = time.time(); t_log = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            med.frame()
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            with med.estagio("mesh"):
                r = fl.detect_for_video(mp_img, int((time.time() - t0) * 1000))

            linhas = [(f"FPS {med.fps():.0f}   mesh {med.media_ms('mesh'):.0f}ms   "
                       f"emo {med.media_ms('emo'):.0f}ms", (0, 255, 180)),
                      (f"{'FLUTUANDO' if braco.livre else 'TRAVADO'}   "
                       f"ROSTOS: {len(r.face_landmarks)}   [ESPACO trava | f | ESC sai]",
                       (150, 150, 150))]
            info = {}
            if r.face_landmarks:
                # maior rosto = índice de maior área
                areas = [(i, (lambda b: (b[2]-b[0])*(b[3]-b[1]))(caixa(lm, w, h)))
                         for i, lm in enumerate(r.face_landmarks)]
                big = max(areas, key=lambda t: t[1])[0]
                # desenha a malha (pontos) de todos; rótulos só do maior
                for lm in r.face_landmarks:
                    for p in lm:
                        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 220, 120), -1)
                lms = r.face_landmarks[big]
                x1, y1, x2, y2 = caixa(lms, w, h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (235, 225, 130), 2)

                bs = blend_dict(r.face_blendshapes[big]) if r.face_blendshapes else {}
                expr, conf = expressao(bs)
                pitch = yaw = roll = 0.0
                if r.facial_transformation_matrixes:
                    pitch, yaw, roll = head_pose(r.facial_transformation_matrixes[big])
                # HSEmotion no crop do maior rosto
                emo = "?"
                crop = rgb[y1:y2, x1:x2]
                if crop.size and crop.shape[0] > 20 and crop.shape[1] > 20:
                    with med.estagio("emo"):
                        emo, _ = emo_rec.predict_emotions(crop, logits=False)
                info = {"expr": expr, "emo": emo, "pitch": round(pitch, 0),
                        "yaw": round(yaw, 0), "roll": round(roll, 0),
                        "sorriso": round(float((bs.get("mouthSmileLeft", 0)+bs.get("mouthSmileRight", 0))/2), 2)}
                linhas += [
                    (f"EMOCAO (HSEmotion): {emo}", (120, 235, 120)),
                    (f"expressao (blendshapes): {expr}", (235, 225, 130)),
                    (f"cabeca: pitch {pitch:+.0f}  yaw {yaw:+.0f}  roll {roll:+.0f} (graus)",
                     (235, 225, 130)),
                ]
            else:
                linhas.append(("nenhum rosto no quadro", (120, 120, 245)))

            hud(frame, linhas)
            cv2.imshow(janela, frame)
            agora = time.time()
            if agora - t_log > 0.5:
                reg.linha(fps=round(med.fps(), 1), mesh_ms=round(med.media_ms("mesh"), 1),
                          emo_ms=round(med.media_ms("emo"), 1),
                          rostos=len(r.face_landmarks), **info)
                t_log = agora

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == 32:
                braco.travar()
            elif k == ord("f"):
                braco.flutuar()
    finally:
        braco.encerrar()
        cap.release()
        cv2.destroyAllWindows()
        reg.fim(resumo=med.resumo())
        print("--- resumo:", med.resumo(), "---")


if __name__ == "__main__":
    main()
