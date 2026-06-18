#!/usr/bin/env python3
"""lab_identidade.py — LAB 2 (IDENTIDADE + CONTAGEM de pessoas).

InsightFace (RetinaFace + ArcFace 512-d): detecta TODOS os rostos do quadro (conta gente),
dá idade/gênero, e RECONHECE quem você cadastrar (tecla 'c') por similaridade de cosseno.
Acorda + flutua o braço (lab_braco) p/ você enquadrar a câmera.

Rodar:  .venv-labs/bin/python lab_identidade.py            (CPU)
        .venv-labs/bin/python lab_identidade.py openvino   (NPU/iGPU; exige onnxruntime-openvino)

Teclas: c cadastra o maior rosto | x limpa cadastros | ESPACO trava | f flutua | ESC sai.
"""

import sys
import time

import numpy as np
import cv2
from insightface.app import FaceAnalysis

import camera
from lab_bench import Medidor, hud, Registro
from lab_braco import Braco

CAMERA = "C920"
LIMIAR = 0.35              # similaridade de cosseno p/ "mesma pessoa" (ArcFace)
DET_SIZE = (640, 640)
PROVIDERS = {"cpu": ["CPUExecutionProvider"],
             "openvino": ["OpenVINOExecutionProvider", "CPUExecutionProvider"]}


def main():
    prov = sys.argv[1].lower() if len(sys.argv) > 1 else "cpu"
    providers = PROVIDERS.get(prov, PROVIDERS["cpu"])

    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- câmera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem câmera pelo nome, tentando idx 0:", e)
        cap = cv2.VideoCapture(0)
    janela = "Lab 2 - Identidade (InsightFace)  [c cadastra | ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, 1280, 760)

    braco = Braco(cap, janela)
    braco.iniciar()

    print(f"--- carregando InsightFace (buffalo_l, {prov}) ---")
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    try:
        prov_real = app.models["detection"].session.get_providers()[0]
    except Exception:
        prov_real = providers[0]

    cadastrados = []          # [(nome, normed_embedding)]
    med = Medidor()
    reg = Registro("identidade")
    reg.linha(tipo="inicio", lab="identidade", modelo="buffalo_l", provider=prov_real)
    t_log = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            med.frame()
            with med.estagio("face"):
                faces = app.get(frame)             # InsightFace usa BGR (cv2) direto
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                       reverse=True)               # maior (mais perto) primeiro

            pessoas = []
            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int)
                nome, sim = "estranho", 0.0
                if cadastrados:
                    emb = f.normed_embedding
                    nm, s = max(((nm, float(np.dot(emb, e))) for nm, e in cadastrados),
                                key=lambda t: t[1])
                    if s > LIMIAR:
                        nome, sim = nm, s
                cor = (120, 235, 120) if nome != "estranho" else (120, 120, 245)
                sexo = getattr(f, "sex", None) or ("M" if int(getattr(f, "gender", 0)) == 1 else "F")
                idade = int(getattr(f, "age", -1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(frame, f"{nome} {sim:.2f}", (x1, max(y1 - 26, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{idade}a {sexo}", (x1, max(y1 - 8, 32)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv2.LINE_AA)
                pessoas.append({"nome": nome, "sim": round(sim, 2), "idade": idade, "sexo": sexo})

            linhas = [
                (f"FPS {med.fps():.0f}   face {med.media_ms('face'):.0f}ms   prov: {prov_real}",
                 (0, 255, 180)),
                (f"PESSOAS NO QUADRO: {len(faces)}    cadastrados: {len(cadastrados)}",
                 (235, 225, 130)),
                (f"{'FLUTUANDO' if braco.livre else 'TRAVADO'}   "
                 "c cadastra | x limpa | ESPACO trava | f flutua | ESC sai", (150, 150, 150)),
            ]
            hud(frame, linhas)
            cv2.imshow(janela, frame)

            agora = time.time()
            if agora - t_log > 0.5:
                reg.linha(fps=round(med.fps(), 1), face_ms=round(med.media_ms("face"), 1),
                          n_pessoas=len(faces), pessoas=pessoas)
                t_log = agora

            k = cv2.waitKey(1) & 0xFF
            if k == 27:                           # ESC
                break
            elif k == ord("c") and faces:         # cadastra o maior rosto
                cadastrados.append((f"P{len(cadastrados) + 1}", faces[0].normed_embedding.copy()))
                print(f"--- cadastrado P{len(cadastrados)} ---")
            elif k == ord("x"):                   # limpa cadastros
                cadastrados.clear()
                print("--- cadastros limpos ---")
            elif k == 32:                         # ESPACO trava
                braco.travar()
            elif k == ord("f"):                   # f flutua
                braco.flutuar()
    finally:
        braco.encerrar()
        cap.release()
        cv2.destroyAllWindows()
        reg.fim(resumo=med.resumo())
        print("--- resumo:", med.resumo(), "---")


if __name__ == "__main__":
    main()
