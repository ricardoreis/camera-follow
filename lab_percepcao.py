#!/usr/bin/env python3
"""lab_percepcao.py — TESTE do P0: a camada percepcao.py ao vivo.

Mostra O ALVO que a percepção publica (rosto → corpo), sem controlar o braço (ele só
FLUTUA p/ você enquadrar). Valida a fundação do tracking robusto: cubra o rosto / sente /
levante / vire de costas e veja o ALVO migrar p/ o CORPO (em vez de sumir).

Rodar (no venv da app):  .venv/bin/python lab_percepcao.py
Teclas: ESPACO trava · f flutua · ESC pousa+sai.
"""

import time

import numpy as np
import cv2

import camera
from percepcao import Percepcao
from lab_bench import Medidor, hud, Registro
from lab_braco import Braco

CAMERA = "C920"


def main():
    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- câmera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem câmera pelo nome, tentando idx 0:", e)
        cap = cv2.VideoCapture(0)
    janela = "P0 - Percepcao (ALVO rosto->corpo)  [ESPACO trava | f | ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, 1280, 760)

    braco = Braco(cap, janela)
    braco.iniciar()

    perc = Percepcao(com_corpo=True)
    med = Medidor()
    reg = Registro("percepcao")
    reg.linha(tipo="inicio", lab="percepcao")
    t0 = time.time(); t_log = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            med.frame()
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            with med.estagio("percep"):
                est = perc.processa(frame, int((time.time() - t0) * 1000))

            # corpo (pontos) — leve
            if est["_lms"] is not None:
                for p in est["_lms"]:
                    if p.visibility >= 0.5:
                        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 2, (0, 200, 120), -1)
            # mira central
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)
            # O ALVO (verde=rosto, laranja=corpo)
            alvo, fonte = est["alvo"], est["fonte"]
            if alvo is not None:
                cor = (120, 235, 120) if fonte == "rosto" else (0, 180, 255)
                cv2.line(frame, (cx, cy), alvo, cor, 2)
                cv2.circle(frame, alvo, 12, cor, 2)
                cv2.circle(frame, alvo, 3, cor, -1)

            linhas = [
                (f"FPS {med.fps():.0f}   percep {med.media_ms('percep'):.0f}ms   {w}x{h}",
                 (0, 255, 180)),
                (f"{'FLUTUANDO' if braco.livre else 'TRAVADO'}   ALVO: "
                 f"{(fonte or '-- (perdido -> fuga/varredura)').upper()}",
                 (120, 235, 120) if fonte == "rosto" else
                 (0, 180, 255) if fonte == "corpo" else (120, 120, 245)),
                (f"rostos: {est['n_rostos']}   corpo: {'sim' if est['tem_corpo'] else 'nao'}   "
                 f"postura: {est['postura'] or '?'}   dist: {est['dist'] or '?'}",
                 (235, 225, 130)),
                ("[ESPACO trava | f flutua | ESC pousa+sai]", (150, 150, 150)),
            ]
            hud(frame, linhas)
            cv2.imshow(janela, frame)

            agora = time.time()
            if agora - t_log > 0.5:
                reg.linha(fps=round(med.fps(), 1), percep_ms=round(med.media_ms("percep"), 1),
                          fonte=fonte, n_rostos=est["n_rostos"], tem_corpo=est["tem_corpo"],
                          postura=est["postura"], dist=est["dist"])
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
