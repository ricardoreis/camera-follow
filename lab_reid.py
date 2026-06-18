#!/usr/bin/env python3
"""lab_reid.py — TESTE do P1b: re-identificação (NÃO trocar de pessoa).

Trava em VOCÊ (tecla 't' = maior rosto) e mostra: você fica com a caixa VERDE "ALVO";
quem mais entrar fica CINZA. O objetivo é provar que, mesmo com outra pessoa em cena, o
ALVO continua em você. O braço só FLUTUA (não controla nada aqui).

Rodar (venv da app):  .venv/bin/python lab_reid.py [dispositivo]
  dispositivo: ovcpu (padrão) | cpu | gpu | npu
Teclas: t trava em você | d destrava | ESPACO trava braço | f flutua | ESC pousa+sai.
"""

import sys
import time

import numpy as np
import cv2

import camera
from reid import ReID
from lab_bench import Medidor, hud, Registro
from lab_braco import Braco

CAMERA = "C920"


def main():
    dispositivo = sys.argv[1].lower() if len(sys.argv) > 1 else "ovcpu"
    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- câmera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem câmera pelo nome, idx 0:", e)
        cap = cv2.VideoCapture(0)
    janela = "Lab re-ID (nao trocar de pessoa)  [t trava em voce | d destrava | ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, 1280, 760)

    braco = Braco(cap, janela)
    braco.iniciar()

    print(f"--- carregando re-ID (InsightFace, {dispositivo}) ---")
    reid = ReID(dispositivo=dispositivo)
    med = Medidor()
    reg = Registro("reid")
    reg.linha(tipo="inicio", lab="reid", dispositivo=dispositivo)
    t_log = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            med.frame()
            reid.submeter(frame)                       # entrega ao worker (não bloqueia)

            n_alvo = 0
            for bbox, is_alvo, sim in reid.faces:       # desenha o último resultado do worker
                x1, y1, x2, y2 = bbox
                cor = (120, 235, 120) if is_alvo else (140, 140, 140)
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2 if is_alvo else 1)
                rotulo = f"ALVO {sim:.2f}" if is_alvo else (f"{sim:.2f}" if reid.tem_alvo else "rosto")
                cv2.putText(frame, rotulo, (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv2.LINE_AA)
                n_alvo += int(is_alvo)

            estado = ("ALVO PRESENTE" if reid.alvo_presente else
                      "ALVO SUMIU" if reid.tem_alvo else "SEM ALVO (tecle t)")
            cor_e = ((120, 235, 120) if reid.alvo_presente else
                     (0, 180, 255) if reid.tem_alvo else (120, 120, 245))
            linhas = [
                (f"FPS {med.fps():.0f}   arcface {reid.ms:.0f}ms (worker)   rostos {len(reid.faces)}",
                 (0, 255, 180)),
                (f"{estado}   sim_alvo {reid.alvo_sim:.2f}", cor_e),
                ("t = trava em VOCE (maior rosto) | d destrava | "
                 f"{'FLUTUANDO' if braco.livre else 'TRAVADO'} | ESPACO | f | ESC sai",
                 (150, 150, 150)),
            ]
            hud(frame, linhas)
            cv2.imshow(janela, frame)

            agora = time.time()
            if agora - t_log > 0.5:
                reg.linha(fps=round(med.fps(), 1), arcface_ms=round(reid.ms, 1),
                          n_rostos=len(reid.faces), alvo_presente=reid.alvo_presente,
                          sim=round(reid.alvo_sim, 2))
                t_log = agora

            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == ord("t"):
                reid.travar(); print("--- travando no maior rosto (voce) ---")
            elif k == ord("d"):
                reid.destravar(); print("--- destravado ---")
            elif k == 32:
                braco.travar()
            elif k == ord("f"):
                braco.flutuar()
    finally:
        reid.encerrar()
        braco.encerrar()
        cap.release()
        cv2.destroyAllWindows()
        reg.fim(resumo=med.resumo())
        print("--- resumo:", med.resumo(), "---")


if __name__ == "__main__":
    main()
