#!/usr/bin/env python3
"""
Fase 5 - Rastreamento final + erro em GRAUS (o que o braço precisa girar).

Junta tudo: detecção (YuNet) -> rastreador (One Euro + Kalman) -> ALVO previsto.
Depois calcula o ERRO entre o alvo e o centro da imagem, em pixels E em graus.

Por que graus? A câmera está na garra. "Centralizar o rosto na imagem" é o
mesmo que "apontar a garra para o rosto". Quanto o rosto está fora do centro,
em graus, é EXATAMENTE quanto os motores pan/tilt precisam girar. Esse será
o 'setpoint' (alvo) do controle do braço na próxima fase.

A conversão usa o campo de visão (FOV) da câmera:
    graus_por_pixel = FOV / tamanho_da_imagem

Uso:
    python 05_rastreador.py --camera ASUS      # teste no laptop
    python 05_rastreador.py                     # câmera do pulso (C920)

Teclas: ESC/q = sair
"""

import argparse
import time

import cv2

import camera
from detector import DetectorFaces
from rastreador import RastreadorAlvo

CAMERA_PULSO = "C920"

# Campo de visão da câmera, em graus. Valores da Logitech C920.
# (Para outra câmera os números mudam, mas o conceito é o mesmo.)
FOV_H = 70.0   # horizontal
FOV_V = 43.0   # vertical


def parse_args():
    p = argparse.ArgumentParser(description="Fase 5 - rastreamento + erro em graus")
    p.add_argument("--camera", default=CAMERA_PULSO)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--escala", type=float, default=0.5)
    p.add_argument("--previsao", type=float, default=60.0,
                   help="Horizonte de previsão em ms (compensa a latência). Padrão: 60")
    p.add_argument("--fov-h", type=float, default=FOV_H)
    p.add_argument("--fov-v", type=float, default=FOV_V)
    return p.parse_args()


def main():
    args = parse_args()
    detector = DetectorFaces()
    rastreador = RastreadorAlvo(t_pred_ms=args.previsao)
    cap, indice = camera.abrir_camera(args.camera, args.width, args.height, args.fps)
    print(f"Câmera '{args.camera}' no índice {indice}. ESC/q para sair.")

    fps_medido = 0.0
    t_anterior = time.time()

    janela = "Camera Follow - Fase 5 (rastreamento + erro em graus)"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler frame. Encerrando.")
            break

        t = time.time()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        faces = detector.detectar(frame, escala=args.escala)
        alvo_face = max(faces, key=lambda f: f.area) if faces else None
        ponto_cru = alvo_face.centro_olhos if alvo_face else None

        suavizado, previsto = rastreador.update(ponto_cru, t)

        # Quantos graus por pixel (aproximação linear, ótima perto do centro).
        graus_px_x = args.fov_h / w
        graus_px_y = args.fov_v / h

        erro_px = erro_graus = None
        if previsto is not None:
            dx = previsto[0] - cx          # + = alvo à direita do centro
            dy = previsto[1] - cy          # + = alvo abaixo do centro
            erro_px = (dx, dy)
            # pan segue x; tilt segue y. (sinais finais dependem da montagem
            # do motor - acertamos isso quando ligar o braço.)
            erro_graus = (dx * graus_px_x, dy * graus_px_y)

        # --- Desenhos ---
        # Mira central = para onde a garra aponta.
        cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)

        if alvo_face is not None:
            cv2.rectangle(frame, (alvo_face.x, alvo_face.y),
                          (alvo_face.x + alvo_face.w, alvo_face.y + alvo_face.h),
                          (80, 80, 80), 1)

        # Alvo previsto (o que a garra deve perseguir) e a "linha de correção".
        if previsto is not None:
            cv2.line(frame, (cx, cy), previsto, (0, 165, 255), 2)
            cv2.circle(frame, previsto, 7, (0, 0, 255), 2)
            cv2.circle(frame, previsto, 2, (0, 0, 255), -1)

        # FPS.
        dt = t - t_anterior
        t_anterior = t
        if dt > 0:
            fps_medido = 0.9 * fps_medido + 0.1 * (1.0 / dt)

        # --- Painel ---
        linhas = [f"{fps_medido:4.1f} FPS   previsao: {args.previsao:.0f} ms   alvo: {'SIM' if alvo_face else 'nao'}"]
        if erro_px is not None:
            linhas.append(f"erro:  {erro_px[0]:+5d} , {erro_px[1]:+5d} px")
            linhas.append(f"girar: PAN {erro_graus[0]:+6.1f} deg   TILT {-erro_graus[1]:+6.1f} deg")
        else:
            linhas.append("erro:  (sem alvo)")

        fonte = cv2.FONT_HERSHEY_SIMPLEX
        larg = max(cv2.getTextSize(s, fonte, 0.6, 1)[0][0] for s in linhas)
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, 8), (6 + larg + 14, 8 + 26 * len(linhas) + 6),
                      (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)
        y = 32
        for s in linhas:
            cv2.putText(frame, s, (12, y), fonte, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            y += 26

        cv2.imshow(janela, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado com limpeza.")


if __name__ == "__main__":
    main()
