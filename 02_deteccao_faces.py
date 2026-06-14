#!/usr/bin/env python3
"""
Fase 2 - Detecção de faces ao vivo (YuNet).

Objetivo: a cada frame, detectar rostos com o YuNet (CNN leve do OpenCV que
roda em CPU, sem GPU). Desenhamos:
  - a caixa de cada rosto,
  - os dois olhos (pontos),
  - o ponto entre os olhos (alvo que a garra vai perseguir na Fase 3),
  - FPS, tempo só da detecção (ms) e nº de rostos.

Uso:
    python 02_deteccao_faces.py                 # câmera do pulso (C920)
    python 02_deteccao_faces.py --escala 1.0    # detecta no frame inteiro (mais preciso, mais lento)
    python 02_deteccao_faces.py --escala 0.5    # detecta em metade do tamanho (mais rápido) [padrão]

Teclas:
    q ou ESC -> sair
"""

import argparse
import time

import cv2

import camera
from detector import DetectorFaces

CAMERA_PULSO = "C920"


def parse_args():
    p = argparse.ArgumentParser(description="Fase 2 - detecção de faces (YuNet)")
    p.add_argument("--camera", default=CAMERA_PULSO,
                   help=f"Nome ou índice da câmera. Padrão: {CAMERA_PULSO}")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--escala", type=float, default=0.5,
                   help="Reduz a imagem só para detectar (0.5 = metade). "
                        "Menor = mais rápido. Padrão: 0.5")
    return p.parse_args()


def main():
    args = parse_args()
    detector = DetectorFaces()
    cap, indice = camera.abrir_camera(args.camera, args.width, args.height, args.fps)
    print(f"Câmera '{args.camera}' no índice {indice}. YuNet em escala "
          f"{args.escala}. Pressione 'q' ou ESC para sair.")

    fps_medido = 0.0
    t_anterior = time.time()

    janela = "Camera Follow - Fase 2 (YuNet)"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler frame da câmera. Encerrando.")
            break

        # Mede só o tempo da detecção.
        t0 = time.perf_counter()
        faces = detector.detectar(frame, escala=args.escala)
        ms_deteccao = (time.perf_counter() - t0) * 1000.0

        # FPS real do loop.
        agora = time.time()
        dt = agora - t_anterior
        t_anterior = agora
        if dt > 0:
            fps_medido = 0.9 * fps_medido + 0.1 * (1.0 / dt)

        # Desenha cada rosto.
        for f in faces:
            # Caixa do rosto.
            cv2.rectangle(frame, (f.x, f.y), (f.x + f.w, f.y + f.h),
                          (0, 255, 0), 2)
            # Os dois olhos (pontos amarelos).
            cv2.circle(frame, f.olho_dir, 3, (0, 255, 255), -1)
            cv2.circle(frame, f.olho_esq, 3, (0, 255, 255), -1)
            # Ponto de mira entre os olhos (alvo) - círculo vermelho.
            cv2.circle(frame, f.centro_olhos, 5, (0, 0, 255), 2)
            # Confiança da detecção, acima da caixa.
            cv2.putText(frame, f"{f.score:.2f}", (f.x, f.y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Mira central (referência de "para onde a garra aponta").
        h_img, w_img = frame.shape[:2]
        cx, cy = w_img // 2, h_img // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 1)

        # Painel de informações.
        texto = f"{fps_medido:4.1f} FPS | deteccao: {ms_deteccao:4.1f} ms | rostos: {len(faces)}"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(janela, frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q") or tecla == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado com limpeza.")


if __name__ == "__main__":
    main()
