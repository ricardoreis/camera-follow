#!/usr/bin/env python3
"""
Fase 1 - Janela da webcam ao vivo.

Objetivo: abrir a câmera do pulso do braço, exibir o vídeo em tempo real,
mostrar o FPS medido e uma mira no centro (referência de "para onde a garra
está olhando", que usaremos nas próximas fases).

A câmera é identificada pelo NOME estável (ex.: "C920"), não pelo número
/dev/videoN — assim funciona mesmo se você trocar a câmera de porta USB.

Uso:
    python 01_webcam.py                 # usa a câmera do pulso (C920)
    python 01_webcam.py --camera C922   # usa outra câmera pelo nome
    python 01_webcam.py --camera 0      # usa pelo índice, se preferir
    python 01_webcam.py --listar        # só lista as câmeras e sai

Teclas durante a execução:
    q  ou  ESC   -> sair
"""

import argparse
import time

import cv2

import camera  # nosso módulo camera.py

# Nome da câmera montada no pulso do braço. Trocou a câmera? Mude só aqui.
CAMERA_PULSO = "C920"


def parse_args():
    p = argparse.ArgumentParser(description="Fase 1 - webcam ao vivo")
    p.add_argument("--camera", default=CAMERA_PULSO,
                   help=f"Nome (ex.: C920) ou índice da câmera. Padrão: {CAMERA_PULSO}")
    p.add_argument("--width", type=int, default=1280, help="Largura. Padrão: 1280")
    p.add_argument("--height", type=int, default=720, help="Altura. Padrão: 720")
    p.add_argument("--fps", type=int, default=30, help="FPS desejado. Padrão: 30")
    p.add_argument("--listar", action="store_true",
                   help="Lista as câmeras disponíveis e sai")
    return p.parse_args()


def main():
    args = parse_args()

    # Atalho útil: só listar as câmeras e sair.
    if args.listar:
        print("Câmeras detectadas (nome estável -> dispositivo atual):")
        for nome, destino in camera.listar_cameras():
            print(f"  {nome}  ->  {destino}")
        return

    # Abre a câmera pelo nome (o módulo resolve o índice atual sozinho).
    cap, indice = camera.abrir_camera(args.camera, args.width, args.height, args.fps)

    largura_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Câmera '{args.camera}' aberta no índice {indice} "
          f"em {largura_real}x{altura_real}. Pressione 'q' ou ESC para sair.")

    # Variáveis para medir o FPS real entregue pela câmera.
    fps_medido = 0.0
    t_anterior = time.time()

    janela = "Camera Follow - Fase 1 (webcam)"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler frame da câmera. Encerrando.")
            break

        # FPS real: tempo entre frames -> FPS instantâneo, suavizado por
        # média exponencial para não "tremer" o número na tela.
        agora = time.time()
        dt = agora - t_anterior
        t_anterior = agora
        if dt > 0:
            fps_medido = 0.9 * fps_medido + 0.1 * (1.0 / dt)

        # Desenhos sobre o frame.
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2  # centro = "para onde a garra aponta"
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 1)

        texto = f"{w}x{h}  |  {fps_medido:4.1f} FPS"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(janela, frame)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q") or tecla == 27:  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado com limpeza.")


if __name__ == "__main__":
    main()
