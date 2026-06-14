#!/usr/bin/env python3
"""
Fase 4 - Laboratório de suavização e predição.

Mostra, sobre o mesmo vídeo, TRÊS versões do ponto-alvo (centro dos olhos do
rosto mais próximo):

    CRU      (vermelho)  - saída direta do YuNet, com jitter.
    ONE EURO (verde)     - suavizado pelo One Euro Filter.
    KALMAN   (azul=pos, ciano=previsto) - suavizado + PREVISTO à frente.

Use as trilhas (rastros) e a métrica de "dispersão" para comparar. Ajuste os
parâmetros ao vivo para sentir o efeito de cada um.

TECLAS
  ESC / q ............ sair
  1 / 2 / 3 .......... liga/desliga CRU / ONE EURO / KALMAN
  h .................. mostra/esconde a ajuda
  r .................. zera as trilhas

  One Euro:  z / x = min_cutoff (-/+)     c / v = beta (-/+)
  Kalman:    n / b = horizonte previsão ms (+/-)   , / m = sigma_a (+/-)

DICA: para LER o jitter, fique com a cabeça PARADA e compare a "dispersão"
do CRU vs ONE EURO. Para sentir a PREDIÇÃO, aumente o horizonte (tecla 'n')
e mova a cabeça de um lado para o outro: o ponto ciano "anda na frente".
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

import camera
from detector import DetectorFaces
from filtros import OneEuroPonto, KalmanCV

CAMERA_PULSO = "C920"

# Cores (BGR).
COR_CRU = (60, 60, 255)      # vermelho
COR_EURO = (60, 255, 60)     # verde
COR_KAL = (255, 170, 0)      # azul
COR_PRED = (255, 255, 0)     # ciano
COR_MIRA = (255, 255, 255)   # branco


def parse_args():
    p = argparse.ArgumentParser(description="Fase 4 - suavização e predição")
    p.add_argument("--camera", default=CAMERA_PULSO)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--escala", type=float, default=0.5)
    p.add_argument("--trilha", type=int, default=40,
                   help="Quantos pontos guardar em cada trilha. Padrão: 40")
    return p.parse_args()


def dispersao(pontos):
    """Distância média dos pontos ao seu centro. Com a cabeça parada, isso é
    justamente o JITTER. (Com a cabeça em movimento, mistura jitter+movimento.)"""
    if len(pontos) < 2:
        return 0.0
    arr = np.array(pontos, dtype=float)
    centro = arr.mean(axis=0)
    return float(np.linalg.norm(arr - centro, axis=1).mean())


def tremor(pontos, n=10):
    """Salto médio (px) de um frame para o outro, nos últimos n pontos.

    Com a cabeça PARADA, isso é exatamente o jitter (quanto o ponto 'pula'
    por frame). Recupera rápido quando você para (só olha ~0,4 s), então é
    bem menos contaminado pelo movimento do que a 'dispersao'."""
    pts = list(pontos)[-n:]
    if len(pts) < 2:
        return 0.0
    saltos = [np.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
              for i in range(1, len(pts))]
    return float(np.mean(saltos))


def desenhar_trilha(frame, pontos, cor):
    if len(pontos) >= 2:
        cv2.polylines(frame, [np.array(pontos, dtype=np.int32)],
                      False, cor, 1, cv2.LINE_AA)


def desenhar_osciloscopio(frame, buf_cru, buf_euro, buf_pred,
                          ver_cru, ver_euro, ver_kal, altura=170):
    """Plota a posição X de cada ponto ao longo do tempo, com ZOOM AUTOMÁTICO.

    O zoom automático é o segredo: a faixa vertical sempre se ajusta ao
    min/max recente. Parado, ele 'amplia' o jitter de 3px até ocupar a tela
    toda (vermelho espinhento vs verde liso). Movendo, ele mostra a onda do
    movimento (verde atrasado à direita, ciano adiantado à esquerda)."""
    h, w = frame.shape[:2]
    y_top = h - altura

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y_top), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)
    cv2.line(frame, (0, y_top), (w, y_top), (130, 130, 130), 1)

    # Autoescala vertical usando só as séries ligadas.
    buffers = []
    if ver_cru:
        buffers.append((buf_cru, COR_CRU))
    if ver_euro:
        buffers.append((buf_euro, COR_EURO))
    if ver_kal:
        buffers.append((buf_pred, COR_PRED))
    valores = [v for buf, _ in buffers for v in buf if v is not None]
    if len(valores) < 2:
        return
    vmin, vmax = min(valores), max(valores)
    if vmax - vmin < 2:                 # quase parado: evita divisão por ~0
        meio = (vmin + vmax) / 2
        vmin, vmax = meio - 1, meio + 1
    pad = (vmax - vmin) * 0.12
    vmin, vmax = vmin - pad, vmax + pad

    for buf, cor in buffers:
        cap = buf.maxlen
        pts = []
        for i, v in enumerate(buf):
            if v is None:
                continue
            x = int(i / (cap - 1) * (w - 1))
            y = int(y_top + (vmax - v) / (vmax - vmin) * (altura - 1))
            pts.append((x, y))
        if len(pts) >= 2:
            cv2.polylines(frame, [np.array(pts, np.int32)], False,
                          cor, 1, cv2.LINE_AA)

    cv2.putText(frame, "osciloscopio: posicao X no tempo (zoom auto) | tecla o",
                (10, y_top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (210, 210, 210), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    detector = DetectorFaces()
    cap, indice = camera.abrir_camera(args.camera, args.width, args.height, args.fps)
    print(f"Câmera '{args.camera}' no índice {indice}. ESC/q para sair, h para ajuda.")

    # Filtros.
    euro = OneEuroPonto(min_cutoff=1.0, beta=0.01)
    kalman = KalmanCV(sigma_a=100.0, sigma_z=3.0)

    # Parâmetros ajustáveis ao vivo.
    min_cutoff, beta = 1.0, 0.01
    sigma_a = 100.0
    t_pred_ms = 0.0   # horizonte de predição (comece em 0 e aumente com 'n')

    # Visibilidade e estado de UI.
    mostra = {"cru": True, "euro": True, "kalman": True}
    mostra_ajuda = True

    # Trilhas.
    tr_cru = deque(maxlen=args.trilha)
    tr_euro = deque(maxlen=args.trilha)
    tr_pred = deque(maxlen=args.trilha)

    # Buffers do osciloscópio (posição X ao longo do tempo, ~7s a 24fps).
    OSC_N = 180
    osc_cru = deque(maxlen=OSC_N)
    osc_euro = deque(maxlen=OSC_N)
    osc_pred = deque(maxlen=OSC_N)
    mostra_osc = True

    fps_medido = 0.0
    t_anterior = time.time()

    janela = "Camera Follow - Fase 4 (suavizacao e predicao)"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Falha ao ler frame. Encerrando.")
            break

        t = time.time()
        faces = detector.detectar(frame, escala=args.escala)

        # Alvo = rosto MAIOR (o mais próximo). Aqui já entra um pouco da Fase 3.
        alvo = max(faces, key=lambda f: f.area) if faces else None

        p_cru = p_euro = None
        if alvo is not None:
            p_cru = alvo.centro_olhos
            euro.set_params(min_cutoff, beta)
            p_euro = euro(p_cru, t)
            kalman.sigma_a = sigma_a
            kalman.atualizar(p_cru, t)
            # Desenha a caixa do alvo, discreta.
            cv2.rectangle(frame, (alvo.x, alvo.y),
                          (alvo.x + alvo.w, alvo.y + alvo.h), (80, 80, 80), 1)
        else:
            kalman.coast(t)  # sem rosto: Kalman desliza sozinho

        p_kal = kalman.posicao()
        p_pred = kalman.prever(t_pred_ms / 1000.0)

        # Atualiza trilhas.
        if p_cru is not None:
            tr_cru.append(p_cru)
        if p_euro is not None:
            tr_euro.append(p_euro)
        if p_pred is not None:
            tr_pred.append(p_pred)

        # Osciloscópio: guarda a coordenada X (ou None quando não há ponto).
        osc_cru.append(p_cru[0] if p_cru is not None else None)
        osc_euro.append(p_euro[0] if p_euro is not None else None)
        osc_pred.append(p_pred[0] if p_pred is not None else None)

        # --- Desenhos ---
        if mostra["cru"]:
            desenhar_trilha(frame, tr_cru, COR_CRU)
            if p_cru is not None:
                cv2.circle(frame, p_cru, 4, COR_CRU, -1)
        if mostra["euro"]:
            desenhar_trilha(frame, tr_euro, COR_EURO)
            if p_euro is not None:
                cv2.circle(frame, p_euro, 5, COR_EURO, 2)
        if mostra["kalman"]:
            desenhar_trilha(frame, tr_pred, COR_PRED)
            if p_kal is not None:
                cv2.circle(frame, p_kal, 5, COR_KAL, 2)
            if p_pred is not None:
                cv2.circle(frame, p_pred, 6, COR_PRED, 2)
            # Linha posição -> previsto: mostra "o quanto miramos à frente".
            if p_kal is not None and p_pred is not None:
                cv2.line(frame, p_kal, p_pred, COR_PRED, 1)

        # Mira central (a "garra").
        h_img, w_img = frame.shape[:2]
        cx, cy = w_img // 2, h_img // 2
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), COR_MIRA, 1)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), COR_MIRA, 1)

        # FPS.
        dt = t - t_anterior
        t_anterior = t
        if dt > 0:
            fps_medido = 0.9 * fps_medido + 0.1 * (1.0 / dt)

        # Osciloscópio na base (a melhor forma de VER a diferença).
        if mostra_osc:
            desenhar_osciloscopio(frame, osc_cru, osc_euro, osc_pred,
                                  mostra["cru"], mostra["euro"], mostra["kalman"])

        # --- Painel ---
        linhas = [
            f"{fps_medido:4.1f} FPS   alvo: {'SIM' if alvo else 'nao'}",
            f"tremor/frame (parado)  CRU: {tremor(tr_cru):4.2f}px   EURO: {tremor(tr_euro):4.2f}px",
        ]
        if mostra_ajuda:
            linhas += [
                "",
                f"[1]CRU {'on ' if mostra['cru'] else 'off'}  "
                f"[2]EURO {'on ' if mostra['euro'] else 'off'}  "
                f"[3]KALMAN {'on ' if mostra['kalman'] else 'off'}",
                f"OneEuro: min_cutoff(z/x)={min_cutoff:.2f}  beta(c/v)={beta:.4f}",
                f"Kalman:  previsao(n/b)={t_pred_ms:.0f}ms  sigma_a(,/m)={sigma_a:.0f}",
                "o=osciloscopio   r=zera trilhas   h=ajuda   ESC/q=sair",
            ]
        # Fundo preto semi-transparente atrás do painel, para legibilidade.
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        escala_fonte, espessura, passo = 0.55, 1, 22
        larg_max = 0
        for ln in linhas:
            (tw, _), _ = cv2.getTextSize(ln, fonte, escala_fonte, espessura)
            larg_max = max(larg_max, tw)
        x0, y0 = 6, 8
        x1, y1 = x0 + larg_max + 12, 26 + (len(linhas) - 1) * passo + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

        # Texto em amarelo.
        y = 26
        for ln in linhas:
            cv2.putText(frame, ln, (10, y), fonte,
                        escala_fonte, (0, 255, 255), espessura, cv2.LINE_AA)
            y += passo

        cv2.imshow(janela, frame)

        # --- Teclado ---
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("1"):
            mostra["cru"] = not mostra["cru"]
        elif k == ord("2"):
            mostra["euro"] = not mostra["euro"]
        elif k == ord("3"):
            mostra["kalman"] = not mostra["kalman"]
        elif k == ord("h"):
            mostra_ajuda = not mostra_ajuda
        elif k == ord("o"):
            mostra_osc = not mostra_osc
        elif k == ord("r"):
            tr_cru.clear(); tr_euro.clear(); tr_pred.clear()
            osc_cru.clear(); osc_euro.clear(); osc_pred.clear()
        # One Euro
        elif k == ord("z"):
            min_cutoff = max(0.05, min_cutoff * 0.7)
        elif k == ord("x"):
            min_cutoff = min(30.0, min_cutoff * 1.4)
        elif k == ord("c"):
            beta = max(0.0, beta - 0.005)
        elif k == ord("v"):
            beta = beta + 0.005
        # Kalman
        elif k == ord("n"):
            t_pred_ms = min(300.0, t_pred_ms + 10.0)
        elif k == ord("b"):
            t_pred_ms = max(0.0, t_pred_ms - 10.0)
        elif k == ord(","):
            sigma_a = min(2000.0, sigma_a * 1.4)
        elif k == ord("m"):
            sigma_a = max(1.0, sigma_a * 0.7)

    cap.release()
    cv2.destroyAllWindows()
    print("Encerrado com limpeza.")


if __name__ == "__main__":
    main()
