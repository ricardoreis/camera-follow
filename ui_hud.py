#!/usr/bin/env python3
"""ui_hud.py — helpers de interface (HUD): cores, painel translúcido, toast e a
tela amigável quando não há comunicação com o braço. Extraído do 10_seguir_ik."""

import numpy as np
import cv2

COR_TXT = (210, 210, 210)
COR_TIT = (0, 215, 255)
COR_OK = (100, 235, 140)
COR_AVISO = (60, 175, 255)
COR_ERRO = (80, 80, 245)
COR_VAL = (235, 225, 130)
COR_DIM = (150, 150, 150)
TOAST_DUR = 3.0


def painel(frame, x0, y0, linhas, escala=0.5, alpha=0.66):
    """Painel translúcido com linhas coloridas. 'linhas' = lista de str ou (str, cor)."""
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    th = int(round(28 * (escala / 0.5)))
    larg = max(cv2.getTextSize(it[0] if isinstance(it, (tuple, list)) else it,
                               fonte, escala, 1)[0][0] for it in linhas)
    x1, y1 = x0 + larg + 24, y0 + th * len(linhas) + 14
    ovl = frame.copy()
    cv2.rectangle(ovl, (x0, y0), (x1, y1), (18, 18, 18), -1)
    cv2.addWeighted(ovl, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (85, 85, 85), 1)
    y = y0 + th
    for it in linhas:
        txt, cor = it if isinstance(it, (tuple, list)) else (it, COR_TXT)
        cv2.putText(frame, txt, (x0 + 12, y), fonte, escala, cor, 1, cv2.LINE_AA)
        y += th
    return x1, y1


def desenha_toast(frame, texto, cor):
    """Aviso grande, centralizado no topo, com borda colorida pelo tipo."""
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    (wt, ht), _ = cv2.getTextSize(texto, fonte, 0.75, 2)
    w = frame.shape[1]
    x0, x1 = (w - wt) // 2 - 20, (w + wt) // 2 + 20
    ovl = frame.copy()
    cv2.rectangle(ovl, (x0, 10), (x1, 36 + ht), (15, 15, 15), -1)
    cv2.addWeighted(ovl, 0.78, frame, 0.22, 0, dst=frame)
    cv2.rectangle(frame, (x0, 10), (x1, 36 + ht), cor, 2)
    cv2.putText(frame, texto, (x0 + 20, 20 + ht), fonte, 0.75, cor, 2, cv2.LINE_AA)


def tela_sem_braco():
    """Janela amigável quando não há comunicação com o braço."""
    img = np.full((300, 940, 3), 25, np.uint8)
    for i, (txt, cor) in enumerate([
            ("SEM COMUNICACAO COM O BRACO B601-DM", COR_ERRO),
            ("Verifique se esta LIGADO e CONECTADO (USB / MotorBridge).", COR_TXT),
            ("Ligue/conecte e rode de novo. (tecle algo p/ sair)", COR_DIM)]):
        cv2.putText(img, txt, (28, 70 + i * 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, cor, 2, cv2.LINE_AA)
    cv2.imshow("Camera Follow IK", img)
    cv2.waitKey(8000)
    cv2.destroyAllWindows()
