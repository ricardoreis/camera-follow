#!/usr/bin/env python3
"""
Fase 6b - Jog seguro de UMA junta + visão da câmera do pulso.

Primeiro movimento comandado do braço. Objetivo: confirmar que controlamos
uma junta com segurança e VER como o giro dela move a imagem da câmera
(descobrir qual junta = pan/tilt e em que direção/sinal).

SEGURANÇA (embutida):
  - Move UMA junta por vez; as outras ficam TRAVADAS na posição inicial.
  - Alvo limitado a +/- LIMITE_DEG da posição inicial (clamp). Não há como
    mandar o braço pra longe.
  - Velocidade baixa (VLIM) e passos pequenos (PASSO_DEG).
  - Ao sair (ESC/q, Ctrl+C ou erro): para o loop e DESLIGA o torque.

  >>> O braço FICA RÍGIDO segurando a posição enquanto roda, e fica MOLE ao
      sair (pode ceder à gravidade). Apoie o braço ao encerrar. <<<

TECLAS (na janela da câmera):
  1..6 ......... seleciona joint1..joint6 (ativa)
  a / d ........ gira a junta ativa  -  / +  (PASSO_DEG)
  c ............ recentra a junta ativa na posição inicial
  h ............ manda TODAS as juntas de volta ao início
  ESC / q ...... sair (desliga o torque)
"""

import sys
import time

import numpy as np
import cv2

# Pacote de controle do braço (não é instalado por pip; importamos pelo caminho).
ARM_REPO = "/home/ricardo-reis/GITHUB/reBotArm_control_py"
sys.path.insert(0, ARM_REPO)
from reBotArm_control_py.actuator import RobotArm  # noqa: E402

import camera  # noqa: E402

CAMERA_PULSO = "C920"

# --- Parâmetros de segurança (conservadores) ---
LIMITE_DEG = 20.0    # alvo nunca passa de +/- isto da posição inicial
VLIM = 0.3           # rad/s (~17 graus/s) - bem devagar
PASSO_DEG = 2.0      # quanto cada tecla a/d move


def main():
    # Estado compartilhado entre o loop de controle (500Hz) e o loop da câmera.
    estado = {"alvo": None}   # np.array de ângulos-alvo (rad)

    arm = RobotArm()
    arm.connect()
    print("--- conectado ---")

    n = arm.num_joints
    home = arm.get_positions().copy()       # posição inicial (rad) = referência
    estado["alvo"] = home.copy()
    vlim = np.full(n, VLIM)
    lim = np.radians(LIMITE_DEG)

    def controlador(ref, dt):
        # Roda a 500Hz: apenas persegue o alvo atual com velocidade limitada.
        ref.pos_vel(estado["alvo"], vlim=vlim)

    cap = None
    try:
        cap, indice = camera.abrir_camera(CAMERA_PULSO)
        print(f"--- câmera {CAMERA_PULSO} (idx {indice}) ---")

        arm.enable()
        arm.mode_pos_vel(vlim=vlim)
        arm.start_control_loop(controlador)
        print("--- braço RÍGIDO segurando posição. Janela da câmera ativa. ---")

        ativa = 4   # começa em joint5 (índice 4) = nosso palpite de PAN
        janela = "Camera Follow - Fase 6b (jog seguro)"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pos = arm.get_positions()         # posição real (rad)
            alvo = estado["alvo"]

            # Mira central (para onde a câmera aponta).
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)

            # Painel: estado de cada junta (delta em graus relativo ao início).
            linhas = [f"JUNTA ATIVA: joint{ativa + 1}   (limite +/-{LIMITE_DEG:.0f} deg)"]
            for i in range(n):
                d_alvo = np.degrees(alvo[i] - home[i])
                d_real = np.degrees(pos[i] - home[i])
                marca = ">>" if i == ativa else "  "
                linhas.append(f"{marca} joint{i+1}: alvo {d_alvo:+6.1f}  real {d_real:+6.1f} deg")
            linhas.append("1-6 seleciona | a/d gira -/+ | c centra | h home | ESC sai")

            fonte = cv2.FONT_HERSHEY_SIMPLEX
            larg = max(cv2.getTextSize(s, fonte, 0.6, 1)[0][0] for s in linhas)
            ov = frame.copy()
            cv2.rectangle(ov, (6, 8), (6 + larg + 14, 8 + 24 * len(linhas) + 6), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)
            y = 30
            for s in linhas:
                cor = (0, 255, 255)
                cv2.putText(frame, s, (12, y), fonte, 0.6, cor, 1, cv2.LINE_AA)
                y += 24

            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif ord("1") <= k <= ord("6"):
                ativa = k - ord("1")
            elif k == ord("a") or k == ord("d"):
                passo = np.radians(PASSO_DEG) * (1 if k == ord("d") else -1)
                novo = alvo[ativa] + passo
                # CLAMP: nunca além de +/- limite da posição inicial.
                novo = np.clip(novo, home[ativa] - lim, home[ativa] + lim)
                alvo[ativa] = novo
            elif k == ord("c"):
                alvo[ativa] = home[ativa]
            elif k == ord("h"):
                estado["alvo"] = home.copy()

    finally:
        print("\n--- encerrando: parando loop e DESLIGANDO torque (apoie o braço) ---")
        try:
            arm.stop_control_loop()
        except Exception:
            pass
        try:
            arm.disable()
        except Exception:
            pass
        try:
            arm.disconnect()
        except Exception:
            pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("--- encerrado ---")


if __name__ == "__main__":
    main()
