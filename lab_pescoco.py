#!/usr/bin/env python3
"""lab_pescoco.py — LAB GUIADO: validar o "pescoço" (pan pelo PUNHO vs pela BASE).

Pergunta a responder: o **punho** (joint5) consegue panear a câmera na horizontal,
suave e numa faixa útil? Se sim, dá pra fazer a cascata "pescoço": movimento pequeno
→ punho (base parada); maior → base entra (como o tilt→altura). Senão, repensamos.

É um TUTORIAL GUIADO: as instruções aparecem na tela, passo a passo; você executa o que
ele pede; tudo é gravado em logs_ik/pescoco_*.jsonl. Reaproveita os módulos (controle
MIT, HUD, log).

Como funciona a medição: o lab cutuca uma junta ±ângulo, mede pra ONDE o rosto se
desloca na imagem (px/grau) — se for sobretudo na HORIZONTAL (dfx), a junta PANEIA
(serve de pescoço); se for vertical (dfy) ou ~0, não serve.

PASSOS: 1) flutue e trave a home  2) testa o PUNHO  3) testa a BASE  4) resumo.
Teclas: [ / ] ajusta o ângulo de teste · m mede · ENTER/ESPACO próximo passo · ESC sai.
"""

import json
import os
import sys
import time

import numpy as np
import cv2

ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)
from reBotArm_control_py.actuator import RobotArm  # noqa: E402

from controle_braco import est, controlador, motor_pronto, KP, KD  # noqa: E402
from ui_hud import (  # noqa: E402
    COR_TXT, COR_TIT, COR_OK, COR_AVISO, COR_VAL, COR_DIM, painel, desenha_toast,
    tela_sem_braco,
)
from diario import Diario  # noqa: E402
import camera  # noqa: E402
from detector import DetectorFaces  # noqa: E402

CAMERA_PULSO = "C920"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_ik")
SETTLE_S = 0.7    # espera o braço chegar à cutucada
MEAS_S = 0.4      # janela de medição do rosto
PUNHO, BASE = 4, 0   # índices: joint5 (punho/pescoço), joint1 (base)


def medir_face(cap, detector, janela, dur, texto):
    """Média do (dx, dy) do rosto vs centro por 'dur' s. None se não houver rosto."""
    t0 = time.time(); xs = []; ys = []
    while time.time() - t0 < dur:
        ret, fr = cap.read()
        if not ret:
            continue
        hh, ww = fr.shape[:2]
        faces = detector.detectar(fr, escala=0.5)
        if faces:
            f = max(faces, key=lambda f: f.area)
            xs.append(f.centro_olhos[0] - ww // 2)
            ys.append(f.centro_olhos[1] - hh // 2)
        desenha_toast(fr, texto, COR_TIT)
        cv2.imshow(janela, fr); cv2.waitKey(1)
    return (float(np.mean(xs)), float(np.mean(ys))) if xs else None


def ir(cap, janela, home, eixo, off_deg, texto):
    """Comanda a junta 'eixo' = home + off_deg e espera assentar (mostrando o frame)."""
    qt = home.copy()
    qt[eixo] = home[eixo] + np.radians(off_deg)
    est["q_target"][:] = qt
    t0 = time.time()
    while time.time() - t0 < SETTLE_S:
        ret, fr = cap.read()
        if ret:
            desenha_toast(fr, texto, COR_TIT)
            cv2.imshow(janela, fr); cv2.waitKey(1)


def nudge_e_mede(cap, detector, janela, home, eixo, ang_deg, nome):
    """Cutuca a junta 'eixo' em ±ang e mede o deslocamento do rosto. Devolve dict."""
    txt = f"MEDINDO {nome} (±{ang_deg:.0f}deg)... fique PARADO, rosto visivel"
    ir(cap, janela, home, eixo, -ang_deg, txt); menos = medir_face(cap, detector, janela, MEAS_S, txt)
    ir(cap, janela, home, eixo, +ang_deg, txt); mais = medir_face(cap, detector, janela, MEAS_S, txt)
    ir(cap, janela, home, eixo, 0.0, "voltando...")
    if menos is None or mais is None:
        return None
    dfx = mais[0] - menos[0]          # deslocamento horizontal do rosto p/ +2*ang
    dfy = mais[1] - menos[1]          # vertical
    ppd_x = abs(dfx) / (2 * ang_deg)  # px/grau na horizontal (= quanto PANEIA)
    ppd_y = abs(dfy) / (2 * ang_deg)
    # PANEIA bem se o efeito é sobretudo horizontal (dfx >> dfy).
    paneia = abs(dfx) > 1.6 * abs(dfy) and ppd_x > 1.0
    return {"ang": ang_deg, "dfx": round(dfx, 1), "dfy": round(dfy, 1),
            "ppd_x": round(ppd_x, 2), "ppd_y": round(ppd_y, 2),
            "sinal": (-1 if dfx > 0 else 1), "paneia": bool(paneia)}


# Instruções de cada passo (lista de (texto, cor)).
def passo_texto(passo, ang, res_punho, res_base):
    def r(d):
        if d is None:
            return "(ainda nao medido — tecle 'm')"
        return (f"px/grau  HORIZ(pan)={d['ppd_x']:.1f}  vert={d['ppd_y']:.1f}  "
                f"sinal={d['sinal']:+d}  -> {'PANEIA :)' if d['paneia'] else 'nao paneia'}")
    if passo == 1:
        return [("LAB PESCOCO  —  PASSO 1/4: POSICIONAR", COR_TIT),
                ("", COR_TXT),
                ("Flutue o braco com a mao ate ele TE ENCARAR,", COR_TXT),
                ("com seu rosto BEM NO CENTRO da imagem.", COR_TXT),
                ("", COR_TXT),
                (">>> Tecle ESPACO para TRAVAR a home e comecar <<<", COR_AVISO)]
    if passo == 2:
        return [("PASSO 2/4: PAN pelo PUNHO (joint5 = 'pescoco')", COR_TIT),
                ("", COR_TXT),
                ("Fique PARADO e centralizado, rosto visivel.", COR_TXT),
                (f"angulo de teste: {ang:.0f} graus   ([ diminui  ] aumenta)", COR_VAL),
                ("Tecle 'm': o braco cutuca SO o punho +/- esse angulo", COR_TXT),
                ("e mede pra onde a camera vira (horizontal = paneia).", COR_DIM),
                ("", COR_TXT),
                ("resultado PUNHO:  " + r(res_punho), COR_OK if res_punho and res_punho["paneia"] else COR_TXT),
                (">>> ENTER/ESPACO = proximo passo (BASE) <<<", COR_AVISO)]
    if passo == 3:
        return [("PASSO 3/4: PAN pela BASE (joint1)", COR_TIT),
                ("", COR_TXT),
                ("Mesma coisa, agora cutucando a BASE (pra comparar).", COR_TXT),
                (f"angulo de teste: {ang:.0f} graus   ([ diminui  ] aumenta)", COR_VAL),
                ("Tecle 'm' para medir.", COR_TXT),
                ("", COR_TXT),
                ("resultado BASE:   " + r(res_base), COR_OK if res_base and res_base["paneia"] else COR_TXT),
                (">>> ENTER/ESPACO = ver o RESUMO <<<", COR_AVISO)]
    # passo 4
    return [("PASSO 4/4: RESUMO", COR_TIT),
            ("", COR_TXT),
            ("PUNHO (joint5):  " + r(res_punho), COR_VAL),
            ("BASE  (joint1):  " + r(res_base), COR_VAL),
            ("", COR_TXT),
            ("Se o PUNHO PANEIA bem (px/grau horizontal decente), da pra", COR_TXT),
            ("fazer a cascata pescoco: pequeno=punho, grande=base.", COR_DIM),
            ("Tudo foi gravado no log. Mande pro assistente analisar.", COR_TXT),
            (">>> ESC para sair (braco segura na home) <<<", COR_AVISO)]


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log = Diario(os.path.join(LOG_DIR, "pescoco_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl"))
    detector = DetectorFaces()
    try:
        arm = RobotArm(); arm.connect()
        pos0 = np.asarray(arm.get_positions(request=True), dtype=float)
    except Exception as e:
        print("!!! sem braco:", e); log.evento("erro_conexao", msg=str(e)); log.close()
        tela_sem_braco(); return
    n = arm.num_joints
    est["kp_hold"] = np.array([j.kp for j in arm._joints], dtype=float)
    est["kd_hold"] = np.array([j.kd for j in arm._joints], dtype=float)
    est["q_target"] = pos0.copy(); est["home"] = pos0.copy()
    est["livre"] = True; est["tracking"] = False
    log.config(lab="pescoco", kp_hold=[round(float(x), 0) for x in est["kp_hold"]])

    cap = None
    try:
        cap, idx = camera.abrir_camera(CAMERA_PULSO)
        arm.enable()
        if not motor_pronto(arm):
            if cap is not None:
                cap.release()
            tela_sem_braco(); return
        arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
        arm.start_control_loop(controlador)
        janela = "Lab Pescoco (guiado)"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL); cv2.resizeWindow(janela, 1280, 760)

        passo = 1
        ang = 8.0
        res_punho = res_base = None
        home = pos0.copy()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            faces = detector.detectar(frame, escala=0.5)
            f = max(faces, key=lambda f: f.area) if faces else None
            cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (255, 255, 255), 1)
            if f is not None:
                cv2.rectangle(frame, (f.x, f.y), (f.x + f.w, f.y + f.h), (0, 230, 0), 2)
            painel(frame, 12, 12, passo_texto(passo, ang, res_punho, res_base), escala=0.6)
            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k != 255:
                log.evento("tecla", passo=passo, cod=int(k))
            if k == 27:                       # ESC
                break
            elif k == ord("]"):
                ang = min(20.0, ang + 1.0)
            elif k == ord("["):
                ang = max(2.0, ang - 1.0)
            elif passo == 1 and k == 32:      # ESPACO trava home
                home = np.asarray(arm.get_positions(), dtype=float).copy()
                est["home"] = home.copy(); est["q_target"][:] = home; est["livre"] = False
                log.evento("home", q_deg=[round(float(np.degrees(x)), 1) for x in home])
                passo = 2
            elif passo == 2 and k == ord("m"):
                res_punho = nudge_e_mede(cap, detector, janela, home, PUNHO, ang, "PUNHO j5")
                log.evento("medida_punho", **(res_punho or {"falhou": True}))
            elif passo == 3 and k == ord("m"):
                res_base = nudge_e_mede(cap, detector, janela, home, BASE, ang, "BASE j1")
                log.evento("medida_base", **(res_base or {"falhou": True}))
            elif passo in (2, 3) and k in (13, 32):   # ENTER/ESPACO -> proximo
                passo += 1
            elif passo == 4 and k in (13, 32):
                pass
    finally:
        print("\n--- encerrando (APOIE o braço) ---")
        for fn in (arm.stop_control_loop, arm.disable, arm.disconnect):
            try:
                fn()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        log.evento("fim"); log.close()
        print("--- encerrado ---")


if __name__ == "__main__":
    main()
