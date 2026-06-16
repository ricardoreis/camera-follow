#!/usr/bin/env python3
"""lab_modo.py — LABORATÓRIO isolado: hold + flutuar SEM trocar de modo.

Objetivo: provar (no hardware, sem câmera/IK/UI) que dá pra:
  1) SEGURAR o braço numa pose levantada, firme, em modo MIT (gravidade + kp);
  2) RAMPAR suave entre "sentado" e "levantado" (o movimento do 'acordar');
  3) FLUTUAR (seguir a mão) e voltar a firme — TUDO sem trocar de modo.

Por que isto existe: na app (10_seguir_ik.py) o 'f' fazia o braço DESPENCAR. A causa
é que trocar POS_VEL→MIT chama `mode_mit()`, que é BLOQUEANTE (~0,5s, junta por junta)
e durante a troca os motores ficam sem comando de sustentação → queda livre. A solução
testada aqui é NÃO trocar de modo: ficar sempre em MIT (como o 08), e o "flutuar" ser
apenas um flag (segue a mão), instantâneo e sem queda.

Uso (braço LIGADO e começando SENTADO):
    .venv/bin/python lab_modo.py
Teclas (na janela preta):
    1   rampa até a pose LEVANTADA (home salva, ou um delta padrão)
    2   rampa de volta à pose SENTADA
    l   liga/desliga FLUTUAR (seguir a mão)
    ESC rampa de volta ao sentado e sai (desliga torque)

Observe: ao levantar (1), ele SEGURA sem ceder? Ao flutuar (l) levantado, ele
mantém a pose (gravidade) e você consegue movê-lo com a mão sem ele despencar?
Tudo é gravado em logs_ik/lab_AAAAMMDD_HHMMSS.jsonl para análise.
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
import pinocchio as pin  # noqa: E402
from reBotArm_control_py.actuator import RobotArm  # noqa: E402
from reBotArm_control_py.dynamics import compute_generalized_gravity  # noqa: E402
from reBotArm_control_py.kinematics import load_robot_model  # noqa: E402

KP, KD, KI = 8.0, 2.0, 3.0          # hold MIT (igual ao 07/08)
VEL_THR, W_THR = 0.04, 0.08         # limiares p/ detectar a mão movendo (flutuar)
DUR_RAMPA = 3.0                     # s da rampa sentado<->levantado
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_seguir_ik.json")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_ik")

_model = load_robot_model()
_data = _model.createData()
_ee_id = _model.getFrameId("end_link")

est = {"q_target": None, "integral": None, "livre": False}


def controlador(arm, dt):
    """500 Hz, SEMPRE MIT: gravidade + kp/kd; se livre, segue a mão."""
    if dt <= 0:
        dt = 0.002
    q = arm.get_positions()
    qd = arm.get_velocities()
    tau_g = compute_generalized_gravity(q=q)
    if est["integral"] is None:
        est["integral"] = np.zeros_like(q)
    integ = est["integral"]
    qt = est["q_target"]
    integ += (qt - q) * KI * dt
    np.clip(integ, -0.5, 0.5, out=integ)
    if est["livre"]:
        pin.computeJointJacobians(_model, _data, q)
        pin.updateFramePlacements(_model, _data)
        J = pin.getFrameJacobian(_model, _data, _ee_id, pin.ReferenceFrame.WORLD)
        v = J @ qd
        if np.linalg.norm(v[:3]) > VEL_THR or np.linalg.norm(v[3:]) > W_THR:
            qt[:] = q
            integ *= 0.9
    n = arm.num_joints
    arm.mit(pos=qt, vel=np.zeros(n),
            kp=np.full(n, KP), kd=np.full(n, KD), tau=tau_g + integ)


def hud(q, q_target, livre, msg=""):
    img = np.full((300, 760, 3), 22, np.uint8)
    def put(t, y, c=(220, 220, 220), s=0.6):
        cv2.putText(img, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)
    put("LAB MODO — hold/flutuar sem trocar de modo (sempre MIT)", 34, (0, 215, 255), 0.6)
    put(f"modo: {'FLUTUANDO (segue a mao)' if livre else 'FIRME (hold)'}",
        70, (60, 175, 255) if livre else (100, 235, 140))
    put("q real (deg):   " + "  ".join(f"{np.degrees(x):+6.1f}" for x in q), 110)
    put("q alvo (deg):   " + "  ".join(f"{np.degrees(x):+6.1f}" for x in q_target), 140)
    sag = np.degrees(np.max(np.abs(q - q_target)))
    put(f"desvio max alvo: {sag:5.1f} deg   (alto = cedendo/caindo)",
        174, (80, 80, 245) if sag > 5 else (150, 150, 150))
    put("1 levantar   2 sentar   l flutuar on/off   ESC sai", 220, (235, 225, 130))
    if msg:
        put(msg, 256, (0, 215, 255))
    cv2.imshow("lab_modo", img)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(os.path.join(LOG_DIR, "lab_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl"), "w")
    t0 = time.time()

    def grava(tipo, **kv):
        kv.update(tipo=tipo, t=round(time.time() - t0, 3))
        log.write(json.dumps(kv) + "\n"); log.flush()

    arm = RobotArm()
    arm.connect()
    pos0 = np.asarray(arm.get_positions(request=True), dtype=float)
    print("--- conectado ---")
    est["q_target"] = pos0.copy()
    sentado = pos0.copy()

    # Pose LEVANTADA de teste: a 'home' salva, ou (sem config) só joint2/3/4 (sobe reto).
    if os.path.exists(CONFIG_PATH):
        cfg = json.load(open(CONFIG_PATH))
        levantada = np.asarray(cfg["home"], dtype=float)
        sentado = np.asarray(cfg.get("repouso", pos0), dtype=float)
    else:
        levantada = pos0.copy()
        levantada[1] += np.radians(20)    # joint2
        levantada[2] -= np.radians(25)    # joint3
        levantada[3] += np.radians(10)    # joint4
    grava("config", sentado=[round(float(np.degrees(x)), 1) for x in sentado],
          levantada=[round(float(np.degrees(x)), 1) for x in levantada], kp=KP)

    n = arm.num_joints
    arm.enable()
    arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))   # MIT UMA vez (arm sentado = seguro)
    arm.start_control_loop(controlador)
    cv2.namedWindow("lab_modo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("lab_modo", 760, 300)
    print("--- MIT ativo. 1 levanta, l flutua, ESC sai. ---")

    def rampa(destino, rotulo):
        """Move o alvo de sentado<->levantado suave; o loop MIT segue. Loga o trajeto."""
        est["livre"] = False
        ini = est["q_target"].copy()
        ti = time.time()
        while True:
            frac = (time.time() - ti) / DUR_RAMPA
            if frac >= 1.0:
                break
            s = frac * frac * (3.0 - 2.0 * frac)
            est["q_target"][:] = ini + (destino - ini) * s
            q = arm.get_positions()
            grava("rampa", rotulo=rotulo,
                  qd=[round(float(np.degrees(x)), 1) for x in est["q_target"]],
                  qr=[round(float(np.degrees(x)), 1) for x in q])
            hud(q, est["q_target"], est["livre"], f"RAMPA: {rotulo}...")
            cv2.waitKey(1)
        est["q_target"][:] = destino

    try:
        while True:
            q = arm.get_positions()
            grava("frame", livre=est["livre"],
                  qd=[round(float(np.degrees(x)), 1) for x in est["q_target"]],
                  qr=[round(float(np.degrees(x)), 1) for x in q])
            hud(q, est["q_target"], est["livre"])
            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                grava("evento", ev="esc")
                rampa(sentado, "sentar (saida)")
                # segura assentado de fato antes de desligar
                tH = time.time()
                while time.time() - tH < 1.5:
                    hud(arm.get_positions(), est["q_target"], False, "ASSENTANDO...")
                    cv2.waitKey(1)
                break
            elif k == ord("1"):
                grava("evento", ev="levantar")
                rampa(levantada, "levantar")
            elif k == ord("2"):
                grava("evento", ev="sentar")
                rampa(sentado, "sentar")
            elif k == ord("l"):
                est["livre"] = not est["livre"]
                if not est["livre"]:                 # ao voltar a firme, fixa onde está
                    est["q_target"][:] = arm.get_positions()
                grava("evento", ev="flutuar", livre=est["livre"],
                      qr=[round(float(np.degrees(x)), 1) for x in arm.get_positions()])
    finally:
        print("\n--- encerrando: desligando torque (APOIE o braço) ---")
        for fn in (arm.stop_control_loop, arm.disable, arm.disconnect):
            try:
                fn()
            except Exception:
                pass
        cv2.destroyAllWindows()
        grava("evento", ev="fim")
        log.close()
        print("--- encerrado ---")


if __name__ == "__main__":
    main()
