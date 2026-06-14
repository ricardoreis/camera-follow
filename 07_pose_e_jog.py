#!/usr/bin/env python3
"""
Fase 6b+ - Posicionar (float+lock) e fazer jog seguro, com a câmera do pulso.

Dois sub-modos, ambos em MIT + compensação de gravidade (não troca de modo):

  FLUTUANDO  -> o braço flutua: empurre com a mão para posicioná-lo; quando
               você solta, ele TRAVA onde está (regra de trava por velocidade
               da ponta, igual ao exemplo 10 que você gosta).
  TRAVADO    -> o alvo só muda pelo teclado (jog), limitado a +/-LIMITE_DEG
               da pose neutra capturada. Bom para testar pan/tilt com folga.

Fluxo sugerido:
  1) Começa FLUTUANDO. Levante o braço com a mão até uma pose neutra boa
     (câmera apontando pra frente, seu rosto perto da cruz central). Solte.
  2) ESPACO -> captura essa pose como "neutra" (home) e TRAVA.
  3) Faça o jog (1-6, a/d) para testar pan (joint5) e tilt (joint4) com folga.
  4) 'f' volta a FLUTUAR para reposicionar, se precisar.

SEGURANCA:
  - Compensação de gravidade: o braço se sustenta (não despenca) enquanto roda.
  - No TRAVADO, alvo limitado a +/-LIMITE_DEG da pose neutra (clamp).
  - kp/kd baixos (braço "complacente", mais seguro).
  - Ao sair: desliga torque -> braço fica mole. APOIE o braço ao encerrar.

TECLAS (na janela da câmera):
  ESPACO ....... captura pose neutra e TRAVA (estando FLUTUANDO)
  f ............ volta a FLUTUAR
  1..6 ......... (travado) seleciona joint1..joint6
  a / d ........ (travado) gira a junta ativa - / +
  c ............ (travado) recentra a junta ativa na pose neutra
  h ............ (travado) manda todas as juntas para a pose neutra
  ESC / q ...... sair (desliga torque)
"""

import sys

import numpy as np
import cv2

ARM_REPO = "/home/ricardo-reis/GITHUB/reBotArm_control_py"
sys.path.insert(0, ARM_REPO)
import pinocchio as pin  # noqa: E402
from reBotArm_control_py.actuator import RobotArm  # noqa: E402
from reBotArm_control_py.dynamics import compute_generalized_gravity  # noqa: E402
from reBotArm_control_py.kinematics import load_robot_model  # noqa: E402

import camera  # noqa: E402
from detector import DetectorFaces  # noqa: E402

CAMERA_PULSO = "C920"

# Controle. KD maior que o exemplo (mais amortecimento) e integral GENTIL
# escalado por dt (ver controlador) para evitar o ciclo-limite no modo travado.
KP, KD, KI = 8.0, 2.0, 3.0
VEL_THR = 0.04      # m/s  - limiar de velocidade linear da ponta para "seguir a mão"
W_THR = 0.08        # rad/s - limiar de velocidade angular
EE_FRAME = "end_link"

# Jog.
LIMITE_DEG = 25.0
PASSO_DEG = 2.0
PAN, TILT = 4, 3    # joint5, joint4 (índices 0-based)

# Modelo cinemático (para a regra de trava por velocidade da ponta).
_model = load_robot_model()
_data = _model.createData()
_ee_id = _model.getFrameId(EE_FRAME)

# Estado compartilhado entre o loop de controle (500Hz) e o loop da câmera.
est = {
    "q_target": None,   # alvo MIT (rad)
    "home": None,       # pose neutra capturada (rad)
    "livre": True,      # True=flutuando, False=travado
    "integral": None,   # termo integral (trim de gravidade)
    "ativa": PAN,
}


def controlador(arm, dt):
    """500 Hz: gravidade + segura/segue o alvo (MIT)."""
    q = arm.get_positions()
    qd = arm.get_velocities()
    tau_g = compute_generalized_gravity(q=q)

    if dt <= 0:
        dt = 0.002
    if est["integral"] is None:
        est["integral"] = np.zeros_like(q)
    integ = est["integral"]
    qt = est["q_target"]

    # Integral GENTIL, escalado por dt (PI correto). A versão antiga somava
    # sem dt -> ganho efetivo ~100x maior -> ciclo-limite no modo travado.
    integ += (qt - q) * KI * dt
    np.clip(integ, -0.5, 0.5, out=integ)

    if est["livre"]:
        # Regra de trava: se a ponta está se movendo (você empurrou), o alvo
        # segue a mão; parado, o alvo congela -> o braço "flutua e trava".
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


def main():
    detector = DetectorFaces()
    arm = RobotArm()
    arm.connect()
    print("--- conectado ---")
    n = arm.num_joints
    est["q_target"] = arm.get_positions(request=True).copy()
    est["home"] = est["q_target"].copy()

    cap = None
    try:
        cap, idx = camera.abrir_camera(CAMERA_PULSO)
        print(f"--- câmera {CAMERA_PULSO} (idx {idx}) ---")
        arm.enable()
        arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
        arm.start_control_loop(controlador)
        print("--- FLUTUANDO: posicione o braço com a mão e tecle ESPACO p/ travar ---")

        janela = "Camera Follow - Fase 6b+ (pose e jog)"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)

        lim = np.radians(LIMITE_DEG)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            # Mostra o ponto entre os olhos (ajuda a centrar a pose neutra).
            faces = detector.detectar(frame, escala=0.5)
            alvo_face = max(faces, key=lambda f: f.area) if faces else None
            if alvo_face is not None:
                cv2.circle(frame, alvo_face.centro_olhos, 6, (0, 0, 255), 2)

            # Mira central.
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)

            pos = arm.get_positions()
            qt = est["q_target"]
            home = est["home"]
            modo = "FLUTUANDO (empurre p/ mover; ESPACO trava)" if est["livre"] \
                else f"TRAVADO  junta ativa: joint{est['ativa']+1}"

            linhas = [f"MODO: {modo}"]
            if not est["livre"]:
                for i in (TILT, PAN):
                    d_alvo = np.degrees(qt[i] - home[i])
                    d_real = np.degrees(pos[i] - home[i])
                    nome = "PAN  joint5" if i == PAN else "TILT joint4"
                    marca = ">>" if i == est["ativa"] else "  "
                    linhas.append(f"{marca} {nome}: alvo {d_alvo:+6.1f}  real {d_real:+6.1f} deg  (lim +/-{LIMITE_DEG:.0f})")
                linhas.append("1-6 sel | a/d gira | c centra | h home | f flutua | ESC sai")
            else:
                linhas.append("Posicione com a mao. ESPACO=travar  ESC=sair")

            fonte = cv2.FONT_HERSHEY_SIMPLEX
            larg = max(cv2.getTextSize(s, fonte, 0.55, 1)[0][0] for s in linhas)
            ov = frame.copy()
            cv2.rectangle(ov, (6, 8), (6 + larg + 14, 8 + 24 * len(linhas) + 6), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)
            y = 30
            for s in linhas:
                cv2.putText(frame, s, (12, y), fonte, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                y += 24

            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == 32:  # ESPACO: captura pose neutra e trava
                est["home"] = arm.get_positions().copy()
                est["q_target"][:] = est["home"]
                est["livre"] = False
                print(f"--- TRAVADO em pose neutra: {np.degrees(est['home']).round(1)} deg ---")
            elif k == ord("f"):
                est["livre"] = True
                print("--- FLUTUANDO ---")
            elif not est["livre"]:
                if ord("1") <= k <= ord("6"):
                    est["ativa"] = k - ord("1")
                elif k in (ord("a"), ord("d")):
                    passo = np.radians(PASSO_DEG) * (1 if k == ord("d") else -1)
                    i = est["ativa"]
                    qt[i] = np.clip(qt[i] + passo, home[i] - lim, home[i] + lim)
                elif k == ord("c"):
                    qt[est["ativa"]] = home[est["ativa"]]
                elif k == ord("h"):
                    qt[:] = home

    finally:
        print("\n--- encerrando: desligando torque (APOIE o braço) ---")
        for fn in (arm.stop_control_loop, arm.disable, arm.disconnect):
            try:
                fn()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("--- encerrado ---")


if __name__ == "__main__":
    main()
