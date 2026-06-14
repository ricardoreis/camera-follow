#!/usr/bin/env python3
"""
Fase 6c - CAMERA FOLLOW: a garra encara o rosto (laço fechado).

Percepção: YuNet -> rastreador (One Euro + Kalman) -> erro em pixels.
Controle: lei PROPORCIONAL ancorada na posição real da junta (à prova de
windup) -> empurra joint5 (pan) e joint4 (tilt) para centralizar o rosto.

  alvo_junta = posicao_atual_da_junta + ganho * erro_angular

Como a câmera gira junto com a junta (~1:1), com ganho <= 1 isso converge
suave e NUNCA passa do ponto (nada de "quicar"/windup).

CALIBRAÇÃO AUTOMÁTICA (tecla k): o programa cutuca cada junta alguns graus,
mede pra que lado o rosto se move na imagem, e deduz sozinho o SINAL e a
ESCALA real (pixels por grau). Sem adivinhação.

Base de hold = MIT + compensação de gravidade (estável, do 07).

SEGURANCA: tracking começa OFF ('t' liga); clamp +/-LIMITE_DEG da neutra;
passo por frame limitado; deadzone; segura sem rosto; desliga torque ao sair.

TECLAS:
  ESPACO ... (flutuando) captura pose neutra e trava
  f ........ volta a flutuar
  k ........ CALIBRAR sinais e escala (fique parado, rosto visível)
  t ........ liga/desliga o TRACKING
  x / y .... inverte manualmente o sinal de pan / tilt
  [ / ] .... ganho - / +
  , / . .... previsão (ms) - / +
  c ........ recentra o olhar na pose neutra
  ESC / q .. sair (desliga torque)
"""

import json
import os
import sys
import time

import numpy as np
import cv2

# Caminho do repo de controle do braço (Seeed). Configurável por variável de
# ambiente REBOT_ARM_REPO para quem clonar o projeto em outra máquina.
ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)
import pinocchio as pin  # noqa: E402
from reBotArm_control_py.actuator import RobotArm  # noqa: E402
from reBotArm_control_py.dynamics import compute_generalized_gravity  # noqa: E402
from reBotArm_control_py.kinematics import load_robot_model  # noqa: E402

import camera  # noqa: E402
from detector import DetectorFaces  # noqa: E402
from rastreador import RastreadorAlvo  # noqa: E402

CAMERA_PULSO = "C920"

# Hold (gravidade + PD + integral gentil), igual ao 07.
KP, KD, KI = 8.0, 2.0, 3.0
VEL_THR, W_THR = 0.04, 0.08
EE_FRAME = "end_link"

PAN, TILT = 4, 3            # joint5, joint4

# Escala inicial (chute por FOV da C920); a calibração substitui pela medida.
FOV_H, FOV_V = 70.0, 43.0

KP_SERVO = 0.20            # ganho proporcional (fração do erro angular)
DEADZONE_PX = 12
MAX_STEP_DEG = 1.5         # passo máx/frame: alto o bastante p/ a lei desacelerar sozinha
LIMITE_DEG = 40.0          # excursão máxima de cada junta (pan/tilt) a partir da neutra
JANELA_W, JANELA_H = 1600, 900   # tamanho inicial da janela de vídeo
PREVISAO_MS = 60.0

# Calibração automática.
DELTA_CAL_DEG = 5.0        # quanto cutuca cada junta ao calibrar
SETTLE_S = 0.6             # espera o braço chegar
MEAS_S = 0.3              # janela de medição do rosto

# Movimentos suaves (acordar / dormir).
DUR_ACORDAR = 3.0          # s para subir à pose neutra
DUR_REPOUSO = 3.5          # s para voltar ao repouso (mais lento/suave)

# Arquivo de config (pose neutra + calibração), salvo com a tecla 'n'.
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_seguir.json")

_model = load_robot_model()
_data = _model.createData()
_ee_id = _model.getFrameId(EE_FRAME)

est = {"q_target": None, "home": None, "livre": True, "integral": None,
       "tracking": False}


def controlador(arm, dt):
    """500 Hz: gravidade + segura/segue o alvo (estável, do 07)."""
    if dt <= 0:
        dt = 0.002
    q = arm.get_positions()
    qd = arm.get_velocities()
    tau_g = compute_generalized_gravity(q=q)

    if est["integral"] is None:
        est["integral"] = np.zeros_like(q)
    integ = est["integral"]
    qt = est["q_target"]

    if est["tracking"]:
        # Durante o tracking o LAÇO VISUAL é o integrador. Zera o integral das
        # juntas: com qt em movimento ele acumularia e causaria bounce.
        integ[:] = 0.0
    else:
        # Hold / float / repouso: integral gentil corrige o erro estático.
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


def medir_erro(cap, detector, dur=MEAS_S):
    """Média do erro (dx, dy) do rosto em relação ao centro, por 'dur' s."""
    t0 = time.time()
    xs, ys = [], []
    while time.time() - t0 < dur:
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        faces = detector.detectar(frame, escala=0.5)
        if faces:
            f = max(faces, key=lambda f: f.area)
            p = f.centro_olhos
            xs.append(p[0] - w // 2)
            ys.append(p[1] - h // 2)
    if not xs:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def calibrar_eixo(cap, detector, eixo, comp):
    """Cutuca a junta 'eixo' em +/-DELTA e mede como o rosto (componente
    'comp': 0=x, 1=y) se desloca. Devolve (sinal, pixels_por_grau) ou None."""
    home, qt = est["home"], est["q_target"]
    delta = np.radians(DELTA_CAL_DEG)

    qt[eixo] = home[eixo] - delta
    time.sleep(SETTLE_S)
    menos = medir_erro(cap, detector)

    qt[eixo] = home[eixo] + delta
    time.sleep(SETTLE_S)
    mais = medir_erro(cap, detector)

    qt[eixo] = home[eixo]
    time.sleep(SETTLE_S)

    if menos is None or mais is None:
        return None
    d = mais[comp] - menos[comp]          # deslocamento do rosto por +2*delta
    if abs(d) < 8:                         # variação pequena demais -> não confio
        return None
    ppd = abs(d) / (2 * DELTA_CAL_DEG)     # pixels por grau (escala real)
    sinal = -1 if d > 0 else +1            # junta+ moveu rosto +d -> centra com o oposto
    return sinal, ppd


def mover_suave(arm, cap, janela, destino, msg, dur):
    """Move o braço da posição atual até 'destino' em 'dur' segundos, com
    aceleração e desaceleração suaves (smoothstep) - "acordar"/"dormir"."""
    est["livre"] = False
    qt = est["q_target"]
    inicio = qt.copy()
    destino = np.asarray(destino, dtype=float)
    t0 = time.time()
    print(f"--- {msg} ---")
    while True:
        frac = (time.time() - t0) / dur
        if frac >= 1.0:
            break
        s = frac * frac * (3.0 - 2.0 * frac)   # smoothstep: ease-in-out
        qt[:] = inicio + (destino - inicio) * s
        ok = False
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                cv2.putText(frame, msg, (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow(janela, frame)
                cv2.waitKey(1)
                ok = True
        if not ok:
            time.sleep(0.02)
    qt[:] = destino


def salvar_config(repouso, neutra, sinal_pan, sinal_tilt, radpx_x, radpx_y,
                  kp_servo, deadzone_px, limite_deg):
    """Salva pose de repouso, pose neutra, calibração e parâmetros de feel."""
    data = {
        "repouso": [float(x) for x in repouso],
        "neutra": [float(x) for x in neutra],
        "sinal_pan": int(sinal_pan),
        "sinal_tilt": int(sinal_tilt),
        "radpx_x": float(radpx_x),
        "radpx_y": float(radpx_y),
        "kp_servo": float(kp_servo),
        "deadzone_px": int(deadzone_px),
        "limite_deg": float(limite_deg),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(data, f, indent=2)


def carregar_config():
    if not os.path.exists(CONFIG_PATH):
        return None
    with open(CONFIG_PATH) as f:
        return json.load(f)


def main():
    detector = DetectorFaces()
    rastreador = RastreadorAlvo(t_pred_ms=PREVISAO_MS)
    arm = RobotArm()
    arm.connect()
    print("--- conectado ---")
    n = arm.num_joints
    est["q_target"] = arm.get_positions(request=True).copy()
    est["home"] = est["q_target"].copy()
    est["repouso"] = est["q_target"].copy()   # pose do início = repouso (comece "sentado")

    kp_servo = KP_SERVO
    deadzone_px = DEADZONE_PX
    limite_deg = LIMITE_DEG
    sinal_pan, sinal_tilt = -1, +1
    previsao_ms = PREVISAO_MS
    tracking = False
    calibrado = False
    # rad por pixel (chute inicial por FOV; calibração substitui).
    radpx_x = np.radians(FOV_H) / 1280.0
    radpx_y = np.radians(FOV_V) / 720.0

    cap = None
    try:
        cap, idx = camera.abrir_camera(CAMERA_PULSO)
        print(f"--- câmera {CAMERA_PULSO} (idx {idx}) ---")
        arm.enable()
        arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
        arm.start_control_loop(controlador)

        janela = "Camera Follow - Fase 6c (a garra te encara)"
        cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(janela, JANELA_W, JANELA_H)
        max_step = np.radians(MAX_STEP_DEG)

        # --- Auto-start: se houver config, ACORDA na pose neutra e já segue ---
        cfg = carregar_config()
        if cfg is not None:
            # neutra relativa ao repouso salvo, aplicada ao repouso atual
            # (robusto a pequenas variações da pose inicial entre execuções).
            delta = np.array(cfg["neutra"]) - np.array(cfg["repouso"])
            neutra_alvo = est["repouso"] + delta
            mover_suave(arm, cap, janela, neutra_alvo, "ACORDANDO...", DUR_ACORDAR)
            est["home"] = neutra_alvo.copy()
            est["livre"] = False
            sinal_pan = cfg["sinal_pan"]
            sinal_tilt = cfg["sinal_tilt"]
            radpx_x = cfg["radpx_x"]
            radpx_y = cfg["radpx_y"]
            kp_servo = cfg.get("kp_servo", KP_SERVO)
            deadzone_px = cfg.get("deadzone_px", DEADZONE_PX)
            limite_deg = cfg.get("limite_deg", LIMITE_DEG)
            calibrado = True
            tracking = True
            print("--- ACORDOU em pose neutra. TRACKING ON (config carregada). ---")
        else:
            print("--- Sem config. Posicione, ESPACO trava, k calibra, t segue, "
                  "n salva tudo p/ auto-start. ---")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2

            faces = detector.detectar(frame, escala=0.5)
            alvo_face = max(faces, key=lambda f: f.area) if faces else None
            ponto_cru = alvo_face.centro_olhos if alvo_face else None
            rastreador.t_pred_ms = previsao_ms
            _, prev = rastreador.update(ponto_cru, time.time())

            qt, home = est["q_target"], est["home"]
            est["tracking"] = tracking and not est["livre"]
            lim = np.radians(limite_deg)

            # ---- LEI DE CONTROLE proporcional (ancorada na posição real) ----
            erro = None
            if tracking and not est["livre"] and ponto_cru is not None and prev is not None:
                q_pos = arm.get_positions()
                dx, dy = prev[0] - cx, prev[1] - cy
                erro = (dx, dy)
                # Cada eixo só se move FORA da zona morta. Dentro dela o alvo
                # CONGELA (não atualiza qt) -> motor firme, tremor ~zero.
                # alvo = posição atual + correção proporcional (desacelera sozinho).
                if abs(dx) >= deadzone_px:
                    des = q_pos[PAN] + sinal_pan * kp_servo * dx * radpx_x
                    des = np.clip(des, home[PAN] - lim, home[PAN] + lim)
                    qt[PAN] += np.clip(des - qt[PAN], -max_step, max_step)
                if abs(dy) >= deadzone_px:
                    des = q_pos[TILT] + sinal_tilt * kp_servo * dy * radpx_y
                    des = np.clip(des, home[TILT] - lim, home[TILT] + lim)
                    qt[TILT] += np.clip(des - qt[TILT], -max_step, max_step)

            # ---- Desenhos ----
            # Zona morta (retângulo): dentro dela a "cabeça" não se mexe.
            cv2.rectangle(frame, (cx - deadzone_px, cy - deadzone_px),
                          (cx + deadzone_px, cy + deadzone_px), (0, 200, 200), 1)
            cv2.line(frame, (cx - 25, cy), (cx + 25, cy), (255, 255, 255), 1)
            cv2.line(frame, (cx, cy - 25), (cx, cy + 25), (255, 255, 255), 1)
            if prev is not None:
                cor = (0, 0, 255) if tracking else (0, 165, 255)
                cv2.line(frame, (cx, cy), prev, cor, 2)
                cv2.circle(frame, prev, 7, cor, 2)

            if est["livre"]:
                estado_txt = "FLUTUANDO (ESPACO trava)"
            else:
                estado_txt = ("TRACKING ON" if tracking else "tracking off") + \
                    ("  [CALIBRADO]" if calibrado else "  [nao calibrado - tecle k]")
            linhas = [
                estado_txt,
                f"erro: {('%+d,%+d px' % erro) if erro else '--'}   ganho([/])={kp_servo:.2f}  zona_morta(o/p)={deadzone_px}px  limite(-/=)={limite_deg:.0f}deg",
                f"sinais: pan(x)={sinal_pan:+d} tilt(y)={sinal_tilt:+d}   "
                f"escala: {1/radpx_x*np.radians(1):.1f}/{1/radpx_y*np.radians(1):.1f} px/deg   prev(,/.)={previsao_ms:.0f}ms",
                "ESPACO trava | t segue | k calibra | n salva | [/] ganho | o/p zona | -/= limite | r reinicia | f flutua | ESC sai",
            ]
            fonte = cv2.FONT_HERSHEY_SIMPLEX
            larg = max(cv2.getTextSize(s, fonte, 0.55, 1)[0][0] for s in linhas)
            ov = frame.copy()
            cv2.rectangle(ov, (6, 8), (6 + larg + 14, 8 + 24 * len(linhas) + 6), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.55, frame, 0.45, 0, dst=frame)
            yy = 30
            for s in linhas:
                cv2.putText(frame, s, (12, yy), fonte, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
                yy += 24

            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                mover_suave(arm, cap, janela, est["repouso"],
                            "RETORNANDO AO REPOUSO...", DUR_REPOUSO)
                break
            elif k == ord("r"):
                # Reiniciar sem sair: volta ao repouso e acorda de novo.
                mover_suave(arm, cap, janela, est["repouso"],
                            "REINICIANDO (repouso)...", DUR_REPOUSO)
                rastreador = RastreadorAlvo(t_pred_ms=previsao_ms)
                if est["integral"] is not None:
                    est["integral"][:] = 0.0
                if cfg is not None:
                    delta = np.array(cfg["neutra"]) - np.array(cfg["repouso"])
                    neutra_alvo = est["repouso"] + delta
                    mover_suave(arm, cap, janela, neutra_alvo, "ACORDANDO...", DUR_ACORDAR)
                    est["home"] = neutra_alvo.copy()
                    est["livre"] = False
                    tracking = True
                else:
                    est["livre"] = True
                    tracking = False
                print("--- REINICIADO ---")
            elif k == 32:  # ESPACO trava
                est["home"] = arm.get_positions().copy()
                est["q_target"][:] = est["home"]
                est["livre"] = False
                print(f"--- TRAVADO em {np.degrees(est['home']).round(1)} deg ---")
            elif k == ord("f"):
                est["livre"] = True
                tracking = False
                print("--- FLUTUANDO ---")
            elif k == ord("k"):
                if est["livre"]:
                    print("!! trave primeiro (ESPACO)")
                else:
                    tracking = False
                    print("--- CALIBRANDO pan (joint5)... fique parado ---")
                    rp = calibrar_eixo(cap, detector, PAN, 0)
                    print("--- CALIBRANDO tilt (joint4)... ---")
                    rt = calibrar_eixo(cap, detector, TILT, 1)
                    if rp and rt:
                        sinal_pan, ppd_x = rp
                        sinal_tilt, ppd_y = rt
                        radpx_x = np.radians(1.0) / ppd_x
                        radpx_y = np.radians(1.0) / ppd_y
                        calibrado = True
                        print(f"--- CALIBRADO: pan sinal={sinal_pan:+d} {ppd_x:.1f}px/deg | "
                              f"tilt sinal={sinal_tilt:+d} {ppd_y:.1f}px/deg ---")
                    else:
                        print("!! calibração falhou (rosto visível e parado?). Tente de novo.")
            elif k == ord("n"):
                salvar_config(est["repouso"], arm.get_positions(),
                              sinal_pan, sinal_tilt, radpx_x, radpx_y,
                              kp_servo, deadzone_px, limite_deg)
                print(f"--- CONFIG SALVA em {CONFIG_PATH} (pose neutra + calibração). "
                      "Próximas execuções acordam e seguem sozinhas. ---")
            elif k == ord("t"):
                if est["livre"]:
                    print("!! trave primeiro (ESPACO)")
                else:
                    tracking = not tracking
                    print(f"--- TRACKING {'ON' if tracking else 'OFF'} ---")
            elif k == ord("x"):
                sinal_pan = -sinal_pan
            elif k == ord("y"):
                sinal_tilt = -sinal_tilt
            elif k == ord("]"):
                kp_servo = min(1.0, kp_servo + 0.05)
            elif k == ord("["):
                kp_servo = max(0.0, kp_servo - 0.05)
            elif k == ord("p"):
                deadzone_px = min(120, deadzone_px + 3)
            elif k == ord("o"):
                deadzone_px = max(0, deadzone_px - 3)
            elif k == ord("="):
                limite_deg = min(80.0, limite_deg + 5.0)
            elif k == ord("-"):
                limite_deg = max(5.0, limite_deg - 5.0)
            elif k == ord("."):
                previsao_ms = min(250.0, previsao_ms + 10.0)
            elif k == ord(","):
                previsao_ms = max(0.0, previsao_ms - 10.0)
            elif k == ord("c"):
                qt[PAN], qt[TILT] = home[PAN], home[TILT]

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
