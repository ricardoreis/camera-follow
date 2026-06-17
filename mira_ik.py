#!/usr/bin/env python3
"""mira_ik.py — modelo cinemático (Pinocchio) + IK do "pescoço fixo".

Extraído do 10_seguir_ik.py. Carrega o modelo UMA vez e oferece:
  geometria(q_home)                            -> (p0, R0, c, r, opt)
  resolver_ik(geom, pan, tilt, q_seed, altura) -> (q, ok, iters, ms)
  sinal_altura_de(geom)                        -> +1/-1 (sentido do acompanhamento de altura)

PAN nasce da BASE (giro da pose-alvo inteira em torno do Z do mundo → joint1).
TILT é um pitch no eixo do CORPO (body-Y, punho). ALTURA desloca a posição na
vertical do mundo. (Ver Aprendizado #9 da DOCUMENTACAO.)

Expõe também o modelo (_model/_data/_ee_id) e os limites de junta (_LO/_HI),
compartilhados com controle_braco.py.
"""

import os
import sys
import time

import numpy as np

ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)
import pinocchio as pin  # noqa: E402
from reBotArm_control_py.kinematics import (  # noqa: E402
    load_robot_model, compute_fk, get_end_effector_frame_id,
)
from reBotArm_control_py.kinematics.inverse_kinematics import (  # noqa: E402
    solve_ik, pos_rot_to_se3, IKParams,
)

# Qual coluna de R0 (eixo do end_link no mundo, na home) é o eixo óptico da câmera.
EIXO_OPTICO_COL = 0        # chute = X
# Eixo do CORPO (coluna de R0) usado para o TILT (pitch do punho). body-Y é suave.
TILT_BODY_COL = 1          # 1 = eixo Y do end_link
# IK em tempo real: warm-start + alvo perto → poucas iterações (ver 09_ik_lab.py).
IK_RT = IKParams(max_iter=200, tolerance=1e-3, step_size=0.5, damping=0.01)

_model = load_robot_model()
_data = _model.createData()
_ee_id = get_end_effector_frame_id(_model)
# Limites de junta (rad), com infinitos saneados — usados para o clamp com margem.
_LO = np.array([x if np.isfinite(x) else -np.pi for x in _model.lowerPositionLimit])
_HI = np.array([x if np.isfinite(x) else np.pi for x in _model.upperPositionLimit])


def geometria(q_home):
    """Deriva (p0, R0, pivô c, raio r, eixo óptico) da pose-home, via FK."""
    p0, R0, _ = compute_fk(_model, q_home)
    opt = R0[:, EIXO_OPTICO_COL].copy()
    r = float(np.linalg.norm(p0))
    c = p0 - r * opt
    return p0, R0, c, r, opt


def sinal_altura_de(geom):
    """+1 se aumentar a altura (z) re-nivela um tilt 'pra cima'; -1 caso contrário.
    Deduz da geometria: como o eixo óptico inclina (componente vertical) com o tilt."""
    p0, R0, c, r, opt = geom
    dd = 0.01
    R_up = pin.AngleAxis(dd, R0[:, TILT_BODY_COL]).matrix() @ R0
    dz = (R_up[:, EIXO_OPTICO_COL][2] - R0[:, EIXO_OPTICO_COL][2]) / dd
    return 1.0 if dz > 0 else -1.0


def resolver_ik(geom, pan, tilt, q_seed, altura=0.0, roll=0.0, reach=0.0):
    """Mira (pan, tilt) em rad + 'altura' (m) + 'roll' (rad) + 'reach' (m) → pose SE(3).

        R_alvo = Rz_world(pan) · R0 · Ry_body(tilt) · Rx_optico(roll)
        p_alvo = Rz_world(pan) · p0 + [0, 0, altura] + reach · (Rz · eixo_optico)

    O 'roll' é uma rotação no EIXO ÓPTICO (head-tilt: inclina a cabeça pro ombro sem
    parar de encarar — vira o joint6). O 'reach' translada a câmera PRA FRENTE no eixo
    óptico (gesto "espreitar": chega o rosto mais perto). Devolve (q, ok, iters, ms)."""
    p0, R0, c, r, opt = geom
    Rz = pin.AngleAxis(pan, np.array([0.0, 0.0, 1.0])).matrix()   # giro na base (mundo Z)
    eixo_tilt = R0[:, TILT_BODY_COL]                              # eixo Y do corpo
    Ry = pin.AngleAxis(tilt, eixo_tilt).matrix()
    R = Rz @ Ry @ R0
    if roll != 0.0:                                              # roll no eixo óptico (X local)
        R = R @ pin.AngleAxis(roll, np.array([1.0, 0.0, 0.0])).matrix()
    p = Rz @ p0
    p = p + np.array([0.0, 0.0, altura])                         # acompanha a altura
    if reach != 0.0:                                            # chega perto no eixo óptico
        p = p + reach * (Rz @ opt)
    alvo = pos_rot_to_se3(p, R)
    t0 = time.perf_counter()
    res = solve_ik(_model, _data, _ee_id, alvo, q_seed.copy(), IK_RT)
    ms = (time.perf_counter() - t0) * 1000.0
    return res.q, res.success, res.iterations, ms
