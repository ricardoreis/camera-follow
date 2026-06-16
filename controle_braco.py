#!/usr/bin/env python3
"""controle_braco.py — loop de controle MIT (gravidade) + estado do braço.

SEMPRE em MIT (nunca troca de modo — ver Aprendizado #8 da DOCUMENTACAO):
  - livre=True  → kp MOLE + segue a mão (flutuar/reposicionar).
  - livre=False → kp FIRME (ganhos de fábrica) → segura/segue sem ceder à gravidade.
  - tracking=True → ZERA o integral (o laço visual da IK é o integrador; sem windup).

Compartilha o modelo Pinocchio com mira_ik. O dict `est` é o estado partilhado
com o loop de 500 Hz (o app preenche/atualiza; o controlador lê).
"""

import os
import sys

import numpy as np

ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)
import pinocchio as pin  # noqa: E402
from reBotArm_control_py.dynamics import compute_generalized_gravity  # noqa: E402

from mira_ik import _model, _data, _ee_id   # compartilha o modelo carregado

# Ganhos do hold MIT. FLUTUAR usa kp MOLE (pra mover com a mão). SEGURAR/SEGUIR usa
# kp FIRME (de fábrica, ~120/18) — senão as juntas que carregam o peso cedem.
KP, KD, KI = 8.0, 2.0, 3.0          # MOLE (flutuar)
VEL_THR, W_THR = 0.04, 0.08         # limiares p/ detectar a mão movendo o braço

# Estado partilhado com o loop de 500 Hz.
#  kp_hold/kd_hold = ganhos FIRMES (preenchidos pelo app após conectar).
est = {"q_target": None, "home": None, "repouso": None,
       "livre": True, "integral": None, "tracking": False,
       "kp_hold": None, "kd_hold": None}


def controlador(arm, dt):
    """Hold por gravidade (MIT). Se 'livre', segue a mão; se 'tracking', zera o
    integral (o laço visual da IK é o integrador). Igual em espírito ao 07/08."""
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
        integ[:] = 0.0          # laço visual é o integrador → sem windup
    else:
        integ += (qt - q) * KI * dt
        np.clip(integ, -0.5, 0.5, out=integ)
        if est["livre"]:        # detecta a mão movendo o braço → acompanha
            pin.computeJointJacobians(_model, _data, q)
            pin.updateFramePlacements(_model, _data)
            J = pin.getFrameJacobian(_model, _data, _ee_id, pin.ReferenceFrame.WORLD)
            v = J @ qd
            if np.linalg.norm(v[:3]) > VEL_THR or np.linalg.norm(v[3:]) > W_THR:
                qt[:] = q
                integ *= 0.9
    n = arm.num_joints
    if est["livre"]:                       # flutuar com a mão → mole
        kp_arr, kd_arr = np.full(n, KP), np.full(n, KD)
    else:                                  # segurar / seguir → firme (não cede c/ gravidade)
        kp_arr, kd_arr = est["kp_hold"], est["kd_hold"]
    arm.mit(pos=qt, vel=np.zeros(n), kp=kp_arr, kd=kd_arr, tau=tau_g + integ)


def motor_pronto(arm):
    """True se TODOS os motores responderam (status_code==1). Detecta braço sem energia."""
    try:
        for jc in arm._joints:
            st = arm._motor_map[jc.name].get_state()
            if st is None or getattr(st, "status_code", None) != 1:
                return False
        return True
    except Exception:
        return False


def status_motores(arm):
    """Código de status de cada motor (1 = ok; >1 = erro/falha; 0 = desabilitado;
    -1 sem leitura). Para detectar a falha (LED vermelho)."""
    out = []
    for jc in arm._joints:
        try:
            st = arm._motor_map[jc.name].get_state()
            out.append(int(getattr(st, "status_code", -1)) if st is not None else -1)
        except Exception:
            out.append(-1)
    return out
