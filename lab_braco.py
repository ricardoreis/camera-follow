#!/usr/bin/env python3
"""lab_braco.py — acordar / flutuar / pousar o braço, reutilizável pelos labs.

Mesma mecânica do seguir_ik (SEMPRE MIT): conecta, sobe suave até a home salva
(config_seguir_ik.json) e deixa FLUTUANDO p/ você posicionar a câmera com as mãos;
ao sair, pousa suave no repouso. Se NÃO houver braço, opera em modo "mock" (o lab roda
só com a câmera). Teclas no lab: ESPAÇO trava · f flutua · ESC pousa+sai.
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

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "config_seguir_ik.json")
DUR_ACORDAR, DUR_REPOUSO = 3.0, 3.5


class Braco:
    """Controla acordar/flutuar/pousar. `arm is None` → sem braço (só câmera)."""

    def __init__(self, cap, janela):
        self.cap, self.janela, self.arm = cap, janela, None

    @property
    def livre(self):
        return bool(est["livre"])

    def iniciar(self):
        """Conecta, entra em MIT, sobe até a home salva e flutua. Devolve True se ok."""
        try:
            arm = RobotArm()
            arm.connect()
            pos0 = np.asarray(arm.get_positions(request=True), dtype=float)
        except Exception as e:
            print("!!! sem braço:", e, "— seguindo só com a câmera")
            return False
        self.arm = arm
        n = arm.num_joints
        est["kp_hold"] = np.array([j.kp for j in arm._joints], dtype=float)
        est["kd_hold"] = np.array([j.kd for j in arm._joints], dtype=float)
        est["q_target"] = pos0.copy()
        est["repouso"] = pos0.copy()
        est["livre"] = False
        est["tracking"] = False
        arm.enable()
        if not motor_pronto(arm):
            print("!!! braço sem comunicação (sem energia?) — seguindo só com a câmera")
            self.arm = None
            return False
        arm.mode_mit(kp=np.full(n, KP), kd=np.full(n, KD))
        arm.start_control_loop(controlador)
        home = pos0
        if os.path.exists(CONFIG_PATH):
            try:
                home = np.asarray(json.load(open(CONFIG_PATH))["home"], dtype=float)
            except Exception:
                pass
        self._ramp(home, "ACORDANDO...", DUR_ACORDAR)
        self.flutuar()
        print("--- FLUTUANDO: posicione a câmera com as mãos. ESPACO trava | f flutua | ESC sai ---")
        return True

    def _quadro(self, msg):
        ok, fr = self.cap.read()
        if ok:
            cv2.putText(fr, msg, (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(self.janela, fr)
            cv2.waitKey(1)

    def _ramp(self, destino, msg, dur, segura=0.0):
        if self.arm is None:
            return
        est["livre"] = False
        est["tracking"] = False
        ini = est["q_target"].copy()
        destino = np.asarray(destino, dtype=float)
        t0 = time.time()
        while True:
            frac = (time.time() - t0) / dur
            if frac >= 1.0:
                break
            s = frac * frac * (3.0 - 2.0 * frac)
            est["q_target"][:] = ini + (destino - ini) * s
            self._quadro(msg)
        est["q_target"][:] = destino
        t1 = time.time()
        while time.time() - t1 < segura:
            self._quadro("ASSENTANDO...")

    def flutuar(self):
        if self.arm is None:
            return
        est["tracking"] = False
        est["q_target"] = np.asarray(self.arm.get_positions(), dtype=float).copy()
        est["integral"] = None
        est["livre"] = True

    def travar(self):
        if self.arm is None:
            return
        est["q_target"] = np.asarray(self.arm.get_positions(), dtype=float).copy()
        est["integral"] = None
        est["livre"] = False

    def encerrar(self):
        if self.arm is None:
            return
        print("--- pousando no repouso (APOIE o braço) ---")
        try:
            self._ramp(est["repouso"], "RETORNANDO AO REPOUSO...", DUR_REPOUSO, segura=2.0)
        except Exception:
            pass
        for fn in (self.arm.stop_control_loop, self.arm.disable, self.arm.disconnect):
            try:
                fn()
            except Exception:
                pass
