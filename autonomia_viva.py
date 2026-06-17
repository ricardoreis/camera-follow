#!/usr/bin/env python3
"""autonomia_viva.py — FUGA/perseguição + VARREDURA (olhar ao redor).

Versão estendida da autonomia.py (que fica intacta, usada pelo seguir_ik.py). Aqui,
quando o rosto some, o braço: vai RETO pro lado que você saiu (perseguir) e, se não te
acha, OLHA AO REDOR (varredura) por alguns ciclos, depois fica OCIOSO esperando — e
tenta varrer de novo. Usado pelo seguir_ik_web.py.

Estados: seguindo → perseguindo → varrendo → ocioso → (varrendo) → … → seguindo.
A "mira" é base_pan/base_tilt (rad, relativa à home; 0 = encarando).

`self.varredura_on` é setado pelo app a cada frame (par["procurar_on"]).
"""

import random
import time

import numpy as np

T_PERDIDO = 0.3        # s sem rosto até começar a PERSEGUIR
T_PURSUIT = 2.0        # s perseguindo o canto antes de varrer
PESO_TILT_BUSCA = 0.5  # suaviza o componente vertical da perseguição
BUSCA_PERIODO = 6.0    # s de um ciclo completo da varredura (olhar ao redor)
BUSCA_AMP_FRAC = 0.8   # fração do 'limite' usada na varredura
N_CICLOS_MIN, N_CICLOS_MAX = 1, 3    # quantas varreduras antes de desistir (sorteado)
OCIOSO_MIN, OCIOSO_MAX = 6.0, 16.0   # s de espera (ocioso) antes de varrer de novo


class Autonomia:
    """Máquina de estados da fuga + varredura. `update()` é chamada a cada frame."""

    def __init__(self, max_step):
        self.max_step = float(max_step)
        self.varredura_on = True       # setado pelo app (par["procurar_on"])
        self.reset()

    def reset(self):
        self.estado = "seguindo"
        self.t_perdido = None
        self.persiga_t0 = None
        self.ultimo_prev = None
        self.alvo_pan = 0.0
        self.alvo_tilt = 0.0
        self.dir_saida = 0.0
        self.busca_t0 = None
        self.n_ciclos = 1
        self.ocioso_ate = 0.0

    def update(self, ponto_cru, prev, cx, cy, base_pan, base_tilt,
               sinal_pan, sinal_tilt, radpx_x, radpx_y, lim, lim_tilt):
        agora = time.time()
        if ponto_cru is not None and prev is not None:
            self.ultimo_prev = prev

        # --- rosto PRESENTE → seguir normal (não mexe na mira) ---
        if ponto_cru is not None:
            self.estado = "seguindo"
            self.t_perdido = None
            return base_pan, base_tilt, self.estado, self.dir_saida

        # --- rosto AUSENTE ---
        if self.t_perdido is None:
            self.t_perdido = agora
            if self.ultimo_prev is not None:        # calcula pra onde você saiu
                ex = self.ultimo_prev[0] - cx
                ey = self.ultimo_prev[1] - cy
                self.dir_saida = float(np.sign(ex))
                denom = max(abs(ex) / max(cx, 1), abs(ey) / max(cy, 1), 1e-3)
                fator = min(1.0 / denom, 3.0)
                self.alvo_pan = float(np.clip(
                    base_pan + sinal_pan * ex * fator * radpx_x, -lim, lim))
                self.alvo_tilt = float(np.clip(
                    base_tilt + sinal_tilt * ey * fator * radpx_y * PESO_TILT_BUSCA,
                    -lim_tilt, lim_tilt))

        if (self.estado == "seguindo" and self.ultimo_prev is not None
                and agora - self.t_perdido > T_PERDIDO):
            self.estado = "perseguindo"
            self.persiga_t0 = agora

        alvo_p, alvo_t = base_pan, base_tilt
        if self.estado == "perseguindo":
            alvo_p, alvo_t = self.alvo_pan, self.alvo_tilt
            if agora - self.persiga_t0 > T_PURSUIT:
                if self.varredura_on:
                    self.estado = "varrendo"
                    self.busca_t0 = agora
                    self.n_ciclos = random.randint(N_CICLOS_MIN, N_CICLOS_MAX)
                else:
                    self.estado = "ocioso"
                    self.ocioso_ate = agora + random.uniform(OCIOSO_MIN, OCIOSO_MAX)

        if self.estado == "varrendo":
            ts = agora - self.busca_t0
            if int(ts / BUSCA_PERIODO) >= self.n_ciclos:
                self.estado = "ocioso"
                self.ocioso_ate = agora + random.uniform(OCIOSO_MIN, OCIOSO_MAX)
            else:                                   # varre em cosseno (viés p/ a saída)
                amp = lim * BUSCA_AMP_FRAC
                sp = self.dir_saida or 1.0
                fase = 2 * np.pi * ts / BUSCA_PERIODO
                alvo_p = amp * sp * np.cos(fase)
                alvo_t = lim_tilt * 0.3 * np.sin(fase * 0.5)

        if self.estado == "ocioso":
            alvo_p, alvo_t = 0.0, 0.0               # volta ao centro e espera
            if self.varredura_on and agora >= self.ocioso_ate:
                self.estado = "varrendo"
                self.busca_t0 = agora
                self.n_ciclos = random.randint(N_CICLOS_MIN, N_CICLOS_MAX)

        base_pan += float(np.clip(alvo_p - base_pan, -self.max_step, self.max_step))
        base_tilt += float(np.clip(alvo_t - base_tilt, -self.max_step, self.max_step))
        return base_pan, base_tilt, self.estado, self.dir_saida
