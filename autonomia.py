#!/usr/bin/env python3
"""autonomia.py — FUGA/perseguição: quando o rosto some, o braço vai RETO pro lado
que você saiu e espera você reaparecer. Portado do 08 (seção 7 "Autonomia") para a
app de IK; agora alcança mais longe (pan pela base, não só o punho).

Estados:
    seguindo    -> rosto presente (o servo visual normal comanda)
    perseguindo -> rosto sumiu há > T_PERDIDO: vai pro canto da saída por T_PURSUIT s
    esperando   -> desistiu (após T_PURSUIT): volta ao centro/home e aguarda

A "mira" é base_pan/base_tilt (rad, relativa à home). Quando o rosto some, esta classe
sobrescreve a mira; quando ele volta, devolve sem alterar (o servo retoma).
"""

import time

import numpy as np

T_PERDIDO = 0.3        # s sem rosto até começar a PERSEGUIR
T_PURSUIT = 2.0        # s perseguindo o canto antes de desistir e voltar ao centro
PESO_TILT_BUSCA = 0.5  # suaviza o componente vertical da perseguição


class Autonomia:
    """Máquina de estados da fuga. `update()` é chamada a cada frame (quando seguindo);
    devolve a mira (possivelmente sobrescrita) + o estado e a direção de saída."""

    def __init__(self, max_step):
        self.max_step = float(max_step)        # passo máx de mira por frame (rad)
        self.reset()

    def reset(self):
        self.estado = "seguindo"
        self.t_perdido = None
        self.persiga_t0 = None
        self.ultimo_prev = None                # último ponto previsto com rosto presente
        self.alvo_pan = 0.0
        self.alvo_tilt = 0.0
        self.dir_saida = 0.0                   # sinal do lado da saída (p/ HUD)

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
            if self.ultimo_prev is not None:   # calcula pra onde você saiu
                ex = self.ultimo_prev[0] - cx
                ey = self.ultimo_prev[1] - cy
                self.dir_saida = float(np.sign(ex))
                # Direção REAL (proporcional), estendida até a borda na mesma proporção
                # → perseguição RETA (horizontal puro se você fugiu na horizontal).
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
                self.estado = "esperando"
        if self.estado == "esperando":
            alvo_p, alvo_t = 0.0, 0.0          # volta ao centro/home e aguarda

        base_pan += float(np.clip(alvo_p - base_pan, -self.max_step, self.max_step))
        base_tilt += float(np.clip(alvo_t - base_tilt, -self.max_step, self.max_step))
        return base_pan, base_tilt, self.estado, self.dir_saida
