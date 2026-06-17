#!/usr/bin/env python3
"""vida.py — comportamentos de "vida" quando você NÃO está interagindo:

    respirar(t, amp)   -> micro-movimento ocioso (pan, tilt em GRAUS): a cabeça
                          "respira" de leve quando parada (não fica estátua).
    Curiosidade        -> quando você fica parado+centralizado por um tempo,
                          sinaliza p/ disparar um gesto sozinho (com cooldown e
                          tempos sorteados). O app escolhe QUAL gesto e dispara.

Usado pelo seguir_ik_web.py (o seguir_ik.py original fica intacto).
"""

import random

import numpy as np


def respirar(t, amp=1.0):
    """Offset suave (pan, tilt) em GRAUS — soma de senos com períodos incomensuráveis
    (não múltiplos), então o padrão não se repete de forma óbvia → parece respiração
    natural. Amplitude pequena de propósito (cabe na zona morta)."""
    pan = 0.6 * np.sin(2 * np.pi * t / 9.0) + 0.25 * np.sin(2 * np.pi * t / 5.3 + 1.0)
    tilt = 0.5 * np.sin(2 * np.pi * t / 4.7) + 0.2 * np.sin(2 * np.pi * t / 7.9 + 2.0)
    return pan * amp, tilt * amp


class Curiosidade:
    """Dispara curiosidade quando você fica parado+centralizado. `update()` devolve
    True UMA vez quando é hora de reagir; o app sorteia o gesto e o toca."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = None              # quando começou a ficar parado/centralizado
        self.cooldown_ate = 0.0     # sem reação até este instante
        self.prox_parado = 5.0      # tempo-alvo (sorteado) parado antes de reagir

    def update(self, agora, pronto, parado_s, cooldown_s):
        """pronto = (seguindo + centralizado + parado + sem gesto + curiosidade ON)."""
        if not pronto:
            self.t0 = None
            return False
        if self.t0 is None:                      # acabou de ficar parado → re-arma
            self.t0 = agora
            self.prox_parado = parado_s * random.uniform(0.8, 1.3)
        if agora < self.cooldown_ate:
            return False
        if agora - self.t0 >= self.prox_parado:  # parado tempo suficiente → reage
            self.cooldown_ate = agora + cooldown_s * random.uniform(0.8, 1.3)
            self.t0 = None
            return True
        return False
