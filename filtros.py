#!/usr/bin/env python3
"""
Filtros de suavização e predição para o ponto de mira (centro dos olhos).

Implementamos os dois "na mão" para você ver a matemática:

  - OneEuro:  passa-baixa ADAPTATIVO. Suaviza muito quando o alvo está parado
              (mata o jitter) e "abre" quando o alvo se move (sem atraso).
              NÃO prediz. É o padrão da indústria para tracking interativo.

  - KalmanCV: filtro de Kalman com modelo de VELOCIDADE CONSTANTE. Estima
              posição E velocidade, então além de suavizar consegue PREVER
              onde o alvo estará daqui a X ms (compensa a latência do sistema).
"""

import math

import numpy as np


# ----------------------------------------------------------------------------
# One Euro Filter
# ----------------------------------------------------------------------------
def _alpha(cutoff, dt):
    """Converte uma frequência de corte (Hz) no fator de suavização alpha.

    alpha perto de 1 -> segue o sinal (pouca suavização).
    alpha perto de 0 -> ignora o sinal (muita suavização).
    """
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro1D:
    """One Euro para um único número (1 dimensão)."""

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.min_cutoff = min_cutoff   # corte em repouso: menor = mais suave
        self.beta = beta               # quanto a velocidade "abre" o filtro
        self.d_cutoff = d_cutoff       # corte do cálculo da derivada
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def __call__(self, x, t):
        if self.x_prev is None:
            self.x_prev, self.t_prev = x, t
            return x

        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-3

        # 1) Estima a velocidade (derivada) e a suaviza um pouco.
        dx = (x - self.x_prev) / dt
        a_d = _alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # 2) A SACADA: a frequência de corte cresce com a velocidade.
        #    Parado -> corte = min_cutoff (suave). Rápido -> corte alto (segue).
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _alpha(cutoff, dt)

        # 3) Passa-baixa com esse corte adaptativo.
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


class OneEuroPonto:
    """One Euro para um ponto 2D (aplica um filtro 1D em x e outro em y)."""

    def __init__(self, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
        self.fx = OneEuro1D(min_cutoff, beta, d_cutoff)
        self.fy = OneEuro1D(min_cutoff, beta, d_cutoff)

    def __call__(self, ponto, t):
        return (int(self.fx(ponto[0], t)), int(self.fy(ponto[1], t)))

    def set_params(self, min_cutoff, beta):
        self.fx.min_cutoff = self.fy.min_cutoff = min_cutoff
        self.fx.beta = self.fy.beta = beta


# ----------------------------------------------------------------------------
# Kalman - modelo de velocidade constante (2D)
# ----------------------------------------------------------------------------
class KalmanCV:
    """Estado = [x, y, vx, vy]. Mede só a posição; estima a velocidade.

    sigma_a: incerteza de aceleração (quanto deixamos o modelo "mudar de
             ideia"). Maior = mais responsivo e menos suave.
    sigma_z: ruído da medição em pixels (o tremor do YuNet). Maior = confia
             menos na medição = mais suave.
    """

    def __init__(self, sigma_a=100.0, sigma_z=3.0):
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z
        self.x = None          # estado (4x1)
        self.P = None          # covariância (incerteza) do estado (4x4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)  # mede posição
        self.t_prev = None

    def _F(self, dt):
        """Matriz de transição: aplica velocidade constante por dt segundos."""
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    def _Q(self, dt):
        """Ruído de processo para modelo de aceleração branca."""
        a = self.sigma_a ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        return a * np.array([[dt4 / 4, 0, dt3 / 2, 0],
                             [0, dt4 / 4, 0, dt3 / 2],
                             [dt3 / 2, 0, dt2, 0],
                             [0, dt3 / 2, 0, dt2]], dtype=float)

    def _predict_to(self, t):
        """Passo de PREDIÇÃO: projeta o estado até o instante t."""
        dt = t - self.t_prev
        if dt <= 0:
            dt = 1e-3
        self.t_prev = t
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)

    def atualizar(self, z, t):
        """Predição + CORREÇÃO com a medição z=(x,y) do detector."""
        z = np.array([[z[0]], [z[1]]], dtype=float)
        if self.x is None:
            # Primeira medição: inicializa posição, velocidade zero, muita incerteza.
            self.x = np.array([[z[0, 0]], [z[1, 0]], [0.0], [0.0]])
            self.P = np.eye(4) * 1000.0
            self.t_prev = t
            return

        self._predict_to(t)

        R = np.eye(2) * (self.sigma_z ** 2)
        y = z - self.H @ self.x                       # inovação (erro)
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)      # ganho de Kalman
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def coast(self, t):
        """Sem medição (rosto perdido): só prediz, deixando o alvo deslizar."""
        if self.x is not None:
            self._predict_to(t)

    def posicao(self):
        if self.x is None:
            return None
        return (int(self.x[0, 0]), int(self.x[1, 0]))

    def prever(self, t_frente):
        """Posição prevista t_frente segundos à frente (NÃO altera o estado).

        É isto que compensa a latência: miramos onde o rosto VAI estar.
        """
        if self.x is None:
            return None
        xp = self._F(t_frente) @ self.x
        return (int(xp[0, 0]), int(xp[1, 0]))

    def velocidade(self):
        if self.x is None:
            return (0.0, 0.0)
        return (float(self.x[2, 0]), float(self.x[3, 0]))
