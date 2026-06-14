#!/usr/bin/env python3
"""
Rastreador de alvo - a arquitetura final de filtragem.

Encadeia os dois filtros decididos na Fase 4:

    ponto cru --> One Euro (suaviza) --> Kalman (estima velocidade e prevê)

Saída:
    suavizado : posição limpa, sem tremor (boa para mostrar onde o rosto ESTÁ).
    previsto  : posição t_pred ms À FRENTE, para compensar a latência do
                sistema (câmera->detecção->motor). É este ponto que a garra
                deve perseguir, para mirar onde o rosto VAI estar.
"""

from filtros import OneEuroPonto, KalmanCV


class RastreadorAlvo:
    def __init__(self, min_cutoff=1.0, beta=0.02,
                 sigma_a=100.0, sigma_z=3.0, t_pred_ms=60.0):
        self.euro = OneEuroPonto(min_cutoff, beta)
        self.kalman = KalmanCV(sigma_a, sigma_z)
        self.t_pred_ms = t_pred_ms

    def update(self, ponto_cru, t):
        """Processa um ponto cru (ou None se o rosto sumiu) no instante t.

        Devolve (suavizado, previsto). Ambos podem ser None no começo, antes
        de o filtro ter dados suficientes.
        """
        if ponto_cru is None:
            # Sem rosto: o Kalman desliza sozinho usando a última velocidade.
            self.kalman.coast(t)
            return None, self.kalman.prever(self.t_pred_ms / 1000.0)

        suavizado = self.euro(ponto_cru, t)          # 1) tira o tremor
        self.kalman.atualizar(suavizado, t)          # 2) alimenta o Kalman limpo
        previsto = self.kalman.prever(self.t_pred_ms / 1000.0)  # 3) prevê à frente
        return suavizado, previsto

    def velocidade(self):
        """Velocidade estimada do alvo (px/s), útil para diagnóstico."""
        return self.kalman.velocidade()
