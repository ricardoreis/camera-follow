#!/usr/bin/env python3
"""gestos.py — perfis temporais de gestos de "cabeça".

HEAD-TILT = um ROLL no eixo óptico da câmera (inclina a cabeça pro ombro, tipo
cachorro/macaco curioso, SEM parar de te encarar — vira o joint6 via IK). Os perfis
devolvem uma fração ao longo do tempo, aplicada à amplitude do roll no app.
"""

HT_SOBE, HT_SEGURA, HT_DESCE = 0.30, 0.9, 0.5   # tempos (s): inclina rápido, segura, volta
HEAD_TILT_DEG = 25.0                            # amplitude padrão do head-tilt


def perfil_head_tilt(e, t_sobe=HT_SOBE, t_segura=HT_SEGURA, t_desce=HT_DESCE):
    """Fração 0..1 do head-tilt ao longo do tempo 'e' (s), ou None quando termina.
    Inclina rápido (smoothstep), segura inclinado ("pensando..."), e volta suave."""
    dur = t_sobe + t_segura + t_desce
    if e >= dur:
        return None
    if e < t_sobe:                          # subida rápida
        s = e / t_sobe
        return s * s * (3 - 2 * s)
    if e < t_sobe + t_segura:               # segura inclinado
        return 1.0
    s = (e - t_sobe - t_segura) / t_desce   # volta
    return 1.0 - s * s * (3 - 2 * s)
