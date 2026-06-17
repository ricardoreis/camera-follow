#!/usr/bin/env python3
"""gestos.py — perfis temporais de gestos de "cabeça" + despachante.

Um gesto é um movimento TRANSITÓRIO (dura alguns segundos e volta ao 0) sobreposto à
mira, enquanto o servo congela. Há dois "estilos":

  • CORPO TODO (via pose do IK): mexe o braço inteiro.
      single / swing -> ROLL no eixo óptico (= joint6): inclina a cabeça pro ombro.
      feliz          -> dTILT oscilando: o corpo sobe/desce (comemora, "pula de feliz").
      dancar         -> dPAN oscilando: o corpo balança (gingado).
      espreitar      -> dREACH: chega o rosto um pouco mais perto e volta.
  • SÓ A CABEÇA (offset DIRETO numa junta do punho): o corpo fica parado.
      sim  -> joint4 (tilt do punho) oscila: acena "sim" só com a cabeça.
      nao  -> joint5 (pan do punho) oscila: balança "não" só com a cabeça.

`aplicar_gesto(tipo, e, params)` devolve um dict com os offsets do frame
({roll, dtilt, dpan, dreach} em rad/m, e/ou `dq`={idx: rad} p/ juntas diretas) ou
None quando o gesto termina.
"""

import numpy as np

HT_SOBE, HT_SEGURA, HT_DESCE = 0.30, 0.9, 0.5   # tempos (s): inclina rápido, segura, volta
HEAD_TILT_DEG = 25.0                            # amplitude padrão do head-tilt

J_NOD = 3       # joint4: tilt do punho (aceno "sim", só a cabeça)
J_SHAKE = 4     # joint5: pan do punho (balança "não", só a cabeça)

# Tipos de gesto (ordem usada para ciclar no painel de AJUSTES).
GESTO_TIPOS = ["single", "swing", "sim", "nao", "feliz", "dancar", "espreitar"]


def perfil_head_tilt(e, t_sobe=HT_SOBE, t_segura=HT_SEGURA, t_desce=HT_DESCE):
    """Fração 0..1 ao longo do tempo 'e' (s), ou None ao terminar. Sobe rápido
    (smoothstep), segura ("pensando..."), e volta suave. É o envelope-base."""
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


def perfil_swing(e, t_sobe=HT_SOBE, t_segura=HT_SEGURA, t_desce=HT_DESCE):
    """Vai a um lado e direto ao inverso, SEM parar no centro. Fração -1..+1
    (o sinal já alterna), ou None ao terminar."""
    t1 = t_sobe                  # 0 -> +1
    t2 = t1 + t_segura           # segura +1
    t3 = t2 + 2 * t_sobe         # +1 -> -1 (passa pelo centro)
    t4 = t3 + t_segura           # segura -1
    t5 = t4 + t_desce            # -1 -> 0
    if e >= t5:
        return None
    if e < t1:
        s = e / t_sobe
        return s * s * (3 - 2 * s)
    if e < t2:
        return 1.0
    if e < t3:
        s = (e - t2) / (2 * t_sobe)
        return 1.0 - 2 * (s * s * (3 - 2 * s))
    if e < t4:
        return -1.0
    s = (e - t4) / t_desce
    return -1.0 + (s * s * (3 - 2 * s))


def perfil_osc(e, t_sobe=HT_SOBE, t_segura=HT_SEGURA, t_desce=HT_DESCE, ciclos=2):
    """Oscilação senoidal (-1..+1) embalada pelo envelope (cresce, oscila, some).
    Usada pelos gestos 'sim'/'nao'. None ao terminar."""
    env = perfil_head_tilt(e, t_sobe, t_segura, t_desce)
    if env is None:
        return None
    dur = t_sobe + t_segura + t_desce
    return env * np.sin(2 * np.pi * ciclos * e / dur)


def aplicar_gesto(tipo, e, p):
    """Offsets transitórios do gesto neste frame, ou None ao terminar.

    p = dict(amp, sobe, segura, desce, dir). 'amp' em graus (gestos angulares);
    no 'espreitar' vira reach em metros (amp*0.0025 → 25deg≈6cm)."""
    sobe, segura, desce = p["sobe"], p["segura"], p.get("desce", HT_DESCE)
    amp = np.radians(p["amp"])
    # --- corpo todo (via pose do IK) ---
    if tipo == "single":
        f = perfil_head_tilt(e, sobe, segura, desce)
        return None if f is None else {"roll": p["dir"] * amp * f}
    if tipo == "swing":
        f = perfil_swing(e, sobe, segura, desce)
        return None if f is None else {"roll": amp * f}
    if tipo == "feliz":                     # corpo sobe/desce (comemora)
        f = perfil_osc(e, sobe, segura, desce, ciclos=2)
        return None if f is None else {"dtilt": amp * f}
    if tipo == "dancar":                    # corpo balança (gingado)
        f = perfil_osc(e, sobe, segura, desce, ciclos=2)
        return None if f is None else {"dpan": amp * f}
    if tipo == "espreitar":                 # chega perto e volta (reach)
        f = perfil_head_tilt(e, sobe, segura, desce)
        return None if f is None else {"dreach": p["amp"] * 0.0025 * f}
    # --- só a cabeça (offset DIRETO numa junta do punho) ---
    if tipo == "sim":                       # acena "sim" só com o punho (joint4)
        f = perfil_osc(e, sobe, segura, desce, ciclos=2)
        return None if f is None else {"dq": {J_NOD: amp * f}}
    if tipo == "nao":                       # balança "não" só com a cabeça (joint5)
        f = perfil_osc(e, sobe, segura, desce, ciclos=2)
        return None if f is None else {"dq": {J_SHAKE: amp * f}}
    return None
