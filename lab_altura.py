#!/usr/bin/env python3
"""lab_altura.py (v2) — LAB GUIADO: mede o ALCANCE VERTICAL do braco (altura) e onde o
IK falha (o "movimento estranho"). Tudo em SIMULACAO (nao move o braco em risco).

Como funciona: a partir de uma POSE DE REFERENCIA, varre a altura PRA CIMA e PRA BAIXO
(tilt=0) ate o IK falhar, achando o range usavel; e ve quanto da p/ olhar p/ baixo no topo.

Referencia (voce escolhe):
  ESPACO -> usa a HOME SALVA (config_seguir_ik.json) = sua pose normal. NAO toque no braco.
  H      -> usa a POSE ATUAL (posicione o braco com a mao antes) -> p/ testar uma home mais alta.

Texto ASCII (sem bug de fonte). ESC pousa e sai. Log em logs_labs/altura_*.jsonl.
Rodar:  .venv/bin/python lab_altura.py
"""

import json
import os
import time

import numpy as np
import cv2

import camera
from mira_ik import geometria, resolver_ik, _LO, _HI
from lab_bench import hud, Registro
from lab_braco import Braco

CAMERA = "C920"
MARGEM = np.radians(4.0)
IK_FLIP = 15.0
PASSO = 0.01            # 1 cm por passo
LIM = 0.45             # testa ate +/-45 cm
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_seguir_ik.json")


def folga(q):
    d = np.minimum(q - (_LO + MARGEM), (_HI - MARGEM) - q)
    i = int(np.argmin(d))
    return float(np.degrees(d[i])), i + 1


def bom(q, q0, ok):
    if not ok:
        return False
    jump = float(np.degrees(np.max(np.abs(q - q0))))
    fol, _ = folga(q)
    return jump <= IK_FLIP and fol > 1.5


def varre(geom, home, sentido, reg):
    """Varre a altura no 'sentido' (+1 cima / -1 baixo) ate falhar. Devolve (cm_ok, junta_lim)."""
    q = home.copy(); melhor = 0.0; junta = None; a = 0.0
    while abs(a) <= LIM + 1e-9:
        qx, ok, iters, _ = resolver_ik(geom, 0.0, 0.0, q, a)
        ok_bom = bom(qx, q, ok)
        fol, ji = folga(qx) if ok else (None, None)
        reg.linha(tipo="vert", alt_cm=round(a * 100, 1), ok=bool(ok), bom=ok_bom,
                  folga_deg=round(fol, 1) if fol else None, junta=ji)
        if ok_bom:
            q = qx; melhor = a
        else:
            junta = ji if ok else "IK falhou"
            break
        a += sentido * PASSO
    return round(melhor * 100), junta


def tilt_no_topo(geom, home, alt_topo_cm, reg):
    """No topo da altura, ate quanto da p/ olhar p/ baixo (tilt negativo) sem o IK falhar."""
    alt = alt_topo_cm / 100.0
    q = home.copy(); a = 0.0
    while a < alt - 1e-9:                       # encadeia o seed ate o topo
        qx, ok, _, _ = resolver_ik(geom, 0.0, 0.0, q, min(a, alt))
        if ok:
            q = qx
        a += 0.02
    tmin = 0
    for td in range(0, -41, -2):
        qx, ok, _, _ = resolver_ik(geom, 0.0, np.radians(td), q, alt)
        reg.linha(tipo="tilt_topo", tilt=td, ok=bool(ok and bom(qx, q, ok)))
        if bom(qx, q, ok):
            q = qx; tmin = td
        else:
            break
    return tmin


def rodar_mapa(geom, home, ref_nome, reg):
    reg.linha(tipo="ref", ref=ref_nome, home_deg=[round(float(np.degrees(x)), 1) for x in home])
    cima, jU = varre(geom, home, +1, reg)
    baixo, jD = varre(geom, home, -1, reg)
    tmin = tilt_no_topo(geom, home, cima, reg)
    res = {"ref": ref_nome, "baixo_cm": baixo, "cima_cm": cima,
           "range_cm": cima - baixo, "junta_cima": jU, "junta_baixo": jD, "tilt_topo": tmin}
    reg.linha(tipo="resumo", **res)
    print("--- RESUMO:", res, "---")
    return res


def instrucoes(res):
    if res is None:
        return [("LAB ALTURA v2 - alcance vertical (simulacao)", (0, 215, 255)),
                ("", (200, 200, 200)),
                ("ESPACO = mapear a partir da HOME SALVA (nao toque no braco)", (210, 210, 210)),
                ("H      = mapear a partir da POSE ATUAL (posicione com a mao)", (210, 210, 210)),
                ("", (200, 200, 200)),
                ("(o braco NAO se move no mapa - e tudo simulacao)", (150, 150, 150)),
                (">>> ESPACO ou H para medir | ESC sai <<<", (60, 175, 255))]
    return [("LAB ALTURA v2 - RESUMO", (0, 215, 255)),
            (f"referencia: {res['ref']}", (150, 150, 150)),
            ("", (200, 200, 200)),
            (f"ALCANCE VERTICAL: de {res['baixo_cm']:+d} cm ate {res['cima_cm']:+d} cm "
             f"(total {res['range_cm']} cm)", (120, 235, 120)),
            (f"limite p/ CIMA:  junta {res['junta_cima']}", (235, 225, 130)),
            (f"limite p/ BAIXO: junta {res['junta_baixo']}", (235, 225, 130)),
            (f"no topo (+{res['cima_cm']}cm), olha p/ baixo ate {res['tilt_topo']} graus", (235, 225, 130)),
            ("", (200, 200, 200)),
            ("ESPACO/H = medir de novo (outra referencia) | ESC pousa e sai", (60, 175, 255))]


def main():
    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- camera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem camera, idx 0:", e); cap = cv2.VideoCapture(0)
    janela = "Lab ALTURA v2  [ESPACO=home salva | H=pose atual | ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL); cv2.resizeWindow(janela, 1280, 760)

    braco = Braco(cap, janela)
    braco.iniciar()
    reg = Registro("altura")

    home_salva = None
    if os.path.exists(CONFIG_PATH):
        try:
            home_salva = np.asarray(json.load(open(CONFIG_PATH))["home"], dtype=float)
        except Exception:
            pass

    res = None
    pedido = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if pedido is not None:                 # mostra "CALCULANDO" antes de travar no calc
                fr = frame.copy()
                hud(fr, [("CALCULANDO... (simulacao, ~1s)", (60, 200, 255))], x=14, y=40)
                cv2.imshow(janela, fr); cv2.waitKey(1)
                ref_nome, home_ref = pedido
                res = rodar_mapa(geometria(home_ref), home_ref, ref_nome, reg)
                pedido = None
            hud(frame, instrucoes(res), x=14, y=30)
            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == 32:                          # ESPACO: home salva
                if home_salva is not None:
                    pedido = ("HOME SALVA", home_salva)
                else:
                    print("!!! sem config_seguir_ik.json (use H com a pose atual)")
            elif k == ord("h") and braco.arm is not None:   # H: pose atual
                pedido = ("POSE ATUAL (a mao)", np.asarray(braco.arm.get_positions(), dtype=float))
    finally:
        braco.encerrar()
        cap.release(); cv2.destroyAllWindows()
        reg.fim()
        print("--- log salvo ---")


if __name__ == "__main__":
    main()
