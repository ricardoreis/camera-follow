#!/usr/bin/env python3
"""lab_altura.py — LAB GUIADO: diagnostica o "MOVIMENTO ESTRANHO" da ALTURA e acha a
ALTURA MÁXIMA SEGURA.

Faz o mapa em SIMULAÇÃO (não move o braço em risco): com a home travada te encarando,
mede até onde o IK resolve BEM conforme a altura sobe (tilt=0), e a zona instável
ALTURA × TILT (incl. o "flip" que a recuperação por home causaria). Loga tudo p/ um
diagnóstico certeiro.

Rodar (no venv da app):  .venv/bin/python lab_altura.py
PASSO 1: flutue e ENCARE-SE; ESPACO trava a home e roda o mapa.
PASSO 2: RESUMO na tela + log em logs_labs/altura_*.jsonl. ESC pousa e sai.
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
ALT_TESTE = 0.45
ALTURAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
TILTS = list(range(25, -31, -3))     # +25 (olha p/ cima) -> -30 (p/ baixo)
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_seguir_ik.json")


def folga_limites(q):
    d = np.minimum(q - (_LO + MARGEM), (_HI - MARGEM) - q)
    i = int(np.argmin(d))
    return float(np.degrees(d[i])), i + 1       # junta 1..6


def mapa_altura(geom, home, reg):
    """Sobe a altura (tilt=0) e mede o IK. Devolve a altura máx 'boa' (cm) + o ponto onde quebrou."""
    q_seed = home.copy(); alt_ok = 0.0; quebra = None; a = 0.0
    while a <= ALT_TESTE + 1e-9:
        q, ok, iters, _ = resolver_ik(geom, 0.0, 0.0, q_seed, a)
        jump = float(np.degrees(np.max(np.abs(q - q_seed)))) if ok else None
        fol, ji = folga_limites(q) if ok else (None, None)
        bom = ok and jump is not None and jump <= IK_FLIP and fol is not None and fol > 1.5
        reg.linha(tipo="alt", alt_cm=round(a * 100, 1), ok=bool(ok), jump=jump,
                  iters=int(iters), folga_deg=fol, junta=ji)
        if bom:
            q_seed = q; alt_ok = a
        else:
            quebra = {"alt_cm": round(a * 100, 1), "ok": bool(ok),
                      "jump": round(jump, 1) if jump else None,
                      "folga": round(fol, 1) if fol else None, "junta": ji}
            break
        a += 0.01
    return round(alt_ok * 100, 1), quebra


def mapa_alt_tilt(geom, home, reg):
    """Pra cada altura, varre o tilt e mede IK + o 'flip' do re-seed por HOME (a recuperação)."""
    linhas = []
    for alt in ALTURAS:
        q_seed = home.copy(); a = 0.0
        while a < alt - 1e-9:                    # encadeia o seed até essa altura
            q, ok, _, _ = resolver_ik(geom, 0.0, 0.0, q_seed, min(a, alt))
            if ok:
                q_seed = q
            a += 0.02
        tilt_min_ok = None; flip_max = 0.0; falhas = 0
        for td in TILTS:
            q, ok, iters, _ = resolver_ik(geom, 0.0, np.radians(td), q_seed, alt)
            jump = float(np.degrees(np.max(np.abs(q - q_seed)))) if ok else None
            bom = ok and jump is not None and jump <= IK_FLIP
            qh, okh, _, _ = resolver_ik(geom, 0.0, np.radians(td), home, alt)   # re-seed por home
            flip = float(np.degrees(np.max(np.abs(qh - q)))) if (ok and okh) else None
            reg.linha(tipo="alt_tilt", alt_cm=round(alt * 100), tilt=td, ok=bool(ok),
                      jump=round(jump, 1) if jump else None,
                      flip_home=round(flip, 1) if flip else None, iters=int(iters))
            if bom:
                q_seed = q; tilt_min_ok = td
            else:
                falhas += 1
            if flip is not None:
                flip_max = max(flip_max, flip)
        linhas.append({"alt_cm": round(alt * 100), "tilt_min_ok": tilt_min_ok,
                       "falhas": falhas, "flip_max": round(flip_max)})
    return linhas


def _instrucoes(passo, alt_ok, quebra, tab):
    if passo == 1:
        return [("LAB ALTURA — PASSO 1/2: POSICIONAR", (0, 215, 255)),
                ("", (200, 200, 200)),
                ("Flutue o braco com a mao ate ele TE ENCARAR de frente,", (210, 210, 210)),
                ("com seu rosto no centro. (a home = referencia do mapa)", (210, 210, 210)),
                ("", (200, 200, 200)),
                (">>> ESPACO trava a home e RODA o mapa (simulacao) <<<", (60, 175, 255))]
    linhas = [("LAB ALTURA — PASSO 2/2: RESUMO", (0, 215, 255)), ("", (200, 200, 200)),
              (f"ALTURA MAXIMA SEGURA (tilt 0): ~{alt_ok:.0f} cm", (120, 235, 120))]
    if quebra:
        m = ("IK falhou" if not quebra["ok"] else
             f"saltou {quebra['jump']}deg" if (quebra.get('jump') or 0) > IK_FLIP else
             f"junta {quebra['junta']} no batente (folga {quebra['folga']}deg)")
        linhas.append((f"  quebrou em {quebra['alt_cm']:.0f} cm: {m}", (60, 175, 255)))
    linhas.append(("ALTURA x TILT (tilt_min = quanto da p/ olhar p/ BAIXO):", (235, 225, 130)))
    for r in tab:
        cor = (120, 120, 245) if (r["falhas"] > 0 or r["flip_max"] > IK_FLIP) else (210, 210, 210)
        tm = f"{r['tilt_min_ok']:+d}deg" if r["tilt_min_ok"] is not None else "FALHA"
        linhas.append((f"  {r['alt_cm']:>3.0f}cm: tilt_min {tm:>7}  falhas {r['falhas']}  "
                       f"flip-home ate {r['flip_max']:.0f}deg", cor))
    linhas += [("", (200, 200, 200)),
               ("Vermelho = zona do 'movimento estranho' (IK falha/flippa).", (150, 150, 150)),
               (">>> log salvo — mande pro assistente.  ESC pousa e sai <<<", (60, 175, 255))]
    return linhas


def main():
    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- câmera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem câmera, idx 0:", e); cap = cv2.VideoCapture(0)
    janela = "Lab ALTURA (guiado)  [ESPACO mapeia | ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL); cv2.resizeWindow(janela, 1280, 760)

    braco = Braco(cap, janela)
    braco.iniciar()
    reg = Registro("altura")

    passo, alt_ok, quebra, tab = 1, 0.0, None, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            hud(frame, _instrucoes(passo, alt_ok, quebra, tab), x=14, y=30)
            cv2.imshow(janela, frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
            elif k == 32 and passo == 1:        # ESPACO: trava home + roda o mapa
                if braco.arm is not None:
                    home = np.asarray(braco.arm.get_positions(), dtype=float)
                    braco.travar()
                elif os.path.exists(CONFIG_PATH):
                    home = np.asarray(json.load(open(CONFIG_PATH))["home"], dtype=float)
                else:
                    home = np.zeros(6)
                geom = geometria(home)
                reg.linha(tipo="home", home_deg=[round(float(np.degrees(x)), 1) for x in home])
                print("--- mapeando (simulacao)... ---")
                alt_ok, quebra = mapa_altura(geom, home, reg)
                tab = mapa_alt_tilt(geom, home, reg)
                reg.linha(tipo="resumo", alt_max_cm=alt_ok, quebra=quebra, tabela=tab)
                print("--- alt_max_ok:", alt_ok, "cm | tabela:", tab, "---")
                passo = 2
    finally:
        braco.encerrar()
        cap.release(); cv2.destroyAllWindows()
        reg.fim()
        print("--- log salvo ---")


if __name__ == "__main__":
    main()
