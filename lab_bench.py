#!/usr/bin/env python3
"""lab_bench.py — medidor de fps/latência + HUD, compartilhado pelos labs de visão.

Sem dependências do braço: só cv2/numpy/time. Uso:

    med = Medidor()
    while ...:
        med.frame()
        with med.estagio("infer"):
            ... # roda o modelo
        hud(frame, [(f"fps {med.fps():.0f}", (0,255,180)), ...])

`med.resumo()` devolve {fps, <estágios em ms>} para gravar/printar no fim.
"""

import collections
import json
import os
import time
import urllib.request

import cv2

MODELOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_labs")
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_labs")


class Registro:
    """Log JSONL simples (1 linha = 1 dict) p/ salvar os sinais dos labs e reusar depois.
    Cada linha ganha um 't' (timestamp). Uso: reg.linha(...); reg.fim(resumo=...)."""

    def __init__(self, nome):
        os.makedirs(LOGS_DIR, exist_ok=True)
        self.caminho = os.path.join(LOGS_DIR, nome + "_" + time.strftime("%Y%m%d_%H%M%S") + ".jsonl")
        self.f = open(self.caminho, "w")
        print(f"--- log: {self.caminho} ---")

    def linha(self, **kw):
        kw.setdefault("t", round(time.time(), 3))
        self.f.write(json.dumps(kw, ensure_ascii=False) + "\n")
        self.f.flush()

    def fim(self, **kw):
        self.linha(tipo="fim", **kw)
        self.f.close()
        print(f"--- log salvo: {self.caminho} ---")


def baixar_modelo(url, nome):
    """Baixa um modelo (.task/.onnx) para models_labs/ se ainda não existir."""
    os.makedirs(MODELOS_DIR, exist_ok=True)
    dst = os.path.join(MODELOS_DIR, nome)
    if not os.path.exists(dst):
        print(f"--- baixando {nome}... ---")
        urllib.request.urlretrieve(url, dst)
        print(f"--- ok: {os.path.getsize(dst) // 1024} KB ---")
    return dst


class Medidor:
    def __init__(self, janela=30):
        self._jan = janela
        self.t_frames = collections.deque(maxlen=janela)
        self.estagios = {}
        self._t0 = None

    def frame(self):
        agora = time.perf_counter()
        if self._t0 is not None:
            self.t_frames.append(agora - self._t0)
        self._t0 = agora

    def estagio(self, nome):
        return _Cron(self, nome)

    def _registra(self, nome, ms):
        self.estagios.setdefault(nome, collections.deque(maxlen=self._jan)).append(ms)

    def fps(self):
        if not self.t_frames:
            return 0.0
        m = sum(self.t_frames) / len(self.t_frames)
        return 1.0 / m if m > 0 else 0.0

    def media_ms(self, nome):
        d = self.estagios.get(nome)
        return (sum(d) / len(d)) if d else 0.0

    def resumo(self):
        return {"fps": round(self.fps(), 1),
                **{k: round(self.media_ms(k), 1) for k in self.estagios}}


class _Cron:
    def __init__(self, med, nome):
        self.med, self.nome = med, nome

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.med._registra(self.nome, (time.perf_counter() - self.t) * 1000.0)


def hud(frame, linhas, x=10, y=26):
    """Desenha linhas (texto, cor_bgr) com fundo escuro (estilo terminal)."""
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    yy = y
    for item in linhas:
        txt = item[0] if isinstance(item, (tuple, list)) else item
        cor = item[1] if isinstance(item, (tuple, list)) and len(item) > 1 else (0, 255, 180)
        (w, h), _ = cv2.getTextSize(txt, fonte, 0.6, 1)
        cv2.rectangle(frame, (x - 4, yy - h - 5), (x + w + 6, yy + 6), (18, 18, 18), -1)
        cv2.putText(frame, txt, (x, yy), fonte, 0.6, cor, 1, cv2.LINE_AA)
        yy += h + 12
