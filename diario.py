#!/usr/bin/env python3
"""diario.py — log estruturado (JSONL) + Tee do stdout. Extraído do 10_seguir_ik.

Diario grava uma linha JSON por registro (config / evento / frame / stdout), pensado
para ser LIDO depois e reconstruir a sessão. Tee espelha o terminal para o log."""

import json
import time


class Diario:
    """Log estruturado: config, eventos (teclas/modos/falhas), saída de terminal e
    telemetria por frame. Flush imediato (se travar/cair, o log sobrevive)."""

    def __init__(self, path):
        self.f = open(path, "w")
        self.t0 = time.time()
        self.path = path

    def _w(self, obj):
        obj["t"] = round(time.time() - self.t0, 3)
        try:
            self.f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self.f.flush()
        except Exception:
            pass

    def config(self, **kv):
        self._w({"tipo": "config", **kv})

    def evento(self, ev, **kv):
        self._w({"tipo": "evento", "ev": ev, **kv})

    def frame(self, **kv):
        self._w({"tipo": "frame", **kv})

    def stdout(self, linha):
        self._w({"tipo": "stdout", "linha": linha})

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


class Tee:
    """Espelha o stdout: escreve no terminal E manda cada linha para um callback
    (para gravar no log toda 'saída de terminal', inclusive a da lib do braço)."""

    def __init__(self, orig, cb):
        self.orig, self.cb, self._buf = orig, cb, ""

    def write(self, s):
        self.orig.write(s)
        self._buf += s
        while "\n" in self._buf:
            linha, self._buf = self._buf.split("\n", 1)
            if linha.strip():
                try:
                    self.cb(linha)
                except Exception:
                    pass

    def flush(self):
        self.orig.flush()
