#!/usr/bin/env python3
"""reid.py — RE-IDENTIFICAÇÃO (não trocar de pessoa).

Worker ASSÍNCRONO: o ArcFace (InsightFace) é pesado p/ rodar todo frame, então roda numa
thread de FUNDO. O loop rápido (30fps) entrega o frame mais recente (`submeter`); a thread
detecta+embute os rostos de tempos em tempos, acha QUEM é o alvo (similaridade de cosseno
com o embedding TRAVADO) e publica o bbox do alvo + a flag is_alvo de cada rosto.

Uso típico no app:
    reid = ReID(dispositivo="ovcpu"); reid.travar()        # trava no maior rosto (você)
    ... por frame:
        reid.submeter(frame)                               # não bloqueia
        i = reid.idx_alvo([f.bbox for f in faces_yunet])   # qual rosto do loop rápido é você
"""

import threading
import time

import numpy as np
from insightface.app import FaceAnalysis

DET_SIZE = (480, 480)
LIMIAR = 0.35              # cosseno acima disso = "é a mesma pessoa"
EMA = 0.1                 # quão rápido o embedding-alvo se adapta (luz/ângulo)
TIMEOUT = 3.0             # s sem ver o alvo → considera "ausente" (folgado p/ gaps do worker)


def make_providers(dispositivo):
    """Providers do onnxruntime conforme o dispositivo (igual ao lab de identidade)."""
    d = (dispositivo or "ovcpu").lower()
    if d == "cpu":
        return ["CPUExecutionProvider"]
    dev = {"ovcpu": "CPU", "gpu": "GPU", "npu": "NPU"}.get(d)
    if dev:
        return [("OpenVINOExecutionProvider", {"device_type": dev}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _area(b):
    return float((b[2] - b[0]) * (b[3] - b[1]))


def _centro(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5)


class ReID:
    def __init__(self, dispositivo="ovcpu", pack="buffalo_s", limiar=LIMIAR, periodo=0.12):
        self.limiar = limiar
        self.periodo = periodo            # intervalo mínimo entre passes do worker (s)
        self.app = FaceAnalysis(name=pack, allowed_modules=["detection", "recognition"],
                                providers=make_providers(dispositivo))
        self.app.prepare(ctx_id=0, det_size=DET_SIZE)

        self._lock = threading.Lock()
        self._frame = None                # último frame submetido (ref; não copiamos)
        self._pedido = None               # "travar" | "destravar" | None
        self._stop = False

        self.alvo_emb = None              # embedding da pessoa-alvo (normalizado)
        self.alvo_bbox = None             # (x1,y1,x2,y2) onde o alvo foi visto por último
        self.alvo_sim = 0.0               # similaridade do último match
        self.alvo_visto = 0.0             # time.time() do último match
        self.faces = []                   # [[bbox, is_alvo, sim], ...] último pass (desenho)
        self.ms = 0.0                     # custo do último pass do ArcFace

        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    # ---- API do loop rápido ----
    def submeter(self, frame):
        """Entrega o frame mais recente p/ o worker (não bloqueia, não copia)."""
        with self._lock:
            self._frame = frame

    def travar(self):
        """Pede p/ travar no MAIOR rosto (mais próximo = você) no próximo pass."""
        self._pedido = "travar"

    def destravar(self):
        self._pedido = "destravar"

    @property
    def tem_alvo(self):
        return self.alvo_emb is not None

    @property
    def alvo_presente(self):
        return self.tem_alvo and (time.time() - self.alvo_visto) < TIMEOUT

    def idx_alvo(self, bboxes):
        """Dado os bboxes (x1,y1,x2,y2) do loop rápido, devolve o índice do que é o ALVO.
        Com UM rosto só (você sozinho) segue direto — re-ID só desambigua quando há vários,
        que é onde ele importa; assim não fica "perdido" piscando quando você está sozinho."""
        if len(bboxes) == 0:
            return None
        if len(bboxes) == 1:
            return 0                                    # 1 pessoa = é ela (sem exigir match)
        if not self.alvo_presente or self.alvo_bbox is None:
            return None                                 # vários e sem confirmação → não chuta estranho
        ca = _centro(self.alvo_bbox)
        diag = max(1.0, (self.alvo_bbox[2] - self.alvo_bbox[0]))
        melhor, dmin = None, 1e9
        for i, b in enumerate(bboxes):
            c = _centro(b)
            d = ((c[0] - ca[0]) ** 2 + (c[1] - ca[1]) ** 2) ** 0.5
            if d < dmin:
                dmin, melhor = d, i
        return melhor if dmin < diag * 1.5 else None    # só se razoavelmente perto

    def encerrar(self):
        self._stop = True

    # ---- thread de fundo ----
    def _worker(self):
        while not self._stop:
            with self._lock:
                frame = self._frame
                self._frame = None
                pedido = self._pedido
                self._pedido = None
            if frame is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            faces = self.app.get(frame)
            self.ms = (time.time() - t0) * 1000.0

            if pedido == "destravar":
                self.alvo_emb = None
                self.alvo_bbox = None
            elif pedido == "travar" and faces:
                alvo = max(faces, key=lambda f: _area(f.bbox))     # maior = mais próximo
                self.alvo_emb = _norm(alvo.normed_embedding.astype(np.float32).copy())
                self.alvo_bbox = tuple(int(v) for v in alvo.bbox)
                self.alvo_visto = time.time()
                self.alvo_sim = 1.0

            # associa cada rosto ao alvo (cosseno) e marca o melhor acima do limiar
            res, melhor, melhor_sim = [], None, -1.0
            for f in faces:
                sim = float(np.dot(f.normed_embedding, self.alvo_emb)) if self.tem_alvo else 0.0
                linha = [tuple(int(v) for v in f.bbox), False, round(sim, 2)]
                res.append(linha)
                if sim > melhor_sim:
                    melhor_sim, melhor = sim, (f, linha)
            if self.tem_alvo and melhor is not None and melhor_sim >= self.limiar:
                f, linha = melhor
                linha[1] = True
                self.alvo_bbox = tuple(int(v) for v in f.bbox)
                self.alvo_sim = melhor_sim
                self.alvo_visto = time.time()
                if melhor_sim > 0.5:                              # adapta devagar (só se confiante)
                    self.alvo_emb = _norm((1 - EMA) * self.alvo_emb + EMA * f.normed_embedding)
            self.faces = res

            dt = time.time() - t0
            if dt < self.periodo:
                time.sleep(self.periodo - dt)
