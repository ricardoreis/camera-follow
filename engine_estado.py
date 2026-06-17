#!/usr/bin/env python3
"""engine_estado.py — ponte thread-safe entre o "cérebro" (loop de visão/IK) e o
servidor web. NÃO comanda o braço: só guarda o último FRAME anotado e o último
ESTADO público (pro vídeo + websocket), e entrega uma FILA de comandos que o loop
do engine drena e aplica na PRÓPRIA thread (sem corrida com a visão/controle).

Princípio de segurança: o engine é autônomo. O servidor só LÊ (frame/estado) e
EMPILHA comandos; quem mexe no braço continua sendo só o loop do engine.
"""

import queue
import threading


class EngineEstado:
    """Singleton compartilhado entre o loop do engine e o servidor web."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None          # último frame anotado (BGR, numpy) — cópia
        self._estado = {}           # último estado público (dict JSON-ável)
        self.comandos = queue.Queue()

    # ---- frame (vídeo) ----
    def publicar_frame(self, frame_bgr):
        """Chamado pelo loop: guarda uma CÓPIA do frame anotado p/ o stream."""
        with self._lock:
            self._frame = frame_bgr

    def frame_bgr(self):
        """Chamado pelo servidor: último frame (ou None se ainda não há)."""
        with self._lock:
            return self._frame

    # ---- estado (websocket) ----
    def publicar_estado(self, d):
        with self._lock:
            self._estado = d

    def estado(self):
        with self._lock:
            return dict(self._estado)

    # ---- comandos (web -> engine) ----
    def enviar_comando(self, cmd):
        """Chamado pelo servidor (thread do web): empilha um comando."""
        self.comandos.put(cmd)

    def proximo_comando(self):
        """Chamado pelo loop do engine: próximo comando ou None (não bloqueia)."""
        try:
            return self.comandos.get_nowait()
        except queue.Empty:
            return None


# Instância única usada por todos.
ESTADO = EngineEstado()
