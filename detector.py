#!/usr/bin/env python3
"""
Detector de faces YuNet - módulo reutilizável entre as fases.

YuNet é uma CNN minúscula e rápida (roda em CPU, sem GPU) que já vem
embutida no OpenCV (cv2.FaceDetectorYN). Não treinamos nada: carregamos um
modelo .onnx pré-pronto (~230 KB) e usamos.

Além da CAIXA do rosto, o YuNet devolve 5 pontos faciais. Para o nosso
projeto, os dois OLHOS são o que importa: a garra vai mirar no ponto entre
eles ("olhar no olho da pessoa").
"""

import os
from dataclasses import dataclass

import cv2

# Caminho do modelo, relativo a este arquivo (funciona de qualquer pasta).
MODELO_PADRAO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_detection_yunet_2023mar.onnx",
)


@dataclass
class Face:
    """Um rosto detectado, com a caixa e os pontos faciais já em coordenadas
    do frame original (pixels inteiros)."""
    x: int
    y: int
    w: int
    h: int
    olho_dir: tuple   # (px, py) do olho direito da pessoa
    olho_esq: tuple   # (px, py) do olho esquerdo da pessoa
    nariz: tuple      # (px, py) da ponta do nariz
    score: float      # confiança da detecção (0..1)

    @property
    def centro_olhos(self):
        """Ponto entre os dois olhos - será o ALVO que a garra persegue."""
        return ((self.olho_dir[0] + self.olho_esq[0]) // 2,
                (self.olho_dir[1] + self.olho_esq[1]) // 2)

    @property
    def area(self):
        """Área da caixa em pixels. Útil para escolher o rosto mais próximo
        (o maior) quando houver várias pessoas."""
        return self.w * self.h


class DetectorFaces:
    """Encapsula o YuNet. Crie uma vez e chame .detectar(frame) por frame."""

    def __init__(self, modelo=MODELO_PADRAO, score=0.8, nms=0.3, top_k=5000):
        if not os.path.exists(modelo):
            raise RuntimeError(
                f"Modelo YuNet não encontrado em {modelo}. "
                f"Baixe o .onnx para a pasta models/."
            )
        # O tamanho de entrada (320,320) é provisório; ajustamos a cada frame
        # com setInputSize, pois o frame pode ter tamanhos diferentes.
        self._det = cv2.FaceDetectorYN.create(
            modelo, "", (320, 320),
            score_threshold=score,   # descarta detecções abaixo dessa confiança
            nms_threshold=nms,        # remove caixas sobrepostas
            top_k=top_k,
        )

    def detectar(self, frame, escala=1.0):
        """Detecta rostos no frame e devolve uma lista de Face.

        'escala' < 1.0 reduz a imagem só para acelerar a detecção; depois
        multiplicamos as coordenadas de volta ao tamanho original.
        """
        if escala != 1.0:
            img = cv2.resize(frame, None, fx=escala, fy=escala,
                             interpolation=cv2.INTER_LINEAR)
        else:
            img = frame

        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(img)

        resultado = []
        if faces is not None:
            inv = 1.0 / escala  # fator para voltar à escala original
            for f in faces:
                # f tem 15 números: x,y,w,h, depois 5 pares (olho dir, olho esq,
                # nariz, canto boca dir, canto boca esq) e por fim o score.
                x, y, bw, bh = (f[0:4] * inv).astype(int)
                olho_dir = tuple((f[4:6] * inv).astype(int))
                olho_esq = tuple((f[6:8] * inv).astype(int))
                nariz = tuple((f[8:10] * inv).astype(int))
                score = float(f[14])
                resultado.append(Face(int(x), int(y), int(bw), int(bh),
                                      olho_dir, olho_esq, nariz, score))
        return resultado
