#!/usr/bin/env python3
"""percepcao.py — camada de PERCEPÇÃO da criatura (P0).

Funde ROSTO (YuNet, já usado no app) + CORPO (MediaPipe Pose) e publica O ALVO (onde
olhar) + sinais de alto nível. É a camada RÁPIDA (por frame). NÃO controla o braço — só
percebe e devolve um estado; quem mira/segue é o servo/IK que já existe.

O ALVO (prioridade):
  1) rosto presente  -> centro dos olhos do maior rosto.
  2) rosto sumiu, corpo presente -> ponto acima dos ombros (onde estaria a cabeça) →
     não perde a pessoa quando ela senta/levanta/vira.
  3) nada -> None (aí o app cai na fuga/varredura).

Próximas etapas (ver PLANO_CRIATURA): re-ID por ArcFace (não trocar de pessoa) e a
camada PESADA assíncrona (identidade/emoção/idade) anotando a pessoa rastreada.
"""

import os
import urllib.request

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from detector import DetectorFaces

MODELOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_labs")
POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
OMB_E, OMB_D = 11, 12
QUAD_E, QUAD_D, JOE_E, JOE_D = 23, 24, 25, 26
VIS_MIN = 0.5


def _baixar(url, nome):
    os.makedirs(MODELOS_DIR, exist_ok=True)
    dst = os.path.join(MODELOS_DIR, nome)
    if not os.path.exists(dst):
        print(f"--- baixando {nome}... ---")
        urllib.request.urlretrieve(url, dst)
    return dst


class Percepcao:
    """Cria os modelos uma vez; chame `processa(frame, ts_ms)` por frame."""

    def __init__(self, com_corpo=True):
        self.detector = DetectorFaces()
        self.com_corpo = com_corpo
        self.pose = None
        if com_corpo:
            mdl = _baixar(POSE_URL, "pose_landmarker_lite.task")
            self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=mdl),
                running_mode=vision.RunningMode.VIDEO, num_poses=1))

    @staticmethod
    def _vis(lms, i):
        return lms[i].visibility >= VIS_MIN

    def _postura(self, lms):
        for q, j in ((QUAD_E, JOE_E), (QUAD_D, JOE_D)):
            if self._vis(lms, q) and self._vis(lms, j):
                return "EM PE" if (lms[j].y - lms[q].y) > 0.22 else "SENTADO"
        return None

    def _dist(self, lms, w):
        if self._vis(lms, OMB_E) and self._vis(lms, OMB_D):
            larg = abs(lms[OMB_E].x - lms[OMB_D].x) * w
            return "PERTO" if larg > w * 0.28 else "LONGE" if larg < w * 0.16 else "media"
        return None

    def processa(self, frame_bgr, ts_ms):
        """Devolve um dict com o ALVO + sinais. `_lms` (pose) sai junto p/ desenho."""
        h, w = frame_bgr.shape[:2]
        faces = self.detector.detectar(frame_bgr, escala=0.5)
        lms = None
        if self.pose is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.pose.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
            lms = res.pose_landmarks[0] if res.pose_landmarks else None

        alvo, fonte, rosto = None, None, None
        if faces:
            rosto = max(faces, key=lambda f: f.area)
            alvo, fonte = rosto.centro_olhos, "rosto"
        elif lms is not None and self._vis(lms, OMB_E) and self._vis(lms, OMB_D):
            cx = (lms[OMB_E].x + lms[OMB_D].x) / 2 * w
            cy = (lms[OMB_E].y + lms[OMB_D].y) / 2 * h
            larg = abs(lms[OMB_E].x - lms[OMB_D].x) * w     # mira acima dos ombros (cabeça)
            alvo, fonte = (int(cx), int(cy - larg * 0.6)), "corpo"

        return {
            "alvo": alvo, "fonte": fonte, "rosto": rosto,
            "n_rostos": len(faces),
            "tem_rosto": bool(faces),
            "tem_corpo": lms is not None,
            "postura": self._postura(lms) if lms is not None else None,
            "dist": self._dist(lms, w) if lms is not None else None,
            "_lms": lms,
        }
