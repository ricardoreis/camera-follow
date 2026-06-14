#!/usr/bin/env python3
"""
Utilitários de câmera compartilhados entre as fases do projeto.

A ideia central aqui é NÃO depender do número /dev/videoN (que muda com a
porta USB / boot), e sim de um nome estável da câmera (ex.: "C920"), que o
Linux mantém em /dev/v4l/by-id/ com base em modelo + número de série.
"""

import glob
import os

import cv2

# Diretório onde o Linux cria links estáveis (modelo+serial) para as câmeras.
BY_ID_DIR = "/dev/v4l/by-id"


def listar_cameras():
    """Devolve uma lista legível das câmeras disponíveis (nós de captura).

    Filtramos só os links que terminam em 'video-index0', que são os nós de
    captura de vídeo (os 'index1' são de metadados e não servem pra abrir).
    """
    cameras = []
    padrao = os.path.join(BY_ID_DIR, "*-video-index0")
    for link in sorted(glob.glob(padrao)):
        nome = os.path.basename(link)
        destino = os.path.realpath(link)  # resolve ../../videoN -> /dev/videoN
        cameras.append((nome, destino))
    return cameras


def resolver_indice(seletor):
    """Converte um 'seletor' de câmera num índice inteiro do V4L2.

    'seletor' pode ser:
      - um número (int ou string "6") -> usado diretamente como índice.
      - um texto (ex.: "C920")        -> procurado entre os identificadores
        estáveis de /dev/v4l/by-id/, devolvendo o índice atual daquela câmera.

    É isso que torna o projeto imune à troca de porta USB: damos o nome,
    ele descobre o número certo na hora.
    """
    # Caso 1: já é um número -> usa direto.
    if isinstance(seletor, int):
        return seletor
    if str(seletor).isdigit():
        return int(seletor)

    # Caso 2: é um nome -> procura nos links by-id (apenas nós de captura).
    padrao = os.path.join(BY_ID_DIR, f"*{seletor}*-video-index0")
    candidatos = sorted(glob.glob(padrao))
    if not candidatos:
        disponiveis = "\n  ".join(n for n, _ in listar_cameras()) or "(nenhuma)"
        raise RuntimeError(
            f"Nenhuma câmera com '{seletor}' no nome em {BY_ID_DIR}.\n"
            f"Câmeras disponíveis:\n  {disponiveis}"
        )

    # O link aponta para algo como ../../video6; resolvemos o caminho real
    # e extraímos o número (6) do nome do arquivo (video6).
    destino = os.path.realpath(candidatos[0])           # ex.: /dev/video6
    numero = int("".join(c for c in os.path.basename(destino) if c.isdigit()))
    return numero


def abrir_camera(seletor, width=1280, height=720, fps=30):
    """Resolve o seletor, abre a câmera e a configura para baixa latência.

    Decisões importantes (mesmas da Fase 1, agora centralizadas aqui):
      - CAP_V4L2  : backend nativo de vídeo do Linux.
      - MJPG      : câmera comprime o vídeo antes de enviar -> 30 FPS em HD.
      - BUFFERSIZE=1 : lemos sempre o frame mais recente -> atraso mínimo.

    Devolve uma tupla (cap, indice) para sabermos qual índice foi usado.
    """
    indice = resolver_indice(seletor)
    cap = cv2.VideoCapture(indice, cv2.CAP_V4L2)

    # A ordem importa: formato (MJPG) ANTES da resolução.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(
            f"Não consegui abrir a câmera '{seletor}' (índice {indice}). "
            f"Ela existe e não está em uso por outro programa?"
        )
    return cap, indice


# Permite testar o módulo sozinho:  python camera.py
if __name__ == "__main__":
    print("Câmeras detectadas (nome estável -> dispositivo atual):")
    for nome, destino in listar_cameras():
        print(f"  {nome}  ->  {destino}")
