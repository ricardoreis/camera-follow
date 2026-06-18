#!/usr/bin/env python3
"""lab_identidade.py — LAB 2 (IDENTIDADE + CONTAGEM de pessoas).

InsightFace (RetinaFace/SCRFD + ArcFace): detecta TODOS os rostos (conta gente), dá
idade/gênero, e RECONHECE quem você cadastrar (tecla 'c' → digita o nome no terminal),
por similaridade de cosseno. Os cadastros são SALVOS em cadastros_identidade.json (uma
"mini base de dados" local) e recarregados ao abrir. Acorda+flutua o braço p/ enquadrar.

Rodar:  .venv-labs/bin/python lab_identidade.py [dispositivo] [pack]
  dispositivo: cpu (padrão) | ovcpu | gpu | npu     (gpu/npu exigem drivers Intel no Linux)
  pack:        s (padrão, buffalo_s, rápido) | l (buffalo_l, preciso/pesado)

Teclas: c cadastra (digite o nome no terminal) | d apaga o último | x limpa tudo |
        ESPACO trava | f flutua | ESC pousa+sai.
"""

import json
import os
import sys
import time

import numpy as np
import cv2
from insightface.app import FaceAnalysis

import camera
from lab_bench import Medidor, hud, Registro
from lab_braco import Braco

CAMERA = "C920"
LIMIAR = 0.35
DET_SIZE = (480, 480)
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cadastros_identidade.json")


def make_providers(dispositivo):
    """Lista de providers do onnxruntime conforme o dispositivo pedido."""
    d = dispositivo.lower()
    if d == "cpu":
        return ["CPUExecutionProvider"]
    dev = {"ovcpu": "CPU", "gpu": "GPU", "npu": "NPU"}.get(d)
    if dev:
        return [("OpenVINOExecutionProvider", {"device_type": dev}), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def carregar_db():
    if not os.path.exists(DB_PATH):
        return []
    try:
        data = json.load(open(DB_PATH))
        return [(e["nome"], np.asarray(e["emb"], dtype=np.float32)) for e in data]
    except Exception:
        return []


def salvar_db(cadastrados):
    json.dump([{"nome": n, "emb": e.tolist()} for n, e in cadastrados], open(DB_PATH, "w"))


def main():
    dispositivo = sys.argv[1].lower() if len(sys.argv) > 1 else "cpu"
    pack = "buffalo_l" if (len(sys.argv) > 2 and sys.argv[2].lower() == "l") else "buffalo_s"
    providers = make_providers(dispositivo)

    try:
        cap, idx = camera.abrir_camera(CAMERA)
        print(f"--- câmera {CAMERA} (idx {idx}) ---")
    except Exception as e:
        print("!!! sem câmera pelo nome, tentando idx 0:", e)
        cap = cv2.VideoCapture(0)
    janela = "Lab 2 - Identidade (InsightFace)  [c cadastra | ESC sai]"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(janela, 1280, 760)

    braco = Braco(cap, janela)
    braco.iniciar()

    print(f"--- carregando InsightFace ({pack}, {dispositivo}) ---")
    app = FaceAnalysis(name=pack, allowed_modules=["detection", "recognition", "genderage"],
                       providers=providers)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    try:
        prov_real = app.models["detection"].session.get_providers()[0]
    except Exception:
        prov_real = providers[0]

    cadastrados = carregar_db()
    print(f"--- {len(cadastrados)} cadastro(s) carregado(s): {[n for n,_ in cadastrados]} ---")
    med = Medidor()
    reg = Registro("identidade")
    reg.linha(tipo="inicio", lab="identidade", pack=pack, dispositivo=dispositivo,
              provider=prov_real)
    t_log = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            med.frame()
            with med.estagio("face"):
                faces = app.get(frame)
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                       reverse=True)

            pessoas = []
            for f in faces:
                x1, y1, x2, y2 = f.bbox.astype(int)
                nome, sim = "estranho", 0.0
                if cadastrados:
                    emb = f.normed_embedding
                    nm, s = max(((nm, float(np.dot(emb, e))) for nm, e in cadastrados),
                                key=lambda t: t[1])
                    if s > LIMIAR:
                        nome, sim = nm, s
                cor = (120, 235, 120) if nome != "estranho" else (120, 120, 245)
                sexo = getattr(f, "sex", None) or ("M" if int(getattr(f, "gender", 0)) == 1 else "F")
                idade = int(getattr(f, "age", -1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(frame, f"{nome} {sim:.2f}", (x1, max(y1 - 26, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2, cv2.LINE_AA)
                cv2.putText(frame, f"{idade}a {sexo}", (x1, max(y1 - 8, 32)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor, 2, cv2.LINE_AA)
                pessoas.append({"nome": nome, "sim": round(sim, 2), "idade": idade, "sexo": sexo})

            linhas = [
                (f"FPS {med.fps():.0f}   face {med.media_ms('face'):.0f}ms   "
                 f"{pack} @ {prov_real}", (0, 255, 180)),
                (f"PESSOAS NO QUADRO: {len(faces)}    cadastrados: {len(cadastrados)} "
                 f"{[n for n,_ in cadastrados][:5]}", (235, 225, 130)),
                (f"{'FLUTUANDO' if braco.livre else 'TRAVADO'}   c cadastra (nome no terminal) | "
                 "d apaga | x limpa | ESPACO trava | f flutua | ESC sai", (150, 150, 150)),
            ]
            hud(frame, linhas)
            cv2.imshow(janela, frame)

            agora = time.time()
            if agora - t_log > 0.5:
                reg.linha(fps=round(med.fps(), 1), face_ms=round(med.media_ms("face"), 1),
                          n_pessoas=len(faces), pessoas=pessoas)
                t_log = agora

            k = cv2.waitKey(1) & 0xFF
            if k == 27:                           # ESC
                break
            elif k == ord("c") and faces:         # cadastra o maior rosto (nome no terminal)
                emb = faces[0].normed_embedding.copy().astype(np.float32)
                nome = input(">>> Nome da pessoa (ENTER cancela): ").strip()
                if nome:
                    cadastrados.append((nome, emb))
                    salvar_db(cadastrados)
                    print(f"--- cadastrado: {nome}  (total {len(cadastrados)}) ---")
            elif k == ord("d") and cadastrados:   # apaga o último
                rem = cadastrados.pop()
                salvar_db(cadastrados)
                print(f"--- removido: {rem[0]} ---")
            elif k == ord("x"):                   # limpa tudo
                cadastrados.clear()
                salvar_db(cadastrados)
                print("--- cadastros limpos ---")
            elif k == 32:                         # ESPACO trava
                braco.travar()
            elif k == ord("f"):                   # f flutua
                braco.flutuar()
    finally:
        braco.encerrar()
        cap.release()
        cv2.destroyAllWindows()
        reg.fim(resumo=med.resumo())
        print("--- resumo:", med.resumo(), "---")


if __name__ == "__main__":
    main()
