#!/usr/bin/env python3
"""servidor_web.py — servidor FastAPI (numa thread daemon) que expõe o engine:

    GET /            -> o painel React (build em web/dist), se existir
    GET /video       -> stream MJPEG do último frame anotado (multipart)
    WS  /ws          -> envia o ESTADO (~10Hz) e recebe COMANDOS (-> fila do engine)
    GET /api/spec    -> a AJUSTES_SPEC (grupos/itens) p/ o front montar controles

Não toca no braço: só lê de engine_estado.ESTADO e empilha comandos. O loop do
engine (seguir_ik_web.py) é quem drena a fila e aplica.
"""

import asyncio
import os
import threading
import time

import cv2

from fastapi import FastAPI, WebSocket
from fastapi.responses import (StreamingResponse, JSONResponse, HTMLResponse,
                               FileResponse)
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect

from engine_estado import ESTADO

WEB_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "dist")
FPS_VIDEO = 20            # taxa do stream MJPEG
JPEG_Q = 70              # qualidade do JPEG (menor = mais leve)
HZ_ESTADO = 12           # taxa de envio do estado no websocket


def _pagina_fallback():
    """Página simples quando o build do React ainda não existe (web/dist)."""
    return HTMLResponse(
        "<html><body style='font-family:sans-serif;background:#111;color:#eee;"
        "text-align:center;padding-top:60px'>"
        "<h2>Camera Follow — servidor no ar</h2>"
        "<p>O painel React ainda não foi buildado (<code>web/dist</code>).</p>"
        "<p>Vídeo ao vivo:</p><img src='/video' style='max-width:90%;border:1px solid #444'/>"
        "</body></html>")


def criar_app(spec=None):
    """Cria o app FastAPI. `spec` = AJUSTES_SPEC (lista) p/ o endpoint /api/spec."""
    app = FastAPI(title="Camera Follow")

    @app.get("/api/spec")
    def api_spec():
        # serializa a spec (grupo, chave, label, ..., fmt) p/ o front
        itens = [{"grupo": g, "chave": c, "label": l, "passo": p,
                  "min": mn, "max": mx, "fmt": f}
                 for (g, c, l, p, mn, mx, f) in (spec or [])]
        return JSONResponse(itens)

    @app.get("/video")
    def video():
        def gen():
            intervalo = 1.0 / FPS_VIDEO
            while True:
                frame = ESTADO.frame_bgr()
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + buf.tobytes() + b"\r\n")
                time.sleep(intervalo)
        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()

        async def enviar():
            while True:
                await sock.send_json(ESTADO.estado())
                await asyncio.sleep(1.0 / HZ_ESTADO)

        async def receber():
            while True:
                cmd = await sock.receive_json()
                ESTADO.enviar_comando(cmd)

        try:
            await asyncio.gather(enviar(), receber())
        except (WebSocketDisconnect, RuntimeError):
            pass

    # O React build (se existir): os ASSETS com hash são imutáveis (cacheáveis), mas o
    # index.html vai com NO-CACHE — senão o navegador guarda um index antigo que aponta
    # pra um JS de hash já apagado pelo rebuild → script 404 → PÁGINA EM BRANCO.
    if os.path.isdir(WEB_DIST):
        assets = os.path.join(WEB_DIST, "assets")
        if os.path.isdir(assets):
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/")
        def raiz():
            return FileResponse(
                os.path.join(WEB_DIST, "index.html"),
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"})
    else:
        @app.get("/")
        def raiz():
            return _pagina_fallback()

    return app


def iniciar(spec=None, porta=8000):
    """Sobe o uvicorn numa thread daemon. Devolve a thread (não bloqueia)."""
    import uvicorn

    app = criar_app(spec)
    # log_config=None: NÃO deixa o uvicorn reconfigurar o logging. O default dele
    # instancia um formatter que chama sys.stdout.isatty() — e o nosso stdout é o
    # Tee (diario.py), que não tem isatty(). Sem o log_config, evitamos esse crash
    # (e não mexemos no diario.py). access_log=False reduz ruído no terminal.
    config = uvicorn.Config(app, host="0.0.0.0", port=porta,
                            log_config=None, access_log=False)
    server = uvicorn.Server(config)

    th = threading.Thread(target=server.run, daemon=True, name="servidor_web")
    th.start()
    return th
