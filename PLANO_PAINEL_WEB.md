# Plano — Painel Web interativo + separação Engine/Interface

## Contexto

A interface hoje é a janela do OpenCV (`cv2`) + teclado, **fundida** ao laço de visão/IK
em [seguir_ik.py](seguir_ik.py). Queremos um **painel de controle profissional** (mouse,
botões, tooltips) que rode no **navegador** (desktop **e** celular pela rede), e preparar
o terreno pro **app do celular** (cliente remoto). Decisões do usuário:

1. **Vídeo + controles no navegador** (engine streama o vídeo em MJPEG).
2. **Web AO LADO do cv2** (incremental): extrair o "cérebro" (engine); o cv2/teclado
   continua como **reserva**; a web é um 2º cliente.
3. **Front-end: React + Tailwind** (Vite).
4. **MVP primeiro**: vídeo + estado ao vivo + alguns controles (tracking, ganho, gestos
   1‑7), provando o caminho engine⇄web ponta a ponta. Depois expandimos.

## Princípio de segurança (inegociável)

O **engine é autônomo e seguro sozinho** (tracking/hold/pouso suave já existem). A web é um
**controle remoto OPCIONAL**: se o navegador cair ou a rede travar, o braço **continua**
seu comportamento com segurança. Nenhuma ação crítica depende da web estar conectada. Só o
loop do engine comanda o braço; comandos da web entram numa **fila** e são aplicados **na
thread do engine** (sem corrida com a visão/controle).

## Arquitetura

```
LAPTOP — engine (Python, 1 processo, várias threads)
  Thread A (já existe): controle do braço MIT 500Hz (controle_braco.controlador)
  Thread B (laço atual da visão ~30fps): frame -> YuNet -> servo/IK/pescoço/altura
      -> comportamentos -> est["q_target"];  publica ESTADO + FRAME anotado;
      drena a FILA DE COMANDOS (aplica vindos do cv2 OU da web)
  Thread C (novo): servidor web FastAPI/uvicorn (daemon)
      GET /        -> React build (estático)
      GET /video   -> stream MJPEG (último frame anotado)
      WS  /ws      -> envia ESTADO (~10-15Hz) + recebe COMANDOS (-> fila)
        ▲ wifi (rede local)
  navegador / celular — React + Tailwind:
      <img src="/video">  +  websocket (estado/comandos)
      controles: tracking on/off, ganho (slider), gestos 1‑7, estado/erro/IK
```

**Fonte única da verdade** = o estado do engine (`est`, `par`, fase, tracking, gesto,
autonomia). **Boundary de comandos** = uma função `aplicar_comando(cmd)` que faz o que as
teclas/painel fazem hoje; tanto o cv2 quanto a web chamam ela.

## Etapas (incrementais, cada uma testável)

### Web‑0 — Boundary de comandos + estado/fila compartilhados (sem web ainda)
- Extrair as AÇÕES das teclas para `aplicar_comando(cmd: dict)` (closure no `main`, reusa
  `entrar_em_seguir`/`voltar_a_flutuar`/`calibrar`/`tocar_gesto_tipo`/`salvar_tudo`/`par`).
  Comandos: `encara`, `flutua`, `calibrar`, `tracking`(on/off), `recentra`, `set_par`
  (chave,val), `gesto`(tipo), `salvar`, `sentado`. As teclas do cv2 passam a chamar isso.
- `engine_estado.py` (novo): objeto thread-safe com (a) último **frame anotado** (lock),
  (b) último **estado público** (dict JSON-ável: fase, tracking, calibrado, erro, mira,
  ik, auto_estado, gesto, `par`), (c) **fila de comandos** (`queue.Queue`).
- No laço: publicar estado+frame por frame; drenar a fila no topo (aplicar comandos).
- **Teste:** cv2/teclado funciona igual; o estado público é montado sem erro (print/log).

### Web‑1 — Servidor: vídeo + estado (read-only)
- `servidor_web.py` (novo): FastAPI + uvicorn em thread daemon, iniciado pelo `seguir_ik`.
  `/video` = `multipart/x-mixed-replace` (MJPEG) do frame compartilhado (~20fps, JPEG);
  `/ws` = empurra o estado público (~10-15Hz). Porta configurável (ex.: 8000), bind
  `0.0.0.0` (acessível na rede).
- **Overlays no vídeo:** o frame publicado mantém os **feedbacks visuais geométricos**
  (caixa da face, zona morta, mira central, linha de trajetória + círculo do ponto
  previsto) — eles já são desenhados no laço, então vêm embutidos no MJPEG (web e celular
  iguais ao cv2). O **painel de TEXTO** (HUD) **não** é queimado no vídeo da web: vira
  widget HTML (Web‑3), alimentado pelo `/ws`. (Para isso, publicar o frame com os overlays
  geométricos mas SEM o painel de texto — ex.: desenhar o HUD só na cópia do cv2, ou um
  flag `hud=False` no frame da web.) Refino futuro: vídeo "limpo" + coords no `/ws` e
  desenhar overlays no navegador (canvas), permitindo p.ex. clicar pra escolher o rosto.
- **Teste:** abrir `http://IP-do-laptop:8000/video` (imagem) e um WS de teste; ver no
  celular pela rede. cv2 segue rodando em paralelo.

### Web‑2 — Comandos pela web (controle real)
- `/ws` recebe comandos JSON e os põe na fila (`aplicar_comando`). MVP: `tracking`,
  `set_par("ganho", …)`, `gesto(1..7)`. (Ações que movem o braço — `calibrar`/`flutua` —
  ficam pra depois ou atrás de confirmação.)
- **Teste:** togglar tracking, mexer ganho e disparar gestos pelo navegador → braço reage;
  estado reflete na tela.

### Web‑3 — Front React + Tailwind (MVP visual)
- `web/` (Vite + React + Tailwind). Componentes: `VideoView` (`<img>`), `StatusBar`
  (estado do WS), `Controles` (toggle tracking, slider ganho, botões gestos 1‑7).
  Hook `useWebSocket` (recebe estado, manda comando). Visual escuro, "painel de
  instrumento". Build (`npm run build`) → `web/dist`, servido pelo FastAPI em `/`.
- **Teste:** painel bonito no desktop e no celular; controla o braço; reconecta sozinho.

## Arquivos
- **Novos:** `engine_estado.py` (estado+frame+fila), `servidor_web.py` (FastAPI),
  `web/` (React+Tailwind+Vite).
- **Editar:** `seguir_ik.py` — `aplicar_comando` + publicar estado/frame + drenar fila +
  subir a thread do servidor (atrás de uma flag `--web`, p/ não obrigar quem não quer).
- **Dependências:** Python `fastapi`, `uvicorn[standard]` (no `.venv`); Node/npm p/ o
  `web/` (build do React). Adicionar `web/node_modules/` e `web/dist/` ao `.gitignore`.
- **Reusa:** `est`/`par` e os closures de ação já existentes; o frame anotado já é
  desenhado no laço (basta publicá-lo).

## Decisões técnicas (recomendadas)
- **MJPEG** (não WebRTC) no MVP: simples, robusto, funciona em tudo; WebRTC fica pro
  futuro se a latência incomodar.
- **Um processo, várias threads** (não micro-serviços): o servidor lê o estado/frame
  compartilhado e enfileira comandos — simples e seguro.
- **`--web` opcional**: sem a flag, roda como hoje (só cv2). Com a flag, sobe o servidor
  também. Zero regressão.
- **Mesma origem**: o FastAPI serve o React build e o WS/vídeo → sem dor de CORS; o
  celular só abre o IP do laptop.

## Verificação (ponta a ponta)
1. `python seguir_ik.py --web` → cv2 abre normal + servidor sobe (log com a URL).
2. Navegador no laptop em `http://localhost:8000`: vídeo + estado ao vivo; togglar
   tracking, mexer ganho, disparar gestos → braço reage; cv2 reflete o mesmo.
3. Celular no mesmo wifi em `http://IP-do-laptop:8000`: idem (vídeo + controle).
4. Derrubar o navegador → o braço continua seguindo (engine autônomo). Reabrir → reconecta.
5. Rodar **sem** `--web` → comportamento idêntico ao de hoje (regressão zero).

## Fora do escopo do MVP (próximos ciclos)
- Todos os grupos de ajuste no painel (pescoço/altura/comportamentos), com tooltips e
  presets visuais. Gráficos ao vivo (erro/IK). WebRTC. App nativo do celular (cliente do
  mesmo WS). Autenticação/limite de acesso na rede. Extração "dura" do engine pra um
  módulo próprio (hoje fica como boundary leve dentro do `seguir_ik`).
