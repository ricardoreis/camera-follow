# 📚 Camera Follow — Documentação Completa

> Documento-mãe do projeto: contexto, decisões, arquitetura, teoria de controle,
> aprendizados, estado atual e próximos passos. Serve de **estudo pessoal** e de
> **guia para quem quiser usar/contribuir**. Mantido junto com o código.

---

## 1. O que é e qual o objetivo

Um **braço robótico que encara o rosto de uma pessoa** e a acompanha com a câmera
montada na garra — como uma **cabeça que segue você com o olhar**. Movimento
**fluido, em tempo real, 100% local**, sem nuvem e sem IA remota.

É um projeto de **aprendizado e exploração** de visão computacional + controle de
robôs, construído **passo a passo, do básico ao avançado**, com foco em *entender
cada peça* (não só fazer funcionar).

**Objetivo de longo prazo:** que o braço pareça uma **criatura viva** observando
você (gestos de cabeça, curiosidade, reações), e servir de base para novas ideias
(microfone, reconhecimento de objetos, etc.).

---

## 2. Hardware

| Item | Detalhe |
|---|---|
| Braço | **Seeed Studio B601-dm**, 6 juntas, motores **Damião** |
| Comunicação | Placa **MotorBridge** em `/dev/ttyACM0` (ponte serial↔CAN; **não** é SocketCAN) |
| Lib de controle | `motorbridge` (PyPI) + repo `reBotArm_control_py` (da Seeed) |
| Câmera | **Logitech C920** (USB/UVC) montada no punho |
| Computador | Notebook/PC **Linux x86, sem GPU** |

**Juntas usadas para o "olhar" (identificadas empiricamente):**
- **PAN (horizontal) = joint5** — `+` vira p/ direita, `−` p/ esquerda.
- **TILT (vertical) = joint4** — `+` olha p/ baixo, `−` p/ cima.
- **HEAD-TILT (inclinar a cabeça) = joint6** (roll).
- São juntas do **punho** → a câmera gira "no lugar" como um olho, sem balançar o
  braço todo (mais natural e seguro). O braço precisa estar **levantado** (não
  "sentado"), senão um limitador mecânico trava o curso.

> ⚠️ **Dependência externa:** o pacote `reBotArm_control_py` (Seeed) **não** está
> neste repositório. Aponte com a variável de ambiente `REBOT_ARM_REPO` ou ajuste
> o caminho no topo de `08_seguir.py`. As **fases de visão (01–05) rodam sem o
> braço**, só com uma webcam.

---

## 3. Stack de tecnologia (e o porquê de cada escolha)

| Camada | Escolha | Por quê |
|---|---|---|
| Captura | OpenCV + V4L2, **MJPG**, buffer=1 | MJPG → 30 FPS em HD; buffer=1 → menor latência |
| Detecção de rosto | **YuNet** (`cv2.FaceDetectorYN`) | CNN leve, roda em CPU (~7-8ms), dá os **pontos dos olhos**; muito melhor que Haar |
| Suavização | **One Euro Filter** | passa-baixa adaptativo: mata o tremor parado, segue rápido em movimento |
| Predição | **Filtro de Kalman** (vel. constante) | estima velocidade e prevê à frente → compensa latência |
| Controle do braço | **Compensação de gravidade** (Pinocchio) + **visual servoing** proporcional | braço se sustenta sozinho; lei proporcional ancorada (à prova de windup) |

> Sobre "sem IA": a restrição real é **não treinar modelos do zero** e **não usar
> serviços remotos**. Bibliotecas locais que internamente usam CNN (como o YuNet)
> são OK.

---

## 4. Arquitetura do código

```
camera-follow/
├── camera.py          # abre/identifica a câmera por nome estável (by-id), baixa latência
├── detector.py        # YuNet -> objetos Face (caixa + olhos + nariz + score)
├── filtros.py         # OneEuro (1D/2D) + KalmanCV (vel. constante 2D)
├── rastreador.py      # RastreadorAlvo = One Euro -> Kalman (suaviza e prevê)
├── 01_webcam.py       # Fase 1: webcam ao vivo
├── 02_deteccao_faces.py # Fase 2: detecção YuNet
├── 04_suavizacao.py   # Fase 4: laboratório (cru x One Euro x Kalman + osciloscópio)
├── 05_rastreador.py   # Fase 5: rastreamento + erro em graus (FOV)
├── 07_pose_e_jog.py   # Posiciona o braço (float+lock por gravidade) + jog
├── 08_seguir.py       # >>> APLICAÇÃO COMPLETA: a garra te encara + gestos + UI
├── models/face_detection_yunet_2023mar.onnx
├── config_seguir.json # (gitignored) calibração específica do SEU braço/câmera
├── gestos.json        # (compartilhável) presets de head-tilt
├── requirements.txt
├── README.md
└── DOCUMENTACAO.md    # este arquivo
```

O projeto é organizado em **fases** (01→08), das mais simples às complexas. Cada
uma roda sozinha e ensina uma parte. O `08_seguir.py` é o produto final.

---

## 5. Como funciona o controle (o coração do projeto)

**Visual servoing "eye-in-hand":** a câmera está na garra, então *centralizar o
rosto na imagem = apontar a garra para a pessoa*.

Fluxo por frame:
1. **Detecta** o rosto (YuNet) → ponto entre os olhos.
2. **Filtra** (One Euro → Kalman) → ponto suavizado e **previsto** à frente.
3. **Erro** = ponto previsto − centro da imagem (em pixels).
4. **Lei de controle (proporcional, ancorada na posição real):**
   ```
   alvo_junta = posição_atual_da_junta + ganho × erro_angular
   ```
   - O erro em pixels vira erro em **graus** via a **escala calibrada** (px/grau).
   - Ancorar na posição **real** (não acumular) deixa a lei **à prova de windup**.
   - A própria lei **desacelera** perto do alvo (não passa do ponto).
5. **Zona morta:** dentro de um raio no centro, o alvo **congela** → motor firme,
   tremor ~zero.
6. **Base de "hold":** MIT + **compensação de gravidade** (o braço se sustenta).
   Modo *float+lock* (estilo do exemplo 10 da Seeed) pra posicionar com a mão.

**Calibração automática (tecla `k`):** cutuca cada junta ±5°, mede como o rosto se
desloca na imagem e deduz sozinho o **sinal** e a **escala real (px/grau)**. Isso é
*identificação de sistema* — substitui qualquer "chute" de FOV.

---

## 6. Aprendizados de controle (a jornada — vale ouro pro estudo)

Os erros e correções que ensinaram mais:

1. **Windup (efeito elástico):** a 1ª lei era *incremental* (`alvo += ganho×erro`)
   = um integrador. Com o braço "mole" (kp baixo), o alvo crescia além do necessário
   antes do braço chegar → **passava do ponto e quicava**. **Correção:** lei
   **proporcional ancorada na posição real** (`alvo = posição + ganho×erro`), que
   desacelera sozinha e nunca acumula.

2. **Amortecimento mal feito:** tentei um termo derivativo usando a **velocidade da
   junta lida a ~24 Hz** (taxa da câmera) num sistema de 500 Hz. Vinha atrasada/ruidosa
   → **piorava** a oscilação (mais amortecimento = mais bounce!). **Lição:** não use
   um sinal rápido amostrado na taxa lenta para realimentação derivativa. **Removido.**

3. **Rate-limiter saturando:** o limite de passo por frame (`max_step`) baixo
   **saturava** longe do alvo e **removia a frenagem proporcional** → overshoot. Subir
   o `max_step` (1,5°) deixou: longe = rápido, perto = desacelera sozinho.

4. **Integral da gravidade sem `dt`:** o exemplo de compensação somava o integral
   sem escalar pelo tempo → a 500 Hz, ganho efetivo ~100× → **ciclo-limite** (o braço
   oscilava sozinho ao travar). **Correção:** `integ += erro × Ki × dt` (gentil) +
   mais amortecimento (kd). E **desligar o integral durante o tracking** (o laço
   visual já é o integrador).

5. **Pose afeta estabilidade:** quanto **menor o px/grau** (pose "baixa"), mais
   agressivo fica o controle → mais propenso a quicar. Pose mais levantada
   (px/grau maior) rastreia mais calmo. A calibração (`k`) vira um "medidor de
   qualidade de pose".

6. **Tremor de repouso:** resolvido **congelando** o alvo de cada eixo dentro da
   zona morta (não atualiza o alvo quando o erro é pequeno).

7. **GPU não ajuda aqui:** o gargalo é **latência física** (câmera + USB + reação do
   motor), não processamento. YuNet já roda em ~8 ms na CPU. O caminho pra mais
   agilidade é **reduzir latência** (captura em thread), não força bruta.

---

## 7. Comportamentos "vivos" (cabeça/animal)

A meta: parecer uma **criatura** observando, não um braço seguindo. Insight do
usuário: o movimento deve ser de **cabeça** (fluido, com inércia, ignora
micro-movimentos), não de **globo ocular** (rápido e preciso demais).

- **Micro-movimento ocioso ("vida"):** soma de senos lentos de períodos diferentes,
  sutil (<1°), por cima do tracking. Teclas `v`/`b`. *(Obs.: com só 2 juntas parece
  mais ruído que vida; deve melhorar com mais juntas.)*
- **HEAD-TILT (joint6):** inclina a cabeça como um cachorro curioso. **Não-bloqueante**
  (continua te encarando durante). Tipos: `single` (um lado), `swing` (vai-e-vem sem
  parar no centro). Tunável: ângulo/velocidade/hold.
- **Presets de gesto (`gestos.json`):** salve variações em slots (`s`+1/2/3), toque
  por número (1/2/3). Cada preset guarda **tipo+ângulo+velocidade+hold** + um
  **contexto de controle** (ganho/zona/limite/vida/prev) que vale **durante** o gesto
  e **reverte sozinho** ao fim ("valor efetivo", sem salvar/restaurar frágil).
- Durante um gesto o **pan/tilt CONGELA** (a câmera inclina sem perseguir o alvo,
  que sai do lugar por causa do roll do joint6).

### Autonomia (a criatura age sozinha) — máquina de estados

Comportamentos independentes (toggles): **TRACKING** (`t`), **CURIOSIDADE** (`m`),
**VARREDURA** (`u`). Estados: `seguindo` → `perseguindo` → `varrendo` → `ocioso`.

- **Perseguição (parte do tracking):** quando você some, o braço calcula a **direção
  REAL** (proporcional, NÃO o `sign()` — que causava tilt espúrio) de onde seu rosto
  saiu e vai **reto** pra lá por alguns segundos.
- **Curiosidade (`m`):** parado e centralizado por X s (sorteado) → head-tilt
  **sorteado** entre os salvos, com **cooldown** (descanso) aleatório.
- **Varredura (`u`):** se a perseguição não te acha, **olha ao redor** (1-3 ciclos),
  desiste, fica **ocioso** (espera) e tenta de novo.
- **Debug:** painel roxo no HUD mostra estado, timers e aleatoriedade. Tecla **`d`**
  grava um **CSV** (posição do rosto, do alvo previsto, erro, ângulos das juntas por
  frame) para análise — foi assim que diagnosticamos o "movimento perdido".

**Ideias futuras:** faixas (min-max) → randomização pra personalidade; "humor/
excitação" global; **gravação por braço-líder** (teleop, LeRobot); gatilhos de
**saudação/aproximação** (com garra + sons); **detecção de corpo/mãos** como pista
quando a cabeça sai mas o corpo aparece.

---

## 8. Interface (08_seguir.py)

- **HUD colorido** (painel translúcido): modo, calibração, parâmetros, head-tilt,
  gestos, dicas.
- **Avisos na tela (toasts):** feedback de eventos (salvar, calibrar, travar, tocar
  gesto...), colorido por tipo. Não precisa olhar o terminal.
- **Tela de ajuda (`a`):** 4 páginas — Teclas, Ajustes, **Conceitos** (explica
  ganho/zona/limite/vida/prev/calibrar), **Tutorial** (gravar gesto).
- **Retângulo na face** + score; **calibração mostra progresso**; **`i`** esconde tudo.

### Teclas (referência completa)

| Tecla | Função |
|---|---|
| `ESPACO` | travar a pose neutra (saindo do flutuar) |
| `t` | liga/desliga o tracking |
| `f` | flutuar (mover o braço com a mão) |
| `c` | recentrar o olhar na neutra |
| `k` | calibrar sinal + escala |
| `n` | salvar config (pose + calibração + ajustes) |
| `m` | curiosidade on/off (head-tilt automático quando parado) |
| `u` | varredura on/off (olhar ao redor quando te perde) |
| `r` | reiniciar a aplicação (recarrega o código) |
| `i` | esconder/mostrar overlays |
| `d` | gravar/parar log de debug (CSV) |
| `a` | ajuda (cicla 4 páginas) |
| `ESC`/`q` | sair (braço volta suave ao repouso) |
| `[` / `]` | ganho − / + |
| `o` / `p` | zona morta − / + |
| `-` / `=` | limite de ângulo − / + |
| `v` / `b` | vida (micro-movimento) − / + |
| `,` / `.` | previsão (ms) − / + |
| `x` / `y` | inverte sinal pan / tilt |
| `h` | head-tilt (alterna o lado) |
| `j` / `l` | head-tilt forçando esquerda / direita |
| `g` | swing (vai-e-vem) |
| `9` / `0` | ângulo do head-tilt − / + |
| `7` / `8` | velocidade do head-tilt (lento / rápido) |
| `4` / `5` | hold do head-tilt − / + |
| `s` + `1/2/3` | salvar gesto no slot |
| `1` / `2` / `3` | tocar gesto salvo |

---

## 9. Configuração e arquivos de dados

- **`config_seguir.json`** (gitignored — específico do seu braço): pose de repouso e
  neutra, sinais, escala (px/grau), ganho, zona morta, limite, vida. Salvo com `n`.
  Na próxima execução o braço **acorda** sozinho na pose neutra e já segue.
- **`gestos.json`** (versionado — portátil): presets de head-tilt. Salvo com `s`+slot.

---

## 10. Como rodar

```bash
# instalar
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# (controle do braço) apontar o repo da Seeed:
export REBOT_ARM_REPO=~/GITHUB/reBotArm_control_py

# só visão (qualquer webcam, sem braço):
.venv/bin/python 02_deteccao_faces.py --camera C920

# aplicação completa (com braço; comece com ele "sentado"):
.venv/bin/python 08_seguir.py
```

Primeira vez (sem config): no `08`, **flutue** o braço com a mão até uma pose neutra
boa (te encarando), **ESPACO** trava, **`k`** calibra, **`n`** salva. Da próxima vez
ele acorda e segue sozinho.

---

## 11. Tutorial: gravar um gesto de head-tilt

Rode `08_seguir.py` e siga:

1. **Defina o "normal".** Deixe o tracking do jeito que gosta (ex.: ganho `0.15`).
   Se quiser fixar, tecle **`n`** (salva config).
2. **Defina o contexto DO GESTO.** Ajuste o ganho que o gesto deve usar **durante a
   execução** (ex.: **`]`** até ~`0.5`). Se quiser, mexa em zona/limite/vida/prev também.
3. **Esculpa o movimento:**
   - **Tipo:** **`h`** (single, alterna lado) ou **`g`** (swing, vai-e-vem).
   - **Ângulo:** **`9`/`0`** (até 110°). *Maior = mais gracioso.*
   - **Velocidade:** **`7`** (lento) / **`8`** (rápido).
   - **Hold:** **`4`/`5`** (tempo "pensando").
   - Vá apertando **`h`**/**`g`** pra **pré-visualizar** enquanto ajusta.
4. **Salve:** tecle **`s`** → aparece *"SALVAR: tecle 1/2/3"* → tecle **`1`**. Pronto,
   virou o **gesto 1** (guarda o tilt **+** o contexto de controle).
5. **Volte ao normal:** tecle **`r`** (reinicia e recarrega o ganho `0.15`) — ou
   ajuste na mão.
6. **Toque:** aperte **`1`**. Durante o gesto o ganho vira `0.5` (vê
   `[GESTO override]` no HUD), e ao fim **volta sozinho** ao normal. 🎯

### Personalidades pra experimentar

| Slot | Caráter | Receita |
|---|---|---|
| **1** | Curioso / alerta | single, ~50°, rápido (`8`), hold curto |
| **2** | Intrigado / pensativo | single, ~80°, lento (`7`), hold longo (`5`) |
| **3** | Negando / wobble | swing (`g`), ~40°, rápido |

> **Dica:** salve, toque, ajuste, salve de novo no mesmo slot pra refinar. Os gestos
> ficam em **`gestos.json`** (portátil/compartilhável!).

### Como funciona por dentro (resumo)

Cada preset guarda `tipo`, `amp` (ângulo), `sobe` (velocidade), `segura` (hold) e o
**contexto de controle** (`ganho`, `zona_morta`, `limite`, `vida`, `prev`). Ao tocar,
esse contexto vira o **valor efetivo** durante o gesto e **reverte sozinho** quando
ele termina (sem salvar/restaurar manual — robusto a interrupções).

---

## 12. Estado atual (status)

✅ **Objetivo central atingido:** a garra segue o rosto, em tempo real, suave, tudo
local. Funcionam: detecção, suavização, predição, controle em malha fechada,
auto-calibração, acordar/dormir suave, ajustes ao vivo, **head-tilt + presets de
gesto com contexto de controle**, e uma **interface visual completa** (HUD, toasts,
ajuda).

Em andamento: gravar/refinar gestos de head-tilt; polir o "feeling".

---

## 13. Próximas etapas (roadmap)

**Prioridade definida com o usuário** (não pular etapas):
1. 🐶 **Comportamento vivo** — presets de gesto ✅; **autonomia** (curiosidade +
   perseguição + varredura) ✅; refino fino da perseguição (desvio vertical, pose com
   px/deg maior); faixas + "humor" (randomização); micro-movimento melhor (mais juntas).
2. 🏃 **Agilidade (latência)** — captura em **thread** separada, afinar suavização/
   predição. É o teto de fluidez (não é GPU).
3. 🦾 **Movimentação completa** — usar **mais juntas** coordenadas via **cinemática
   inversa** (Pinocchio): controlar a **pose do end-effector** (posição + orientação)
   em vez de juntas isoladas. Inspiração: projeto UR_Facetracking (UR5 + RTDE + IK).
4. 🎬 **Gravação por braço-líder** (teleop) pra gestos orgânicos multi-junta.
5. 🔔 **Gatilhos** — parado X s → gesto; futuramente microfone, objetos.
6. 🌐 **Interface no navegador** (futuro) — ver vídeo, botões, comandos no browser.
7. 📦 **Comunidade** — modo simulação (sem braço), vídeo/GIF no README.

---

## 14. Glossário de conceitos

- **GANHO:** força da correção. Alto = rápido (pode quicar). Baixo = suave/lento.
- **ZONA MORTA:** raio no centro onde o braço **não** se mexe. Maior = cabeça calma.
- **LIMITE:** quanto pan/tilt podem girar a partir da pose neutra (graus).
- **VIDA:** amplitude do micro-movimento ocioso ("respirar"). 0 = estátua.
- **PREVISÃO:** quantos ms à frente o alvo é projetado (compensa latência).
- **ESCALA (px/grau):** quantos pixels o rosto anda na imagem por grau de junta —
  medida pela calibração.
- **WINDUP:** quando um integrador acumula além do necessário → overshoot/oscilação.
- **VISUAL SERVOING:** controlar o robô a partir do erro medido na imagem.
- **COMPENSAÇÃO DE GRAVIDADE:** torque que segura o peso do braço (via modelo
  dinâmico/Pinocchio) → o braço "flutua" e fica firme com pouca rigidez.

---

## 15. Troubleshooting / armadilhas

- **Braço "quica" (efeito elástico):** ganho alto pra latência/pose atual. Baixe o
  ganho (`[`) ou use pose mais levantada (px/grau ≥ 3 na calibração `k`).
- **A garra foge do rosto (vai pro limite):** sinal invertido → tecle `x` (pan) ou
  `y` (tilt). (A calibração `k` resolve sozinho.)
- **Tremor parado:** aumente a zona morta (`p`).
- **Face some quando a cabeça inclina muito:** o YuNet é treinado em rostos em pé;
  com roll grande ele perde. Normal durante o gesto (é rápido).
- **Pose "sentada" trava o tilt:** levante o braço (limitador mecânico).
- **Saída suave só com `ESC`** (não `Ctrl+C`, que desliga o torque na hora).
- **Laptop a 400 MHz após suspender:** bug de PROCHOT do Zenbook — reinicie.

---

## 16. Inspiração estudada

**UR_Facetracking** (robin-gdwl, ~2018, braço UR5): usa Haar + RTDE + cinemática
inversa, controlando a **pose do end-effector** (translação + rotação). **Sacada que
adotamos no roadmap:** comandar a pose e deixar a **IK** mover o braço todo
coordenado (nossa direção #3). Nós já estamos à frente em detecção (YuNet),
suavização (One Euro), predição (Kalman) e auto-calibração — que ele não tem.

---

*Documento vivo — atualizado conforme o projeto evolui.*
