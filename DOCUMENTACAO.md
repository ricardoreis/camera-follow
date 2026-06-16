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
├── 08_seguir.py       # >>> APLICAÇÃO COMPLETA (2 juntas): a garra te encara + gestos + UI
├── 09_ik_lab.py       # Bancada de IK (SEM braço): geometria, envelope, latência (Etapa 0)
├── 10_seguir_ik.py    # Etapa 1 via IK — versão MONOLÍTICA (referência; mantida intacta)
├── seguir_ik.py       # >>> Etapa 1 via IK, MODULAR (use esta): laço + teclas + servo/altura
├── mira_ik.py         #   módulo: modelo Pinocchio + IK (geometria, resolver_ik, sinal_altura)
├── controle_braco.py  #   módulo: loop MIT (gravidade) + estado `est` + motores
├── autonomia.py       #   módulo: fuga/perseguição (vai pro lado que você sumiu)
├── gestos.py          #   módulo: perfil do head-tilt (roll no eixo óptico)
├── ui_hud.py          #   módulo: cores, painel, toast, tela sem braço
├── diario.py          #   módulo: log JSONL + Tee do terminal
├── lab_modo.py        # Bancada de CONTROLE (só braço): isola hold/flutuar (MIT vs POS_VEL)
├── lab_pescoco.py     # Bancada GUIADA: valida o pan pelo PUNHO vs BASE (o "pescoço")
├── models/face_detection_yunet_2023mar.onnx
├── config_seguir.json     # (gitignored) calibração do 08 (2 juntas)
├── config_seguir_ik.json  # (gitignored) calibração do 10 (IK: home+repouso+escala+ajustes)
├── logs_ik/           # (gitignored) logs JSONL detalhados do 10/lab (debug por frame)
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

### Aprendizados da Etapa 1 — braço todo via IK (`10_seguir_ik.py`)

Esta etapa rendeu vários aprendizados duros (e caros em tempo) — ficam aqui pra não
se repetirem:

8. **POS_VEL × MIT — a armadilha que mais custou.** A decisão inicial (registrada
   nesta doc) foi *"POS_VEL para seguir + MIT só pra capturar a home"*, só porque o
   exemplo de IK da Seeed (`ArmEndPos`) usa POS_VEL. **Estava errada.** O detalhe fatal
   só apareceu lendo o `arm.py` da lib: **trocar de modo (`mode_mit`/`mode_pos_vel`) é
   BLOQUEANTE ~0,5 s** (junta por junta, com `sleep`) — durante a troca os motores
   ficam **sem comando de sustentação** → o braço **DESPENCA** (e duas threads na
   serial dão *"broken pipe"*). **Solução: ficar SEMPRE em MIT** (hold + seguir +
   flutuar), **nunca trocar de modo** — exatamente o que o `08` já fazia. **Lição
   dupla:** (a) quando há contra-evidência na mão (o `08` MIT-sempre funcionava),
   **questione a decisão registrada**; (b) **isolar o problema** numa bancada mínima
   (`lab_modo.py`, sem câmera/IK/UI) foi o que destravou o diagnóstico — depurar na
   app inteira era às cegas.

9. **Pan tem que nascer da BASE; girar nos eixos do MUNDO trava a IK.** Parametrizar
   pan/tilt como rotação da orientação em torno dos eixos do **mundo** fazia, na home
   flutuada, o "pan" virar rotação do end_link em torno do seu próprio eixo ≈ vertical
   (`Z_link ≈ vertical`) → a IK precisava **virar a configuração** (salto > 15°) → a
   trava de segurança revertia todo frame → **o braço não paneava** (e a calibração
   media ~0 px/deg, porque o nudge de pan não movia nada). **Correção:** **PAN = girar
   a pose-alvo INTEIRA em torno da base** (`p = Rz(pan)·p0`, `R = Rz(pan)·R0·…`) → a IK
   resolve com o **joint1**, suave e 1:1. **TILT = pitch no eixo do CORPO** (body-Y, o
   punho). Validado por simulação em 3 homes: 0 reverts, < 1,5°/frame.

10. **A gravidade carrega o TILT, não o PAN → kp firme só onde pesa.** O pan (joint1,
    base) **não luta contra a gravidade** → kp mole basta (só fica um pouco lento). O
    tilt usa **ombro/cotovelo/punho**, que **seguram o peso do braço** → com kp mole
    (8, herdado do `08`) eles **cediam** → o tilt ficava "elástico" (bounce sem parar,
    parando no teto/chão). **Correção:** **kp FIRME** (os ganhos de fábrica do MIT:
    ~120 nas juntas grandes, ~18 no punho) para **segurar/seguir**, e **kp mole (8) só
    para flutuar** com a mão (escolha automática pelo flag `livre`, sem trocar modo).
    No `08` o tilt era o **punho (leve)**, por isso o problema não aparecia.

11. **A calibração mede o que REALMENTE acontece.** A escala *"pan 0,07 px/deg"* da
    auto-calibração **não era a óptica** — era o pan **travado** (a IK revertia o
    nudge). A auto-calibração (malha aberta, parado) é o melhor diagnóstico de plant:
    número saudável (~15 px/deg) = funcionando; ~0 = algo bloqueando o movimento.

12. **Drop ao desligar = física, não bug.** Quando o ESC desliga o torque, a gravidade
    deixa de ser compensada e o braço cede até **apoiar**. Some-se a isso o torque
    cortar antes de o braço **chegar** ao alvo. **Mitigação:** segurar até assentar
    (esperar `|q−alvo| < ~1°`) antes de desligar, e marcar o "sentado" com o braço
    **apoiado de verdade** (não suspenso).

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

✅ **Objetivo central atingido e superado.** A garra segue o rosto em tempo real,
suave, tudo local. **Funcionando hoje:**
- Detecção (YuNet), suavização (One Euro), predição (Kalman), controle em malha
  fechada (visual servoing proporcional, anti-windup, zona morta que congela).
- **Auto-calibração** (sinal + escala px/grau); **acordar/dormir** suave; **pouso
  suave** ao sentar (segura o torque até chegar); **reinício** completo (`r`).
- **Head-tilt** (single/swing) + **presets de gesto** (`gestos.json`) com contexto de
  controle; durante o gesto o pan/tilt **congela** (não persegue o alvo deslocado).
- **Autonomia:** persegue (linha reta, direção proporcional) o canto pra onde você
  fugiu, fica **curioso** (head-tilt sozinho), **procura** (varredura) e fica
  **ocioso**. Toggles `t`/`m`/`u`. Debug roxo no HUD + **log CSV** (tecla `d`).
- **Interface completa:** HUD colorido, toasts, ajuda em 4 páginas, mensagem amigável
  quando o braço está sem energia.

**Obstáculos já superados (aprendizados):** windup/efeito elástico; amortecimento mal
feito (velocidade amostrada na taxa lenta); saturação do rate-limiter; integral da
gravidade sem `dt` (ciclo-limite); tremor de repouso (congelar na zona morta);
head-tilt enlouquecendo o tracking (roll perde a face → resolvido congelando pan/tilt
no gesto); perseguição com tilt espúrio (era o `sign()` → trocado por direção
proporcional). Ver seção 6.

**Limitação conhecida do `08` (motivou a IK):** com só 2 juntas do punho e escala
~2,5 px/deg, o alcance angular é pequeno (±limite cobre ~200px de tela). Na
perseguição a câmera vai na direção certa **mas não alcança** quem foge pra bem longe.

### Etapa 1 da IK (`10_seguir_ik.py`) — ✅ funcional

O braço **inteiro** segue o rosto via IK ("pescoço fixo"): **pan pela base** (joint1,
faixa ampla) e **tilt pelo punho**, suave e firme, sem despencar no flutuar e subindo
reto ao acordar. Acorda sozinho na pose salva, auto-calibra (`k`), salva config (`n`).
Arquitetura **MIT-sempre**. Detalhes e a jornada (com as armadilhas) na seção 13 e nos
Aprendizados 8–12 (seção 6). **Próximo:** re-plugar autonomia + gestos (Etapa 2).

---

## 13. Próximas etapas (roadmap)

### ➡️ EM ANDAMENTO: 🦾 Braço todo via Cinemática Inversa (IK)

Controlar a **pose do end-effector** (posição + orientação da câmera) com o **braço
inteiro** via IK, em vez de só 2 juntas do punho. **Por que (cura a limitação raiz):**
alcance maior (a busca **realmente alcança**), câmera na **altura dos olhos**, e torna
a detecção de corpo menos necessária.

#### O que já existe no repo do braço (não vamos reescrever IK)

O `reBotArm_control_py` já traz uma stack de IK completa e robusta. **Vamos só
alimentar poses-alvo de câmera.**

- **Controlador de alto nível `ArmEndPos`** (`controllers/arm_endpos_controller.py`):
  - `.start()` → conecta, entra em **modo POS_VEL**, habilita e roda **sozinho um loop
    de 500 Hz** (manda `pos_vel` para um `_q_target` interno).
  - `.move_to_ik(x, y, z, roll, pitch, yaw)` → resolve a IK na hora (CLIK) e seta o alvo.
  - `.move_to_traj(...)` → trajetória **suave** (min-jerk geodésica em SE(3) + CLIK).
  - `.safe_home()` / `.end()` → volta à zero com segurança.
- **Solver CLIK** (`kinematics/inverse_kinematics.py`): mínimos quadrados amortecidos
  (Levenberg-Marquardt, damping adaptativo, line-search, *retry* aleatório, respeita
  limites de junta). Modelo Pinocchio carregado do URDF. Frame da ponta = **`end_link`**.

#### A geometria do B601-dm (medida via Pinocchio)

| Junta | Função | Limite |
|---|---|---|
| **joint1** | **giro da base (pan!)** | **±160°** |
| joint2 | ombro | [−180°, 0°] |
| **join3** *(sic — typo no próprio URDF)* | cotovelo | [−180°, 0°] |
| joint4 | punho (tilt) | [−107°, +90°] |
| joint5 | punho | ±90° |
| joint6 | rotação (roll) | ±180° |

Ponto-chave: hoje o pan/tilt usa **2 juntas do punho** (~±40° úteis). Com a IK, a
**joint1 sozinha dá ±160° de pan** e o conjunto ombro/cotovelo/punho dá uma faixa
grande de tilt — **é a cura da limitação de alcance**. Alcance ~0,26 m à frente,
e dá pra levantar a câmera até ~0,37 m de altura. (`nq=6`, FK em zero → `end_link`
em `(0.26, 0, 0.19)` m.)

#### ⚙️ A arquitetura de controle (CORRIGIDA — ver Aprendizados 8–10)

> **Nota histórica:** a decisão inicial era *"POS_VEL para seguir + MIT só pra
> capturar a home"* (porque o exemplo `ArmEndPos` usa POS_VEL). **Isso se mostrou
> errado** — ver o Aprendizado #8. A arquitetura final é a abaixo.

**SEMPRE em modo MIT + compensação de gravidade** (igual ao `08`), com um **único loop
de 500 Hz que nunca para e nunca troca de modo**. Trocar POS_VEL↔MIT é bloqueante
(~0,5 s) e deixava o braço despencar — então não trocamos. O mesmo loop MIT faz tudo:
- **Segurar/seguir** (`livre=False`): kp **firme** (ganhos de fábrica do MIT, ~120/18)
  → não cede sob gravidade; durante o tracking o integral é zerado (o laço visual é o
  integrador).
- **Flutuar** (`livre=True`): kp **mole** (8) + segue a mão → reposicionar.
O "atuador" no fim do loop deixou de ser 2 juntas e passou a ser **a IK** (q_target das
6 juntas). Toda a visão (servo proporcional, deadzone, predição, calibração) é a mesma.

#### A decisão de design (com o usuário)

Nosso servo calcula um **olhar desejado** (pan/tilt); convertemos isso numa **pose-alvo
SE(3)** e a IK resolve as 6 juntas. Em duas fases:

- **Fase 1 — "pescoço fixo" (FEITA):** a câmera fica num ponto e só **re-mira**.
  Parametrização que funciona (ver Aprendizado #9):
  - **PAN = girar a pose-alvo inteira em torno da BASE** → `p_alvo = Rz(pan)·p0`,
    `R_alvo = Rz(pan)·R0·…` → a IK usa o **joint1** (suave, 1:1, faixa grande).
  - **TILT = pitch no eixo do CORPO** (body-Y, punho) → `R_alvo = …·R0·Ry_body(tilt)`.
  - *(O modelo "órbita" da bancada **falhava na prática**: girar a orientação nos eixos
    do mundo travava a IK quando `Z_link≈vertical`. Por isso a versão final é
    ponto-fixo com pan-pela-base, não a órbita.)*
- **Fase 2 — "pescoço que se estica" (futuro):** a ponta **translada** pra manter a
  altura dos olhos e "esticar" o alcance — exige posição 3D do rosto (direção +
  distância pelo tamanho da caixa) + extrínseco da câmera.

Modo de controle: **MIT-sempre + poses-alvo pra IK** (sem POS_VEL, sem troca de modo).

#### Plano — "pescoço fixo" via IK (em etapas)

- **Etapa 0 — Bancada de IK (`09_ik_lab.py`, SEM braço):** ✅ **feita.** Geometria,
  limites, convenção de mira e latência do solver. *(ver resultados abaixo.)*
- **Etapa 1 — Núcleo MIT + IK (`seguir_ik.py`, modular):** ✅ **FUNCIONAL.** Segue o
  rosto (pan com **pescoço** punho→base + desenrolar, tilt pelo punho) + **acompanha a
  altura**. Suave, firme, com auto-calibração, salvar/acordar e recuperação anti-trava.
- **Etapa 3 (vertical) — altura dos olhos:** ✅ **FEITA**: o braço sobe/desce devagar
  pra re-nivelar o olhar (cascata tilt→altura, simétrica ao pescoço pan).
- **Etapa 2 — autonomia + gestos:** **fuga/perseguição** ✅, **head-tilt** ✅,
  **pescoço (pan punho+base)** ✅. Falta **"dar vida"**: curiosidade (head-tilt sozinho
  quando você fica parado), **varredura** (olhar ao redor quando te perde de vez) e
  micro-movimento "respirar". ← **EM PLANEJAMENTO**

#### 🗂️ Fila de próximos passos (depois do "dar vida")

1. **🦒 Pescoço que se ESTICA — parte 2 (lean/reach):** aproximar/afastar a câmera
   quando você se inclina pra frente/trás (profundidade). Completa o "pescoço que se
   estica" físico. **Precisa estimar a distância** (pelo tamanho do rosto na imagem).
2. **🎭 Personalidade / gestos:** salvar **vários** head-tilts (estilos/presets, tipo
   `gestos.json` do `08`), **randomização** (pra não repetir igual) e um **"humor/
   excitação" global** que modula os movimentos.
3. **🔧 Pendência:** corrigir o bug de **re-engatar o tracking depois do `f`** (flutuar
   → ESPACO → `t` nem sempre reengata) — investigar com log.

#### Etapa 1 — o que a app de IK já faz (`seguir_ik.py`)

> **Arquitetura modular:** a app foi refatorada (lado-a-lado, sem tocar no `10`, que
> fica como referência) em **`seguir_ik.py`** (orquestração: laço + teclas + servo/
> altura) + módulos **`mira_ik.py`** (modelo + IK), **`controle_braco.py`** (loop MIT +
> estado `est` + motores), **`ui_hud.py`** (HUD) e **`diario.py`** (log). O `10`
> monolítico (927 linhas) virou `seguir_ik` (638) + 4 módulos focados. **Use o
> `seguir_ik.py`.**

- **Sempre MIT** (sem troca de modo). **PAN pela base** + **TILT pelo punho**; kp
  **firme** pra seguir / **mole** pra flutuar (ver Aprendizados 8–10).
- **Auto-calibração (`k`)**, malha aberta com você **parado**: cutuca a mira ±8° e mede
  **sinal + escala (px/grau)** reais (~15 px/deg, saudável).
- **Acompanhamento de ALTURA (`h` liga/desliga, `v`/`b` ajusta o alcance):** se o tilt
  se MANTÉM (você ficou mais alto/baixo), o braço **sobe/desce devagar** pra te encarar
  na **altura dos olhos** (cascata "olhar guia a altura", sem estimar profundidade).
  É a 1ª parte da Etapa 3 (só vertical), trazida pra cá. Tilt fica **modesto (±30°)**
  porque a altura cobre o vertical → evita poses extremas.
- **Fuga/perseguição (`u`, módulo `autonomia.py`):** quando o rosto some, vai **reto pro
  lado** que você saiu (alcança longe via base), espera, e re-centraliza ao reaparecer.
- **Head-tilt (`g`, módulo `gestos.py`):** inclina a cabeça pro ombro (roll no eixo
  óptico = joint6) **sem parar de te encarar**; durante o gesto pan/tilt/fuga congelam.
- **"Pescoço" (cascata pan):** pan PEQUENO pelo **PUNHO** (joint5, ~18 px/deg —
  validado em `lab_pescoco.py`); ao saturar, a **BASE** (joint1) assume o resto. E o
  **"desenrolar"**: a base vira devagar enquanto o punho **volta ao 0** (a cabeça
  endireita, o corpo compensa) — a câmera segue te encarando o tempo todo (camera pan
  = punho + base = pan total). Teclas: `9`/`0` faixa do punho · `w`/`e` velocidade do
  desenrolar · `j` inverte sinal. Tudo ajustável ao vivo e salvo no `n`.
- **Salvar/Acordar (`n` / `z`):** salva home + repouso (sentado) + calibração + ajustes
  em `config_seguir_ik.json`. Na próxima vez **acorda na pose** (rampa suave, sobe reto)
  e fica pronto pra seguir (tecle `t`). `z` marca o "sentado" limpo.
- **Segurança:** envelope do pan ajustável; se a IK "virar", **desfaz** o passo (sem
  disparada); se ficar **PRESA** (>8 frames), **recua e re-semeia do home** pra
  destravar; detecção de falha de motor (congela); ESC volta ao repouso segurando até
  assentar.
- **Log JSONL detalhado** (`logs_ik/`): config, teclas, eventos, saída de terminal e
  **telemetria por frame** (rosto/erro, mira, altura, q_alvo e q_real das 6 juntas,
  status dos motores). Foi a ferramenta que permitiu diagnosticar tudo à distância.
- **Teclas:** `ESPACO` trava a home · `k` calibra · `t` segue · `f` flutua · `z` marca
  sentado · `n` salva · `h` altura on/off · `v`/`b` alcance de altura · `o`/`p` zona ·
  `[`/`]` ganho · `-`/`=` envelope · `,`/`.` previsão · `x`/`y` sinais · `c` recentra ·
  `i` esconde · `ESC` sai.
- **Bancada de controle `lab_modo.py`** (só o braço, sem câmera/IK): isola hold/flutuar
  — onde provamos que **MIT-sempre** resolve a queda do `f`. Fica como ferramenta.
- **Pendência conhecida (não-crítica):** depois do `f` (flutuar), voltar ao tracking
  (ESPACO → `t`) nem sempre reengata — a investigar com log.

#### 📊 Resultados da bancada (`09_ik_lab.py`)

A bancada revelou **duas coisas decisivas**:

1. **Latência da IK é irrelevante.** Com *warm-start* (parte do `q` atual + alvo
   pertinho, como o servo faz frame a frame): **~0,03 ms, 1 iteração**. O orçamento por
   frame a 24 FPS é ~42 ms → a IK não é gargalo nenhum.
2. **Fixar a câmera num ponto ABSOLUTO limita o pan a ±60°** (pra girar parada no mesmo
   ponto o braço se contorce e esgota juntas). O modelo que dá a faixa grande é o de
   **cabeça num pescoço** — a câmera **orbita** um pivô fixo (raio fixo), mirando
   radialmente pra fora:

   ```
   R_alvo = Rz(pan)·Ry(tilt)·R0                       (gira a mira)
   p_alvo = c + r·(Rz(pan)·Ry(tilt)·eixo_óptico)       (orbita o pivô c, raio r)
   ```

| Modelo (na bancada) | PAN | TILT |
|---|---|---|
| Hoje (2 juntas do punho) | ~±40° | ~±40° |
| IK ponto absoluto fixo | ±60° | ±40° |
| IK pivô/órbita (teórico) | ±120° | ±40° |

   > ⚠️ **Mas a órbita falhou no hardware** (girar a orientação nos eixos do mundo
   > travava a IK — Aprendizado #9). A **versão final** usa **PAN pela base** (joint1,
   > faixa ampla, suave) + **TILT pelo punho** — não a órbita. O tilt fica em ±40°
   > (limite vertical do braço), suficiente pra seguir um rosto.

> **De-risca a Etapa 1:** como nosso servo é de **malha fechada** (corrige o erro em
> pixels via auto-calibração), a geometria do pivô **não precisa ser exata** — basta a
> câmera apontar pra fora que o servo zera o erro.

**Segurança (Etapa 1):** warm-start a cada frame (sem saltos); se a IK não convergir,
**segura a última pose**; clamp do envelope (começa pequeno, ~±25°); ESC com pouso
suave (`safe_home`/repouso); começa **sentado**. Inspiração conceitual:
UR_Facetracking (UR5 + RTDE + IK, controla pose do end-effector).

### Depois do IK
1. 🧍 **Detecção de corpo/pessoa** — só se ainda fizer falta após o IK (pista quando a
   cabeça sai mas o corpo aparece). **Mãos/objetos** → interação (futuro).
2. 🎭 **Personalidade / "humor" + ÁUDIO** — faixas (randomização) + sons de reação
   junto dos movimentos. (Adiado pelo usuário pra quando entrar áudio.)
3. 🏃 **Agilidade (captura em thread)** — menos latência. Pouco relevante pro
   comportamento de "cabeça" (não é globo ocular); fica pra depois.
4. 🎬 **Gravação por braço-líder** (teleop) pra gestos orgânicos multi-junta.
5. 🔔 **Gatilhos** de saudação/aproximação (com garra + sons).
6. 🌐 **Interface no navegador**; 📦 **modo simulação** (sem braço) pra comunidade.

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
