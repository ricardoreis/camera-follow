# Plano — A "Criatura": percepção avançada → comportamentos

> Documento **vivo**: planejamento + **registro dos labs** (o que testamos/medimos) +
> **catálogo de funcionalidades** (brainstorm p/ consulta, mesmo o que não for implementado).
> Companheiro de [PLANO_VISAO.md](PLANO_VISAO.md) (labs de visão) e da [DOCUMENTACAO.md](DOCUMENTACAO.md).

---

## 1. Visão

Transformar o braço (que hoje segue o rosto) numa **criatura** que percebe muito mais e
reage: reconhece pessoas, lê corpo/mãos/expressão, segue a pessoa de forma robusta,
imita, responde a gestos e tem "vida própria". Objetivo: **poderoso, amigável, capaz,
surpreendente e útil** — uma criatura que impressione.

## 2. Decisões deste planejamento (16–18/06/2026)

- **Fazer tudo**, mas **começar pela fundação: TRACKING ROBUSTO DA PESSOA** (rosto + corpo +
  identidade) — é o que sustenta espelhar, mãos, etc.
- **Multi-pessoa:** implementar várias políticas de foco (atuam em contextos diferentes);
  **inicialmente segue a mais próxima/maior**.
- **Áudio (voz/som):** entra **em breve** — planejar junto (módulo separado).
- **Estilo:** **as duas coisas** — autônoma (vida própria) **e** comandável (responde a gestos).
- **Ambiente:** os modelos de visão (MediaPipe, InsightFace, onnxruntime-openvino,
  hsemotion-onnx) **convivem** com o controle do braço (pinocchio + Seeed) e numpy 2.4.6
  (provado no `.venv-labs`). A app integrada usará esse conjunto (consolidar o venv).

---

## 3. Registro dos LABS (o que já testamos — documentação)

Labs em arquivos `lab_*.py`, no venv isolado **`.venv-labs`** (não toca no `.venv` do
braço). Modelos baixam sob demanda (`models_labs/`, `~/.insightface`); logs em `logs_labs/`.

### Lab 1 — Corpo + Mãos (`lab_pose.py`)  ✅ validado
- **MediaPipe Pose** (33 pts) + **Hands** (21 pts/mão), Tasks API, CPU.
- Deriva: presença/postura (em pé/sentado), distância (ombros px), **olhar aprox.**
  (cima/baixo) + roll, mãos + **dedos levantados** + handedness.
- Toggles `b`/`m` (liga/desliga corpo/mãos p/ ver fps). Acorda+flutua o braço.
- **Medido:** ~**30fps** só corpo; **~22fps** corpo+mãos (pose ~13ms + mãos ~12–23ms).

### Lab 2 — Identidade + contagem (`lab_identidade.py`)  ✅ validado
- **InsightFace** (SCRFD/RetinaFace + **ArcFace** 512-d) — detecta TODOS os rostos
  (**conta pessoas**), **idade/gênero**, e **reconhece** quem foi cadastrado (similaridade
  de cosseno). Cadastro **persistente com nome** (`cadastros_identidade.json`).
- Modelo `buffalo_s` + `allowed_modules` (sem os 2 de landmark) p/ velocidade.
- **Benchmark CPU × OpenVINO (Intel 258V), 1 rosto:**
  | dispositivo | latência/face | fps |
  |---|---|---|
  | onnxruntime-CPU (`cpu`) | ~32 ms | ~21 |
  | **OpenVINO-CPU (`ovcpu`)** | **~9 ms** | **~30** |
  | `gpu`/`npu` | ~32 ms | ~21 (caiu no CPU) |
  - **OpenVINO-CPU = ~3,5× de graça** (virou o padrão). **iGPU/NPU** precisam de **drivers
    Intel no Linux** (sudo/reboot) — **adiado**; só compensam ao empilhar modelos pesados.
  - `buffalo_l` (preciso) ~600ms/rosto em CPU = 2fps → usar só async/baixa freq.

### Lab 3 — Emoção/Expressão + pose da cabeça (`lab_emocao.py`)  ✅ validado
- **MediaPipe Face Mesh** (478 pts) → **blendshapes** (sorriso/surpresa/franzir/piscar) +
  **matriz de pose da cabeça** (pitch/yaw/roll **preciso**). **HSEmotion** (ONNX) → emoção
  rotulada (8 classes), ~12ms.
- **Medido:** ~**27fps** (mesh ~13ms + emo ~7–14ms).

### Tecnologias escolhidas (resumo)
| capacidade | tech | onde roda | nota |
|---|---|---|---|
| rosto (rápido) | YuNet (já no app) ou Face Mesh | CPU | YuNet p/ tracking; Face Mesh p/ expressão |
| corpo/pose | MediaPipe Pose | CPU | seguir pelo corpo |
| mãos/gestos | MediaPipe Hands | CPU | comandos + seguir a mão |
| identidade | InsightFace ArcFace | **OpenVINO-CPU** | reconhecer/memorizar |
| emoção | HSEmotion (ONNX) | CPU | rótulo de emoção |
| expressão/pose cabeça | Face Mesh blendshapes/matriz | CPU | espelhar |

---

## 4. Catálogo de funcionalidades (brainstorm — p/ consulta)

Legenda: 🟢 no roadmap · 🔵 ideia (talvez) · ⚪ depende de áudio/futuro.

### Tracking & presença
- 🟢 **Tracking robusto da pessoa** — funde rosto+corpo+ID; segue pelo corpo quando a
  cabeça sai (senta/levanta), persiste de costas/andando, **re-ID** p/ não trocar de alvo.
- 🟢 **Contagem de pessoas** + reação a grupo/plateia.
- 🟢 **Políticas de foco** (multi-pessoa): mais próxima · conhecida (você) · "ativa"
  (se move/acabou de chegar) · alternar o olhar entre todos.
- 🔵 **Detecção de "saiu/chegou"** → acenar adeus quando você sai; animar quando chega.
- 🔵 **Proxêmica** — chega perto se você está longe/quieto; recua se você chega demais.

### Espelhar / imitar
- 🟢 **Espelhar pose da cabeça** (pitch/yaw/roll) — espelho curioso.
- 🟢 **Espelhar expressões** — você inclina→ele inclina; sorri→anima; surpresa→arregala.
- 🔵 **Atenção conjunta (gaze)** — segue pra onde **você olha** (íris do Face Mesh).
- 🔵 **Imitar gestos de mão** (acena→acena de volta).

### Mãos / comandos
- 🟢 **Comandos por gesto:** 👋 oi · ✋ pare/pausa · 👍 feliz · 👆 apontar→"olha lá" ·
  🤙 vem cá · ✌️/dedos→escolher.
- 🟢 **Modo "siga minha mão"** — o braço acompanha a posição da mão (ativado por gesto).
- 🔵 **Calibrar/ajustar por gesto** (sem teclado).

### Identidade / memória
- 🟢 **Reconhecer** (você vs estranho) + comportamento diferente.
- 🟢 **Memória persistente** — lembra pessoas entre sessões; "há quanto tempo não te via",
  nº de encontros.
- ⚪ **Cumprimentar por nome** (falado — depende de áudio).
- 🔵 **Auto-cadastro** — aprende um rosto novo sozinho ("quem é você?").

### Emoção / humor / vida
- 🟢 **Humor/excitação global** — sobe com sorriso/interação, cai parado; modula
  frequência/amplitude de gestos, respiro, curiosidade.
- 🟢 **Reagir ao estado** — bravo→recua; feliz→anima; entediado→cutuca.
- 🟢 (já existe) **curiosidade · varredura · respirar** — enriquecer com os sinais novos.
- 🔵 **Personalidade/preset de humor** (mais tímido vs extrovertido).

### Social / demos / brincadeiras
- 🔵 **Espelho**, **pedra-papel-tesoura** (gestos!), **siga a mão**, **estátua / quem mexe
  primeiro**, **staring contest**.
- 🔵 **"Modo foto"** — centraliza, conta 3, "tira foto" num gesto.

### Áudio (planejar junto)
- ⚪ **Voz (TTS)** — saudação por nome, falas de reação.
- ⚪ **Sons** — efeitos de emoção (curioso, feliz, surpreso).
- ⚪ **Escuta** (STT/wake-word) — responder a comandos de voz (bem futuro).

---

## 5. Arquitetura de integração (`percepcao.py`)

**Princípio: camadas por frequência** (senão a fps despenca — ver Lab 2/3):

```
THREAD do braço (já existe): controle MIT 500Hz.
THREAD/loop de visão (~30fps) — camada RÁPIDA (por frame):
   • rosto (YuNet) + corpo (Pose) [+ mãos só no modo-mão] → atualiza o ALVO.
   • o ALVO (pan/tilt) alimenta o servo/IK que JÁ existe (não muda o controle).
WORKER assíncrono (~1–2Hz) — camada PESADA (não trava o frame):
   • identidade (ArcFace/ovcpu), emoção (HSEmotion), idade/gênero, pose fina da cabeça.
   • resultados ficam "grudados" na pessoa rastreada (cache).
```

**O ALVO (a pessoa rastreada)** — objeto com: caixa do rosto, centro do corpo, embedding
de identidade, emoção/expressão, última vez visto. Regras:
1. rosto visível → mira no rosto (servo atual).
2. rosto sumiu, corpo visível → mira no **centro do corpo** (não perde quando senta/vira).
3. ambos sumiram → **fuga/varredura** (já existe).
4. rosto novo aparece → **re-ID**: se bate com o alvo, continua; senão, aplica a política
   de foco (inicial: mais próximo).

**Barramento de sinais** (o que a percepção publica p/ os comportamentos consumirem):
`alvo{pan,tilt,dist}, postura, maos[gestos], identidade{nome,conhecido}, emocao,
expressao, head_pose, n_pessoas`. (No app web, isso também vira HUD/terminal/painel.)

**Como pluga no app atual:** a percepção produz o **ponto-alvo** que substitui o "centro
do rosto" de hoje → o servo/IK/pescoço/altura **não mudam**. Comportamentos novos
(espelhar, seguir-mão, comandos) entram como **modos** que modulam o alvo/gestos. Reaproveita
`autonomia_viva` (fuga/varredura), `gestos`, `vida` (curiosidade/respirar), painel web.

**Ambiente:** consolidar as libs de visão no venv da app (validado que convivem). Decidir:
estender o `.venv` do braço **ou** basear a app nova no `.venv-labs`. (Definir na Etapa P1.)

---

## 6. Roadmap de implementação (etapas testáveis)

Cada etapa: arquivos NOVOS quando possível (preservar o que funciona), validar no hardware,
commitar, atualizar este doc + DOCUMENTACAO.

- **P0 — Consolidar ambiente + esqueleto** `percepcao.py`: venv unificado; um loop que roda
  rosto+corpo e publica o ALVO (sem mexer no controle). Medir fps.
- **P1 — TRACKING ROBUSTO DA PESSOA** (fundação): fusão rosto→corpo→fuga + **re-ID** por
  ArcFace (worker async) + política "mais próximo". *Teste:* sentar/levantar/virar de
  costas/andar sem perder; outra pessoa passa e não rouba o alvo.
- **P2 — Mãos:** comandos por gesto (oi/pare/apontar/vem-cá) + **modo "siga minha mão"**.
- **P3 — Espelhar:** cabeça (pose) + expressões (head-tilt/anima ao sorrir).
- **P4 — Identidade/memória:** conhecido vs estranho, memória persistente, saudação (tela).
- **P5 — Emoção + HUMOR global:** modula vida/gestos; reações ao estado.
- **P6 — Atenção/gaze, plateia (políticas de foco), proxêmica.**
- **A — ÁUDIO:** módulo de voz (TTS) + sons; saudação por nome falada.
- **Demos/brincadeiras:** espelho, pedra-papel-tesoura, etc. (quando der).

## 7. Riscos / decisões técnicas pendentes
- **Orçamento de fps:** rosto+corpo todo frame ≈ 22fps; talvez rodar **corpo em baixa
  freq** ou só quando o rosto some. Ajustar na P0/P1.
- **Re-ID custo:** ArcFace por rosto (~9ms ovcpu) — rodar só quando um rosto aparece/some,
  não todo frame.
- **Segurança:** o "siga a mão" move o braço por um alvo externo — manter os clamps/limites
  e o pouso suave; talvez exigir um gesto p/ ativar.
- **iGPU/NPU:** adiado (drivers de sistema); revisitar se a P5/P6 pesarem.
- **Áudio:** escolher TTS local (ex.: piper) vs nuvem; latência; idioma PT-BR.

## 8. Verificação (por etapa)
Sim (sem braço) onde der + hardware: cada etapa tem um **teste de aceitação** (acima).
Logs JSONL dos sinais p/ análise. Atualizar as tabelas de fps/latência aqui conforme medir.
