# Plano — Percepção avançada (labs de visão)

## Objetivo

Deixar o braço mais "poderoso/inteligente": testar tecnologias de visão além do YuNet
(que faz só rosto + 5 pontos), em **labs novos** (sem tocar no que existe), medir o que
roda bem **em CPU, NPU/iGPU (OpenVINO) e — se valer — GPU dedicada**, e então decidir o
que entra na app principal. O usuário quer **entender onde investir tempo/energia/dinheiro**
(inclusive se compensa comprar uma GPU).

## Ordem de prioridade (definida pelo usuário)
1. **Corpo** (pose) → 2. **Identidade** (quem é) → 3. **Mãos** (gestos) → 4. **Emoção/idade/gênero**.
Testar **tudo**, nessa ordem. Para cada um: **CPU primeiro**, depois **NPU/iGPU (OpenVINO)**.

## Hardware do usuário
Zenbook S14 / **Intel 258V (Lunar Lake)**: CPU forte + **iGPU Arc** + **NPU** — **sem
GPU NVIDIA/CUDA**. Local = CPU (`onnxruntime`) ou **OpenVINO** (iGPU/NPU). FP16 ≈ 2× no
NPU/GPU. **MediaPipe** roda em CPU (TFLite/XNNPACK), não acelera bem no NPU Intel → serve
de **baseline CPU**; a comparação CPU↔NPU↔GPU é feita com os modelos **ONNX**.

## Metodologia de benchmark (igual em todos os labs)
Helper comum **`lab_bench.py`**: mede **fps** e **latência por estágio** (ms), overlay
na tela + resumo no fim (e log JSONL reusando `diario.py`). Cada lab roda o mesmo modelo
em **provedores diferentes** quando aplicável (CPUExecutionProvider × OpenVINO CPU/GPU/NPU)
e reporta a tabela. Critérios: **fps**, **latência**, **qualidade/feel** (estabilidade,
acerto), **facilidade de setup**. Saída: uma tabela por capacidade → decisão.

## Os labs (arquivos NOVOS, na raiz; reaproveitam `camera.py`)

### Lab 1 — Corpo / Pose  (`lab_pose.py`)  [prioridade #1]
- **MediaPipe Pose** (33 landmarks): desenha o esqueleto, deriva **em pé/sentado**,
  **perto/longe** (tamanho/!proporção), ombros/braços. Baseline CPU (fps esperado alto).
- Depois: **RTMPose (ONNX)** via `onnxruntime` (CPU) e `onnxruntime-openvino`
  (CPU/iGPU/NPU) → a real comparação CPU↔NPU↔GPU de corpo + precisão vs MediaPipe.
- Uso no robô (futuro): seguir pelo **corpo** quando o rosto some; saber postura/distância.

### Lab 2 — Identidade  (`lab_identidade.py`)  [#2]
- **InsightFace** (RetinaFace + **ArcFace** 512-d; ONNX) — cadastrar "você" (algumas
  fotos/segundos) → reconhecer ao vivo (similaridade de cosseno). Mede CPU × OpenVINO
  (NPU/iGPU). Dá **idade+gênero** de brinde (modelo genderage).
- **face_recognition (dlib)** (128-d) — baseline simples (se instalar fácil).
- Uso no robô: **cumprimentar por nome**, agir diferente com estranhos.

### Lab 3 — Mãos / Gestos  (`lab_maos.py`)  [#3]
- **MediaPipe Hands** (21 pts/mão) + um classificador simples de **gestos** (aceno,
  joinha, palma aberta/"pare", apontar) por regras dos landmarks. CPU.
- Uso no robô: interação **sem toque** (vem/vai/reage ao aceno).

### Lab 4 — Emoção / Idade / Gênero  (`lab_emocao.py`)  [#4]
- **MediaPipe Face Landmarker + blendshapes** (sorriso/surpresa/piscar — de graça, CPU).
- **HSEmotion** (ONNX, emoção leve) e **DeepFace.analyze** (all-in-one) — comparar
  qualidade/velocidade; CPU × OpenVINO onde der.
- Uso no robô: reagir ao **humor** (sorriu → anima; franziu → recua/curioso).

## Onde vale investir (o que os labs vão responder)
- **Quanto o NPU/iGPU acelera** cada modelo ONNX vs CPU (e se a precisão FP16 cai).
- **Se o CPU já basta** (provável p/ MediaPipe e modelos leves) → **não precisa GPU**.
- **Quais modelos são pesados** (RTMPose-large, recog. em multidão, emoção multimodal) →
  aí sim uma **GPU dedicada** (eGPU/desktop) daria salto — os números dirão se compensa.
- Serviço pago por API: só se for **muito** melhor que o melhor local (raro nesses casos).

## Arquitetura (para a futura integração na app)
O loop de tracking é rápido (cada frame). Modelos pesados (identidade/emoção) rodam
**assíncronos, em baixa frequência** (~1–2×/s, thread separada), anotando o rosto já
rastreado — sem travar os 30 fps. Os labs já medem isolado; a integração vem depois.

## Ambiente isolado (NÃO mexer no `.venv` do braço)
As libs de visão (mediapipe, insightface, deepface, tensorflow, onnxruntime…) podem
conflitar com o ambiente do braço (numpy 2.4.6 + opencv + pinocchio + libs da Seeed).
Por isso os labs rodam num venv **separado**: **`.venv-labs`** (gitignored). O `.venv`
principal fica intacto. Os labs só dependem de `camera.py` (OpenCV) — fácil de reusar.

## Dependências (instaladas por lab, sob demanda, no `.venv-labs`)
`mediapipe` (labs 1/3/4), `insightface`+`onnxruntime` (lab 2), `hsemotion`/`deepface`
(lab 4), `onnxruntime-openvino` (comparação NPU/iGPU). Modelos `.onnx`/`.task` baixam na
1ª execução → **gitignore** (não versionar os pesos).

## Verificação / entregáveis
- Cada lab: roda ao vivo (overlay + fps), grava um **resumo** (tabela CPU×NPU×… + feel).
- Ao fim de cada um, decidimos **se/como** entra na app principal (provável via um módulo
  novo `percepcao.py`, com o pesado em thread).
