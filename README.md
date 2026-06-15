# Camera Follow 🤖👁️

Um braço robótico que **encara o rosto de uma pessoa** e a acompanha com a câmera
montada na garra — como uma **cabeça que segue você com o olhar**, com gestos de
"curiosidade" (head-tilt). Movimento **fluido, em tempo real, 100% local e sem IA
remota**.

Projeto de aprendizado e exploração de **visão computacional + controle de robôs**,
construído passo a passo, do básico ao avançado.

![status](https://img.shields.io/badge/status-funcional-success)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

> 📚 **Documentação completa** (contexto, decisões, teoria de controle, aprendizados,
> roadmap): **[DOCUMENTACAO.md](DOCUMENTACAO.md)**

---

## ✨ O que ele faz

- **Segue o rosto** mantendo-o centralizado (a garra "olha no seu olho").
- **Suave e em tempo real** — sem tremor parado, sem "quicar", com predição que
  compensa a latência.
- **Comportamento "vivo"** — micro-movimento ocioso ("respirar") e gestos de
  **head-tilt** (inclinar a cabeça como um cachorro curioso), criáveis e salváveis.
- **Acorda e dorme** suave (sobe à pose neutra ao iniciar, volta ao repouso ao sair).
- **Auto-calibração** — mede sozinho o sentido e a escala da câmera/braço.
- **Interface completa** — HUD colorido, avisos na tela e tela de ajuda com tutoriais.

## 🧰 Tecnologias (todas locais, CPU, sem GPU)

| Camada | Tecnologia |
|---|---|
| Captura | OpenCV + V4L2, MJPG, buffer mínimo (baixa latência) |
| Detecção de rosto | **YuNet** (`cv2.FaceDetectorYN`) — CNN leve com pontos dos olhos |
| Suavização | **One Euro Filter** (anti-tremor adaptativo) |
| Predição | **Filtro de Kalman** (compensa a latência) |
| Controle | **Compensação de gravidade** (Pinocchio) + **visual servoing** proporcional |

## 🦾 Hardware

- Braço **Seeed Studio B601-dm** (6 juntas, motores Damião) via `motorbridge`
  (placa MotorBridge em `/dev/ttyACM0`).
- Webcam **Logitech C920** no punho.
- PC/notebook **Linux x86, sem GPU**.

> ⚠️ O controle do braço usa o pacote `reBotArm_control_py` (Seeed), **não incluído**
> aqui. Aponte com `REBOT_ARM_REPO` ou ajuste o caminho em `08_seguir.py`. As fases
> de **visão (01–05) rodam sem o braço**, só com uma webcam.

## 🚀 Instalação

```bash
git clone https://github.com/ricardoreis/camera-follow.git
cd camera-follow
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export REBOT_ARM_REPO=~/GITHUB/reBotArm_control_py   # repo de controle da Seeed
```

O modelo do YuNet já vem incluído em `models/`.

## ▶️ Como usar

O projeto é organizado em **fases**, das simples às complexas:

| Arquivo | O que faz | Precisa do braço? |
|---|---|---|
| `01_webcam.py` | Webcam ao vivo (FPS, baixa latência) | Não |
| `02_deteccao_faces.py` | Detecção de rostos (YuNet) | Não |
| `04_suavizacao.py` | Laboratório: cru × One Euro × Kalman (+ osciloscópio) | Não |
| `05_rastreador.py` | Rastreamento + erro em graus | Não |
| `07_pose_e_jog.py` | Posiciona o braço (float+lock) e faz jog | **Sim** |
| `08_seguir.py` | **Aplicação completa: a garra te encara + gestos** | **Sim** |

```bash
# só visão (qualquer webcam):
.venv/bin/python 02_deteccao_faces.py --camera C920

# aplicação completa (comece com o braço "sentado"):
.venv/bin/python 08_seguir.py
```

**Primeira vez** (sem config): no `08`, **flutue** o braço com a mão até a pose
neutra (te encarando), **ESPACO** trava, **`k`** calibra, **`n`** salva. Da próxima
vez ele **acorda e segue sozinho**.

## ⌨️ Controles (08_seguir.py)

Tecle **`a`** dentro do app para a ajuda completa (4 páginas com teclas, conceitos e
tutorial). Resumo:

| | |
|---|---|
| **Tracking** | `ESPACO` trava · `t` segue · `f` flutua · `c` recentra · `k` calibra · `n` salva |
| **Ajustes** | `[`/`]` ganho · `o`/`p` zona morta · `-`/`=` limite · `v`/`b` vida · `,`/`.` previsão · `x`/`y` sinais |
| **Head-tilt** | `h` alterna · `j`/`l` fixa lado · `g` swing · `9/0` ângulo · `7/8` velocidade · `4/5` hold |
| **Gestos** | `s`+`1/2/3` salva · `1`/`2`/`3` toca |
| **Autonomia** | `m` curiosidade (head-tilt sozinho) · `u` varredura (olhar ao redor) |
| **Sistema** | `a` ajuda · `i` esconde · `d` log de debug (CSV) · `r` reinicia · `ESC` sai (suave) |

O braço age sozinho: segue seu rosto, **persegue** o canto pra onde você fugiu, fica
**curioso** (head-tilt) quando você para, e **procura** você quando some.

## 🎬 Gravar um gesto de head-tilt

1. Tracking normal como gosta → `n` (salva o "normal").
2. Ajuste o ganho **do gesto** (`]`), e tipo/ângulo/velocidade/hold (`h`/`g`, `9/0`, `7/8`, `4/5`).
3. Salve: `s` → `1`.
4. Volte ao normal: `r`.
5. Toque: aperte `1`. O gesto usa seu próprio contexto e reverte sozinho ao fim.

Os gestos ficam em `gestos.json` (portátil/compartilhável). Detalhes em
[DOCUMENTACAO.md](DOCUMENTACAO.md#11-tutorial-gravar-um-gesto-de-head-tilt).

## 🗺️ Roadmap

🐶 Comportamento vivo (gestos ✅, randomização, busca ao perder alvo) · 🏃 Agilidade
(captura em thread) · 🦾 Braço todo via cinemática inversa · 🎬 Gravação por
braço-líder · 🔔 Gatilhos (mic, objetos) · 🌐 Interface no navegador.

Detalhes em [DOCUMENTACAO.md](DOCUMENTACAO.md#13-próximas-etapas-roadmap).

## 🤝 Contribuindo

Ideias e PRs são bem-vindos! É um projeto de aprendizado — sugestões didáticas
também. Veja o contexto completo em [DOCUMENTACAO.md](DOCUMENTACAO.md).

## 📄 Licença

[MIT](LICENSE)
