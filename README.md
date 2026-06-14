# Camera Follow 🤖👁️

Um braço robótico que **encara o rosto de uma pessoa** e a acompanha com a câmera
montada na garra — como uma cabeça que segue você com o olhar. Movimento **fluido,
em tempo real, 100% local e sem depender de nuvem/IA remota**.

Projeto de aprendizado e exploração de **visão computacional + controle de robôs**,
construído passo a passo, do básico ao avançado.

![status](https://img.shields.io/badge/status-funcional-success)
![python](https://img.shields.io/badge/python-3.10%2B-blue)

## O que ele faz

A câmera na ponta do braço detecta o rosto mais próximo, calcula o quanto ele está
fora do centro da imagem e move duas juntas do punho (**pan** + **tilt**) para
manter o rosto centralizado — ou seja, a garra "olha no seu olho" e te segue.

## Tecnologias (todas locais, rodam em CPU, sem GPU)

| Camada | Tecnologia |
|---|---|
| Captura | OpenCV + V4L2, MJPG, buffer mínimo (baixa latência) |
| Detecção de rosto | **YuNet** (`cv2.FaceDetectorYN`) — CNN leve embutida no OpenCV, com pontos dos olhos |
| Suavização | **One Euro Filter** (anti-tremor adaptativo) |
| Predição | **Filtro de Kalman** (velocidade constante) para compensar a latência |
| Controle do braço | **Compensação de gravidade** (Pinocchio) + **visual servoing** proporcional |
| Hardware | Braço Seeed **B601-dm** (motores Damião, CAN via MotorBridge) |

## Hardware usado

- Braço robótico **Seeed Studio B601-dm** (6 juntas, motores Damião) controlado pela
  lib [`motorbridge`](https://pypi.org/project/motorbridge/) através do repositório de
  controle [reBotArm_control_py](https://github.com/Seeed-Studio) (placa MotorBridge em `/dev/ttyACM0`).
- Webcam **Logitech C920** montada no punho.
- PC/notebook Linux (x86), sem GPU.

> ⚠️ **Dependência externa:** o controle do braço usa o pacote `reBotArm_control_py`
> (da Seeed), que **não** está incluído aqui. Aponte para ele com a variável de
> ambiente `REBOT_ARM_REPO`, ou ajuste o caminho no topo de `08_seguir.py`.
> As fases de **visão** (01–05) funcionam **sem o braço**, só com uma webcam.

## Instalação

```bash
git clone https://github.com/ricardoreis/camera-follow.git
cd camera-follow
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

O modelo do YuNet (`models/face_detection_yunet_2023mar.onnx`) já vem no repositório.

## Como usar

O projeto está organizado em **fases**, das mais simples às mais complexas.
Cada uma roda sozinha e ensina uma parte:

| Arquivo | O que faz | Precisa do braço? |
|---|---|---|
| `01_webcam.py` | Abre a webcam ao vivo (FPS, baixa latência) | Não |
| `02_deteccao_faces.py` | Detecta rostos com YuNet | Não |
| `04_suavizacao.py` | Laboratório: compara cru × One Euro × Kalman (com osciloscópio) | Não |
| `05_rastreador.py` | Rastreamento final + erro em graus | Não |
| `07_pose_e_jog.py` | Posiciona o braço (float+lock por gravidade) e faz jog | **Sim** |
| `08_seguir.py` | **A aplicação completa: a garra te encara** | **Sim** |

Exemplo (só visão, qualquer webcam):

```bash
.venv/bin/python 02_deteccao_faces.py --camera C920
```

A aplicação principal:

```bash
.venv/bin/python 08_seguir.py
```

### Controles do `08_seguir.py`

Com a config salva, o braço **acorda** sozinho e já começa a seguir. Teclas:

```
ESPACO  trava a pose neutra        t   liga/desliga o tracking
k       auto-calibra sinal+escala  n   salva config (auto-start)
[ / ]   ganho - / +                o / p   zona morta - / +
- / =   limite de ângulo - / +     , / .   predição (ms) - / +
x / y   inverte sinal pan / tilt   c   recentra o olhar
f       flutuar (reposicionar)     r   reiniciar    ESC  sair (suave)
```

## Como funciona o controle (resumo)

1. A câmera está na garra → "centralizar o rosto na imagem" = "apontar a garra para você".
2. O erro em pixels vira erro em **graus** (calibrado automaticamente).
3. Uma lei **proporcional** ancorada na posição real da junta move pan/tilt sem
   acumular erro (à prova de *windup*), com **zona morta** (não treme) e
   desaceleração natural perto do alvo (não passa do ponto).
4. A base de "hold" usa **compensação de gravidade** (o braço se sustenta sozinho)
   e um modo *float+lock* para posicionar com a mão.

## Status e roadmap

✅ Funcional: detecção, suavização, predição, controle em malha fechada, auto-calibração,
acordar/dormir suave, parâmetros ajustáveis ao vivo.

🔜 Ideias: captura em *thread* (mais agilidade), comportamento mais "vivo" (gestos da
garra), re-calibração por pose, múltiplos alvos.

## Contribuindo

Contribuições e ideias são bem-vindas! Abra uma *issue* ou um *pull request*.
Este é um projeto de aprendizado — sugestões didáticas também são ótimas.

## Licença

[MIT](LICENSE)
