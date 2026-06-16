#!/usr/bin/env python3
"""09_ik_lab.py — Bancada de Cinemática Inversa (IK), SEM o braço.

Esta é a Etapa 0 do caminho para o "braço todo via IK". Ela NÃO liga o braço:
roda só na sua máquina, com o modelo Pinocchio carregado do URDF do B601-dm.
O objetivo é ENTENDER a IK e preparar a Etapa 1 (pescoço fixo), respondendo:

  1) Qual a geometria/limites do braço e quanto a IK nos dá de faixa angular?
     (a "cura" da limitação das 2 juntas do punho — joint1 sozinha dá ±160°.)
  2) Como mapear o nosso servo visual (pan, tilt) numa ORIENTAÇÃO-alvo da câmera,
     mantendo a ponta num PONTO FIXO no espaço ("pescoço fixo")?
  3) A IK é rápida o bastante para rodar a cada frame? (com "warm-start", ela parte
     do q atual + alvo pertinho → tem que convergir em 1-2 iterações.)

Conceito-chave do "pescoço fixo" (modelo PIVÔ/ÓRBITA):
  A bancada nos ensinou que fixar a câmera num ponto ABSOLUTO limita o pan a ~±60°
  (pra girar parado no mesmo ponto o braço se contorce e esgota as juntas). O modelo
  que dá a faixa grande é o de uma CABEÇA NUM PESCOÇO: a câmera orbita um ponto-pivô
  fixo (o "pescoço") num raio fixo, sempre mirando RADIALMENTE pra fora. Isso dá
  ~±120° de pan (a joint1 varre o arco) e ~±40° de tilt.

  A cada frame entregamos à IK uma pose-alvo SE(3) = (posição, orientação):

      R_alvo = Rz(pan) · Ry(tilt) · R0                  (gira a mira)
      p_alvo = c + r · (Rz(pan) · Ry(tilt) · eixo_óptico)   (orbita o pivô c, raio r)

  onde c (pivô) e r (raio) saem da pose-home, e eixo_óptico é o eixo do end_link que
  "olha pra frente" (chute = +X; confirma-se no hardware). Como nosso servo é de
  MALHA FECHADA (corrige o erro em pixels via auto-calibração — teclas k/x/y do 08),
  a geometria do pivô não precisa ser exata: basta a câmera apontar pra fora.

Uso:
    .venv/bin/python 09_ik_lab.py            # roda tudo (info + envelope + latência)
    .venv/bin/python 09_ik_lab.py --pan 30 --tilt -10   # testa uma mira específica
"""

import argparse
import os
import sys
import time

import numpy as np

# Mesmo padrão do 08_seguir.py: aponte com REBOT_ARM_REPO ou ajuste o caminho.
ARM_REPO = os.environ.get("REBOT_ARM_REPO",
                          os.path.expanduser("~/GITHUB/reBotArm_control_py"))
sys.path.insert(0, ARM_REPO)

import pinocchio as pin  # noqa: E402
from reBotArm_control_py.kinematics import (  # noqa: E402
    load_robot_model,
    compute_fk,
    get_joint_names,
    get_end_effector_frame_id,
)
from reBotArm_control_py.kinematics.inverse_kinematics import (  # noqa: E402
    solve_ik,
    pos_rot_to_se3,
    IKParams,
)


# ───────────────────────── parâmetros da bancada ──────────────────────────────

# Pose-home (graus por junta). É a postura onde a câmera "te encara" e a partir da
# qual ela gira. Na Etapa 1, no hardware, isto será CAPTURADO flutuando o braço com
# a mão; aqui usamos um chute razoável para estudar a IK em torno dele.
HOME_DEG = [0.0, -60.0, -60.0, 0.0, 0.0, 0.0]

# Qual coluna de R0 (eixo do frame end_link, no mundo) é o eixo ÓPTICO da câmera
# (pra onde ela "olha"). Chute = 0 (eixo X). Confirma-se no hardware com a auto-cal.
EIXO_OPTICO_COL = 0

# Parâmetros do solver para uso "em tempo real" (warm-start, alvo perto → poucas
# iterações). tolerance 1e-3 ≈ 1 mm / 1 mrad, bom o suficiente para mirar.
IK_RT = IKParams(max_iter=200, tolerance=1e-3, step_size=0.5, damping=0.01)

GRAUS = np.pi / 180.0


# ───────────────────────────── helpers ────────────────────────────────────────

def carrega():
    """Carrega o modelo e devolve (model, data, frame_id da ponta, nomes)."""
    model = load_robot_model()
    data = model.createData()
    fid = get_end_effector_frame_id(model)
    nomes = get_joint_names(model)
    return model, data, fid, nomes


def orientacao_alvo(R0: np.ndarray, pan_deg: float, tilt_deg: float) -> np.ndarray:
    """R_alvo = Rz(pan) · Ry(tilt) · R0  — gira a home por pan/tilt nos eixos do mundo.

    rpyToMatrix(r, p, y) = Rz(y)·Ry(p)·Rx(r); então passamos roll=0, pitch=tilt,
    yaw=pan para obter Rz(pan)·Ry(tilt), e multiplicamos pela orientação-home.
    """
    R_giro = pin.rpy.rpyToMatrix(0.0, tilt_deg * GRAUS, pan_deg * GRAUS)
    return R_giro @ R0


def dentro_dos_limites(model, q) -> bool:
    """True se q respeita os limites de junta do URDF (ignora limites infinitos)."""
    lo, hi = model.lowerPositionLimit, model.upperPositionLimit
    for i in range(model.nq):
        if np.isfinite(lo[i]) and q[i] < lo[i] - 1e-6:
            return False
        if np.isfinite(hi[i]) and q[i] > hi[i] + 1e-6:
            return False
    return True


def pivo_e_raio(p0, R0):
    """Deriva o pivô c e o raio r da pose-home: a câmera fica a r do pivô, ao longo
    do eixo óptico. c = p0 − r·eixo_óptico, com r = |p0| (pivô perto da base)."""
    opt = R0[:, EIXO_OPTICO_COL]
    r = float(np.linalg.norm(p0))
    c = p0 - r * opt
    return c, r, opt


def pose_alvo(modelo, p0, R0, c, r, opt, pan, tilt):
    """Pose-alvo SE(3) para a mira (pan, tilt). 'fixo' = ponto absoluto; 'orbit' =
    câmera orbita o pivô c (mira radial)."""
    R = orientacao_alvo(R0, pan, tilt)
    if modelo == "fixo":
        p = p0
    else:  # orbit
        R_giro = pin.rpy.rpyToMatrix(0.0, tilt * GRAUS, pan * GRAUS)
        p = c + r * (R_giro @ opt)
    return pos_rot_to_se3(p, R), p


def resolve(model, data, fid, geom, pan, tilt, q_seed, modelo="orbit"):
    """Resolve a IK para a mira (pan, tilt), partindo de q_seed.

    geom = (p0, R0, c, r, opt). Devolve (result, dt_ms, p_alvo).
    """
    p0, R0, c, r, opt = geom
    alvo, p = pose_alvo(modelo, p0, R0, c, r, opt, pan, tilt)
    t0 = time.perf_counter()
    result = solve_ik(model, data, fid, alvo, q_seed.copy(), IK_RT)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return result, dt_ms, p


# ───────────────────────────── seções do estudo ───────────────────────────────

def secao_info(model, fid, nomes, q_home, geom):
    p0, R0, c, r, opt = geom
    print("=" * 64)
    print("  1) GEOMETRIA E LIMITES DO BRAÇO")
    print("=" * 64)
    print(f"  modelo: {model.name}   |   juntas (nq={model.nq}): {nomes}")
    print("  (nota: 'join3' é um typo do próprio URDF — é a joint3/cotovelo)\n")
    print("  limites de junta (graus):")
    for i, n in enumerate(nomes):
        lo = np.degrees(model.lowerPositionLimit[i])
        hi = np.degrees(model.upperPositionLimit[i])
        print(f"    {n:8s}: [{lo:+7.1f}, {hi:+7.1f}]")
    print()
    print(f"  POSE-HOME usada na bancada (graus): {HOME_DEG}")
    print(f"    posição da câmera p0 (m)   : [{p0[0]:+.3f}, {p0[1]:+.3f}, {p0[2]:+.3f}]")
    rpy0 = np.degrees(pin.rpy.matrixToRpy(R0))
    print(f"    orientação-home rpy (graus): [{rpy0[0]:+.1f}, {rpy0[1]:+.1f}, {rpy0[2]:+.1f}]")
    print(f"    eixo óptico (chute, col {EIXO_OPTICO_COL}): {np.round(opt, 2)}")
    print(f"    PIVÔ c (m): [{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}]   RAIO r: {r:.3f} m\n")


def secao_envelope(model, data, fid, q_home, geom):
    print("=" * 64)
    print("  2) ENVELOPE ANGULAR — até onde a IK consegue MIRAR")
    print("=" * 64)
    print("  (varre pan/tilt; marca onde a IK converge E respeita os limites)")
    print("  FIXO = ponto absoluto  |  ORBIT = câmera orbita o pivô (o que usaremos)\n")

    vals = np.arange(-170, 171, 10)
    for modelo in ("fixo", "orbit"):
        for eixo in ("pan", "tilt"):
            bons = []
            for v in vals:
                pan, tilt = (v, 0.0) if eixo == "pan" else (0.0, v)
                res, _, _ = resolve(model, data, fid, geom, pan, tilt, q_home, modelo)
                if res.success and dentro_dos_limites(model, res.q):
                    bons.append(v)
            faixa = (f"de {min(bons):+.0f}° a {max(bons):+.0f}°  ({len(bons)} pts)"
                     if bons else "nenhum ponto convergiu")
            print(f"  {modelo.upper():5s} {eixo.upper():4s}: {faixa}")
        print()
    print("  comparação: hoje (2 juntas do punho) usamos ~±40° de pan.\n")


def secao_latencia(model, data, fid, q_home, geom):
    print("=" * 64)
    print("  3) LATÊNCIA DA IK — cabe no orçamento de cada frame?")
    print("=" * 64)
    print("  (simula o tempo real: warm-start do q anterior + passos pequenos de mira)\n")

    # Cold-start: do home, alvo a 20° (como um salto grande).
    res, dt, _ = resolve(model, data, fid, geom, 20.0, -10.0, q_home)
    print(f"  COLD (home → 20°,-10°): {dt:6.2f} ms  "
          f"iters={res.iterations:4d}  ok={res.success}  err={res.error:.1e}")

    # Warm-start: passinhos de 1° (como o servo faz frame a frame).
    q = q_home.copy()
    pan, tilt = 0.0, 0.0
    tempos, iters = [], []
    for _ in range(200):
        pan += 1.0 * np.sign(np.sin(time.perf_counter()))  # zigue-zague pequeno
        pan = float(np.clip(pan, -40, 40))
        res, dt, _ = resolve(model, data, fid, geom, pan, tilt, q)
        if res.success:
            q = res.q.copy()
        tempos.append(dt)
        iters.append(res.iterations)
    tempos = np.array(tempos)
    print(f"  WARM (passos de ~1°, 200x): média {tempos.mean():5.2f} ms  "
          f"p95 {np.percentile(tempos, 95):5.2f} ms  máx {tempos.max():5.2f} ms")
    print(f"                              iterações média {np.mean(iters):.1f}")
    orcamento = 1000.0 / 24.0  # ~24 FPS
    print(f"\n  orçamento por frame @24 FPS ≈ {orcamento:.1f} ms. "
          f"{'✅ folgado' if tempos.mean() < orcamento * 0.3 else '⚠️ revisar'}\n")


def secao_teste_unico(model, data, fid, q_home, geom, pan, tilt, nomes):
    p0 = geom[0]
    print("=" * 64)
    print(f"  TESTE DE MIRA ÚNICA (modelo ORBIT): pan={pan:+.1f}°  tilt={tilt:+.1f}°")
    print("=" * 64)
    res, dt, p_alvo = resolve(model, data, fid, geom, pan, tilt, q_home)
    print(f"  convergiu: {res.success}   iters: {res.iterations}   "
          f"erro: {res.error:.2e}   tempo: {dt:.2f} ms")
    print(f"  dentro dos limites: {dentro_dos_limites(model, res.q)}\n")
    print("  ângulos de junta da solução (graus):")
    for n, deg in zip(nomes, np.degrees(res.q)):
        print(f"    {n:8s} = {deg:+8.2f}")
    # Confere: a FK da solução bate com a pose-alvo pedida?
    pos, _, _ = compute_fk(model, res.q)
    print(f"\n  conferência FK → câmera em (m): {np.round(pos, 3)}   "
          f"alvo pedido: {np.round(p_alvo, 3)}")
    print(f"  a câmera ORBITA o pivô: deslocou {np.linalg.norm(pos - p0) * 1000:.0f} mm "
          "da home (esperado > 0).\n")


# ──────────────────────────────── main ────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Bancada de IK (sem o braço).")
    ap.add_argument("--pan", type=float, default=None, help="testa uma mira: pan (graus)")
    ap.add_argument("--tilt", type=float, default=0.0, help="testa uma mira: tilt (graus)")
    args = ap.parse_args()

    model, data, fid, nomes = carrega()
    q_home = np.radians(HOME_DEG)
    p0, R0, _ = compute_fk(model, q_home)
    c, r, opt = pivo_e_raio(p0, R0)
    geom = (p0, R0, c, r, opt)

    print()
    if args.pan is not None:
        secao_teste_unico(model, data, fid, q_home, geom, args.pan, args.tilt, nomes)
        return

    secao_info(model, fid, nomes, q_home, geom)
    secao_envelope(model, data, fid, q_home, geom)
    secao_latencia(model, data, fid, q_home, geom)
    print("Pronto. Próximo passo (Etapa 1): trocar o núcleo do 08 por POS_VEL + IK,")
    print("segurando esta home e mirando com envelope pequeno e tracking OFF.\n")


if __name__ == "__main__":
    main()
