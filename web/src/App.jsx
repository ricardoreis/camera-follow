import { useEffect, useRef, useState, useCallback, useMemo } from 'react'

/* ─── conexão com o engine (websocket + spec) ─────────────────────────────── */
function useEngine() {
  const [estado, setEstado] = useState(null)
  const [spec, setSpec] = useState([])
  const [conectado, setConectado] = useState(false)
  const wsRef = useRef(null)

  useEffect(() => {
    fetch('/api/spec').then((r) => r.json()).then(setSpec).catch(() => {})
    let vivo = true, timer = null
    const conectar = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws
      ws.onopen = () => setConectado(true)
      ws.onclose = () => { setConectado(false); if (vivo) timer = setTimeout(conectar, 1000) }
      ws.onmessage = (ev) => { try { setEstado(JSON.parse(ev.data)) } catch {} }
    }
    conectar()
    return () => { vivo = false; clearTimeout(timer); wsRef.current?.close() }
  }, [])

  const enviar = useCallback((cmd) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd))
  }, [])

  return { estado, spec, conectado, enviar }
}

/* ─── helpers ─────────────────────────────────────────────────────────────── */
const PANEL_MIN = 400
const GRUPO_LABEL = { TRACKING: 'Tracking', PESCOCO: 'Pescoço', ALTURA: 'Altura', GESTOS: 'Gestos' }
const HINTS = {
  ganho: 'Força da correção. Alto = rápido (pode quicar); baixo = suave e calmo.',
  zona: 'Raio central onde o braço NÃO se mexe (mata o tremor com você parado).',
  limite: 'Quanto o pan pode girar a partir do centro.',
  limite_tilt: 'Quanto o tilt pode inclinar a partir do centro.',
  previsao: 'Mira X ms à frente para compensar a latência.',
  max_step: 'Passo máximo da mira por frame (suavidade).',
  neck_max: 'Quanto o PUNHO paneia antes da BASE entrar (pescoço).',
  neck_relax: 'Velocidade do "desenrolar": a cabeça endireita e o corpo compensa.',
  sinal_neck: 'Sentido do pescoço (inverta se virar para o lado errado).',
  altura_on: 'Sobe/desce o braço para manter você na altura dos olhos.',
  alt_max: 'Alcance vertical do acompanhamento de altura.',
  altura_ganho: 'Velocidade do sobe/desce da altura.',
  altura_zona: 'Tilt mínimo (sustentado) para começar a subir/descer.',
  gesto_tipo: 'Gesto selecionado para editar/testar (cada um tem a sua config).',
  g_amp: 'Amplitude do gesto selecionado.',
  g_vel: 'Velocidade (tempo de subida) do gesto.',
  g_hold: 'Tempo segurando no ápice do gesto.',
  g_curioso: 'Se este gesto entra nas reações automáticas (curiosidade).',
}

function fmtVal(v, fmt) {
  if (v === undefined || v === null) return '—'
  switch (fmt) {
    case 'bool': return v ? 'ON' : 'off'
    case 'sinal': return (v >= 0 ? '+' : '') + v
    case 'px': return `${Math.round(v)} px`
    case 'ms': return `${Math.round(v)} ms`
    case 'deg': return `${Math.round(v)}°`
    case 'degf1': return `${Number(v).toFixed(1)}°`
    case 'cm': return `${Math.round(v * 100)} cm`
    case 'f1ps': return `${Number(v).toFixed(1)}/s`
    case 'f2': return Number(v).toFixed(2)
    case 'f2s': return `${Number(v).toFixed(2)}s`
    case 'f1s': return `${Number(v).toFixed(1)}s`
    case 'tipo': return String(v)
    default: return Number(v).toFixed(2)
  }
}
function valorDe(item, e) {
  if (!e) return undefined
  if (item.chave.startsWith('g_')) return e.gestos?.[e.par?.gesto_tipo]?.[item.chave.slice(2)]
  return e.par?.[item.chave]
}

/* ─── componentes ─────────────────────────────────────────────────────────── */
function Secao({ titulo, children, extra }) {
  return (
    <section className="break-inside-avoid">
      <div className="mb-2 flex items-center justify-between">
        <h3 className="text-[11px] font-semibold uppercase tracking-wider text-gray-400">{titulo}</h3>
        {extra}
      </div>
      {children}
    </section>
  )
}

function Metric({ label, value, ok }) {
  const cor = ok === true ? 'text-emerald-600' : ok === false ? 'text-rose-500' : 'text-gray-800'
  return (
    <div className="rounded-xl border border-gray-200 bg-white px-3 py-2">
      <div className="text-[11px] text-gray-400">{label}</div>
      <div className={`text-sm font-semibold ${cor}`}>{value}</div>
    </div>
  )
}

function MiniBtn({ children, onClick, title }) {
  return (
    <button onClick={onClick} title={title}
      className="flex h-8 w-8 items-center justify-center rounded-lg border border-gray-300 bg-white
                 text-lg font-semibold text-gray-700 hover:bg-gray-100 active:scale-95">
      {children}
    </button>
  )
}

function ParRow({ item, e, enviar }) {
  const v = valorDe(item, e)
  const nudge = (d) => enviar({ cmd: 'nudge', sel: item.sel, d })
  let controle
  if (item.fmt === 'bool') {
    const on = !!v
    controle = (
      <button onClick={() => nudge(1)}
        className={`relative h-6 w-11 rounded-full transition ${on ? 'bg-emerald-500' : 'bg-gray-300'}`}>
        <span className={`absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition ${on ? 'left-[22px]' : 'left-0.5'}`} />
      </button>
    )
  } else if (item.fmt === 'tipo') {
    controle = (
      <div className="flex items-center gap-2">
        <MiniBtn onClick={() => nudge(-1)} title="anterior">‹</MiniBtn>
        <span className="w-20 text-center text-sm font-semibold capitalize text-gray-800">{fmtVal(v, item.fmt)}</span>
        <MiniBtn onClick={() => nudge(1)} title="próximo">›</MiniBtn>
      </div>
    )
  } else {
    controle = (
      <div className="flex items-center gap-2">
        <MiniBtn onClick={() => nudge(-1)} title="diminuir">−</MiniBtn>
        <span className="w-16 text-center text-sm font-semibold tabular-nums text-gray-800">{fmtVal(v, item.fmt)}</span>
        <MiniBtn onClick={() => nudge(1)} title="aumentar">+</MiniBtn>
      </div>
    )
  }
  return (
    <div className="flex items-center justify-between py-1.5">
      <span className="text-sm text-gray-600" title={HINTS[item.chave] || ''}>
        {item.label}
        {HINTS[item.chave] && <span className="ml-1 cursor-help text-gray-300">ⓘ</span>}
      </span>
      {controle}
    </div>
  )
}

function Sparkline({ data }) {
  const W = 320, H = 40
  if (!data.length) return <div style={{ height: H }} />
  const max = Math.max(30, ...data)
  const pts = data.map((v, i) => `${(i / (data.length - 1 || 1)) * W},${H - (v / max) * H}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="h-10 w-full">
      <polyline points={pts} fill="none" stroke="#6366f1" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

// Notificação flutuante: aparece quando chega um toast NOVO (timestamp diferente).
function Toast({ estado }) {
  const [msg, setMsg] = useState(null)
  const lastT = useRef(0)
  useEffect(() => {
    const t = estado?.toast
    if (t && t.t && t.t !== lastT.current) {
      lastT.current = t.t
      setMsg(t)
      const id = setTimeout(() => setMsg(null), 4000)   // some sozinho após 4s
      return () => clearTimeout(id)
    }
  }, [estado])
  if (!msg) return null
  const cor = { ok: 'bg-emerald-600', erro: 'bg-rose-600', aviso: 'bg-amber-500' }[msg.kind] || 'bg-gray-800'
  return (
    <div className={`fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-3 rounded-xl ${cor}
                     px-5 py-2.5 text-sm font-medium text-white shadow-xl ring-1 ring-black/10`}>
      <span>{msg.txt}</span>
      <button onClick={() => setMsg(null)} title="fechar"
        className="-mr-1 flex h-6 w-6 items-center justify-center rounded-md text-white/80 hover:bg-white/20 hover:text-white">
        ✕
      </button>
    </div>
  )
}

/* ─── app ─────────────────────────────────────────────────────────────────── */
export default function App() {
  const { estado, spec, conectado, enviar } = useEngine()
  const e = estado || {}
  const [aberto, setAberto] = useState(true)
  const [width, setWidth] = useState(() => {
    const v = Number(localStorage.getItem('panelW'))
    return v >= PANEL_MIN ? v : PANEL_MIN
  })
  const [arrastando, setArrastando] = useState(false)
  const arrastRef = useRef(false)
  const [hist, setHist] = useState([])

  useEffect(() => { localStorage.setItem('panelW', String(width)) }, [width])

  useEffect(() => {
    if (!estado) return
    const mag = estado.erro ? Math.hypot(estado.erro[0], estado.erro[1]) : 0
    setHist((h) => [...h.slice(-59), mag])
  }, [estado])

  // arrastar a divisória para redimensionar o painel
  useEffect(() => {
    const move = (ev) => {
      if (!arrastRef.current) return
      const max = Math.min(1400, window.innerWidth - 360)   // deixa ao menos 360px de vídeo
      setWidth(Math.min(Math.max(window.innerWidth - ev.clientX, PANEL_MIN), max))
    }
    const up = () => {
      if (arrastRef.current) { arrastRef.current = false; setArrastando(false); document.body.style.userSelect = '' }
    }
    window.addEventListener('mousemove', move)
    window.addEventListener('mouseup', up)
    return () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
  }, [])

  const grupos = useMemo(() => {
    const m = {}
    spec.forEach((it, i) => { (m[it.grupo] ??= []).push({ ...it, sel: i }) })
    return m
  }, [spec])

  const tracking = !!e.tracking
  const tipos = e.tipos || []
  const cols = Math.max(1, Math.min(4, Math.floor(width / 320)))  // 1..4 colunas responsivas

  return (
    <div className="flex h-screen flex-col bg-gray-100 text-gray-900">
      {/* Cabeçalho */}
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-5 py-3">
        <div className="flex items-baseline gap-2">
          <span className="text-xl">🦾</span>
          <h1 className="text-lg font-semibold tracking-tight">Camera Follow</h1>
          <span className="text-sm text-gray-400">painel de controle</span>
        </div>
        <div className="flex items-center gap-4">
          <span className={`flex items-center gap-2 text-xs ${conectado ? 'text-emerald-600' : 'text-rose-500'}`}>
            <span className={`h-2 w-2 rounded-full ${conectado ? 'bg-emerald-500' : 'bg-rose-500'}`} />
            {conectado ? 'conectado' : 'reconectando…'}
          </span>
          <button onClick={() => setAberto((a) => !a)}
            className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50">
            {aberto ? 'Ocultar painel ›' : '‹ Mostrar painel'}
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Vídeo (esquerda, centralizado, responsivo) */}
        <main className="flex min-w-0 flex-1 items-center justify-center p-6">
          <img src="/video" alt="vídeo ao vivo"
            className="max-h-full max-w-full rounded-2xl bg-black object-contain shadow-lg ring-1 ring-gray-200" />
        </main>

        {/* Painel (direita) — drawer redimensionável + colapsável */}
        <aside
          style={{ width: aberto ? width : 0 }}
          className={`relative shrink-0 overflow-hidden border-l border-gray-200 bg-white
                      ${arrastando ? '' : 'transition-[width] duration-300 ease-in-out'}`}>
          {/* alça de redimensionar (divisória) */}
          <div onMouseDown={() => { arrastRef.current = true; setArrastando(true); document.body.style.userSelect = 'none' }}
            title="arraste para redimensionar"
            className={`absolute left-0 top-0 z-10 h-full w-1.5 cursor-col-resize
                        ${arrastando ? 'bg-indigo-400' : 'hover:bg-indigo-300'}`} />

          <div className="flex h-full flex-col pl-1.5" style={{ width }}>
            <div className="flex items-center justify-between border-b border-gray-100 px-5 py-3">
              <span className="text-sm font-semibold text-gray-700">Controles</span>
              <button onClick={() => setAberto(false)} title="fechar"
                className="flex h-7 w-7 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700">✕</button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4">
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap: '1rem', alignItems: 'start' }}>
                {/* Tracking */}
                <Secao titulo="Controle">
                  <div className="flex items-center justify-between rounded-xl border border-gray-200 bg-white px-4 py-3">
                    <div>
                      <div className="font-medium">Tracking</div>
                      {!e.calibrado && <div className="text-xs text-amber-600">calibre no teclado (k) p/ ativar</div>}
                    </div>
                    <button onClick={() => enviar({ cmd: 'tracking', val: !tracking })}
                      className={`rounded-xl px-5 py-2 font-semibold transition ${tracking
                        ? 'bg-emerald-500 text-white hover:bg-emerald-400'
                        : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}>
                      {tracking ? 'ON' : 'OFF'}
                    </button>
                  </div>
                </Secao>

                {/* Detecção */}
                <Secao titulo="Detecção">
                  <div className="grid grid-cols-2 gap-2">
                    <Metric label="Status" value={e.tem_rosto ? 'Detectado' : '—'} ok={e.tem_rosto} />
                    <Metric label="Fase" value={e.fase ?? '—'} />
                    <Metric label="Pan / Tilt" value={`${e.pan ?? 0}° / ${e.tilt ?? 0}°`} />
                    <Metric label="IK" value={e.ik_ok ? `${e.ik_ms} ms` : 'revertido'} ok={e.ik_ok} />
                  </div>
                  <div className="mt-2 rounded-xl border border-gray-200 bg-white px-3 py-2">
                    <div className="mb-1 flex justify-between text-[11px] text-gray-400">
                      <span>erro (px)</span><span>{e.erro ? `${e.erro[0]}, ${e.erro[1]}` : '—'}</span>
                    </div>
                    <Sparkline data={hist} />
                  </div>
                </Secao>

                {/* Tocar gesto */}
                {tipos.length > 0 && (
                  <Secao titulo="Tocar gesto">
                    <div className="grid grid-cols-3 gap-2">
                      {tipos.map((t, i) => (
                        <button key={t} onClick={() => enviar({ cmd: 'gesto', tipo: t })}
                          className={`rounded-xl border px-2 py-2 text-sm font-medium capitalize transition active:scale-95
                            ${e.par?.gesto_tipo === t
                              ? 'border-indigo-300 bg-indigo-50 text-indigo-700'
                              : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50'}`}>
                          <span className="mr-1 text-[10px] text-gray-400">{i + 1}</span>{t}
                        </button>
                      ))}
                    </div>
                  </Secao>
                )}

                {/* Ajustes (gerados da AJUSTES_SPEC) */}
                {Object.entries(grupos).map(([g, itens]) => (
                  <Secao key={g} titulo={`Ajustes · ${GRUPO_LABEL[g] || g}`}>
                    <div className="divide-y divide-gray-100 rounded-xl border border-gray-200 bg-white px-4">
                      {itens.map((it) => <ParRow key={it.chave} item={it} e={e} enviar={enviar} />)}
                    </div>
                  </Secao>
                ))}
              </div>
            </div>

            {/* Ações (fixas no rodapé) */}
            <div className="flex gap-2 border-t border-gray-100 px-5 py-3">
              <button onClick={() => enviar({ cmd: 'recentra' })}
                className="flex-1 rounded-xl border border-gray-300 bg-white px-3 py-2 font-medium text-gray-700 hover:bg-gray-50">
                Recentrar
              </button>
              <button onClick={() => enviar({ cmd: 'salvar' })}
                className="flex-1 rounded-xl bg-emerald-600 px-3 py-2 font-semibold text-white hover:bg-emerald-500">
                Salvar config
              </button>
            </div>
          </div>
        </aside>
      </div>

      <Toast estado={estado} />
    </div>
  )
}
