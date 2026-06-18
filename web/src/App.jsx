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

/* ─── atalhos de teclado ──────────────────────────────────────────────────── */
const SHORTCUTS = [
  { k: 'ESPAÇO', d: 'Encarar — trava a pose neutra e entra no modo seguir' },
  { k: 'f', d: 'Flutuar — mover o braço com a mão para reposicionar' },
  { k: 'k', d: 'Calibrar — fique parado, rosto visível (alguns segundos)' },
  { k: 't', d: 'Tracking ON/OFF — pausar / voltar a seguir' },
  { k: 'g', d: 'Tocar o gesto selecionado' },
  { k: '1 … 7', d: 'Tocar cada gesto: single, swing, sim, não, feliz, dançar, espreitar' },
  { k: 'u', d: 'Procurar ON/OFF — perseguir + varrer quando você some' },
  { k: 'm', d: 'Curiosidade ON/OFF — gesto sozinho quando você fica parado' },
  { k: 'c', d: 'Recentrar o olhar' },
  { k: 'z', d: 'Marcar a pose "sentado" (repouso) — só flutuando' },
  { k: 'n', d: 'Salvar config' },
  { k: 'ESC', d: 'Parar — pousa o braço no repouso e encerra a aplicação' },
]
function comandoDaTecla(key, est) {
  if (key >= '1' && key <= '7') {
    const tipo = (est?.tipos || [])[Number(key) - 1]
    return tipo ? { cmd: 'gesto', tipo } : null
  }
  switch (key) {
    case ' ': return { cmd: 'encara' }
    case 'f': return { cmd: 'flutua' }
    case 'k': return { cmd: 'calibrar' }
    case 't': return { cmd: 'tracking' }
    case 'g': return { cmd: 'gesto', tipo: est?.par?.gesto_tipo || 'single' }
    case 'u': return { cmd: 'procurar' }
    case 'm': return { cmd: 'curioso' }
    case 'c': return { cmd: 'recentra' }
    case 'z': return { cmd: 'sentado' }
    case 'n': return { cmd: 'salvar' }
    case 'Escape': return { cmd: 'parar' }
    default: return null
  }
}

/* ─── helpers do painel ───────────────────────────────────────────────────── */
const GRUPO_LABEL = {
  TRACKING: 'Tracking', PESCOCO: 'Pescoço', ALTURA: 'Altura',
  GESTOS: 'Gestos', COMPORTAMENTOS: 'Comportamentos',
}
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
  procurar_on: 'Quando você some: persegue o canto e depois olha ao redor (varredura).',
  curioso_on: 'Quando você fica parado, ele faz um gesto sozinho (curiosidade).',
  parado_s: 'Quanto tempo parado/centralizado até disparar a curiosidade.',
  cooldown_s: 'Intervalo mínimo entre reações automáticas.',
  vel_parado: 'Abaixo desta velocidade (px/s) você é considerado "parado".',
  respirar_on: 'Micro-movimento ocioso: a cabeça "respira" de leve (não vira estátua).',
  respirar_amp: 'Intensidade do respiro (0 = parado/estátua).',
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
function Secao({ titulo, children }) {
  return (
    <section className="break-inside-avoid">
      <h3 className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-gray-400">{titulo}</h3>
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
function ReidBadge({ reid }) {
  if (!reid) return (
    <div className="mt-2 flex items-center justify-between rounded-xl border border-gray-200 bg-gray-50 px-3 py-2 text-sm">
      <span className="font-semibold text-gray-500">re-ID (não trocar de pessoa)</span>
      <span className="font-medium text-gray-400">desligado</span>
    </div>
  )
  const ok = reid.presente
  return (
    <div className={`mt-2 flex items-center justify-between rounded-xl border px-3 py-2 text-sm ${ok ? 'border-emerald-200 bg-emerald-50' : 'border-amber-200 bg-amber-50'}`}>
      <span className="font-semibold text-gray-700">re-ID · {ok ? 'travado em VOCÊ' : 'pessoa sumiu'}</span>
      <span className={`font-mono font-semibold ${ok ? 'text-emerald-600' : 'text-amber-600'}`}>sim {Number(reid.sim).toFixed(2)}</span>
    </div>
  )
}
function MiniBtn({ children, onClick, title }) {
  return (
    <button onClick={onClick} title={title}
      className="flex h-8 w-8 items-center justify-center rounded-lg border border-gray-300 bg-white
                 text-lg font-semibold text-gray-700 hover:bg-gray-100 active:scale-95">{children}</button>
  )
}
function ParRow({ item, e, enviar }) {
  const v = valorDe(item, e)
  const nudge = (d) => enviar({ cmd: 'nudge', sel: item.sel, d })
  let controle
  if (item.fmt === 'bool') {
    const on = !!v
    controle = (
      <button onClick={() => nudge(1)} className={`relative h-6 w-11 rounded-full transition ${on ? 'bg-emerald-500' : 'bg-gray-300'}`}>
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
        {item.label}{HINTS[item.chave] && <span className="ml-1 cursor-help text-gray-300">ⓘ</span>}
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
function Toast({ estado }) {
  const [msg, setMsg] = useState(null)
  const lastT = useRef(0)
  useEffect(() => {
    const t = estado?.toast
    if (t && t.t && t.t !== lastT.current) {
      lastT.current = t.t
      setMsg(t)
      const id = setTimeout(() => setMsg(null), 4000)
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
        className="-mr-1 flex h-6 w-6 items-center justify-center rounded-md text-white/80 hover:bg-white/20 hover:text-white">✕</button>
    </div>
  )
}
// Terminal de eventos (o "cérebro" em tempo real), estilo log dark.
function Terminal({ eventos }) {
  const ref = useRef(null)
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight }, [eventos])
  const cor = { ok: 'text-emerald-400', erro: 'text-rose-400', aviso: 'text-amber-300', info: 'text-sky-300' }
  return (
    <div className="shrink-0 overflow-hidden rounded-xl bg-gray-950 ring-1 ring-gray-800">
      <div className="flex items-center gap-2 border-b border-gray-800 px-3 py-1.5">
        <span className="h-2 w-2 rounded-full bg-emerald-500" />
        <span className="text-[11px] font-medium uppercase tracking-wider text-gray-400">cérebro · log ao vivo</span>
      </div>
      <div ref={ref} className="h-44 overflow-y-auto px-3 py-2 font-mono text-xs leading-relaxed">
        {(eventos || []).map((ev) => (
          <div key={ev.id}>
            <span className="text-gray-600">{ev.t}</span>{'  '}
            <span className={cor[ev.kind] || 'text-gray-300'}>{ev.txt}</span>
          </div>
        ))}
        {(!eventos || !eventos.length) && <span className="text-gray-600">— aguardando eventos —</span>}
      </div>
    </div>
  )
}
// Modal de Ajuda com a legenda dos atalhos.
function Ajuda({ onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div className="max-h-[80vh] w-full max-w-lg overflow-y-auto rounded-2xl bg-white p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold">Atalhos de teclado</h2>
          <button onClick={onClose} className="flex h-8 w-8 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700">✕</button>
        </div>
        <div className="divide-y divide-gray-100">
          {SHORTCUTS.map((s) => (
            <div key={s.k} className="flex items-start gap-3 py-2">
              <kbd className="min-w-[64px] rounded-md border border-gray-300 bg-gray-50 px-2 py-1 text-center text-xs font-semibold text-gray-700">{s.k}</kbd>
              <span className="text-sm text-gray-600">{s.d}</span>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-gray-400">As mesmas ações estão nos botões/painel. Os atalhos funcionam com a página em foco.</p>
      </div>
    </div>
  )
}

/* ─── app ─────────────────────────────────────────────────────────────────── */
const PANEL_MIN = 400
export default function App() {
  const { estado, spec, conectado, enviar } = useEngine()
  const e = estado || {}
  const [aberto, setAberto] = useState(true)
  const [ajuda, setAjuda] = useState(false)
  const [width, setWidth] = useState(() => {
    const v = Number(localStorage.getItem('panelW'))
    return v >= PANEL_MIN ? v : PANEL_MIN
  })
  const [arrastando, setArrastando] = useState(false)
  const arrastRef = useRef(false)
  const [hist, setHist] = useState([])
  const estadoRef = useRef(estado)
  const ajudaRef = useRef(ajuda)
  useEffect(() => { estadoRef.current = estado }, [estado])
  useEffect(() => { ajudaRef.current = ajuda }, [ajuda])

  useEffect(() => { localStorage.setItem('panelW', String(width)) }, [width])
  useEffect(() => {
    if (!estado) return
    const mag = estado.erro ? Math.hypot(estado.erro[0], estado.erro[1]) : 0
    setHist((h) => [...h.slice(-59), mag])
  }, [estado])

  // atalhos de teclado (página em foco)
  useEffect(() => {
    const onKey = (ev) => {
      const tag = ev.target?.tagName
      if (tag && /INPUT|TEXTAREA|SELECT/.test(tag)) return
      if (ev.ctrlKey || ev.metaKey || ev.altKey) return
      if (ev.key === 'Escape' && ajudaRef.current) { setAjuda(false); return }  // ESC fecha a ajuda
      const cmd = comandoDaTecla(ev.key, estadoRef.current)
      if (cmd) { ev.preventDefault(); enviar(cmd) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enviar])

  // arrastar a divisória para redimensionar o painel
  useEffect(() => {
    const move = (ev) => {
      if (!arrastRef.current) return
      const max = Math.min(1400, window.innerWidth - 360)
      setWidth(Math.min(Math.max(window.innerWidth - ev.clientX, PANEL_MIN), max))
    }
    const up = () => { if (arrastRef.current) { arrastRef.current = false; setArrastando(false); document.body.style.userSelect = '' } }
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
  const cols = Math.max(1, Math.min(4, Math.floor(width / 320)))

  return (
    <div className="flex h-screen flex-col bg-gray-100 text-gray-900">
      {/* Cabeçalho */}
      <header className="flex items-center justify-between border-b border-gray-200 bg-white px-5 py-3">
        <div className="flex items-baseline gap-2">
          <span className="text-xl">🦾</span>
          <h1 className="text-lg font-semibold tracking-tight">Camera Follow</h1>
          <span className="text-sm text-gray-400">painel de controle</span>
        </div>
        <div className="flex items-center gap-3">
          <span className={`flex items-center gap-2 text-xs ${conectado ? 'text-emerald-600' : 'text-rose-500'}`}>
            <span className={`h-2 w-2 rounded-full ${conectado ? 'bg-emerald-500' : 'bg-rose-500'}`} />
            {conectado ? 'conectado' : 'reconectando…'}
          </span>
          <button onClick={() => setAjuda(true)} title="atalhos de teclado"
            className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50">? Ajuda</button>
          <button onClick={() => enviar({ cmd: 'parar' })} title="parar (pousa e encerra)"
            className="rounded-lg bg-rose-600 px-3 py-1.5 text-sm font-semibold text-white hover:bg-rose-500">■ Stop</button>
          <button onClick={() => setAberto((a) => !a)}
            className="rounded-lg border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 hover:bg-gray-50">
            {aberto ? 'Ocultar painel ›' : '‹ Mostrar painel'}
          </button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Esquerda: vídeo + terminal de eventos */}
        <main className="flex min-w-0 flex-1 flex-col gap-3 p-4">
          <div className="flex min-h-0 flex-1 items-center justify-center">
            <img src="/video" alt="vídeo ao vivo"
              className="max-h-full max-w-full rounded-2xl bg-black object-contain shadow-lg ring-1 ring-gray-200" />
          </div>
          <Terminal eventos={e.eventos} />
        </main>

        {/* Direita: painel redimensionável + colapsável */}
        <aside style={{ width: aberto ? width : 0 }}
          className={`relative shrink-0 overflow-hidden border-l border-gray-200 bg-white ${arrastando ? '' : 'transition-[width] duration-300 ease-in-out'}`}>
          <div onMouseDown={() => { arrastRef.current = true; setArrastando(true); document.body.style.userSelect = 'none' }}
            title="arraste para redimensionar"
            className={`absolute left-0 top-0 z-10 h-full w-1.5 cursor-col-resize ${arrastando ? 'bg-indigo-400' : 'hover:bg-indigo-300'}`} />
          <div className="flex h-full flex-col pl-1.5" style={{ width }}>
            <div className="flex items-center justify-between border-b border-gray-100 px-5 py-3">
              <span className="text-sm font-semibold text-gray-700">Controles</span>
              <button onClick={() => setAberto(false)} title="fechar"
                className="flex h-7 w-7 items-center justify-center rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700">✕</button>
            </div>

            <div className="flex-1 overflow-y-auto px-5 py-4">
              <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap: '1rem', alignItems: 'start' }}>
                <Secao titulo="Controle">
                  <div className="flex items-center justify-between rounded-xl border border-gray-200 bg-white px-4 py-3">
                    <div>
                      <div className="font-medium">Tracking</div>
                      {!e.calibrado && <div className="text-xs text-amber-600">calibre (k) p/ ativar</div>}
                    </div>
                    <button onClick={() => enviar({ cmd: 'tracking', val: !tracking })}
                      className={`rounded-xl px-5 py-2 font-semibold transition ${tracking ? 'bg-emerald-500 text-white hover:bg-emerald-400' : 'bg-gray-200 text-gray-700 hover:bg-gray-300'}`}>
                      {tracking ? 'ON' : 'OFF'}
                    </button>
                  </div>
                </Secao>

                <Secao titulo="Detecção">
                  <div className="grid grid-cols-2 gap-2">
                    <Metric label="Status" value={e.tem_rosto ? 'Detectado' : '—'} ok={e.tem_rosto} />
                    <Metric label="Fase" value={e.fase ?? '—'} />
                    <Metric label="Pan / Tilt" value={`${e.pan ?? 0}° / ${e.tilt ?? 0}°`} />
                    <Metric label="IK" value={e.ik_ok ? `${e.ik_ms} ms` : 'revertido'} ok={e.ik_ok} />
                  </div>
                  <ReidBadge reid={e.reid} />
                  <div className="mt-2 rounded-xl border border-gray-200 bg-white px-3 py-2">
                    <div className="mb-1 flex justify-between text-[11px] text-gray-400">
                      <span>erro (px)</span><span>{e.erro ? `${e.erro[0]}, ${e.erro[1]}` : '—'}</span>
                    </div>
                    <Sparkline data={hist} />
                  </div>
                </Secao>

                {tipos.length > 0 && (
                  <Secao titulo="Tocar gesto">
                    <div className="grid grid-cols-3 gap-2">
                      {tipos.map((t, i) => (
                        <button key={t} onClick={() => enviar({ cmd: 'gesto', tipo: t })}
                          className={`rounded-xl border px-2 py-2 text-sm font-medium capitalize transition active:scale-95 ${e.par?.gesto_tipo === t ? 'border-indigo-300 bg-indigo-50 text-indigo-700' : 'border-gray-200 bg-white text-gray-700 hover:bg-gray-50'}`}>
                          <span className="mr-1 text-[10px] text-gray-400">{i + 1}</span>{t}
                        </button>
                      ))}
                    </div>
                  </Secao>
                )}

                {Object.entries(grupos).map(([g, itens]) => (
                  <Secao key={g} titulo={`Ajustes · ${GRUPO_LABEL[g] || g}`}>
                    <div className="divide-y divide-gray-100 rounded-xl border border-gray-200 bg-white px-4">
                      {itens.map((it) => <ParRow key={it.chave} item={it} e={e} enviar={enviar} />)}
                    </div>
                  </Secao>
                ))}
              </div>
            </div>

            <div className="flex gap-2 border-t border-gray-100 px-5 py-3">
              <button onClick={() => enviar({ cmd: 'recentra' })}
                className="flex-1 rounded-xl border border-gray-300 bg-white px-3 py-2 font-medium text-gray-700 hover:bg-gray-50">Recentrar</button>
              <button onClick={() => enviar({ cmd: 'salvar' })}
                className="flex-1 rounded-xl bg-emerald-600 px-3 py-2 font-semibold text-white hover:bg-emerald-500">Salvar config</button>
            </div>
          </div>
        </aside>
      </div>

      {ajuda && <Ajuda onClose={() => setAjuda(false)} />}
      <Toast estado={estado} />
    </div>
  )
}
