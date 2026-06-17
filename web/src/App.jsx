import { useEffect, useRef, useState, useCallback } from 'react'

// Hook: conecta no websocket do engine, recebe o ESTADO e expõe enviar(cmd).
// Reconecta sozinho se cair. Em dev o Vite faz proxy de /ws -> :8000.
function useEngine() {
  const [estado, setEstado] = useState(null)
  const [conectado, setConectado] = useState(false)
  const wsRef = useRef(null)

  useEffect(() => {
    let vivo = true
    let timer = null
    const conectar = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}/ws`)
      wsRef.current = ws
      ws.onopen = () => setConectado(true)
      ws.onclose = () => {
        setConectado(false)
        if (vivo) timer = setTimeout(conectar, 1000) // reconecta
      }
      ws.onmessage = (ev) => {
        try { setEstado(JSON.parse(ev.data)) } catch {}
      }
    }
    conectar()
    return () => { vivo = false; clearTimeout(timer); wsRef.current?.close() }
  }, [])

  const enviar = useCallback((cmd) => {
    const ws = wsRef.current
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(cmd))
  }, [])

  return { estado, conectado, enviar }
}

function Chip({ label, value, ok }) {
  const cor = ok === true ? 'text-emerald-400' : ok === false ? 'text-rose-400' : 'text-sky-300'
  return (
    <div className="rounded-lg bg-slate-800/60 px-3 py-2 ring-1 ring-slate-700">
      <div className="text-[10px] uppercase tracking-wide text-slate-400">{label}</div>
      <div className={`text-sm font-semibold ${cor}`}>{value}</div>
    </div>
  )
}

export default function App() {
  const { estado, conectado, enviar } = useEngine()
  const e = estado || {}
  const tracking = !!e.tracking
  const ganho = e.par?.ganho ?? 0.08
  const tipos = e.tipos || ['single', 'swing', 'sim', 'nao', 'feliz', 'dancar', 'espreitar']

  return (
    <div className="min-h-full text-slate-100">
      {/* Cabeçalho */}
      <header className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
        <h1 className="text-lg font-bold tracking-tight">
          🦾 Camera&nbsp;Follow <span className="text-slate-500 font-normal">— painel</span>
        </h1>
        <span className={`flex items-center gap-2 text-xs ${conectado ? 'text-emerald-400' : 'text-rose-400'}`}>
          <span className={`h-2 w-2 rounded-full ${conectado ? 'bg-emerald-400' : 'bg-rose-400'}`} />
          {conectado ? 'conectado' : 'reconectando…'}
        </span>
      </header>

      <main className="mx-auto max-w-3xl space-y-4 p-4">
        {/* Vídeo ao vivo (MJPEG) */}
        <div className="overflow-hidden rounded-2xl bg-black ring-1 ring-slate-800">
          <img src="/video" alt="vídeo ao vivo" className="w-full object-contain" />
        </div>

        {/* Estado */}
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
          <Chip label="fase" value={e.fase ?? '—'} />
          <Chip label="tracking" value={tracking ? 'ON' : 'off'} ok={tracking} />
          <Chip label="calibrado" value={e.calibrado ? 'OK' : 'não'} ok={!!e.calibrado} />
          <Chip label="rosto" value={e.tem_rosto ? 'sim' : 'não'} ok={!!e.tem_rosto} />
          <Chip label="erro px" value={e.erro ? `${e.erro[0]},${e.erro[1]}` : '—'} />
          <Chip label="IK" value={e.ik_ok ? `${e.ik_ms}ms` : 'rev'} ok={e.ik_ok} />
        </div>

        {/* Controles */}
        <div className="space-y-4 rounded-2xl bg-slate-900/60 p-4 ring-1 ring-slate-800">
          {/* Tracking */}
          <div className="flex items-center justify-between gap-3">
            <span className="font-medium">Tracking</span>
            <button
              onClick={() => enviar({ cmd: 'tracking', val: !tracking })}
              className={`rounded-xl px-5 py-2 font-semibold transition ${
                tracking ? 'bg-emerald-500 text-emerald-950 hover:bg-emerald-400'
                         : 'bg-slate-700 hover:bg-slate-600'}`}>
              {tracking ? 'ON' : 'OFF'}
            </button>
          </div>

          {/* Ganho */}
          <div>
            <div className="mb-1 flex justify-between text-sm">
              <span className="font-medium">Ganho</span>
              <span className="tabular-nums text-sky-300">{Number(ganho).toFixed(2)}</span>
            </div>
            <input
              type="range" min="0" max="1" step="0.02" value={ganho}
              onChange={(ev) => enviar({ cmd: 'set_par', chave: 'ganho', val: Number(ev.target.value) })}
              className="w-full accent-sky-400"
            />
          </div>

          {/* Gestos */}
          <div>
            <div className="mb-2 text-sm font-medium">Gestos</div>
            <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
              {tipos.map((t, i) => (
                <button key={t}
                  onClick={() => enviar({ cmd: 'gesto', tipo: t })}
                  className="rounded-xl bg-indigo-600/80 px-2 py-3 text-sm font-semibold capitalize
                             ring-1 ring-indigo-500/40 transition hover:bg-indigo-500 active:scale-95">
                  <span className="block text-[10px] text-indigo-200/70">{i + 1}</span>{t}
                </button>
              ))}
            </div>
          </div>

          {/* Ações */}
          <div className="flex gap-2 pt-1">
            <button onClick={() => enviar({ cmd: 'recentra' })}
              className="flex-1 rounded-xl bg-slate-700 px-3 py-2 font-medium hover:bg-slate-600">
              Recentrar
            </button>
            <button onClick={() => enviar({ cmd: 'salvar' })}
              className="flex-1 rounded-xl bg-amber-500/90 px-3 py-2 font-semibold text-amber-950 hover:bg-amber-400">
              Salvar config
            </button>
          </div>
        </div>

        <p className="pb-6 text-center text-xs text-slate-500">
          O braço é autônomo — este painel é um controle remoto. A janela do PC (cv2/teclado)
          segue funcionando em paralelo.
        </p>
      </main>
    </div>
  )
}
