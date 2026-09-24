import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Loader, Upload, Play, Navigation } from 'lucide-react'
import api from '../api'
import { Card, Label, PageHeader, Btn, Slider, Select, InfoBanner, Empty, TextInput } from '../components/ui'

interface Obstacle {
  class_name: string; cx: number; prox: number; conf: number
  x: number; y: number; w: number; h: number
}
interface Guidance {
  action: 'straight' | 'left' | 'right' | 'stop'
  urgency: number
  hud: { arrow: number | null; text: string; colour: string }
  corridors: { left: number; centre: number; right: number }
  obstacles: Obstacle[]
  reasons: string[]
  centre_clear: boolean
  meta?: { model: string; time_ms: number }
}
interface RunItem { id: number; status: string; model_base: string }

// Big, glanceable on-road navigation arrow - the whole point of the helmet HUD.
function Arrow({ action, colour, text }: {
  action: 'straight' | 'left' | 'right' | 'stop'; colour: string; text: string
}) {
  // curved path + arrow-head placement for each cue, drawn on a mock road
  const CURVES: Record<string, { d: string; tip: [number, number]; rot: number } | null> = {
    straight: { d: 'M50,100 L50,32',                                   tip: [50, 30],  rot: 0 },
    left:     { d: 'M62,102 C62,76 42,72 26,44',                       tip: [25, 42],  rot: -48 },
    right:    { d: 'M38,102 C38,76 58,72 74,44',                       tip: [75, 42],  rot: 48 },
    stop:     null,
  }
  const c = CURVES[action]
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <div style={{ width: 160, height: 160, borderRadius: 16, overflow: 'hidden',
        background: 'linear-gradient(#1c1c22 0%, #2b2b33 48%, #3a3a44 50%, #26262e 100%)',
        border: `2px solid ${colour}`, boxShadow: `0 0 26px ${colour}55` }}>
        <svg width="160" height="160" viewBox="0 0 100 100">
          {c === null ? (
            <>
              <rect x="16" y="52" width="68" height="12" rx="4" fill={colour} opacity="0.85" />
              <path d="M30,30 L70,74 M70,30 L30,74" stroke={colour} strokeWidth="6"
                strokeLinecap="round" />
            </>
          ) : (
            <>
              <path d={c.d} stroke={colour} strokeWidth="12" fill="none" strokeLinecap="round"
                opacity="0.55" />
              <path d={c.d} stroke="#fff" strokeWidth="1.6" fill="none" strokeLinecap="round" />
              <polygon points="0,-13 11,8 -11,8" fill={colour}
                transform={`translate(${c.tip[0]},${c.tip[1]}) rotate(${c.rot})`} />
            </>
          )}
        </svg>
      </div>
      <span style={{ fontSize: 20, fontWeight: 800, letterSpacing: '0.06em', color: colour }}>{text}</span>
    </div>
  )
}

// Three bars showing how blocked each corridor is.
function Corridors({ c }: { c: { left: number; centre: number; right: number } }) {
  const bar = (name: string, v: number) => {
    const pct = Math.min(100, Math.round(v * 100))
    const col = v >= 0.25 ? 'var(--danger)' : 'var(--success)'
    return (
      <div key={name} style={{ flex: 1 }}>
        <div style={{ fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase',
          letterSpacing: '0.05em', marginBottom: 4 }}>{name}</div>
        <div style={{ height: 8, borderRadius: 4, background: 'var(--surface3)', overflow: 'hidden' }}>
          <div style={{ width: `${pct}%`, height: '100%', background: col, transition: 'width 180ms' }} />
        </div>
        <div style={{ fontSize: 10, color: 'var(--text2)', marginTop: 3, fontFamily: 'JetBrains Mono, monospace' }}>
          {v.toFixed(2)}
        </div>
      </div>
    )
  }
  return <div style={{ display: 'flex', gap: 10 }}>
    {bar('left', c.left)}{bar('centre (path)', c.centre)}{bar('right', c.right)}
  </div>
}

export default function Guidance() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate = useNavigate()

  const [runs, setRuns]     = useState<RunItem[]>([])
  const [runId, setRunId]   = useState('')
  const [mode, setMode]     = useState<'image' | 'video'>('image')
  const [file, setFile]     = useState<File | null>(null)
  const [fileName, setFileName] = useState('')
  const [conf, setConf]     = useState(0.3)

  const [running, setRunning] = useState(false)
  const [result, setResult]   = useState<Guidance | null>(null)
  const [error, setError]     = useState('')
  const [jobId, setJobId]     = useState<string | null>(null)
  const [job, setJob]         = useState<any>(null)
  // Live (realtime) mode
  const [liveOn,    setLiveOn]    = useState(false)
  const [liveSrc,   setLiveSrc]   = useState('0')
  const [liveEvery, setLiveEvery] = useState(2)
  const [liveImg,   setLiveImg]   = useState(320)
  const [stats,     setStats]     = useState<any>(null)
  // camera calibration (auto-detected, still nudgable by hand)
  const [calRunning, setCalRunning] = useState(false)
  const [cal, setCal] = useState<any>({
    horizon: 0.47, road_half: 1.8, road_bottom: 0.99, confidence: null, basis: 'default',
  })

  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    api.get(`/projects/${projectId}/training/runs`)
      .then(r => setRuns((r.data as RunItem[]).filter(x => x.status === 'done')))
      .catch(() => {})
  }, [projectId])

  const pick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f); setFileName(f.name); setResult(null); setJob(null); setJobId(null); setError('')
    setMode(f.type.startsWith('image/') ? 'image' : 'video')
  }

  const run = async () => {
    if (!file) return
    setRunning(true); setError(''); setResult(null); setJob(null); setJobId(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      if (runId) fd.append('run_id', runId)
      fd.append('conf', String(conf))
      if (mode === 'image') {
        const r = await api.post(`/projects/${projectId}/guidance/analyze`, fd,
          { headers: { 'Content-Type': 'multipart/form-data' } })
        setResult(r.data)
      } else {
        const r = await api.post(`/projects/${projectId}/guidance/analyze-video`, fd,
          { headers: { 'Content-Type': 'multipart/form-data' } })
        setJobId(r.data.job_id); setJob({ status: 'running', processed: 0, total_frames: 0 })
      }
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Guidance failed.')
    } finally { setRunning(false) }
  }

  useEffect(() => {
    if (!jobId) return
    const t = setInterval(async () => {
      try {
        const r = await api.get(`/projects/${projectId}/guidance/video/${jobId}/status`)
        setJob(r.data)
        if (r.data.status === 'done' || r.data.status === 'failed') clearInterval(t)
      } catch { clearInterval(t) }
    }, 1200)
    return () => clearInterval(t)
  }, [jobId, projectId])

  // poll live stats while the realtime stream is on
  useEffect(() => {
    if (!liveOn) { setStats(null); return }
    const t = setInterval(async () => {
      try {
        const r = await api.get(`/projects/${projectId}/guidance/live/stats`)
        setStats(r.data)
      } catch { /* keep last value */ }
    }, 1000)
    return () => clearInterval(t)
  }, [liveOn, projectId])

  // MJPEG stream URL - the browser renders it straight in an <img>
  const liveUrl = `${window.location.origin}/api/projects/${projectId}/guidance/live` +
    `?source=${encodeURIComponent(liveSrc)}&imgsz=${liveImg}` +
    `&horizon=${cal.horizon}&road_half=${cal.road_half}&road_bottom=${cal.road_bottom}` +
    `&every=${liveEvery}`

  // Auto-calibrate the camera knobs from the picture itself
  const autoCal = async () => {
    if (!file) { setError('Pick an image or video first, then auto-calibrate.'); return }
    setCalRunning(true); setError('')
    try {
      const fd = new FormData()
      fd.append('file', file)
      const r = await api.post(`/projects/${projectId}/guidance/calibrate`, fd)
      const d = r.data
      if (d.ok) {
        setCal({
          horizon: d.horizon, road_half: d.road_half, road_bottom: d.road_bottom,
          confidence: d.confidence, basis: d.basis,
        })
      } else {
        setCal((c: any) => ({ ...c, confidence: 0, basis: d.reason || 'not found' }))
      }
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Auto-calibration failed.')
    } finally { setCalRunning(false) }
  }

  return (
    <div style={{ maxWidth: 980, margin: '0 auto' }}>
      <PageHeader back={() => navigate(`/projects/${projectId}/images`)}
        title="Steering Guide" subtitle="Which way to move the wheel" />

      <InfoBanner>
        Built for a helmet camera: the bottom-centre of the frame is the path ahead.
        The guide watches for obstacles in that path and tells you <b>GO</b>,
        <b> KEEP LEFT</b>, <b>KEEP RIGHT</b> or <b>STOP</b>. This is a
        <b> decision aid, not a safety-certified device</b> — always ride to the conditions.
      </InfoBanner>

      <Card style={{ padding: 16, margin: '12px 0' }}>
        <Label>Input</Label>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 8 }}>
          <div>
            <span style={{ fontSize: 12, color: 'var(--text2)', display: 'block', marginBottom: 5 }}>Model</span>
            <Select value={runId} onChange={setRunId}>
              <option value="">Built-in COCO model (instant)</option>
              {runs.map(r => <option key={r.id} value={String(r.id)}>Run #{r.id} · {r.model_base}</option>)}
            </Select>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            <Btn variant="secondary" size="sm" onClick={() => fileRef.current?.click()}>
              <Upload size={12} /> Upload image or video
            </Btn>
            <input ref={fileRef} type="file" accept="image/*,video/*" style={{ display: 'none' }} onChange={pick} />
            <span style={{ fontSize: 12, color: 'var(--text3)' }}>{fileName || 'No file selected'}</span>
          </div>
          <Slider label="Confidence Threshold" value={conf} onChange={setConf}
            min={0.05} max={0.95} format={v => `${Math.round(v * 100)}%`} />
          <Btn variant="primary" size="sm" onClick={run} disabled={!file || running}>
            {running ? <><Loader size={12} className="animate-spin" /> Analysing…</>
              : <><Play size={12} strokeWidth={2.5} /> Give me the cue</>}
          </Btn>
          {error && <p style={{ fontSize: 12, color: 'var(--danger)' }}>{error}</p>}
        </div>
      </Card>

      {/* Camera calibration: auto-detect the 3 knobs, hand-tweak if needed */}
      <Card style={{ padding: 16, margin: '12px 0' }}>
        <Label>Camera calibration</Label>
        <p style={{ fontSize: 12, color: 'var(--text3)', margin: '6px 0 12px' }}>
          Give it one typical frame from the camera and it works out the road geometry itself.
          A helmet cam, a dashcam behind a bonnet and a bar mount all need different values.
        </p>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
          <Btn variant="secondary" size="sm" onClick={autoCal} disabled={calRunning}>
            {calRunning ? 'Detecting...' : 'Auto-calibrate'}
          </Btn>
          {cal.confidence !== null && (
            <span style={{ fontSize: 12, color: cal.confidence > 0.35 ? 'var(--success)' : 'var(--text3)' }}>
              confidence {Math.round((cal.confidence || 0) * 100)}% · {cal.basis}
            </span>
          )}
        </div>
        <Slider label="Horizon" value={cal.horizon}
          onChange={v => setCal((c: any) => ({ ...c, horizon: v }))}
          min={0.2} max={0.7} step={0.01} format={v => v.toFixed(2)} />
        <Slider label="Path width (metres each side)" value={cal.road_half}
          onChange={v => setCal((c: any) => ({ ...c, road_half: v }))}
          min={1.2} max={4} step={0.1} format={v => `${v.toFixed(1)} m`} />
        <Slider label="Road starts at (below = bonnet/bars)" value={cal.road_bottom}
          onChange={v => setCal((c: any) => ({ ...c, road_bottom: v }))}
          min={0.4} max={1} step={0.01} format={v => v.toFixed(2)} />
      </Card>

      {/* Realtime (live) mode - the stream a helmet device would show */}
      <Card style={{ padding: 16, margin: '12px 0' }}>
        <Label>Realtime (live) mode</Label>
        <p style={{ fontSize: 12, color: 'var(--text3)', margin: '6px 0 12px' }}>
          Runs the guide continuously. Source <code>0</code> = webcam, or paste a video
          file path to loop it. Detection can run every Nth frame while the cue keeps
          easing on every frame, which is what keeps it smooth at speed.
        </p>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
          <TextInput value={liveSrc} onChange={setLiveSrc} placeholder="0 (webcam) or C:/path/clip.mp4" />
          <Select value={String(liveImg)} onChange={v => setLiveImg(Number(v))}>
            <option value="320">320 px (fastest)</option>
            <option value="416">416 px</option>
            <option value="640">640 px (most precise)</option>
          </Select>
          <Select value={String(liveEvery)} onChange={v => setLiveEvery(Number(v))}>
            <option value="1">detect every frame</option>
            <option value="2">detect every 2nd frame</option>
            <option value="3">detect every 3rd frame</option>
          </Select>
          <Btn variant={liveOn ? 'danger' : 'primary'} size="sm" onClick={() => setLiveOn(v => !v)}>
            {liveOn ? 'Stop live' : 'Start live'}
          </Btn>
        </div>
        {liveOn && (
          <>
            <img src={liveUrl} alt="live steering guide"
              style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)' }} />
            {stats && stats.running && (
              <div style={{ display: 'flex', gap: 16, marginTop: 10, fontSize: 12,
                color: 'var(--text2)', flexWrap: 'wrap' }}>
                <span>FPS <b>{stats.fps}</b></span>
                <span>frame time <b>{stats.latency_ms} ms</b></span>
                <span>cue <b>{stats.action}</b></span>
                <span>urgency <b>{stats.urgency}</b></span>
                <span>frames <b>{stats.frame}</b></span>
              </div>
            )}
          </>
        )}
      </Card>

      {/* Image result: the HUD */}
      {result && (
        <Card style={{ padding: 20 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: 24, alignItems: 'start' }}>
            <Arrow action={result.action} colour={result.hud.colour} text={result.hud.text} />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text3)', textTransform: 'uppercase',
                  letterSpacing: '0.05em', marginBottom: 6 }}>Corridor blocking</div>
                <Corridors c={result.corridors} />
              </div>
              <div>
                <div style={{ fontSize: 11, color: 'var(--text3)', textTransform: 'uppercase',
                  letterSpacing: '0.05em', marginBottom: 4 }}>Urgency</div>
                <div style={{ height: 6, borderRadius: 3, background: 'var(--surface3)', overflow: 'hidden' }}>
                  <div style={{ width: `${Math.round(result.urgency * 100)}%`, height: '100%',
                    background: result.urgency > 0.7 ? 'var(--danger)' : result.urgency > 0.3 ? '#f59e0b' : 'var(--success)' }} />
                </div>
              </div>
              {result.reasons.length > 0 && (
                <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--text2)' }}>
                  {result.reasons.map((r, i) => <li key={i}>{r}</li>)}
                </ul>
              )}
              {result.obstacles.length > 0 && (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                  <thead><tr>
                    {['Obstacle', 'Proximity', 'Where'].map(h => (
                      <th key={h} style={{ textAlign: 'left', fontSize: 10, color: 'var(--text3)',
                        textTransform: 'uppercase', letterSpacing: '0.04em', paddingBottom: 4 }}>{h}</th>
                    ))}
                  </tr></thead>
                  <tbody>
                    {result.obstacles.slice(0, 8).map((o, i) => (
                      <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                        <td style={{ padding: '4px 0' }}>{o.class_name}</td>
                        <td style={{ padding: '4px 0', fontFamily: 'JetBrains Mono, monospace',
                          color: o.prox > 0.7 ? 'var(--danger)' : 'var(--text2)' }}>{o.prox.toFixed(2)}</td>
                        <td style={{ padding: '4px 0', color: 'var(--text2)' }}>
                          {o.cx < 0.35 ? 'left' : o.cx <= 0.65 ? 'centre' : 'right'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
              {result.meta && (
                <p style={{ fontSize: 11, color: 'var(--text3)' }}>
                  model: {result.meta.model} · {result.meta.time_ms} ms
                </p>
              )}
            </div>
          </div>
        </Card>
      )}

      {/* Video result */}
      {job && (
        <Card style={{ padding: 20 }}>
          <Label>Video guidance</Label>
          {job.status !== 'done'
            ? <p style={{ fontSize: 13, color: 'var(--text2)', marginTop: 8 }}>
                {job.status === 'failed' ? `Failed: ${job.error}`
                  : `Processing… ${job.processed}/${job.total_frames} frames`}
              </p>
            : (
              <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 12 }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10 }}>
                  {Object.entries(job.summary?.tally ?? {}).map(([k, v]) => (
                    <div key={k} style={{ padding: '8px 10px', borderRadius: 8, background: 'var(--surface2)',
                      border: '1px solid var(--border)' }}>
                      <div style={{ fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase' }}>{k}</div>
                      <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text)' }}>{v as number}</div>
                    </div>
                  ))}
                </div>
                <video controls style={{ maxWidth: '100%', borderRadius: 8 }}
                  src={`/api/projects/${projectId}/guidance/video/${jobId}/download`} />
                {(job.summary?.events?.length ?? 0) > 0 && (
                  <div>
                    <div style={{ fontSize: 11, color: 'var(--text3)', textTransform: 'uppercase',
                      letterSpacing: '0.05em', marginBottom: 4 }}>Cue changes</div>
                    <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12, color: 'var(--text2)' }}>
                      {job.summary.events.slice(0, 8).map((e: any, i: number) => (
                        <li key={i}>frame {e.frame}: <b>{e.action.toUpperCase()}</b>
                          {' '}({e.reasons.join('; ') || '—'})</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
        </Card>
      )}

      {!result && !job && (
        <Empty icon={Navigation} message="No cue yet"
          sub="Upload a helmet-cam image or clip and press the button" />
      )}
    </div>
  )
}
