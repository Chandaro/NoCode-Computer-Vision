import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Loader, Upload, Bike, ShieldCheck, ShieldAlert, Play } from 'lucide-react'
import api from '../api'
import { Card, Label, PageHeader, MetricPill, Btn, Slider, Select, Toggle, InfoBanner, Empty } from '../components/ui'

// One analysed object on the road.
interface RoadObject {
  track_id: number | null
  class_name: string
  conf: number
  x: number; y: number; w: number; h: number
  role: 'rider' | 'pedestrian' | 'vehicle'
  helmet: 'yes' | 'no' | 'unknown' | null
  indicator: 'ok' | 'warn' | 'unknown' | 'info'
}

interface RoadSummary {
  total: number; riders: number; helmets: number; no_helmet: number; unknown: number
  helmet_rate: number | null; safety_score: number | null
  counts: Record<string, number>; alerts: string[]
}

interface RoadResult {
  objects: RoadObject[]
  summary: RoadSummary
  image_w: number; image_h: number
  meta: { model: string; tracked: boolean; clip: boolean; time_ms: number }
}

interface RunItem { id: number; status: string; model_base: string }

// Indicator colours match the backend's scheme.
const INDICATOR_COLOR: Record<string, string> = {
  ok: '#22c55e', warn: '#ef4444', unknown: '#38bdf8', info: '#f59e0b',
}

const HELMET_LABEL: Record<string, string> = {
  yes: 'helmet ✓', no: 'no helmet ✗', unknown: 'helmet ?',
}

export default function RoadSafety() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate = useNavigate()

  const [runs, setRuns]       = useState<RunItem[]>([])
  const [runId, setRunId]     = useState<string>('')          // '' = COCO instant
  const [status, setStatus]   = useState<any>(null)

  const [mode, setMode]       = useState<'image' | 'video'>('image')
  const [file, setFile]       = useState<File | null>(null)
  const [fileName, setFileName] = useState('')
  const [conf, setConf]       = useState(0.3)
  const [iou, setIou]         = useState(0.45)
  const [useClip, setUseClip] = useState(false)
  const [track, setTrack]     = useState(true)

  const [running, setRunning] = useState(false)
  const [result, setResult]   = useState<RoadResult | null>(null)
  const [error, setError]     = useState('')

  // video job
  const [jobId, setJobId]     = useState<string | null>(null)
  const [job, setJob]         = useState<any>(null)

  const fileRef   = useRef<HTMLInputElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef<HTMLImageElement | null>(null)

  // load trained detection runs (for the model picker)
  useEffect(() => {
    api.get(`/projects/${projectId}/training/runs`)
      .then(r => setRuns((r.data as RunItem[]).filter(x => x.status === 'done')))
      .catch(() => {})
  }, [projectId])

  // report which helmet source will be used
  useEffect(() => {
    const q = runId ? `?run_id=${runId}` : ''
    api.get(`/projects/${projectId}/road-safety/status${q}`)
      .then(r => setStatus(r.data)).catch(() => setStatus(null))
  }, [projectId, runId])

  const pickFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f); setFileName(f.name); setResult(null); setJob(null); setJobId(null); setError('')

    if (f.type.startsWith('image/')) {
      setMode('image')
      const url = URL.createObjectURL(f)
      const img = new Image()
      img.onload = () => {
        imgRef.current = img
        const c = canvasRef.current
        if (!c) return
        c.width = img.width; c.height = img.height
        c.getContext('2d')!.drawImage(img, 0, 0)
      }
      img.src = url
    } else {
      setMode('video')
    }
  }

  const draw = (res: RoadResult) => {
    const c = canvasRef.current, img = imgRef.current
    if (!c || !img) return
    const ctx = c.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    res.objects.forEach(o => {
      const colour = INDICATOR_COLOR[o.indicator] ?? '#64748b'
      const x = o.x * c.width, y = o.y * c.height
      const w = o.w * c.width, h = o.h * c.height
      ctx.strokeStyle = colour; ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      let text = o.class_name
      if (o.track_id != null) text += ` #${o.track_id}`
      if (o.role === 'rider' && o.helmet) text += ` · ${HELMET_LABEL[o.helmet]}`
      ctx.font = '600 12px Inter, sans-serif'
      const tw = ctx.measureText(text).width + 8
      ctx.fillStyle = colour
      ctx.fillRect(x, Math.max(0, y - 18), tw, 18)
      ctx.fillStyle = '#fff'
      ctx.fillText(text, x + 4, Math.max(12, y - 5))
    })
  }

  const analyse = async () => {
    if (!file) return
    setRunning(true); setError(''); setResult(null); setJob(null); setJobId(null)
    try {
      const fd = new FormData()
      fd.append('file', file)
      if (runId) fd.append('run_id', runId)
      fd.append('conf', String(conf))
      fd.append('iou', String(iou))
      fd.append('use_clip', String(useClip))
      fd.append('track', String(track))

      if (mode === 'image') {
        const res = await api.post(
          `/projects/${projectId}/road-safety/analyze`, fd,
          { headers: { 'Content-Type': 'multipart/form-data' } })
        setResult(res.data)
        setTimeout(() => draw(res.data), 50)
      } else {
        const res = await api.post(
          `/projects/${projectId}/road-safety/analyze-video`, fd,
          { headers: { 'Content-Type': 'multipart/form-data' } })
        setJobId(res.data.job_id)
        setJob({ status: 'running', processed: 0, total_frames: 0 })
      }
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Analysis failed.')
    } finally {
      setRunning(false)
    }
  }

  // poll the video job
  useEffect(() => {
    if (!jobId) return
    const t = setInterval(async () => {
      try {
        const r = await api.get(`/projects/${projectId}/road-safety/video/${jobId}/status`)
        setJob(r.data)
        if (r.data.status === 'done' || r.data.status === 'failed') clearInterval(t)
      } catch { clearInterval(t) }
    }, 1200)
    return () => clearInterval(t)
  }, [jobId, projectId])

  const s = result?.summary

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader
        back={() => navigate(`/projects/${projectId}/images`)}
        title="Road Safety"
        subtitle="Track road users and flag helmet use"
      />

      <InfoBanner>
        Works instantly with the built-in model: people, bicycles, motorcycles and
        vehicles are detected and tracked. Helmet status appears when a helmet model
        is available (train one here) or with zero-shot CLIP
        {status ? ` — ${status.helmet_from_model ? 'a helmet model is available' : status.clip_available ? 'CLIP is available' : 'no helmet source yet, riders show as unknown'}` : ''}.
      </InfoBanner>

      {/* Controls */}
      <Card style={{ padding: 16, marginBottom: 12, marginTop: 12 }}>
        <Label>Input</Label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 8 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div>
              <span style={{ fontSize: 12, color: 'var(--text2)', display: 'block', marginBottom: 5 }}>Model</span>
              <Select value={runId} onChange={setRunId}>
                <option value="">Built-in COCO model (instant, no training)</option>
                {runs.map(r => (
                  <option key={r.id} value={String(r.id)}>Run #{r.id} · {r.model_base}</option>
                ))}
              </Select>
            </div>

            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <Btn variant="secondary" size="sm" onClick={() => fileRef.current?.click()}>
                <Upload size={12} /> Upload image or video
              </Btn>
              <input ref={fileRef} type="file" accept="image/*,video/*"
                style={{ display: 'none' }} onChange={pickFile} />
              <span style={{ fontSize: 12, color: 'var(--text3)', overflow: 'hidden',
                textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {fileName || 'No file selected'}
              </span>
            </div>

            <Slider label="Confidence Threshold" value={conf} onChange={setConf}
              min={0.05} max={0.95} format={v => `${Math.round(v * 100)}%`} />
            <Slider label="IoU Threshold" value={iou} onChange={setIou}
              min={0.1} max={0.9} format={v => `${Math.round(v * 100)}%`} />
            <Toggle label="Track across frames (ByteTrack)" checked={track} onChange={setTrack} />
            <Toggle label="Zero-shot helmet check (CLIP)" checked={useClip} onChange={setUseClip} />

            <Btn variant="primary" size="sm" onClick={analyse} disabled={!file || running}>
              {running ? <><Loader size={12} className="animate-spin" /> Analysing…</>
                : <><Play size={12} strokeWidth={2.5} /> Analyse {mode === 'video' ? 'Video' : 'Image'}</>}
            </Btn>
            {error && <p style={{ fontSize: 12, color: 'var(--danger)' }}>{error}</p>}
          </div>

          {/* Preview / result canvas */}
          <div style={{ border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden',
            background: '#07070a', display: 'flex', alignItems: 'center', justifyContent: 'center',
            minHeight: 200 }}>
            {mode === 'image' && file
              ? <canvas ref={canvasRef} style={{ maxWidth: '100%', maxHeight: 340, display: 'block', objectFit: 'contain' }} />
              : job
                ? <div style={{ textAlign: 'center', padding: 20 }}>
                    <p style={{ fontSize: 13, color: 'var(--text2)' }}>
                      {job.status === 'done' ? 'Done' : job.status === 'failed' ? 'Failed' : 'Processing…'}
                      {job.total_frames ? ` (${job.processed}/${job.total_frames} frames)` : ''}
                    </p>
                    {job.error && <p style={{ fontSize: 12, color: 'var(--danger)' }}>{job.error}</p>}
                    {job.status === 'done' && jobId && (
                      <video controls style={{ maxWidth: '100%', maxHeight: 300, marginTop: 10 }}
                        src={`/api/projects/${projectId}/road-safety/video/${jobId}/download`} />
                    )}
                    {job.summary && (
                      <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 8 }}>
                        {job.summary.unique_tracks} unique tracks · {job.summary.frames_with_unprotected_rider} frames with an unprotected rider
                      </p>
                    )}
                  </div>
                : <p style={{ fontSize: 12, color: 'var(--text3)' }}>Upload an image or video to analyse</p>}
          </div>
        </div>
      </Card>

      {/* Dashboard */}
      {s && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5,1fr)', gap: 10, marginBottom: 12 }}>
            <MetricPill label="Road users" value={`${s.total}`} color="var(--accent)" />
            <MetricPill label="Riders" value={`${s.riders}`} color="#f59e0b" />
            <MetricPill label="Helmet ✓" value={`${s.helmets}`} color="var(--success)" />
            <MetricPill label="No helmet ✗" value={`${s.no_helmet}`} color="var(--danger)" />
            <MetricPill label="Safety score" value={s.safety_score != null ? `${s.safety_score}` : '—'}
              color={s.safety_score != null && s.safety_score < 60 ? 'var(--danger)' : 'var(--success)'} />
          </div>

          {s.alerts.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 12 }}>
              {s.alerts.map((a, i) => (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13,
                  padding: '8px 12px', borderRadius: 8, background: 'rgba(239,68,68,0.08)',
                  border: '1px solid rgba(239,68,68,0.35)', color: 'var(--text)' }}>
                  <ShieldAlert size={14} style={{ color: 'var(--danger)' }} /> {a}
                </div>
              ))}
            </div>
          )}

          <Card style={{ padding: 16 }}>
            <Label>Detected road users</Label>
            {result!.objects.length === 0
              ? <Empty icon={Bike} message="No road users found" sub="Try lowering the confidence threshold" />
              : (
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, marginTop: 8 }}>
                  <thead>
                    <tr>
                      {['#', 'Type', 'Role', 'Helmet', 'Conf', 'Indicator'].map(h => (
                        <th key={h} style={{ textAlign: h === '#' || h === 'Type' || h === 'Role' || h === 'Helmet' ? 'left' : 'right',
                          color: 'var(--text3)', fontWeight: 500, fontSize: 10, paddingBottom: 6,
                          textTransform: 'uppercase', letterSpacing: '0.04em' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result!.objects.map((o, i) => (
                      <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                        <td style={{ padding: '5px 0' }}>{o.track_id ?? '—'}</td>
                        <td style={{ padding: '5px 0' }}>
                          <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 2,
                            marginRight: 6, background: INDICATOR_COLOR[o.indicator] }} />
                          {o.class_name}
                        </td>
                        <td style={{ padding: '5px 0', color: 'var(--text2)' }}>{o.role}</td>
                        <td style={{ padding: '5px 0', color: o.helmet === 'no' ? 'var(--danger)' : o.helmet === 'yes' ? 'var(--success)' : 'var(--text3)' }}>
                          {o.helmet ? (o.helmet === 'yes'
                            ? <><ShieldCheck size={12} style={{ verticalAlign: -2 }} /> yes</>
                            : o.helmet === 'no'
                              ? <><ShieldAlert size={12} style={{ verticalAlign: -2 }} /> no</>
                              : 'unknown') : '—'}
                        </td>
                        <td style={{ padding: '5px 0', textAlign: 'right', fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent)' }}>
                          {(o.conf * 100).toFixed(0)}%
                        </td>
                        <td style={{ padding: '5px 0', textAlign: 'right', color: INDICATOR_COLOR[o.indicator] }}>
                          {o.indicator}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            {result && (
              <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 10 }}>
                model: {result.meta.model} · {result.meta.time_ms} ms
                {result.meta.clip ? ' · CLIP helmet check on' : ''}
              </p>
            )}
          </Card>
        </>
      )}
    </div>
  )
}
