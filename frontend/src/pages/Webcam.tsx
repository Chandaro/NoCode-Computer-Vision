import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Camera, CameraOff, Zap } from 'lucide-react'
import api, { type TrainingRun, type Project } from '../api'
import { PageHeader, Card, Slider, Select, Field, Badge, Btn } from '../components/ui'

const CLASS_COLORS = [
  '#8b5cf6','#22c55e','#f59e0b','#38bdf8','#f87171',
  '#ec4899','#a3e635','#fb923c','#34d399','#c084fc',
]
function getClsColor(id: number) { return CLASS_COLORS[id % CLASS_COLORS.length] }

interface Detection {
  x: number; y: number; w: number; h: number
  conf: number; class_id: number; class_name: string
  mask?: number[][]
}

const INFER_INTERVAL_MS = 150  // ~7 fps inference rate

export default function Webcam() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project, setProject] = useState<Project | null>(null)
  const [runs,    setRuns]    = useState<TrainingRun[]>([])
  const [runId,   setRunId]   = useState('')
  const [conf,    setConf]    = useState(0.25)
  const [active,  setActive]  = useState(false)
  const [fps,     setFps]     = useState(0)
  const [detections, setDetections] = useState<Detection[]>([])
  const [videoSize,  setVideoSize]  = useState({ w: 640, h: 480 })
  const [renderedSize, setRenderedSize] = useState({ w: 640, h: 480 })
  const [error, setError] = useState('')

  const videoRef    = useRef<HTMLVideoElement>(null)
  const canvasRef   = useRef<HTMLCanvasElement>(null)
  const streamRef   = useRef<MediaStream | null>(null)
  const pendingRef  = useRef(false)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fpsCountRef = useRef(0)
  const fpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    api.get(`/projects/${projectId}`).then(r => setProject(r.data))
    api.get(`/projects/${projectId}/training/runs`).then(r => {
      const done = r.data.filter((run: TrainingRun) => run.status === 'done')
      setRuns(done)
      if (done.length > 0) setRunId(String(done[done.length - 1].id))
    })
  }, [projectId])

  const updateRenderedSize = useCallback(() => {
    if (videoRef.current) {
      const el = videoRef.current
      setRenderedSize({ w: el.clientWidth, h: el.clientHeight })
    }
  }, [])

  useEffect(() => {
    window.addEventListener('resize', updateRenderedSize)
    return () => window.removeEventListener('resize', updateRenderedSize)
  }, [updateRenderedSize])

  const startCamera = async () => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' }
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        const track = stream.getVideoTracks()[0]
        const settings = track.getSettings()
        setVideoSize({ w: settings.width ?? 640, h: settings.height ?? 480 })
        updateRenderedSize()
      }
      setActive(true)
      startInferLoop()
      fpsTimerRef.current = setInterval(() => {
        setFps(fpsCountRef.current)
        fpsCountRef.current = 0
      }, 1000)
    } catch (e: any) {
      setError('Camera access denied or not available: ' + e.message)
    }
  }

  const stopCamera = () => {
    if (intervalRef.current)  clearInterval(intervalRef.current)
    if (fpsTimerRef.current)  clearInterval(fpsTimerRef.current)
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setActive(false)
    setDetections([])
    setFps(0)
    pendingRef.current = false
  }

  useEffect(() => () => stopCamera(), [])

  const captureAndInfer = useCallback(async () => {
    if (pendingRef.current) return
    if (!videoRef.current || !canvasRef.current || !runId) return

    const video  = videoRef.current
    const canvas = canvasRef.current
    // Scale down to max 640px wide before sending — reduces payload ~4x vs 1280p
    const scale  = Math.min(1, 640 / video.videoWidth)
    canvas.width  = Math.round(video.videoWidth  * scale)
    canvas.height = Math.round(video.videoHeight * scale)
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

    pendingRef.current = true
    canvas.toBlob(async (blob) => {
      if (!blob) { pendingRef.current = false; return }
      try {
        const fd = new FormData()
        fd.append('frame', blob, 'frame.jpg')
        fd.append('conf',  String(conf))
        fd.append('iou',   '0.45')
        const res = await api.post(
          `/projects/${projectId}/training/runs/${runId}/webcam-frame`, fd,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        )
        setDetections(res.data.detections ?? [])
        fpsCountRef.current += 1
        updateRenderedSize()
      } catch { /* drop frame on error */ }
      finally { pendingRef.current = false }
    }, 'image/jpeg', 0.75)
  }, [projectId, runId, conf, updateRenderedSize])

  const startInferLoop = useCallback(() => {
    if (intervalRef.current) clearInterval(intervalRef.current)
    intervalRef.current = setInterval(captureAndInfer, INFER_INTERVAL_MS)
  }, [captureAndInfer])

  // Restart loop when conf/runId changes so the closure captures fresh values
  useEffect(() => {
    if (active) startInferLoop()
  }, [conf, runId, active, startInferLoop])

  const hasMasks = detections.some(d => d.mask && d.mask.length > 0)

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader back={() => navigate(`/projects/${projectId}/images`)}
        title="Live Webcam Inference" subtitle={project?.name} />

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 16 }}>

        {/* ── Controls ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Model</p>
            <Field label="Detection run">
              <Select value={runId} onChange={v => { setRunId(v); setDetections([]) }}>
                <option value="">Select a run…</option>
                {runs.map(r => (
                  <option key={r.id} value={r.id}>
                    Run #{r.id} — {r.model_base}
                  </option>
                ))}
              </Select>
            </Field>
            <Slider label="Confidence" value={conf} onChange={setConf}
              min={0.1} max={0.9} step={0.05} format={v => `${Math.round(v * 100)}%`} />
          </Card>

          {/* Stats */}
          <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Live Stats</p>
            {[
              { label: 'Inference FPS', value: active ? `${fps}` : '—' },
              { label: 'Detections',    value: active ? `${detections.length}` : '—' },
              { label: 'Resolution',    value: active ? `${videoSize.w}×${videoSize.h}` : '—' },
            ].map(({ label, value }) => (
              <div key={label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, color: 'var(--text3)' }}>{label}</span>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)',
                  fontFamily: 'JetBrains Mono, monospace' }}>{value}</span>
              </div>
            ))}
          </Card>

          {/* Class breakdown */}
          {active && detections.length > 0 && (
            <Card style={{ padding: 16 }}>
              <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 10 }}>Detected</p>
              {Object.entries(
                detections.reduce((acc, d) => {
                  acc[d.class_name] = (acc[d.class_name] ?? 0) + 1
                  return acc
                }, {} as Record<string, number>)
              ).map(([cls, count], i) => (
                <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 5 }}>
                  <div style={{ width: 8, height: 8, borderRadius: '50%', background: getClsColor(i), flexShrink: 0 }} />
                  <span style={{ fontSize: 12, color: 'var(--text2)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{cls}</span>
                  <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', fontFamily: 'JetBrains Mono, monospace' }}>{count}</span>
                </div>
              ))}
            </Card>
          )}

          <Btn variant={active ? 'ghost' : 'primary'} onClick={active ? stopCamera : startCamera}
            disabled={!runId && !active}
            style={{ width: '100%', justifyContent: 'center', padding: '9px 14px',
              ...(active ? { color: 'var(--red, #f87171)', borderColor: 'var(--red, #f87171)' } : {}) }}>
            {active
              ? <><CameraOff size={14} /> Stop Camera</>
              : <><Camera size={14} /> Start Camera</>}
          </Btn>
          {error && <p style={{ fontSize: 11, color: 'var(--red, #f87171)' }}>{error}</p>}
        </div>

        {/* ── Video + overlay ── */}
        <Card style={{ padding: 0, overflow: 'hidden', position: 'relative', background: '#000' }}>
          {/* Status bar */}
          <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 10, display: 'flex', gap: 6 }}>
            {active && <Badge color="green"><Zap size={9} /> Live</Badge>}
            {active && fps > 0 && (
              <span style={{ fontSize: 10, background: 'rgba(0,0,0,0.6)', color: '#fff',
                padding: '3px 7px', borderRadius: 4, fontFamily: 'JetBrains Mono, monospace' }}>
                {fps} fps
              </span>
            )}
            {!active && !error && (
              <span style={{ fontSize: 10, background: 'rgba(0,0,0,0.5)', color: 'var(--text3)',
                padding: '3px 7px', borderRadius: 4 }}>Camera off</span>
            )}
          </div>

          <div ref={containerRef} style={{ position: 'relative', lineHeight: 0, minHeight: 360,
            display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {!active && (
              <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center', gap: 12 }}>
                <Camera size={40} style={{ color: 'var(--text3)', opacity: 0.4 }} />
                <p style={{ fontSize: 13, color: 'var(--text3)' }}>
                  {!runId ? 'Select a run to start' : 'Press Start Camera'}
                </p>
              </div>
            )}

            <video
              ref={videoRef}
              autoPlay playsInline muted
              onLoadedMetadata={updateRenderedSize}
              style={{ width: '100%', display: active ? 'block' : 'none',
                maxHeight: '65vh', objectFit: 'contain' }}
            />

            {/* SVG mask overlay for seg models */}
            {active && hasMasks && (
              <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                viewBox={`0 0 ${renderedSize.w} ${renderedSize.h}`}
                preserveAspectRatio="none">
                {detections.map((d, i) => {
                  if (!d.mask || d.mask.length === 0) return null
                  const color = getClsColor(d.class_id)
                  const pts = d.mask.map(([x, y]) =>
                    `${x * renderedSize.w},${y * renderedSize.h}`).join(' ')
                  return <polygon key={i} points={pts} fill={color}
                    fillOpacity={0.3} stroke={color} strokeWidth={1.5} />
                })}
              </svg>
            )}

            {/* Bounding box overlays */}
            {active && detections.map((d, i) => {
              const color = getClsColor(d.class_id)
              return (
                <div key={i} style={{
                  position: 'absolute',
                  left:   `${d.x * 100}%`, top:    `${d.y * 100}%`,
                  width:  `${d.w * 100}%`, height: `${d.h * 100}%`,
                  border: `2px solid ${color}`,
                  boxSizing: 'border-box', pointerEvents: 'none',
                }}>
                  <span style={{
                    position: 'absolute', top: -18, left: -1,
                    background: color, color: '#fff',
                    fontSize: 9, fontWeight: 600,
                    padding: '2px 5px', borderRadius: 3,
                    whiteSpace: 'nowrap', lineHeight: 1.4,
                    fontFamily: 'JetBrains Mono, monospace',
                  }}>
                    {d.class_name} {Math.round(d.conf * 100)}%
                  </span>
                </div>
              )
            })}
          </div>
          {/* Hidden canvas for frame capture */}
          <canvas ref={canvasRef} style={{ display: 'none' }} />
        </Card>
      </div>
    </div>
  )
}
