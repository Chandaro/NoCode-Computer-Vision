import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Zap, Download, RefreshCw, ChevronDown, ChevronUp, BarChart2, FileCode, Loader, Cpu, Square, Trash2, FlaskConical, Upload, X, Eye, Video, Camera } from 'lucide-react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'
import api, { type Project, type TrainingRun } from '../api'
import { Card, Field, Select, Slider, Btn, Badge, PageHeader, LogTerminal, ProgressBar } from '../components/ui'

interface MetricPoint { epoch: number; mAP50: number; precision: number; recall: number }
interface Progress   { epoch: number; total: number; mAP50: number; precision: number; recall: number }

// ── Detection overlay (box per detection drawn as absolute divs) ───────────────
const CLASS_COLORS = [
  '#8b5cf6','#22c55e','#f59e0b','#38bdf8','#f87171',
  '#ec4899','#a3e635','#fb923c','#34d399','#c084fc',
]
function getClsColor(id: number) { return CLASS_COLORS[id % CLASS_COLORS.length] }

interface BatchDetection {
  x: number; y: number; w: number; h: number
  conf: number; class_id: number; class_name: string
  mask?: number[][]
}
interface BatchImageResult {
  filename: string; detections: BatchDetection[]
  count: number; image_w: number; image_h: number
}
interface BatchSummary {
  total_images: number; total_detections: number
  avg_detections_per_image: number; images_with_detections: number
  class_counts: Record<string, number>
}

function TestImageCard({ file, src, result }: { file?: File; src?: string; result: BatchImageResult }) {
  const [url, setUrl] = useState('')
  const containerRef = useRef<HTMLDivElement>(null)
  const [renderedSize, setRenderedSize] = useState({ w: 1, h: 1 })
  const hasMasks = result.detections.some(d => d.mask && d.mask.length > 0)

  useEffect(() => {
    if (file) {
      const u = URL.createObjectURL(file)
      setUrl(u)
      return () => URL.revokeObjectURL(u)
    } else if (src) {
      setUrl(src)
    }
  }, [file, src])

  return (
    <div style={{
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: 8, overflow: 'hidden',
    }}>
      <div ref={containerRef} style={{ position: 'relative', lineHeight: 0 }}>
        {url && (
          <img
            src={url}
            onLoad={e => {
              const el = e.currentTarget
              setRenderedSize({ w: el.clientWidth, h: el.clientHeight })
            }}
            style={{ width: '100%', display: 'block', maxHeight: 220, objectFit: 'contain', background: '#000' }}
            alt={result.filename}
          />
        )}
        {/* SVG mask polygons (seg models) */}
        {url && hasMasks && (
          <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
            viewBox={`0 0 ${renderedSize.w} ${renderedSize.h}`}
            preserveAspectRatio="none">
            {result.detections.map((d, i) => {
              if (!d.mask || d.mask.length === 0) return null
              const color = getClsColor(d.class_id)
              const pts = d.mask.map(([x, y]) => `${x * renderedSize.w},${y * renderedSize.h}`).join(' ')
              return (
                <g key={i}>
                  <polygon points={pts} fill={color} fillOpacity={0.25} stroke={color} strokeWidth={1.5} />
                </g>
              )
            })}
          </svg>
        )}
        {/* Bounding box overlays */}
        {url && result.detections.map((d, i) => {
          const color = getClsColor(d.class_id)
          return (
            <div key={i} style={{
              position: 'absolute',
              left: `${d.x * 100}%`, top: `${d.y * 100}%`,
              width: `${d.w * 100}%`, height: `${d.h * 100}%`,
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
      <div style={{ padding: '7px 10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <span style={{ fontSize: 11, color: 'var(--text3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: '70%' }}>
          {result.filename}
        </span>
        <span style={{
          fontSize: 11, fontWeight: 600,
          color: result.count > 0 ? 'var(--success)' : 'var(--text3)',
          fontFamily: 'JetBrains Mono, monospace', flexShrink: 0,
        }}>
          {result.count} {hasMasks ? 'seg' : 'det'}
        </span>
      </div>
    </div>
  )
}

export default function Train() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project, setProject]     = useState<Project | null>(null)
  const [runs, setRuns]           = useState<TrainingRun[]>([])
  const [epochs, setEpochs]       = useState(50)
  const [imgsz, setImgsz]         = useState(640)
  const [batch, setBatch]         = useState(16)
  const [modelBase, setModelBase] = useState('yolo11n.pt')
  const [valSplit, setValSplit]   = useState(0.2)
  const [showAug, setShowAug]     = useState(false)

  // Optimizer
  const [optimizer,     setOptimizer]     = useState('auto')
  const [lr0,           setLr0]           = useState(0.01)
  const [lrf,           setLrf]           = useState(0.01)
  const [momentum,      setMomentum]      = useState(0.937)
  const [weightDecay,   setWeightDecay]   = useState(0.0005)
  const [warmupEpochs,  setWarmupEpochs]  = useState(3.0)
  const [patience,      setPatience]      = useState(50)
  // Geometric aug
  const [fliplr,    setFliplr]    = useState(0.5)
  const [flipud,    setFlipud]    = useState(0.0)
  const [degrees,   setDegrees]   = useState(0.0)
  const [translate, setTranslate] = useState(0.1)
  const [scale,     setScale]     = useState(0.5)
  const [shear,     setShear]     = useState(0.0)
  const [perspective, setPerspective] = useState(0.0)
  // Color aug
  const [hsvH,      setHsvH]      = useState(0.015)
  const [hsvS,      setHsvS]      = useState(0.7)
  const [hsvV,      setHsvV]      = useState(0.4)
  // Mixing / cutout
  const [mosaic,    setMosaic]    = useState(1.0)
  const [mixup,     setMixup]     = useState(0.0)
  const [copyPaste, setCopyPaste] = useState(0.0)
  const [erasing,   setErasing]   = useState(0.4)

  const [streaming, setStreaming]   = useState(false)
  const [logs, setLogs]             = useState<string[]>([])
  const [progress, setProgress]     = useState<Progress | null>(null)
  const [chartData, setChartData]   = useState<MetricPoint[]>([])
  const [onnxStatus,     setOnnxStatus]     = useState<Record<number, string>>({})
  const [tfliteStatus,   setTfliteStatus]   = useState<Record<number, string>>({})
  const [trtStatus,      setTrtStatus]      = useState<Record<number, string>>({})
  const [resumeRunId,    setResumeRunId]     = useState('')
  const [stoppingRun,    setStoppingRun]    = useState<Set<number>>(new Set())
  const [deletingRun,    setDeletingRun]    = useState<Set<number>>(new Set())

  // ── Aug preview ─────────────────────────────────────────────────────────────
  const [augPreviewing,  setAugPreviewing]  = useState(false)
  const [augPreviews,    setAugPreviews]    = useState<string[]>([])
  const [showAugModal,   setShowAugModal]   = useState(false)

  // ── Video inference ──────────────────────────────────────────────────────────
  const [videoRunId,     setVideoRunId]     = useState('')
  const videoFileRef                        = useRef<HTMLInputElement>(null)
  const [videoFile,      setVideoFile]      = useState<File | null>(null)
  const [videoInputMode, setVideoInputMode] = useState<'file'|'url'>('file')
  const [videoUrl,       setVideoUrl]       = useState('')
  const [videoConf,      setVideoConf]      = useState(0.25)
  const [videoTracker,   setVideoTracker]   = useState(false)
  const [videoJobId,     setVideoJobId]     = useState('')
  const [videoStatus,    setVideoStatus]    = useState<'idle'|'uploading'|'running'|'done'|'failed'>('idle')
  const [videoProgress,  setVideoProgress]  = useState({ processed: 0, total: 0 })
  const videoPollerRef                      = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Test set ────────────────────────────────────────────────────────────────
  const [testRunId,      setTestRunId]      = useState('')
  const [testFiles,      setTestFiles]      = useState<File[]>([])
  const [testInputMode,  setTestInputMode]  = useState<'file'|'url'>('file')
  const [testUrl,        setTestUrl]        = useState('')
  const [testUrlDisplay, setTestUrlDisplay] = useState('')
  const [testConf,       setTestConf]       = useState(0.25)
  const [testRunning,    setTestRunning]    = useState(false)
  const [testResults,    setTestResults]    = useState<BatchImageResult[] | null>(null)
  const [testSummary,    setTestSummary]    = useState<BatchSummary | null>(null)
  const testFileRef = useRef<HTMLInputElement>(null)

  const logRef = useRef<HTMLDivElement>(null)
  const esRef  = useRef<EventSource | null>(null)

  useEffect(() => {
    api.get(`/projects/${projectId}`).then(r => setProject(r.data))
    loadRuns()
  }, [projectId])

  const loadRuns = () =>
    api.get(`/projects/${projectId}/training/runs`).then(r => setRuns(r.data))

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])
  useEffect(() => () => { esRef.current?.close() }, [])
  useEffect(() => () => { if (videoPollerRef.current) clearInterval(videoPollerRef.current) }, [])

  const startTraining = async () => {
    esRef.current?.close()
    setLogs([]); setChartData([]); setProgress(null)
    const res = await api.post(`/projects/${projectId}/training/start`, {
      epochs, imgsz, batch, model_base: modelBase, val_split: valSplit,
      optimizer, lr0, lrf, momentum, weight_decay: weightDecay,
      warmup_epochs: warmupEpochs, patience,
      fliplr, flipud, degrees, translate, scale, shear, perspective,
      hsv_h: hsvH, hsv_s: hsvS, hsv_v: hsvV,
      mosaic, mixup, copy_paste: copyPaste, erasing,
      resume_run_id: resumeRunId ? Number(resumeRunId) : null,
    })
    const run: TrainingRun = res.data
    setStreaming(true)
    const es = new EventSource(`/api/projects/${projectId}/training/runs/${run.id}/logs`)
    esRef.current = es
    es.onmessage = (e) => {
      const msg: string = e.data
      if (msg === '__END__') { es.close(); esRef.current = null; setStreaming(false); loadRuns(); return }
      if (msg.startsWith('__PROGRESS__:')) {
        const parts = msg.split(':')
        if (parts.length < 5) return
        const [, ep, m50, prec, rec] = parts
        const epochNum = Number(ep.split('/')[0]), totalNum = Number(ep.split('/')[1])
        const mAP50Val = Number(m50), precVal = Number(prec), recVal = Number(rec)
        setProgress({ epoch: epochNum, total: totalNum, mAP50: mAP50Val, precision: precVal, recall: recVal })
        setChartData(prev => [...prev, { epoch: epochNum, mAP50: mAP50Val, precision: precVal, recall: recVal }])
        return
      }
      if (msg.startsWith('__DONE__:') || msg === '__FAILED__') { setStreaming(false); loadRuns() }
      setLogs(prev => [...prev, msg])
    }
    es.onerror = () => { es.close(); esRef.current = null; setStreaming(false); loadRuns(); setLogs(prev => [...prev, '[WARN] Connection lost']) }
  }

  const startExport = async (
    runId: number, fmt: 'onnx' | 'tflite' | 'tensorrt',
    setStatus: React.Dispatch<React.SetStateAction<Record<number, string>>>
  ) => {
    setStatus(s => ({ ...s, [runId]: 'running' }))
    const fmtPath = fmt === 'tensorrt' ? 'tensorrt' : fmt
    try {
      await api.post(`/projects/${projectId}/training/runs/${runId}/export/${fmtPath}`)
      const poll = setInterval(async () => {
        const r = await api.get(
          `/projects/${projectId}/training/runs/${runId}/export/${fmtPath}/status`
        )
        const status = r.data.status as string
        setStatus(s => ({ ...s, [runId]: status }))
        if (status === 'done' || status === 'failed') clearInterval(poll)
      }, 2000)
    } catch { setStatus(s => ({ ...s, [runId]: 'failed' })) }
  }

  const exportOnnx = (runId: number) => startExport(runId, 'onnx', setOnnxStatus)

  const stopRun = async (runId: number) => {
    setStoppingRun(prev => new Set(prev).add(runId))
    try {
      await api.post(`/projects/${projectId}/training/runs/${runId}/stop`)
      await loadRuns()
    } finally {
      setStoppingRun(prev => { const s = new Set(prev); s.delete(runId); return s })
    }
  }

  const deleteRun = async (runId: number) => {
    if (!confirm(`Delete Run #${runId} and all its files? This cannot be undone.`)) return
    setDeletingRun(prev => new Set(prev).add(runId))
    try {
      await api.delete(`/projects/${projectId}/training/runs/${runId}`)
      setRuns(prev => prev.filter(r => r.id !== runId))
    } finally {
      setDeletingRun(prev => { const s = new Set(prev); s.delete(runId); return s })
    }
  }

  const runAugPreview = async () => {
    setAugPreviewing(true)
    try {
      const res = await api.post(`/projects/${projectId}/training/augmentation-preview`, {
        fliplr, flipud, degrees, translate, scale,
        hsv_h: hsvH, hsv_s: hsvS, hsv_v: hsvV, mosaic, n: 6,
      })
      setAugPreviews(res.data.previews ?? [])
      setShowAugModal(true)
    } catch (e: any) {
      alert('Preview failed: ' + (e?.response?.data?.detail ?? e.message))
    } finally {
      setAugPreviewing(false)
    }
  }

  const startVideoInfer = async () => {
    if (!videoRunId) return
    if (videoInputMode === 'file' && !videoFile) return
    if (videoInputMode === 'url' && !videoUrl.trim()) return
    if (videoPollerRef.current) clearInterval(videoPollerRef.current)
    setVideoStatus('uploading'); setVideoJobId('')
    setVideoProgress({ processed: 0, total: 0 })
    try {
      const fd = new FormData()
      fd.append('conf', String(videoConf))
      fd.append('iou', '0.45')
      fd.append('tracker', String(videoTracker))
      let endpoint: string
      if (videoInputMode === 'url') {
        fd.append('url', videoUrl.trim())
        endpoint = `/projects/${projectId}/training/runs/${videoRunId}/video-infer-url`
      } else {
        fd.append('file', videoFile!)
        endpoint = `/projects/${projectId}/training/runs/${videoRunId}/video-infer`
      }
      const res = await api.post(endpoint, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      const jobId: string = res.data.job_id
      setVideoJobId(jobId)
      setVideoStatus('running')
      videoPollerRef.current = setInterval(async () => {
        try {
          const s = await api.get(`/projects/${projectId}/training/runs/${videoRunId}/video-infer/${jobId}/status`)
          setVideoProgress({ processed: s.data.processed, total: s.data.total_frames })
          if (s.data.stage) setVideoStatus(s.data.stage === 'downloading' ? 'uploading' : 'running')
          if (s.data.status === 'done') {
            clearInterval(videoPollerRef.current!)
            setVideoStatus('done')
          } else if (s.data.status === 'failed') {
            clearInterval(videoPollerRef.current!)
            setVideoStatus('failed')
            alert('Video processing failed: ' + (s.data.error ?? 'unknown error'))
          }
        } catch { /* keep polling */ }
      }, 1500)
    } catch (e: any) {
      setVideoStatus('failed')
      alert('Upload failed: ' + (e?.response?.data?.detail ?? e.message))
    }
  }

  const runBatchTest = async () => {
    if (testInputMode === 'file' && testFiles.length === 0) return
    if (testInputMode === 'url' && !testUrl.trim()) return
    if (!testRunId) return
    setTestRunning(true); setTestResults(null); setTestSummary(null)
    try {
      const fd = new FormData()
      fd.append('conf', String(testConf))
      if (testInputMode === 'url') {
        fd.append('url', testUrl.trim())
        fd.append('iou', '0.45')
        const res = await api.post(
          `/projects/${projectId}/training/runs/${testRunId}/infer-url`, fd,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        )
        const imgResult: BatchImageResult = {
          filename: testUrl.split('/').pop()?.slice(0, 64) || 'image',
          detections: res.data.detections,
          count: res.data.count,
          image_w: res.data.image_w,
          image_h: res.data.image_h,
        }
        setTestResults([imgResult])
        setTestUrlDisplay(testUrl.trim())
        const cc: Record<string, number> = {}
        res.data.detections.forEach((d: BatchDetection) => { cc[d.class_name] = (cc[d.class_name] || 0) + 1 })
        setTestSummary({
          total_images: 1, total_detections: res.data.count,
          avg_detections_per_image: res.data.count,
          images_with_detections: res.data.count > 0 ? 1 : 0,
          class_counts: cc,
        })
      } else {
        testFiles.forEach(f => fd.append('files', f))
        fd.append('iou', '0.45')
        const res = await api.post(
          `/projects/${projectId}/training/runs/${testRunId}/test-batch`, fd,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        )
        setTestResults(res.data.images)
        setTestSummary(res.data.summary)
      }
    } catch (e: any) {
      alert('Test failed: ' + (e?.response?.data?.detail ?? e.message))
    } finally {
      setTestRunning(false)
    }
  }

  const statusBadge = (s: string) => {
    if (s === 'done')    return <Badge color="green">Done</Badge>
    if (s === 'failed')  return <Badge color="red">Failed</Badge>
    if (s === 'stopped') return <Badge color="gray">Stopped</Badge>
    if (s === 'running') return <Badge color="yellow">Running</Badge>
    return <Badge color="gray">Pending</Badge>
  }

  const S = ({ label, value, clr = 'var(--text)' }: { label: string; value: string; clr?: string }) => (
    <div style={{ textAlign: 'center' }}>
      <p style={{ fontSize: 11, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 4 }}>{label}</p>
      <p style={{ fontSize: 15, fontWeight: 600, color: clr, fontFamily: 'JetBrains Mono, monospace' }}>{value}</p>
    </div>
  )

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>

      {/* ── Aug Preview Modal ── */}
      {showAugModal && (
        <div style={{
          position: 'fixed', inset: 0, zIndex: 200,
          background: 'rgba(0,0,0,0.72)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }} onClick={() => setShowAugModal(false)}>
          <div style={{
            background: 'var(--surface)', border: '1px solid var(--border)',
            borderRadius: 12, padding: 20, maxWidth: 680, width: '90%',
            maxHeight: '85vh', overflowY: 'auto',
          }} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)' }}>Augmentation Preview</p>
              <button onClick={() => setShowAugModal(false)}
                style={{ border: 'none', background: 'transparent', color: 'var(--text3)', cursor: 'pointer', padding: 4 }}>
                <X size={16} />
              </button>
            </div>
            {augPreviews.length === 0
              ? <p style={{ fontSize: 12, color: 'var(--text3)' }}>No annotated images found in project.</p>
              : <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 8 }}>
                  {augPreviews.map((src, i) => (
                    <img key={i} src={src} alt={`aug-${i}`}
                      style={{ width: '100%', borderRadius: 6, display: 'block', border: '1px solid var(--border)' }} />
                  ))}
                </div>}
            <p style={{ fontSize: 10, color: 'var(--text3)', marginTop: 10 }}>
              Each image is sampled from your annotated dataset and has your current augmentation settings applied. Results vary on each preview.
            </p>
          </div>
        </div>
      )}

      <PageHeader back={() => navigate(`/projects/${projectId}/images`)}
        title={modelBase.includes('-seg') ? 'Instance Segmentation Training' : 'Object Detection Training'}
        subtitle={project?.name} />

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 16 }}>

        {/* ── Config ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Model</p>

            <Field label={
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                Architecture
                <span style={{
                  fontSize: 9, fontWeight: 700, letterSpacing: '0.06em',
                  padding: '2px 6px', borderRadius: 3,
                  background: modelBase.includes('-seg') ? 'rgba(34,197,94,0.12)' : 'rgba(139,92,246,0.12)',
                  color: modelBase.includes('-seg') ? '#22c55e' : 'var(--accent)',
                }}>
                  {modelBase.includes('-seg') ? 'SEGMENT' : 'DETECT'}
                </span>
              </span>
            }>
              <Select value={modelBase} onChange={setModelBase}>
                <optgroup label="YOLO11 (latest)">
                  <option value="yolo11n.pt">YOLO11 Nano</option>
                  <option value="yolo11s.pt">YOLO11 Small</option>
                  <option value="yolo11m.pt">YOLO11 Medium</option>
                  <option value="yolo11l.pt">YOLO11 Large</option>
                  <option value="yolo11x.pt">YOLO11 XLarge</option>
                </optgroup>
                <optgroup label="YOLOv10">
                  <option value="yolov10n.pt">YOLOv10 Nano</option>
                  <option value="yolov10s.pt">YOLOv10 Small</option>
                  <option value="yolov10m.pt">YOLOv10 Medium</option>
                  <option value="yolov10b.pt">YOLOv10 Base</option>
                  <option value="yolov10l.pt">YOLOv10 Large</option>
                  <option value="yolov10x.pt">YOLOv10 XLarge</option>
                </optgroup>
                <optgroup label="YOLOv9">
                  <option value="yolov9t.pt">YOLOv9 Tiny</option>
                  <option value="yolov9s.pt">YOLOv9 Small</option>
                  <option value="yolov9m.pt">YOLOv9 Medium</option>
                  <option value="yolov9c.pt">YOLOv9 Compact</option>
                  <option value="yolov9e.pt">YOLOv9 Extended</option>
                </optgroup>
                <optgroup label="YOLOv8">
                  <option value="yolov8n.pt">YOLOv8 Nano</option>
                  <option value="yolov8s.pt">YOLOv8 Small</option>
                  <option value="yolov8m.pt">YOLOv8 Medium</option>
                  <option value="yolov8l.pt">YOLOv8 Large</option>
                  <option value="yolov8x.pt">YOLOv8 XLarge</option>
                </optgroup>
                <optgroup label="— Segmentation —">
                  <option value="yolo11n-seg.pt">YOLO11 Nano Seg</option>
                  <option value="yolo11s-seg.pt">YOLO11 Small Seg</option>
                  <option value="yolo11m-seg.pt">YOLO11 Medium Seg</option>
                  <option value="yolov8n-seg.pt">YOLOv8 Nano Seg</option>
                  <option value="yolov8s-seg.pt">YOLOv8 Small Seg</option>
                  <option value="yolov8m-seg.pt">YOLOv8 Medium Seg</option>
                </optgroup>
              </Select>
            </Field>

            <Slider label="Epochs" value={epochs} onChange={setEpochs} min={1} max={300} step={1} />
            <Field label="Image Size">
              <Select value={imgsz} onChange={v => setImgsz(Number(v))}>
                {[320, 416, 640, 1280].map(s => <option key={s} value={s}>{s}px{s===640?' — default':''}</option>)}
              </Select>
            </Field>
            <Slider label="Batch Size" value={batch} onChange={setBatch} min={1} max={64} step={1} />
            <Slider label="Validation Split" value={valSplit} onChange={setValSplit} min={0.1} max={0.4}
              format={v => `${Math.round(v * 100)}%`} />
            <Slider label="Early Stop Patience" value={patience} onChange={setPatience} min={0} max={100} step={1}
              format={v => v === 0 ? 'Off' : `${v} ep`} />

            <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginTop: 4 }}>Optimizer</p>
            <Field label="Algorithm">
              <Select value={optimizer} onChange={setOptimizer}>
                <option value="auto">Auto (recommended)</option>
                <option value="SGD">SGD</option>
                <option value="Adam">Adam</option>
                <option value="AdamW">AdamW</option>
                <option value="NAdam">NAdam</option>
                <option value="RAdam">RAdam</option>
                <option value="RMSProp">RMSProp</option>
              </Select>
            </Field>
            <Slider label="Learning Rate (lr0)" value={lr0} onChange={setLr0} min={0.0001} max={0.1} step={0.0001} format={v => v.toFixed(4)} />
            <Slider label="Final LR (lrf)" value={lrf} onChange={setLrf} min={0.0001} max={0.1} step={0.0001} format={v => v.toFixed(4)} />
            <Slider label="Momentum" value={momentum} onChange={setMomentum} min={0.6} max={0.98} step={0.001} format={v => v.toFixed(3)} />
            <Slider label="Weight Decay" value={weightDecay} onChange={setWeightDecay} min={0} max={0.001} step={0.00001} format={v => v.toFixed(5)} />
            <Slider label="Warmup Epochs" value={warmupEpochs} onChange={setWarmupEpochs} min={0} max={10} step={0.5} format={v => `${v}`} />
          </Card>

          {/* Augmentation */}
          <Card>
            <button onClick={() => setShowAug(v => !v)}
              style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                padding: '12px 16px', background: 'transparent', border: 'none', cursor: 'pointer',
                color: 'var(--text2)', fontSize: 12, fontWeight: 600 }}>
              <span style={{ textTransform: 'uppercase', letterSpacing: '0.07em', fontSize: 11, color: 'var(--text3)' }}>Augmentation</span>
              {showAug ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
            </button>
            {showAug && (
              <div style={{ padding: '0 16px 16px', display: 'flex', flexDirection: 'column', gap: 10,
                borderTop: '1px solid var(--border)' }}>
                <div style={{ marginTop: 12 }}>
                  <p style={{ fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Geometric</p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <Slider label="Flip LR"      value={fliplr}      onChange={setFliplr}      min={0} max={1} />
                    <Slider label="Flip UD"      value={flipud}      onChange={setFlipud}      min={0} max={1} />
                    <Slider label="Rotation"     value={degrees}     onChange={setDegrees}     min={0} max={180} step={1} format={v => `${v}°`} />
                    <Slider label="Translate"    value={translate}   onChange={setTranslate}   min={0} max={0.5} />
                    <Slider label="Scale"        value={scale}       onChange={setScale}       min={0} max={0.9} />
                    <Slider label="Shear"        value={shear}       onChange={setShear}       min={0} max={45} step={0.5} format={v => `${v}°`} />
                    <Slider label="Perspective"  value={perspective} onChange={setPerspective} min={0} max={0.001} step={0.0001} format={v => v.toFixed(4)} />
                  </div>
                </div>
                <div>
                  <p style={{ fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Color (HSV)</p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <Slider label="Hue"        value={hsvH} onChange={setHsvH} min={0} max={0.1} step={0.005} />
                    <Slider label="Saturation" value={hsvS} onChange={setHsvS} min={0} max={1} />
                    <Slider label="Brightness" value={hsvV} onChange={setHsvV} min={0} max={1} />
                  </div>
                </div>
                <div>
                  <p style={{ fontSize: 10, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 8 }}>Mixing & Cutout</p>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <Slider label="Mosaic"      value={mosaic}     onChange={setMosaic}     min={0} max={1} />
                    <Slider label="Mixup"       value={mixup}      onChange={setMixup}      min={0} max={1} />
                    <Slider label="Copy-Paste"  value={copyPaste}  onChange={setCopyPaste}  min={0} max={1} />
                    <Slider label="Erasing"     value={erasing}    onChange={setErasing}    min={0} max={0.9} />
                  </div>
                </div>
                <Btn variant="secondary" onClick={runAugPreview} disabled={augPreviewing}
                  style={{ width: '100%', justifyContent: 'center', marginTop: 4 }}>
                  {augPreviewing
                    ? <><Loader size={12} className="animate-spin" /> Generating…</>
                    : <><Eye size={12} /> Preview Augmentation</>}
                </Btn>
              </div>
            )}
          </Card>

          {runs.filter(r => r.status === 'done').length > 0 && (
            <Field label="Resume from run (optional)">
              <Select value={resumeRunId} onChange={setResumeRunId}>
                <option value="">— Start fresh —</option>
                {runs.filter(r => r.status === 'done').map(r => (
                  <option key={r.id} value={r.id}>
                    Run #{r.id} · {r.model_base} · {r.epochs} ep
                  </option>
                ))}
              </Select>
            </Field>
          )}

          <Btn variant="primary" onClick={startTraining} disabled={streaming}
            style={{ width: '100%', justifyContent: 'center', padding: '9px 14px' }}>
            {streaming ? <><Loader size={13} className="animate-spin" /> Training…</> : <><Zap size={13} strokeWidth={2.5} /> Start Training</>}
          </Btn>
        </div>

        {/* ── Right ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Progress */}
          {(streaming || progress) && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
                <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', letterSpacing: '0.01em' }}>Training Progress</p>
                {streaming
                  ? <Badge color="yellow"><Loader size={10} className="animate-spin" /> Running</Badge>
                  : <Badge color="green">Complete</Badge>}
              </div>
              <ProgressBar value={progress?.epoch ?? 0} max={progress?.total ?? epochs}
                label={`Epoch ${progress?.epoch ?? 0} of ${progress?.total ?? epochs}`} />
              {progress && progress.mAP50 > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 8, marginTop: 12 }}>
                  <S label="mAP50"     value={`${(progress.mAP50 * 100).toFixed(1)}%`}     clr="var(--accent)" />
                  <S label="Precision" value={`${(progress.precision * 100).toFixed(1)}%`}  clr="var(--success)" />
                  <S label="Recall"    value={`${(progress.recall * 100).toFixed(1)}%`}     clr="var(--warn)" />
                </div>
              )}
            </Card>
          )}

          {/* Chart */}
          {chartData.length > 1 && (
            <Card style={{ padding: 16 }}>
              <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 14 }}>Metrics</p>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="epoch" stroke="var(--text3)" tick={{ fontSize: 10 }} />
                  <YAxis stroke="var(--text3)" tick={{ fontSize: 10 }} domain={[0, 1]} />
                  <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11 }}
                    formatter={(v: unknown) => [((v as number) * 100).toFixed(2) + '%']} />
                  <Line type="monotone" dataKey="mAP50"     stroke="var(--accent)" strokeWidth={2}   dot={false} name="mAP50" />
                  <Line type="monotone" dataKey="precision" stroke="var(--success)" strokeWidth={1.5} dot={false} name="Precision" strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="recall"    stroke="var(--warn)"    strokeWidth={1.5} dot={false} name="Recall"    strokeDasharray="4 2" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Log */}
          <Card style={{ padding: 16 }}>
            <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 10 }}>Console</p>
            <LogTerminal logs={logs} logRef={logRef} />
          </Card>

          {/* Run comparison */}
          {runs.filter(r => r.status === 'done' && r.results.mAP50 > 0).length > 1 && (
            <Card style={{ padding: 16 }}>
              <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 14 }}>Run Comparison — mAP50</p>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={runs.filter(r => r.status === 'done').map(r => ({
                  name: `#${r.id}`, mAP50: r.results.mAP50 ? r.results.mAP50 * 100 : 0
                }))} margin={{ left: -16 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
                  <XAxis dataKey="name" stroke="var(--text3)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--text3)" tick={{ fontSize: 11 }} domain={[0, 100]}
                    tickFormatter={v => `${v}%`} />
                  <Tooltip contentStyle={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11 }}
                    formatter={(v: unknown) => [`${(v as number).toFixed(1)}%`, 'mAP50']} />
                  <Bar dataKey="mAP50" radius={[3, 3, 0, 0]}>
                    {runs.filter(r => r.status === 'done').map((_, i) => (
                      <Cell key={i} fill={['#5865f2','#22c55e','#f59e0b','#06b6d4','#a855f7','#ec4899'][i % 6]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Runs */}
          <Card style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
              <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Run History</p>
              <button onClick={loadRuns} style={{ border: 'none', background: 'transparent', color: 'var(--text3)', cursor: 'pointer', padding: 4, borderRadius: 4 }}>
                <RefreshCw size={12} />
              </button>
            </div>
            {runs.length === 0
              ? <p style={{ fontSize: 12, color: 'var(--text3)' }}>No runs yet.</p>
              : <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {[...runs].reverse().map(run => {
                    const onnxSt = onnxStatus[run.id] ?? (run.onnx_path ? 'done' : 'idle')
                    return (
                      <div key={run.id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                        padding: '10px 12px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 6 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                          {statusBadge(run.status)}
                          <div>
                            <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text)' }}>Run #{run.id}</span>
                            <span style={{ fontSize: 11, color: 'var(--text3)', marginLeft: 8 }}>{run.model_base} · {run.epochs} ep</span>
                          </div>
                        </div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                          {run.results.mAP50 !== undefined && run.results.mAP50 > 0 && (
                            <span style={{ fontSize: 12, color: 'var(--text2)', fontFamily: 'JetBrains Mono, monospace' }}>
                              {(run.results.mAP50 * 100).toFixed(1)}%
                            </span>
                          )}
                          {run.status === 'running' && (
                            <Btn variant="ghost" size="sm" disabled={stoppingRun.has(run.id)} onClick={() => stopRun(run.id)}
                              style={{ color: 'var(--red, #f87171)' }}>
                              {stoppingRun.has(run.id)
                                ? <><Loader size={10} className="animate-spin" /> Stopping</>
                                : <><Square size={10} /> Stop</>}
                            </Btn>
                          )}
                          {run.status !== 'running' && (
                            <Btn variant="ghost" size="sm" disabled={deletingRun.has(run.id)} onClick={() => deleteRun(run.id)}
                              style={{ color: 'var(--text3)' }}>
                              {deletingRun.has(run.id)
                                ? <Loader size={10} className="animate-spin" />
                                : <Trash2 size={10} />}
                            </Btn>
                          )}
                          {run.status === 'done' && (
                            <Btn variant="ghost" size="sm" onClick={() => navigate(`/projects/${projectId}/eval/${run.id}`)}>
                              <BarChart2 size={11} /> Eval
                            </Btn>
                          )}
                          {run.status === 'done' && (
                            <Btn variant="ghost" size="sm" onClick={() => navigate(`/projects/${projectId}/webcam`)}>
                              <Camera size={11} /> Live
                            </Btn>
                          )}
                          {run.status === 'done' && run.model_path && (
                            <Btn variant="secondary" size="sm" href={`/api/projects/${projectId}/training/runs/${run.id}/download`}>
                              <Download size={11} /> .pt
                            </Btn>
                          )}
                          {run.status === 'done' && (() => {
                            const tflSt = tfliteStatus[run.id] ?? 'idle'
                            const trtSt = trtStatus[run.id]    ?? 'idle'
                            return (<>
                              {onnxSt === 'done'
                                ? <Btn variant="secondary" size="sm" href={`/api/projects/${projectId}/training/runs/${run.id}/export/onnx/download`}>
                                    <Download size={11} /> .onnx
                                  </Btn>
                                : <Btn variant="ghost" size="sm" disabled={onnxSt === 'running'} onClick={() => exportOnnx(run.id)}>
                                    {onnxSt === 'running' ? <><Loader size={10} className="animate-spin" /> Exporting</> : <><FileCode size={11} /> ONNX</>}
                                  </Btn>}
                              {tflSt === 'done'
                                ? <Btn variant="secondary" size="sm" href={`/api/projects/${projectId}/training/runs/${run.id}/export/tflite/download`}>
                                    <Download size={11} /> .tflite
                                  </Btn>
                                : <Btn variant="ghost" size="sm" disabled={tflSt === 'running'} onClick={() => startExport(run.id, 'tflite', setTfliteStatus)}>
                                    {tflSt === 'running' ? <><Loader size={10} className="animate-spin" /> TFLite</> : 'TFLite'}
                                  </Btn>}
                              {trtSt === 'done'
                                ? <Btn variant="secondary" size="sm" href={`/api/projects/${projectId}/training/runs/${run.id}/export/tensorrt/download`}>
                                    <Download size={11} /> .engine
                                  </Btn>
                                : <Btn variant="ghost" size="sm" disabled={trtSt === 'running'} onClick={() => startExport(run.id, 'tensorrt', setTrtStatus)}>
                                    {trtSt === 'running' ? <><Loader size={10} className="animate-spin" /> TRT</> : <><Cpu size={10} /> TRT</>}
                                  </Btn>}
                            </>)
                          })()}
                        </div>
                      </div>
                    )
                  })}
                </div>}
          </Card>

          {/* ── Test Set ── */}
          {runs.some(r => r.status === 'done') && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                <FlaskConical size={14} style={{ color: 'var(--accent)' }} />
                <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Test Set Evaluation</p>
              </div>

              {/* Run selector */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <div style={{ display: 'flex', gap: 8 }}>
                  <select
                    value={testRunId}
                    onChange={e => { setTestRunId(e.target.value); setTestResults(null); setTestSummary(null) }}
                    style={{
                      flex: 1, padding: '7px 10px',
                      background: 'var(--surface2)', border: '1px solid var(--border)',
                      borderRadius: 6, color: 'var(--text)', fontSize: 12,
                      fontFamily: 'inherit', cursor: 'pointer',
                    }}
                  >
                    <option value="">Select a run…</option>
                    {runs.filter(r => r.status === 'done').map(r => (
                      <option key={r.id} value={r.id}>
                        Run #{r.id} — {r.model_base} · {r.epochs} ep
                        {r.results?.mAP50 ? ` · mAP50 ${(r.results.mAP50 * 100).toFixed(1)}%` : ''}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Confidence */}
                <Slider label="Confidence threshold" value={testConf} onChange={setTestConf}
                  min={0.05} max={0.95} step={0.05} format={v => `${Math.round(v * 100)}%`} />

                {/* File / URL toggle */}
                <div style={{ display: 'flex', gap: 4, background: 'var(--surface2)', padding: 3, borderRadius: 7 }}>
                  {(['file','url'] as const).map(m => (
                    <button key={m} onClick={() => { setTestInputMode(m); setTestResults(null); setTestSummary(null) }}
                      style={{
                        flex: 1, padding: '4px 8px', border: 'none', borderRadius: 5, cursor: 'pointer',
                        fontSize: 11, fontWeight: 500, fontFamily: 'inherit',
                        background: testInputMode === m ? 'var(--surface)' : 'transparent',
                        color: testInputMode === m ? 'var(--text)' : 'var(--text3)',
                        transition: 'all 0.15s',
                      }}>
                      {m === 'file' ? 'Upload Files' : 'Paste URL'}
                    </button>
                  ))}
                </div>

                {testInputMode === 'file' ? (
                  <>
                    <div
                      onClick={() => testFileRef.current?.click()}
                      style={{
                        border: `2px dashed ${testFiles.length > 0 ? 'var(--accent)' : 'var(--border2)'}`,
                        borderRadius: 8, padding: '14px 12px',
                        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
                        cursor: 'pointer', transition: 'border-color 0.15s',
                        background: testFiles.length > 0 ? 'var(--accent-t)' : 'transparent',
                      }}
                    >
                      <Upload size={18} style={{ color: testFiles.length > 0 ? 'var(--accent)' : 'var(--text3)' }} />
                      {testFiles.length === 0 ? (
                        <span style={{ fontSize: 12, color: 'var(--text3)' }}>Click to upload test images</span>
                      ) : (
                        <span style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 500 }}>
                          {testFiles.length} image{testFiles.length !== 1 ? 's' : ''} selected
                        </span>
                      )}
                      <span style={{ fontSize: 11, color: 'var(--text3)' }}>JPG, PNG, BMP — multiple allowed</span>
                    </div>
                    <input
                      ref={testFileRef} type="file" multiple accept="image/*"
                      style={{ display: 'none' }}
                      onChange={e => {
                        const files = Array.from(e.target.files ?? [])
                        setTestFiles(files)
                        setTestResults(null); setTestSummary(null)
                      }}
                    />
                  </>
                ) : (
                  <input
                    type="url"
                    placeholder="https://example.com/image.jpg"
                    value={testUrl}
                    onChange={e => { setTestUrl(e.target.value); setTestResults(null); setTestSummary(null) }}
                    style={{
                      width: '100%', padding: '8px 10px', boxSizing: 'border-box',
                      background: 'var(--surface2)', border: '1px solid var(--border)',
                      borderRadius: 6, color: 'var(--text)', fontSize: 12, fontFamily: 'inherit',
                    }}
                  />
                )}

                {/* Clear + Run */}
                <div style={{ display: 'flex', gap: 8 }}>
                  {(testInputMode === 'file' ? testFiles.length > 0 : testUrl.trim().length > 0) && (
                    <Btn variant="ghost" size="sm" onClick={() => {
                      setTestFiles([]); setTestUrl(''); setTestResults(null); setTestSummary(null)
                    }}>
                      <X size={12} /> Clear
                    </Btn>
                  )}
                  <Btn
                    variant="primary" size="sm"
                    disabled={!testRunId || (testInputMode === 'file' ? testFiles.length === 0 : !testUrl.trim()) || testRunning}
                    onClick={runBatchTest}
                    style={{ flex: 1, justifyContent: 'center' }}
                  >
                    {testRunning
                      ? <><Loader size={12} className="animate-spin" /> Running…</>
                      : <><FlaskConical size={12} /> Run Test</>}
                  </Btn>
                </div>
              </div>

              {/* ── Summary ── */}
              {testSummary && (
                <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)' }}>
                  <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>
                    Summary
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8, marginBottom: 12 }}>
                    {[
                      { label: 'Images tested',    value: testSummary.total_images },
                      { label: 'Total detections', value: testSummary.total_detections },
                      { label: 'Avg / image',      value: testSummary.avg_detections_per_image.toFixed(1) },
                      { label: 'With detections',  value: testSummary.images_with_detections },
                    ].map(({ label, value }) => (
                      <div key={label} style={{
                        background: 'var(--surface3)', borderRadius: 6,
                        padding: '10px 12px', textAlign: 'center',
                      }}>
                        <p style={{ fontSize: 18, fontWeight: 700, color: 'var(--text)', fontFamily: 'JetBrains Mono, monospace', fontVariantNumeric: 'tabular-nums' }}>{value}</p>
                        <p style={{ fontSize: 10, color: 'var(--text3)', marginTop: 3 }}>{label}</p>
                      </div>
                    ))}
                  </div>

                  {/* Class breakdown */}
                  {Object.keys(testSummary.class_counts).length > 0 && (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                      <p style={{ fontSize: 10, color: 'var(--text3)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>By class</p>
                      {Object.entries(testSummary.class_counts)
                        .sort((a, b) => b[1] - a[1])
                        .map(([cls, count], i) => {
                          const total = testSummary.total_detections
                          const pct = total > 0 ? (count / total) * 100 : 0
                          return (
                            <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                              <div style={{ width: 8, height: 8, borderRadius: '50%', background: getClsColor(i), flexShrink: 0 }} />
                              <span style={{ fontSize: 12, color: 'var(--text2)', flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{cls}</span>
                              <div style={{ width: 80, height: 4, background: 'var(--surface3)', borderRadius: 99, overflow: 'hidden', flexShrink: 0 }}>
                                <div style={{ width: `${pct}%`, height: '100%', background: getClsColor(i), borderRadius: 99 }} />
                              </div>
                              <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'JetBrains Mono, monospace', flexShrink: 0, minWidth: 24, textAlign: 'right' }}>{count}</span>
                            </div>
                          )
                        })}
                    </div>
                  )}
                </div>
              )}

              {/* ── Image grid with overlays ── */}
              {testResults && testResults.length > 0 && (
                <div style={{ marginTop: 16, paddingTop: 14, borderTop: '1px solid var(--border)' }}>
                  <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 10 }}>
                    Results
                  </p>
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
                    gap: 8,
                    maxHeight: 480, overflowY: 'auto',
                  }}>
                    {testResults.map((result, i) => (
                      <TestImageCard key={i}
                        file={testInputMode === 'file' ? testFiles[i] : undefined}
                        src={testInputMode === 'url' ? testUrlDisplay : undefined}
                        result={result} />
                    ))}
                  </div>
                </div>
              )}
            </Card>
          )}

          {/* ── Video Inference ── */}
          {runs.some(r => r.status === 'done') && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14 }}>
                <Video size={14} style={{ color: 'var(--accent)' }} />
                <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Video Inference</p>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                <select
                  value={videoRunId}
                  onChange={e => { setVideoRunId(e.target.value); setVideoStatus('idle'); setVideoJobId('') }}
                  style={{
                    padding: '7px 10px', background: 'var(--surface2)',
                    border: '1px solid var(--border)', borderRadius: 6,
                    color: 'var(--text)', fontSize: 12, fontFamily: 'inherit', cursor: 'pointer',
                  }}
                >
                  <option value="">Select a run…</option>
                  {runs.filter(r => r.status === 'done').map(r => (
                    <option key={r.id} value={r.id}>
                      Run #{r.id} — {r.model_base} · {r.epochs} ep
                    </option>
                  ))}
                </select>

                <Slider label="Confidence threshold" value={videoConf} onChange={setVideoConf}
                  min={0.05} max={0.95} step={0.05} format={v => `${Math.round(v * 100)}%`} />

                {/* Tracker toggle */}
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <p style={{ fontSize: 12, color: 'var(--text2)', fontWeight: 500 }}>Object Tracking</p>
                    <p style={{ fontSize: 10, color: 'var(--text3)', marginTop: 2 }}>Assign persistent IDs across frames</p>
                  </div>
                  <button
                    onClick={() => setVideoTracker(v => !v)}
                    style={{
                      width: 36, height: 20, borderRadius: 10, border: 'none', cursor: 'pointer',
                      background: videoTracker ? 'var(--accent)' : 'var(--surface3)',
                      position: 'relative', transition: 'background 0.2s', flexShrink: 0,
                    }}
                  >
                    <span style={{
                      position: 'absolute', top: 3, left: videoTracker ? 18 : 3,
                      width: 14, height: 14, borderRadius: '50%',
                      background: '#fff', transition: 'left 0.2s',
                    }} />
                  </button>
                </div>

                {/* File / URL toggle */}
                <div style={{ display: 'flex', gap: 4, background: 'var(--surface2)', padding: 3, borderRadius: 7 }}>
                  {(['file','url'] as const).map(m => (
                    <button key={m} onClick={() => { setVideoInputMode(m); setVideoStatus('idle'); setVideoJobId('') }}
                      style={{
                        flex: 1, padding: '5px 8px', border: 'none', borderRadius: 5, cursor: 'pointer',
                        fontSize: 11, fontWeight: 500, fontFamily: 'inherit',
                        background: videoInputMode === m ? 'var(--surface)' : 'transparent',
                        color: videoInputMode === m ? 'var(--text)' : 'var(--text3)',
                        boxShadow: videoInputMode === m ? 'var(--shadow-xs)' : 'none',
                        transition: 'all 0.15s',
                      }}>
                      {m === 'file' ? 'Upload File' : 'Paste URL'}
                    </button>
                  ))}
                </div>

                {videoInputMode === 'file' ? (
                  <>
                    <div onClick={() => videoFileRef.current?.click()} style={{
                      border: `2px dashed ${videoFile ? 'var(--accent)' : 'var(--border2)'}`,
                      borderRadius: 8, padding: '14px 12px',
                      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6,
                      cursor: 'pointer', background: videoFile ? 'var(--accent-t)' : 'transparent',
                    }}>
                      <Video size={18} style={{ color: videoFile ? 'var(--accent)' : 'var(--text3)' }} />
                      {!videoFile
                        ? <span style={{ fontSize: 12, color: 'var(--text3)' }}>Click to upload video</span>
                        : <span style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 500 }}>{videoFile.name}</span>}
                      <span style={{ fontSize: 11, color: 'var(--text3)' }}>MP4, AVI, MOV, MKV</span>
                    </div>
                    <input ref={videoFileRef} type="file" accept="video/*" style={{ display: 'none' }}
                      onChange={e => { setVideoFile(e.target.files?.[0] ?? null); setVideoStatus('idle'); setVideoJobId('') }} />
                  </>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    <input
                      type="url"
                      placeholder="https://youtube.com/watch?v=... or any video URL"
                      value={videoUrl}
                      onChange={e => { setVideoUrl(e.target.value); setVideoStatus('idle') }}
                      style={{
                        padding: '8px 10px', background: 'var(--surface2)',
                        border: `1px solid ${videoUrl ? 'var(--accent)' : 'var(--border)'}`,
                        borderRadius: 6, color: 'var(--text)', fontSize: 12,
                        fontFamily: 'inherit', outline: 'none', width: '100%', boxSizing: 'border-box',
                      }}
                    />
                    <p style={{ fontSize: 10, color: 'var(--text3)' }}>
                      YouTube, Vimeo, Twitter, TikTok, and 1000+ sites supported via yt-dlp.
                      Video is capped at 720p.
                    </p>
                  </div>
                )}

                {/* Progress bar when running */}
                {(videoStatus === 'running' || videoStatus === 'uploading') && (
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span style={{ fontSize: 11, color: 'var(--text3)' }}>
                        {videoStatus === 'uploading' ? (videoInputMode === 'url' ? 'Downloading video…' : 'Uploading…') : 'Processing frames…'}
                      </span>
                      {videoStatus === 'running' && videoProgress.total > 0 && (
                        <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'JetBrains Mono, monospace' }}>
                          {videoProgress.processed} / {videoProgress.total}
                        </span>
                      )}
                    </div>
                    <ProgressBar
                      value={videoProgress.processed}
                      max={videoProgress.total || 1}
                      label=""
                    />
                  </div>
                )}

                <div style={{ display: 'flex', gap: 8 }}>
                  {(videoFile || videoUrl) && videoStatus !== 'running' && videoStatus !== 'uploading' && (
                    <Btn variant="ghost" size="sm" onClick={() => {
                      setVideoFile(null); setVideoUrl(''); setVideoStatus('idle')
                    }}>
                      <X size={12} /> Clear
                    </Btn>
                  )}
                  <Btn
                    variant="primary" size="sm"
                    disabled={!videoRunId || (videoInputMode === 'file' ? !videoFile : !videoUrl.trim()) || videoStatus === 'running' || videoStatus === 'uploading'}
                    onClick={startVideoInfer}
                    style={{ flex: 1, justifyContent: 'center' }}
                  >
                    {videoStatus === 'uploading' || videoStatus === 'running'
                      ? <><Loader size={12} className="animate-spin" /> Processing…</>
                      : <><Video size={12} /> Run Inference</>}
                  </Btn>
                  {videoStatus === 'done' && videoJobId && (
                    <Btn variant="secondary" size="sm"
                      href={`/api/projects/${projectId}/training/runs/${videoRunId}/video-infer/${videoJobId}/download`}>
                      <Download size={12} /> Download
                    </Btn>
                  )}
                </div>

                {videoStatus === 'done' && (
                  <p style={{ fontSize: 11, color: 'var(--success)' }}>
                    Processing complete — {videoProgress.processed} frames annotated.
                  </p>
                )}
                {videoStatus === 'failed' && (
                  <p style={{ fontSize: 11, color: 'var(--red, #f87171)' }}>
                    Processing failed. Check that OpenCV is installed on the server.
                  </p>
                )}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
