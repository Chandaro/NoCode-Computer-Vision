import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Upload, Download, Loader, Camera, CameraOff, Video, Image as ImageIcon, Layers } from 'lucide-react'
import api, { type Project } from '../api'
import { PageHeader, Card, Field, Select, Slider, Btn, ProgressBar } from '../components/ui'

// 20-color palette for class instances
const CLASS_COLORS = [
  '#8b5cf6','#22c55e','#f59e0b','#38bdf8','#f87171',
  '#ec4899','#a3e635','#fb923c','#34d399','#c084fc',
  '#818cf8','#4ade80','#fbbf24','#67e8f9','#fca5a5',
  '#f9a8d4','#bef264','#fdba74','#6ee7b7','#e879f9',
]
function getClsColor(id: number) { return CLASS_COLORS[id % CLASS_COLORS.length] }

interface Detection {
  x: number; y: number; w: number; h: number
  conf: number; class_id: number; class_name: string
  mask: number[][]
}

// ── SVG mask polygon overlay ──────────────────────────────────────────────────
function MaskOverlay({ detections, width, height }: {
  detections: Detection[]; width: number; height: number
}) {
  return (
    <svg
      style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
    >
      {detections.map((d, i) => {
        if (!d.mask || d.mask.length === 0) return null
        const color = getClsColor(d.class_id)
        const pts = d.mask.map(([x, y]) => `${x * width},${y * height}`).join(' ')
        return (
          <g key={i}>
            <polygon points={pts} fill={color} fillOpacity={0.3}
              stroke={color} strokeWidth={1.5} strokeLinejoin="round" />
          </g>
        )
      })}
    </svg>
  )
}

type Mode = 'image' | 'video' | 'webcam'
const INFER_MS = 150

export default function Segmentation() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project,   setProject]   = useState<Project | null>(null)
  const [mode,      setMode]      = useState<Mode>('image')
  const [modelName, setModelName] = useState('yolo11n-seg.pt')
  const [conf,      setConf]      = useState(0.25)

  // ── Image mode ───────────────────────────────────────────────────────────────
  const [imgInputMode, setImgInputMode] = useState<'file'|'url'>('file')
  const [imgUrl,       setImgUrl]       = useState('')
  const [imgSrc,       setImgSrc]       = useState('')
  const [imgDets,      setImgDets]      = useState<Detection[]>([])
  const [imgRendered,  setImgRendered]  = useState({ w: 1, h: 1 })
  const [imgRunning,   setImgRunning]   = useState(false)
  const imgFileRef = useRef<HTMLInputElement>(null)

  // ── Video mode ───────────────────────────────────────────────────────────────
  const [videoFile,      setVideoFile]      = useState<File | null>(null)
  const [videoInputMode, setVideoInputMode] = useState<'file'|'url'>('file')
  const [videoUrl,       setVideoUrl]       = useState('')
  const [videoStatus,    setVideoStatus]    = useState<'idle'|'uploading'|'running'|'done'|'failed'>('idle')
  const [videoJobId,     setVideoJobId]     = useState('')
  const [videoProgress,  setVideoProgress]  = useState({ processed: 0, total: 0 })
  const videoFileRef   = useRef<HTMLInputElement>(null)
  const videoPollerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Webcam mode ──────────────────────────────────────────────────────────────
  const [camActive,   setCamActive]   = useState(false)
  const [camDets,     setCamDets]     = useState<Detection[]>([])
  const [camFps,      setCamFps]      = useState(0)
  const [camRendered, setCamRendered] = useState({ w: 640, h: 480 })
  const [camError,    setCamError]    = useState('')
  const camVideoRef    = useRef<HTMLVideoElement>(null)
  const camCanvasRef   = useRef<HTMLCanvasElement>(null)
  const camStreamRef   = useRef<MediaStream | null>(null)
  const camPendingRef  = useRef(false)
  const camIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const camFpsCountRef = useRef(0)
  const camFpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    api.get(`/projects/${projectId}`).then(r => setProject(r.data))
  }, [projectId])
  useEffect(() => () => {
    stopCam()
    if (videoPollerRef.current) clearInterval(videoPollerRef.current)
  }, [])

  // ── Image inference ──────────────────────────────────────────────────────────
  const runImageInfer = async (file: File) => {
    setImgSrc(URL.createObjectURL(file)); setImgDets([])
    setImgRunning(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('model_name', modelName)
      fd.append('conf', String(conf))
      const res = await api.post('/segment/infer', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setImgDets(res.data.detections ?? [])
    } catch (e: any) {
      alert('Inference failed: ' + (e?.response?.data?.detail ?? e.message))
    } finally { setImgRunning(false) }
  }

  const runImageInferUrl = async () => {
    if (!imgUrl.trim()) return
    setImgSrc(imgUrl.trim()); setImgDets([])
    setImgRunning(true)
    try {
      const fd = new FormData()
      fd.append('url', imgUrl.trim())
      fd.append('model_name', modelName)
      fd.append('conf', String(conf))
      const res = await api.post('/segment/infer-url', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setImgDets(res.data.detections ?? [])
    } catch (e: any) {
      alert('Inference failed: ' + (e?.response?.data?.detail ?? e.message))
    } finally { setImgRunning(false) }
  }

  // ── Video inference ──────────────────────────────────────────────────────────
  const startVideoInfer = async () => {
    if (videoInputMode === 'file' && !videoFile) return
    if (videoInputMode === 'url'  && !videoUrl.trim()) return
    if (videoPollerRef.current) clearInterval(videoPollerRef.current)
    setVideoStatus('uploading'); setVideoJobId('')
    setVideoProgress({ processed: 0, total: 0 })
    try {
      const fd = new FormData()
      fd.append('model_name', modelName)
      fd.append('conf', String(conf))
      let endpoint: string
      if (videoInputMode === 'url') {
        fd.append('url', videoUrl.trim())
        endpoint = '/segment/video-infer-url'
      } else {
        fd.append('file', videoFile!)
        endpoint = '/segment/video-infer'
      }
      const res = await api.post(endpoint, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      const jobId: string = res.data.job_id
      setVideoJobId(jobId); setVideoStatus('running')
      videoPollerRef.current = setInterval(async () => {
        try {
          const s = await api.get(`/segment/video-infer/${jobId}/status`)
          setVideoProgress({ processed: s.data.processed, total: s.data.total_frames })
          if (s.data.stage) setVideoStatus(s.data.stage === 'downloading' ? 'uploading' : 'running')
          if (s.data.status === 'done') {
            clearInterval(videoPollerRef.current!); setVideoStatus('done')
          } else if (s.data.status === 'failed') {
            clearInterval(videoPollerRef.current!); setVideoStatus('failed')
            alert('Processing failed: ' + (s.data.error ?? 'unknown'))
          }
        } catch { /* keep polling */ }
      }, 1500)
    } catch (e: any) {
      setVideoStatus('failed')
      alert('Upload failed: ' + (e?.response?.data?.detail ?? e.message))
    }
  }

  // ── Webcam ───────────────────────────────────────────────────────────────────
  const updateCamSize = useCallback(() => {
    if (camVideoRef.current) {
      const el = camVideoRef.current
      setCamRendered({ w: el.clientWidth, h: el.clientHeight })
    }
  }, [])
  useEffect(() => {
    window.addEventListener('resize', updateCamSize)
    return () => window.removeEventListener('resize', updateCamSize)
  }, [updateCamSize])

  const captureAndSegment = useCallback(async () => {
    if (camPendingRef.current) return
    if (!camVideoRef.current || !camCanvasRef.current) return
    const video = camVideoRef.current, canvas = camCanvasRef.current
    const scale = Math.min(1, 640 / video.videoWidth)
    canvas.width  = Math.round(video.videoWidth  * scale)
    canvas.height = Math.round(video.videoHeight * scale)
    canvas.getContext('2d')!.drawImage(video, 0, 0, canvas.width, canvas.height)
    camPendingRef.current = true
    canvas.toBlob(async (blob) => {
      if (!blob) { camPendingRef.current = false; return }
      try {
        const fd = new FormData()
        fd.append('frame', blob, 'frame.jpg')
        fd.append('model_name', modelName)
        fd.append('conf', String(conf))
        const res = await api.post('/segment/webcam-frame', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
        setCamDets(res.data.detections ?? [])
        camFpsCountRef.current += 1
        updateCamSize()
      } catch { /* drop frame */ }
      finally { camPendingRef.current = false }
    }, 'image/jpeg', 0.75)
  }, [modelName, conf, updateCamSize])

  const startCamLoop = useCallback(() => {
    if (camIntervalRef.current) clearInterval(camIntervalRef.current)
    camIntervalRef.current = setInterval(captureAndSegment, INFER_MS)
  }, [captureAndSegment])

  useEffect(() => { if (camActive) startCamLoop() }, [conf, modelName, camActive, startCamLoop])

  const startCam = async () => {
    setCamError(''); setCamDets([])
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }
      })
      camStreamRef.current = stream
      if (camVideoRef.current) {
        camVideoRef.current.srcObject = stream
        await camVideoRef.current.play()
        updateCamSize()
      }
      setCamActive(true); startCamLoop()
      camFpsTimerRef.current = setInterval(() => {
        setCamFps(camFpsCountRef.current); camFpsCountRef.current = 0
      }, 1000)
    } catch (e: any) { setCamError('Camera access denied: ' + e.message) }
  }

  const stopCam = () => {
    if (camIntervalRef.current) clearInterval(camIntervalRef.current)
    if (camFpsTimerRef.current) clearInterval(camFpsTimerRef.current)
    camStreamRef.current?.getTracks().forEach(t => t.stop())
    camStreamRef.current = null
    if (camVideoRef.current) camVideoRef.current.srcObject = null
    setCamActive(false); setCamDets([]); setCamFps(0); camPendingRef.current = false
  }

  // ── Derived state ────────────────────────────────────────────────────────────
  const activeDets = mode === 'image' ? imgDets : mode === 'webcam' ? camDets : []
  const classCounts = activeDets.reduce((acc, d) => {
    acc[d.class_name] = (acc[d.class_name] || 0) + 1; return acc
  }, {} as Record<string, number>)

  const MODES: { key: Mode; label: string; icon: React.ReactNode }[] = [
    { key: 'image',  label: 'Image',  icon: <ImageIcon size={13} /> },
    { key: 'video',  label: 'Video',  icon: <Video size={13} /> },
    { key: 'webcam', label: 'Webcam', icon: <Camera size={13} /> },
  ]

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader back={() => navigate(`/projects/${projectId}/images`)}
        title="Instance Segmentation" subtitle={project?.name} />

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 16 }}>

        {/* ── Sidebar ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>

            {/* Mode tabs */}
            <div style={{ display: 'flex', gap: 4, background: 'var(--surface2)', padding: 3, borderRadius: 7 }}>
              {MODES.map(m => (
                <button key={m.key} onClick={() => setMode(m.key)}
                  style={{
                    flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
                    padding: '5px 8px', border: 'none', borderRadius: 5, cursor: 'pointer',
                    fontSize: 11, fontWeight: 500, fontFamily: 'inherit',
                    background: mode === m.key ? 'var(--surface)' : 'transparent',
                    color: mode === m.key ? 'var(--text)' : 'var(--text3)',
                    boxShadow: mode === m.key ? 'var(--shadow-xs)' : 'none',
                    transition: 'all 0.15s',
                  }}>
                  {m.icon}{m.label}
                </button>
              ))}
            </div>

            <Field label="Model">
              <Select value={modelName} onChange={setModelName}>
                <optgroup label="YOLO11 Seg (recommended)">
                  <option value="yolo11n-seg.pt">YOLO11 Nano  · fastest</option>
                  <option value="yolo11s-seg.pt">YOLO11 Small</option>
                  <option value="yolo11m-seg.pt">YOLO11 Medium</option>
                  <option value="yolo11l-seg.pt">YOLO11 Large</option>
                  <option value="yolo11x-seg.pt">YOLO11 XLarge · most accurate</option>
                </optgroup>
                <optgroup label="YOLOv8 Seg">
                  <option value="yolov8n-seg.pt">YOLOv8 Nano</option>
                  <option value="yolov8s-seg.pt">YOLOv8 Small</option>
                  <option value="yolov8m-seg.pt">YOLOv8 Medium</option>
                  <option value="yolov8l-seg.pt">YOLOv8 Large</option>
                </optgroup>
              </Select>
            </Field>

            <Slider label="Confidence" value={conf} onChange={setConf}
              min={0.1} max={0.9} step={0.05} format={v => `${Math.round(v * 100)}%`} />
          </Card>

          {/* Class breakdown */}
          {Object.keys(classCounts).length > 0 && (
            <Card style={{ padding: 14 }}>
              <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>
                {mode === 'webcam' ? 'Live Objects' : 'Detected'}
              </p>
              {Object.entries(classCounts).sort((a, b) => b[1] - a[1]).slice(0, 12).map(([cls, cnt], i) => {
                const clsId = activeDets.find(d => d.class_name === cls)?.class_id ?? i
                return (
                  <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: getClsColor(clsId), flexShrink: 0 }} />
                    <span style={{ fontSize: 11, color: 'var(--text2)', flex: 1 }}>{cls}</span>
                    <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'monospace' }}>{cnt}</span>
                  </div>
                )
              })}
            </Card>
          )}

          {/* Webcam FPS */}
          {mode === 'webcam' && camActive && (
            <Card style={{ padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 11, color: 'var(--text3)' }}>FPS</span>
                <span style={{
                  fontSize: 18, fontWeight: 700, fontFamily: 'monospace',
                  color: camFps >= 5 ? 'var(--success)' : 'var(--warn)',
                }}>{camFps}</span>
              </div>
            </Card>
          )}
        </div>

        {/* ── Main panel ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* IMAGE MODE */}
          {mode === 'image' && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 12 }}>

                {/* File / URL toggle */}
                <div style={{ display: 'flex', gap: 4, background: 'var(--surface2)', padding: 3, borderRadius: 7 }}>
                  {(['file','url'] as const).map(m => (
                    <button key={m} onClick={() => { setImgInputMode(m); setImgSrc(''); setImgDets([]) }}
                      style={{
                        flex: 1, padding: '4px 8px', border: 'none', borderRadius: 5, cursor: 'pointer',
                        fontSize: 11, fontWeight: 500, fontFamily: 'inherit',
                        background: imgInputMode === m ? 'var(--surface)' : 'transparent',
                        color: imgInputMode === m ? 'var(--text)' : 'var(--text3)',
                        transition: 'all 0.15s',
                      }}>
                      {m === 'file' ? 'Upload File' : 'Paste URL'}
                    </button>
                  ))}
                </div>

                {imgInputMode === 'file' ? (
                  <>
                    <Btn variant="primary" onClick={() => imgFileRef.current?.click()} disabled={imgRunning}>
                      {imgRunning
                        ? <><Loader size={12} className="animate-spin" /> Processing…</>
                        : <><Upload size={12} /> Upload Image</>}
                    </Btn>
                    <input ref={imgFileRef} type="file" accept="image/*" style={{ display: 'none' }}
                      onChange={e => { const f = e.target.files?.[0]; if (f) runImageInfer(f) }} />
                  </>
                ) : (
                  <div style={{ display: 'flex', gap: 6 }}>
                    <input
                      type="url" placeholder="https://example.com/image.jpg"
                      value={imgUrl} onChange={e => setImgUrl(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter') runImageInferUrl() }}
                      style={{
                        flex: 1, padding: '8px 10px',
                        background: 'var(--surface2)', border: '1px solid var(--border)',
                        borderRadius: 6, color: 'var(--text)', fontSize: 12, fontFamily: 'inherit',
                      }}
                    />
                    <Btn variant="primary" onClick={runImageInferUrl} disabled={imgRunning || !imgUrl.trim()}>
                      {imgRunning ? <Loader size={12} className="animate-spin" /> : 'Go'}
                    </Btn>
                  </div>
                )}
              </div>

              {imgSrc ? (
                <div style={{ position: 'relative', lineHeight: 0 }}>
                  <img src={imgSrc} alt="segmentation"
                    onLoad={e => {
                      const el = e.currentTarget
                      setImgRendered({ w: el.clientWidth, h: el.clientHeight })
                    }}
                    style={{ width: '100%', borderRadius: 6, display: 'block',
                      border: '1px solid var(--border)' }}
                  />
                  {imgDets.length > 0 && (
                    <MaskOverlay detections={imgDets} width={imgRendered.w} height={imgRendered.h} />
                  )}
                  {/* Bounding box labels */}
                  {imgDets.map((d, i) => (
                    <div key={i} style={{
                      position: 'absolute',
                      left: `${d.x * 100}%`,
                      top: `${d.y * 100}%`,
                      transform: 'translateY(-100%)',
                      background: getClsColor(d.class_id),
                      color: '#fff', fontSize: 10, fontWeight: 600,
                      padding: '1px 5px', borderRadius: 3, whiteSpace: 'nowrap',
                      boxShadow: '0 1px 3px rgba(0,0,0,0.4)',
                      pointerEvents: 'none',
                    }}>
                      {d.class_name} {Math.round(d.conf * 100)}%
                    </div>
                  ))}
                </div>
              ) : (
                <div
                  style={{
                    border: '2px dashed var(--border2)', borderRadius: 8, padding: 48,
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
                    cursor: imgInputMode === 'file' ? 'pointer' : 'default',
                  }}
                  onClick={() => imgInputMode === 'file' && imgFileRef.current?.click()}
                >
                  <Layers size={36} style={{ color: 'var(--text3)', opacity: 0.35 }} />
                  <p style={{ fontSize: 13, color: 'var(--text3)' }}>Upload an image to run segmentation</p>
                  <p style={{ fontSize: 11, color: 'var(--text3)', opacity: 0.6 }}>
                    Uses pre-trained COCO weights — no training required
                  </p>
                </div>
              )}

              {imgDets.length > 0 && (
                <div style={{ marginTop: 10, padding: '8px 12px', background: 'var(--surface2)',
                  borderRadius: 6, fontSize: 12, color: 'var(--text2)' }}>
                  {imgDets.length} object{imgDets.length !== 1 ? 's' : ''} segmented
                  {imgDets.filter(d => d.mask.length === 0).length > 0 && (
                    <span style={{ color: 'var(--warn)', marginLeft: 8 }}>
                      · some masks missing — make sure you selected a -seg model
                    </span>
                  )}
                </div>
              )}
            </Card>
          )}

          {/* VIDEO MODE */}
          {mode === 'video' && (
            <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>

              {/* File / URL toggle */}
              <div style={{ display: 'flex', gap: 4, background: 'var(--surface2)', padding: 3, borderRadius: 7 }}>
                {(['file','url'] as const).map(m => (
                  <button key={m} onClick={() => { setVideoInputMode(m); setVideoStatus('idle') }}
                    style={{
                      flex: 1, padding: '5px 8px', border: 'none', borderRadius: 5, cursor: 'pointer',
                      fontSize: 11, fontWeight: 500, fontFamily: 'inherit',
                      background: videoInputMode === m ? 'var(--surface)' : 'transparent',
                      color: videoInputMode === m ? 'var(--text)' : 'var(--text3)',
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
                    borderRadius: 8, padding: '20px 12px',
                    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                    cursor: 'pointer', background: videoFile ? 'var(--accent-t)' : 'transparent',
                  }}>
                    <Video size={24} style={{ color: videoFile ? 'var(--accent)' : 'var(--text3)' }} />
                    {!videoFile
                      ? <span style={{ fontSize: 12, color: 'var(--text3)' }}>Click to upload video</span>
                      : <span style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 500 }}>{videoFile.name}</span>}
                    <span style={{ fontSize: 11, color: 'var(--text3)' }}>MP4, AVI, MOV</span>
                  </div>
                  <input ref={videoFileRef} type="file" accept="video/*" style={{ display: 'none' }}
                    onChange={e => { setVideoFile(e.target.files?.[0] ?? null); setVideoStatus('idle') }} />
                </>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <input
                    type="url" placeholder="https://... video URL"
                    value={videoUrl}
                    onChange={e => { setVideoUrl(e.target.value); setVideoStatus('idle') }}
                    style={{
                      width: '100%', padding: '9px 12px', boxSizing: 'border-box',
                      background: 'var(--surface2)', border: '1px solid var(--border)',
                      borderRadius: 6, color: 'var(--text)', fontSize: 12, fontFamily: 'inherit',
                    }}
                  />
                  <p style={{ fontSize: 11, color: 'var(--text3)' }}>
                    Bilibili, Vimeo, Twitter, and 1000+ sites supported. Capped at 720p.
                  </p>
                </div>
              )}

              {/* Progress bar */}
              {(videoStatus === 'uploading' || videoStatus === 'running') && (
                <div>
                  <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 6 }}>
                    {videoStatus === 'uploading' ? 'Downloading video…' : 'Processing frames…'}
                  </p>
                  <ProgressBar
                    value={videoProgress.processed} max={videoProgress.total || 1}
                    label={videoProgress.total > 0
                      ? `${videoProgress.processed} / ${videoProgress.total} frames`
                      : 'Please wait…'} />
                </div>
              )}

              <div style={{ display: 'flex', gap: 8 }}>
                <Btn variant="primary"
                  disabled={
                    videoStatus === 'uploading' || videoStatus === 'running' ||
                    (videoInputMode === 'file' ? !videoFile : !videoUrl.trim())
                  }
                  onClick={startVideoInfer}
                  style={{ flex: 1, justifyContent: 'center' }}>
                  {videoStatus === 'uploading' || videoStatus === 'running'
                    ? <><Loader size={12} className="animate-spin" /> Processing…</>
                    : <><Layers size={12} /> Run Segmentation</>}
                </Btn>
                {videoStatus === 'done' && videoJobId && (
                  <Btn variant="secondary" href={`/api/segment/video-infer/${videoJobId}/download`}>
                    <Download size={12} /> Download
                  </Btn>
                )}
              </div>

              {videoStatus === 'done' && (
                <p style={{ fontSize: 12, color: 'var(--success)' }}>
                  ✓ Done — download your segmented video above
                </p>
              )}
              {videoStatus === 'failed' && (
                <p style={{ fontSize: 12, color: 'var(--red, #f87171)' }}>
                  ✗ Processing failed
                </p>
              )}
            </Card>
          )}

          {/* WEBCAM MODE */}
          {mode === 'webcam' && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                <Btn
                  variant={camActive ? 'ghost' : 'primary'}
                  onClick={camActive ? stopCam : startCam}
                  style={camActive ? { color: 'var(--red, #f87171)' } : {}}>
                  {camActive
                    ? <><CameraOff size={12} /> Stop</>
                    : <><Camera size={12} /> Start Camera</>}
                </Btn>
              </div>

              {camError && (
                <p style={{ fontSize: 12, color: 'var(--red, #f87171)', marginBottom: 8 }}>{camError}</p>
              )}

              <div style={{
                position: 'relative', lineHeight: 0,
                background: '#000', borderRadius: 6, overflow: 'hidden',
                minHeight: camActive ? undefined : 240,
              }}>
                <video ref={camVideoRef} muted playsInline
                  onLoadedMetadata={updateCamSize}
                  style={{ width: '100%', display: 'block', maxHeight: 520, objectFit: 'contain' }} />
                <canvas ref={camCanvasRef} style={{ display: 'none' }} />

                {camActive && camDets.length > 0 && (
                  <>
                    <MaskOverlay detections={camDets} width={camRendered.w} height={camRendered.h} />
                    {camDets.map((d, i) => (
                      <div key={i} style={{
                        position: 'absolute',
                        left: `${d.x * 100}%`, top: `${d.y * 100}%`,
                        transform: 'translateY(-100%)',
                        background: getClsColor(d.class_id),
                        color: '#fff', fontSize: 9, fontWeight: 600,
                        padding: '1px 4px', borderRadius: 2, whiteSpace: 'nowrap',
                        pointerEvents: 'none',
                      }}>
                        {d.class_name}
                      </div>
                    ))}
                  </>
                )}

                {!camActive && (
                  <div style={{
                    position: 'absolute', inset: 0,
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                  }}>
                    <div style={{ textAlign: 'center', color: 'var(--text3)' }}>
                      <Layers size={40} style={{ opacity: 0.25, marginBottom: 10 }} />
                      <p style={{ fontSize: 13 }}>Start camera for live segmentation</p>
                    </div>
                  </div>
                )}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
