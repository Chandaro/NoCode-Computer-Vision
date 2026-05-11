import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Upload, Download, Loader, Camera, CameraOff, Video, Image as ImageIcon, Activity } from 'lucide-react'
import api, { type Project } from '../api'
import { PageHeader, Card, Field, Select, Slider, Btn, Badge, ProgressBar } from '../components/ui'

// COCO 17-keypoint skeleton connections
const SKELETON = [
  [0,1],[0,2],[1,3],[2,4],
  [5,7],[7,9],[6,8],[8,10],
  [5,6],[5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],
]
// Left / right / center keypoint groups for coloring
const KP_LEFT   = new Set([1,3,5,7,9,11,13,15])
const KP_RIGHT  = new Set([2,4,6,8,10,12,14,16])

function kpColor(idx: number) {
  if (KP_LEFT.has(idx))  return '#38bdf8'  // left = sky blue
  if (KP_RIGHT.has(idx)) return '#fb923c'  // right = orange
  return '#e2e8f0'                          // center = white-ish
}

interface Keypoint  { x: number; y: number }
interface Person {
  x: number; y: number; w: number; h: number; conf: number
  keypoints: number[][]; kp_conf: number[]
}

// ── Skeleton SVG overlay ──────────────────────────────────────────────────────
function SkeletonOverlay({ persons, width, height, skeleton }:
  { persons: Person[]; width: number; height: number; skeleton: number[][] }) {
  return (
    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
      viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      {persons.map((person, pi) => {
        const kps: Keypoint[] = person.keypoints.map(([x, y]) => ({ x: x * width, y: y * height }))
        const confs = person.kp_conf
        return (
          <g key={pi}>
            {/* Skeleton lines */}
            {skeleton.map(([a, b], si) => {
              const ka = kps[a], kb = kps[b]
              if (!ka || !kb) return null
              const ca = confs[a] ?? 1, cb = confs[b] ?? 1
              if (ca < 0.3 || cb < 0.3) return null
              if (ka.x === 0 && ka.y === 0) return null
              if (kb.x === 0 && kb.y === 0) return null
              return (
                <line key={si}
                  x1={ka.x} y1={ka.y} x2={kb.x} y2={kb.y}
                  stroke="rgba(255,255,255,0.7)" strokeWidth={2} strokeLinecap="round" />
              )
            })}
            {/* Keypoint circles */}
            {kps.map((kp, ki) => {
              const c = confs[ki] ?? 1
              if (c < 0.3) return null
              if (kp.x === 0 && kp.y === 0) return null
              return (
                <circle key={ki} cx={kp.x} cy={kp.y} r={4}
                  fill={kpColor(ki)} stroke="rgba(0,0,0,0.5)" strokeWidth={1} />
              )
            })}
          </g>
        )
      })}
    </svg>
  )
}

type Mode = 'image' | 'video' | 'webcam'

const INFER_MS = 150

export default function Pose() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project,    setProject]    = useState<Project | null>(null)
  const [mode,       setMode]       = useState<Mode>('image')
  const [modelName,  setModelName]  = useState('yolo11n-pose.pt')
  const [conf,       setConf]       = useState(0.25)

  // ── Image mode ──────────────────────────────────────────────────────────────
  const [imgInputMode, setImgInputMode] = useState<'file'|'url'>('file')
  const [imgUrl,       setImgUrl]       = useState('')
  const [imgSrc,     setImgSrc]     = useState('')
  const [imgPersons, setImgPersons] = useState<Person[]>([])
  const [imgSkeleton,setImgSkeleton]= useState<number[][]>(SKELETON)
  const [imgRendered,setImgRendered]= useState({ w: 1, h: 1 })
  const [imgRunning, setImgRunning] = useState(false)
  const imgFileRef = useRef<HTMLInputElement>(null)
  const imgRef     = useRef<HTMLImageElement>(null)

  // ── Video mode ──────────────────────────────────────────────────────────────
  const [videoFile,      setVideoFile]      = useState<File | null>(null)
  const [videoInputMode, setVideoInputMode] = useState<'file'|'url'>('file')
  const [videoUrl,       setVideoUrl]       = useState('')
  const [videoStatus,    setVideoStatus]    = useState<'idle'|'uploading'|'running'|'done'|'failed'>('idle')
  const [videoJobId,     setVideoJobId]     = useState('')
  const [videoProgress,  setVideoProgress]  = useState({ processed: 0, total: 0 })
  const videoFileRef   = useRef<HTMLInputElement>(null)
  const videoPollerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Webcam mode ─────────────────────────────────────────────────────────────
  const [camActive,    setCamActive]    = useState(false)
  const [camPersons,   setCamPersons]   = useState<Person[]>([])
  const [camFps,       setCamFps]       = useState(0)
  const [camRendered,  setCamRendered]  = useState({ w: 640, h: 480 })
  const [camError,     setCamError]     = useState('')
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
  useEffect(() => () => { stopCam(); if (videoPollerRef.current) clearInterval(videoPollerRef.current) }, [])

  // ── Image inference ──────────────────────────────────────────────────────────
  const runImageInfer = async (file: File) => {
    const url = URL.createObjectURL(file)
    setImgSrc(url); setImgPersons([])
    setImgRunning(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('model_name', modelName)
      fd.append('conf', String(conf))
      const res = await api.post('/pose/infer', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setImgPersons(res.data.persons ?? [])
      setImgSkeleton(res.data.skeleton ?? SKELETON)
    } catch (e: any) {
      alert('Inference failed: ' + (e?.response?.data?.detail ?? e.message))
    } finally { setImgRunning(false) }
  }

  const runImageInferUrl = async () => {
    if (!imgUrl.trim()) return
    setImgSrc(imgUrl.trim()); setImgPersons([])
    setImgRunning(true)
    try {
      const fd = new FormData()
      fd.append('url', imgUrl.trim())
      fd.append('model_name', modelName)
      fd.append('conf', String(conf))
      const res = await api.post('/pose/infer-url', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setImgPersons(res.data.persons ?? [])
      setImgSkeleton(res.data.skeleton ?? SKELETON)
    } catch (e: any) {
      alert('Inference failed: ' + (e?.response?.data?.detail ?? e.message))
    } finally { setImgRunning(false) }
  }

  // ── Video inference ──────────────────────────────────────────────────────────
  const startVideoInfer = async () => {
    if (videoInputMode === 'file' && !videoFile) return
    if (videoInputMode === 'url' && !videoUrl.trim()) return
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
        endpoint = '/pose/video-infer-url'
      } else {
        fd.append('file', videoFile!)
        endpoint = '/pose/video-infer'
      }
      const res = await api.post(endpoint, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      const jobId: string = res.data.job_id
      setVideoJobId(jobId); setVideoStatus('running')
      videoPollerRef.current = setInterval(async () => {
        try {
          const s = await api.get(`/pose/video-infer/${jobId}/status`)
          setVideoProgress({ processed: s.data.processed, total: s.data.total_frames })
          if (s.data.stage) setVideoStatus(s.data.stage === 'downloading' ? 'uploading' : 'running')
          if (s.data.status === 'done') { clearInterval(videoPollerRef.current!); setVideoStatus('done') }
          else if (s.data.status === 'failed') {
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
  useEffect(() => { window.addEventListener('resize', updateCamSize); return () => window.removeEventListener('resize', updateCamSize) }, [updateCamSize])

  const captureAndPose = useCallback(async () => {
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
        const res = await api.post('/pose/webcam-frame', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
        setCamPersons(res.data.persons ?? [])
        camFpsCountRef.current += 1
        updateCamSize()
      } catch { /* drop frame */ }
      finally { camPendingRef.current = false }
    }, 'image/jpeg', 0.75)
  }, [modelName, conf, updateCamSize])

  const startCamLoop = useCallback(() => {
    if (camIntervalRef.current) clearInterval(camIntervalRef.current)
    camIntervalRef.current = setInterval(captureAndPose, INFER_MS)
  }, [captureAndPose])

  useEffect(() => { if (camActive) startCamLoop() }, [conf, modelName, camActive, startCamLoop])

  const startCam = async () => {
    setCamError(''); setCamPersons([])
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } })
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
    setCamActive(false); setCamPersons([]); setCamFps(0); camPendingRef.current = false
  }

  const MODES: { key: Mode; label: string; icon: React.ReactNode }[] = [
    { key: 'image',  label: 'Image',  icon: <ImageIcon size={13} /> },
    { key: 'video',  label: 'Video',  icon: <Video size={13} /> },
    { key: 'webcam', label: 'Webcam', icon: <Camera size={13} /> },
  ]

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader back={() => navigate(`/projects/${projectId}/images`)}
        title="Pose Estimation" subtitle={project?.name} />

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
                <optgroup label="YOLO11 Pose">
                  <option value="yolo11n-pose.pt">YOLO11 Nano</option>
                  <option value="yolo11s-pose.pt">YOLO11 Small</option>
                  <option value="yolo11m-pose.pt">YOLO11 Medium</option>
                </optgroup>
                <optgroup label="YOLOv8 Pose">
                  <option value="yolov8n-pose.pt">YOLOv8 Nano</option>
                  <option value="yolov8s-pose.pt">YOLOv8 Small</option>
                  <option value="yolov8m-pose.pt">YOLOv8 Medium</option>
                </optgroup>
              </Select>
            </Field>
            <Slider label="Confidence" value={conf} onChange={setConf}
              min={0.1} max={0.9} step={0.05} format={v => `${Math.round(v * 100)}%`} />
          </Card>

          {/* Keypoint legend */}
          <Card style={{ padding: 14 }}>
            <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>Skeleton</p>
            {[['#38bdf8','Left side'],['#fb923c','Right side'],['#e2e8f0','Center']].map(([c,l]) => (
              <div key={l} style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5 }}>
                <div style={{ width: 10, height: 10, borderRadius: '50%', background: c, flexShrink: 0 }} />
                <span style={{ fontSize: 11, color: 'var(--text3)' }}>{l}</span>
              </div>
            ))}
          </Card>
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
                    <button key={m} onClick={() => { setImgInputMode(m); setImgSrc(''); setImgPersons([]) }}
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
                      {imgRunning ? <><Loader size={12} className="animate-spin" /> Processing…</> : <><Upload size={12} /> Upload Image</>}
                    </Btn>
                    <input ref={imgFileRef} type="file" accept="image/*" style={{ display: 'none' }}
                      onChange={e => { const f = e.target.files?.[0]; if (f) runImageInfer(f) }} />
                  </>
                ) : (
                  <div style={{ display: 'flex', gap: 6 }}>
                    <input
                      type="url"
                      placeholder="https://example.com/image.jpg"
                      value={imgUrl}
                      onChange={e => setImgUrl(e.target.value)}
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
                  <img ref={imgRef} src={imgSrc} alt="pose"
                    onLoad={e => { const el = e.currentTarget; setImgRendered({ w: el.clientWidth, h: el.clientHeight }) }}
                    style={{ width: '100%', borderRadius: 6, display: 'block', border: '1px solid var(--border)' }} />
                  {imgPersons.length > 0 && (
                    <SkeletonOverlay persons={imgPersons}
                      width={imgRendered.w} height={imgRendered.h} skeleton={imgSkeleton} />
                  )}
                </div>
              ) : (
                <div style={{ border: '2px dashed var(--border2)', borderRadius: 8, padding: 48,
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
                  cursor: 'pointer' }} onClick={() => imgFileRef.current?.click()}>
                  <Activity size={32} style={{ color: 'var(--text3)', opacity: 0.4 }} />
                  <p style={{ fontSize: 13, color: 'var(--text3)' }}>Upload an image to detect poses</p>
                </div>
              )}

              {imgPersons.length > 0 && (
                <div style={{ marginTop: 10, padding: '8px 12px', background: 'var(--surface2)',
                  borderRadius: 6, fontSize: 12, color: 'var(--text2)' }}>
                  {imgPersons.length} person{imgPersons.length !== 1 ? 's' : ''} detected
                  {imgPersons.length > 0 && ` · avg confidence ${(imgPersons.reduce((s, p) => s + p.conf, 0) / imgPersons.length * 100).toFixed(0)}%`}
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
                    YouTube, Vimeo, Twitter, TikTok, and 1000+ sites supported. Capped at 720p.
                  </p>
                </div>
              )}

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
                  <ProgressBar value={videoProgress.processed} max={videoProgress.total || 1} label="" />
                </div>
              )}

              <div style={{ display: 'flex', gap: 8 }}>
                <Btn variant="primary" style={{ flex: 1, justifyContent: 'center' }}
                  disabled={(videoInputMode === 'file' ? !videoFile : !videoUrl.trim()) || videoStatus === 'running' || videoStatus === 'uploading'}
                  onClick={startVideoInfer}>
                  {videoStatus === 'running' || videoStatus === 'uploading'
                    ? <><Loader size={12} className="animate-spin" /> Processing…</>
                    : <><Activity size={12} /> Run Pose Detection</>}
                </Btn>
                {videoStatus === 'done' && videoJobId && (
                  <Btn variant="secondary"
                    href={`/api/pose/video-infer/${videoJobId}/download`}>
                    <Download size={12} /> Download
                  </Btn>
                )}
              </div>

              {videoStatus === 'done' && (
                <p style={{ fontSize: 11, color: 'var(--success)' }}>
                  Done — {videoProgress.processed} frames processed.
                </p>
              )}
            </Card>
          )}

          {/* WEBCAM MODE */}
          {mode === 'webcam' && (
            <Card style={{ padding: 0, overflow: 'hidden', background: '#000', position: 'relative' }}>
              <div style={{ position: 'absolute', top: 10, right: 10, zIndex: 10, display: 'flex', gap: 6 }}>
                {camActive && <Badge color="green"><Activity size={9} /> Live</Badge>}
                {camActive && camFps > 0 && (
                  <span style={{ fontSize: 10, background: 'rgba(0,0,0,0.6)', color: '#fff',
                    padding: '3px 7px', borderRadius: 4, fontFamily: 'JetBrains Mono, monospace' }}>
                    {camFps} fps
                  </span>
                )}
              </div>

              <div style={{ position: 'relative', lineHeight: 0, minHeight: 360,
                display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {!camActive && (
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center', gap: 12 }}>
                    <Camera size={40} style={{ color: 'var(--text3)', opacity: 0.4 }} />
                    <p style={{ fontSize: 13, color: 'var(--text3)' }}>Press Start Camera</p>
                  </div>
                )}

                <video ref={camVideoRef} autoPlay playsInline muted
                  onLoadedMetadata={updateCamSize}
                  style={{ width: '100%', display: camActive ? 'block' : 'none',
                    maxHeight: '65vh', objectFit: 'contain' }} />

                {camActive && camPersons.length > 0 && (
                  <SkeletonOverlay persons={camPersons}
                    width={camRendered.w} height={camRendered.h} skeleton={SKELETON} />
                )}
              </div>
              <canvas ref={camCanvasRef} style={{ display: 'none' }} />

              <div style={{ padding: '10px 16px', borderTop: '1px solid var(--border)', display: 'flex', gap: 8, alignItems: 'center' }}>
                <Btn variant={camActive ? 'ghost' : 'primary'} onClick={camActive ? stopCam : startCam}
                  style={{ ...(camActive ? { color: 'var(--red, #f87171)', borderColor: 'var(--red, #f87171)' } : {}) }}>
                  {camActive ? <><CameraOff size={13} /> Stop</> : <><Camera size={13} /> Start Camera</>}
                </Btn>
                {camActive && (
                  <span style={{ fontSize: 12, color: 'var(--text3)' }}>
                    {camPersons.length} person{camPersons.length !== 1 ? 's' : ''} detected
                  </span>
                )}
                {camError && <span style={{ fontSize: 11, color: 'var(--red, #f87171)' }}>{camError}</span>}
              </div>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
