import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Upload, Download, Loader, Camera, CameraOff, Video,
  Image as ImageIcon, Layers, Info, ChevronDown, ChevronUp, AlertTriangle,
  Box, CheckCircle, XCircle, X,
} from 'lucide-react'
import api, { type Project } from '../api'
import { PageHeader, Card, Field, Select, Slider, Btn, Badge, ProgressBar } from '../components/ui'
import PointCloudViewer from '../components/PointCloudViewer'

type ReconJobStatus = 'idle' | 'uploading' | 'queued' | 'running' | 'done' | 'failed'

// ─── Types ────────────────────────────────────────────────────────────────────
interface DepthStats {
  min_raw: number; max_raw: number; mean_raw: number
  is_metric: boolean; unit: string; uniform_warning: boolean
}
interface DepthResult {
  depth_colorized_b64: string
  depth_gray_b64: string
  sidebyside_b64: string
  depth_stats: DepthStats
  model_used: string; colormap: string; image_w: number; image_h: number
  result_id: string
}

type Mode        = 'image' | 'video' | 'webcam'
type DisplayMode = 'depth' | 'sidebyside' | 'overlay'
type VideoStatus = 'idle' | 'uploading' | 'running' | 'done' | 'failed'

const INFER_MS = 1200   // webcam inference interval — depth is slower than pose

// Colormap gradient stops (CSS approximation for the legend bar)
const CMAP_GRADIENTS: Record<string, string> = {
  inferno:  'linear-gradient(to right, #000004, #3b0f70, #8c2981, #dd4968, #fd9347, #fcffa4)',
  magma:    'linear-gradient(to right, #000004, #3b0f70, #8c2981, #c0576a, #f89441, #fcfdbf)',
  plasma:   'linear-gradient(to right, #0d0887, #6a00a8, #b12a90, #e16462, #fca636, #f0f921)',
  viridis:  'linear-gradient(to right, #440154, #30678d, #35b779, #fde725)',
  turbo:    'linear-gradient(to right, #30123b, #4040ff, #00e4ff, #80ff00, #ff8000, #7a0403)',
  jet:      'linear-gradient(to right, #000080, #0000ff, #00ffff, #00ff00, #ffff00, #ff0000)',
  hot:      'linear-gradient(to right, #000000, #ff0000, #ffff00, #ffffff)',
}

const MODEL_IS_METRIC = (id: string) =>
  id.includes('Metric-Indoor') || id.includes('Metric-Outdoor')

export default function Depth() {
  const { id }     = useParams<{ id: string }>()
  const projectId  = Number(id)
  const navigate   = useNavigate()

  const [project,   setProject]   = useState<Project | null>(null)
  const [pageMode,  setPageMode]  = useState<'depth' | 'reconstruct'>('depth')
  const [mode,      setMode]      = useState<Mode>('image')

  // ── 3D Reconstruction state ──────────────────────────────────────────────────
  const [reconFiles,    setReconFiles]    = useState<File[]>([])
  const [reconPreviews, setReconPreviews] = useState<string[]>([])
  const [reconJobId,    setReconJobId]    = useState('')
  const [reconStatus,   setReconStatus]   = useState<ReconJobStatus>('idle')
  const [reconMsg,      setReconMsg]      = useState('')
  const [reconError,    setReconError]    = useState('')
  const [reconShow3D,   setReconShow3D]   = useState(false)
  const [reconNiter,    setReconNiter]    = useState(200)
  const [reconModelStatus, setReconModelStatus] =
    useState<{ installed: boolean; model_cached: boolean; loaded: boolean } | null>(null)
  const reconFileRef   = useRef<HTMLInputElement>(null)
  const reconPollerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Config ──────────────────────────────────────────────────────────────────
  const [modelId,      setModelId]      = useState('depth-anything/Depth-Anything-V2-Small-hf')
  const [colormap,     setColormap]     = useState('inferno')
  const [displayMode,  setDisplayMode]  = useState<DisplayMode>('sidebyside')
  const [opacity,      setOpacity]      = useState(0.7)
  const [showInfo,     setShowInfo]     = useState(false)

  // ── Image mode ──────────────────────────────────────────────────────────────
  const [imgInputMode, setImgInputMode] = useState<'file' | 'url'>('file')
  const [imgUrl,       setImgUrl]       = useState('')
  const [origSrc,      setOrigSrc]      = useState('')   // blob URL of original
  const [depthResult,  setDepthResult]  = useState<DepthResult | null>(null)
  const [imgRunning,   setImgRunning]   = useState(false)
  const [imgError,       setImgError]       = useState('')
  const [showPointCloud, setShowPointCloud] = useState(false)
  const [pointSize,      setPointSize]      = useState(2)
  const imgFileRef = useRef<HTMLInputElement>(null)

  // ── Video mode ──────────────────────────────────────────────────────────────
  const [videoFile,      setVideoFile]      = useState<File | null>(null)
  const [videoInputMode, setVideoInputMode] = useState<'file' | 'url'>('file')
  const [videoUrl,       setVideoUrl]       = useState('')
  const [videoStatus,    setVideoStatus]    = useState<VideoStatus>('idle')
  const [videoJobId,     setVideoJobId]     = useState('')
  const [videoProgress,  setVideoProgress]  = useState({ processed: 0, total: 0, stage: '' })
  const [videoError,     setVideoError]     = useState('')
  const videoFileRef   = useRef<HTMLInputElement>(null)
  const videoPollerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Webcam mode ─────────────────────────────────────────────────────────────
  const [camActive,   setCamActive]   = useState(false)
  const [camDepthSrc, setCamDepthSrc] = useState('')   // base64 depth frame
  const [camFps,      setCamFps]      = useState(0)
  const [camError,    setCamError]    = useState('')
  const [camUniform,  setCamUniform]  = useState(false)
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

  // Check DUSt3R model readiness whenever the user opens 3D Reconstruction
  useEffect(() => {
    if (pageMode !== 'reconstruct') return
    api.get('/reconstruct3d/model-status')
      .then(r => setReconModelStatus(r.data))
      .catch(() => setReconModelStatus({ installed: false, model_cached: false, loaded: false }))
  }, [pageMode])

  useEffect(() => () => {
    stopCam()
    if (videoPollerRef.current) clearInterval(videoPollerRef.current)
  }, [])

  // ── Image inference ──────────────────────────────────────────────────────────
  const runImageInfer = async (file: File) => {
    const blobUrl = URL.createObjectURL(file)
    setOrigSrc(blobUrl); setDepthResult(null); setImgError(''); setShowPointCloud(false)
    setImgRunning(true)
    try {
      const fd = new FormData()
      fd.append('file', file)
      fd.append('model_id', modelId)
      fd.append('colormap', colormap)
      const res = await api.post('/depth/infer', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setDepthResult(res.data)
    } catch (e: any) {
      setImgError(e?.response?.data?.detail ?? e?.message ?? 'Inference failed')
    } finally { setImgRunning(false) }
  }

  const runImageInferUrl = async () => {
    if (!imgUrl.trim()) return
    setOrigSrc(imgUrl.trim()); setDepthResult(null); setImgError(''); setShowPointCloud(false)
    setImgRunning(true)
    try {
      const fd = new FormData()
      fd.append('url', imgUrl.trim())
      fd.append('model_id', modelId)
      fd.append('colormap', colormap)
      const res = await api.post('/depth/infer-url', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      setDepthResult(res.data)
    } catch (e: any) {
      setImgError(e?.response?.data?.detail ?? e?.message ?? 'Inference failed')
    } finally { setImgRunning(false) }
  }

  // ── Video inference ──────────────────────────────────────────────────────────
  const startVideoInfer = async () => {
    if (videoInputMode === 'file' && !videoFile) return
    if (videoInputMode === 'url'  && !videoUrl.trim()) return
    if (videoPollerRef.current) clearInterval(videoPollerRef.current)
    setVideoStatus('uploading'); setVideoJobId(''); setVideoError('')
    setVideoProgress({ processed: 0, total: 0, stage: '' })
    try {
      const fd = new FormData()
      fd.append('model_id', modelId)
      fd.append('colormap', colormap)
      let endpoint: string
      if (videoInputMode === 'url') {
        fd.append('url', videoUrl.trim())
        endpoint = '/depth/video-infer-url'
      } else {
        fd.append('file', videoFile!)
        endpoint = '/depth/video-infer'
      }
      const res = await api.post(endpoint, fd, { headers: { 'Content-Type': 'multipart/form-data' } })
      const jobId: string = res.data.job_id
      setVideoJobId(jobId); setVideoStatus('running')
      videoPollerRef.current = setInterval(async () => {
        try {
          const s = await api.get(`/depth/video-infer/${jobId}/status`)
          const stage = s.data.stage ?? ''
          setVideoProgress({ processed: s.data.processed, total: s.data.total_frames * 2, stage })
          if (s.data.status === 'done') {
            clearInterval(videoPollerRef.current!); setVideoStatus('done')
          } else if (s.data.status === 'failed') {
            clearInterval(videoPollerRef.current!); setVideoStatus('failed')
            setVideoError(s.data.error ?? 'Unknown error')
          }
        } catch { /* keep polling */ }
      }, 1500)
    } catch (e: any) {
      setVideoStatus('failed')
      setVideoError(e?.response?.data?.detail ?? e?.message ?? 'Upload failed')
    }
  }

  // ── Webcam ───────────────────────────────────────────────────────────────────
  const captureAndDepth = useCallback(async () => {
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
        fd.append('model_id', modelId)
        fd.append('colormap', colormap)
        const res = await api.post('/depth/webcam-frame', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
        setCamDepthSrc('data:image/png;base64,' + res.data.depth_b64)
        setCamUniform(res.data.uniform_warning ?? false)
        camFpsCountRef.current += 1
      } catch { /* drop frame */ }
      finally { camPendingRef.current = false }
    }, 'image/jpeg', 0.75)
  }, [modelId, colormap])

  const startCamLoop = useCallback(() => {
    if (camIntervalRef.current) clearInterval(camIntervalRef.current)
    camIntervalRef.current = setInterval(captureAndDepth, INFER_MS)
  }, [captureAndDepth])

  useEffect(() => { if (camActive) startCamLoop() }, [modelId, colormap, camActive, startCamLoop])

  const startCam = async () => {
    setCamError(''); setCamDepthSrc('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
      })
      camStreamRef.current = stream
      if (camVideoRef.current) {
        camVideoRef.current.srcObject = stream
        await camVideoRef.current.play()
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
    setCamActive(false); setCamDepthSrc(''); setCamFps(0); camPendingRef.current = false
  }

  // ── Reconstruction helpers ───────────────────────────────────────────────────
  const addReconFiles = useCallback((incoming: FileList | File[]) => {
    const imgs = Array.from(incoming).filter(f => f.type.startsWith('image/'))
    if (!imgs.length) return
    setReconFiles(prev => {
      const combined = [...prev, ...imgs].slice(0, 12)
      reconPreviews.forEach(URL.revokeObjectURL)
      setReconPreviews(combined.map(f => URL.createObjectURL(f)))
      return combined
    })
  }, [reconPreviews])

  const removeReconFile = (idx: number) => {
    setReconFiles(prev => {
      const next = prev.filter((_, i) => i !== idx)
      reconPreviews.forEach(URL.revokeObjectURL)
      setReconPreviews(next.map(f => URL.createObjectURL(f)))
      return next
    })
  }

  const clearRecon = () => {
    reconPreviews.forEach(URL.revokeObjectURL)
    setReconFiles([]); setReconPreviews([])
    setReconJobId(''); setReconStatus('idle'); setReconMsg(''); setReconError(''); setReconShow3D(false)
    if (reconPollerRef.current) clearInterval(reconPollerRef.current)
  }

  const startReconstruction = async () => {
    if (reconFiles.length < 2) return
    if (reconPollerRef.current) clearInterval(reconPollerRef.current)
    setReconStatus('uploading'); setReconJobId(''); setReconMsg('Uploading images…')
    setReconError(''); setReconShow3D(false)
    try {
      const fd = new FormData()
      reconFiles.forEach(f => fd.append('files', f))
      fd.append('niter', String(reconNiter))
      const res = await api.post('/reconstruct3d/start', fd,
        { headers: { 'Content-Type': 'multipart/form-data' } })
      const jid: string = res.data.job_id
      setReconJobId(jid); setReconStatus('queued'); setReconMsg('Queued…')
      reconPollerRef.current = setInterval(async () => {
        try {
          const s = await api.get(`/reconstruct3d/${jid}/status`)
          setReconMsg(s.data.msg)
          if      (s.data.status === 'done')   { clearInterval(reconPollerRef.current!); setReconStatus('done') }
          else if (s.data.status === 'failed') { clearInterval(reconPollerRef.current!); setReconStatus('failed'); setReconError(s.data.error ?? 'Unknown error') }
          else setReconStatus('running')
        } catch { /* keep polling */ }
      }, 2000)
    } catch (e: any) {
      setReconStatus('failed')
      setReconError(e?.response?.data?.detail ?? e?.message ?? 'Upload failed')
    }
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────
  const downloadB64 = (b64: string, filename: string) => {
    const a = document.createElement('a')
    a.href = 'data:image/png;base64,' + b64
    a.download = filename; a.click()
  }

  const MODES = [
    { key: 'image'  as Mode, label: 'Image',  icon: <ImageIcon size={13} /> },
    { key: 'video'  as Mode, label: 'Video',  icon: <Video size={13} /> },
    { key: 'webcam' as Mode, label: 'Webcam', icon: <Camera size={13} /> },
  ]

  const DISPLAY_MODES = [
    { key: 'depth'      as DisplayMode, label: 'Depth Only' },
    { key: 'sidebyside' as DisplayMode, label: 'Side by Side' },
    { key: 'overlay'    as DisplayMode, label: 'Overlay' },
  ]

  const isMetric = MODEL_IS_METRIC(modelId)
  const stats    = depthResult?.depth_stats

  // ── Result image src based on display mode ───────────────────────────────────
  const resultImageSrc = (() => {
    if (!depthResult) return ''
    if (displayMode === 'depth')      return 'data:image/png;base64,' + depthResult.depth_colorized_b64
    if (displayMode === 'sidebyside') return 'data:image/png;base64,' + depthResult.sidebyside_b64
    return ''  // overlay handled separately in JSX
  })()

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader
        back={() => navigate(`/projects/${projectId}/images`)}
        title={pageMode === 'reconstruct' ? 'Multi-View 3D Reconstruction' : 'Depth Estimation'}
        subtitle={project?.name}
      />

      {/* ── Page mode toggle ── */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 16, borderRadius: 8, overflow: 'hidden',
        border: '1px solid var(--border)', width: 'fit-content' }}>
        {([
          { key: 'depth'       as const, label: 'Depth Map',         icon: <Layers size={13} /> },
          { key: 'reconstruct' as const, label: '3D Reconstruction', icon: <Box    size={13} /> },
        ]).map(({ key, icon, label }) => (
          <button key={key} onClick={() => setPageMode(key)}
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '8px 18px', border: 'none', cursor: 'pointer',
              fontSize: 12, fontWeight: pageMode === key ? 600 : 400,
              background: pageMode === key ? 'var(--accent)' : 'var(--surface)',
              color: pageMode === key ? '#fff' : 'var(--text2)',
              transition: 'all 0.15s',
            }}>
            {icon}{label}
          </button>
        ))}
      </div>

      {/* ── 3D Reconstruction panel ── */}
      {pageMode === 'reconstruct' && (
        <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 16 }}>

          {/* Sidebar */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <Card style={{ padding: 16 }}>
              <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
                textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 10 }}>
                How to get best results
              </p>
              {[
                ['📸', '4–8 photos', 'Walk around every 30–45°'],
                ['🔄', 'Full coverage', 'Front, sides, back, top'],
                ['💡', 'Even lighting', 'No harsh shadows or reflections'],
                ['🎯', '~60% overlap', 'Each photo shares area with the next'],
                ['📐', 'Hold steady', 'Motion blur hurts feature matching'],
              ].map(([icon, title, desc]) => (
                <div key={String(title)} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                  <span style={{ fontSize: 14, flexShrink: 0 }}>{icon}</span>
                  <div>
                    <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', lineHeight: 1.3 }}>{title}</p>
                    <p style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.4 }}>{desc}</p>
                  </div>
                </div>
              ))}
            </Card>
            <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 10 }}>
              <Slider label="Alignment quality" value={reconNiter} onChange={setReconNiter}
                min={50} max={500} step={50}
                format={v => v <= 100 ? `${v} — fast` : v >= 400 ? `${v} — best` : String(v)} />
              <p style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.6 }}>
                Higher = better alignment, longer processing. 200 recommended.
              </p>
              {reconShow3D && (
                <Slider label="Point size" value={pointSize} onChange={setPointSize}
                  min={1} max={6} step={0.5} format={v => String(v)} />
              )}
            </Card>
            <Card style={{ padding: 12, background: 'rgba(94,106,210,0.06)',
              border: '1px solid rgba(94,106,210,0.2)' }}>
              <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', marginBottom: 4 }}>First-time setup</p>
              <p style={{ fontSize: 11, color: 'var(--text2)', lineHeight: 1.6 }}>
                Run <code style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>setup_dust3r.bat</code> then
                restart the server. GPU: ~20–30 s · CPU: ~5–15 min
              </p>
            </Card>
          </div>

          {/* Main panel */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

            {/* Model-readiness banner */}
            {reconModelStatus && !reconModelStatus.installed && (
              <Card style={{ padding: 14, background: 'rgba(248,81,73,0.08)',
                border: '1px solid rgba(248,81,73,0.3)' }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                  <XCircle size={16} style={{ color: '#f85149', flexShrink: 0, marginTop: 1 }} />
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 600, color: '#f85149', marginBottom: 3 }}>
                      DUSt3R not installed
                    </p>
                    <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.6 }}>
                      Run <code style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>setup_dust3r.bat</code> in
                      the NoCode CV folder, then restart the server. The setup also downloads
                      the model (~2.3 GB) so reconstruction starts instantly.
                    </p>
                  </div>
                </div>
              </Card>
            )}
            {reconModelStatus && reconModelStatus.installed && !reconModelStatus.model_cached && (
              <Card style={{ padding: 14, background: 'rgba(210,153,34,0.08)',
                border: '1px solid rgba(210,153,34,0.3)' }}>
                <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                  <AlertTriangle size={16} style={{ color: '#d29922', flexShrink: 0, marginTop: 1 }} />
                  <div>
                    <p style={{ fontSize: 13, fontWeight: 600, color: '#d29922', marginBottom: 3 }}>
                      Model not downloaded yet (~2.3 GB)
                    </p>
                    <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.6 }}>
                      The first reconstruction will download the model and take several minutes.
                      To download it ahead of time, run <code style={{ fontFamily: 'monospace',
                      color: 'var(--accent)' }}>setup_dust3r.bat</code> — it fetches the model so
                      future runs start instantly.
                    </p>
                  </div>
                </div>
              </Card>
            )}

            {/* Upload zone */}
            <Card style={{ padding: 16 }}>
              {/* Drop zone */}
              <div
                onDragOver={e => e.preventDefault()}
                onDrop={e => { e.preventDefault(); addReconFiles(e.dataTransfer.files) }}
                onClick={() => reconFiles.length < 12 && reconFileRef.current?.click()}
                style={{
                  border: `2px dashed ${reconFiles.length >= 2 ? 'var(--accent)' : 'var(--border2)'}`,
                  borderRadius: 8, padding: '20px 16px', textAlign: 'center',
                  cursor: reconFiles.length < 12 ? 'pointer' : 'default', marginBottom: 12,
                  background: reconFiles.length >= 2 ? 'rgba(94,106,210,0.04)' : 'transparent',
                }}>
                <Camera size={26} style={{ color: 'var(--text3)', opacity: 0.5, marginBottom: 6 }} />
                <p style={{ fontSize: 13, color: 'var(--text2)', marginBottom: 2 }}>
                  {reconFiles.length === 0 ? 'Drag & drop photos, or click to browse'
                    : reconFiles.length < 2 ? `${reconFiles.length} photo — add at least 1 more`
                    : `${reconFiles.length} / 12 photos${reconFiles.length < 12 ? ' — click to add more' : ' — max'}`}
                </p>
                <p style={{ fontSize: 11, color: 'var(--text3)' }}>JPG, PNG, WebP · 2–12 images</p>
              </div>
              <input ref={reconFileRef} type="file" multiple accept="image/*" style={{ display: 'none' }}
                onChange={e => { if (e.target.files) addReconFiles(e.target.files); e.target.value = '' }} />

              {/* Thumbnails */}
              {reconFiles.length > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
                  gap: 6, marginBottom: 12 }}>
                  {reconFiles.map((f, i) => (
                    <div key={i} style={{ position: 'relative', borderRadius: 6, overflow: 'hidden',
                      border: '1px solid var(--border)', aspectRatio: '1' }}>
                      <img src={reconPreviews[i]} alt={f.name}
                        style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                      <button onClick={e => { e.stopPropagation(); removeReconFile(i) }}
                        style={{ position: 'absolute', top: 3, right: 3, width: 16, height: 16,
                          borderRadius: '50%', background: 'rgba(0,0,0,0.65)', border: 'none',
                          color: '#fff', cursor: 'pointer', display: 'flex',
                          alignItems: 'center', justifyContent: 'center' }}>
                        <X size={9} />
                      </button>
                      <span style={{ position: 'absolute', bottom: 2, left: 4, fontSize: 9,
                        color: 'rgba(255,255,255,0.7)', fontFamily: 'monospace' }}>{i + 1}</span>
                    </div>
                  ))}
                </div>
              )}

              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                <Btn variant="primary" disabled={reconFiles.length < 2 ||
                  reconStatus === 'uploading' || reconStatus === 'queued' || reconStatus === 'running'}
                  onClick={startReconstruction}>
                  {(reconStatus === 'uploading' || reconStatus === 'queued' || reconStatus === 'running')
                    ? <><Loader size={12} className="animate-spin" /> Reconstructing…</>
                    : <><Box size={12} /> Reconstruct 3D</>}
                </Btn>
                {reconFiles.length > 0 && reconStatus === 'idle' && (
                  <Btn variant="ghost" size="sm" onClick={clearRecon}><X size={12} /> Clear</Btn>
                )}
              </div>
            </Card>

            {/* Status */}
            {reconStatus !== 'idle' && (
              <Card style={{ padding: 16 }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <div style={{ flexShrink: 0, marginTop: 2 }}>
                    {reconStatus === 'done'   && <CheckCircle size={18} style={{ color: '#3fb950' }} />}
                    {reconStatus === 'failed' && <XCircle     size={18} style={{ color: '#f85149' }} />}
                    {(reconStatus === 'uploading' || reconStatus === 'queued' || reconStatus === 'running')
                      && <Loader size={18} className="animate-spin" style={{ color: 'var(--accent)' }} />}
                  </div>
                  <div style={{ flex: 1 }}>
                    <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
                      {reconStatus === 'done' ? '3D reconstruction complete'
                        : reconStatus === 'failed' ? 'Reconstruction failed' : 'Processing…'}
                    </p>
                    <p style={{ fontSize: 12, color: 'var(--text2)' }}>{reconMsg}</p>
                    {reconError && (
                      <div style={{ marginTop: 8, padding: '8px 12px', borderRadius: 6, fontSize: 12,
                        background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.3)',
                        color: '#f85149' }}>{reconError}</div>
                    )}
                  </div>
                </div>
              </Card>
            )}

            {/* Viewer */}
            {reconStatus === 'done' && reconJobId && (
              <Card style={{ padding: 16 }}>
                <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap', alignItems: 'center' }}>
                  <Btn variant={reconShow3D ? 'primary' : 'secondary'} size="sm"
                    onClick={() => setReconShow3D(v => !v)}>
                    <Box size={12} />{reconShow3D ? 'Hide 3D View' : 'View 3D Point Cloud'}
                  </Btn>
                  <Btn variant="ghost" size="sm"
                    onClick={() => window.open(`/api/reconstruct3d/${reconJobId}/download`, '_blank')}>
                    <Download size={12} /> Download PLY
                  </Btn>
                </div>
                {reconShow3D && (
                  <PointCloudViewer
                    fetchUrl={`/reconstruct3d/${reconJobId}/result`}
                    pointSize={pointSize}
                  />
                )}
              </Card>
            )}

          </div>
        </div>
      )}

      {/* ── Depth Map panel ── */}
      {pageMode === 'depth' && (
      <div style={{ display: 'grid', gridTemplateColumns: '270px 1fr', gap: 16 }}>

        {/* ═══ SIDEBAR ═══════════════════════════════════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Mode tabs + controls */}
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

            {/* Model */}
            <Field label="Model">
              <Select value={modelId} onChange={setModelId}>
                <optgroup label="Relative Depth (near = bright)">
                  <option value="depth-anything/Depth-Anything-V2-Small-hf">DA-V2 Small — Fast</option>
                  <option value="depth-anything/Depth-Anything-V2-Base-hf">DA-V2 Base — Balanced</option>
                  <option value="Intel/dpt-hybrid-midas">MiDaS DPT-Hybrid — Sharp edges</option>
                </optgroup>
                <optgroup label="Metric Depth (values in metres)">
                  <option value="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf">Metric Indoor (≤10 m)</option>
                  <option value="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf">Metric Outdoor (≤80 m)</option>
                </optgroup>
              </Select>
            </Field>

            {/* Colormap */}
            <Field label="Colormap">
              <Select value={colormap} onChange={setColormap}>
                <option value="inferno">Inferno (default, colorblind-safe)</option>
                <option value="turbo">Turbo (vivid, Google 2019)</option>
                <option value="magma">Magma</option>
                <option value="plasma">Plasma</option>
                <option value="viridis">Viridis</option>
                <option value="jet">Jet (classic, not perceptual)</option>
                <option value="hot">Hot</option>
              </Select>
            </Field>

            {/* Colormap gradient preview */}
            <div>
              <div style={{ height: 10, borderRadius: 4, marginBottom: 4,
                background: CMAP_GRADIENTS[colormap] || CMAP_GRADIENTS.inferno }} />
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text3)' }}>
                <span>Far / Deep</span><span>Near / Close</span>
              </div>
            </div>

            {/* Display mode — image mode only */}
            {mode === 'image' && (
              <Field label="Display">
                <div style={{ display: 'flex', gap: 3, background: 'var(--surface2)', padding: 3, borderRadius: 7 }}>
                  {DISPLAY_MODES.map(d => (
                    <button key={d.key} onClick={() => setDisplayMode(d.key)}
                      style={{
                        flex: 1, padding: '4px 4px', border: 'none', borderRadius: 5, cursor: 'pointer',
                        fontSize: 10, fontWeight: 500, fontFamily: 'inherit', whiteSpace: 'nowrap',
                        background: displayMode === d.key ? 'var(--surface)' : 'transparent',
                        color: displayMode === d.key ? 'var(--text)' : 'var(--text3)',
                        transition: 'all 0.15s',
                      }}>
                      {d.label}
                    </button>
                  ))}
                </div>
              </Field>
            )}

            {/* Opacity — overlay mode only */}
            {mode === 'image' && displayMode === 'overlay' && (
              <Slider label="Overlay Opacity" value={opacity} onChange={setOpacity}
                min={0.1} max={1.0} step={0.05} format={v => `${Math.round(v * 100)}%`} />
            )}
          </Card>

          {/* Info card */}
          <Card style={{ padding: 0, overflow: 'hidden' }}>
            <button onClick={() => setShowInfo(v => !v)}
              style={{
                width: '100%', display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', padding: '10px 14px',
                background: 'transparent', border: 'none', cursor: 'pointer',
                color: 'var(--text2)', fontSize: 12,
              }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Info size={12} style={{ color: 'var(--accent)' }} /> About depth estimation
              </span>
              {showInfo ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            </button>
            {showInfo && (
              <div style={{ padding: '0 14px 14px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                <div style={{
                  padding: '10px 12px', borderRadius: 6, fontSize: 12, lineHeight: 1.7,
                  background: isMetric ? 'rgba(88,166,255,0.08)' : 'rgba(63,185,80,0.08)',
                  border: `1px solid ${isMetric ? 'rgba(88,166,255,0.2)' : 'rgba(63,185,80,0.2)'}`,
                  color: 'var(--text2)',
                }}>
                  {isMetric ? (
                    <>
                      <span style={{ color: '#58a6ff', fontWeight: 600, display: 'block', marginBottom: 4 }}>
                        📏 Metric depth — values in metres
                      </span>
                      Bright pixels are close (small metre value), dark pixels are far (large metre value).
                      The min/max values shown below are estimated real-world distances.
                    </>
                  ) : (
                    <>
                      <span style={{ color: '#3fb950', fontWeight: 600, display: 'block', marginBottom: 4 }}>
                        🌈 Relative depth — no unit
                      </span>
                      Bright pixels are close, dark pixels are far, but values are relative to
                      this specific image — they do not represent real-world distances.
                    </>
                  )}
                </div>
                <p style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.6 }}>
                  Model: <code style={{ color: 'var(--accent)', fontFamily: 'monospace', fontSize: 10 }}>{modelId.split('/').pop()}</code>
                  <br />First run downloads the model (~80–400 MB). Subsequent runs use the local cache.
                </p>
              </div>
            )}
          </Card>

          {/* Stats card — shown after image inference */}
          {stats && (
            <Card style={{ padding: 14 }}>
              <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
                textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 10 }}>
                Depth Statistics
              </p>
              {stats.uniform_warning && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8,
                  padding: '6px 10px', borderRadius: 5,
                  background: 'rgba(210,153,34,0.1)', border: '1px solid rgba(210,153,34,0.3)',
                  fontSize: 11, color: '#d29922' }}>
                  <AlertTriangle size={11} /> Uniform image — depth unreliable
                </div>
              )}
              {[
                ['Closest', `${stats.min_raw.toFixed(stats.is_metric ? 2 : 4)} ${stats.is_metric ? 'm' : ''}`],
                ['Farthest', `${stats.max_raw.toFixed(stats.is_metric ? 2 : 4)} ${stats.is_metric ? 'm' : ''}`],
                ['Mean',    `${stats.mean_raw.toFixed(stats.is_metric ? 2 : 4)} ${stats.is_metric ? 'm' : ''}`],
              ].map(([label, val]) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between',
                  fontSize: 12, marginBottom: 5 }}>
                  <span style={{ color: 'var(--text3)' }}>{label}</span>
                  <span style={{ color: 'var(--text)', fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>
                    {val}
                  </span>
                </div>
              ))}
              <div style={{ marginTop: 8 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between',
                  fontSize: 10, color: 'var(--text3)', marginBottom: 3 }}>
                  <span>{stats.is_metric ? `${stats.min_raw.toFixed(1)} m` : 'Near'}</span>
                  <span>{stats.is_metric ? `${stats.max_raw.toFixed(1)} m` : 'Far'}</span>
                </div>
                <div style={{ height: 8, borderRadius: 4, background: CMAP_GRADIENTS[colormap] || CMAP_GRADIENTS.inferno }} />
              </div>
            </Card>
          )}
        </div>

        {/* ═══ MAIN PANEL ════════════════════════════════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* ── IMAGE MODE ─────────────────────────────────────────────────── */}
          {mode === 'image' && (
            <Card style={{ padding: 16 }}>
              {/* Upload / URL toggle */}
              <div style={{ display: 'flex', gap: 4, background: 'var(--surface2)',
                padding: 3, borderRadius: 7, marginBottom: 12 }}>
                {(['file', 'url'] as const).map(m => (
                  <button key={m} onClick={() => { setImgInputMode(m); setOrigSrc(''); setDepthResult(null) }}
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
                <div style={{ marginBottom: 12 }}>
                  <Btn variant="primary" onClick={() => imgFileRef.current?.click()} disabled={imgRunning}>
                    {imgRunning
                      ? <><Loader size={12} className="animate-spin" /> Estimating depth…</>
                      : <><Upload size={12} /> Upload Image</>}
                  </Btn>
                  <input ref={imgFileRef} type="file" accept="image/*" style={{ display: 'none' }}
                    onChange={e => { const f = e.target.files?.[0]; if (f) runImageInfer(f) }} />
                </div>
              ) : (
                <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
                  <input type="url" placeholder="https://example.com/photo.jpg"
                    value={imgUrl} onChange={e => setImgUrl(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') runImageInferUrl() }}
                    style={{ flex: 1, padding: '8px 10px', background: 'var(--surface2)',
                      border: '1px solid var(--border)', borderRadius: 6,
                      color: 'var(--text)', fontSize: 12, fontFamily: 'inherit', outline: 'none' }} />
                  <Btn variant="primary" onClick={runImageInferUrl} disabled={imgRunning || !imgUrl.trim()}>
                    {imgRunning ? <Loader size={12} className="animate-spin" /> : 'Go'}
                  </Btn>
                </div>
              )}

              {/* Error */}
              {imgError && (
                <div style={{ marginBottom: 12, padding: '8px 12px', borderRadius: 6, fontSize: 12,
                  background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.3)', color: '#f85149' }}>
                  {imgError}
                </div>
              )}

              {/* Result display */}
              {depthResult ? (
                <>
                  {/* Download buttons */}
                  <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
                    <Btn variant="secondary" size="sm"
                      onClick={() => downloadB64(depthResult.depth_colorized_b64, 'depth_colorized.png')}>
                      <Download size={11} /> Colorized
                    </Btn>
                    <Btn variant="secondary" size="sm"
                      onClick={() => downloadB64(depthResult.depth_gray_b64, 'depth_grayscale.png')}>
                      <Download size={11} /> Grayscale
                    </Btn>
                    <Btn variant="ghost" size="sm"
                      onClick={() => downloadB64(depthResult.sidebyside_b64, 'depth_comparison.png')}>
                      <Download size={11} /> Comparison
                    </Btn>
                  </div>

                  {/* Image display */}
                  {displayMode === 'overlay' ? (
                    <div style={{ position: 'relative', lineHeight: 0, borderRadius: 6, overflow: 'hidden' }}>
                      <img src={origSrc} alt="original"
                        style={{ width: '100%', display: 'block', maxHeight: 500, objectFit: 'contain', background: '#000' }} />
                      <img src={'data:image/png;base64,' + depthResult.depth_colorized_b64} alt="depth"
                        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%',
                          objectFit: 'contain', opacity, pointerEvents: 'none' }} />
                    </div>
                  ) : (
                    <img src={resultImageSrc} alt="depth result"
                      style={{ width: '100%', borderRadius: 6, border: '1px solid var(--border)',
                        display: 'block', maxHeight: 500, objectFit: 'contain', background: '#000' }} />
                  )}

                  <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 8 }}>
                    {depthResult.image_w} × {depthResult.image_h} px · {depthResult.colormap} colormap ·{' '}
                    {depthResult.depth_stats.is_metric ? 'metric depth (metres)' : 'relative depth'}
                  </p>

                  {/* ── 3D Point Cloud section ── */}
                  <div style={{ marginTop: 16, borderTop: '1px solid var(--border)', paddingTop: 14 }}>
                    {!depthResult.result_id && (
                      <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 8 }}>
                        ⚠ Restart the backend to enable 3D point cloud (result ID missing from older run).
                      </p>
                    )}
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 10, flexWrap: 'wrap' }}>
                      <Btn variant={showPointCloud ? 'primary' : 'secondary'} size="sm"
                        disabled={!depthResult.result_id}
                        onClick={() => setShowPointCloud(v => !v)}>
                        <Box size={12} />
                        {showPointCloud ? 'Hide 3D View' : 'View 3D Point Cloud'}
                      </Btn>
                      {showPointCloud && (
                        <Btn variant="ghost" size="sm"
                          onClick={() => window.open(`/api/depth/pointcloud/${depthResult.result_id}/download?subsample=4`, '_blank')}>
                          <Download size={12} /> Download PLY
                        </Btn>
                      )}
                      {showPointCloud && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginLeft: 'auto' }}>
                          <span style={{ fontSize: 11, color: 'var(--text3)' }}>Point size</span>
                          <input type="range" min={1} max={6} step={0.5} value={pointSize}
                            onChange={e => setPointSize(Number(e.target.value))}
                            style={{ width: 80, accentColor: 'var(--accent)' }} />
                          <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'monospace' }}>
                            {pointSize}
                          </span>
                        </div>
                      )}
                    </div>

                    {showPointCloud && depthResult.result_id && (
                      <>
                        <PointCloudViewer
                          fetchUrl={'/depth/pointcloud/' + depthResult.result_id + '/data?subsample=4'}
                          pointSize={pointSize}
                        />
                        <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 6 }}>
                          Coloured by original photo RGB · XYZ back-projected from depth map ·
                          PLY opens in MeshLab, Blender, CloudCompare
                        </p>
                      </>
                    )}
                  </div>
                </>
              ) : (
                !imgRunning && (
                  <div style={{ border: '1px dashed var(--border2)', borderRadius: 8, padding: '48px 24px',
                    textAlign: 'center', color: 'var(--text3)' }}>
                    <Layers size={28} style={{ marginBottom: 10, opacity: 0.4 }} />
                    <p style={{ fontSize: 13, marginBottom: 4 }}>Upload an image to estimate depth</p>
                    <p style={{ fontSize: 11 }}>Supports JPG, PNG, WebP · Max 1024 px (auto-resized)</p>
                  </div>
                )
              )}
            </Card>
          )}

          {/* ── VIDEO MODE ─────────────────────────────────────────────────── */}
          {mode === 'video' && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', gap: 4, background: 'var(--surface2)',
                padding: 3, borderRadius: 7, marginBottom: 12 }}>
                {(['file', 'url'] as const).map(m => (
                  <button key={m} onClick={() => { setVideoInputMode(m); setVideoFile(null); setVideoStatus('idle') }}
                    style={{
                      flex: 1, padding: '4px 8px', border: 'none', borderRadius: 5, cursor: 'pointer',
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
                <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                  <Btn variant="secondary" onClick={() => videoFileRef.current?.click()}>
                    <Upload size={12} /> {videoFile ? videoFile.name : 'Select Video'}
                  </Btn>
                  <input ref={videoFileRef} type="file" accept="video/*" style={{ display: 'none' }}
                    onChange={e => { const f = e.target.files?.[0]; if (f) { setVideoFile(f); setVideoStatus('idle') } }} />
                  <Btn variant="primary" disabled={!videoFile || videoStatus === 'running' || videoStatus === 'uploading'}
                    onClick={startVideoInfer}>
                    {videoStatus === 'uploading' || videoStatus === 'running'
                      ? <><Loader size={12} className="animate-spin" /> Processing…</>
                      : <><Layers size={12} /> Estimate Depth</>}
                  </Btn>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 12 }}>
                  <input type="url" placeholder="https://example.com/video.mp4"
                    value={videoUrl} onChange={e => { setVideoUrl(e.target.value); setVideoStatus('idle') }}
                    style={{ padding: '8px 10px', background: 'var(--surface2)',
                      border: '1px solid var(--border)', borderRadius: 6,
                      color: 'var(--text)', fontSize: 12, fontFamily: 'inherit', outline: 'none' }} />
                  <Btn variant="primary"
                    disabled={!videoUrl.trim() || videoStatus === 'running' || videoStatus === 'uploading'}
                    onClick={startVideoInfer}>
                    {videoStatus === 'uploading' || videoStatus === 'running'
                      ? <><Loader size={12} className="animate-spin" /> Processing…</>
                      : <><Layers size={12} /> Estimate Depth</>}
                  </Btn>
                </div>
              )}

              {/* Info about two-pass */}
              <div style={{ padding: '8px 12px', borderRadius: 6, marginBottom: 12, fontSize: 12,
                background: 'rgba(94,106,210,0.08)', border: '1px solid rgba(94,106,210,0.18)',
                color: 'var(--text2)', lineHeight: 1.6 }}>
                <strong style={{ color: 'var(--accent)', display: 'block', marginBottom: 2, fontSize: 11 }}>
                  Two-pass processing
                </strong>
                Pass 1 analyses all frames to find the global depth range. Pass 2 applies consistent
                coloring — preventing the flickering that single-pass depth videos suffer from.
                Processing time is ~2× longer but results look professional.
              </div>

              {/* Progress */}
              {(videoStatus === 'uploading' || videoStatus === 'running') && (
                <div style={{ marginBottom: 12 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4,
                    fontSize: 11, color: 'var(--text3)' }}>
                    <span style={{ textTransform: 'capitalize' }}>
                      {videoStatus === 'uploading' ? 'Uploading…' :
                       videoProgress.stage === 'analyzing' ? 'Pass 1 — analysing frames…' :
                       videoProgress.stage === 'rendering' ? 'Pass 2 — rendering depth…' :
                       videoProgress.stage === 'downloading' ? 'Downloading video…' : 'Processing…'}
                    </span>
                    {videoProgress.total > 0 && (
                      <span>{videoProgress.processed}/{videoProgress.total} frames</span>
                    )}
                  </div>
                  <ProgressBar value={videoProgress.total > 0
                    ? Math.round(videoProgress.processed / videoProgress.total * 100) : 0} max={100} />
                </div>
              )}

              {videoStatus === 'done' && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 12,
                  padding: '10px 14px', borderRadius: 8, marginBottom: 8,
                  background: 'rgba(63,185,80,0.1)', border: '1px solid rgba(63,185,80,0.3)' }}>
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#3fb950' }}>✓ Depth video ready</span>
                  <Btn variant="primary" size="sm"
                    onClick={() => window.open(`/api/depth/video-infer/${videoJobId}/download`, '_blank')}>
                    <Download size={12} /> Download MP4
                  </Btn>
                </div>
              )}

              {videoError && (
                <div style={{ padding: '8px 12px', borderRadius: 6, fontSize: 12,
                  background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.3)', color: '#f85149' }}>
                  {videoError}
                </div>
              )}
            </Card>
          )}

          {/* ── WEBCAM MODE ────────────────────────────────────────────────── */}
          {mode === 'webcam' && (
            <Card style={{ padding: 16 }}>
              {/* Controls */}
              <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
                {!camActive ? (
                  <Btn variant="primary" onClick={startCam}>
                    <Camera size={13} /> Start Camera
                  </Btn>
                ) : (
                  <Btn variant="secondary" onClick={stopCam}>
                    <CameraOff size={13} /> Stop Camera
                  </Btn>
                )}
                {camActive && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Badge color={camFps > 0 ? 'green' : 'gray'}>
                      {camFps} FPS
                    </Badge>
                    {camUniform && (
                      <Badge color="yellow">
                        <AlertTriangle size={10} /> Uniform frame
                      </Badge>
                    )}
                  </div>
                )}
              </div>

              {camError && (
                <div style={{ padding: '8px 12px', borderRadius: 6, marginBottom: 12, fontSize: 12,
                  background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.3)', color: '#f85149' }}>
                  {camError}
                </div>
              )}

              {/* Live feed + depth side by side */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                {/* Original feed */}
                <div>
                  <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
                    textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
                    Live Feed
                  </p>
                  <div style={{ position: 'relative', background: '#000', borderRadius: 6, overflow: 'hidden',
                    border: '1px solid var(--border)', aspectRatio: '4/3' }}>
                    <video ref={camVideoRef} playsInline muted
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                    {!camActive && (
                      <div style={{ position: 'absolute', inset: 0, display: 'flex',
                        alignItems: 'center', justifyContent: 'center' }}>
                        <Camera size={28} style={{ color: 'var(--text3)', opacity: 0.4 }} />
                      </div>
                    )}
                  </div>
                </div>

                {/* Depth output */}
                <div>
                  <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
                    textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>
                    Depth Map
                  </p>
                  <div style={{ background: '#000', borderRadius: 6, overflow: 'hidden',
                    border: '1px solid var(--border)', aspectRatio: '4/3',
                    display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {camDepthSrc ? (
                      <img src={camDepthSrc} alt="depth"
                        style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                    ) : (
                      <div style={{ textAlign: 'center', color: 'var(--text3)' }}>
                        <Layers size={24} style={{ opacity: 0.3, marginBottom: 6 }} />
                        <p style={{ fontSize: 11 }}>{camActive ? 'Processing…' : 'Start camera to begin'}</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Colormap legend */}
              <div style={{ marginTop: 10 }}>
                <div style={{ height: 8, borderRadius: 4, background: CMAP_GRADIENTS[colormap] || CMAP_GRADIENTS.inferno }} />
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text3)', marginTop: 3 }}>
                  <span>Far / Deep</span>
                  <span>Near / Close</span>
                </div>
              </div>

              <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 8 }}>
                Note: Depth inference runs at ~{Math.round(1000 / INFER_MS)} req/s — GPU recommended for smooth output.
                Changing model or colormap applies to the next captured frame.
              </p>
              <canvas ref={camCanvasRef} style={{ display: 'none' }} />
            </Card>
          )}

        </div>
      </div>
      )} {/* end depth panel */}

    </div>
  )
}
