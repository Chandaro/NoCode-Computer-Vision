import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Upload, Download, Loader, Camera, CameraOff, Video, Image as ImageIcon, Search, X, Sparkles } from 'lucide-react'
import api, { type Project, type FindDetection, type FindStatus } from '../api'
import { PageHeader, Card, Field, Select, Slider, Btn, ProgressBar, TextInput, InfoBanner } from '../components/ui'

// Open-vocabulary detection: the user types what to look for, YOLOE-26 finds it.
// No training, no fixed class list — the prompt list *is* the class list.

const COLORS = [
  '#8b5cf6','#22c55e','#f59e0b','#38bdf8','#f87171',
  '#ec4899','#a3e635','#fb923c','#34d399','#c084fc',
]
const colorFor = (i: number) => COLORS[i % COLORS.length]

// Starter prompt sets so first-time users see something work immediately.
const EXAMPLES: { label: string; prompts: string[] }[] = [
  { label: 'Street',  prompts: ['person', 'car', 'bus', 'bicycle', 'traffic light'] },
  { label: 'Safety',  prompts: ['person', 'helmet', 'safety vest'] },
  { label: 'Kitchen', prompts: ['cup', 'bottle', 'knife', 'bowl'] },
  { label: 'Animals', prompts: ['dog', 'cat', 'bird', 'cow'] },
]

const MODELS = [
  { value: 'yoloe-26n-seg.pt', label: 'Nano · fastest',        hint: 'best for webcam' },
  { value: 'yoloe-26s-seg.pt', label: 'Small · balanced',      hint: 'recommended' },
  { value: 'yoloe-26m-seg.pt', label: 'Medium · more accurate', hint: '' },
  { value: 'yoloe-26l-seg.pt', label: 'Large · most accurate', hint: 'slow on CPU' },
]

type Mode = 'image' | 'video' | 'webcam'
const INFER_MS = 150

function errMsg(e: unknown): string {
  const ax = e as { response?: { data?: { detail?: string } }; message?: string }
  return ax?.response?.data?.detail ?? ax?.message ?? 'unknown error'
}

function Overlay({ detections, prompts, opacity }: {
  detections: FindDetection[]; prompts: string[]; opacity: number
}) {
  const idx = (name: string) => Math.max(0, prompts.indexOf(name))
  return (
    <>
      <svg viewBox="0 0 1 1" preserveAspectRatio="none"
        style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
        {detections.map((d, i) => {
          const c = colorFor(idx(d.class_name))
          return d.mask.length >= 3
            ? <polygon key={i} points={d.mask.map(([x, y]) => `${x},${y}`).join(' ')}
                fill={c} fillOpacity={opacity} stroke={c} strokeWidth={1.5}
                vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
            : <rect key={i} x={d.x} y={d.y} width={d.w} height={d.h} fill="none"
                stroke={c} strokeWidth={2} vectorEffect="non-scaling-stroke" />
        })}
      </svg>
      {detections.map((d, i) => (
        <div key={i} style={{
          position: 'absolute', left: `${d.x * 100}%`, top: `${d.y * 100}%`,
          transform: 'translateY(-100%)', background: colorFor(idx(d.class_name)),
          color: '#fff', fontSize: 10, fontWeight: 600, padding: '1px 5px', borderRadius: 3,
          whiteSpace: 'nowrap', boxShadow: '0 1px 3px rgba(0,0,0,0.4)', pointerEvents: 'none',
        }}>
          {d.class_name} {Math.round(d.conf * 100)}%
        </div>
      ))}
    </>
  )
}

function Tabs<T extends string>({ value, onChange, items }: {
  value: T; onChange: (v: T) => void; items: { key: T; label: string; icon?: React.ReactNode }[]
}) {
  return (
    <div style={{ display: 'flex', gap: 4, background: 'var(--surface2)', padding: 3, borderRadius: 7 }}>
      {items.map(it => (
        <button key={it.key} onClick={() => onChange(it.key)} style={{
          flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5,
          padding: '5px 8px', border: 'none', borderRadius: 5, cursor: 'pointer',
          fontSize: 11, fontWeight: 500, fontFamily: 'inherit',
          background: value === it.key ? 'var(--surface)' : 'transparent',
          color: value === it.key ? 'var(--text)' : 'var(--text3)',
          boxShadow: value === it.key ? 'var(--shadow-xs)' : 'none', transition: 'all 0.15s',
        }}>
          {it.icon}{it.label}
        </button>
      ))}
    </div>
  )
}

export default function FindAnything() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project, setProject] = useState<Project | null>(null)
  const [status,  setStatus]  = useState<FindStatus | null>(null)
  const [mode,    setMode]    = useState<Mode>('image')

  // ── What to find ────────────────────────────────────────────────────────────
  const [prompts, setPrompts] = useState<string[]>(EXAMPLES[0].prompts)
  const [draft,   setDraft]   = useState('')
  const [model,   setModel]   = useState('yoloe-26s-seg.pt')
  const [conf,    setConf]    = useState(0.25)
  const [opacity, setOpacity] = useState(0.35)

  const addDraft = () => {
    const next = [...prompts]
    for (const p of draft.split(',').map(s => s.trim()).filter(Boolean)) {
      if (!next.some(n => n.toLowerCase() === p.toLowerCase())) next.push(p)
    }
    setPrompts(next.slice(0, 20)); setDraft('')
  }
  const promptForm = (fd: FormData) => {
    fd.append('prompts', prompts.join(','))
    fd.append('model_name', model)
    fd.append('conf', String(conf))
  }

  // ── Image ───────────────────────────────────────────────────────────────────
  const [imgInput,   setImgInput]   = useState<'file' | 'url'>('file')
  const [imgUrl,     setImgUrl]     = useState('')
  const [imgFile,    setImgFile]    = useState<File | null>(null)
  const [imgSrc,     setImgSrc]     = useState('')
  const [imgDets,    setImgDets]    = useState<FindDetection[]>([])
  const [imgPrompts, setImgPrompts] = useState<string[]>([])  // prompts the shown result used
  const [imgCounts,  setImgCounts]  = useState<Record<string, number>>({})
  const [imgRunning, setImgRunning] = useState(false)
  const [imgError,   setImgError]   = useState('')
  const imgFileRef = useRef<HTMLInputElement>(null)

  // ── Video ───────────────────────────────────────────────────────────────────
  const [videoInput,    setVideoInput]    = useState<'file' | 'url'>('file')
  const [videoFile,     setVideoFile]     = useState<File | null>(null)
  const [videoUrl,      setVideoUrl]      = useState('')
  const [videoStatus,   setVideoStatus]   = useState<'idle' | 'uploading' | 'running' | 'done' | 'failed'>('idle')
  const [videoJobId,    setVideoJobId]    = useState('')
  const [videoError,    setVideoError]    = useState('')
  const [videoProgress, setVideoProgress] = useState({ processed: 0, total: 0 })
  const videoFileRef   = useRef<HTMLInputElement>(null)
  const videoPollerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Webcam ──────────────────────────────────────────────────────────────────
  const [camActive, setCamActive] = useState(false)
  const [camDets,   setCamDets]   = useState<FindDetection[]>([])
  const [camCounts, setCamCounts] = useState<Record<string, number>>({})
  const [camFps,    setCamFps]    = useState(0)
  const [camError,  setCamError]  = useState('')
  const camVideoRef    = useRef<HTMLVideoElement>(null)
  const camCanvasRef   = useRef<HTMLCanvasElement>(null)
  const camStreamRef   = useRef<MediaStream | null>(null)
  const camPendingRef  = useRef(false)
  const camIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const camFpsCountRef = useRef(0)
  const camFpsTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  // The capture loop reads settings from here so edits apply to the next frame
  // without restarting the loop (and without stale closures).
  const liveRef = useRef({ prompts, model, conf })
  useEffect(() => { liveRef.current = { prompts, model, conf } }, [prompts, model, conf])

  useEffect(() => {
    api.get(`/projects/${projectId}`).then(r => setProject(r.data)).catch(() => {})
    api.get('/find/status').then(r => setStatus(r.data)).catch(() => {})
  }, [projectId])

  const stopCam = useCallback(() => {
    if (camIntervalRef.current) clearInterval(camIntervalRef.current)
    if (camFpsTimerRef.current) clearInterval(camFpsTimerRef.current)
    camStreamRef.current?.getTracks().forEach(t => t.stop())
    camStreamRef.current = null
    if (camVideoRef.current) camVideoRef.current.srcObject = null
    setCamActive(false); setCamDets([]); setCamFps(0); camPendingRef.current = false
  }, [])

  useEffect(() => () => {
    stopCam()
    if (videoPollerRef.current) clearInterval(videoPollerRef.current)
  }, [stopCam])

  const firstRun = status !== null && !status.text_encoder_ready
  const refreshStatus = () => api.get('/find/status').then(r => setStatus(r.data)).catch(() => {})

  // ── Image search ────────────────────────────────────────────────────────────
  const searchImage = async (file: File | null = imgFile) => {
    if (!prompts.length) { setImgError('Add at least one thing to find.'); return }
    const fromUrl = imgInput === 'url'
    if (fromUrl ? !imgUrl.trim() : !file) return
    setImgRunning(true); setImgError('')
    try {
      const fd = new FormData()
      promptForm(fd)
      if (fromUrl) fd.append('url', imgUrl.trim())
      else fd.append('file', file!)
      const res = await api.post(fromUrl ? '/find/infer-url' : '/find/infer', fd)
      setImgDets(res.data.detections ?? [])
      setImgCounts(res.data.counts ?? {})
      setImgPrompts(res.data.prompts ?? prompts)
      if (firstRun) refreshStatus()
    } catch (e) {
      setImgDets([]); setImgError(errMsg(e))
    } finally { setImgRunning(false) }
  }

  const pickImage = (f: File) => {
    setImgFile(f); setImgSrc(URL.createObjectURL(f)); setImgDets([]); setImgCounts({})
    searchImage(f)
  }

  // ── Video ───────────────────────────────────────────────────────────────────
  const startVideo = async () => {
    if (!prompts.length) { setVideoError('Add at least one thing to find.'); return }
    if (videoInput === 'file' ? !videoFile : !videoUrl.trim()) return
    if (videoPollerRef.current) clearInterval(videoPollerRef.current)
    setVideoStatus('uploading'); setVideoJobId(''); setVideoError('')
    setVideoProgress({ processed: 0, total: 0 })
    try {
      const fd = new FormData()
      promptForm(fd)
      if (videoInput === 'url') fd.append('url', videoUrl.trim())
      else fd.append('file', videoFile!)
      const res = await api.post(videoInput === 'url' ? '/find/video-infer-url' : '/find/video-infer', fd)
      const jobId: string = res.data.job_id
      setVideoJobId(jobId); setVideoStatus('running')
      videoPollerRef.current = setInterval(async () => {
        try {
          const s = await api.get(`/find/video-infer/${jobId}/status`)
          setVideoProgress({ processed: s.data.processed, total: s.data.total_frames })
          if (s.data.status === 'done') {
            clearInterval(videoPollerRef.current!); setVideoStatus('done')
          } else if (s.data.status === 'failed') {
            clearInterval(videoPollerRef.current!); setVideoStatus('failed')
            setVideoError(s.data.error ?? 'unknown error')
          } else {
            setVideoStatus(s.data.stage === 'downloading' ? 'uploading' : 'running')
          }
        } catch { /* keep polling */ }
      }, 1500)
    } catch (e) {
      setVideoStatus('failed'); setVideoError(errMsg(e))
    }
  }

  // ── Webcam ──────────────────────────────────────────────────────────────────
  const captureFrame = useCallback(() => {
    const video = camVideoRef.current, canvas = camCanvasRef.current
    const live = liveRef.current
    if (camPendingRef.current || !video || !canvas || !video.videoWidth || !live.prompts.length) return
    const scale = Math.min(1, 640 / video.videoWidth)
    canvas.width  = Math.round(video.videoWidth  * scale)
    canvas.height = Math.round(video.videoHeight * scale)
    canvas.getContext('2d')!.drawImage(video, 0, 0, canvas.width, canvas.height)
    camPendingRef.current = true
    canvas.toBlob(async blob => {
      if (!blob) { camPendingRef.current = false; return }
      try {
        const fd = new FormData()
        fd.append('frame', blob, 'frame.jpg')
        fd.append('prompts', live.prompts.join(','))
        fd.append('model_name', live.model)
        fd.append('conf', String(live.conf))
        const res = await api.post('/find/webcam-frame', fd)
        setCamDets(res.data.detections ?? [])
        setCamCounts(res.data.counts ?? {})
        camFpsCountRef.current += 1
      } catch { /* drop frame */ }
      finally { camPendingRef.current = false }
    }, 'image/jpeg', 0.75)
  }, [])

  const startCam = async () => {
    setCamError(''); setCamDets([])
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } } })
      camStreamRef.current = stream
      if (camVideoRef.current) {
        camVideoRef.current.srcObject = stream
        await camVideoRef.current.play()
      }
      setCamActive(true)
      camIntervalRef.current = setInterval(captureFrame, INFER_MS)
      camFpsTimerRef.current = setInterval(() => {
        setCamFps(camFpsCountRef.current); camFpsCountRef.current = 0
      }, 1000)
    } catch (e) { setCamError('Camera access denied: ' + errMsg(e)) }
  }

  // ── Derived ─────────────────────────────────────────────────────────────────
  const shownCounts  = mode === 'webcam' ? camCounts : mode === 'image' ? imgCounts : {}
  const shownPrompts = mode === 'image' && imgPrompts.length ? imgPrompts : prompts
  const hasResult    = Object.keys(shownCounts).length > 0
  const busyVideo    = videoStatus === 'uploading' || videoStatus === 'running'

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader back={() => navigate(`/projects/${projectId}/images`)}
        title="Find Anything" subtitle={project?.name} />

      <InfoBanner>
        <strong>Type what you want to find</strong> — "helmet", "red car", "cracked tile" — and the model
        looks for it. No labelling or training needed. This is <em>open-vocabulary detection</em>: the model
        learned to match pictures with words, so it can find things it was never specifically trained on.
      </InfoBanner>

      {firstRun && (
        <div style={{ padding: '10px 14px', marginBottom: 16, borderRadius: 'var(--radius)',
          background: 'var(--warn-s)', color: 'var(--text2)', fontSize: 12, lineHeight: 1.6 }}>
          <strong style={{ color: 'var(--warn)' }}>First search takes a few minutes.</strong> The app downloads
          the model (~30 MB) and the text model that reads your words (~250 MB) once. After that it works offline.
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 16 }}>

        {/* ── Sidebar ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <Tabs value={mode} onChange={v => { if (v !== 'webcam' && camActive) stopCam(); setMode(v) }} items={[
              { key: 'image',  label: 'Image',  icon: <ImageIcon size={13} /> },
              { key: 'video',  label: 'Video',  icon: <Video size={13} /> },
              { key: 'webcam', label: 'Webcam', icon: <Camera size={13} /> },
            ]} />

            <Field label="Things to find" hint="comma-separated">
              <div style={{ display: 'flex', gap: 6 }}>
                <TextInput value={draft} onChange={setDraft} placeholder="e.g. helmet, red car"
                  onKeyDown={e => { if (e.key === 'Enter') addDraft() }} />
                <Btn size="sm" onClick={addDraft} disabled={!draft.trim()}>Add</Btn>
              </div>
            </Field>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
              {prompts.map((p, i) => (
                <span key={p} style={{
                  display: 'inline-flex', alignItems: 'center', gap: 4, padding: '2px 4px 2px 8px',
                  borderRadius: 999, fontSize: 11, color: '#fff', background: colorFor(i),
                }}>
                  {p}
                  <button aria-label={`Remove ${p}`} onClick={() => setPrompts(prompts.filter(x => x !== p))}
                    style={{ border: 'none', background: 'transparent', color: '#fff', cursor: 'pointer',
                      padding: 0, display: 'flex' }}>
                    <X size={11} />
                  </button>
                </span>
              ))}
              {prompts.length === 0 && (
                <span style={{ fontSize: 11, color: 'var(--text3)' }}>Nothing yet — add a word or pick an example.</span>
              )}
            </div>

            <div>
              <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 6 }}>Examples</p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {EXAMPLES.map(ex => (
                  <Btn key={ex.label} size="sm" variant="ghost" onClick={() => setPrompts(ex.prompts)}
                    style={{ border: '1px solid var(--border)' }}>
                    {ex.label}
                  </Btn>
                ))}
              </div>
            </div>

            <Field label="Model" hint={MODELS.find(m => m.value === model)?.hint}>
              <Select value={model} onChange={setModel}>
                {MODELS.map(m => <option key={m.value} value={m.value}>YOLOE-26 {m.label}</option>)}
              </Select>
            </Field>

            <div>
              <Slider label="Confidence" value={conf} onChange={setConf}
                min={0.05} max={0.9} step={0.05} format={v => `${Math.round(v * 100)}%`} />
              <p style={{ fontSize: 10, color: 'var(--text3)', lineHeight: 1.5, marginTop: 4 }}>
                Lower it if nothing is found; raise it if you get wrong matches.
              </p>
            </div>
            <Slider label="Mask opacity" value={opacity} onChange={setOpacity}
              min={0} max={0.9} step={0.05} format={v => `${Math.round(v * 100)}%`} />
          </Card>

          {/* Per-prompt results — zero counts are shown on purpose: "not found" teaches too */}
          {hasResult && (
            <Card style={{ padding: 14 }}>
              <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase',
                letterSpacing: '0.07em', marginBottom: 8 }}>
                {mode === 'webcam' ? 'Seeing now' : 'Found'}
              </p>
              {shownPrompts.map((p, i) => {
                const n = shownCounts[p] ?? 0
                return (
                  <div key={p} style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 5, opacity: n ? 1 : 0.5 }}>
                    <div style={{ width: 10, height: 10, borderRadius: 2, background: colorFor(i), flexShrink: 0 }} />
                    <span style={{ fontSize: 11, color: 'var(--text2)', flex: 1 }}>{p}</span>
                    <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'monospace' }}>{n}</span>
                  </div>
                )
              })}
              {mode === 'webcam' && camActive && (
                <p style={{ fontSize: 10, color: 'var(--text3)', marginTop: 6, fontFamily: 'monospace' }}>
                  {camFps} FPS
                </p>
              )}
            </Card>
          )}

          <Card style={{ padding: 14 }}>
            <p style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, fontWeight: 600,
              color: 'var(--text2)', marginBottom: 6 }}>
              <Sparkles size={12} /> Tips
            </p>
            <ul style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.6, paddingLeft: 16, margin: 0 }}>
              <li>Use short, everyday nouns: "dog", not "a picture of a dog".</li>
              <li>Add colour or detail to narrow it down: "red car", "person with helmet".</li>
              <li>A trained detector (Train page) is still more accurate for one specific job.
                Use Find Anything to explore, or to check what's in your images before you label them.</li>
            </ul>
          </Card>
        </div>

        {/* ── Main panel ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {mode === 'image' && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginBottom: 12 }}>
                <Tabs value={imgInput} onChange={v => { setImgInput(v); setImgSrc(''); setImgDets([]); setImgCounts({}) }}
                  items={[{ key: 'file', label: 'Upload file' }, { key: 'url', label: 'Paste URL' }]} />
                {imgInput === 'file' ? (
                  <div style={{ display: 'flex', gap: 8 }}>
                    <Btn variant={imgFile ? 'secondary' : 'primary'} onClick={() => imgFileRef.current?.click()} disabled={imgRunning}>
                      <Upload size={12} /> {imgFile ? 'Change image' : 'Upload image'}
                    </Btn>
                    {imgFile && (
                      <Btn variant="primary" onClick={() => searchImage()} disabled={imgRunning || !prompts.length}>
                        {imgRunning ? <><Loader size={12} className="animate-spin" /> Searching…</> : <><Search size={12} /> Search again</>}
                      </Btn>
                    )}
                    <input ref={imgFileRef} type="file" accept="image/*" style={{ display: 'none' }}
                      onChange={e => { const f = e.target.files?.[0]; if (f) pickImage(f); e.target.value = '' }} />
                  </div>
                ) : (
                  <div style={{ display: 'flex', gap: 6 }}>
                    <TextInput type="url" value={imgUrl} onChange={setImgUrl} placeholder="https://example.com/photo.jpg"
                      onKeyDown={e => { if (e.key === 'Enter') { setImgSrc(imgUrl.trim()); searchImage() } }} />
                    <Btn variant="primary" disabled={imgRunning || !imgUrl.trim() || !prompts.length}
                      onClick={() => { setImgSrc(imgUrl.trim()); searchImage() }}>
                      {imgRunning ? <Loader size={12} className="animate-spin" /> : <><Search size={12} /> Find</>}
                    </Btn>
                  </div>
                )}
              </div>

              {imgError && <p style={{ fontSize: 12, color: 'var(--danger)', marginBottom: 10 }}>{imgError}</p>}

              {imgSrc ? (
                <div style={{ position: 'relative', lineHeight: 0 }}>
                  <img src={imgSrc} alt="search result"
                    style={{ width: '100%', borderRadius: 6, display: 'block', border: '1px solid var(--border)' }} />
                  <Overlay detections={imgDets} prompts={imgPrompts} opacity={opacity} />
                  {imgRunning && (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
                      justifyContent: 'center', background: 'rgba(0,0,0,0.35)', borderRadius: 6,
                      color: '#fff', fontSize: 13, gap: 8, lineHeight: 1.4 }}>
                      <Loader size={16} className="animate-spin" />
                      {firstRun ? 'Downloading models (first time only)…' : 'Searching…'}
                    </div>
                  )}
                </div>
              ) : (
                <div onClick={() => imgInput === 'file' && imgFileRef.current?.click()} style={{
                  border: '2px dashed var(--border2)', borderRadius: 8, padding: 48,
                  display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
                  cursor: imgInput === 'file' ? 'pointer' : 'default',
                }}>
                  <Search size={36} style={{ color: 'var(--text3)', opacity: 0.35 }} />
                  <p style={{ fontSize: 13, color: 'var(--text3)' }}>Upload a photo, then see what it finds</p>
                  <p style={{ fontSize: 11, color: 'var(--text3)', opacity: 0.7 }}>
                    Change the words and press "Search again" to compare
                  </p>
                </div>
              )}

              {!imgRunning && imgSrc && !imgError && imgPrompts.length > 0 && imgDets.length === 0 && (
                <p style={{ marginTop: 10, fontSize: 12, color: 'var(--text3)' }}>
                  Nothing found. Try a simpler word or lower the confidence.
                </p>
              )}
            </Card>
          )}

          {mode === 'video' && (
            <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <Tabs value={videoInput} onChange={v => { setVideoInput(v); setVideoStatus('idle') }}
                items={[{ key: 'file', label: 'Upload file' }, { key: 'url', label: 'Paste URL' }]} />
              {videoInput === 'file' ? (
                <>
                  <div onClick={() => videoFileRef.current?.click()} style={{
                    border: `2px dashed ${videoFile ? 'var(--accent)' : 'var(--border2)'}`, borderRadius: 8,
                    padding: '20px 12px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                    cursor: 'pointer', background: videoFile ? 'var(--accent-t)' : 'transparent',
                  }}>
                    <Video size={24} style={{ color: videoFile ? 'var(--accent)' : 'var(--text3)' }} />
                    <span style={{ fontSize: 12, color: videoFile ? 'var(--accent)' : 'var(--text3)' }}>
                      {videoFile ? videoFile.name : 'Click to upload video'}
                    </span>
                    <span style={{ fontSize: 11, color: 'var(--text3)' }}>MP4, AVI, MOV</span>
                  </div>
                  <input ref={videoFileRef} type="file" accept="video/*" style={{ display: 'none' }}
                    onChange={e => { setVideoFile(e.target.files?.[0] ?? null); setVideoStatus('idle') }} />
                </>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <TextInput type="url" value={videoUrl} placeholder="https://... video link"
                    onChange={v => { setVideoUrl(v); setVideoStatus('idle') }} />
                  <p style={{ fontSize: 11, color: 'var(--text3)' }}>YouTube, Vimeo and 1000+ sites. Capped at 720p.</p>
                </div>
              )}

              {busyVideo && (
                <div>
                  <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 6 }}>
                    {videoStatus === 'uploading' ? 'Downloading video…' : 'Finding things frame by frame…'}
                  </p>
                  <ProgressBar value={videoProgress.processed} max={videoProgress.total || 1}
                    label={videoProgress.total > 0 ? `${videoProgress.processed} / ${videoProgress.total} frames` : 'Please wait…'} />
                </div>
              )}

              <div style={{ display: 'flex', gap: 8 }}>
                <Btn variant="primary" onClick={startVideo} style={{ flex: 1, justifyContent: 'center' }}
                  disabled={busyVideo || !prompts.length || (videoInput === 'file' ? !videoFile : !videoUrl.trim())}>
                  {busyVideo ? <><Loader size={12} className="animate-spin" /> Processing…</> : <><Search size={12} /> Find in video</>}
                </Btn>
                {videoStatus === 'done' && videoJobId && (
                  <Btn variant="secondary" href={`/api/find/video-infer/${videoJobId}/download`}>
                    <Download size={12} /> Download
                  </Btn>
                )}
              </div>
              {videoStatus === 'done' && <p style={{ fontSize: 12, color: 'var(--success)' }}>✓ Done. Download the marked-up video above.</p>}
              {videoStatus === 'failed' && <p style={{ fontSize: 12, color: 'var(--danger)' }}>✗ {videoError || 'Processing failed'}</p>}
            </Card>
          )}

          {mode === 'webcam' && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center' }}>
                <Btn variant={camActive ? 'danger' : 'primary'} onClick={camActive ? stopCam : startCam}>
                  {camActive ? <><CameraOff size={12} /> Stop</> : <><Camera size={12} /> Start camera</>}
                </Btn>
                <span style={{ fontSize: 11, color: 'var(--text3)' }}>
                  Edit the words while it runs — it updates instantly.
                </span>
              </div>
              {camError && <p style={{ fontSize: 12, color: 'var(--danger)', marginBottom: 8 }}>{camError}</p>}
              <div style={{ position: 'relative', lineHeight: 0, background: '#000', borderRadius: 6,
                overflow: 'hidden', minHeight: camActive ? undefined : 240 }}>
                <video ref={camVideoRef} muted playsInline style={{ width: '100%', display: 'block' }} />
                <canvas ref={camCanvasRef} style={{ display: 'none' }} />
                {camActive && <Overlay detections={camDets} prompts={prompts} opacity={opacity} />}
                {!camActive && (
                  <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
                    justifyContent: 'center', color: 'var(--text3)', fontSize: 13, lineHeight: 1.4 }}>
                    Start the camera, then type what to look for
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
