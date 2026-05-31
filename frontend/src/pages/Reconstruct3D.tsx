import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  Download, Loader, X, Box, Camera,
  CheckCircle, XCircle, Info,
} from 'lucide-react'
import api, { type Project } from '../api'
import { PageHeader, Card, Btn, Slider } from '../components/ui'
import PointCloudViewer from '../components/PointCloudViewer'

type JobStatus = 'idle' | 'uploading' | 'queued' | 'running' | 'done' | 'failed'

const STATUS_STEPS = [
  'Loading DUSt3R model…',
  'Preparing images…',
  'Running pairwise inference…',
  'Computing global alignment…',
  'Extracting point cloud…',
]

export default function Reconstruct3D() {
  const { id }    = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project, setProject] = useState<Project | null>(null)

  // ── Image upload ─────────────────────────────────────────────────────────
  const [files,    setFiles]    = useState<File[]>([])
  const [previews, setPreviews] = useState<string[]>([])
  const fileInputRef = useRef<HTMLInputElement>(null)
  const dropRef      = useRef<HTMLDivElement>(null)

  // ── Job state ────────────────────────────────────────────────────────────
  const [jobId,     setJobId]     = useState('')
  const [jobStatus, setJobStatus] = useState<JobStatus>('idle')
  const [jobMsg,    setJobMsg]    = useState('')
  const [jobError,  setJobError]  = useState('')

  // ── Viewer ───────────────────────────────────────────────────────────────
  const [show3D,    setShow3D]    = useState(false)
  const [pointSize, setPointSize] = useState(2)

  // ── Config ───────────────────────────────────────────────────────────────
  const [niter, setNiter] = useState(200)

  const pollerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    api.get(`/projects/${projectId}`).then(r => setProject(r.data))
  }, [projectId])

  useEffect(() => () => {
    previews.forEach(URL.revokeObjectURL)
    if (pollerRef.current) clearInterval(pollerRef.current)
  }, [])

  // ── File handling ─────────────────────────────────────────────────────────
  const addFiles = useCallback((incoming: FileList | File[]) => {
    const imgs = Array.from(incoming).filter(f => f.type.startsWith('image/'))
    if (!imgs.length) return
    setFiles(prev => {
      const combined = [...prev, ...imgs].slice(0, 12)
      const urls = combined.map(f => URL.createObjectURL(f))
      previews.forEach(URL.revokeObjectURL)
      setPreviews(urls)
      return combined
    })
  }, [previews])

  const removeFile = (idx: number) => {
    setFiles(prev => {
      const next = prev.filter((_, i) => i !== idx)
      const urls = next.map(f => URL.createObjectURL(f))
      previews.forEach(URL.revokeObjectURL)
      setPreviews(urls)
      return next
    })
  }

  const clearAll = () => {
    previews.forEach(URL.revokeObjectURL)
    setFiles([]); setPreviews([])
    setJobId(''); setJobStatus('idle'); setJobMsg(''); setJobError('')
    setShow3D(false)
    if (pollerRef.current) clearInterval(pollerRef.current)
  }

  // Drag-and-drop
  const onDragOver = (e: React.DragEvent) => { e.preventDefault() }
  const onDrop     = (e: React.DragEvent) => {
    e.preventDefault()
    addFiles(e.dataTransfer.files)
  }

  // ── Run reconstruction ───────────────────────────────────────────────────
  const startReconstruction = async () => {
    if (files.length < 2) return
    if (pollerRef.current) clearInterval(pollerRef.current)
    setJobStatus('uploading'); setJobId(''); setJobMsg('Uploading images…')
    setJobError(''); setShow3D(false)

    try {
      const fd = new FormData()
      files.forEach(f => fd.append('files', f))
      fd.append('niter', String(niter))
      const res = await api.post('/reconstruct3d/start', fd,
        { headers: { 'Content-Type': 'multipart/form-data' } })
      const jid: string = res.data.job_id
      setJobId(jid)
      setJobStatus('queued')
      setJobMsg('Queued…')

      // Poll for status
      pollerRef.current = setInterval(async () => {
        try {
          const s = await api.get(`/reconstruct3d/${jid}/status`)
          setJobMsg(s.data.msg)
          if (s.data.status === 'done') {
            clearInterval(pollerRef.current!)
            setJobStatus('done')
          } else if (s.data.status === 'failed') {
            clearInterval(pollerRef.current!)
            setJobStatus('failed')
            setJobError(s.data.error ?? 'Unknown error')
          } else {
            setJobStatus('running')
          }
        } catch { /* keep polling */ }
      }, 2000)
    } catch (e: any) {
      setJobStatus('failed')
      setJobError(e?.response?.data?.detail ?? e?.message ?? 'Upload failed')
    }
  }

  const isProcessing = jobStatus === 'uploading' || jobStatus === 'queued' || jobStatus === 'running'

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader
        back={() => navigate(`/projects/${projectId}/images`)}
        title="Multi-View 3D Reconstruction"
        subtitle={project?.name}
      />

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 16 }}>

        {/* ═══ SIDEBAR ══════════════════════════════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* How to shoot */}
          <Card style={{ padding: 16 }}>
            <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
              textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 10 }}>
              How to get best results
            </p>
            {[
              ['📸', '4–8 photos', 'Walk around the object every 30–45°'],
              ['🔄', 'Full coverage', 'Capture front, sides, back, and top'],
              ['💡', 'Even lighting', 'Avoid harsh shadows & reflections'],
              ['🎯', 'Overlap', 'Each photo should share ~60% with the next'],
              ['📐', 'Keep still', 'Hold steady — motion blur hurts matching'],
            ].map(([icon, title, desc]) => (
              <div key={title} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                <span style={{ fontSize: 14, flexShrink: 0 }}>{icon}</span>
                <div>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', lineHeight: 1.3 }}>{title}</p>
                  <p style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.4 }}>{desc}</p>
                </div>
              </div>
            ))}
          </Card>

          {/* Quality slider */}
          <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
            <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
              textTransform: 'uppercase', letterSpacing: '0.07em' }}>
              Quality
            </p>
            <Slider label="Alignment iterations" value={niter} onChange={setNiter}
              min={50} max={500} step={50}
              format={v => v <= 100 ? `${v} (fast)` : v >= 400 ? `${v} (best)` : String(v)} />
            <p style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.6 }}>
              More iterations = better alignment but longer processing.
              200 is a good default for most scenes.
            </p>

            {/* Point size (only relevant after done) */}
            {show3D && (
              <Slider label="Point size" value={pointSize} onChange={setPointSize}
                min={1} max={6} step={0.5} format={v => String(v)} />
            )}
          </Card>

          {/* Hardware note */}
          <Card style={{ padding: 14 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
              <Info size={13} style={{ color: 'var(--accent)', flexShrink: 0, marginTop: 1 }} />
              <div>
                <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
                  GPU recommended
                </p>
                <p style={{ fontSize: 11, color: 'var(--text3)', lineHeight: 1.6 }}>
                  4 images on GPU: ~20–30 s<br />
                  4 images on CPU: ~5–15 min<br />
                  Model: ~330 MB (auto-downloads)
                </p>
              </div>
            </div>
          </Card>

          {/* DUSt3R install note */}
          <Card style={{ padding: 14, background: 'rgba(94,106,210,0.06)',
            border: '1px solid rgba(94,106,210,0.2)' }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--accent)', marginBottom: 4 }}>
              First-time setup
            </p>
            <p style={{ fontSize: 11, color: 'var(--text2)', lineHeight: 1.6 }}>
              Run <code style={{ fontFamily: 'monospace', color: 'var(--accent)' }}>setup_dust3r.bat</code> in
              the NoCode CV folder to install DUSt3R, then restart the server.
            </p>
          </Card>
        </div>

        {/* ═══ MAIN PANEL ══════════════════════════════════════════════ */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Upload zone */}
          <Card style={{ padding: 16 }}>
            {/* Drop zone */}
            <div ref={dropRef} onDragOver={onDragOver} onDrop={onDrop}
              onClick={() => files.length < 12 && fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${files.length >= 2 ? 'var(--accent)' : 'var(--border2)'}`,
                borderRadius: 8, padding: '24px 16px', textAlign: 'center',
                cursor: files.length < 12 ? 'pointer' : 'default',
                transition: 'border-color 0.15s',
                background: files.length >= 2 ? 'rgba(94,106,210,0.04)' : 'transparent',
                marginBottom: files.length > 0 ? 14 : 0,
              }}>
              <Camera size={28} style={{ color: 'var(--text3)', opacity: 0.5, marginBottom: 8 }} />
              <p style={{ fontSize: 13, color: 'var(--text2)', marginBottom: 4 }}>
                {files.length === 0
                  ? 'Drag & drop photos here, or click to browse'
                  : files.length < 2
                  ? `${files.length} photo — add at least 1 more`
                  : `${files.length} / 12 photos — ${files.length < 12 ? 'click to add more' : 'maximum reached'}`}
              </p>
              <p style={{ fontSize: 11, color: 'var(--text3)' }}>
                JPG, PNG, WebP · 2–12 images · walk around the subject
              </p>
            </div>
            <input ref={fileInputRef} type="file" multiple accept="image/*"
              style={{ display: 'none' }}
              onChange={e => { if (e.target.files) addFiles(e.target.files); e.target.value = '' }} />

            {/* Image preview grid */}
            {files.length > 0 && (
              <div style={{ display: 'grid',
                gridTemplateColumns: 'repeat(auto-fill, minmax(90px, 1fr))',
                gap: 6, marginBottom: 14 }}>
                {files.map((f, i) => (
                  <div key={i} style={{ position: 'relative', borderRadius: 6, overflow: 'hidden',
                    border: '1px solid var(--border)', aspectRatio: '1' }}>
                    <img src={previews[i]} alt={f.name}
                      style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} />
                    <button onClick={e => { e.stopPropagation(); removeFile(i) }}
                      disabled={isProcessing}
                      style={{ position: 'absolute', top: 3, right: 3, width: 18, height: 18,
                        borderRadius: '50%', background: 'rgba(0,0,0,0.65)',
                        border: 'none', color: '#fff', cursor: 'pointer',
                        display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <X size={10} />
                    </button>
                    <span style={{ position: 'absolute', bottom: 3, left: 4,
                      fontSize: 9, color: 'rgba(255,255,255,0.7)',
                      fontFamily: 'JetBrains Mono, monospace' }}>
                      {i + 1}
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Action bar */}
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
              <Btn variant="primary"
                disabled={files.length < 2 || isProcessing}
                onClick={startReconstruction}>
                {isProcessing
                  ? <><Loader size={12} className="animate-spin" /> Reconstructing…</>
                  : <><Box size={12} /> Reconstruct 3D</>}
              </Btn>

              {files.length > 0 && !isProcessing && (
                <Btn variant="ghost" size="sm" onClick={clearAll}>
                  <X size={12} /> Clear all
                </Btn>
              )}

              {files.length < 2 && files.length > 0 && (
                <span style={{ fontSize: 11, color: 'var(--warn)' }}>
                  ⚠ Need at least 2 images
                </span>
              )}
            </div>
          </Card>

          {/* Status card */}
          {jobStatus !== 'idle' && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
                <div style={{ flexShrink: 0, marginTop: 2 }}>
                  {jobStatus === 'done'   && <CheckCircle size={18} style={{ color: '#3fb950' }} />}
                  {jobStatus === 'failed' && <XCircle     size={18} style={{ color: '#f85149' }} />}
                  {isProcessing          && <Loader size={18} className="animate-spin" style={{ color: 'var(--accent)' }} />}
                </div>
                <div style={{ flex: 1 }}>
                  <p style={{ fontSize: 13, fontWeight: 600, color: 'var(--text)', marginBottom: 4 }}>
                    {jobStatus === 'done'   && '3D reconstruction complete'}
                    {jobStatus === 'failed' && 'Reconstruction failed'}
                    {isProcessing          && 'Processing…'}
                  </p>
                  <p style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.6 }}>
                    {jobMsg}
                  </p>
                  {jobError && (
                    <div style={{ marginTop: 8, padding: '8px 12px', borderRadius: 6,
                      background: 'rgba(248,81,73,0.1)', border: '1px solid rgba(248,81,73,0.3)',
                      fontSize: 12, color: '#f85149' }}>
                      {jobError}
                    </div>
                  )}

                  {/* Processing steps indicator */}
                  {isProcessing && (
                    <div style={{ marginTop: 10, display: 'flex', flexDirection: 'column', gap: 4 }}>
                      {STATUS_STEPS.map((step, i) => {
                        const active = jobMsg.includes(step.split('…')[0].split(' ').slice(-2).join(' '))
                        const done   = STATUS_STEPS.slice(0, i).some(s =>
                          jobMsg.includes(s.split('…')[0].split(' ').slice(-2).join(' ')))
                        return (
                          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 7,
                            fontSize: 11,
                            color: active ? 'var(--accent)' : done ? 'var(--text3)' : 'var(--text3)',
                            opacity: active ? 1 : done ? 0.5 : 0.35 }}>
                            <div style={{ width: 6, height: 6, borderRadius: '50%', flexShrink: 0,
                              background: active ? 'var(--accent)' : done ? 'var(--success)' : 'var(--border2)' }} />
                            {step}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </div>
              </div>
            </Card>
          )}

          {/* 3D Viewer */}
          {jobStatus === 'done' && jobId && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8,
                marginBottom: 12, flexWrap: 'wrap' }}>
                <Btn variant={show3D ? 'primary' : 'secondary'} size="sm"
                  onClick={() => setShow3D(v => !v)}>
                  <Box size={12} />
                  {show3D ? 'Hide 3D View' : 'View 3D Point Cloud'}
                </Btn>
                <Btn variant="ghost" size="sm"
                  onClick={() => window.open(`/api/reconstruct3d/${jobId}/download`, '_blank')}>
                  <Download size={12} /> Download PLY
                </Btn>
                <span style={{ fontSize: 11, color: 'var(--text3)', marginLeft: 'auto' }}>
                  Opens in MeshLab · Blender · CloudCompare
                </span>
              </div>

              {show3D && (
                <>
                  <PointCloudViewer
                    fetchUrl={`/reconstruct3d/${jobId}/result`}
                    pointSize={pointSize}
                  />
                  <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 8 }}>
                    Reconstructed from {files.length} photos using DUSt3R ·
                    Coloured by original image RGB · Full-resolution PLY available for download
                  </p>
                </>
              )}
            </Card>
          )}

          {/* Empty state */}
          {jobStatus === 'idle' && files.length === 0 && (
            <Card style={{ padding: 40, textAlign: 'center' }}>
              <Box size={40} style={{ color: 'var(--text3)', opacity: 0.3, margin: '0 auto 12px' }} />
              <p style={{ fontSize: 14, color: 'var(--text2)', marginBottom: 6 }}>
                Upload 2–12 photos of the same object or scene
              </p>
              <p style={{ fontSize: 12, color: 'var(--text3)', maxWidth: 400, margin: '0 auto' }}>
                Walk around the subject taking a photo every 30–45°.
                DUSt3R will reconstruct a real 3D point cloud from multiple viewpoints —
                no COLMAP, no training, no special equipment required.
              </p>
            </Card>
          )}

        </div>
      </div>
    </div>
  )
}
