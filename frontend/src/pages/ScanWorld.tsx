import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Box, Crosshair, Download, Film, Link2, Loader, MousePointerClick, RotateCcw, Search, Upload } from 'lucide-react'
import api from '../api'
import { PageHeader, Card, Btn, Slider, InfoBanner, ProgressBar, Badge } from '../components/ui'
import SplatViewer, { type ViewMode } from '../components/SplatViewer'

interface ModelStatus { mapanything: boolean; sam2: boolean; cuda: boolean; gpu: string | null; vram_gb: number | null; ready: boolean }
interface Obj { id: string; name: string; n_splats: number; size_m: number[]; center: number[]; radius: number; frames_seen: number }
interface Job {
  status: 'running' | 'done' | 'error'; stage: string; progress: number; msg: string; error?: string | null
  n_views?: number; n_splats?: number; width?: number; height?: number; center?: number[]; radius?: number; objects?: Obj[]
}
interface Hit { id: string; title: string; channel: string; duration: number; thumbnail: string; url: string }
interface Click { x: number; y: number; label: 0 | 1 }

const STAGES = [['download', 'Download'], ['frames', 'Frames'], ['model', 'Model'], ['geometry', '3D geometry'], ['splats', 'Splats']] as const
const keyFor = (pid?: string) => `kh-scan-job-${pid}`
// "1:20", "80" or "1:02:03" -> seconds; empty -> undefined
const toSec = (t: string): number | undefined => {
  const v = t.trim(); if (!v) return undefined
  const parts = v.split(':').map(Number)
  if (parts.some(n => !Number.isFinite(n) || n < 0)) return NaN
  return parts.reduce((a, n) => a * 60 + n, 0)
}
const fmt = (m: number) => m >= 1 ? `${m.toFixed(2)} m` : `${(m * 100).toFixed(0)} cm`

export default function ScanWorld() {
  const { id: projectId } = useParams()
  const navigate = useNavigate()
  const [ms, setMs] = useState<ModelStatus | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [source, setSource] = useState<'file' | 'link'>('file')
  const [url, setUrl] = useState('')
  const [from, setFrom] = useState('')
  const [to, setTo] = useState('')
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<Hit[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [picked, setPicked] = useState<string | null>(null)
  const [frames, setFrames] = useState(40)
  const [detail, setDetail] = useState(2)
  const [jobId, setJobId] = useState<string | null>(() => { try { return localStorage.getItem(keyFor(projectId)) } catch { return null } })
  const [job, setJob] = useState<Job | null>(null)
  const [err, setErr] = useState('')
  const [frame, setFrame] = useState(0)
  const [clicks, setClicks] = useState<Click[]>([])
  const [segBusy, setSegBusy] = useState(false)
  const [active, setActive] = useState<Obj | null>(null)
  const [mode, setMode] = useState<ViewMode>('highlight')
  const dropRef = useRef<HTMLLabelElement>(null)

  useEffect(() => { api.get('/scan3d/model-status').then(r => setMs(r.data)).catch(() => setMs(null)) }, [])

  // poll the running job
  useEffect(() => {
    if (!jobId) return
    let live = true
    const tick = async () => {
      try {
        const r = await api.get(`/scan3d/${jobId}/status`)
        if (!live) return
        setJob(r.data)
        if (r.data.status === 'running') setTimeout(tick, 1000)
      } catch { if (live) { setJob(null); setJobId(null) } }
    }
    tick()
    return () => { live = false }
  }, [jobId])

  const isVideo = source === 'link' || files.some(f => f.type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(f.name))
  const t0 = toSec(from), t1 = toSec(to)
  const rangeBad = Number.isNaN(t0) || Number.isNaN(t1) || (t0 !== undefined && t1 !== undefined && t1 - t0 < 2)
  const canStart = !!ms?.ready && job?.status !== 'running' && !rangeBad && (source === 'link' ? /^https?:\/\/\S+/i.test(url.trim()) : files.length > 0)

  const start = async () => {
    if (!canStart) return
    setErr(''); setActive(null); setClicks([])
    try {
      let r
      if (source === 'link') {
        r = await api.post('/scan3d/start-url', { url: url.trim(), frames, detail, start: t0, end: t1 })
      } else {
        const fd = new FormData()
        files.forEach(f => fd.append('files', f))
        fd.append('frames', String(frames)); fd.append('detail', String(detail))
        if (t0 !== undefined) fd.append('start', String(t0))
        if (t1 !== undefined) fd.append('end', String(t1))
        r = await api.post('/scan3d/start', fd)
      }
      setJobId(r.data.job_id)
      try { localStorage.setItem(keyFor(projectId), r.data.job_id) } catch { /* ignore */ }
    } catch (e: unknown) {
      setErr((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? "Couldn't start the scan.")
    }
  }

  const runSearch = async () => {
    if (q.trim().length < 2 || searching) return
    setSearching(true); setErr('')
    try {
      const r = await api.get('/scan3d/search', { params: { q: q.trim() } })
      setHits(r.data.results)
    } catch (e: unknown) {
      setErr((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Search failed.')
      setHits(null)
    } finally { setSearching(false) }
  }
  const pick = (h: Hit) => { setUrl(h.url); setPicked(h.id); setFrom(''); setTo('') }
  const mmss = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`

  const findObject = async () => {
    if (!jobId || !clicks.length) return
    setSegBusy(true); setErr('')
    try {
      const r = await api.post(`/scan3d/${jobId}/segment`, { frame, points: clicks.map(c => [c.x, c.y]), labels: clicks.map(c => c.label) })
      setJob(j => j ? { ...j, objects: [...(j.objects ?? []), r.data] } : j)
      setActive(r.data); setMode('highlight'); setClicks([])
    } catch (e: unknown) {
      setErr((e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Segmentation failed.')
    } finally { setSegBusy(false) }
  }

  const onPick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    e.preventDefault()
    const r = e.currentTarget.getBoundingClientRect()
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height
    setClicks(c => [...c, { x, y, label: e.button === 2 || e.shiftKey ? 0 : 1 }])
  }, [])

  const done = job?.status === 'done'
  const sceneUrl = jobId ? `/api/scan3d/${jobId}/scene.ply` : ''

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader back={() => navigate(`/projects/${projectId}/images`)} title="Scan the World" khmer="ស្កេនពិភពពិត"
        subtitle="Film a place or object, get a real-world 3D scene, then click anything to cut it out" />

      {ms && !ms.ready && (
        <InfoBanner>
          <b>One-time setup needed.</b> Close NoCode CV, double-click <code>setup_scan3d.bat</code> in the NoCode CV folder, then start the app again.
          It installs MapAnything and SAM 2 (both Apache-2.0); no CUDA compiling is needed. The model weights (about 5 GB) download on the first scan.
        </InfoBanner>
      )}
      {ms?.ready && !ms.cuda && <InfoBanner>No NVIDIA GPU was found, so scans will be very slow on the CPU.</InfoBanner>}

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 340px) minmax(0, 1fr)', gap: 16, alignItems: 'start' }} className="scan-grid">
        {/* ── capture ─────────────────────────────────────────────── */}
        <Card style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}><Film size={15} style={{ color: 'var(--accent)' }} /><b style={{ fontSize: 13.5 }}>1. Capture</b></div>
          <div role="tablist" style={{ display: 'flex', border: '1px solid var(--border)', borderRadius: 999, padding: 3, gap: 3 }}>
            {([['file', 'Upload', Upload], ['link', 'Paste a link', Link2]] as const).map(([k, label, Icon]) => (
              <button key={k} role="tab" aria-selected={source === k} onClick={() => setSource(k)}
                style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6, padding: '6px 10px', border: 'none', borderRadius: 999, cursor: 'pointer',
                  fontSize: 12.5, fontWeight: 600, fontFamily: 'inherit', background: source === k ? 'var(--accent)' : 'transparent', color: source === k ? 'var(--on-accent)' : 'var(--text2)' }}>
                <Icon size={13} /> {label}
              </button>
            ))}
          </div>
          {source === 'file' ? (
          <label ref={dropRef}
              onDragOver={e => { e.preventDefault(); dropRef.current?.style.setProperty('border-color', 'var(--accent)') }}
              onDragLeave={() => dropRef.current?.style.removeProperty('border-color')}
              onDrop={e => { e.preventDefault(); dropRef.current?.style.removeProperty('border-color'); setFiles(Array.from(e.dataTransfer.files)) }}
              style={{ border: '1.5px dashed var(--border2)', borderRadius: 'var(--radius)', padding: '20px 14px', textAlign: 'center', cursor: 'pointer', display: 'block' }}>
              <input type="file" accept="video/*,image/*" multiple hidden onChange={e => setFiles(Array.from(e.target.files ?? []))} />
              <Upload size={20} style={{ color: 'var(--accent)', margin: '0 auto 6px', display: 'block' }} />
              <p style={{ fontSize: 13, color: 'var(--text)' }}>{files.length ? `${files.length} file${files.length > 1 ? 's' : ''} ready` : 'Drop a video or photos'}</p>
              <p style={{ fontSize: 11.5, color: 'var(--text3)', marginTop: 2 }}>{files.length ? files.slice(0, 2).map(f => f.name).join(', ') + (files.length > 2 ? '…' : '') : 'or click to choose'}</p>
            </label>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div style={{ display: 'flex', gap: 6 }}>
                <input value={q} onChange={e => setQ(e.target.value)} placeholder="Search short clips, e.g. Bayon drone" aria-label="Search YouTube"
                  onKeyDown={e => { if (e.key === 'Enter') runSearch() }}
                  style={{ flex: 1, minWidth: 0, padding: '9px 12px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit' }} />
                <Btn variant="secondary" onClick={runSearch} disabled={q.trim().length < 2 || searching}>
                  {searching ? <Loader size={13} className="animate-spin" /> : <Search size={13} />}
                </Btn>
              </div>
              {hits && (hits.length === 0
                ? <p style={{ fontSize: 12, color: 'var(--text3)' }}>No clips under 3 minutes found. Try words like drone, walk around, 360.</p>
                : <div style={{ display: 'flex', flexDirection: 'column', gap: 6, maxHeight: 300, overflowY: 'auto', paddingRight: 2 }}>
                    {hits.map(h => (
                      <button key={h.id} onClick={() => pick(h)} title={h.title}
                        style={{ display: 'flex', gap: 8, alignItems: 'center', padding: 4, textAlign: 'left', cursor: 'pointer', fontFamily: 'inherit',
                          borderRadius: 8, border: `1px solid ${picked === h.id ? 'var(--accent)' : 'transparent'}`, background: picked === h.id ? 'var(--accent-t)' : 'transparent' }}>
                        <span style={{ position: 'relative', flexShrink: 0 }}>
                          <img src={h.thumbnail} alt="" loading="lazy" style={{ width: 96, height: 54, objectFit: 'cover', borderRadius: 5, display: 'block' }} />
                          <span style={{ position: 'absolute', right: 3, bottom: 3, fontSize: 10, padding: '0 4px', borderRadius: 3, background: 'rgba(0,0,0,.75)', color: '#fff', fontFamily: 'var(--font-mono)' }}>{mmss(h.duration)}</span>
                        </span>
                        <span style={{ minWidth: 0 }}>
                          <span style={{ display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden', fontSize: 12.5, color: 'var(--text)', lineHeight: 1.35 }}>{h.title}</span>
                          <span style={{ display: 'block', fontSize: 11, color: 'var(--text3)', marginTop: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{h.channel}</span>
                        </span>
                      </button>
                    ))}
                  </div>)}
              <input value={url} onChange={e => setUrl(e.target.value)} placeholder="https://www.youtube.com/watch?v=…" aria-label="Video link"
                onKeyDown={e => { if (e.key === 'Enter') start() }}
                style={{ width: '100%', padding: '9px 12px', background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', color: 'var(--text)', fontSize: 13, fontFamily: 'inherit' }} />
              <p style={{ fontSize: 11.5, color: 'var(--text3)', lineHeight: 1.5 }}>Search above, or paste a link from YouTube, Vimeo, TikTok and most video sites. Only use videos you own or that are licensed for reuse.</p>
            </div>
          )}
          {isVideo && (
            <div>
              <p style={{ fontSize: 12.5, color: 'var(--text2)', fontWeight: 500, marginBottom: 6 }}>Part of the video to scan <span style={{ color: 'var(--text3)', fontWeight: 400 }}>optional</span></p>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                {([['From', from, setFrom, '0:00'], ['To', to, setTo, 'end']] as const).map(([label, val, set, ph]) => (
                  <label key={label} style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--text3)' }}>
                    {label}
                    <input value={val} onChange={e => set(e.target.value)} placeholder={ph} inputMode="numeric" aria-label={`${label} time`}
                      style={{ width: '100%', padding: '7px 10px', background: 'var(--surface2)', border: `1px solid ${rangeBad ? 'var(--danger)' : 'var(--border)'}`, borderRadius: 'var(--radius)', color: 'var(--text)', fontSize: 13, fontFamily: 'var(--font-mono)' }} />
                  </label>
                ))}
              </div>
              <p style={{ fontSize: 11.5, color: rangeBad ? 'var(--danger)' : 'var(--text3)', marginTop: 5 }}>
                {rangeBad ? 'Use times like 1:20, with the end at least 2 s after the start.' : 'Pick 20–60 s where the camera moves around one place, e.g. 1:20 to 1:50.'}
              </p>
            </div>
          )}
          <Slider label="Frames to use" value={frames} onChange={v => setFrames(Math.round(v))} min={12} max={80} step={4} format={v => `${Math.round(v)}`} />
          <Slider label="Detail" value={detail} onChange={v => setDetail(Math.round(v))} min={1} max={3} step={1}
            format={v => ['', 'Fast', 'Balanced', 'Sharp'][Math.round(v)]} />
          <Btn variant="primary" onClick={start} disabled={!canStart} style={{ justifyContent: 'center' }}>
            <Box size={14} /> {job?.status === 'running' ? 'Scanning…' : 'Build 3D scene'}
          </Btn>
          <div style={{ fontSize: 12, color: 'var(--text2)', lineHeight: 1.6 }}>
            <b style={{ color: 'var(--text)' }}>For a good scan:</b> walk slowly around the subject, keep it in frame, use even light.
            Avoid glass, mirrors and plain white walls. Things that move (people, cars) will blur. 20–60 seconds of video is plenty.
          </div>
        </Card>

        {/* ── result ──────────────────────────────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, minWidth: 0 }}>
          {err && <div role="alert" style={{ color: 'var(--danger)', fontSize: 13, padding: '10px 14px', background: 'var(--danger-s)', borderRadius: 'var(--radius)' }}>{err}</div>}

          {job && job.status !== 'done' && (
            <Card style={{ padding: 18, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {STAGES.map(([k, label]) => {
                  const idx = STAGES.findIndex(s => s[0] === job.stage), me = STAGES.findIndex(s => s[0] === k)
                  return <Badge key={k} color={job.status === 'error' && me === idx ? 'red' : me < idx ? 'green' : me === idx ? 'yellow' : 'gray'}>{label}</Badge>
                })}
              </div>
              {job.status === 'error'
                ? <p style={{ color: 'var(--danger)', fontSize: 13 }}>{job.error}</p>
                : <><ProgressBar value={Math.round(job.progress * 100)} max={100} label={job.msg} /><p style={{ fontSize: 12, color: 'var(--text3)' }}><Loader size={11} className="animate-spin" style={{ display: 'inline', verticalAlign: -1 }} /> This takes about a minute for 40 frames on an RTX-class GPU.</p></>}
            </Card>
          )}

          {done && job.center && job.radius != null && (
            <Card style={{ padding: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '2px 4px 10px', flexWrap: 'wrap' }}>
                <b style={{ fontSize: 13.5 }}>2. Explore</b>
                <span style={{ fontSize: 12, color: 'var(--text3)' }}>{job.msg}. Drag to orbit, scroll to zoom, right-drag to pan.</span>
                <div style={{ flex: 1 }} />
                {active && (['highlight', 'object', 'hide', 'all'] as ViewMode[]).map(m => (
                  <Btn key={m} size="sm" variant={mode === m ? 'primary' : 'ghost'} onClick={() => setMode(m)}>
                    {{ highlight: 'Highlight', object: 'Object only', hide: 'Remove object', all: 'Everything' }[m]}
                  </Btn>
                ))}
                <Btn size="sm" variant="secondary" href={sceneUrl}><Download size={13} /> Scene .ply</Btn>
              </div>
              <SplatViewer sceneUrl={sceneUrl} mode={mode}
                objectUrl={active ? `/api/scan3d/${jobId}/object/${active.id}.ply?part=object` : undefined}
                restUrl={active ? `/api/scan3d/${jobId}/object/${active.id}.ply?part=rest` : undefined}
                center={active && mode === 'object' ? active.center : job.center} radius={active && mode === 'object' ? Math.max(active.radius, 0.05) * 2.2 : job.radius} />
            </Card>
          )}

          {done && job.n_views != null && (
            <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 12 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Crosshair size={15} style={{ color: 'var(--accent)' }} /><b style={{ fontSize: 13.5 }}>3. Pick an object</b>
                <span style={{ fontSize: 12, color: 'var(--text3)' }}>Click it on any frame (shift or right-click to exclude a part). SAM 2 follows it through every frame.</span>
              </div>
              <div style={{ display: 'flex', gap: 6, overflowX: 'auto', paddingBottom: 4 }}>
                {Array.from({ length: job.n_views }, (_, i) => (
                  <button key={i} onClick={() => { setFrame(i); setClicks([]) }} aria-label={`Frame ${i + 1}`}
                    style={{ flexShrink: 0, padding: 0, border: `2px solid ${i === frame ? 'var(--accent)' : 'transparent'}`, borderRadius: 6, background: 'none', cursor: 'pointer' }}>
                    <img src={`/api/scan3d/${jobId}/frame/${i}.jpg`} alt="" loading="lazy" style={{ display: 'block', height: 52, borderRadius: 4 }} />
                  </button>
                ))}
              </div>
              <div onMouseDown={onPick} onContextMenu={e => e.preventDefault()}
                style={{ position: 'relative', cursor: 'crosshair', alignSelf: 'flex-start', maxWidth: '100%', borderRadius: 8, overflow: 'hidden' }}>
                <img src={`/api/scan3d/${jobId}/frame/${frame}.jpg`} alt={`Frame ${frame + 1}`} draggable={false}
                  style={{ display: 'block', maxWidth: '100%', maxHeight: 360, userSelect: 'none' }} />
                {clicks.map((c, i) => (
                  <span key={i} style={{ position: 'absolute', left: `${c.x * 100}%`, top: `${c.y * 100}%`, width: 14, height: 14, marginLeft: -7, marginTop: -7, borderRadius: '50%',
                    border: '2px solid #fff', background: c.label ? 'var(--accent)' : 'var(--danger)', boxShadow: '0 0 0 2px rgba(0,0,0,.4)' }} />
                ))}
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <Btn variant="primary" onClick={findObject} disabled={!clicks.length || segBusy}>
                  {segBusy ? <><Loader size={13} className="animate-spin" /> Tracking through {job.n_views} frames…</> : <><MousePointerClick size={13} /> Find in 3D</>}
                </Btn>
                {clicks.length > 0 && <Btn variant="ghost" onClick={() => setClicks([])}><RotateCcw size={13} /> Clear clicks</Btn>}
              </div>
              {(job.objects?.length ?? 0) > 0 && (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 8 }}>
                  {job.objects!.map(o => (
                    <div key={o.id} onClick={() => { setActive(o); setMode('highlight') }} role="button" tabIndex={0}
                      onKeyDown={e => { if (e.key === 'Enter') { setActive(o); setMode('highlight') } }}
                      style={{ padding: '10px 12px', borderRadius: 'var(--radius)', cursor: 'pointer', background: active?.id === o.id ? 'var(--accent-t)' : 'var(--surface2)',
                        border: `1px solid ${active?.id === o.id ? 'var(--accent)' : 'var(--border)'}` }}>
                      <p style={{ fontSize: 13, fontWeight: 600 }}>{o.name}</p>
                      <p style={{ fontSize: 12, color: 'var(--text2)' }}>{fmt(o.size_m[0])} × {fmt(o.size_m[1])}{o.size_m[2] > 0.01 ? ` × ${fmt(o.size_m[2])}` : ''}</p>
                      <p style={{ fontSize: 11.5, color: 'var(--text3)' }}>{o.n_splats.toLocaleString()} splats, seen in {o.frames_seen} frames</p>
                      <a href={`/api/scan3d/${jobId}/object/${o.id}.ply?part=object`} onClick={e => e.stopPropagation()} style={{ fontSize: 12, color: 'var(--accent)' }}>Download object .ply</a>
                    </div>
                  ))}
                </div>
              )}
              <p style={{ fontSize: 11.5, color: 'var(--text3)' }}>Sizes are in real metres from MapAnything's metric depth; expect a few percent error.</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
