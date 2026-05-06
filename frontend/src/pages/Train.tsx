import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Zap, Download, RefreshCw, ChevronDown, ChevronUp, BarChart2, FileCode, Loader, Cpu, Square, Trash2 } from 'lucide-react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from 'recharts'
import api, { type Project, type TrainingRun } from '../api'
import { Card, Field, Select, Slider, Btn, Badge, PageHeader, LogTerminal, ProgressBar } from '../components/ui'

interface MetricPoint { epoch: number; mAP50: number; precision: number; recall: number }
interface Progress   { epoch: number; total: number; mAP50: number; precision: number; recall: number }

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
    es.onerror = () => { es.close(); esRef.current = null; setStreaming(false); loadRuns(); setLogs(prev => [...prev, '⚠️ Connection lost']) }
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
      <PageHeader back={() => navigate(`/projects/${projectId}/images`)}
        title="Object Detection Training" subtitle={project?.name} />

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 16 }}>

        {/* ── Config ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <Card style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <p style={{ fontSize: 11, fontWeight: 600, color: 'var(--text3)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Model</p>

            <Field label="Architecture">
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
        </div>
      </div>
    </div>
  )
}
