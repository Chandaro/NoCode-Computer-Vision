import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Zap, Download, Loader, RefreshCw, CheckCircle, XCircle, Upload } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import api, { type Project, type ClassificationRun, type ClsInferResult } from '../api'
import {
  Card, Label, PageHeader, Btn, Field, Select, Slider,
  LogTerminal, ProgressBar, Badge,
} from '../components/ui'

interface Progress   { epoch: number; total: number; accuracy: number }
interface ChartPoint { epoch: number; accuracy: number }

const BASE_MODELS = [
  { value: 'resnet18',           label: 'ResNet-18 (fastest)'         },
  { value: 'resnet50',           label: 'ResNet-50'                    },
  { value: 'mobilenet_v3_small', label: 'MobileNet V3 Small'           },
  { value: 'efficientnet_b0',    label: 'EfficientNet-B0 (best acc.)'  },
]

const TT = {
  contentStyle: {
    background: 'var(--surface2)', border: '1px solid var(--border)',
    borderRadius: 6, fontSize: 11,
  },
}

export default function Classification() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project,   setProject]   = useState<Project | null>(null)
  const [runs,      setRuns]      = useState<ClassificationRun[]>([])
  const [epochs,    setEpochs]    = useState(10)
  const [imgsz,     setImgsz]     = useState(224)
  const [batch,     setBatch]     = useState(32)
  const [baseModel, setBaseModel] = useState('resnet18')
  const [lr,        setLr]        = useState('0.001')
  const [freeze,    setFreeze]    = useState(true)

  const [streaming, setStreaming]  = useState(false)
  const [logs,      setLogs]       = useState<string[]>([])
  const [progress,  setProgress]   = useState<Progress | null>(null)
  const [chartData, setChartData]  = useState<ChartPoint[]>([])

  // ── Inference ──────────────────────────────────────────────────────────────
  const [inferRunId,   setInferRunId]   = useState('')
  const [inferFile,    setInferFile]    = useState<File | null>(null)
  const [inferName,    setInferName]    = useState('')
  const [inferRunning, setInferRunning] = useState(false)
  const [inferResult,  setInferResult]  = useState<ClsInferResult | null>(null)
  const inferFileRef = useRef<HTMLInputElement>(null)

  // ── Confusion matrix expand ────────────────────────────────────────────────
  const [expandedCm, setExpandedCm] = useState<number | null>(null)

  const logRef = useRef<HTMLDivElement>(null)
  const esRef  = useRef<EventSource | null>(null)

  useEffect(() => {
    api.get(`/projects/${projectId}`).then(r => setProject(r.data))
    loadRuns()
  }, [projectId])

  const loadRuns = () =>
    api.get(`/projects/${projectId}/classification/runs`).then(r => {
      setRuns(r.data)
    })

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [logs])

  useEffect(() => () => { esRef.current?.close() }, [])

  const startTraining = async () => {
    esRef.current?.close()
    setLogs([]); setChartData([]); setProgress(null)
    const res = await api.post(`/projects/${projectId}/classification/start`, {
      epochs, imgsz, batch, base_model: baseModel,
      lr: Number(lr), freeze_backbone: freeze,
    })
    const run: ClassificationRun = res.data
    setStreaming(true)
    const es = new EventSource(
      `/api/projects/${projectId}/classification/runs/${run.id}/logs`
    )
    esRef.current = es
    es.onmessage = (e) => {
      const msg: string = e.data
      if (msg === '__END__') {
        es.close(); esRef.current = null; setStreaming(false); loadRuns(); return
      }
      if (msg.startsWith('__PROGRESS__:')) {
        const parts = msg.split(':')
        if (parts.length < 5) return
        const [, ep,, acc] = parts
        const epochNum = Number(ep.split('/')[0])
        const totalNum = Number(ep.split('/')[1])
        const accVal   = Number(acc)
        setProgress({ epoch: epochNum, total: totalNum, accuracy: accVal })
        setChartData(prev => [...prev, { epoch: epochNum, accuracy: accVal }])
        return
      }
      if (msg.startsWith('__DONE__:') || msg === '__FAILED__') {
        setStreaming(false); loadRuns()
      }
      setLogs(prev => [...prev, msg])
    }
    es.onerror = () => {
      es.close(); esRef.current = null; setStreaming(false); loadRuns()
      setLogs(prev => [...prev, '⚠️ Connection lost'])
    }
  }

  const runInference = async () => {
    if (!inferFile || !inferRunId) return
    setInferRunning(true); setInferResult(null)
    try {
      const fd = new FormData()
      fd.append('file', inferFile)
      const res = await api.post(
        `/projects/${projectId}/classification/runs/${inferRunId}/infer`, fd,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setInferResult(res.data)
    } finally {
      setInferRunning(false)
    }
  }

  const pct = progress ? Math.round((progress.epoch / progress.total) * 100) : 0

  const statusBadge = (s: string) => {
    if (s === 'done')    return <Badge color="green"><CheckCircle size={11} /> done</Badge>
    if (s === 'failed')  return <Badge color="red"><XCircle size={11} /> failed</Badge>
    if (s === 'running') return <Badge color="yellow"><Loader size={11} className="animate-spin" /> running</Badge>
    return <Badge color="gray">pending</Badge>
  }

  const doneRuns = runs.filter(r => r.status === 'done')

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader
        back={() => navigate(`/projects/${projectId}/images`)}
        title="Image Classification"
        subtitle={`${project?.name ?? '…'} · Transfer Learning`}
      />

      {/* Info banner */}
      <div style={{ padding: '10px 14px', background: 'rgba(88,101,242,0.08)',
        border: '1px solid rgba(88,101,242,0.22)', borderRadius: 8, marginBottom: 20,
        fontSize: 12, color: '#a5b4fc', lineHeight: 1.6 }}>
        ℹ️ Classification uses each image's <strong>first annotation class</strong> as its label.
        Each class needs at least 2 annotated images.
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: 12, alignItems: 'start' }}>

        {/* ── Config ── */}
        <Card style={{ padding: 16 }}>
          <Label>Model Config</Label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginTop: 8 }}>
            <Field label="Base Model">
              <Select value={baseModel} onChange={setBaseModel}>
                {BASE_MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </Select>
            </Field>
            <Slider label="Epochs" value={epochs} onChange={setEpochs}
              min={1} max={100} step={1} format={v => String(v)} />
            <Field label="Image Size">
              <Select value={String(imgsz)} onChange={v => setImgsz(Number(v))}>
                {[128, 224, 256, 384].map(s => (
                  <option key={s} value={s}>{s}{s === 224 ? ' (default)' : ''}</option>
                ))}
              </Select>
            </Field>
            <Slider label="Batch Size" value={batch} onChange={setBatch}
              min={4} max={128} step={4} format={v => String(v)} />
            <Field label="Learning Rate">
              <Select value={lr} onChange={setLr}>
                {['0.01','0.001','0.0001','0.00001'].map(v => (
                  <option key={v} value={v}>{v}</option>
                ))}
              </Select>
            </Field>

            {/* Freeze backbone toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: 10,
              cursor: 'pointer', userSelect: 'none' }}>
              <div onClick={() => setFreeze(f => !f)}
                style={{ width: 32, height: 18, borderRadius: 9,
                  background: freeze ? 'var(--accent)' : 'var(--surface3)',
                  border: `1px solid ${freeze ? 'var(--accent)' : 'var(--border2)'}`,
                  position: 'relative', flexShrink: 0,
                  transition: 'background 0.15s', cursor: 'pointer' }}>
                <div style={{ position: 'absolute', top: 2, left: freeze ? 14 : 2,
                  width: 12, height: 12, borderRadius: '50%', background: '#fff',
                  transition: 'left 0.15s' }} />
              </div>
              <span style={{ fontSize: 12, color: 'var(--text2)' }}>Freeze backbone (faster)</span>
            </label>

            <Btn variant="primary" onClick={startTraining} disabled={streaming}
              style={{ width: '100%', justifyContent: 'center', marginTop: 4 }}>
              {streaming
                ? <><Loader size={13} className="animate-spin" /> Training…</>
                : <><Zap size={13} strokeWidth={2.5} /> Start Training</>}
            </Btn>
          </div>
        </Card>

        {/* ── Right column ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>

          {/* Progress */}
          {(streaming || progress) && (
            <Card style={{ padding: 16 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between',
                alignItems: 'center', marginBottom: 12 }}>
                <Label>Progress</Label>
                {streaming
                  ? <span style={{ display: 'flex', alignItems: 'center', gap: 6,
                      fontSize: 12, color: 'var(--warn)' }}>
                      <Loader size={12} className="animate-spin" /> Running
                    </span>
                  : <span style={{ display: 'flex', alignItems: 'center', gap: 6,
                      fontSize: 12, color: 'var(--success)' }}>
                      <CheckCircle size={12} /> Done
                    </span>}
              </div>
              <ProgressBar value={progress?.epoch ?? 0} max={progress?.total ?? epochs}
                label={`Epoch ${progress?.epoch ?? 0} / ${progress?.total ?? epochs}`}
                sublabel={`${pct}%`} />
              {progress && (
                <div style={{ marginTop: 12, display: 'inline-block',
                  background: 'var(--surface2)', border: '1px solid var(--border)',
                  borderRadius: 6, padding: '8px 16px', textAlign: 'center' }}>
                  <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 2 }}>Val Accuracy</p>
                  <p style={{ fontSize: 18, fontWeight: 600,
                    fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent)' }}>
                    {(progress.accuracy * 100).toFixed(1)}%
                  </p>
                </div>
              )}
            </Card>
          )}

          {/* Accuracy chart */}
          {chartData.length > 1 && (
            <Card style={{ padding: 16 }}>
              <Label>Validation Accuracy</Label>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={chartData} margin={{ left: -16 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
                  <XAxis dataKey="epoch" stroke="var(--text3)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--text3)" tick={{ fontSize: 11 }} domain={[0, 1]}
                    tickFormatter={v => `${Math.round(v * 100)}%`} />
                  <Tooltip {...TT} formatter={(v: unknown) =>
                    [`${((v as number) * 100).toFixed(2)}%`]} />
                  <Line type="monotone" dataKey="accuracy" stroke="var(--accent)"
                    strokeWidth={2} dot={false} name="Accuracy" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Log */}
          <Card style={{ padding: 16 }}>
            <Label>Log</Label>
            <LogTerminal logs={logs} logRef={logRef} />
          </Card>

          {/* ── Inference ── */}
          {doneRuns.length > 0 && (
            <Card style={{ padding: 16 }}>
              <Label>Test Classification</Label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginTop: 8 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  <Field label="Run">
                    <Select value={inferRunId} onChange={setInferRunId}>
                      <option value="">Select a run…</option>
                      {doneRuns.map(r => {
                        const res = r.results as Record<string, unknown>
                        const acc = typeof res?.top1_acc === 'number'
                          ? ` — ${(res.top1_acc * 100).toFixed(1)}%` : ''
                        return <option key={r.id} value={r.id}>
                          Run #{r.id} · {r.base_model}{acc}
                        </option>
                      })}
                    </Select>
                  </Field>
                  <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <Btn variant="secondary" size="sm"
                      onClick={() => inferFileRef.current?.click()}>
                      <Upload size={12} /> Image
                    </Btn>
                    <input ref={inferFileRef} type="file" accept="image/*"
                      style={{ display: 'none' }}
                      onChange={e => {
                        const f = e.target.files?.[0]
                        if (f) { setInferFile(f); setInferName(f.name); setInferResult(null) }
                      }} />
                    <span style={{ fontSize: 11, color: 'var(--text3)',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {inferName || 'No file'}
                    </span>
                  </div>
                  <Btn variant="primary" size="sm" onClick={runInference}
                    disabled={!inferFile || !inferRunId || inferRunning}>
                    {inferRunning
                      ? <><Loader size={12} className="animate-spin" /> Running…</>
                      : <><Zap size={12} strokeWidth={2.5} /> Classify</>}
                  </Btn>
                </div>

                {/* Result */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {inferResult ? (
                    <>
                      {inferResult.top1 && (
                        <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)',
                          borderRadius: 8, padding: '10px 14px', marginBottom: 4 }}>
                          <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 4 }}>Top prediction</p>
                          <p style={{ fontSize: 16, fontWeight: 600, color: 'var(--text)' }}>
                            {inferResult.top1.class_name}
                          </p>
                          <p style={{ fontSize: 13, fontFamily: 'JetBrains Mono, monospace',
                            color: 'var(--accent)' }}>
                            {(inferResult.top1.probability * 100).toFixed(1)}%
                          </p>
                        </div>
                      )}
                      {inferResult.top5.slice(1).map((p, i) => (
                        <div key={i} style={{ display: 'flex', justifyContent: 'space-between',
                          alignItems: 'center', fontSize: 12, color: 'var(--text2)',
                          padding: '3px 0', borderTop: i === 0 ? '1px solid var(--border)' : 'none' }}>
                          <span>{p.class_name}</span>
                          <span style={{ fontFamily: 'JetBrains Mono, monospace',
                            color: 'var(--text3)', fontSize: 11 }}>
                            {(p.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </>
                  ) : (
                    <p style={{ fontSize: 12, color: 'var(--text3)', marginTop: 8 }}>
                      Results will appear here
                    </p>
                  )}
                </div>
              </div>
            </Card>
          )}

          {/* ── Run history ── */}
          <Card style={{ padding: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between',
              alignItems: 'center', marginBottom: 12 }}>
              <Label>Run History</Label>
              <button onClick={loadRuns}
                style={{ border: 'none', background: 'transparent',
                  color: 'var(--text3)', cursor: 'pointer', display: 'flex',
                  padding: 4, borderRadius: 4 }}>
                <RefreshCw size={13} />
              </button>
            </div>

            {runs.length === 0
              ? <p style={{ fontSize: 13, color: 'var(--text3)' }}>No runs yet.</p>
              : <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {[...runs].reverse().map(run => {
                    const res  = run.results as Record<string, unknown>
                    const acc1 = typeof res?.top1_acc === 'number' ? res.top1_acc : null
                    const acc5 = typeof res?.top5_acc === 'number' ? res.top5_acc : null
                    const cm   = Array.isArray(res?.confusion_matrix)
                      ? res.confusion_matrix as number[][] : null
                    const isCmExpanded = expandedCm === run.id

                    return (
                      <div key={run.id} style={{ background: 'var(--surface2)',
                        border: '1px solid var(--border)', borderRadius: 7 }}>
                        <div style={{ display: 'flex', alignItems: 'center',
                          justifyContent: 'space-between', padding: '10px 14px' }}>
                          <div>
                            <div style={{ display: 'flex', alignItems: 'center',
                              gap: 8, marginBottom: 3 }}>
                              {statusBadge(run.status)}
                              <span style={{ fontSize: 13, color: 'var(--text)',
                                fontWeight: 500 }}>Run #{run.id}</span>
                              <span style={{ fontSize: 11, color: 'var(--text3)',
                                fontFamily: 'JetBrains Mono, monospace' }}>
                                {run.base_model} · {run.epochs} ep
                              </span>
                            </div>
                            <p style={{ fontSize: 11, color: 'var(--text3)' }}>
                              {run.created_at.split('T')[0]}
                            </p>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                            {acc1 !== null && (
                              <div style={{ textAlign: 'right' }}>
                                <span style={{ fontSize: 12, fontFamily: 'JetBrains Mono, monospace',
                                  color: 'var(--accent)' }}>
                                  top-1: {(acc1 * 100).toFixed(1)}%
                                </span>
                                {acc5 !== null && (
                                  <span style={{ fontSize: 11, fontFamily: 'JetBrains Mono, monospace',
                                    color: 'var(--text3)', marginLeft: 8 }}>
                                    top-5: {(acc5 * 100).toFixed(1)}%
                                  </span>
                                )}
                              </div>
                            )}
                            {cm && (
                              <Btn variant="ghost" size="sm"
                                onClick={() => setExpandedCm(isCmExpanded ? null : run.id)}>
                                CM
                              </Btn>
                            )}
                            {run.status === 'done' && run.model_path && (
                              <Btn variant="secondary" size="sm"
                                href={`/api/projects/${projectId}/classification/runs/${run.id}/download`}>
                                <Download size={12} /> .pth
                              </Btn>
                            )}
                          </div>
                        </div>

                        {/* Confusion matrix */}
                        {isCmExpanded && cm && project && (
                          <div style={{ padding: '0 14px 14px',
                            borderTop: '1px solid var(--border)' }}>
                            <p style={{ fontSize: 11, color: 'var(--text3)', margin: '10px 0 6px',
                              textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
                              Confusion Matrix
                            </p>
                            <div style={{ overflowX: 'auto' }}>
                              <table style={{ borderCollapse: 'collapse', fontSize: 11 }}>
                                <thead>
                                  <tr>
                                    <th style={{ padding: '3px 8px', color: 'var(--text3)',
                                      textAlign: 'right', fontWeight: 400 }}>↓ True \ Pred →</th>
                                    {project.classes.map((c, j) => (
                                      <th key={j} style={{ padding: '3px 8px',
                                        color: 'var(--text3)', fontWeight: 500,
                                        whiteSpace: 'nowrap', maxWidth: 80,
                                        overflow: 'hidden', textOverflow: 'ellipsis' }}>
                                        {c}
                                      </th>
                                    ))}
                                  </tr>
                                </thead>
                                <tbody>
                                  {cm.map((row, i) => {
                                    const rowTotal = row.reduce((a, b) => a + b, 0)
                                    return (
                                      <tr key={i}>
                                        <td style={{ padding: '3px 8px', color: 'var(--text2)',
                                          fontWeight: 500, whiteSpace: 'nowrap' }}>
                                          {project.classes[i] ?? `cls${i}`}
                                        </td>
                                        {row.map((val, j) => {
                                          const intensity = rowTotal > 0 ? val / rowTotal : 0
                                          const isCorrect = i === j
                                          return (
                                            <td key={j} style={{
                                              padding: '3px 8px', textAlign: 'center',
                                              background: isCorrect
                                                ? `rgba(34,197,94,${intensity * 0.6})`
                                                : intensity > 0.1 ? `rgba(248,113,113,${intensity * 0.5})` : 'transparent',
                                              color: intensity > 0.4 ? '#fff' : 'var(--text2)',
                                              fontFamily: 'JetBrains Mono, monospace',
                                              fontWeight: isCorrect ? 600 : 400,
                                              borderRadius: 3,
                                            }}>
                                              {val}
                                            </td>
                                          )
                                        })}
                                      </tr>
                                    )
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        )}
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
