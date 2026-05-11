import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Zap, Download, Loader, RefreshCw, CheckCircle, XCircle, Upload, Square, Trash2, ChevronDown, ChevronUp } from 'lucide-react'
import { LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import api, { type Project, type ClassificationRun, type ClsInferResult } from '../api'
import {
  Card, Label, PageHeader, Btn, Field, Select, Slider,
  LogTerminal, ProgressBar, Badge,
} from '../components/ui'

interface Progress   { epoch: number; total: number; accuracy: number }
interface ChartPoint { epoch: number; accuracy: number; trainLoss: number; valLoss: number }

const BASE_MODELS = [
  { value: 'resnet18',           label: 'ResNet-18 (fastest)'             },
  { value: 'resnet34',           label: 'ResNet-34'                       },
  { value: 'resnet50',           label: 'ResNet-50'                       },
  { value: 'mobilenet_v3_small', label: 'MobileNet V3 Small (lightweight)'},
  { value: 'efficientnet_b0',    label: 'EfficientNet-B0'                 },
  { value: 'efficientnet_b1',    label: 'EfficientNet-B1 (best acc.)'     },
  { value: 'convnext_tiny',      label: 'ConvNeXt-Tiny (modern)'          },
]

const TT = {
  contentStyle: {
    background: 'var(--surface2)', border: '1px solid var(--border)',
    borderRadius: 6, fontSize: 11,
  },
}

const SEC = { fontSize: 10, fontWeight: 600, color: 'var(--text3)',
  textTransform: 'uppercase' as const, letterSpacing: '0.07em', marginTop: 6 }

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
  const [valSplit,  setValSplit]  = useState(0.2)
  const [patience,  setPatience]  = useState(0)
  const [resumeRunId, setResumeRunId] = useState('')
  // Optimizer
  const [optimizer,    setOptimizer]    = useState('Adam')
  const [weightDecay,  setWeightDecay]  = useState(0.0)
  const [momentum,     setMomentum]     = useState(0.9)
  const [warmupEpochs, setWarmupEpochs] = useState(0)
  // LR Scheduler
  const [lrScheduler, setLrScheduler] = useState('cosine')
  const [stepSize,    setStepSize]    = useState(10)
  const [stepGamma,   setStepGamma]   = useState(0.1)
  // Regularisation
  const [labelSmoothing, setLabelSmoothing] = useState(0.0)
  const [dropoutHead,    setDropoutHead]    = useState(0.0)
  // Augmentation
  const [showAug,    setShowAug]    = useState(false)
  const [fliplr,     setFliplr]     = useState(0.5)
  const [flipud,     setFlipud]     = useState(0.0)
  const [degrees,    setDegrees]    = useState(0.0)
  const [translate,  setTranslate]  = useState(0.0)
  const [scale,      setScale]      = useState(0.0)
  const [brightness, setBrightness] = useState(0.2)
  const [contrast,   setContrast]   = useState(0.2)
  const [saturation, setSaturation] = useState(0.2)
  const [erasing,    setErasing]    = useState(0.0)
  const [mixup,      setMixup]      = useState(0.0)

  const [streaming, setStreaming]  = useState(false)
  const [logs,      setLogs]       = useState<string[]>([])
  const [progress,  setProgress]   = useState<Progress | null>(null)
  const [chartData, setChartData]  = useState<ChartPoint[]>([])

  // ── Inference ──────────────────────────────────────────────────────────────
  const [inferRunId,    setInferRunId]    = useState('')
  const [inferFile,     setInferFile]     = useState<File | null>(null)
  const [inferName,     setInferName]     = useState('')
  const [inferPreview,  setInferPreview]  = useState<string | null>(null)
  const [inferRunning,  setInferRunning]  = useState(false)
  const [inferResult,   setInferResult]   = useState<ClsInferResult | null>(null)
  const [onnxExporting, setOnnxExporting] = useState<number | null>(null)
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
      val_split: valSplit, patience,
      resume_run_id: resumeRunId ? Number(resumeRunId) : null,
      optimizer, weight_decay: weightDecay, momentum, warmup_epochs: warmupEpochs,
      lr_scheduler: lrScheduler, step_size: stepSize, step_gamma: stepGamma,
      label_smoothing: labelSmoothing, dropout_head: dropoutHead,
      fliplr, flipud, degrees, translate, scale,
      brightness, contrast, saturation, erasing, mixup,
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
        // format: __PROGRESS__:epoch/total:acc:train_loss:val_loss
        const parts = msg.split(':')
        if (parts.length < 5) return
        const [, ep, accStr, tlossStr, vlossStr] = parts
        const epochNum = Number(ep.split('/')[0])
        const totalNum = Number(ep.split('/')[1])
        const accVal   = Number(accStr)
        const tloss    = Number(tlossStr)
        const vloss    = Number(vlossStr)
        setProgress({ epoch: epochNum, total: totalNum, accuracy: accVal })
        setChartData(prev => [...prev, { epoch: epochNum, accuracy: accVal, trainLoss: tloss, valLoss: vloss }])
        return
      }
      if (msg.startsWith('__DONE__:') || msg === '__FAILED__') {
        setStreaming(false); loadRuns()
      }
      setLogs(prev => [...prev, msg])
    }
    es.onerror = () => {
      es.close(); esRef.current = null; setStreaming(false); loadRuns()
      setLogs(prev => [...prev, '[WARN] Connection lost'])
    }
  }

  const exportOnnx = async (runId: number) => {
    setOnnxExporting(runId)
    try {
      const res = await api.post(
        `/projects/${projectId}/classification/runs/${runId}/export-onnx`,
        {}, { responseType: 'blob' }
      )
      const url = URL.createObjectURL(res.data)
      const a = document.createElement('a')
      a.href = url; a.download = `cls_model_run${runId}.onnx`; a.click()
      URL.revokeObjectURL(url)
    } finally {
      setOnnxExporting(null)
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
    if (s === 'stopped') return <Badge color="gray"><Square size={11} /> stopped</Badge>
    return <Badge color="gray">pending</Badge>
  }

  const stopRun = async (runId: number) => {
    await api.post(`/projects/${projectId}/classification/runs/${runId}/stop`)
    esRef.current?.close(); esRef.current = null; setStreaming(false)
    loadRuns()
  }

  const deleteRun = async (runId: number) => {
    await api.delete(`/projects/${projectId}/classification/runs/${runId}`)
    loadRuns()
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
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <Card style={{ padding: 16 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 13 }}>
            <p style={SEC}>Model</p>
            <Field label="Architecture">
              <Select value={baseModel} onChange={setBaseModel}>
                {BASE_MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
              </Select>
            </Field>
            <Slider label="Epochs"     value={epochs} onChange={setEpochs} min={1} max={200} step={1} format={v => String(v)} />
            <Field label="Image Size">
              <Select value={String(imgsz)} onChange={v => setImgsz(Number(v))}>
                {[128, 224, 256, 384].map(s => (
                  <option key={s} value={s}>{s}{s === 224 ? ' (default)' : ''}</option>
                ))}
              </Select>
            </Field>
            <Slider label="Batch Size" value={batch} onChange={setBatch} min={4} max={128} step={4} format={v => String(v)} />
            <Slider label="Val Split"  value={valSplit} onChange={setValSplit} min={0.1} max={0.4} step={0.05} format={v => `${Math.round(v*100)}%`} />
            <Slider label="Early Stop Patience" value={patience} onChange={setPatience} min={0} max={50} step={1} format={v => v === 0 ? 'Off' : `${v} ep`} />

            {/* Freeze backbone toggle */}
            <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', userSelect: 'none' }}>
              <div onClick={() => setFreeze(f => !f)}
                style={{ width: 32, height: 18, borderRadius: 9,
                  background: freeze ? 'var(--accent)' : 'var(--surface3)',
                  border: `1px solid ${freeze ? 'var(--accent)' : 'var(--border2)'}`,
                  position: 'relative', flexShrink: 0, transition: 'background 0.15s', cursor: 'pointer' }}>
                <div style={{ position: 'absolute', top: 2, left: freeze ? 14 : 2,
                  width: 12, height: 12, borderRadius: '50%', background: '#fff', transition: 'left 0.15s' }} />
              </div>
              <span style={{ fontSize: 12, color: 'var(--text2)' }}>Freeze backbone</span>
            </label>

            <p style={SEC}>Optimizer</p>
            <Field label="Algorithm">
              <Select value={optimizer} onChange={setOptimizer}>
                <option value="Adam">Adam (default)</option>
                <option value="AdamW">AdamW</option>
                <option value="SGD">SGD</option>
              </Select>
            </Field>
            <Field label="Learning Rate">
              <input
                type="number" value={lr}
                onChange={e => setLr(e.target.value)}
                step="0.0001" min="0.00001" max="1"
                style={{
                  width: '100%', background: 'var(--surface2)',
                  border: '1px solid var(--border)', borderRadius: 5,
                  color: 'var(--text)', padding: '5px 8px', fontSize: 12,
                }}
              />
            </Field>
            <Slider label="Weight Decay"   value={weightDecay}  onChange={setWeightDecay}  min={0} max={0.01} step={0.0001} format={v => v.toFixed(4)} />
            <Slider label="Momentum (SGD)" value={momentum}     onChange={setMomentum}     min={0.5} max={0.99} step={0.01} format={v => v.toFixed(2)} />
            <Slider label="Warmup Epochs"  value={warmupEpochs} onChange={setWarmupEpochs} min={0} max={10} step={1} format={v => String(v)} />

            <p style={SEC}>LR Scheduler</p>
            <Field label="Schedule">
              <Select value={lrScheduler} onChange={setLrScheduler}>
                <option value="cosine">Cosine Annealing (default)</option>
                <option value="step">Step LR</option>
                <option value="none">None (constant LR)</option>
              </Select>
            </Field>
            {lrScheduler === 'step' && (<>
              <Slider label="Step Size"  value={stepSize}  onChange={setStepSize}  min={1} max={50} step={1} format={v => `${v} ep`} />
              <Slider label="Step Gamma" value={stepGamma} onChange={setStepGamma} min={0.01} max={0.9} step={0.01} format={v => v.toFixed(2)} />
            </>)}

            <p style={SEC}>Regularisation</p>
            <Slider label="Label Smoothing" value={labelSmoothing} onChange={setLabelSmoothing} min={0} max={0.3} step={0.01} format={v => v.toFixed(2)} />
            <Slider label="Head Dropout"    value={dropoutHead}    onChange={setDropoutHead}    min={0} max={0.7} step={0.05} format={v => v.toFixed(2)} />

            {runs.filter(r => r.status === 'done').length > 0 && (<>
              <p style={SEC}>Resume</p>
              <Field label="Continue from run">
                <Select value={resumeRunId} onChange={setResumeRunId}>
                  <option value="">— Start fresh —</option>
                  {runs.filter(r => r.status === 'done').map(r => {
                    const acc = (r.results as Record<string, unknown>)?.top1_acc
                    return <option key={r.id} value={r.id}>
                      Run #{r.id} · {r.base_model}{typeof acc === 'number' ? ` · ${(acc*100).toFixed(1)}%` : ''}
                    </option>
                  })}
                </Select>
              </Field>
            </>)}
          </div>
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
            <div style={{ padding: '0 16px 16px', display: 'flex', flexDirection: 'column', gap: 8, borderTop: '1px solid var(--border)' }}>
              <p style={{ ...SEC, marginTop: 12 }}>Geometric</p>
              <Slider label="Flip LR"    value={fliplr}    onChange={setFliplr}    min={0} max={1} step={0.05} />
              <Slider label="Flip UD"    value={flipud}    onChange={setFlipud}    min={0} max={1} step={0.05} />
              <Slider label="Rotation"   value={degrees}   onChange={setDegrees}   min={0} max={180} step={1} format={v => `${v}°`} />
              <Slider label="Translate"  value={translate} onChange={setTranslate} min={0} max={0.5} step={0.05} />
              <Slider label="Scale"      value={scale}     onChange={setScale}     min={0} max={0.5} step={0.05} />
              <p style={SEC}>Color</p>
              <Slider label="Brightness" value={brightness} onChange={setBrightness} min={0} max={0.8} step={0.05} />
              <Slider label="Contrast"   value={contrast}   onChange={setContrast}   min={0} max={0.8} step={0.05} />
              <Slider label="Saturation" value={saturation} onChange={setSaturation} min={0} max={0.8} step={0.05} />
              <p style={SEC}>Mixing & Cutout</p>
              <Slider label="Mixup"   value={mixup}   onChange={setMixup}   min={0} max={1} step={0.05} />
              <Slider label="Erasing" value={erasing} onChange={setErasing} min={0} max={0.9} step={0.05} />
            </div>
          )}
        </Card>

        <Btn variant="primary" onClick={startTraining} disabled={streaming}
          style={{ width: '100%', justifyContent: 'center', padding: '9px 14px' }}>
          {streaming
            ? <><Loader size={13} className="animate-spin" /> Training…</>
            : <><Zap size={13} strokeWidth={2.5} /> Start Training</>}
        </Btn>
        </div>

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
                    strokeWidth={2} dot={false} name="Val Accuracy" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          )}

          {/* Loss chart */}
          {chartData.length > 1 && (
            <Card style={{ padding: 16 }}>
              <Label>Training & Validation Loss</Label>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={chartData} margin={{ left: -16 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
                  <XAxis dataKey="epoch" stroke="var(--text3)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--text3)" tick={{ fontSize: 11 }}
                    tickFormatter={v => (v as number).toFixed(2)} />
                  <Tooltip {...TT} formatter={(v: unknown) =>
                    [(v as number).toFixed(4)]} />
                  <Line type="monotone" dataKey="trainLoss" stroke="#f97316"
                    strokeWidth={2} dot={false} name="Train Loss" strokeDasharray="4 2" />
                  <Line type="monotone" dataKey="valLoss" stroke="#22c55e"
                    strokeWidth={2} dot={false} name="Val Loss" />
                </LineChart>
              </ResponsiveContainer>
              <div style={{ display: 'flex', gap: 16, marginTop: 6 }}>
                <span style={{ fontSize: 11, color: '#f97316', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ display: 'inline-block', width: 14, height: 2, background: '#f97316', borderTop: '2px dashed #f97316' }} />
                  Train Loss
                </span>
                <span style={{ fontSize: 11, color: '#22c55e', display: 'flex', alignItems: 'center', gap: 4 }}>
                  <span style={{ display: 'inline-block', width: 14, height: 2, background: '#22c55e' }} />
                  Val Loss
                </span>
              </div>
            </Card>
          )}

          {/* Run comparison */}
          {runs.filter(r => r.status === 'done').length > 1 && (
            <Card style={{ padding: 16 }}>
              <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 14 }}>Run Comparison — Top-1 Accuracy</p>
              <ResponsiveContainer width="100%" height={110}>
                <BarChart data={runs.filter(r => r.status === 'done').map(r => {
                  const acc = (r.results as Record<string, unknown>)?.top1_acc
                  return { name: `#${r.id}`, acc: typeof acc === 'number' ? acc * 100 : 0 }
                })} margin={{ left: -16 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
                  <XAxis dataKey="name" stroke="var(--text3)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--text3)" tick={{ fontSize: 11 }} domain={[0, 100]} tickFormatter={v => `${v}%`} />
                  <Tooltip {...TT} formatter={(v: unknown) => [`${(v as number).toFixed(1)}%`, 'Top-1']} />
                  <Bar dataKey="acc" radius={[3,3,0,0]}>
                    {runs.filter(r => r.status === 'done').map((_, i) => (
                      <Cell key={i} fill={['#5865f2','#22c55e','#f59e0b','#06b6d4','#a855f7','#ec4899'][i % 6]} />
                    ))}
                  </Bar>
                </BarChart>
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
                        if (f) {
                          setInferFile(f); setInferName(f.name); setInferResult(null)
                          setInferPreview(URL.createObjectURL(f))
                        }
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

                {/* Preview + Result */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {inferPreview && (
                    <img src={inferPreview} alt="preview"
                      style={{ width: '100%', maxHeight: 120, objectFit: 'cover',
                        borderRadius: 6, border: '1px solid var(--border)' }} />
                  )}
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
                    const res        = run.results as Record<string, unknown>
                    const acc1       = typeof res?.top1_acc === 'number' ? res.top1_acc : null
                    const acc5       = typeof res?.top5_acc === 'number' ? res.top5_acc : null
                    const cm         = Array.isArray(res?.confusion_matrix)
                      ? res.confusion_matrix as number[][] : null
                    const perClass   = (res?.per_class && typeof res.per_class === 'object' && !Array.isArray(res.per_class))
                      ? res.per_class as Record<string, Record<string, number>> : null
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
                            {run.status === 'done' && run.model_path && (<>
                              <Btn variant="secondary" size="sm"
                                href={`/api/projects/${projectId}/classification/runs/${run.id}/download`}>
                                <Download size={12} /> .pth
                              </Btn>
                              <Btn variant="secondary" size="sm"
                                disabled={onnxExporting === run.id}
                                onClick={() => exportOnnx(run.id)}>
                                {onnxExporting === run.id
                                  ? <><Loader size={11} className="animate-spin" /> ONNX…</>
                                  : <><Download size={11} /> .onnx</>}
                              </Btn>
                            </>)}
                            {run.status === 'running' && (
                              <Btn variant="ghost" size="sm"
                                onClick={() => stopRun(run.id)}
                                style={{ color: 'var(--warn)' }}>
                                <Square size={12} /> Stop
                              </Btn>
                            )}
                            {run.status !== 'running' && (
                              <Btn variant="ghost" size="sm"
                                onClick={() => deleteRun(run.id)}
                                style={{ color: 'var(--error, #f87171)' }}>
                                <Trash2 size={12} />
                              </Btn>
                            )}
                          </div>
                        </div>

                        {/* Expanded details: per-class metrics + confusion matrix */}
                        {isCmExpanded && (
                          <div style={{ padding: '0 14px 14px', borderTop: '1px solid var(--border)' }}>

                            {/* Per-class Precision / Recall / F1 table */}
                            {perClass && (
                              <>
                                <p style={{ fontSize: 11, color: 'var(--text3)', margin: '10px 0 6px',
                                  textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 500 }}>
                                  Per-Class Metrics
                                </p>
                                <div style={{ overflowX: 'auto', marginBottom: 12 }}>
                                  <table style={{ borderCollapse: 'collapse', fontSize: 11, width: '100%' }}>
                                    <thead>
                                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                                        {['Class','Accuracy','Precision','Recall','F1','Support'].map(h => (
                                          <th key={h} style={{ padding: '3px 8px', color: 'var(--text3)',
                                            fontWeight: 500, textAlign: h === 'Class' ? 'left' : 'center' }}>
                                            {h}
                                          </th>
                                        ))}
                                      </tr>
                                    </thead>
                                    <tbody>
                                      {Object.entries(perClass).map(([cls, m]) => (
                                        <tr key={cls} style={{ borderBottom: '1px solid var(--border2,var(--border))' }}>
                                          <td style={{ padding: '4px 8px', color: 'var(--text2)', fontWeight: 500 }}>{cls}</td>
                                          {(['accuracy','precision','recall','f1'] as const).map(k => {
                                            const v = m[k] as number
                                            const good = v >= 0.7
                                            return (
                                              <td key={k} style={{ padding: '4px 8px', textAlign: 'center',
                                                fontFamily: 'JetBrains Mono, monospace',
                                                color: good ? 'var(--success,#22c55e)' : v < 0.4 ? 'var(--danger,#f87171)' : 'var(--warn,#f59e0b)' }}>
                                                {(v * 100).toFixed(1)}%
                                              </td>
                                            )
                                          })}
                                          <td style={{ padding: '4px 8px', textAlign: 'center',
                                            color: 'var(--text3)', fontFamily: 'JetBrains Mono, monospace' }}>
                                            {m.support}
                                          </td>
                                        </tr>
                                      ))}
                                    </tbody>
                                  </table>
                                </div>
                              </>
                            )}

                            {/* Confusion matrix */}
                            {cm && project && (
                              <>
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
                                          <th key={j} style={{ padding: '3px 8px', color: 'var(--text3)',
                                            fontWeight: 500, whiteSpace: 'nowrap' }}>
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
                              </>
                            )}
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
