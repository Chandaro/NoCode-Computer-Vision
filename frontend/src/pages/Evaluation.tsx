import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Loader, Image as ImageIcon, Upload, Zap } from 'lucide-react'
import api, { type EvalData, type InferResult } from '../api'
import { Card, Label, PageHeader, MetricPill, Empty, Btn, Slider } from '../components/ui'

const PALETTE = ['#5865f2','#22c55e','#f59e0b','#06b6d4','#a855f7','#ec4899','#64748b','#ef4444']

const PLOT_LABELS: Record<string, string> = {
  'confusion_matrix.png':            'Confusion Matrix',
  'confusion_matrix_normalized.png': 'Confusion Matrix (Norm.)',
  'results.png':                     'Training Results',
  'BoxF1_curve.png':                 'F1 Curve',
  'BoxPR_curve.png':                 'PR Curve',
  'BoxP_curve.png':                  'Precision Curve',
  'BoxR_curve.png':                  'Recall Curve',
  'labels.jpg':                      'Label Distribution',
  'labels_correlogram.jpg':          'Label Correlogram',
}

function isValBatch(name: string) { return name.startsWith('val_batch') }

export default function Evaluation() {
  const { id, runId } = useParams<{ id: string; runId: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [data, setData]         = useState<EvalData | null>(null)
  const [loading, setLoading]   = useState(true)
  const [expanded, setExpanded] = useState<string | null>(null)

  // ── Inference state ──────────────────────────────────────────────────────
  const [inferFile,    setInferFile]    = useState<File | null>(null)
  const [inferName,    setInferName]    = useState('')
  const [inferConf,    setInferConf]    = useState(0.25)
  const [inferIou,     setInferIou]     = useState(0.45)
  const [inferRunning, setInferRunning] = useState(false)
  const [inferResult,  setInferResult]  = useState<InferResult | null>(null)
  const inferFileRef   = useRef<HTMLInputElement>(null)
  const inferCanvasRef = useRef<HTMLCanvasElement>(null)
  const inferImgRef    = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    api.get(`/projects/${projectId}/training/runs/${runId}/eval`)
      .then(r => setData(r.data))
      .finally(() => setLoading(false))
  }, [projectId, runId])

  const handleInferFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    setInferFile(f)
    setInferName(f.name)
    setInferResult(null)

    const url = URL.createObjectURL(f)
    const img = new Image()
    img.onload = () => {
      inferImgRef.current = img
      const canvas = inferCanvasRef.current
      if (!canvas) return
      canvas.width  = img.width
      canvas.height = img.height
      const ctx = canvas.getContext('2d')!
      ctx.drawImage(img, 0, 0)
    }
    img.src = url
  }

  const drawDetections = (result: InferResult) => {
    const canvas = inferCanvasRef.current
    const img    = inferImgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')!
    ctx.drawImage(img, 0, 0)
    result.detections.forEach((d) => {
      const color = PALETTE[d.class_id % PALETTE.length]
      const x = d.x * canvas.width,  y = d.y * canvas.height
      const w = d.w * canvas.width,  h = d.h * canvas.height
      ctx.strokeStyle = color; ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)
      ctx.fillStyle = color + 'cc'
      ctx.fillRect(x, y - 18, Math.min(w, ctx.measureText(`${d.class_name} ${(d.conf*100).toFixed(0)}%`).width + 8), 18)
      ctx.fillStyle = '#fff'; ctx.font = '12px Inter, sans-serif'
      ctx.fillText(`${d.class_name} ${(d.conf*100).toFixed(0)}%`, x + 4, y - 5)
    })
  }

  const runInfer = async () => {
    if (!inferFile) return
    setInferRunning(true)
    try {
      const fd = new FormData()
      fd.append('file', inferFile)
      fd.append('conf', String(inferConf))
      fd.append('iou',  String(inferIou))
      const res = await api.post(
        `/projects/${projectId}/training/runs/${runId}/infer`, fd,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      )
      setInferResult(res.data)
      setTimeout(() => drawDetections(res.data), 50)
    } finally {
      setInferRunning(false)
    }
  }

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
      height: 260, gap: 8, color: 'var(--text2)' }}>
      <Loader size={16} className="animate-spin" /> Loading evaluation…
    </div>
  )
  if (!data) return (
    <p style={{ textAlign: 'center', marginTop: 80, color: 'var(--danger)', fontSize: 14 }}>
      Failed to load evaluation data.
    </p>
  )

  const mainPlots  = data.available_plots.filter(p => !isValBatch(p))
  const valBatches = data.available_plots.filter(isValBatch)
  const plotUrl    = (f: string) =>
    `/api/projects/${projectId}/training/runs/${runId}/eval/plots/${f}`

  const metricColor = (k: string) => {
    if (k.toLowerCase().includes('map'))  return 'var(--accent)'
    if (k.toLowerCase().includes('prec')) return 'var(--success)'
    if (k.toLowerCase().includes('rec'))  return '#f59e0b'
    return 'var(--text)'
  }

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader
        back={() => navigate(`/projects/${projectId}/train`)}
        title="Model Evaluation"
        subtitle={`Run #${runId} · ${data.model_base} · ${data.epochs} epochs`}
      />

      {/* Overview metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 20 }}>
        {Object.entries(data.overall).map(([k, v]) => (
          <MetricPill key={k} label={k} value={`${(v * 100).toFixed(1)}%`} color={metricColor(k)} />
        ))}
      </div>

      {/* ── Inference ── */}
      <Card style={{ padding: 16, marginBottom: 12 }}>
        <Label>Test Inference</Label>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginTop: 8 }}>

          {/* Controls */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <Btn variant="secondary" size="sm" onClick={() => inferFileRef.current?.click()}>
                <Upload size={12} /> Upload Image
              </Btn>
              <input ref={inferFileRef} type="file" accept="image/*"
                style={{ display: 'none' }} onChange={handleInferFile} />
              <span style={{ fontSize: 12, color: 'var(--text3)',
                overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {inferName || 'No file selected'}
              </span>
            </div>
            <Slider label="Confidence Threshold" value={inferConf} onChange={setInferConf}
              min={0.05} max={0.95} format={v => `${Math.round(v * 100)}%`} />
            <Slider label="IoU Threshold" value={inferIou} onChange={setInferIou}
              min={0.1} max={0.9} format={v => `${Math.round(v * 100)}%`} />
            <Btn variant="primary" size="sm" onClick={runInfer}
              disabled={!inferFile || inferRunning}>
              {inferRunning
                ? <><Loader size={12} className="animate-spin" /> Running…</>
                : <><Zap size={12} strokeWidth={2.5} /> Run Inference</>}
            </Btn>

            {/* Detections table */}
            {inferResult && (
              <div>
                <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 6 }}>
                  {inferResult.count} detection{inferResult.count !== 1 ? 's' : ''}
                </p>
                {inferResult.detections.length > 0 && (
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11 }}>
                    <thead>
                      <tr>
                        {['Class','Conf','X','Y','W','H'].map(h => (
                          <th key={h} style={{ color: 'var(--text3)', fontWeight: 500,
                            textAlign: h === 'Class' ? 'left' : 'right', paddingBottom: 6,
                            letterSpacing: '0.04em', textTransform: 'uppercase', fontSize: 10 }}>
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {inferResult.detections.map((d, i) => (
                        <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                          <td style={{ padding: '5px 0', display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ width: 8, height: 8, borderRadius: 2, flexShrink: 0,
                              background: PALETTE[d.class_id % PALETTE.length] }} />
                            {d.class_name}
                          </td>
                          <td style={{ padding: '5px 0', textAlign: 'right',
                            fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent)' }}>
                            {(d.conf * 100).toFixed(1)}%
                          </td>
                          {[d.x, d.y, d.w, d.h].map((v, j) => (
                            <td key={j} style={{ padding: '5px 0', textAlign: 'right',
                              fontFamily: 'JetBrains Mono, monospace', color: 'var(--text2)' }}>
                              {v.toFixed(3)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </div>

          {/* Canvas preview */}
          <div style={{ border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden',
            background: '#07070a', display: 'flex', alignItems: 'center', justifyContent: 'center',
            minHeight: 180 }}>
            {inferFile
              ? <canvas ref={inferCanvasRef}
                  style={{ maxWidth: '100%', maxHeight: 320, display: 'block', objectFit: 'contain' }} />
              : <p style={{ fontSize: 12, color: 'var(--text3)' }}>Upload an image to preview</p>}
          </div>
        </div>
      </Card>

      {/* Per-class metrics */}
      {Object.keys(data.per_class).length > 0 && (
        <Card style={{ padding: 16, marginBottom: 12 }}>
          <Label>Per-Class Metrics</Label>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                {['Class', 'AP50', 'Precision', 'Recall'].map((h, i) => (
                  <th key={h} style={{
                    fontSize: 11, fontWeight: 500, color: 'var(--text3)',
                    letterSpacing: '0.05em', textTransform: 'uppercase',
                    textAlign: i === 0 ? 'left' : 'right', paddingBottom: 10,
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.per_class).map(([cls, m]) => (
                <tr key={cls} style={{ borderTop: '1px solid var(--border)' }}>
                  <td style={{ padding: '9px 0', fontSize: 13, color: 'var(--text)', fontWeight: 500 }}>{cls}</td>
                  <td style={{ padding: '9px 0', textAlign: 'right', fontSize: 13,
                    fontFamily: 'JetBrains Mono, monospace', color: 'var(--accent)' }}>
                    {(m.ap50 * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '9px 0', textAlign: 'right', fontSize: 13,
                    fontFamily: 'JetBrains Mono, monospace', color: 'var(--success)' }}>
                    {(m.precision * 100).toFixed(1)}%
                  </td>
                  <td style={{ padding: '9px 0', textAlign: 'right', fontSize: 13,
                    fontFamily: 'JetBrains Mono, monospace', color: '#f59e0b' }}>
                    {(m.recall * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}

      {/* Training plots */}
      {mainPlots.length > 0 && (
        <Card style={{ padding: 16, marginBottom: 12 }}>
          <Label>Training Plots</Label>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 10, marginTop: 4 }}>
            {mainPlots.map(plot => (
              <div key={plot} style={{ cursor: 'pointer' }} onClick={() => setExpanded(plot)}>
                <div style={{ border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden',
                  transition: 'border-color 0.15s' }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}>
                  <img src={plotUrl(plot)} alt={plot}
                    style={{ width: '100%', height: 148, objectFit: 'contain',
                      background: '#fff', display: 'block', padding: 4 }}
                    onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
                </div>
                <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 5, textAlign: 'center',
                  overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                  {PLOT_LABELS[plot] ?? plot}
                </p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Validation batches */}
      {valBatches.length > 0 && (
        <Card style={{ padding: 16, marginBottom: 12 }}>
          <Label>Validation Predictions</Label>
          <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 10 }}>
            Left = ground truth · Right = model predictions
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {valBatches.map(f => (
              <div key={f} style={{ cursor: 'pointer' }} onClick={() => setExpanded(f)}>
                <div style={{ border: '1px solid var(--border)', borderRadius: 6, overflow: 'hidden',
                  transition: 'border-color 0.15s' }}
                  onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--accent)')}
                  onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}>
                  <img src={plotUrl(f)} alt={f}
                    style={{ width: '100%', display: 'block', background: '#fff' }}
                    onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
                </div>
                <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 5, textAlign: 'center' }}>{f}</p>
              </div>
            ))}
          </div>
        </Card>
      )}

      {mainPlots.length === 0 && valBatches.length === 0 && (
        <div style={{ border: '1px dashed var(--border2)', borderRadius: 10 }}>
          <Empty icon={ImageIcon} message="No evaluation plots found"
            sub="Plots are generated during training and stored in the run directory." />
        </div>
      )}

      {/* Lightbox */}
      {expanded && (
        <div onClick={() => setExpanded(null)}
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.88)', zIndex: 200,
            display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 24,
            backdropFilter: 'blur(6px)' }}>
          <img src={plotUrl(expanded)} alt={expanded}
            style={{ maxWidth: '100%', maxHeight: '100%', borderRadius: 8,
              boxShadow: '0 24px 64px rgba(0,0,0,0.7)', background: '#fff', padding: 8 }} />
        </div>
      )}
    </div>
  )
}
