/**
 * EstimateCard — pre-flight GPU VRAM / system RAM estimate for a training run.
 * Renders a Fits/Tight/Too-big verdict, a VRAM bar with a free-memory marker,
 * and a one-click "use batch N" when the config would OOM.
 */
interface Estimate {
  device: 'GPU' | 'CPU'
  est_vram_gb: number
  note?: string
  gpu: null | {
    name: string; total_gb: number; free_gb: number
    verdict: 'fits' | 'tight' | 'over'; suggested_batch: number | null
  }
  ram?: { est_gb?: number; available_gb?: number }
}

export default function EstimateCard(
  { estimate, onUseBatch }: { estimate: Estimate; onUseBatch?: (b: number) => void }
) {
  const g = estimate.gpu
  const v = g?.verdict
  const c = v === 'fits' ? '#22c55e' : v === 'tight' ? '#f59e0b' : v === 'over' ? '#ef4444' : 'var(--text3)'
  const pct = g ? Math.min(100, Math.round(estimate.est_vram_gb / g.total_gb * 100)) : 0
  const freePct = g ? Math.round(g.free_gb / g.total_gb * 100) : 0
  return (
    <div style={{ border: `1px solid ${c}55`, borderRadius: 8, padding: '10px 12px',
      background: 'var(--surface2)', display: 'flex', flexDirection: 'column', gap: 7 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 11, fontWeight: 600, color: 'var(--text2)' }}>
          Estimated load · {estimate.device}
        </span>
        {g && (
          <span style={{ fontSize: 10, fontWeight: 700, color: c, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            {v === 'fits' ? '✓ Fits' : v === 'tight' ? '⚠ Tight' : '✕ Too big'}
          </span>
        )}
      </div>
      {g ? (
        <>
          <div style={{ position: 'relative', height: 8, borderRadius: 4, background: 'var(--surface)', overflow: 'hidden' }}>
            <div style={{ position: 'absolute', left: `${freePct}%`, top: 0, bottom: 0, width: 2, background: 'var(--text3)' }} title="Free VRAM" />
            <div style={{ height: '100%', width: `${pct}%`, background: c, borderRadius: 4, transition: 'width 0.2s' }} />
          </div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text3)', fontFamily: 'JetBrains Mono, monospace' }}>
            <span>~{estimate.est_vram_gb} GB needed</span>
            <span>{g.free_gb} GB free / {g.total_gb} GB</span>
          </div>
          <div style={{ fontSize: 10, color: 'var(--text3)' }}>{g.name}</div>
          {v === 'over' && g.suggested_batch && onUseBatch && (
            <button onClick={() => onUseBatch(g.suggested_batch!)}
              style={{ fontSize: 10, padding: '4px 8px', borderRadius: 5, cursor: 'pointer',
                border: `1px solid ${c}`, background: `${c}22`, color: c, alignSelf: 'flex-start' }}>
              Use batch {g.suggested_batch} to fit
            </button>
          )}
        </>
      ) : (
        <span style={{ fontSize: 10, color: '#f59e0b' }}>{estimate.note}</span>
      )}
      <span style={{ fontSize: 9.5, color: 'var(--text3)', lineHeight: 1.4 }}>
        + ~{estimate.ram?.est_gb} GB system RAM
        {estimate.ram?.available_gb != null ? ` · ${estimate.ram.available_gb} GB free` : ''}
        {'  ·  estimate, not exact'}
      </span>
    </div>
  )
}
