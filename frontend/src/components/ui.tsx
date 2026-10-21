import { type ReactNode, type CSSProperties } from 'react'

// ── Card ─────────────────────────────────────────────────────────────────────
export function Card({ children, style, className = '' }: {
  children: ReactNode; style?: CSSProperties; className?: string
}) {
  return (
    <div className={`card-hover ${className}`} style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      ...style,
    }}>
      {children}
    </div>
  )
}

// ── Section label ─────────────────────────────────────────────────────────────
export function Label({ children }: { children: ReactNode }) {
  return (
    <p style={{
      fontSize: 12, fontWeight: 600, letterSpacing: '0.01em',
      color: 'var(--text2)', marginBottom: 10,
    }}>
      {children}
    </p>
  )
}

// ── Divider ───────────────────────────────────────────────────────────────────
export function Divider() {
  return <div style={{ height: 1, background: 'var(--border)', margin: '4px 0' }} />
}

// ── Stat tile ─────────────────────────────────────────────────────────────────
export function StatTile({ label, value, sub }: {
  label: string; value: string | number; sub?: string
}) {
  return (
    <div style={{
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: '16px 20px',
    }}>
      <p className="font-display" style={{
        fontSize: 34, color: 'var(--text)',
        lineHeight: 1, fontVariantNumeric: 'tabular-nums',
      }}>{value}</p>
      <p style={{ fontSize: 12, color: 'var(--text3)', marginTop: 6, fontWeight: 500 }}>{label}</p>
      {sub && <p style={{ fontSize: 11, color: 'var(--accent)', marginTop: 4, fontWeight: 500 }}>{sub}</p>}
    </div>
  )
}

// ── Button ────────────────────────────────────────────────────────────────────
type BtnVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export function Btn({
  children, onClick, disabled, variant = 'secondary', size = 'md',
  type = 'button', href, style,
}: {
  children: ReactNode; onClick?: () => void; disabled?: boolean
  variant?: BtnVariant; size?: 'sm' | 'md'; type?: 'button' | 'submit'
  href?: string; style?: CSSProperties
}) {
  const base: CSSProperties = {
    display: 'inline-flex', alignItems: 'center', gap: 6,
    cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.35 : 1,
    border: '1px solid transparent',
    textDecoration: 'none',
    fontFamily: 'inherit', fontWeight: 600,
    borderRadius: 999,
    fontSize: size === 'sm' ? 12 : 13,
    padding: size === 'sm' ? '5px 13px' : '8px 18px',
    whiteSpace: 'nowrap', letterSpacing: '0.01em',
    lineHeight: 1.4,
  }
  const variants: Record<BtnVariant, CSSProperties> = {
    primary:   { background: 'var(--accent)', color: 'var(--on-accent)', borderColor: 'transparent' },
    secondary: { background: 'var(--secondary-t)', color: 'var(--secondary)', borderColor: 'rgba(60,196,180,0.35)' },
    ghost:     { background: 'transparent', color: 'var(--text2)', borderColor: 'transparent' },
    danger:    { background: 'transparent', color: 'var(--danger)', borderColor: 'var(--danger)' },
  }
  const cls = `btn-${variant}`
  const combined = { ...base, ...variants[variant], ...style }
  if (href) return <a href={href} className={cls} style={combined}>{children}</a>
  return (
    <button type={type} className={cls} onClick={onClick} disabled={disabled} style={combined}>
      {children}
    </button>
  )
}

// ── Badge ─────────────────────────────────────────────────────────────────────
type BadgeColor = 'green' | 'red' | 'yellow' | 'gray' | 'blue' | 'purple'
export function Badge({ children, color = 'gray' }: { children: ReactNode; color?: BadgeColor }) {
  const colors: Record<BadgeColor, CSSProperties> = {
    green:  { background: 'var(--success-s)',   color: 'var(--success)', border: '1px solid rgba(106,203,142,0.22)' },
    red:    { background: 'var(--danger-s)',     color: 'var(--danger)',  border: '1px solid rgba(240,106,94,0.22)' },
    yellow: { background: 'var(--accent-s)',     color: 'var(--accent)',  border: '1px solid rgba(228,180,63,0.25)' },
    gray:   { background: 'var(--surface2)',     color: 'var(--text2)',   border: '1px solid var(--border)' },
    blue:   { background: 'var(--secondary-s)', color: 'var(--secondary)', border: '1px solid rgba(60,196,180,0.25)' },
    purple: { background: 'rgba(88,102,232,0.14)', color: '#9aa4ff',      border: '1px solid rgba(88,102,232,0.32)' },
  }
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 8px',
      borderRadius: 999,
      fontSize: 11, fontWeight: 600, lineHeight: '18px',
      ...colors[color],
    }}>
      {children}
    </span>
  )
}

// ── Select ────────────────────────────────────────────────────────────────────
export function Select({ value, onChange, children, style }: {
  value: string | number; onChange: (v: string) => void
  children: ReactNode; style?: CSSProperties
}) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} style={{
      width: '100%', padding: '7px 12px',
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius)', color: 'var(--text)',
      fontSize: 13, fontFamily: 'inherit', cursor: 'pointer', outline: 'none',
      ...style,
    }}>
      {children}
    </select>
  )
}

// ── Slider ────────────────────────────────────────────────────────────────────
export function Slider({ label, value, onChange, min, max, step = 0.05, format }: {
  label: string; value: number; onChange: (v: number) => void
  min: number; max: number; step?: number; format?: (v: number) => string
}) {
  const display = format ? format(value) : value
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <span style={{ fontSize: 13, color: 'var(--text2)', fontWeight: 500 }}>{label}</span>
        <span style={{ fontSize: 12, color: 'var(--accent)', fontWeight: 600, fontFamily: 'var(--font-mono)' }}>
          {display}
        </span>
      </div>
      <div style={{ position: 'relative', height: 28, display: 'flex', alignItems: 'center' }}>
        {/* Track */}
        <div style={{ position: 'absolute', left: 0, right: 0, height: 4, borderRadius: 4, background: 'var(--surface3)' }} />
        {/* Fill */}
        <div style={{ position: 'absolute', left: 0, width: `${pct}%`, height: 4, borderRadius: 4,
          background: 'linear-gradient(90deg, var(--secondary), var(--accent))' }} />
        {/* Native input — invisible, handles all interaction */}
        <input
          type="range" min={min} max={max} step={step} value={value}
          onChange={e => onChange(Number(e.target.value))}
          style={{ position: 'absolute', width: '100%', opacity: 0, cursor: 'grab', height: '100%', margin: 0, padding: 0, zIndex: 2 } as CSSProperties}
        />
        {/* Eye thumb: gold ring, iris, pupil */}
        <div style={{
          position: 'absolute',
          left: `calc(${pct}% - 9px)`,
          width: 18, height: 18, borderRadius: '50%',
          pointerEvents: 'none', zIndex: 1,
          background: 'radial-gradient(circle, #05090b 0 2.5px, var(--iris) 3px 5px, var(--accent) 5.5px 8px, var(--bg) 8.5px)',
          boxShadow: '0 0 0 1px var(--accent), 0 2px 8px rgba(0,0,0,0.5)',
        }} />
      </div>
    </div>
  )
}

// ── Field ─────────────────────────────────────────────────────────────────────
export function Field({ label, children, hint }: {
  label: ReactNode; children: ReactNode; hint?: string
}) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <label style={{ fontSize: 12, color: 'var(--text2)', fontWeight: 500, letterSpacing: '0.01em' }}>
        {label}
        {hint && <span style={{ fontSize: 11, color: 'var(--text3)', marginLeft: 6, fontWeight: 400 }}>{hint}</span>}
      </label>
      {children}
    </div>
  )
}

// ── TextInput ─────────────────────────────────────────────────────────────────
export function TextInput({ value, onChange, placeholder, onKeyDown, type = 'text' }: {
  value: string; onChange: (v: string) => void
  placeholder?: string; onKeyDown?: (e: React.KeyboardEvent) => void
  type?: string
}) {
  return (
    <input
      type={type}
      value={value} onChange={e => onChange(e.target.value)}
      placeholder={placeholder} onKeyDown={onKeyDown}
      style={{
        width: '100%', padding: '7px 12px',
        background: 'var(--surface2)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius)', color: 'var(--text)',
        fontSize: 13, fontFamily: 'inherit', outline: 'none',
      }}
    />
  )
}

// ── PageHeader ────────────────────────────────────────────────────────────────
export function PageHeader({ back, title, subtitle, actions }: {
  back?: () => void; title: string; subtitle?: string; actions?: ReactNode
}) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap',
      marginBottom: 24,
      paddingBottom: 18,
      borderBottom: '1px solid var(--border)',
    }}>
      {back && (
        <button
          onClick={back}
          className="btn-ghost"
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: 34, height: 34,
            border: '1px solid var(--border2)',
            borderRadius: '50%',
            background: 'var(--surface2)',
            color: 'var(--text2)', cursor: 'pointer', flexShrink: 0,
            fontSize: 14,
          }}
          aria-label="Go back"
        >
          ←
        </button>
      )}
      <div style={{ flex: 1, minWidth: 0 }}>
        <h1 className="font-display" style={{
          fontSize: 28, color: 'var(--text)', lineHeight: 1.1,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
        }}>
          {title}
        </h1>
        {subtitle && (
          <p style={{ fontSize: 13, color: 'var(--text2)', marginTop: 4, fontWeight: 400 }}>{subtitle}</p>
        )}
      </div>
      {actions && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0, flexWrap: 'wrap' }}>
          {actions}
        </div>
      )}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────
export function Empty({ icon: Icon, message, sub }: {
  icon: React.ElementType; message: string; sub?: string
}) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', alignItems: 'center',
      justifyContent: 'center', padding: '56px 24px', gap: 14,
    }}>
      <div style={{
        width: 52, height: 52,
        background: 'var(--surface2)', border: '1px solid var(--border2)',
        borderRadius: '50%',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <Icon size={20} style={{ color: 'var(--text3)' }} strokeWidth={1.5} />
      </div>
      <div style={{ textAlign: 'center', maxWidth: 280 }}>
        <p style={{ color: 'var(--text2)', fontSize: 14, fontWeight: 600 }}>{message}</p>
        {sub && <p style={{ color: 'var(--text3)', fontSize: 13, marginTop: 6, lineHeight: 1.6 }}>{sub}</p>}
      </div>
    </div>
  )
}

// ── Log terminal ──────────────────────────────────────────────────────────────
import { type RefObject } from 'react'

function termColor(l: string): string {
  const lo = l.toLowerCase()
  if (l.startsWith('[ERROR]') || lo.includes('error') || lo.includes('traceback')) return '#f06a5e'
  if (l.startsWith('[DONE]')  || lo.includes('finished') || lo.includes('complete')) return '#6acb8e'
  if (l.startsWith('[WARN]')  || l.startsWith('[STOPPED]') || lo.includes('warning')) return '#f39a4a'
  if (lo.includes('epoch'))  return '#3cc4b4'
  if (lo.includes('loss') || lo.includes('map') || lo.includes('precision') || lo.includes('recall')) return '#e4b43f'
  if (l.startsWith('[INFO]')) return '#3cc4b4'
  return '#c9d6d1'
}

export function LogTerminal({ logs, logRef }: {
  logs: string[]; logRef: RefObject<HTMLDivElement | null>
}) {
  return (
    <div style={{
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)',
      overflow: 'hidden',
      flexShrink: 0,
    }}>
      {/* ── Title bar ── */}
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '9px 14px',
        background: 'var(--surface2)',
        borderBottom: '1px solid var(--border)',
        userSelect: 'none',
      }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: logs.length ? 'var(--success)' : 'var(--text3)',
          boxShadow: logs.length ? '0 0 0 3px rgba(106,203,142,0.18)' : 'none' }} />
        <span style={{ fontSize: 12, color: 'var(--text2)', fontFamily: 'var(--font-mono)' }}>
          training.log
        </span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
          {logs.length} lines
        </span>
      </div>

      {/* ── Terminal body ── */}
      <div
        ref={logRef}
        style={{
          height: 220, overflowY: 'auto',
          background: '#040d0e',
          padding: '12px 16px',
          fontFamily: 'var(--font-mono)',
          fontSize: 11.5, lineHeight: 2,
        }}
      >
        {logs.length === 0 ? (
          <span style={{ color: 'var(--text3)' }}>
            <span style={{ color: 'var(--secondary)' }}>nocode-cv</span>
            <span style={{ color: 'var(--accent)' }}> $ </span>
            <span>waiting for training to start</span>
            <span style={{ display: 'inline-block', width: 7, height: '1.1em', background: 'var(--accent)', marginLeft: 3, verticalAlign: 'text-bottom', opacity: 0.7 }} />
          </span>
        ) : logs.map((l, i) => (
          <div key={i} style={{ display: 'flex', gap: 0 }}>
            <span style={{
              color: '#2a4448', userSelect: 'none', flexShrink: 0, minWidth: 34,
              fontVariantNumeric: 'tabular-nums', textAlign: 'right', paddingRight: 14,
            }}>
              {i + 1}
            </span>
            <span style={{ color: '#1a3236', flexShrink: 0, marginRight: 14 }}>│</span>
            <span style={{ color: termColor(l), wordBreak: 'break-all' }}>{l}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Progress bar ──────────────────────────────────────────────────────────────
export function ProgressBar({ value, max, label, sublabel }: {
  value: number; max: number; label?: string; sublabel?: string
}) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div>
      {(label || sublabel) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 7 }}>
          <span style={{ fontSize: 12, color: 'var(--text2)', fontWeight: 500 }}>{label}</span>
          <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'var(--font-mono)' }}>
            {sublabel ?? `${pct}%`}
          </span>
        </div>
      )}
      <div style={{ height: 5, background: 'var(--surface3)', borderRadius: 5, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`, borderRadius: 5,
          background: 'linear-gradient(90deg, var(--secondary), var(--accent))',
          transition: 'width 0.3s',
        }} />
      </div>
    </div>
  )
}

// ── MetricPill ────────────────────────────────────────────────────────────────
export function MetricPill({ label, value, color = 'var(--accent)' }: {
  label: string; value: string; color?: string
}) {
  return (
    <div style={{
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-lg)', padding: '14px 18px', textAlign: 'center',
    }}>
      <p style={{
        fontSize: 12, color: 'var(--text2)', marginBottom: 6, fontWeight: 600,
      }}>{label}</p>
      <p style={{
        fontSize: 22, fontWeight: 600, color,
        fontFamily: 'var(--font-mono)', letterSpacing: '-0.02em',
        fontVariantNumeric: 'tabular-nums',
      }}>{value}</p>
    </div>
  )
}

// ── SectionHeader ─────────────────────────────────────────────────────────────
export function SectionHeader({ children }: { children: ReactNode }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, margin: '4px 0' }}>
      <span style={{
        fontSize: 12, fontWeight: 600, color: 'var(--text2)', whiteSpace: 'nowrap',
      }}>{children}</span>
      <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
    </div>
  )
}

// ── Toggle ────────────────────────────────────────────────────────────────────
export function Toggle({ checked, onChange, label }: {
  checked: boolean; onChange: (v: boolean) => void; label: string
}) {
  return (
    <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', userSelect: 'none' }}>
      <div
        onClick={() => onChange(!checked)}
        style={{
          width: 32, height: 18, borderRadius: 18,
          background: checked ? 'var(--accent)' : 'var(--surface3)',
          border: `1px solid ${checked ? 'var(--accent)' : 'var(--border2)'}`,
          position: 'relative', flexShrink: 0,
          transition: 'background 0.1s, border-color 0.1s', cursor: 'pointer',
        }}
      >
        <div style={{
          position: 'absolute', top: 2, left: checked ? 15 : 2,
          width: 12, height: 12, borderRadius: '50%', background: checked ? 'var(--on-accent)' : 'var(--text2)',
          transition: 'left 0.1s',
        }} />
      </div>
      <span style={{ fontSize: 13, color: 'var(--text2)', fontWeight: 500 }}>{label}</span>
    </label>
  )
}

// ── InfoBanner ────────────────────────────────────────────────────────────────
export function InfoBanner({ children }: { children: ReactNode }) {
  return (
    <div style={{
      padding: '10px 14px',
      background: 'var(--accent-t)',
      border: '1px solid rgba(228,180,63,0.22)',
      borderLeft: '3px solid var(--accent)',
      borderRadius: 'var(--radius)',
      marginBottom: 20,
      fontSize: 13, color: 'var(--text2)', lineHeight: 1.6,
    }}>
      {children}
    </div>
  )
}
