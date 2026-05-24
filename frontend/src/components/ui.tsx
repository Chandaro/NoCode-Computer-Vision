import { type ReactNode, type CSSProperties } from 'react'

// ── Card ─────────────────────────────────────────────────────────────────────
export function Card({ children, style, className = '' }: {
  children: ReactNode; style?: CSSProperties; className?: string
}) {
  return (
    <div className={`card-hover ${className}`} style={{
      background: 'var(--surface)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius)',
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
      fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
      textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 10,
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
      borderRadius: 'var(--radius)', padding: '16px 20px',
    }}>
      <p style={{
        fontSize: 28, fontWeight: 700, color: 'var(--text)',
        letterSpacing: '-0.04em', lineHeight: 1, fontVariantNumeric: 'tabular-nums',
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
    fontFamily: 'inherit', fontWeight: 500,
    borderRadius: 'var(--radius-sm)',
    fontSize: size === 'sm' ? 12 : 13,
    padding: size === 'sm' ? '5px 12px' : '8px 16px',
    whiteSpace: 'nowrap', letterSpacing: '0.01em',
    lineHeight: 1.4,
  }
  const variants: Record<BtnVariant, CSSProperties> = {
    primary:   { background: 'var(--accent)', color: '#fff', borderColor: 'transparent' },
    secondary: { background: 'var(--surface2)', color: 'var(--text2)', borderColor: 'var(--border)' },
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
type BadgeColor = 'green' | 'red' | 'yellow' | 'gray' | 'blue'
export function Badge({ children, color = 'gray' }: { children: ReactNode; color?: BadgeColor }) {
  const colors: Record<BadgeColor, CSSProperties> = {
    green:  { background: 'var(--success-s)', color: 'var(--success)', border: '1px solid rgba(74,222,128,0.15)' },
    red:    { background: 'var(--danger-s)',  color: 'var(--danger)',  border: '1px solid rgba(248,113,113,0.15)' },
    yellow: { background: 'var(--warn-s)',    color: 'var(--warn)',    border: '1px solid rgba(251,191,36,0.15)' },
    gray:   { background: 'var(--surface2)',  color: 'var(--text2)',   border: '1px solid var(--border)' },
    blue:   { background: 'var(--accent-s)',  color: '#93b4ff',        border: '1px solid rgba(37,99,235,0.2)' },
  }
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 7px',
      borderRadius: 'var(--radius-sm)',
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
      borderRadius: 'var(--radius-sm)', color: 'var(--text)',
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
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
        <span style={{ fontSize: 13, color: 'var(--text2)', fontWeight: 500 }}>{label}</span>
        <span style={{
          fontSize: 12, color: 'var(--accent)', fontWeight: 600,
          fontFamily: 'JetBrains Mono, monospace',
        }}>
          {display}
        </span>
      </div>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', cursor: 'pointer', accentColor: 'var(--accent)' } as CSSProperties}
      />
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
        borderRadius: 'var(--radius-sm)', color: 'var(--text)',
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
      display: 'flex', alignItems: 'center', gap: 14,
      marginBottom: 24,
      paddingBottom: 16,
      borderBottom: '1px solid var(--border)',
    }}>
      {back && (
        <button
          onClick={back}
          className="btn-ghost"
          style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            width: 28, height: 28,
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
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
        <h1 style={{
          fontSize: 17, fontWeight: 700, color: 'var(--text)',
          letterSpacing: '-0.02em', lineHeight: 1.2,
        }}>
          {title}
        </h1>
        {subtitle && (
          <p style={{ fontSize: 12, color: 'var(--text3)', marginTop: 2, fontWeight: 400 }}>{subtitle}</p>
        )}
      </div>
      {actions && (
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexShrink: 0 }}>
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
        width: 44, height: 44,
        background: 'var(--surface2)', border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
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
export function LogTerminal({ logs, logRef }: {
  logs: string[]; logRef: RefObject<HTMLDivElement | null>
}) {
  return (
    <div
      ref={logRef}
      style={{
        height: 200, overflowY: 'auto', flexShrink: 0,
        background: '#080808',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '12px 14px',
        fontFamily: 'JetBrains Mono, monospace',
        fontSize: 11.5, lineHeight: 1.8,
      }}
    >
      {logs.length === 0
        ? <span style={{ color: 'var(--text3)' }}>Waiting for training to start…</span>
        : logs.map((l, i) => {
          const color =
            l.startsWith('[ERROR]') || l.toLowerCase().includes('error') ? 'var(--danger)' :
            l.startsWith('[DONE]')                                         ? 'var(--success)' :
            l.startsWith('[WARN]') || l.startsWith('[STOPPED]')           ? 'var(--warn)' :
            'var(--text3)'
          return (
            <div key={i} style={{ color, display: 'flex', gap: 8 }}>
              <span style={{ color: '#2a2a2a', userSelect: 'none', flexShrink: 0 }}>
                {String(i + 1).padStart(3, ' ')}
              </span>
              <span>{l}</span>
            </div>
          )
        })}
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
          <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'JetBrains Mono, monospace' }}>
            {sublabel ?? `${pct}%`}
          </span>
        </div>
      )}
      <div style={{ height: 3, background: 'var(--surface3)', borderRadius: 0, overflow: 'hidden' }}>
        <div style={{
          height: '100%', width: `${pct}%`,
          background: 'var(--accent)',
          transition: 'width 0.2s',
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
      borderRadius: 'var(--radius)', padding: '14px 18px', textAlign: 'center',
    }}>
      <p style={{
        fontSize: 10, color: 'var(--text3)', marginBottom: 6,
        letterSpacing: '0.08em', textTransform: 'uppercase', fontWeight: 600,
      }}>{label}</p>
      <p style={{
        fontSize: 22, fontWeight: 700, color,
        fontFamily: 'JetBrains Mono, monospace', letterSpacing: '-0.03em',
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
        fontSize: 10, fontWeight: 700, letterSpacing: '0.08em',
        textTransform: 'uppercase', color: 'var(--text3)', whiteSpace: 'nowrap',
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
          width: 30, height: 16, borderRadius: 0,
          background: checked ? 'var(--accent)' : 'var(--surface3)',
          border: `1px solid ${checked ? 'var(--accent)' : 'var(--border2)'}`,
          position: 'relative', flexShrink: 0,
          transition: 'background 0.1s, border-color 0.1s', cursor: 'pointer',
        }}
      >
        <div style={{
          position: 'absolute', top: 2, left: checked ? 13 : 2,
          width: 10, height: 10, borderRadius: 0, background: '#fff',
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
      border: '1px solid rgba(37,99,235,0.2)',
      borderRadius: 'var(--radius)',
      marginBottom: 20,
      fontSize: 13, color: 'var(--text2)', lineHeight: 1.6,
    }}>
      {children}
    </div>
  )
}
