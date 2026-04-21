import { type ReactNode, type CSSProperties } from 'react'

// ── Card ─────────────────────────────────────────────────────────────────────
export function Card({ children, style, className = '' }: { children: ReactNode; style?: CSSProperties; className?: string }) {
  return (
    <div className={className} style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 8, ...style,
    }}>
      {children}
    </div>
  )
}

// ── Section label (12px uppercase) ───────────────────────────────────────────
export function Label({ children }: { children: ReactNode }) {
  return (
    <p style={{ fontSize: 11, fontWeight: 500, letterSpacing: '0.07em', textTransform: 'uppercase', color: 'var(--text3)', marginBottom: 6 }}>
      {children}
    </p>
  )
}

// ── Divider ───────────────────────────────────────────────────────────────────
export function Divider() {
  return <div style={{ height: 1, background: 'var(--border)', margin: '4px 0' }} />
}

// ── Stat tile ─────────────────────────────────────────────────────────────────
export function StatTile({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 8, padding: '14px 16px' }}>
      <p style={{ fontSize: 22, fontWeight: 600, color: 'var(--text)', letterSpacing: '-0.02em', lineHeight: 1 }}>{value}</p>
      <p style={{ fontSize: 12, color: 'var(--text2)', marginTop: 4 }}>{label}</p>
      {sub && <p style={{ fontSize: 11, color: 'var(--accent)', marginTop: 2 }}>{sub}</p>}
    </div>
  )
}

// ── Button ────────────────────────────────────────────────────────────────────
type BtnVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
export function Btn({
  children, onClick, disabled, variant = 'secondary', size = 'md', type = 'button', href, style,
}: {
  children: ReactNode; onClick?: () => void; disabled?: boolean
  variant?: BtnVariant; size?: 'sm' | 'md'; type?: 'button' | 'submit'
  href?: string; style?: CSSProperties
}) {
  const base: CSSProperties = {
    display: 'inline-flex', alignItems: 'center', gap: 6, cursor: disabled ? 'not-allowed' : 'pointer',
    opacity: disabled ? 0.45 : 1, border: 'none', textDecoration: 'none',
    fontFamily: 'inherit', fontWeight: 500, borderRadius: 6,
    fontSize: size === 'sm' ? 12 : 13,
    padding: size === 'sm' ? '4px 10px' : '7px 14px',
    transition: 'background 0.12s, color 0.12s, border-color 0.12s',
    whiteSpace: 'nowrap',
  }
  const variants: Record<BtnVariant, CSSProperties> = {
    primary:   { background: 'var(--accent)', color: '#fff', border: '1px solid var(--accent)' },
    secondary: { background: 'transparent', color: 'var(--text2)', border: '1px solid var(--border2)' },
    ghost:     { background: 'transparent', color: 'var(--text2)', border: '1px solid transparent' },
    danger:    { background: 'transparent', color: 'var(--danger)', border: '1px solid var(--danger)' },
  }
  const combined = { ...base, ...variants[variant], ...style }
  if (href) return <a href={href} style={combined}>{children}</a>
  return <button type={type} onClick={onClick} disabled={disabled} style={combined}>{children}</button>
}

// ── Badge ─────────────────────────────────────────────────────────────────────
type BadgeColor = 'green' | 'red' | 'yellow' | 'gray' | 'blue'
export function Badge({ children, color = 'gray' }: { children: ReactNode; color?: BadgeColor }) {
  const colors: Record<BadgeColor, CSSProperties> = {
    green:  { background: 'rgba(34,197,94,0.1)',  color: '#4ade80',  border: '1px solid rgba(34,197,94,0.2)' },
    red:    { background: 'rgba(248,113,113,0.1)', color: '#f87171', border: '1px solid rgba(248,113,113,0.2)' },
    yellow: { background: 'rgba(251,191,36,0.1)', color: '#fbbf24', border: '1px solid rgba(251,191,36,0.2)' },
    gray:   { background: 'var(--surface2)',       color: 'var(--text2)', border: '1px solid var(--border)' },
    blue:   { background: 'rgba(88,101,242,0.1)',  color: '#818cf8', border: '1px solid rgba(88,101,242,0.2)' },
  }
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, padding: '2px 8px',
      borderRadius: 4, fontSize: 11, fontWeight: 500, lineHeight: '18px', ...colors[color] }}>
      {children}
    </span>
  )
}

// ── Select ────────────────────────────────────────────────────────────────────
export function Select({ value, onChange, children, style }: {
  value: string | number; onChange: (v: string) => void; children: ReactNode; style?: CSSProperties
}) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} style={{
      width: '100%', padding: '7px 10px', background: 'var(--surface2)',
      border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)',
      fontSize: 13, fontFamily: 'inherit', cursor: 'pointer', outline: 'none', ...style,
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
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
        <span style={{ fontSize: 12, color: 'var(--text2)' }}>{label}</span>
        <span style={{ fontSize: 12, color: 'var(--text)', fontFamily: 'JetBrains Mono, monospace' }}>
          {format ? format(value) : value}
        </span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }} />
    </div>
  )
}

// ── Field ─────────────────────────────────────────────────────────────────────
export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <label style={{ display: 'block', fontSize: 12, color: 'var(--text2)', marginBottom: 5, fontWeight: 500 }}>
        {label}
      </label>
      {children}
    </div>
  )
}

// ── TextInput ─────────────────────────────────────────────────────────────────
export function TextInput({ value, onChange, placeholder, onKeyDown }: {
  value: string; onChange: (v: string) => void; placeholder?: string
  onKeyDown?: (e: React.KeyboardEvent) => void
}) {
  return (
    <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
      onKeyDown={onKeyDown}
      style={{ width: '100%', padding: '7px 10px', background: 'var(--surface2)',
        border: '1px solid var(--border)', borderRadius: 6, color: 'var(--text)',
        fontSize: 13, fontFamily: 'inherit', outline: 'none' }} />
  )
}

// ── PageHeader ────────────────────────────────────────────────────────────────
export function PageHeader({ back, title, subtitle, actions }: {
  back?: () => void; title: string; subtitle?: string; actions?: ReactNode
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 28 }}>
      {back && (
        <button onClick={back} style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
          width: 28, height: 28, border: '1px solid var(--border)', borderRadius: 6,
          background: 'transparent', color: 'var(--text2)', cursor: 'pointer', flexShrink: 0 }}>
          ←
        </button>
      )}
      <div style={{ flex: 1 }}>
        <h1 style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)', letterSpacing: '-0.02em', lineHeight: 1.2 }}>
          {title}
        </h1>
        {subtitle && <p style={{ fontSize: 13, color: 'var(--text2)', marginTop: 2 }}>{subtitle}</p>}
      </div>
      {actions && <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>{actions}</div>}
    </div>
  )
}

// ── Empty state ───────────────────────────────────────────────────────────────
export function Empty({ icon: Icon, message, sub }: { icon: React.ElementType; message: string; sub?: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      padding: '60px 24px', gap: 12 }}>
      <Icon size={32} style={{ color: 'var(--text3)' }} strokeWidth={1.5} />
      <div style={{ textAlign: 'center' }}>
        <p style={{ color: 'var(--text2)', fontSize: 14 }}>{message}</p>
        {sub && <p style={{ color: 'var(--text3)', fontSize: 12, marginTop: 4 }}>{sub}</p>}
      </div>
    </div>
  )
}

// ── Log terminal ──────────────────────────────────────────────────────────────
import { type RefObject } from 'react'
export function LogTerminal({ logs, logRef }: { logs: string[]; logRef: RefObject<HTMLDivElement | null> }) {
  return (
    <div ref={logRef} style={{ height: 180, overflowY: 'auto', background: '#07070a',
      border: '1px solid var(--border)', borderRadius: 6, padding: '10px 12px',
      fontFamily: 'JetBrains Mono, monospace', fontSize: 12, lineHeight: 1.7 }}>
      {logs.length === 0
        ? <span style={{ color: 'var(--text3)' }}>Waiting…</span>
        : logs.map((l, i) => (
          <div key={i} style={{ color:
            l.startsWith('❌') ? 'var(--danger)' :
            l.startsWith('✅') ? 'var(--success)' :
            l.startsWith('⚠️') ? 'var(--warn)' :
            l.startsWith('🚀') ? '#818cf8' : 'var(--text2)'
          }}>{l}</div>
        ))}
    </div>
  )
}

// ── Progress bar ──────────────────────────────────────────────────────────────
export function ProgressBar({ value, max, label, sublabel }: { value: number; max: number; label?: string; sublabel?: string }) {
  const pct = max > 0 ? Math.round((value / max) * 100) : 0
  return (
    <div>
      {(label || sublabel) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: 12, color: 'var(--text2)' }}>{label}</span>
          <span style={{ fontSize: 12, color: 'var(--text3)', fontFamily: 'JetBrains Mono, monospace' }}>{sublabel ?? `${pct}%`}</span>
        </div>
      )}
      <div style={{ height: 4, background: 'var(--surface3)', borderRadius: 2, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: 'var(--accent)', borderRadius: 2, transition: 'width 0.3s ease' }} />
      </div>
    </div>
  )
}

// ── MetricPill ────────────────────────────────────────────────────────────────
export function MetricPill({ label, value, color = 'var(--accent)' }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 6, padding: '8px 12px', textAlign: 'center' }}>
      <p style={{ fontSize: 11, color: 'var(--text3)', marginBottom: 3, letterSpacing: '0.04em', textTransform: 'uppercase' }}>{label}</p>
      <p style={{ fontSize: 17, fontWeight: 600, color, fontFamily: 'JetBrains Mono, monospace', letterSpacing: '-0.01em' }}>{value}</p>
    </div>
  )
}
