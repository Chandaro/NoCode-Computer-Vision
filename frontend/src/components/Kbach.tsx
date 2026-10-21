// Khmer ornament kit (kbach), drawn as SVG so it stays sharp and themable.
//
//   KbachCorner  flame-scroll (kbach phni plerng) corner piece for panels
//   Rosette      lotus ceiling rosette, as carved on Angkor's gallery ceilings
//   NagaRail     the naga balustrade: a scaled body ending in a seven-headed hood
//   AngkorMini   the five lotus-bud towers at sunrise, for small headers
//   LotusPip     a lotus bud that opens when its nav item is active
//   HolBand      ikat diamond lattice from Khmer hol silk, as a background

import type { CSSProperties } from 'react'

export function KbachCorner({ size = 34, style, className }: { size?: number; style?: CSSProperties; className?: string }) {
  return (
    <svg width={size} height={size} viewBox="0 0 40 40" aria-hidden className={className}
      style={{ position: 'absolute', pointerEvents: 'none', ...style }}>
      <g fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round">
        <path d="M2 38 V13 C2 6.5 6.5 2 13 2 H38" strokeWidth="1.3" opacity="0.9" />
        {/* flame tongues curling in from the corner */}
        <path d="M7 31 C7 19 11.5 12 21 9.5 C15.5 13.5 14.2 17.5 16 21 C17.6 16 21.5 13.5 26.5 14 C21.5 16.3 19.5 20.5 22 25" strokeWidth="1.2" />
        <path d="M7 31 C9.5 26.5 12.5 24.5 16 24.5" strokeWidth="1" opacity="0.75" />
        <path d="M31 7 C26.5 9.5 24.5 12.5 24.5 16" strokeWidth="1" opacity="0.75" />
      </g>
      <circle cx="6.5" cy="6.5" r="1.6" fill="currentColor" />
      <circle cx="22.5" cy="25.5" r="1.1" fill="currentColor" opacity="0.8" />
    </svg>
  )
}

/** Four corners around a positioned parent. */
export function KbachFrame({ color = 'var(--accent)', size = 30, inset = 6, opacity = 0.55 }: {
  color?: string; size?: number; inset?: number; opacity?: number
}) {
  const base: CSSProperties = { color, opacity }
  return (
    <span aria-hidden className="kh-frame">
      <KbachCorner size={size} style={{ ...base, top: inset, left: inset }} />
      <KbachCorner size={size} style={{ ...base, top: inset, right: inset, transform: 'scaleX(-1)' }} />
      <KbachCorner size={size} style={{ ...base, bottom: inset, left: inset, transform: 'scaleY(-1)' }} />
      <KbachCorner size={size} style={{ ...base, bottom: inset, right: inset, transform: 'scale(-1,-1)' }} />
    </span>
  )
}

const PETAL = 'M0 -46 C 9 -34, 10 -22, 0 -12 C -10 -22, -9 -34, 0 -46 Z'
const PETAL_IN = 'M0 -30 C 6 -23, 6.5 -16, 0 -9 C -6.5 -16, -6 -23, 0 -30 Z'

export function Rosette({ size = 180, spin = true, style, className }: {
  size?: number; spin?: boolean; style?: CSSProperties; className?: string
}) {
  return (
    <svg width={size} height={size} viewBox="-50 -50 100 100" aria-hidden className={`${spin ? 'kh-spin' : ''} ${className ?? ''}`}
      style={{ pointerEvents: 'none', ...style }}>
      <g fill="none" stroke="currentColor" strokeWidth="0.9">
        <circle r="48" opacity="0.5" />
        <circle r="44.5" strokeDasharray="1.5 3" opacity="0.7" />
        {Array.from({ length: 16 }, (_, i) => (
          <path key={`o${i}`} d={PETAL} transform={`rotate(${i * 22.5})`} />
        ))}
        {Array.from({ length: 8 }, (_, i) => (
          <path key={`i${i}`} d={PETAL_IN} transform={`rotate(${i * 45 + 22.5})`} opacity="0.85" />
        ))}
        <circle r="9" />
        <path d="M0 -6 L1.8 -1.8 L6 0 L1.8 1.8 L0 6 L-1.8 1.8 L-6 0 L-1.8 -1.8 Z" fill="currentColor" opacity="0.7" />
      </g>
    </svg>
  )
}

/** A seven-headed naga hood, fanned out like the end of an Angkor causeway railing. */
export function NagaHead({ size = 64, style }: { size?: number; style?: CSSProperties }) {
  const heads = [-66, -44, -22, 0, 22, 44, 66]
  return (
    <svg width={size} height={size * 0.78} viewBox="-50 -62 100 78" aria-hidden style={{ pointerEvents: 'none', ...style }}>
      <g stroke="currentColor" strokeWidth="1.3" strokeLinejoin="round">
        {/* hood behind the heads */}
        <path d="M-44 10 C -50 -20, -30 -50, 0 -56 C 30 -50, 50 -20, 44 10 Z" fill="var(--surface2)" opacity="0.95" />
        {heads.map((a, i) => (
          <g key={i} transform={`rotate(${a} 0 10)`}>
            <path d="M0 10 C -4 -10, -6 -24, -6.5 -32 C -6.5 -40, 6.5 -40, 6.5 -32 C 6 -24, 4 -10, 0 10 Z"
              fill="var(--surface)" />
            <circle cx="-2.4" cy="-33" r="1.1" fill="currentColor" stroke="none" />
            <circle cx="2.4" cy="-33" r="1.1" fill="currentColor" stroke="none" />
          </g>
        ))}
        <path d="M-14 12 C -8 4, 8 4, 14 12" fill="none" />
      </g>
    </svg>
  )
}

/** Scaled naga body running the full width, with the hood at the right end. */
export function NagaRail({ className }: { className?: string }) {
  return (
    <div aria-hidden className={`kh-naga ${className ?? ''}`}>
      <div className="kh-naga-body" />
      <NagaHead size={84} style={{ color: 'var(--accent)', flexShrink: 0, marginLeft: -10, marginBottom: -4 }} />
    </div>
  )
}

export function AngkorMini({ width = 180, height = 88 }: { width?: number; height?: number }) {
  // towers drawn as lotus buds on a stepped base; the sun pulses gently behind
  const bud = (cx: number, base: number, top: number, hw: number) =>
    `M${cx - hw} ${base} C ${cx - hw} ${base - (base - top) * 0.55}, ${cx - hw * 0.45} ${top + (base - top) * 0.2}, ${cx} ${top} C ${cx + hw * 0.45} ${top + (base - top) * 0.2}, ${cx + hw} ${base - (base - top) * 0.55}, ${cx + hw} ${base} Z`
  return (
    <svg className="kh-mini" width="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid slice" aria-hidden style={{ display: 'block' }}>
      <defs>
        <linearGradient id="kh-mini-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#1f1a46" />
          <stop offset="0.6" stopColor="#7a3f78" />
          <stop offset="1" stopColor="#ea8fa2" />
        </linearGradient>
        <radialGradient id="kh-mini-sun">
          <stop offset="0" stopColor="#fff1d6" />
          <stop offset="0.5" stopColor="#f0b15a" />
          <stop offset="1" stopColor="#f0b15a" stopOpacity="0" />
        </radialGradient>
      </defs>
      <rect width={width} height={height} fill="url(#kh-mini-sky)" />
      <circle className="kh-sun-pulse" cx={width / 2} cy={height * 0.5} r={height * 0.32} fill="url(#kh-mini-sun)" />
      <g fill="#0b0918">
        <path d={bud(width / 2, height * 0.72, height * 0.12, 7)} />
        <path d={bud(width / 2 - 20, height * 0.72, height * 0.32, 5.5)} />
        <path d={bud(width / 2 + 20, height * 0.72, height * 0.32, 5.5)} />
        <path d={bud(width / 2 - 36, height * 0.74, height * 0.42, 5)} />
        <path d={bud(width / 2 + 36, height * 0.74, height * 0.42, 5)} />
        <rect x={width / 2 - 46} y={height * 0.7} width="92" height={height * 0.08} />
        <rect x={width / 2 - 78} y={height * 0.77} width="156" height={height * 0.06} />
        <rect x="0" y={height * 0.83} width={width} height={height * 0.17} />
        {/* sugar palm */}
        <rect x={width - 26} y={height * 0.26} width="1.6" height={height * 0.6} />
        <circle cx={width - 25.2} cy={height * 0.26} r="7" />
        <rect x={22} y={height * 0.38} width="1.4" height={height * 0.47} />
        <circle cx={22.7} cy={height * 0.38} r="5.5" />
      </g>
      <rect x="0" y={height * 0.83} width={width} height={height * 0.17} fill="#ea8fa2" opacity="0.12" />
    </svg>
  )
}

export function LotusPip() {
  return (
    <svg className="kh-pip" width="14" height="14" viewBox="-8 -12 16 14" aria-hidden>
      <path className="lp-l" d="M0 0 C -2 -3, -2.5 -6.5, 0 -10 C 2.5 -6.5, 2 -3, 0 0 Z" />
      <path className="lp-r" d="M0 0 C -2 -3, -2.5 -6.5, 0 -10 C 2.5 -6.5, 2 -3, 0 0 Z" />
      <path className="lp-c" d="M0 0 C -2 -3, -2.5 -6.5, 0 -10 C 2.5 -6.5, 2 -3, 0 0 Z" />
    </svg>
  )
}
