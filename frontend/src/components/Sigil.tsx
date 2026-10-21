// Small Khmer motifs drawn as SVG.
//   PrasatMark: the app mark, Angkor's lotus-bud towers inside detection brackets.
//   Lotus: each project's sigil; the seed picks the petal colour and how open it is.

export function hashSeed(n: number | string): number {
  const s = String(n)
  let h = 2166136261
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) }
  h ^= h >>> 15; h = Math.imul(h, 2246822507); h ^= h >>> 13; h = Math.imul(h, 3266489909); h ^= h >>> 16
  return (h >>> 0) / 4294967295
}

// [petal, inner line]
const PETALS: [string, string][] = [
  ['#ea8fa2', '#f7c9d2'],   // pink lotus
  ['#e0a83a', '#f6d690'],   // gold leaf
  ['#efe4cf', '#ffffff'],   // white lotus
  ['#f0913a', '#f9c48c'],   // saffron
  ['#9b7fd1', '#cdbdf0'],   // blue-violet water lily
  ['#4fb3a9', '#a7e0d9'],   // moat teal
]

const PETAL = 'M16 5 C 20.2 10.5, 20.2 18, 16 24 C 11.8 18, 11.8 10.5, 16 5 Z'

export function Lotus({ size = 32, seed = 0, title }: { size?: number; seed?: number | string; title?: string }) {
  const r = hashSeed(seed)
  const [fill, line] = PETALS[Math.floor(r * PETALS.length) % PETALS.length]
  const open = 22 + (r * 97 % 1) * 16     // degrees between petals
  const petals = [
    { a: -open * 2, s: 0.72 }, { a: open * 2, s: 0.72 },
    { a: -open, s: 0.86 }, { a: open, s: 0.86 },
    { a: 0, s: 1 },
  ]
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden={title ? undefined : true}
      role={title ? 'img' : undefined} style={{ flexShrink: 0, display: 'block' }}>
      {title && <title>{title}</title>}
      <ellipse cx="16" cy="26.5" rx="12.5" ry="3" fill="#1f4a40" />
      <path d="M5 26.5 Q16 23.5 27 26.5" stroke="#2f6b5c" strokeWidth="1" fill="none" />
      {petals.map((p, i) => (
        <g key={i} transform={`rotate(${p.a} 16 24) translate(16 24) scale(${p.s}) translate(-16 -24)`}>
          <path d={PETAL} fill={fill} stroke="#100e22" strokeWidth="1.1" />
          <path d="M16 8.5 C 17.6 13, 17.6 18, 16 21.5" stroke={line} strokeWidth="0.9" fill="none" opacity="0.8" />
        </g>
      ))}
    </svg>
  )
}

export function PrasatMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden style={{ flexShrink: 0, display: 'block' }}>
      {/* detection brackets */}
      <path d="M2 8 V2 H8 M24 2 H30 V8 M30 24 V30 H24 M8 30 H2 V24" stroke="#ea8fa2" strokeWidth="2" fill="none" strokeLinecap="round" />
      {/* towers: centre lotus bud flanked by two smaller ones, on a stepped base */}
      <path d="M16 5.5 C 18.6 9, 19.4 14, 19 19 H13 C 12.6 14, 13.4 9, 16 5.5 Z" fill="#e0a83a" />
      <path d="M10.2 11 C 12 13.4, 12.5 16.5, 12.3 19 H8.1 C 7.9 16.5, 8.4 13.4, 10.2 11 Z" fill="#e0a83a" opacity="0.85" />
      <path d="M21.8 11 C 23.6 13.4, 24.1 16.5, 23.9 19 H19.7 C 19.5 16.5, 20 13.4, 21.8 11 Z" fill="#e0a83a" opacity="0.85" />
      <path d="M6 19 H26 V21.5 H6 Z M7.5 21.5 H24.5 V24.5 H7.5 Z" fill="#e0a83a" />
    </svg>
  )
}
