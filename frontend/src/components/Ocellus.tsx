// A single peacock "eye" (ocellus), drawn as SVG. Used for the brand mark and
// as each project's sigil. The seed shifts hue and ring proportions so every
// project gets its own eye without the palette leaving the Argus family.

export function hashSeed(n: number | string): number {
  const s = String(n)
  let h = 2166136261
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619) }
  // final avalanche so neighbouring ids land far apart
  h ^= h >>> 15; h = Math.imul(h, 2246822507); h ^= h >>> 13; h = Math.imul(h, 3266489909); h ^= h >>> 16
  return (h >>> 0) / 4294967295
}

const IRISES = ['#5866e8', '#3cc4b4', '#7a5be0', '#2f9fd8', '#4bb37a', '#c46a9a']

export function Ocellus({ size = 28, seed = 0, lookX = 0, lookY = 0, title }: {
  size?: number; seed?: number | string; lookX?: number; lookY?: number; title?: string
}) {
  const r = hashSeed(seed)
  const iris = IRISES[Math.floor(r * IRISES.length) % IRISES.length]
  const ring = 0.62 + (r * 7 % 1) * 0.1        // gold ring radius
  const irisR = 0.4 + (r * 13 % 1) * 0.08
  const px = 16 + lookX * 3, py = 17 + lookY * 3
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden={title ? undefined : true}
      role={title ? 'img' : undefined} style={{ flexShrink: 0, display: 'block' }}>
      {title && <title>{title}</title>}
      {/* teardrop feather vane */}
      <path d="M16 1.5 C 25 8, 29.5 14, 29.5 19.5 A 13.5 12 0 0 1 2.5 19.5 C 2.5 14, 7 8, 16 1.5 Z"
        fill="#0f3a3a" stroke="#1f6a63" strokeWidth="1" />
      <ellipse cx="16" cy="18.5" rx={12 * ring + 1.5} ry={11 * ring + 1.5} fill="#e4b43f" />
      <ellipse cx="16" cy="18.5" rx={12 * ring} ry={11 * ring} fill="#0d2b2d" />
      <circle cx={px} cy={py + 1} r={12 * irisR} fill={iris} />
      <circle cx={px} cy={py + 1} r={12 * irisR * 0.5} fill="#05090b" />
      <circle cx={px + 1.6} cy={py - 0.6} r="1.1" fill="#ecf1e7" opacity="0.9" />
    </svg>
  )
}
