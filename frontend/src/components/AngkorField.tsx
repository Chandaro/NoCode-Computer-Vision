import { useEffect, useRef, useState } from 'react'

/*
 * AngkorField: Angkor Wat at sunrise, as seen by a vision model.
 *
 * The scene (sky, rising sun, the five lotus-bud towers, sugar palms, the moat
 * and its reflection, lotus buds) is evaluated per pixel into a low-resolution
 * buffer and ordered-dithered into a small palette, then scaled up with
 * nearest-neighbour sampling, so it reads like a printed temple rubbing.
 *
 * Interaction: the pointer is a lens. Inside it the scene switches to the
 * model's view (flat class colours with outlined edges) and a tag names the
 * class under the pointer. Once the sun is up, detection boxes draw in.
 */

type RGB = [number, number, number]
const hex = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]
const ramp = (...c: string[]) => c.map(hex)

const SKY   = ramp('#100e22', '#1f1a46', '#3c2a66', '#7a3f78', '#c66184', '#ea8fa2', '#f0b15a', '#f7d79a')
const SUN   = ramp('#f0b15a', '#f7d79a', '#fff1d6')
const STONE = ramp('#0b0918', '#161230', '#241c45')
const WATER = ramp('#0b0a1c', '#171537', '#2e2456', '#5b3466', '#9a4f74', '#c98a5c', '#f0b15a')
const PAD   = ramp('#0d2620', '#1f4a40', '#2f6b5c')
const BUD   = ramp('#7a3550', '#ea8fa2', '#f7c9d2')

// Classes for the segmentation lens
const C = { SKY: 0, SUN: 1, TEMPLE: 2, PALM: 3, BANK: 4, WATER: 5, PAD: 6, LOTUS: 7 } as const
const MASK = ['#3b3fa8', '#f5d33a', '#ef6a55', '#86c27a', '#8a6a4a', '#2f9e9a', '#3f8f55', '#ea8fa2'].map(hex)
const EDGE = hex('#fff1d6')
const RING = hex('#e0a83a')
export const CLASS_NAMES: [string, string][] = [
  ['មេឃ', 'sky'], ['ព្រះអាទិត្យ', 'sun'], ['ប្រាសាទ', 'temple'], ['ដើមត្នោត', 'sugar palm'],
  ['ច្រាំង', 'riverbank'], ['ទឹក', 'water'], ['ស្លឹកឈូក', 'lotus leaf'], ['ផ្កាឈូក', 'lotus'],
]

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => (v + 0.5) / 16)
const PX = 3
const LENS = 30          // lens radius, buffer px

interface Box { x: number; y: number; w: number; h: number; km: string; en: string; delay: number }

function geometry(W: number, H: number) {
  const U = H * 0.8
  const hz = Math.round(H * 0.74)
  const cx = W * (W / H > 2.2 ? 0.7 : 0.62)
  const towers = [
    { ax: 0, base: 0.2, top: 0.66, hw: 0.068 },
    { ax: -0.2, base: 0.2, top: 0.5, hw: 0.054 }, { ax: 0.2, base: 0.2, top: 0.5, hw: 0.054 },
    { ax: -0.36, base: 0.14, top: 0.43, hw: 0.05 }, { ax: 0.36, base: 0.14, top: 0.43, hw: 0.05 },
  ]
  const palms = [
    { x: cx + 0.7 * U, h: 0.62 }, { x: cx + 0.86 * U, h: 0.47 },
    { x: cx - 0.72 * U, h: 0.5 },
  ].filter(p => p.x - 0.1 * U > W * 0.48 && p.x + 0.1 * U < W - 2)
  const pads = [
    { x: cx - 0.55 * U, y: H * 0.94, r: 0.075, bud: true },
    { x: cx - 0.3 * U, y: H * 0.985, r: 0.06, bud: false },
    { x: cx + 0.55 * U, y: H * 0.9, r: 0.065, bud: true },
    { x: cx + 0.8 * U, y: H * 0.975, r: 0.085, bud: false },
  ].filter(p => p.x > W * 0.45)
  return { U, hz, cx, towers, palms, pads }
}

function budW(t: number) {
  if (t < 0) return 0
  if (t < 0.07) return 1.06
  if (t > 1) return t < 1.07 ? 0.09 : 0
  return Math.pow(1 - Math.pow(t, 1.7), 0.5) * (1 - 0.08 * ((t * 8) % 1))
}

export default function AngkorField({ height = 420 }: { height?: number }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const tagRef = useRef<HTMLDivElement>(null)
  const [boxes, setBoxes] = useState<Box[]>([])

  useEffect(() => {
    const canvas = canvasRef.current, wrap = wrapRef.current, tag = tagRef.current
    if (!canvas || !wrap || !tag) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    let W = 0, H = 0, img: ImageData | null = null
    let sil = new Uint8Array(0)        // static silhouette class (0 = none) for y < hz
    let lab = new Uint8Array(0)        // per-frame class map
    let val = new Float32Array(0)      // per-frame tone 0..1
    let g = geometry(1, 1)
    let raf = 0, visible = true, last = 0
    const t0 = performance.now()
    let lens: { x: number; y: number } | null = null
    // Once the sun has risen the sky and temple stop changing, so their
    // colours are cached and only the moat (and the lens) are redrawn.
    let skyDone = false
    let aboveCache: Uint8ClampedArray | null = null

    const build = () => {
      const rect = canvas.getBoundingClientRect()
      W = Math.max(1, Math.ceil(rect.width / PX)); H = Math.max(1, Math.ceil(rect.height / PX))
      canvas.width = W; canvas.height = H
      img = ctx.createImageData(W, H)
      lab = new Uint8Array(W * H); val = new Float32Array(W * H); sil = new Uint8Array(W * H)
      g = geometry(W, H)
      skyDone = false; aboveCache = null
      const { U, hz, cx, towers, palms } = g
      for (let y = 0; y < hz; y++) {
        const h = (hz - y) / U
        for (let x = 0; x < W; x++) {
          const a = (x - cx) / U
          let c = 0
          // riverbank and the low tree line along it
          const tree = 0.03 + 0.018 * (Math.sin(x * 0.37) * 0.5 + Math.sin(x * 0.13 + 1) * 0.5 + 1) / 2
          if (h < 0.022 || (Math.abs(a) > 0.8 && h < tree)) c = C.BANK
          // outer gallery, centre gopura, terraces
          if (Math.abs(a) < 0.8 && h < 0.075) c = C.TEMPLE
          if (Math.abs(a) < 0.07 && h < 0.075 + 0.06 * budW((h - 0.075) / 0.06)) c = C.TEMPLE
          if ((Math.abs(a - 0.8) < 0.035 || Math.abs(a + 0.8) < 0.035) && h < 0.075 + 0.045 * budW((h - 0.075) / 0.045)) c = C.TEMPLE
          if (Math.abs(a) < 0.5 && h < 0.14) c = C.TEMPLE
          if (Math.abs(a) < 0.41 && h < 0.2) c = C.TEMPLE
          for (const tw of towers) {
            const t = (h - tw.base) / (tw.top - tw.base)
            if (t >= 0 && Math.abs(a - tw.ax) < tw.hw * budW(t)) c = C.TEMPLE
          }
          // sugar palms (thnaot): thin trunk, round crown of fronds
          for (const p of palms) {
            const top = p.h
            const tx = p.x + Math.sin(h * 4) * 0.012 * U
            if (h < top && Math.abs(x - tx) < Math.max(0.7, 0.009 * U)) c = C.PALM
            const dx = (x - (p.x + Math.sin(top * 4) * 0.012 * U)) / U, dy = h - top
            const r = Math.hypot(dx, dy), th = Math.atan2(dy, dx)
            if (r < 0.085 * (0.45 + 0.55 * Math.pow(Math.abs(Math.cos(th * 4.5)), 0.5))) c = C.PALM
          }
          sil[y * W + x] = c === 0 ? 0 : c + 1
        }
      }
      // detection boxes in css px
      const bx: Box[] = []
      const toCss = (v: number) => v * PX
      bx.push({ x: toCss(cx - 0.44 * U), y: toCss(hz - 0.69 * U), w: toCss(0.88 * U), h: toCss(0.69 * U), km: 'ប្រាសាទ', en: 'temple', delay: 0 })
      palms.forEach((p, i) => bx.push({ x: toCss(p.x - 0.1 * U), y: toCss(hz - (p.h + 0.1) * U), w: toCss(0.2 * U), h: toCss((p.h + 0.1) * U), km: 'ដើមត្នោត', en: 'sugar palm', delay: 0.25 + i * 0.18 }))
      g.pads.filter(p => p.bud).forEach((p, i) => bx.push({ x: toCss(p.x - 0.045 * U), y: toCss(p.y - 0.12 * U), w: toCss(0.09 * U), h: toCss(0.12 * U), km: 'ផ្កាឈូក', en: 'lotus', delay: 0.9 + i * 0.2 }))
      setBoxes(bx)
    }

    const frame = (now: number) => {
      if (!img) return
      const t = now - t0
      const { U, hz, cx, pads } = g
      const p = reduced ? 1 : easeOut(Math.min(1, t / 4200))
      const sunR = 0.1 * U, sunX = cx, sunY = hz - (0.08 + 0.3 * p) * U
      const dawn = 0.35 + 0.65 * p

      // pass 1: above the waterline (skipped once the sunrise has finished)
      if (!skyDone) for (let y = 0; y < hz; y++) {
        const v = y / hz
        for (let x = 0; x < W; x++) {
          const i = y * W + x
          const s = sil[i]
          if (s) { lab[i] = s - 1; val[i] = s - 1 === C.TEMPLE && y % 3 === 0 ? 0.55 : 0.25; continue }
          const d = Math.hypot(x - sunX, y - sunY)
          if (d < sunR) { lab[i] = C.SUN; val[i] = 0.45 + 0.55 * (1 - d / sunR); continue }
          const glow = Math.exp(-d / (0.55 * U))
          lab[i] = C.SKY
          val[i] = Math.min(1, 0.06 + 0.55 * Math.pow(v, 1.8) * dawn + 0.5 * glow * dawn)
        }
      }
      // pass 2: the moat, reflecting pass 1 with ripples
      const tt = reduced ? 0 : t * 0.0022
      for (let y = hz; y < H; y++) {
        const depth = (y - hz) / Math.max(1, H - hz)
        for (let x = 0; x < W; x++) {
          const i = y * W + x
          let cls: number = C.WATER, tone: number
          const rip = Math.sin(y * 0.32 + tt * 2 + Math.sin(x * 0.03 + tt) * 1.2) * (0.5 + depth * 1.6)
          const sx = Math.min(W - 1, Math.max(0, Math.round(x + rip)))
          const sy = Math.min(hz - 1, Math.max(0, Math.round(2 * hz - y - 1 + Math.sin(x * 0.2 + tt * 2) * 0.6)))
          const j = sy * W + sx
          const src = lab[j]
          tone = src === C.SKY || src === C.SUN ? val[j] * 0.72 - depth * 0.14 : 0.06
          if (Math.sin(x * 0.11 + y * 2.1 + tt * 4) > 0.985 - depth * 0.02 && src === C.SKY) tone += 0.25
          for (const pd of pads) {
            const ex = (x - pd.x) / (pd.r * U), ey = (y - pd.y) / (pd.r * U * 0.26)
            if (ex * ex + ey * ey < 1) { cls = C.PAD; tone = 0.35 + 0.5 * (1 - (ex * ex + ey * ey)) }
            if (pd.bud) {
              const bh = (pd.y - y) / (0.11 * U)
              if (bh >= 0 && bh <= 1.02 && Math.abs(x - pd.x) < 0.032 * U * budW(bh)) { cls = C.LOTUS; tone = 0.35 + 0.6 * bh }
            }
          }
          lab[i] = cls; val[i] = Math.max(0, Math.min(1, tone))
        }
      }
      // colour pass, with the lens
      const d = img.data
      const paint = (y0: number, y1: number, x0: number, x1: number, withLens: boolean) => {
        for (let y = y0; y < y1; y++) {
          for (let x = x0; x < x1; x++) {
            const i = y * W + x
            const th = BAYER[(x & 3) + ((y & 3) << 2)]
            const cl = lab[i]
            const rmp = cl === C.SKY ? SKY : cl === C.SUN ? SUN : cl === C.WATER ? WATER
              : cl === C.PAD ? PAD : cl === C.LOTUS ? BUD : STONE
            const f = val[i] * (rmp.length - 1)
            const lo = Math.floor(f)
            let col = rmp[Math.min(rmp.length - 1, lo + (f - lo > th ? 1 : 0))]
            if (withLens && lens) {
              const ld = Math.hypot(x - lens.x, y - lens.y)
              if (ld < LENS) {
                const edge = (x + 1 < W && lab[i + 1] !== cl) || (y + 1 < H && lab[i + W] !== cl)
                col = edge ? EDGE : th < 0.8 ? MASK[cl] : col
              } else if (ld < LENS + 1) col = RING
            }
            const k = i << 2
            d[k] = col[0]; d[k + 1] = col[1]; d[k + 2] = col[2]; d[k + 3] = 255
          }
        }
      }
      if (aboveCache) {
        d.set(aboveCache)
        paint(hz, H, 0, W, true)
      } else {
        paint(0, hz, 0, W, false)
        if (p >= 1) { skyDone = true; aboveCache = d.slice(0, hz * W * 4) }
        paint(hz, H, 0, W, true)
      }
      if (lens && lens.y - LENS - 2 < hz) {
        const ly0 = Math.max(0, Math.floor(lens.y - LENS - 2)), ly1 = Math.min(hz, Math.ceil(lens.y + LENS + 2))
        const lx0 = Math.max(0, Math.floor(lens.x - LENS - 2)), lx1 = Math.min(W, Math.ceil(lens.x + LENS + 2))
        paint(ly0, ly1, lx0, lx1, true)
      }
      ctx.putImageData(img, 0, 0)
    }

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      if (!visible || document.hidden) return
      // full rate while the sun rises or the lens is in use, then just the slow ripple
      if (now - last < (lens || !skyDone ? 50 : 100)) return
      last = now
      frame(now)
    }

    const onMove = (ev: PointerEvent) => {
      const r = canvas.getBoundingClientRect()
      const x = (ev.clientX - r.left) / PX, y = (ev.clientY - r.top) / PX
      if (x < 0 || y < 0 || x >= W || y >= H) { onLeave(); return }
      lens = { x, y }
      const cl = lab[Math.floor(y) * W + Math.floor(x)]
      const [km, en] = CLASS_NAMES[cl] ?? CLASS_NAMES[0]
      tag.style.opacity = '1'
      tag.style.transform = `translate(${ev.clientX - r.left + 18}px, ${ev.clientY - r.top + 18}px)`
      tag.firstElementChild!.textContent = km
      tag.lastElementChild!.textContent = en
      if (reduced) frame(performance.now())
    }
    const onLeave = () => { lens = null; tag.style.opacity = '0'; if (reduced) frame(performance.now()) }

    const io = new IntersectionObserver(([en]) => { visible = en.isIntersecting })
    io.observe(canvas)
    const ro = new ResizeObserver(() => { build(); frame(performance.now()) })
    ro.observe(canvas)
    wrap.addEventListener('pointermove', onMove)
    wrap.addEventListener('pointerleave', onLeave)
    build()
    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf); io.disconnect(); ro.disconnect()
      wrap.removeEventListener('pointermove', onMove)
      wrap.removeEventListener('pointerleave', onLeave)
    }
  }, [])

  return (
    <div ref={wrapRef} style={{ position: 'absolute', inset: 0, cursor: 'crosshair' }}>
      <canvas ref={canvasRef} aria-hidden="true"
        style={{ display: 'block', width: '100%', height, imageRendering: 'pixelated' }} />
      <svg className="kh-boxes" aria-hidden="true" width="100%" height={height} style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
        {boxes.map((b, i) => (
          <g key={`${i}-${Math.round(b.x)}`} className="kh-box">
            <rect x={b.x} y={b.y} width={b.w} height={b.h} rx="2"
              style={{ animationDelay: `${4.2 + b.delay}s` }} />
            <g className="tag" style={{ animationDelay: `${4.9 + b.delay}s` }}>
              <rect x={b.x - 0.75} y={b.y - 21} width={b.en.length * 6.6 + b.km.length * 7 + 22} height="20" rx="2"
                style={{ fill: 'var(--accent)', stroke: 'none', strokeDasharray: 'none', animation: 'none' }} />
              <text x={b.x + 7} y={b.y - 7} style={{ font: '600 11px var(--font-body)', fill: 'var(--on-accent)' }}>
                {b.km}  {b.en}
              </text>
            </g>
          </g>
        ))}
      </svg>
      <div ref={tagRef} aria-hidden="true" style={{
        position: 'absolute', left: 0, top: 0, opacity: 0, pointerEvents: 'none',
        transition: 'opacity 0.15s', display: 'flex', gap: 8, alignItems: 'baseline',
        padding: '4px 10px', borderRadius: 6, background: 'rgba(16,14,34,0.88)',
        border: '1px solid var(--accent)', whiteSpace: 'nowrap',
      }}>
        <span style={{ fontFamily: 'var(--font-body)', fontSize: 13, color: 'var(--text)' }} />
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--accent)' }} />
      </div>
    </div>
  )
}

function easeOut(k: number) { return 1 - Math.pow(1 - Math.min(1, Math.max(0, k)), 3) }
