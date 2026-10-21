import { useEffect, useRef } from 'react'

/*
 * ArgusField: a live, ordered-dither field of watching eyes.
 *
 * The scene is rendered per pixel into a low-resolution buffer (one buffer
 * pixel = PX css pixels) and dithered with a 4x4 Bayer matrix into a small
 * peacock palette, then scaled up with nearest-neighbour sampling. That gives
 * the printed, 1-bit-ish texture without any image assets.
 *
 * Eyes follow the pointer anywhere on the page, look at whatever element calls
 * setArgusGaze(), blink on their own, and flinch in a ripple when clicked.
 */

type RGB = [number, number, number]
const NIGHT: RGB = [7, 19, 21]
const DEEP: RGB  = [14, 44, 46]
const BARB: RGB  = [31, 106, 99]
const TEAL: RGB  = [60, 196, 180]
const GOLD: RGB  = [228, 180, 63]
const IRIS: RGB  = [88, 102, 232]
const DOWN: RGB  = [236, 241, 231]

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => (v + 0.5) / 16)

const PX = 3          // css px per buffer pixel
const SX = 30         // eye spacing (buffer px)
const SY = 22
const RX = 10.5       // eye half-width
const RY = 5.6        // eye max half-height

interface Eye {
  cx: number; cy: number; s: number; present: boolean
  open: number; wake: number
  lx: number; ly: number; ease: number
  nextBlink: number; blinkStart: number
  wanderPhase: number
}

// Shared gaze target, in client (viewport) coordinates.
let gazeTarget: { x: number; y: number } | null = null
export function setArgusGaze(el: Element | null) {
  if (!el) { gazeTarget = null; return }
  const r = el.getBoundingClientRect()
  gazeTarget = { x: r.left + r.width / 2, y: r.top + r.height / 2 }
}

function rand(i: number) {
  const x = Math.sin(i * 127.1 + 311.7) * 43758.5453
  return x - Math.floor(x)
}

export default function ArgusField({ height = 380, className, quietLeft = true }: {
  height?: number; className?: string; quietLeft?: boolean
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    let W = 0, H = 0
    let img: ImageData | null = null
    let eyes: Eye[] = []
    let grid: (Eye | null)[][] = []
    let raf = 0
    let visible = true
    const t0 = performance.now()
    let pointer: { x: number; y: number; t: number } | null = null

    const build = () => {
      const rect = canvas.getBoundingClientRect()
      W = Math.max(1, Math.ceil(rect.width / PX))
      H = Math.max(1, Math.ceil(rect.height / PX))
      canvas.width = W; canvas.height = H
      img = ctx.createImageData(W, H)
      eyes = []; grid = []
      const rows = Math.ceil(H / SY) + 1
      const cols = Math.ceil(W / SX) + 1
      for (let r = 0; r <= rows; r++) {
        grid[r] = []
        for (let c = 0; c <= cols; c++) {
          const id = r * 997 + c * 13
          const off = r % 2 ? SX / 2 : 0
          const cx = c * SX + off + (rand(id) - 0.5) * 4
          const cy = r * SY + (rand(id + 1) - 0.5) * 3
          // Keep the left third calmer so the headline stays legible.
          const xn = cx / W
          const calm = quietLeft ? Math.min(1, Math.max(0, (xn - 0.08) / 0.42)) : 1
          const present = rand(id + 2) < 0.25 + 0.65 * calm
          const s = (0.62 + rand(id + 3) * 0.5) * (0.7 + 0.3 * calm)
          const e: Eye = {
            cx, cy, s, present,
            open: reduced ? 1 : 0,
            wake: reduced ? 0 : 250 + xn * 1100 + rand(id + 4) * 350,   // intro: eyes open left to right
            lx: 0, ly: 0, ease: 0.06 + rand(id + 5) * 0.08,
            nextBlink: 2500 + rand(id + 6) * 9000, blinkStart: -1,
            wanderPhase: rand(id + 7) * 100,
          }
          grid[r][c] = e
          eyes.push(e)
        }
      }
    }

    const step = (now: number) => {
      const t = now - t0
      const rect = canvas.getBoundingClientRect()
      // Gaze target in buffer coords
      let tx: number | null = null, ty: number | null = null
      const live = pointer && now - pointer.t < 5000 ? pointer : null
      const tgt = gazeTarget ?? live
      if (tgt) { tx = (tgt.x - rect.left) / PX; ty = (tgt.y - rect.top) / PX }

      for (const e of eyes) {
        if (!e.present) continue
        // open / blink
        let open = 1
        if (!reduced) {
          if (t < e.wake) open = 0
          else if (t < e.wake + 420) open = easeOut((t - e.wake) / 420)
          if (t > e.nextBlink && e.blinkStart < 0) e.blinkStart = t
          if (e.blinkStart >= 0) {
            const k = (t - e.blinkStart) / 190
            if (k >= 1) { e.blinkStart = -1; e.nextBlink = t + 3000 + Math.random() * 9000 }
            else open = Math.min(open, Math.abs(Math.cos(k * Math.PI)))
          }
        }
        e.open = open
        // look
        let gx: number, gy: number
        if (tx !== null && ty !== null) {
          const dx = tx - e.cx, dy = ty - e.cy
          const d = Math.hypot(dx, dy) || 1
          const m = Math.min(1, d / 45)
          gx = dx / d * m; gy = dy / d * m
        } else {
          const w = reduced ? 0 : t / 2400 + e.wanderPhase
          gx = Math.sin(w) * 0.55; gy = Math.cos(w * 0.7) * 0.4
        }
        e.lx += (gx - e.lx) * e.ease
        e.ly += (gy - e.ly) * e.ease
      }
    }

    const render = () => {
      if (!img) return
      const d = img.data
      for (let y = 0; y < H; y++) {
        const r = Math.round(y / SY)
        const rowEyes = grid[r]
        const off = r % 2 ? SX / 2 : 0
        for (let x = 0; x < W; x++) {
          const th = BAYER[(x & 3) + ((y & 3) << 2)]
          let a: RGB = NIGHT, b: RGB = DEEP, tv: number
          // background barbs: slanted threads that drift slightly with y
          const barb = Math.sin((x * 0.9 + y * 0.42) * 0.5 + Math.sin(y * 0.045) * 2.2)
          tv = 0.1 + 0.11 * (barb * 0.5 + 0.5)

          const c = Math.round((x - off) / SX)
          const e = rowEyes ? rowEyes[c] : null
          if (e && e.present) {
            const dx = x - e.cx, dy = y - e.cy
            const rx = RX * e.s, ry = RY * e.s
            const u = dx / rx
            // feather vane halo around each eye
            const ev = Math.hypot(dx / (rx * 1.38), dy / (ry * 2.15))
            if (ev > 0.9 && ev < 1.0) { a = NIGHT; b = BARB; tv = 0.55 }
            else if (ev <= 0.9) { tv = 0.22 }

            if (Math.abs(u) < 1) {
              const h = ry * (1 - u * u) * e.open
              const ady = Math.abs(dy)
              if (h > 0.6 && ady < h) {
                const pcx = e.cx + e.lx * rx * 0.42
                const pcy = e.cy + e.ly * ry * 0.3
                const pd = Math.hypot(x - pcx, y - pcy)
                const ir = ry * 0.95, pr = ir * 0.46, gr = ir * 1.28
                if (ady > h - 0.85) { a = DEEP; b = TEAL; tv = 0.9 }           // lid line
                else if (pd < pr) {
                  if (Math.hypot(x - (pcx + 1.1), y - (pcy - 1.1)) < 0.95) { a = DOWN; b = DOWN; tv = 1 }
                  else { a = NIGHT; b = NIGHT; tv = 0 }
                }
                else if (pd < ir) { a = NIGHT; b = IRIS; tv = 1.02 - 0.55 * (pd - pr) / (ir - pr) }
                else if (pd < gr) { a = DEEP; b = GOLD; tv = 0.86 }
                else { a = DEEP; b = BARB; tv = 0.5 }
              } else if (e.open < 0.2 && ady < 0.8 && Math.abs(u) < 0.92) {
                a = DEEP; b = TEAL; tv = 0.75                                   // closed lid
              }
            }
          }
          const col = tv > th ? b : a
          const i = (y * W + x) << 2
          d[i] = col[0]; d[i + 1] = col[1]; d[i + 2] = col[2]; d[i + 3] = 255
        }
      }
      ctx.putImageData(img, 0, 0)
    }

    let last = 0
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      if (!visible || document.hidden) return
      if (now - last < 33) return        // ~30 fps is plenty for a dithered field
      last = now
      step(now)
      render()
    }

    const onMove = (ev: PointerEvent) => { pointer = { x: ev.clientX, y: ev.clientY, t: performance.now() } }
    const onDown = (ev: PointerEvent) => {
      const rect = canvas.getBoundingClientRect()
      const bx = (ev.clientX - rect.left) / PX, by = (ev.clientY - rect.top) / PX
      const t = performance.now() - t0
      for (const e of eyes) {
        const dist = Math.hypot(e.cx - bx, e.cy - by)
        if (dist < 140) { e.blinkStart = -1; e.nextBlink = t + dist * 4 }
      }
    }
    const io = new IntersectionObserver(([en]) => { visible = en.isIntersecting })
    io.observe(canvas)
    const ro = new ResizeObserver(() => { build(); step(performance.now()); render() })
    ro.observe(canvas)
    window.addEventListener('pointermove', onMove, { passive: true })
    canvas.addEventListener('pointerdown', onDown)
    build()
    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf)
      io.disconnect(); ro.disconnect()
      window.removeEventListener('pointermove', onMove)
      canvas.removeEventListener('pointerdown', onDown)
    }
  }, [quietLeft])

  return (
    <canvas ref={canvasRef} className={className} aria-hidden="true"
      style={{ display: 'block', width: '100%', height, imageRendering: 'pixelated' }} />
  )
}

function easeOut(k: number) { return 1 - Math.pow(1 - Math.min(1, Math.max(0, k)), 3) }
