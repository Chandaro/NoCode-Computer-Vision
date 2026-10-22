import { useEffect, useRef, useState } from 'react'
import type { World } from './KhmerWorlds'

/*
 * PixelScene: draws a world's scene in the same pixel-dither style as the home
 * hero. The SVG scene is rasterised once at low resolution (one buffer pixel =
 * PX css pixels), reduced to a small palette taken from the scene itself, and
 * ordered-dithered with a 4x4 Bayer matrix, then shown with nearest-neighbour
 * scaling. Motion is done on the pixels (water rows shimmer, light sweeps,
 * firelight flickers, glints, a sprite crossing) on a cached base image, and
 * the pointer is a lens that shows the scene as an edge-detection model sees it.
 */

const PX = 3
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => v / 16 - 0.5)

type RGB = [number, number, number]

// ── scene markup: the SVG is rendered once into a hidden node, read as text,
// then dropped from the page; later instances of the same world reuse it ──
const markupCache = new Map<string, string>()

function rasterise(markup: string, w: number, h: number): Promise<ImageData> {
  // draw at 2x then average down, so the dither works from clean colours
  const svg = markup.replace('<svg', `<svg xmlns="http://www.w3.org/2000/svg" width="${w * 2}" height="${h * 2}"`)
  const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml' }))
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const big = document.createElement('canvas'); big.width = w * 2; big.height = h * 2
      big.getContext('2d')!.drawImage(img, 0, 0)
      const small = document.createElement('canvas'); small.width = w; small.height = h
      const sc = small.getContext('2d', { willReadFrequently: true })!
      sc.imageSmoothingQuality = 'high'
      sc.drawImage(big, 0, 0, w, h)
      URL.revokeObjectURL(url)
      resolve(sc.getImageData(0, 0, w, h))
    }
    img.onerror = e => { URL.revokeObjectURL(url); reject(e) }
    img.src = url
  })
}

// ── palette from the scene (k-means on its pixels) ──
function palette(d: Uint8ClampedArray, k: number): RGB[] {
  const pts: RGB[] = []
  const step = Math.max(1, Math.floor(d.length / 4 / 3000))
  for (let i = 0; i < d.length; i += 4 * step) if (d[i + 3] > 128) pts.push([d[i], d[i + 1], d[i + 2]])
  if (!pts.length) return [[0, 0, 0]]
  // seed by luminance spread so dark and light both get colours
  const sorted = [...pts].sort((a, b) => (a[0] * 3 + a[1] * 6 + a[2]) - (b[0] * 3 + b[1] * 6 + b[2]))
  let c: RGB[] = Array.from({ length: k }, (_, i) => [...sorted[Math.floor((i + 0.5) / k * sorted.length)]] as RGB)
  for (let it = 0; it < 8; it++) {
    const sum = c.map(() => [0, 0, 0, 0])
    for (const p of pts) {
      let bi = 0, bd = Infinity
      for (let j = 0; j < c.length; j++) { const dr = p[0] - c[j][0], dg = p[1] - c[j][1], db = p[2] - c[j][2]; const dd = dr * dr + dg * dg + db * db; if (dd < bd) { bd = dd; bi = j } }
      const sm = sum[bi]; sm[0] += p[0]; sm[1] += p[1]; sm[2] += p[2]; sm[3]++
    }
    c = c.map((old, j) => sum[j][3] ? [sum[j][0] / sum[j][3], sum[j][1] / sum[j][3], sum[j][2] / sum[j][3]] as RGB : old)
  }
  return c
}

function nearest(pal: RGB[], r: number, g: number, b: number) {
  let bi = 0, bd = Infinity
  for (let j = 0; j < pal.length; j++) { const dr = r - pal[j][0], dg = g - pal[j][1], db = b - pal[j][2]; const dd = dr * dr + dg * dg + db * db; if (dd < bd) { bd = dd; bi = j } }
  return pal[bi]
}

function dither(src: ImageData, pal: RGB[], spread = 38, keepAlpha = false): ImageData {
  const { width: W, height: H, data } = src
  const out = new ImageData(W, H)
  const o = out.data
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = (y * W + x) * 4
    if (keepAlpha && data[i + 3] < 110) { o[i + 3] = 0; continue }
    const t = BAYER[(x & 3) + ((y & 3) << 2)] * spread
    const c = nearest(pal, data[i] + t, data[i + 1] + t, data[i + 2] + t)
    o[i] = c[0]; o[i + 1] = c[1]; o[i + 2] = c[2]; o[i + 3] = 255
  }
  return out
}

const hexRgb = (h: string): RGB => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]

export default function PixelScene({ world, live = false, className }: { world: World; live?: boolean; className?: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const hostRef = useRef<HTMLDivElement>(null)
  const needSprite = !!world.fx.sprite
  const ready = () => markupCache.has(`${world.id}|base`) && (!needSprite || markupCache.has(`${world.id}|sprite`))
  const [captured, setCaptured] = useState(ready)

  // capture markup from the hidden render, then stop rendering it
  useEffect(() => {
    if (captured || !hostRef.current) return
    const svgs = hostRef.current.querySelectorAll('svg')
    if (svgs[0]) markupCache.set(`${world.id}|base`, svgs[0].outerHTML)
    if (needSprite && svgs[1]) markupCache.set(`${world.id}|sprite`, svgs[1].outerHTML)
    setCaptured(true)
  }, [captured, world.id, needSprite])

  useEffect(() => {
    const wrap = wrapRef.current, canvas = canvasRef.current
    if (!wrap || !canvas || !captured) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let cancelled = false, raf = 0, last = 0, visible = true
    let W = 0, H = 0
    let base: ImageData | null = null, sprite: ImageData | null = null, frame: ImageData | null = null
    let pal: RGB[] = [], edges: Uint8Array | null = null
    let spriteBox = { x0: 0, x1: 0, y0: 0, y1: 0 }
    let lens: { x: number; y: number } | null = null
    const t0 = performance.now()
    const accent = hexRgb(world.vars['--accent'])
    const cream: RGB = [255, 241, 214]
    const fx = world.fx

    const build = async () => {
      const r = wrap.getBoundingClientRect()
      const w = Math.max(8, Math.ceil(r.width / PX)), h = Math.max(4, Math.ceil(r.height / PX))
      if (w === W && h === H && base) return
      W = w; H = h
      canvas.width = W; canvas.height = H
      try {
        const src = await rasterise(markupCache.get(`${world.id}|base`)!, W, H)
        if (cancelled) return
        pal = palette(src.data, 14)
        base = dither(src, pal)
        frame = new ImageData(new Uint8ClampedArray(base.data), W, H)
        // edge map for the lens: luminance gradient of the dithered-free source
        edges = new Uint8Array(W * H)
        const L = (i: number) => src.data[i * 4] * 0.3 + src.data[i * 4 + 1] * 0.59 + src.data[i * 4 + 2] * 0.11
        for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
          const i = y * W + x
          const gx = L(i + 1) - L(i - 1), gy = L(i + W) - L(i - W)
          edges[i] = gx * gx + gy * gy > 900 ? 1 : 0
        }
        if (fx.sprite) {
          const s = await rasterise(markupCache.get(`${world.id}|sprite`)!, W, H)
          if (cancelled) return
          sprite = dither(s, pal, 38, true)
          let x0 = W, x1 = 0, y0 = H, y1 = 0
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (sprite.data[(y * W + x) * 4 + 3]) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y) }
          spriteBox = { x0, x1, y0, y1 }
        }
        draw(performance.now())
      } catch { /* leave the canvas empty if the browser cannot rasterise */ }
    }

    const draw = (now: number) => {
      if (!base || !frame || !ctx) return
      const t = (now - t0) / 1000
      const b = base.data, f = frame.data
      f.set(b)
      if (live && !reduced) {
        // water rows shimmer: each row slides a pixel or two, in a slow wave
        if (fx.ripple) {
          const y0 = Math.floor(fx.ripple[0] * H), y1 = Math.min(H, Math.ceil(fx.ripple[1] * H))
          for (let y = y0; y < y1; y++) {
            const depth = (y - y0) / Math.max(1, y1 - y0)
            const dx = Math.round(Math.sin(y * 0.9 + t * 2.4) * (0.6 + depth * 1.6))
            if (!dx) continue
            for (let x = 0; x < W; x++) {
              const sx = Math.min(W - 1, Math.max(0, x - dx))
              const i = (y * W + x) * 4, j = (y * W + sx) * 4
              f[i] = b[j]; f[i + 1] = b[j + 1]; f[i + 2] = b[j + 2]
            }
          }
        }
        // a band of light (or shade) crossing the scene, re-dithered into the palette
        if (fx.sweep) {
          const period = fx.sweep === 'shadow' ? 12 : 7
          const cxB = ((t % period) / period) * (W * 1.4) - W * 0.2, half = W * 0.08
          const sign = fx.sweep === 'shadow' ? -1 : 1
          for (let y = 0; y < H; y++) for (let x = Math.max(0, Math.floor(cxB - half)); x < Math.min(W, cxB + half); x++) {
            const k = 1 - Math.abs(x - cxB) / half
            const i = (y * W + x) * 4, th = BAYER[(x & 3) + ((y & 3) << 2)] * 38, a = sign * 46 * k
            const c = nearest(pal, b[i] + a + th, b[i + 1] + a + th, b[i + 2] + a * 0.8 + th)
            f[i] = c[0]; f[i + 1] = c[1]; f[i + 2] = c[2]
          }
        }
        // firelight: the whole screen breathes brighter and dimmer
        if (fx.flicker) {
          const k = 0.86 + 0.14 * Math.abs(Math.sin(t * 7.3) * Math.sin(t * 3.1 + 1))
          for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
            const i = (y * W + x) * 4, th = BAYER[(x & 3) + ((y & 3) << 2)] * 30
            const c = nearest(pal, b[i] * k + th, b[i + 1] * k + th, b[i + 2] * k + th)
            f[i] = c[0]; f[i + 1] = c[1]; f[i + 2] = c[2]
          }
        }
        // the sprite (Tonle Sap's dragon boat) crossing on its own layer
        if (sprite && fx.sprite) {
          const sw = spriteBox.x1 - spriteBox.x0 + 1
          const off = Math.round(((t % fx.sprite) / fx.sprite) * (W + sw)) - sw - spriteBox.x0
          const bob = Math.round(Math.sin(t * 2) * 0.6)
          const s = sprite.data
          for (let y = spriteBox.y0; y <= spriteBox.y1; y++) for (let x = spriteBox.x0; x <= spriteBox.x1; x++) {
            const j = (y * W + x) * 4
            if (!s[j + 3]) continue
            const tx = x + off, ty = y + bob
            if (tx < 0 || tx >= W || ty < 0 || ty >= H) continue
            const i = (ty * W + tx) * 4
            f[i] = s[j]; f[i + 1] = s[j + 1]; f[i + 2] = s[j + 2]
          }
        }
        // single pixels glinting
        if (fx.twinkle) {
          const y0 = Math.floor(fx.twinkle[0] * H), y1 = Math.ceil(fx.twinkle[1] * H)
          for (let n = 0; n < 14; n++) {
            const ph = Math.sin(n * 91.7) * 43758.5
            const x = Math.floor((ph - Math.floor(ph)) * W), y = y0 + Math.floor(((ph * 7) % 1 + 1) % 1 * (y1 - y0))
            if (Math.sin(t * (1.3 + (n % 5) * 0.4) + n) > 0.6) { const i = (y * W + x) * 4; f[i] = 255; f[i + 1] = 241; f[i + 2] = 214 }
          }
        }
      }
      // the lens: how an edge-detection model sees the scene
      if (lens && edges) {
        const R = Math.max(10, Math.min(28, H * 0.42))
        for (let y = Math.max(0, Math.floor(lens.y - R - 1)); y < Math.min(H, lens.y + R + 1); y++)
          for (let x = Math.max(0, Math.floor(lens.x - R - 1)); x < Math.min(W, lens.x + R + 1); x++) {
            const d = Math.hypot(x - lens.x, y - lens.y), i = (y * W + x) * 4
            if (d < R) {
              const lum = (f[i] * 0.3 + f[i + 1] * 0.59 + f[i + 2] * 0.11) / 255
              const c: RGB = edges[y * W + x] ? cream : lum + BAYER[(x & 3) + ((y & 3) << 2)] * 0.5 > 0.42 ? accent : [16, 14, 34]
              f[i] = c[0]; f[i + 1] = c[1]; f[i + 2] = c[2]
            } else if (d < R + 1) { f[i] = accent[0]; f[i + 1] = accent[1]; f[i + 2] = accent[2] }
          }
      }
      ctx.putImageData(frame, 0, 0)
    }

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      if (!visible || document.hidden || now - last < 110) return
      last = now
      draw(now)
    }

    const onMove = (e: PointerEvent) => {
      const r = canvas.getBoundingClientRect()
      lens = { x: (e.clientX - r.left) / PX, y: (e.clientY - r.top) / PX }
      if (!live || reduced) draw(performance.now())
    }
    const onLeave = () => { lens = null; draw(performance.now()) }

    const ro = new ResizeObserver(() => { build() })
    ro.observe(wrap)
    const io = new IntersectionObserver(([en]) => { visible = en.isIntersecting })
    io.observe(wrap)
    if (live) {
      wrap.addEventListener('pointermove', onMove)
      wrap.addEventListener('pointerleave', onLeave)
      if (!reduced) raf = requestAnimationFrame(loop)
    }
    build()
    return () => {
      cancelled = true; cancelAnimationFrame(raf); ro.disconnect(); io.disconnect()
      wrap.removeEventListener('pointermove', onMove)
      wrap.removeEventListener('pointerleave', onLeave)
    }
  }, [world, live, captured])

  return (
    <div ref={wrapRef} className={className} style={{ position: 'absolute', inset: 0, cursor: live ? 'crosshair' : undefined }}>
      <canvas ref={canvasRef} aria-hidden style={{ display: 'block', width: '100%', height: '100%', imageRendering: 'pixelated' }} />
      {!captured && (
        <div ref={hostRef} hidden aria-hidden>
          <world.Scene layer="base" />
          {needSprite && <world.Scene layer="sprite" />}
        </div>
      )}
    </div>
  )
}
