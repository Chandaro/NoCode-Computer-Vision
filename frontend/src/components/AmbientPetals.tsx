import { useEffect, useRef } from 'react'

/*
 * Slow ambient layer behind every page: lotus petals drifting down like
 * offerings on the moat, and specks of gold leaf that glint. Deliberately
 * faint and low-frame-rate so it never competes with the work on screen.
 * Pauses when the tab is hidden; does nothing with reduced motion.
 */
export default function AmbientPetals({ density = 1 }: { density?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    let W = 0, H = 0, raf = 0, last = 0
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const resize = () => {
      W = window.innerWidth; H = window.innerHeight
      c.width = W * dpr; c.height = H * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const rnd = (a: number, b: number) => a + Math.random() * (b - a)
    const COLORS = ['234,143,162', '242,166,182', '247,201,210', '234,143,162']
    const petals = Array.from({ length: Math.round(14 * density) }, () => ({
      x: rnd(0, W), y: rnd(-H, H), s: rnd(6, 13), r: rnd(0, Math.PI * 2),
      vr: rnd(-0.004, 0.004), vy: rnd(0.12, 0.32), sway: rnd(0, 100), c: COLORS[Math.floor(rnd(0, 4))],
      a: rnd(0.22, 0.42),
    }))
    const dust = Array.from({ length: Math.round(46 * density) }, () => ({
      x: rnd(0, W), y: rnd(0, H), r: rnd(0.6, 1.6), ph: rnd(0, 100), sp: rnd(0.0006, 0.0016),
    }))
    const petal = (s: number) => {
      ctx.beginPath()
      ctx.moveTo(0, -s)
      ctx.bezierCurveTo(s * 0.55, -s * 0.6, s * 0.5, s * 0.4, 0, s * 0.7)
      ctx.bezierCurveTo(-s * 0.5, s * 0.4, -s * 0.55, -s * 0.6, 0, -s)
      ctx.fill()
    }
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      if (document.hidden || now - last < 40) return
      last = now
      ctx.clearRect(0, 0, W, H)
      for (const d of dust) {
        const tw = Math.max(0, Math.sin(now * d.sp + d.ph))
        if (tw < 0.05) continue
        ctx.fillStyle = `rgba(240,177,90,${0.5 * tw * tw})`
        ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2); ctx.fill()
      }
      for (const p of petals) {
        p.y += p.vy; p.r += p.vr
        const x = p.x + Math.sin(now / 3000 + p.sway) * 18
        if (p.y > H + 20) { p.y = -20; p.x = rnd(0, W) }
        ctx.save(); ctx.translate(x, p.y); ctx.rotate(p.r)
        const gr = ctx.createLinearGradient(0, -p.s, 0, p.s)
        gr.addColorStop(0, `rgba(255,241,214,${p.a})`)
        gr.addColorStop(0.45, `rgba(${p.c},${p.a})`)
        gr.addColorStop(1, `rgba(${p.c},${p.a * 0.4})`)
        ctx.fillStyle = gr
        petal(p.s)
        ctx.restore()
      }
    }
    raf = requestAnimationFrame(loop)
    window.addEventListener('resize', resize)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [density])
  return <canvas ref={ref} aria-hidden style={{ position: 'fixed', inset: 0, width: '100vw', height: '100vh', pointerEvents: 'none', zIndex: 0 }} />
}
