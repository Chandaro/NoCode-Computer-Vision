import { useEffect, useRef } from 'react'

/*
 * Slow ambient layer behind every page. Each Khmer art world has its own:
 *   petals     lotus petals drifting down (Angkor Wat)
 *   roses      rose-sandstone petals and dust (Banteay Srei)
 *   leaves     jungle leaves tumbling (Ta Prohm)
 *   embers     sparks rising from the coconut-husk fire (Sbek Thom)
 *   gold       gold-leaf flakes glinting (Royal Palace)
 *   fireflies  fireflies over the lake (Tonle Sap)
 *   motes      stone dust drifting in sunlight (Bayon)
 * Deliberately faint and low-frame-rate so it never competes with the work.
 * Pauses when the tab is hidden; does nothing with reduced motion.
 */
export type ParticleMode = 'petals' | 'roses' | 'leaves' | 'embers' | 'gold' | 'fireflies' | 'motes'

const STYLE: Record<ParticleMode, { colors: string[]; shape: 'petal' | 'leaf' | 'dot' | 'flake'; n: number; vy: [number, number]; size: [number, number]; alpha: [number, number]; glow?: boolean }> = {
  petals:    { colors: ['234,143,162', '242,166,182', '247,201,210'], shape: 'petal', n: 14, vy: [0.12, 0.32], size: [6, 13], alpha: [0.22, 0.42] },
  roses:     { colors: ['240,146,124', '244,194,138', '242,176,154'], shape: 'petal', n: 16, vy: [0.1, 0.28], size: [5, 11], alpha: [0.22, 0.45] },
  leaves:    { colors: ['122,170,90', '166,209,122', '200,180,110'], shape: 'leaf', n: 13, vy: [0.18, 0.4], size: [7, 14], alpha: [0.25, 0.45] },
  embers:    { colors: ['255,179,71', '255,122,41', '255,213,154'], shape: 'dot', n: 34, vy: [-0.55, -0.25], size: [1, 2.4], alpha: [0.5, 0.9], glow: true },
  gold:      { colors: ['242,201,76', '246,214,112', '255,241,200'], shape: 'flake', n: 22, vy: [0.08, 0.22], size: [2, 4.5], alpha: [0.35, 0.7] },
  motes:     { colors: ['220,226,214', '201,194,164', '143,196,255'], shape: 'dot', n: 30, vy: [-0.06, 0.1], size: [0.8, 1.8], alpha: [0.25, 0.55] },
  fireflies: { colors: ['220,255,170', '255,230,140', '170,240,220'], shape: 'dot', n: 26, vy: [-0.08, 0.08], size: [1.2, 2.2], alpha: [0.5, 0.95], glow: true },
}

export default function AmbientPetals({ density = 1, mode = 'petals' }: { density?: number; mode?: ParticleMode }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c) return
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const ctx = c.getContext('2d')
    if (!ctx) return
    const st = STYLE[mode]
    let W = 0, H = 0, raf = 0, last = 0
    const dpr = Math.min(2, window.devicePixelRatio || 1)
    const resize = () => {
      W = window.innerWidth; H = window.innerHeight
      c.width = W * dpr; c.height = H * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    resize()
    const rnd = (a: number, b: number) => a + Math.random() * (b - a)
    const ps = Array.from({ length: Math.round(st.n * density) }, () => ({
      x: rnd(0, W), y: rnd(0, H), s: rnd(st.size[0], st.size[1]), r: rnd(0, Math.PI * 2),
      vr: rnd(-0.006, 0.006), vy: rnd(st.vy[0], st.vy[1]), sway: rnd(0, 100),
      c: st.colors[Math.floor(rnd(0, st.colors.length))], a: rnd(st.alpha[0], st.alpha[1]), ph: rnd(0, 100),
    }))
    const dust = Array.from({ length: Math.round(36 * density) }, () => ({
      x: rnd(0, W), y: rnd(0, H), r: rnd(0.6, 1.5), ph: rnd(0, 100), sp: rnd(0.0006, 0.0016),
    }))
    const petal = (s: number) => {
      ctx.beginPath()
      ctx.moveTo(0, -s)
      ctx.bezierCurveTo(s * 0.55, -s * 0.6, s * 0.5, s * 0.4, 0, s * 0.7)
      ctx.bezierCurveTo(-s * 0.5, s * 0.4, -s * 0.55, -s * 0.6, 0, -s)
      ctx.fill()
    }
    const leaf = (s: number) => {
      ctx.beginPath()
      ctx.moveTo(0, -s)
      ctx.quadraticCurveTo(s * 0.7, 0, 0, s)
      ctx.quadraticCurveTo(-s * 0.7, 0, 0, -s)
      ctx.fill()
      ctx.strokeStyle = 'rgba(20,40,20,0.35)'; ctx.lineWidth = 0.7
      ctx.beginPath(); ctx.moveTo(0, -s); ctx.lineTo(0, s); ctx.stroke()
    }
    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      if (document.hidden || now - last < 40) return
      last = now
      ctx.clearRect(0, 0, W, H)
      if (mode !== 'embers' && mode !== 'fireflies') {
        for (const d of dust) {
          const tw = Math.max(0, Math.sin(now * d.sp + d.ph))
          if (tw < 0.05) continue
          ctx.fillStyle = `rgba(${st.colors[0]},${0.45 * tw * tw})`
          ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2); ctx.fill()
        }
      }
      for (const p of ps) {
        p.y += p.vy; p.r += p.vr
        let x = p.x + Math.sin(now / 3000 + p.sway) * 18
        if (mode === 'fireflies') { x = p.x + Math.sin(now / 1700 + p.sway) * 30; p.y += Math.cos(now / 2100 + p.ph) * 0.25 }
        if (p.y > H + 20) { p.y = -20; p.x = rnd(0, W) }
        if (p.y < -20) { p.y = H + 20; p.x = rnd(0, W) }
        let a = p.a
        if (st.glow) a *= 0.45 + 0.55 * Math.max(0, Math.sin(now / 600 + p.ph))
        ctx.save(); ctx.translate(x, p.y); ctx.rotate(p.r)
        if (st.shape === 'dot') {
          if (st.glow) {
            const g = ctx.createRadialGradient(0, 0, 0, 0, 0, p.s * 5)
            g.addColorStop(0, `rgba(${p.c},${a * 0.5})`); g.addColorStop(1, `rgba(${p.c},0)`)
            ctx.fillStyle = g; ctx.beginPath(); ctx.arc(0, 0, p.s * 5, 0, Math.PI * 2); ctx.fill()
          }
          ctx.fillStyle = `rgba(${p.c},${a})`; ctx.beginPath(); ctx.arc(0, 0, p.s, 0, Math.PI * 2); ctx.fill()
        } else if (st.shape === 'flake') {
          const tw = 0.4 + 0.6 * Math.abs(Math.sin(now / 900 + p.ph))
          ctx.fillStyle = `rgba(${p.c},${a * tw})`
          ctx.beginPath(); ctx.moveTo(0, -p.s); ctx.lineTo(p.s * 0.6, 0); ctx.lineTo(0, p.s); ctx.lineTo(-p.s * 0.6, 0); ctx.closePath(); ctx.fill()
        } else {
          const gr = ctx.createLinearGradient(0, -p.s, 0, p.s)
          gr.addColorStop(0, `rgba(255,241,214,${a})`)
          gr.addColorStop(0.45, `rgba(${p.c},${a})`)
          gr.addColorStop(1, `rgba(${p.c},${a * 0.4})`)
          ctx.fillStyle = gr
          if (st.shape === 'leaf') { ctx.fillStyle = `rgba(${p.c},${a})`; leaf(p.s) } else petal(p.s)
        }
        ctx.restore()
      }
    }
    raf = requestAnimationFrame(loop)
    window.addEventListener('resize', resize)
    return () => { cancelAnimationFrame(raf); window.removeEventListener('resize', resize) }
  }, [density, mode])
  return <canvas ref={ref} aria-hidden style={{ position: 'fixed', inset: 0, width: '100vw', height: '100vh', pointerEvents: 'none', zIndex: 0 }} />
}
