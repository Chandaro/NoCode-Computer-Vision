import { useEffect, useState, type CSSProperties, type ReactNode } from 'react'

/*
 * Slow ambient layer behind every page. Each Khmer art world has its own:
 *   petals     lotus petals drifting down (Angkor Wat)
 *   roses      rose-sandstone petals and dust (Banteay Srei)
 *   leaves     jungle leaves tumbling (Ta Prohm)
 *   embers     sparks rising from the coconut-husk fire (Sbek Thom)
 *   gold       gold-leaf flakes glinting (Royal Palace, murals)
 *   fireflies  fireflies over the lake (Tonle Sap)
 *   motes      stone dust drifting in sunlight (Bayon)
 *
 * Performance: each particle shape is pre-rendered once into a small sprite and
 * moved with CSS transform/opacity animations, which run on the compositor, so
 * the main thread stays free. Nothing is rendered with reduced motion.
 */
export type ParticleMode = 'petals' | 'roses' | 'leaves' | 'embers' | 'gold' | 'fireflies' | 'motes'

type Shape = 'petal' | 'leaf' | 'dot' | 'glow' | 'flake'
const STYLE: Record<ParticleMode, { colors: string[]; shape: Shape; n: number; vy: [number, number]; size: [number, number]; alpha: [number, number]; twinkle?: boolean; dust: boolean }> = {
  petals:    { colors: ['234,143,162', '242,166,182', '247,201,210'], shape: 'petal', n: 12, vy: [0.2, 0.5], size: [6, 13], alpha: [0.25, 0.45], dust: true },
  roses:     { colors: ['240,146,124', '244,194,138', '242,176,154'], shape: 'petal', n: 14, vy: [0.18, 0.45], size: [5, 11], alpha: [0.25, 0.45], dust: true },
  leaves:    { colors: ['122,170,90', '166,209,122', '200,180,110'], shape: 'leaf', n: 11, vy: [0.3, 0.6], size: [7, 14], alpha: [0.3, 0.5], dust: true },
  embers:    { colors: ['255,179,71', '255,122,41', '255,213,154'], shape: 'glow', n: 26, vy: [-0.9, -0.4], size: [1.2, 2.4], alpha: [0.5, 0.9], twinkle: true, dust: false },
  gold:      { colors: ['242,201,76', '246,214,112', '255,241,200'], shape: 'flake', n: 18, vy: [0.12, 0.35], size: [2, 4.5], alpha: [0.35, 0.7], twinkle: true, dust: true },
  fireflies: { colors: ['220,255,170', '255,230,140', '170,240,220'], shape: 'glow', n: 20, vy: [-0.12, 0.12], size: [1.2, 2.2], alpha: [0.5, 0.95], twinkle: true, dust: false },
  motes:     { colors: ['220,226,214', '201,194,164', '143,196,255'], shape: 'dot', n: 24, vy: [-0.1, 0.15], size: [0.8, 1.8], alpha: [0.3, 0.6], dust: false },
}

function sprite(shape: Shape, rgb: string, size: number): HTMLCanvasElement {
  const pad = shape === 'glow' ? size * 5 : size
  const S = Math.ceil((size + pad) * 2)
  const c = document.createElement('canvas'); c.width = S; c.height = S
  const x = c.getContext('2d')!
  x.translate(S / 2, S / 2)
  if (shape === 'glow') {
    const g = x.createRadialGradient(0, 0, 0, 0, 0, size * 5)
    g.addColorStop(0, `rgba(${rgb},0.55)`); g.addColorStop(1, `rgba(${rgb},0)`)
    x.fillStyle = g; x.beginPath(); x.arc(0, 0, size * 5, 0, Math.PI * 2); x.fill()
    x.fillStyle = `rgb(${rgb})`; x.beginPath(); x.arc(0, 0, size, 0, Math.PI * 2); x.fill()
  } else if (shape === 'dot') {
    x.fillStyle = `rgb(${rgb})`; x.beginPath(); x.arc(0, 0, size, 0, Math.PI * 2); x.fill()
  } else if (shape === 'flake') {
    x.fillStyle = `rgb(${rgb})`; x.beginPath()
    x.moveTo(0, -size); x.lineTo(size * 0.6, 0); x.lineTo(0, size); x.lineTo(-size * 0.6, 0); x.closePath(); x.fill()
  } else if (shape === 'leaf') {
    x.fillStyle = `rgb(${rgb})`; x.beginPath()
    x.moveTo(0, -size); x.quadraticCurveTo(size * 0.7, 0, 0, size); x.quadraticCurveTo(-size * 0.7, 0, 0, -size); x.fill()
    x.strokeStyle = 'rgba(20,40,20,0.35)'; x.lineWidth = 0.7
    x.beginPath(); x.moveTo(0, -size); x.lineTo(0, size); x.stroke()
  } else {
    const g = x.createLinearGradient(0, -size, 0, size)
    g.addColorStop(0, 'rgb(255,241,214)'); g.addColorStop(0.45, `rgb(${rgb})`); g.addColorStop(1, `rgba(${rgb},0.4)`)
    x.fillStyle = g; x.beginPath()
    x.moveTo(0, -size)
    x.bezierCurveTo(size * 0.55, -size * 0.6, size * 0.5, size * 0.4, 0, size * 0.7)
    x.bezierCurveTo(-size * 0.5, size * 0.4, -size * 0.55, -size * 0.6, 0, -size)
    x.fill()
  }
  return c
}

const spriteCache = new Map<string, string>()
function spriteUrl(shape: Shape, rgb: string, size: number) {
  const key = `${shape}|${rgb}|${size.toFixed(1)}`
  let u = spriteCache.get(key)
  if (!u) { u = sprite(shape, rgb, size).toDataURL(); spriteCache.set(key, u) }
  return u
}

// Particles are plain elements animated with CSS transforms and opacity, so the
// browser's compositor moves them and the main thread does no per-frame work.
export default function AmbientPetals({ density = 1, mode = 'petals' }: { density?: number; mode?: ParticleMode }) {
  const [items, setItems] = useState<ReactNode[]>([])
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { setItems([]); return }
    const st = STYLE[mode]
    const rnd = (a: number, b: number) => a + Math.random() * (b - a)
    const out: ReactNode[] = []
    const n = Math.round(st.n * density)
    const rises = st.vy[1] < 0, hovers = mode === 'fireflies' || mode === 'motes'
    for (let i = 0; i < n; i++) {
      const size = rnd(st.size[0], st.size[1])
      const url = spriteUrl(st.shape, st.colors[i % st.colors.length], size)
      const speed = rnd(Math.abs(st.vy[0]), Math.abs(st.vy[1])) || 0.15
      const dur = hovers ? rnd(10, 18) : Math.min(90, Math.max(14, 900 / (speed * 20)))
      const style: CSSProperties = {
        left: `${rnd(0, 100)}%`, top: hovers ? `${rnd(5, 95)}%` : undefined,
        animationDuration: `${dur}s`, animationDelay: `${-rnd(0, dur)}s`,
        opacity: rnd(st.alpha[0], st.alpha[1]),
      }
      const cls = hovers ? 'kp kp-hover' : rises ? 'kp kp-rise' : 'kp kp-fall'
      out.push(
        <span key={i} className={cls} style={style}>
          <span className={`kp-sway${st.twinkle ? ' kp-twinkle' : ''}`} style={{ animationDuration: `${rnd(4, 7)}s, ${rnd(1.6, 3.2)}s`, animationDelay: `${-rnd(0, 6)}s` }}>
            <img src={url} alt="" className={st.shape === 'petal' || st.shape === 'leaf' || st.shape === 'flake' ? 'kp-spin' : undefined}
              style={{ animationDuration: `${rnd(8, 16)}s`, animationDirection: i % 2 ? 'reverse' : 'normal' }} />
          </span>
        </span>,
      )
    }
    if (st.dust) {
      const dust = spriteUrl('dot', st.colors[0], 1.2)
      for (let i = 0; i < Math.round(22 * density); i++) {
        out.push(<img key={`d${i}`} src={dust} alt="" className="kp-dust"
          style={{ left: `${rnd(0, 100)}%`, top: `${rnd(0, 100)}%`, animationDuration: `${rnd(4, 9)}s`, animationDelay: `${-rnd(0, 9)}s` }} />)
      }
    }
    setItems(out)
  }, [density, mode])
  return <div aria-hidden className="kp-layer">{items}</div>
}
