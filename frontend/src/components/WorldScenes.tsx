import { useId, type ReactNode } from 'react'

/*
 * Scenes for the Khmer art worlds, drawn from reference photographs
 * (Wikimedia Commons) of each place or art form. viewBox is 600x200 and the
 * art sits on the right two thirds, so banner titles stay readable.
 */

type P = { className?: string; layer?: 'base' | 'sprite' }
const useSid = () => useId().replace(/:/g, '')

/* ── helpers ─────────────────────────────────────────────────────────── */

// lotus-bud (corn-cob) tower with stepped tiers
function budTower(cx: number, base: number, top: number, hw: number, fill: string, tierFill?: string) {
  const h = base - top
  const d = `M${cx - hw} ${base} C ${cx - hw} ${base - h * 0.55}, ${cx - hw * 0.42} ${top + h * 0.18}, ${cx} ${top} C ${cx + hw * 0.42} ${top + h * 0.18}, ${cx + hw} ${base - h * 0.55}, ${cx + hw} ${base} Z`
  return (
    <g>
      <path d={d} fill={fill} />
      {tierFill && Array.from({ length: 6 }, (_, i) => {
        const y = base - h * (0.18 + i * 0.13)
        const w = hw * Math.pow(1 - (i * 0.13 + 0.18), 0.55) * 0.96
        return <rect key={i} x={cx - w} y={y} width={w * 2} height="1.2" fill={tierFill} />
      })}
      <rect x={cx - 0.8} y={top - 7} width="1.6" height="8" fill={fill} />
    </g>
  )
}

function tree(cx: number, base: number, r: number, fill: string) {
  return (
    <g fill={fill}>
      <rect x={cx - r * 0.08} y={base - r * 1.6} width={r * 0.16} height={r * 1.6} />
      <circle cx={cx} cy={base - r * 1.7} r={r} /><circle cx={cx - r * 0.7} cy={base - r * 1.35} r={r * 0.7} /><circle cx={cx + r * 0.75} cy={base - r * 1.4} r={r * 0.72} />
    </g>
  )
}

function sugarPalm(x: number, base: number, h: number, fill: string) {
  const top = base - h
  return (
    <g fill={fill} stroke={fill}>
      <path d={`M${x} ${base} Q ${x + 3} ${base - h * 0.5} ${x + 1} ${top}`} fill="none" strokeWidth="2" />
      {Array.from({ length: 14 }, (_, i) => {
        const a = (i / 14) * Math.PI * 2
        return <path key={i} d={`M${x + 1} ${top} l ${Math.cos(a) * 13} ${Math.sin(a) * 11}`} strokeWidth="2.2" strokeLinecap="round" />
      })}
      <circle cx={x + 1} cy={top} r="5" stroke="none" />
    </g>
  )
}

/* ── Angkor Wat: five towers at dawn over the western pond ───────────── */
export function AngkorScene({ className }: P) {
  const id = useSid()
  const ink = '#0b0918'
  const land = (
    <g fill={ink}>
      {budTower(430, 128, 30, 15, ink)}
      {budTower(395, 128, 62, 11, ink)}{budTower(465, 128, 62, 11, ink)}
      {budTower(366, 130, 74, 10, ink)}{budTower(494, 130, 74, 10, ink)}
      <rect x="352" y="124" width="156" height="10" /><rect x="318" y="132" width="224" height="8" />
      <rect x="250" y="139" width="370" height="7" />
      {[270, 300, 560, 590].map(x => <rect key={x} x={x - 7} y="130" width="14" height="10" />)}
      {tree(250, 146, 15, ink)}{tree(600, 146, 18, ink)}{tree(330, 146, 9, ink)}{tree(545, 146, 11, ink)}
      {sugarPalm(286, 146, 70, ink)}{sugarPalm(575, 146, 82, ink)}{sugarPalm(522, 146, 58, ink)}
      <rect x="220" y="145" width="380" height="6" />
    </g>
  )
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#1d1840" /><stop offset="0.42" stopColor="#5a3570" />
          <stop offset="0.62" stopColor="#d0607a" /><stop offset="0.74" stopColor="#f6a85a" />
        </linearGradient>
        <radialGradient id={`${id}u`}><stop offset="0" stopColor="#fff3d8" /><stop offset="0.4" stopColor="#f7b863" /><stop offset="1" stopColor="#f7b863" stopOpacity="0" /></radialGradient>
        <linearGradient id={`${id}w`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#e08a6a" /><stop offset="0.5" stopColor="#5a3570" /><stop offset="1" stopColor="#120f26" /></linearGradient>
      </defs>
      <rect width="600" height="151" fill={`url(#${id}s)`} />
      <circle className="kw-pulse" cx="430" cy="118" r="70" fill={`url(#${id}u)`} />
      {land}
      <rect y="151" width="600" height="49" fill={`url(#${id}w)`} />
      <g opacity="0.45" transform="translate(0 302) scale(1 -1)"><g className="kw-ripple">{land}</g></g>
      {[0, 1, 2, 3, 4, 5, 6].map(i => <rect key={i} className="kw-glint" x={350 + i * 24} y={157 + (i % 3) * 8} width={20 - (i % 3) * 5} height="1.4" fill="#ffd9a0" style={{ animationDelay: `${i * 0.6}s` }} />)}
      {/* reeds and lotus in the foreground */}
      <g stroke="#0b0918" strokeWidth="1.6" strokeLinecap="round">
        {[232, 238, 244, 596, 590].map((x, i) => <path key={x} d={`M${x} 200 q ${i % 2 ? 4 : -3} -22 ${i % 2 ? 2 : -1} -${30 + (i % 3) * 8}`} fill="none" />)}
      </g>
      <g fill="#f2a6b6"><path d="M318 186 C 314 179, 314 172, 318 164 C 322 172, 322 179, 318 186 Z" /><path d="M548 194 C 545 188, 545 181, 548 175 C 551 181, 551 188, 548 194 Z" /></g>
      <g fill="#14302a"><ellipse cx="318" cy="187" rx="16" ry="3" /><ellipse cx="548" cy="195" rx="18" ry="3" /><ellipse cx="440" cy="194" rx="20" ry="2.6" /></g>
    </svg>
  )
}

/* ── Banteay Srei: layered pink sandstone pediments with flame lobes ─── */
// Banteay Srei frames are multi-lobed (ogee) arches: the band rises from
// outward-curling spirals, swells, and closes in a pointed crown, repeated
// as nested frames around a tympanum crowded with figures.
function lobedArch(cx: number, base: number, hw: number, h: number) {
  const s = hw / 100
  return `M${cx - hw} ${base}
    C ${cx - hw} ${base - 22 * s}, ${cx - hw * 0.78} ${base - 30 * s}, ${cx - hw * 0.86} ${base - 44 * s}
    C ${cx - hw * 0.96} ${base - 62 * s}, ${cx - hw * 0.62} ${base - 74 * s}, ${cx - hw * 0.62} ${base - h * 0.62}
    C ${cx - hw * 0.62} ${base - h * 0.86}, ${cx - hw * 0.22} ${base - h * 0.84}, ${cx} ${base - h}
    C ${cx + hw * 0.22} ${base - h * 0.84}, ${cx + hw * 0.62} ${base - h * 0.86}, ${cx + hw * 0.62} ${base - h * 0.62}
    C ${cx + hw * 0.62} ${base - 74 * s}, ${cx + hw * 0.96} ${base - 62 * s}, ${cx + hw * 0.86} ${base - 44 * s}
    C ${cx + hw * 0.78} ${base - 30 * s}, ${cx + hw} ${base - 22 * s}, ${cx + hw} ${base}`
}

function reliefFigures(x0: number, x1: number, y: number, size: number, color: string, key: string) {
  // a frieze of small carved figures: heads with crowns, shoulders, raised arms
  const out: ReactNode[] = []
  const n = Math.max(2, Math.floor((x1 - x0) / (size * 1.25)))
  for (let i = 0; i < n; i++) {
    const x = x0 + (i + 0.5) * (x1 - x0) / n, k = size / 10
    const arm = i % 2 ? -1 : 1
    out.push(
      <g key={`${key}${i}`} fill={color}>
        <path d={`M${x} ${y - 13 * k} l ${-1.6 * k} ${3 * k} h ${3.2 * k} Z`} />
        <circle cx={x} cy={y - 8.5 * k} r={2.4 * k} />
        <path d={`M${x - 4 * k} ${y} C ${x - 4 * k} ${y - 4 * k}, ${x - 2.6 * k} ${y - 6 * k}, ${x} ${y - 6 * k} C ${x + 2.6 * k} ${y - 6 * k}, ${x + 4 * k} ${y - 4 * k}, ${x + 4 * k} ${y} Z`} />
        <path d={`M${x + 3 * arm * k} ${y - 5 * k} l ${3.5 * arm * k} ${-4 * k}`} stroke={color} strokeWidth={1.3 * k} strokeLinecap="round" />
      </g>,
    )
  }
  return out
}

function flameFringe(d: string, color: string, id: string) {
  // flame tongues standing along the outer edge of a frame
  return (
    <g id={id}>
      <path d={d} fill="none" stroke="#5a2a24" strokeWidth="22" strokeDasharray="5 4" strokeLinecap="butt" />
      <path d={d} fill="none" stroke={color} strokeWidth="20" strokeDasharray="3.5 5.5" strokeLinecap="butt" />
    </g>
  )
}

export function BanteayScene({ className }: P) {
  const id = useSid()
  const cx = 425
  const outer = lobedArch(cx, 150, 132, 136)
  const mid = lobedArch(cx, 150, 104, 112)
  const inner = lobedArch(cx, 150, 78, 90)
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#4f7fb8" /><stop offset="1" stopColor="#a9c6dc" /></linearGradient>
        <linearGradient id={`${id}st`} x1="0" y1="0" x2="0.3" y2="1"><stop offset="0" stopColor="#d98a6e" /><stop offset="1" stopColor="#a65a48" /></linearGradient>
        <linearGradient id={`${id}l`} x1="0" x2="1"><stop offset="0" stopColor="#fff" stopOpacity="0" /><stop offset="0.5" stopColor="#ffe9dc" stopOpacity="0.28" /><stop offset="1" stopColor="#fff" stopOpacity="0" /></linearGradient>
        <clipPath id={`${id}c`}><rect x="250" y="0" width="350" height="200" /></clipPath>
        <clipPath id={`${id}ty`}><path d={`${inner} Z`} /></clipPath>
      </defs>
      <rect width="600" height="200" fill={`url(#${id}s)`} />
      {/* jungle canopy behind, as at the site */}
      <g fill="#3f6a3a" opacity="0.85"><circle cx="270" cy="40" r="34" /><circle cx="310" cy="20" r="26" /><circle cx="590" cy="30" r="36" /><circle cx="555" cy="8" r="24" /></g>
      <g clipPath={`url(#${id}c)`}>
        {/* stone mass of the gopura behind the pediment */}
        <path d="M282 170 V 52 L 296 40 H 554 L 568 52 V 170 Z" fill="#8f4a3d" />
        <path d="M282 52 H 568 M 282 92 H 568 M 282 132 H 568" stroke="#6e3428" strokeWidth="1" opacity="0.5" />
        {/* three nested lobed frames, each with a flame fringe */}
        {flameFringe(outer, '#e8a088', `${id}f1`)}
        <path d={`${outer} Z`} fill="#b86a55" stroke="#5a2a24" strokeWidth="2" />
        {flameFringe(mid, '#f0ad94', `${id}f2`)}
        <path d={`${mid} Z`} fill="#c77a62" stroke="#5a2a24" strokeWidth="2" />
        <path d={`${inner} Z`} fill={`url(#${id}st)`} stroke="#5a2a24" strokeWidth="2.4" />
        {/* tympanum: tiers of carved figures around a central deity */}
        <g clipPath={`url(#${id}ty)`}>
          <rect x="340" y="50" width="170" height="100" fill="#8a4638" opacity="0.35" />
          {reliefFigures(348, 502, 148, 9, '#5e2a20', 'r1')}
          {reliefFigures(356, 494, 134, 8.5, '#5e2a20', 'r2')}
          {reliefFigures(366, 484, 120, 8, '#5e2a20', 'r3')}
          {reliefFigures(378, 472, 106, 7.5, '#5e2a20', 'r4')}
          {/* foliage scrolls filling the gaps between the figures */}
          {Array.from({ length: 18 }, (_, i) => {
            const x = 352 + (i * 37) % 146, y = 96 + ((i * 23) % 50)
            return <path key={i} d={`M${x} ${y} c 0 -3 4 -4 5 -1 c 1 2 -1 3 -2.4 2.4`} fill="none" stroke="#f0ad94" strokeWidth="0.8" opacity="0.7" />
          })}
          <path d="M346 141 H 504 M 352 127 H 498 M 362 113 H 488 M 374 99 H 476" stroke="#e9a68c" strokeWidth="1" opacity="0.8" />
          <g fill="#6e3428"><circle cx={cx} cy="74" r="5" /><path d={`M${cx} 62 l -3 6 h 6 Z`} /><path d={`M${cx - 9} 88 C ${cx - 9} 80, ${cx - 5} 78, ${cx} 78 C ${cx + 5} 78, ${cx + 9} 80, ${cx + 9} 88 Z`} /></g>
        </g>
        {/* crown finial */}
        <path d={`M${cx} 2 C ${cx + 6} 8, ${cx + 4} 14, ${cx} 16 C ${cx - 4} 14, ${cx - 6} 8, ${cx} 2 Z`} fill="#e8a088" />
        {/* spirals (makara curls) where the frames end */}
        {[[cx - 132, 150, -1], [cx + 132, 150, 1], [cx - 104, 150, -1], [cx + 104, 150, 1]].map(([x, y, d], i) => (
          <path key={i} d={`M${x} ${y} c ${10 * d} -2 ${14 * d} -12 ${6 * d} -18 c ${-6 * d} -4 ${-12 * d} 2 ${-8 * d} 7 c ${3 * d} 3 ${7 * d} 0 ${5 * d} -3`} fill="none" stroke="#f0ad94" strokeWidth="3.5" strokeLinecap="round" />
        ))}
        {/* lintel: garlands looping from a central figure, medallions in the loops */}
        <rect x="292" y="150" width="266" height="34" fill="#b86a55" stroke="#5a2a24" strokeWidth="1.5" />
        <path d="M300 156 Q 340 188 380 158 Q 404 150 425 158 Q 446 150 470 158 Q 510 188 550 156" fill="none" stroke="#5a2a24" strokeWidth="8" />
        <path d="M300 156 Q 340 188 380 158 Q 404 150 425 158 Q 446 150 470 158 Q 510 188 550 156" fill="none" stroke="#e8a088" strokeWidth="5" strokeDasharray="2 2" />
        {[318, 362, 488, 532].map(x => <path key={x} d={`M${x} ${x < 425 ? 170 : 170} v 8 l -3 4 h 6 l -3 -4`} stroke="#5a2a24" strokeWidth="1.2" fill="#e8a088" />)}
        {[340, 510].map(x => <g key={x} transform={`translate(${x} 164)`}><circle r="7" fill="#d98a6e" stroke="#5a2a24" strokeWidth="1.2" /><path d="M-4 0 c 0 -3 4 -4 5 -1 c 1 2 -1 3 -2.5 2.5" fill="none" stroke="#5a2a24" strokeWidth="1.2" /></g>)}
        <g fill="#6e3428"><circle cx={cx} cy="160" r="4" /><path d={`M${cx - 7} 176 C ${cx - 7} 168, ${cx - 4} 166, ${cx} 166 C ${cx + 4} 166, ${cx + 7} 168, ${cx + 7} 176 Z`} /></g>
        <path d="M292 184 H 558" stroke="#5a2a24" strokeWidth="2" />
        <rect x="292" y="184" width="266" height="16" fill="#8a4638" />
        {Array.from({ length: 22 }, (_, i) => <path key={i} d={`M${296 + i * 12} 196 c 2 -6 6 -6 8 0`} fill="none" stroke="#e8a088" strokeWidth="1.2" />)}
        <rect className="kw-rake" x="200" y="0" width="80" height="200" fill={`url(#${id}l)`} />
      </g>
    </svg>
  )
}

/* ── Ta Prohm: grey galleries swallowed by silk-cotton roots ──────────── */
export function TaProhmScene({ className }: P) {
  const id = useSid()
  const blocks: ReactNode[] = []
  for (let r = 0; r < 7; r++) for (let c = 0; c < 10; c++) {
    const w = 36, x = 250 + c * w + (r % 2 ? 18 : 0), y = 52 + r * 21
    const g = 92 + ((r * 7 + c * 13) % 5) * 8
    blocks.push(<rect key={`${r}-${c}`} x={x} y={y} width={w - 2} height="19" rx="1.5" fill={`rgb(${g - 6},${g + 2},${g - 14})`} />)
    if ((r * 3 + c) % 7 === 0) blocks.push(<ellipse key={`m${r}-${c}`} cx={x + 10} cy={y + 12} rx="8" ry="4" fill="#5f7a3a" opacity="0.55" />)
  }
  const root = '#bdb29c', rootDark = '#6f6656'
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#173320" /><stop offset="1" stopColor="#24402a" /></linearGradient>
        <radialGradient id={`${id}sun`} cx="0.72" cy="0.05" r="0.7"><stop offset="0" stopColor="#f2f7c8" stopOpacity="0.5" /><stop offset="1" stopColor="#f2f7c8" stopOpacity="0" /></radialGradient>
        <linearGradient id={`${id}r`} x1="0" x2="1"><stop offset="0" stopColor={rootDark} /><stop offset="0.45" stopColor={root} /><stop offset="1" stopColor="#a49b86" /></linearGradient>
      </defs>
      <rect width="600" height="200" fill={`url(#${id}s)`} />
      <g>{blocks}</g>
      {/* gopura doorway with carved lintel */}
      <path d="M380 200 V 112 L 392 100 H 428 L 440 112 V 200 Z" fill="#07120c" />
      <rect x="372" y="92" width="76" height="10" fill="#a9a98f" /><rect x="376" y="102" width="68" height="4" fill="#7d7f68" />
      {[[296, 84], [530, 126], [334, 156], [500, 74]].map(([x, y], i) => (
        <g key={i} transform={`translate(${x} ${y})`} fill="none" stroke="#c9ccb0" strokeWidth="1.1" opacity="0.8">
          <circle r="7.5" /><circle r="3.5" /><path d="M0 -7.5 V 7.5 M -7.5 0 H 7.5" />
        </g>
      ))}
      {/* the roots: broad, pale, braided, poured over the wall */}
      <g className="kw-roots-fill" fill={`url(#${id}r)`} stroke={rootDark} strokeWidth="1">
        <path d="M452 0 C 448 40, 430 60, 438 100 C 444 128, 424 150, 420 200 L 446 200 C 448 160, 466 136, 462 104 C 458 70, 476 40, 478 0 Z" />
        <path d="M478 0 C 488 40, 520 64, 514 110 C 510 140, 528 170, 548 200 L 572 200 C 546 166, 536 140, 540 108 C 546 64, 512 36, 504 0 Z" />
        <path d="M440 20 C 400 34, 372 46, 352 74 C 334 100, 330 150, 310 200 L 326 200 C 344 152, 350 108, 368 84 C 388 58, 418 46, 446 38 Z" />
        <path d="M504 30 C 548 46, 584 76, 600 120 V 150 C 586 104, 556 70, 506 50 Z" />
        <path d="M438 100 C 456 126, 474 156, 470 200 H 482 C 486 158, 468 124, 446 96 Z" />
      </g>
      {/* grooves and braided strands along the roots */}
      <g fill="none" stroke="#5a5244" strokeWidth="1.1" opacity="0.75" strokeLinecap="round">
        <path d="M458 4 C 452 44, 436 64, 444 102 C 448 130, 430 156, 428 200" /><path d="M468 4 C 464 44, 450 70, 454 104 C 458 134, 440 160, 438 200" />
        <path d="M486 4 C 498 44, 528 68, 522 110 C 518 142, 534 172, 556 200" /><path d="M496 4 C 506 44, 536 70, 530 110 C 526 142, 544 170, 564 200" />
        <path d="M436 28 C 398 44, 372 60, 358 86 C 342 112, 336 156, 316 200" />
        <path d="M520 44 C 556 64, 584 92, 598 132" />
      </g>
      <g fill="none" stroke="#e6dfcf" strokeWidth="0.9" opacity="0.5">
        <path d="M462 10 C 456 50, 444 72, 450 102" /><path d="M492 10 C 504 50, 530 74, 526 112" />
      </g>
      {/* thin aerial roots hanging straight down */}
      <g stroke="#a89d86" strokeWidth="1.6" opacity="0.85">
        {[392, 404, 536, 548, 578].map((x, i) => <path key={x} d={`M${x} ${20 + i * 4} V ${120 + (i % 3) * 30}`} />)}
      </g>
      <g fill="#1e4a2a" opacity="0.9">
        <circle cx="300" cy="-6" r="30" /><circle cx="350" cy="4" r="22" />
      </g>
      <g fill="#14321f">
        <circle cx="470" cy="-14" r="64" /><circle cx="548" cy="-4" r="52" /><circle cx="392" cy="-10" r="46" /><circle cx="600" cy="26" r="40" />
      </g>
      <rect className="kw-dapple" width="600" height="200" fill={`url(#${id}sun)`} />
    </svg>
  )
}

/* ── Sbek Thom: lace-cut leather panel before the fire-lit screen ─────── */
export function SbekScene({ className }: P) {
  const id = useSid()
  const shield = 'M420 14 C 470 14, 506 30, 508 70 C 510 112, 500 146, 420 150 C 340 146, 330 112, 332 70 C 334 30, 370 14, 420 14 Z'
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <radialGradient id={`${id}scr`} cx="0.6" cy="0.6" r="0.75">
          <stop offset="0" stopColor="#fff0cf" /><stop offset="0.3" stopColor="#f7b25e" /><stop offset="0.7" stopColor="#8a3c12" /><stop offset="1" stopColor="#1a0c05" />
        </radialGradient>
        {/* filigree: snail-curls (kbach hien) and leaves cut out of the hide */}
        <pattern id={`${id}fil`} width="14" height="14" patternUnits="userSpaceOnUse">
          <rect width="14" height="14" fill="#fff" />
          <path d="M3 8 c 0 -3 4 -4 5 -1 c 1 2 -1 3.5 -2.6 2.6" fill="none" stroke="#000" strokeWidth="1.9" strokeLinecap="round" />
          <path d="M11 1 c 2 1.5 2 4 0 5.5 c -2 -1.5 -2 -4 0 -5.5 Z M 11 8 c 2 1.5 2 4 0 5.5 c -2 -1.5 -2 -4 0 -5.5 Z" fill="#000" />
          <circle cx="2" cy="2" r="1.1" fill="#000" /><circle cx="6" cy="12.5" r="1.1" fill="#000" />
        </pattern>
        <mask id={`${id}m`}>
          <rect width="600" height="200" fill={`url(#${id}fil)`} />
          <g fill="none" stroke="#fff">
            <path d={shield} strokeWidth="7" />
            <path d="M420 22 C 462 22, 494 36, 496 70 C 498 106, 490 136, 420 140 C 350 136, 342 106, 344 70 C 346 36, 378 22, 420 22 Z" strokeWidth="2.5" />
            {/* solid leather lines that carry the image: a pavilion under a bodhi tree */}
            <path d="M372 132 V 100 L 420 70 L 468 100 V 132" strokeWidth="5" />
            <path d="M362 102 L 420 62 L 478 102" strokeWidth="6" />
            <path d="M384 132 V 104 M 456 132 V 104" strokeWidth="4" />
            <path d="M362 102 c -8 -2 -10 -10 -5 -14 M 478 102 c 8 -2 10 -10 5 -14" strokeWidth="3" />
            <path d="M414 62 L 420 40 L 426 62" strokeWidth="3" />
            <path d="M350 136 H 490" strokeWidth="5" />
            <path d="M398 136 V 110 Q 420 96 442 110 V 136" strokeWidth="3" />
            <path d="M420 40 V 24 M 404 30 Q 420 18 436 30" strokeWidth="2.5" />
          </g>
        </mask>
      </defs>
      <rect width="600" height="200" fill="#0e0a07" />
      <rect x="236" y="0" width="364" height="176" fill={`url(#${id}scr)`} />
      <rect x="236" y="0" width="364" height="176" fill="none" stroke="#2a1408" strokeWidth="6" />
      <g className="kw-sway" style={{ transformOrigin: '420px 200px' }}>
        <path d={shield} fill="#2a1408" mask={`url(#${id}m)`} />
        <rect x="404" y="148" width="3.5" height="60" fill="#4a2e16" /><rect x="432" y="148" width="3.5" height="60" fill="#4a2e16" />
      </g>
      <g className="kw-fire">
        <path d="M396 200 C 392 186, 400 180, 398 168 C 406 178, 412 182, 410 200 Z" fill="#ffb347" />
        <path d="M418 200 C 416 190, 424 184, 424 172 C 432 184, 434 190, 430 200 Z" fill="#ff7a29" />
        <path d="M440 200 C 438 192, 444 188, 444 180 C 450 188, 452 194, 448 200 Z" fill="#ffd59a" />
      </g>
      <rect y="176" width="600" height="24" fill="#120a05" />
    </svg>
  )
}

/* ── Royal Palace: Throne Hall with tiered spire and layered gables ───── */
function gable(cx: number, base: number, w: number, h: number, roof: string, trim: string, face: string) {
  return (
    <g>
      {/* roof planes either side */}
      <path d={`M${cx - w * 1.5} ${base} L ${cx - w * 0.35} ${base - h * 0.7} L ${cx} ${base - h} L ${cx} ${base} Z`} fill={roof} />
      <path d={`M${cx + w * 1.5} ${base} L ${cx + w * 0.35} ${base - h * 0.7} L ${cx} ${base - h} L ${cx} ${base} Z`} fill={roof} opacity="0.9" />
      <path d={`M${cx - w * 1.5} ${base} L ${cx - w * 0.35} ${base - h * 0.7} M ${cx + w * 1.5} ${base} L ${cx + w * 0.35} ${base - h * 0.7}`} stroke={trim} strokeWidth="3" />
      {/* gable face */}
      <path d={`M${cx - w} ${base} L ${cx} ${base - h} L ${cx + w} ${base} Z`} fill={face} stroke="#f6d670" strokeWidth="2.4" />
      <path d={`M${cx - w * 0.62} ${base - 3} L ${cx} ${base - h * 0.72} L ${cx + w * 0.62} ${base - 3} Z`} fill="none" stroke="#b9861e" strokeWidth="1.2" />
      <circle cx={cx} cy={base - h * 0.34} r={w * 0.12} fill="#b9861e" />
      {/* chovear: the long horn at the apex, and hooked hang hong at the eaves */}
      <path d={`M${cx} ${base - h} q 4 -8 12 -14`} fill="none" stroke="#f6d670" strokeWidth="2.6" strokeLinecap="round" />
      <path d={`M${cx - w} ${base} c -6 0 -8 -6 -4 -10 M ${cx + w} ${base} c 6 0 8 -6 4 -10`} fill="none" stroke="#f6d670" strokeWidth="2.2" strokeLinecap="round" />
    </g>
  )
}

export function PalaceScene({ className }: P) {
  const id = useSid()
  const roof = '#d98a24', trim = '#2f8a5a', face = '#e8b43a'
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#0f1638" /><stop offset="0.7" stopColor="#2c3478" /><stop offset="1" stopColor="#7a5b7a" /></linearGradient>
        <linearGradient id={`${id}g`} x1="0" x2="1"><stop offset="0" stopColor="#fff6c8" stopOpacity="0" /><stop offset="0.5" stopColor="#fff6c8" stopOpacity="0.45" /><stop offset="1" stopColor="#fff6c8" stopOpacity="0" /></linearGradient>
        <linearGradient id={`${id}sp`} x1="0" x2="1"><stop offset="0" stopColor="#b9861e" /><stop offset="0.5" stopColor="#f6d670" /><stop offset="1" stopColor="#c9962a" /></linearGradient>
        <clipPath id={`${id}c`}><rect x="250" y="0" width="350" height="200" /></clipPath>
      </defs>
      <rect width="600" height="200" fill={`url(#${id}s)`} />
      {Array.from({ length: 18 }, (_, i) => <circle key={i} className="kw-star" cx={250 + (i * 97) % 350} cy={8 + (i * 41) % 64} r={i % 3 ? 0.9 : 1.4} fill="#f0ead8" style={{ animationDelay: `${(i % 5) * 0.8}s` }} />)}
      <g clipPath={`url(#${id}c)`}>
        {/* tiered spire */}
        {Array.from({ length: 9 }, (_, i) => {
          const y = 88 - i * 9, w = 15 - i * 1.4
          return <path key={i} d={`M${425 - w} ${y} Q 425 ${y - 9} ${425 + w} ${y} Z`} fill={`url(#${id}sp)`} />
        })}
        <rect x="424" y="4" width="2" height="12" fill="#f6d670" />
        <rect x="410" y="86" width="30" height="10" fill="#e8b43a" />
        {gable(425, 116, 34, 26, roof, trim, face)}
        {gable(425, 140, 52, 30, roof, trim, face)}
        {gable(425, 164, 72, 32, roof, trim, face)}
        {/* walls and gilded columns */}
        <rect x="335" y="164" width="180" height="24" fill="#f3ecdd" />
        {Array.from({ length: 9 }, (_, i) => <rect key={i} x={340 + i * 20} y="166" width="5" height="22" fill="#d9b24a" />)}
        <rect x="318" y="188" width="214" height="5" fill="#cdbfa4" /><rect x="304" y="193" width="242" height="7" fill="#a8977c" />
        <path d="M400 200 L 410 188 H 440 L 450 200 Z" fill="#b8322a" />
        <rect className="kw-shine" x="150" y="0" width="60" height="200" fill={`url(#${id}g)`} />
      </g>
      {/* lamp posts */}
      {[300, 556].map(x => <g key={x}><rect x={x} y="120" width="2" height="80" fill="#2f8a5a" /><circle className="kw-lamp" cx={x + 1} cy="118" r="4" fill="#ffe7a0" /></g>)}
    </svg>
  )
}

function boatOnly() {
  return (
    <g transform="translate(0 166)">
      <path d="M0 0 H 160 C 172 0, 178 -12, 178 -26 C 184 -18, 184 -6, 176 2 C 170 8, 150 9, 130 9 H 10 C 2 9, -6 4, -10 -10 C -4 -4, -2 -1, 0 0 Z" fill="#b8322a" stroke="#f2c94c" strokeWidth="1.4" />
      <circle cx="177" cy="-23" r="1.5" fill="#f2c94c" />
      <path d="M4 3 H 168" stroke="#f2c94c" strokeWidth="1" />
      <g stroke="#f6e9d6" strokeWidth="1.6" strokeLinecap="round">
        {Array.from({ length: 14 }, (_, i) => <path key={i} d={`M${14 + i * 10.5} -2 l -6 12`} />)}
      </g>
      <g fill="#1b1424">{Array.from({ length: 14 }, (_, i) => <circle key={i} cx={16 + i * 10.5} cy="-5" r="2.4" />)}</g>
    </g>
  )
}

/* ── Tonle Sap: tall stilt houses, long boats, the dragon-boat race ───── */
export function TonleSapScene({ className, layer }: P) {
  const id = useSid()
  const house = (x: number, base: number, lift: number, w: number, wall: string, trim: string) => {
    const y = base - lift
    return (
      <g>
        {[0.08, 0.36, 0.64, 0.92].map(p => <rect key={p} x={x + w * p - 1} y={y} width="2.2" height={lift + 6} fill="#3a2418" />)}
        <path d={`M${x + w * 0.08} ${y + 8} L ${x + w * 0.36} ${base} M ${x + w * 0.64} ${y + 8} L ${x + w * 0.92} ${base}`} stroke="#3a2418" strokeWidth="1" />
        <rect x={x - 4} y={y - 2} width={w + 8} height="3" fill="#5a3a24" />
        <rect x={x} y={y - 22} width={w} height="20" fill={wall} />
        {[0.2, 0.55].map(p => <rect key={p} x={x + w * p} y={y - 17} width={w * 0.18} height="9" fill={trim} className="kw-lamp" />)}
        <path d={`M${x - 6} ${y - 22} L ${x + w / 2} ${y - 38} L ${x + w + 6} ${y - 22} Z`} fill="#8a8f98" />
        {Array.from({ length: 6 }, (_, i) => <path key={i} d={`M${x - 6 + i * (w + 12) / 6} ${y - 22} L ${x + w / 2} ${y - 38}`} stroke="#6a6f78" strokeWidth="0.8" />)}
      </g>
    )
  }
  if (layer === 'sprite') return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      {boatOnly()}
    </svg>
  )
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#1f2858" /><stop offset="0.5" stopColor="#b9566a" /><stop offset="0.78" stopColor="#f6a25a" /></linearGradient>
        <linearGradient id={`${id}w`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#c9805a" /><stop offset="1" stopColor="#3a2a2a" /></linearGradient>
      </defs>
      <rect width="600" height="150" fill={`url(#${id}s)`} />
      <circle cx="520" cy="132" r="24" fill="#ffd59a" opacity="0.92" />
      <rect y="148" width="600" height="52" fill={`url(#${id}w)`} />
      {[0, 1, 2, 3, 4, 5, 6].map(i => <rect key={i} className="kw-glint" x={470 + (i % 4) * 24} y={154 + i * 5} width={36 - i * 4} height="1.4" fill="#ffe0a8" style={{ animationDelay: `${i * 0.5}s` }} />)}
      {house(262, 150, 44, 42, '#3f8fb0', '#ffd27a')}{house(318, 150, 52, 50, '#c0463a', '#ffd27a')}
      {house(382, 150, 40, 38, '#4aa36a', '#ffd27a')}{house(560, 150, 46, 40, '#e2b34a', '#fff0b0')}
      {/* moored long boat */}
      <path d="M270 166 H 360 C 366 166, 370 162, 372 156 C 368 168, 360 172, 350 172 H 280 C 274 172, 268 170, 266 162 Z" fill="#2f6fa8" stroke="#1b3a5a" strokeWidth="1" />
      <g fill="none" stroke="#1b1424" strokeWidth="1.6" className="kw-birds">
        <path d="M330 50 q 5 -5 10 0 q 5 -5 10 0" /><path d="M362 38 q 4 -4 8 0 q 4 -4 8 0" /><path d="M392 58 q 4 -4 8 0 q 4 -4 8 0" />
      </g>
      {layer !== 'base' && (<>
      {/* Bon Om Touk racing boat: long hull, high curling prow, rows of paddlers */}
      <g className="kw-boat" transform="translate(0 166)">
        <path d="M0 0 H 160 C 172 0, 178 -12, 178 -26 C 184 -18, 184 -6, 176 2 C 170 8, 150 9, 130 9 H 10 C 2 9, -6 4, -10 -10 C -4 -4, -2 -1, 0 0 Z" fill="#b8322a" stroke="#f2c94c" strokeWidth="1.4" />
        <circle cx="177" cy="-23" r="1.5" fill="#f2c94c" />
        <path d="M4 3 H 168" stroke="#f2c94c" strokeWidth="1" />
        <g className="kw-oars" stroke="#f6e9d6" strokeWidth="1.6" strokeLinecap="round">
          {Array.from({ length: 14 }, (_, i) => <path key={i} d={`M${14 + i * 10.5} -2 l -6 12`} />)}
        </g>
        <g fill="#1b1424">{Array.from({ length: 14 }, (_, i) => <circle key={i} cx={16 + i * 10.5} cy="-5" r="2.4" />)}</g>
      </g>
      </>)}
    </svg>
  )
}

/* ── Bayon: weathered face towers in a forest of towers ─────────────── */
// rough outline: jitter the edges so the stone reads as eroded blocks
function rough(points: [number, number][], amp: number, seed: number) {
  let r = seed
  const rnd = () => { r = (r * 9301 + 49297) % 233280; return r / 233280 - 0.5 }
  return 'M' + points.map(([x, y]) => `${(x + rnd() * amp).toFixed(1)} ${(y + rnd() * amp).toFixed(1)}`).join(' L ') + ' Z'
}

function towerSilhouette(cx: number, base: number, w: number, h: number, seed: number) {
  // stacked, shrinking courses ending in a lotus-bud top
  const pts: [number, number][] = []
  const steps = 9
  for (let i = 0; i <= steps; i++) {
    const t = i / steps, y = base - t * h
    const hw = w * (t < 0.55 ? 1 - t * 0.12 : (1 - 0.066) * Math.pow(1 - (t - 0.55) / 0.45, 0.6))
    pts.push([cx - hw, y]); if (i < steps) pts.push([cx - hw * 0.94, y - h / steps * 0.5])
  }
  const right = pts.map(([x, y]) => [2 * cx - x, y] as [number, number]).reverse()
  return rough([...pts, [cx, base - h - 6], ...right], 2.4, seed)
}

function bayonFace(cx: number, top: number, s: number, key: string) {
  const L = '#b9ae92', M = '#8c826c', D = '#4a4336', K = '#2a261f'
  const w = 40 * s
  return (
    <g key={key}>
      {/* tall diadem with floral pendants */}
      <rect x={cx - w} y={top} width={w * 2} height={20 * s} fill={M} />
      {Array.from({ length: 8 }, (_, i) => {
        const x = cx - w + (i + 0.5) * (w * 2 / 8)
        return <g key={i}><path d={`M${x} ${top + 3 * s} l ${-3 * s} ${6 * s} l ${3 * s} ${8 * s} l ${3 * s} ${-8 * s} Z`} fill={L} /><circle cx={x} cy={top + 18 * s} r={1.6 * s} fill={D} /></g>
      })}
      <rect x={cx - w} y={top + 20 * s} width={w * 2} height={2.5 * s} fill={K} opacity="0.6" />
      {/* cheeks and jaw */}
      <path d={`M${cx - w} ${top + 34 * s} C ${cx - w} ${top + 62 * s}, ${cx - 30 * s} ${top + 80 * s}, ${cx - 14 * s} ${top + 86 * s} H ${cx + 14 * s} C ${cx + 30 * s} ${top + 80 * s}, ${cx + w} ${top + 62 * s}, ${cx + w} ${top + 34 * s}`} fill={L} />
      <path d={`M${cx + 8 * s} ${top + 44 * s} C ${cx + 30 * s} ${top + 48 * s}, ${cx + 36 * s} ${top + 66 * s}, ${cx + 14 * s} ${top + 86 * s} H ${cx + 8 * s} Z`} fill={M} opacity="0.7" />
      {/* forehead and brow ridge in shadow */}
      <path d={`M${cx - w} ${top + 22 * s} H ${cx + w} V ${top + 34 * s} Q ${cx} ${top + 28 * s} ${cx - w} ${top + 34 * s} Z`} fill={M} />
      <path d={`M${cx - 33 * s} ${top + 34 * s} Q ${cx - 18 * s} ${top + 27 * s} ${cx - 3 * s} ${top + 34 * s} L ${cx - 3 * s} ${top + 38 * s} Q ${cx - 18 * s} ${top + 32 * s} ${cx - 33 * s} ${top + 38 * s} Z M ${cx + 3 * s} ${top + 34 * s} Q ${cx + 18 * s} ${top + 27 * s} ${cx + 33 * s} ${top + 34 * s} L ${cx + 33 * s} ${top + 38 * s} Q ${cx + 18 * s} ${top + 32 * s} ${cx + 3 * s} ${top + 38 * s} Z`} fill={D} />
      {/* lowered eyelids */}
      <path d={`M${cx - 30 * s} ${top + 42 * s} Q ${cx - 18 * s} ${top + 38 * s} ${cx - 6 * s} ${top + 42 * s} Q ${cx - 18 * s} ${top + 46 * s} ${cx - 30 * s} ${top + 42 * s} Z M ${cx + 6 * s} ${top + 42 * s} Q ${cx + 18 * s} ${top + 38 * s} ${cx + 30 * s} ${top + 42 * s} Q ${cx + 18 * s} ${top + 46 * s} ${cx + 6 * s} ${top + 42 * s} Z`} fill={K} />
      {/* broad flat nose with shadowed sides */}
      <path d={`M${cx - 4 * s} ${top + 36 * s} L ${cx - 12 * s} ${top + 60 * s} Q ${cx} ${top + 66 * s} ${cx + 12 * s} ${top + 60 * s} L ${cx + 4 * s} ${top + 36 * s} Z`} fill={L} />
      <path d={`M${cx + 4 * s} ${top + 36 * s} L ${cx + 12 * s} ${top + 60 * s} Q ${cx + 6 * s} ${top + 64 * s} ${cx + 2 * s} ${top + 63 * s} Z`} fill={D} opacity="0.8" />
      <path d={`M${cx - 11 * s} ${top + 61 * s} q ${4 * s} ${3 * s} ${7 * s} ${1 * s} M ${cx + 4 * s} ${top + 62 * s} q ${3 * s} ${2 * s} ${7 * s} ${-1 * s}`} stroke={K} strokeWidth={2 * s} fill="none" strokeLinecap="round" />
      {/* the smile: full lips, upturned corners */}
      <path d={`M${cx - 20 * s} ${top + 70 * s} Q ${cx} ${top + 66 * s} ${cx + 20 * s} ${top + 70 * s} Q ${cx} ${top + 74 * s} ${cx - 20 * s} ${top + 70 * s} Z`} fill={M} />
      <path d={`M${cx - 20 * s} ${top + 70 * s} Q ${cx} ${top + 76 * s} ${cx + 20 * s} ${top + 70 * s}`} stroke={K} strokeWidth={2.2 * s} fill="none" strokeLinecap="round" />
      <path d={`M${cx - 20 * s} ${top + 70 * s} q ${-3 * s} ${-2 * s} ${-4 * s} ${-5 * s} M ${cx + 20 * s} ${top + 70 * s} q ${3 * s} ${-2 * s} ${4 * s} ${-5 * s}`} stroke={K} strokeWidth={1.6 * s} fill="none" strokeLinecap="round" />
      <path d={`M${cx - 15 * s} ${top + 77 * s} Q ${cx} ${top + 84 * s} ${cx + 15 * s} ${top + 77 * s}`} fill={M} />
      {/* the face is carved across stacked blocks: course lines and joints */}
      <g stroke={K} strokeWidth={0.9} opacity="0.45">
        {[46, 58, 72].map(y => <path key={y} d={`M${cx - w} ${top + y * s} H ${cx + w}`} />)}
        <path d={`M${cx - 22 * s} ${top + 22 * s} V ${top + 46 * s} M ${cx + 26 * s} ${top + 46 * s} V ${top + 58 * s} M ${cx - 28 * s} ${top + 58 * s} V ${top + 72 * s} M ${cx + 18 * s} ${top + 72 * s} V ${top + 86 * s}`} />
      </g>
      {/* long ear lobes */}
      <path d={`M${cx - w - 2 * s} ${top + 34 * s} q ${-7 * s} ${12 * s} ${-2 * s} ${40 * s} q ${4 * s} ${2 * s} ${5 * s} ${-4 * s} V ${top + 34 * s} Z`} fill={M} />
      <path d={`M${cx + w + 2 * s} ${top + 34 * s} q ${7 * s} ${12 * s} ${2 * s} ${40 * s} q ${-4 * s} ${2 * s} ${-5 * s} ${-4 * s} V ${top + 34 * s} Z`} fill={D} />
    </g>
  )
}

export function BayonScene({ className }: P) {
  const id = useSid()
  const stone = '#6f685a', dark = '#3d382f'
  const courses = (x0: number, x1: number, y0: number, y1: number, k: string) =>
    Array.from({ length: Math.floor((y1 - y0) / 8) }, (_, i) => (
      <path key={`${k}${i}`} d={`M${x0} ${y0 + i * 8} H ${x1}`} stroke={dark} strokeWidth="0.7" opacity="0.5" />
    ))
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#2f73c8" /><stop offset="1" stopColor="#8fbfe6" /></linearGradient>
        <linearGradient id={`${id}lit`} x1="0" x2="1"><stop offset="0" stopColor="#fff2d0" stopOpacity="0.18" /><stop offset="1" stopColor="#000" stopOpacity="0.3" /></linearGradient>
        <linearGradient id={`${id}sh`} x1="0" x2="1"><stop offset="0" stopColor="#000" stopOpacity="0" /><stop offset="0.5" stopColor="#000" stopOpacity="0.22" /><stop offset="1" stopColor="#000" stopOpacity="0" /></linearGradient>
        <clipPath id={`${id}c`}><rect x="240" y="0" width="360" height="200" /></clipPath>
      </defs>
      <rect width="600" height="200" fill={`url(#${id}s)`} />
      <g clipPath={`url(#${id}c)`}>
        {/* far towers, the forest of faces */}
        <g fill="#5d584c" opacity="0.75">
          <path d={towerSilhouette(292, 160, 22, 112, 3)} /><path d={towerSilhouette(348, 160, 20, 96, 7)} />
          <path d={towerSilhouette(520, 160, 22, 118, 11)} /><path d={towerSilhouette(588, 160, 20, 100, 5)} />
        </g>
        {/* side tower turned three quarters, its face catching the light */}
        <clipPath id={`${id}tb`}><path d={towerSilhouette(530, 176, 44, 150, 17)} /></clipPath>
        <clipPath id={`${id}ta`}><path d={towerSilhouette(420, 184, 60, 176, 23)} /></clipPath>
        <path d={towerSilhouette(530, 176, 44, 150, 17)} fill={stone} />
        <g clipPath={`url(#${id}tb)`}>
          {courses(480, 580, 30, 176, 'b')}
          <g transform="translate(530 0) scale(0.78 1) translate(-530 0)">{bayonFace(530, 70, 0.95, 'side')}</g>
          <rect x="480" y="20" width="100" height="160" fill={`url(#${id}lit)`} />
        </g>
        {/* main tower and its face */}
        <path d={towerSilhouette(420, 184, 60, 176, 23)} fill={stone} />
        <g clipPath={`url(#${id}ta)`}>{courses(356, 484, 14, 184, 'a')}</g>
        {bayonFace(420, 52, 1.32, 'main')}
        {/* lichen and moss in the crevices */}
        {[[372, 64], [470, 150], [384, 168], [452, 40], [540, 60], [500, 160], [300, 120]].map(([x, y], i) => <ellipse key={i} cx={x} cy={y} rx="10" ry="4" fill="#6f7d44" opacity="0.55" />)}
        {/* gallery and doorway at the foot */}
        <rect x="240" y="168" width="360" height="32" fill="#4d4739" />
        {[262, 318, 560].map(x => <path key={x} d={`M${x} 200 V 180 L ${x + 8} 174 L ${x + 16} 180 V 200 Z`} fill="#1f1c17" />)}
        <rect className="kw-shadow" x="250" y="0" width="120" height="200" fill={`url(#${id}sh)`} />
      </g>
      <g fill="none" stroke="#1b2430" strokeWidth="1.5" className="kw-birds">
        <path d="M300 26 q 5 -5 10 0 q 5 -5 10 0" /><path d="M330 16 q 4 -4 8 0 q 4 -4 8 0" />
      </g>
    </svg>
  )
}

/* ── Reamker mural: the painted galleries of the Silver Pagoda ───────── */
export function MuralScene({ className }: P) {
  const id = useSid()
  const mtree = (x: number, y: number, r: number, c: string) => (
    <g>
      <rect x={x - 1.5} y={y} width="3" height={r * 1.3} fill="#5a3a24" />
      {[[0, 0, 1], [-0.7, 0.35, 0.7], [0.7, 0.35, 0.7], [0, -0.6, 0.65]].map(([dx, dy, k], i) => (
        <circle key={i} cx={x + dx * r} cy={y + dy * r} r={r * k} fill={c} stroke="#1f3a1a" strokeWidth="1" />
      ))}
    </g>
  )
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#1c2f7a" /><stop offset="1" stopColor="#3f62b0" /></linearGradient>
      </defs>
      {/* flat painted bands: blue sky, red earth, as in the gallery murals */}
      <rect width="600" height="200" fill={`url(#${id}s)`} />
      {[[300, 30], [380, 46], [520, 24]].map(([x, y], i) => (
        <path key={i} className="kw-cloud" style={{ animationDelay: `${i * 2}s` }} d={`M${x} ${y} q 8 -10 18 -2 q 10 -8 18 2 q 8 0 6 6 h -44 q -4 -6 2 -6 Z`} fill="#e8eef8" opacity="0.9" />
      ))}
      <path d="M240 96 Q 330 80 420 92 T 600 88 V 200 H 240 Z" fill="#c4472b" />
      <path d="M240 150 Q 360 136 460 150 T 600 146 V 200 H 240 Z" fill="#a8371f" />
      {/* river with stylised wave lines */}
      <path d="M240 122 Q 340 112 440 124 T 600 118 V 134 Q 520 140 440 136 T 240 138 Z" fill="#3f8fb0" />
      {[0, 1, 2].map(i => <path key={i} className="kw-wave" d={`M${250 + i * 110} 128 q 8 -4 16 0 t 16 0 t 16 0`} fill="none" stroke="#bfe6f2" strokeWidth="1.2" />)}
      {mtree(272, 74, 16, '#3f7a3a')}{mtree(318, 82, 12, '#4a8a42')}{mtree(574, 70, 18, '#3f7a3a')}{mtree(536, 84, 11, '#4a8a42')}
      {/* gold pavilion with tiered roof */}
      <g>
        <rect x="408" y="78" width="84" height="44" fill="#f3e2b0" stroke="#7a4a12" strokeWidth="1.2" />
        {[0, 1, 2, 3].map(i => <rect key={i} x={414 + i * 22} y="82" width="6" height="40" fill="#d9a63a" />)}
        <path d="M398 80 L 450 56 L 502 80 Z" fill="#e0a83a" stroke="#7a4a12" strokeWidth="1.2" />
        <path d="M410 60 L 450 40 L 490 60 Z" fill="#e8b84a" stroke="#7a4a12" strokeWidth="1.2" />
        <path d="M424 42 L 450 26 L 476 42 Z" fill="#f2c95a" stroke="#7a4a12" strokeWidth="1.2" />
        <path d="M450 26 V 8" stroke="#e0a83a" strokeWidth="2" />
        <path d="M398 80 c -6 0 -8 -6 -4 -9 M 502 80 c 6 0 8 -6 4 -9" fill="none" stroke="#e0a83a" strokeWidth="2" />
        <path d="M440 122 V 100 Q 450 90 460 100 V 122 Z" fill="#7a2a1a" />
      </g>
      {/* white elephant on the bank, as in the Reamker procession scenes */}
      <g className="kw-sway" style={{ transformOrigin: '340px 156px' }} stroke="#4a3a2a" strokeWidth="1">
        <g fill="#f4f1ea">
          {[316, 326, 348, 358].map(x => <rect key={x} x={x} y="140" width="7" height="16" rx="1.5" />)}
          <ellipse cx="338" cy="134" rx="26" ry="15" />
          <circle cx="366" cy="128" r="11" />
          <path d="M374 132 C 382 138, 382 150, 376 156 C 374 152, 376 144, 370 138 Z" />
          <path d="M360 122 C 352 122, 350 138, 360 140 C 362 134, 362 128, 360 122 Z" fill="#e8e2d6" />
          <path d="M312 130 c -6 4 -6 12 -4 16" fill="none" />
        </g>
        <path d="M322 120 H 352 L 356 140 H 318 Z" fill="#c4472b" />
        <path d="M322 120 H 352 L 354 128 H 320 Z" fill="#e0a83a" />
        {[324, 334, 344].map(x => <circle key={x} cx={x} cy="134" r="1.6" fill="#e0a83a" stroke="none" />)}
        <circle cx="369" cy="126" r="1.3" fill="#2a1a12" stroke="none" />
        <path d="M374 136 q 6 2 8 -2" fill="none" stroke="#f4f1ea" strokeWidth="2" />
      </g>
      {/* painted border, as at the foot of the murals */}
      <rect x="240" y="188" width="360" height="12" fill="#6a1e14" />
      {Array.from({ length: 18 }, (_, i) => <path key={i} d={`M${244 + i * 20} 194 l 5 -4 5 4 -5 4 Z`} fill="#e0a83a" />)}
    </svg>
  )
}
