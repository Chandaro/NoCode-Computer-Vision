import { createContext, useContext, useEffect, useId, useState, type ReactNode } from 'react'
import { hashSeed } from './Sigil'

/*
 * Khmer art worlds. Each project gets one, so projects stop looking alike.
 * A world is a palette (applied to the whole app while inside the project),
 * an animated scene for banners and cards, a texture, a border band and an
 * ambient particle style. Sources:
 *   angkor    Angkor Wat at dawn: lotus-bud towers, naga causeway (Kbach Angkor)
 *   banteay   Banteay Srei: pink sandstone pediments, flame kbach (phñi phleung),
 *             and its alternating checkerboard carving
 *   taprohm   Ta Prohm: jungle temple, silk-cotton roots, circular medallions
 *   sbek      Sbek Thom: uncoloured cow-hide shadow panels before a fire-lit screen
 *   palace    Royal Palace, Phnom Penh: tiered roofs, chovear finials, naga eaves
 *   tonlesap  Tonle Sap: stilt houses and the Bon Om Touk dragon-boat race
 */

export type WorldId = 'angkor' | 'banteay' | 'taprohm' | 'sbek' | 'palace' | 'tonlesap'

export interface World {
  id: WorldId
  en: string
  km: string
  blurb: string
  vars: Record<string, string>
  particles: 'petals' | 'roses' | 'leaves' | 'embers' | 'gold' | 'fireflies'
  Scene: (p: { className?: string }) => ReactNode
}

const v = (bg: string, s: string, s2: string, s3: string, b: string, b2: string, t: string, t2: string, t3: string,
  a: string, ah: string, aRgb: string, on: string, sec: string, secRgb: string) => ({
  '--bg': bg, '--surface': s, '--surface2': s2, '--surface3': s3, '--border': b, '--border2': b2,
  '--text': t, '--text2': t2, '--text3': t3,
  '--accent': a, '--accent-h': ah, '--accent-s': `rgba(${aRgb},0.15)`, '--accent-t': `rgba(${aRgb},0.07)`, '--on-accent': on,
  '--secondary': sec, '--secondary-h': sec, '--secondary-s': `rgba(${secRgb},0.13)`, '--secondary-t': `rgba(${secRgb},0.06)`,
  '--world-rgb': aRgb,
})

/* ── Scenes (viewBox 600x200, art on the right so titles stay readable) ── */

const bud = (cx: number, base: number, top: number, hw: number) =>
  `M${cx - hw} ${base} C ${cx - hw} ${base - (base - top) * 0.55}, ${cx - hw * 0.45} ${top + (base - top) * 0.2}, ${cx} ${top} C ${cx + hw * 0.45} ${top + (base - top) * 0.2}, ${cx + hw} ${base - (base - top) * 0.55}, ${cx + hw} ${base} Z`

function AngkorScene({ className }: { className?: string }) {
  const id = useId().replace(/:/g, '')
  const temple = (
    <g fill="#0b0918">
      <path d={bud(430, 132, 34, 15)} />
      <path d={bud(388, 132, 66, 11)} /><path d={bud(472, 132, 66, 11)} />
      <path d={bud(356, 134, 80, 10)} /><path d={bud(504, 134, 80, 10)} />
      <rect x="340" y="128" width="180" height="10" /><rect x="300" y="136" width="260" height="9" />
      <rect x="230" y="143" width="400" height="8" />
      <rect x="566" y="62" width="2.6" height="84" /><circle cx="567" cy="60" r="13" />
      <rect x="276" y="84" width="2.2" height="60" /><circle cx="277" cy="82" r="10" />
    </g>
  )
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#14112e" /><stop offset="0.45" stopColor="#3c2a66" />
          <stop offset="0.68" stopColor="#c66184" /><stop offset="0.76" stopColor="#f0b15a" />
        </linearGradient>
        <radialGradient id={`${id}u`}><stop offset="0" stopColor="#fff1d6" /><stop offset="0.45" stopColor="#f0b15a" /><stop offset="1" stopColor="#f0b15a" stopOpacity="0" /></radialGradient>
        <linearGradient id={`${id}w`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#7a3f78" /><stop offset="1" stopColor="#100e22" /></linearGradient>
      </defs>
      <rect width="600" height="151" fill={`url(#${id}s)`} />
      <circle className="kw-pulse" cx="430" cy="112" r="64" fill={`url(#${id}u)`} />
      {temple}
      <rect y="151" width="600" height="49" fill={`url(#${id}w)`} />
      <g opacity="0.32" transform="translate(0 302) scale(1 -1)"><g className="kw-ripple">{temple}</g></g>
      {[0, 1, 2, 3, 4, 5].map(i => <rect key={i} className="kw-glint" x={360 + i * 26} y={160 + (i % 3) * 9} width={18 - (i % 3) * 4} height="1.5" fill="#f7d79a" style={{ animationDelay: `${i * 0.7}s` }} />)}
      <g fill="#ea8fa2"><path d="M300 178 C 297 172, 297 166, 300 160 C 303 166, 303 172, 300 178 Z" /><path d="M530 186 C 527 180, 527 174, 530 168 C 533 174, 533 180, 530 186 Z" /></g>
      <g fill="#1f4a40"><ellipse cx="300" cy="179" rx="16" ry="3" /><ellipse cx="530" cy="187" rx="18" ry="3" /><ellipse cx="410" cy="194" rx="22" ry="3" /></g>
    </svg>
  )
}

// flame curls (kbach phñi phleung) placed along a curve
function flamesAlong(pts: [number, number][], size: number, flip = false) {
  return pts.map(([x, y], i) => (
    <path key={i} transform={`translate(${x} ${y}) scale(${flip ? -size : size} ${size})`}
      d="M0 0 C -2 -4, -1 -8, 3 -11 C 2 -8, 3 -6, 5 -5 C 5 -8, 7 -10, 9 -10 C 7 -7, 8 -3, 5 0 Z" />
  ))
}

function BanteayScene({ className }: { className?: string }) {
  const id = useId().replace(/:/g, '')
  // flame crest follows the pediment's two quadratic edges, meeting at the apex
  const left: [number, number][] = [], right: [number, number][] = []
  for (let i = 1; i <= 8; i++) {
    const t = i / 9, u = 1 - t
    const x = u * u * 290 + 2 * u * t * 334 + t * t * 420
    const y = u * u * 162 + 2 * u * t * 84 + t * t * 22
    left.push([x - 4, y - 2]); right.push([840 - x + 4, y - 2])
  }
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#2a1220" /><stop offset="1" stopColor="#5a2a32" /></linearGradient>
        <linearGradient id={`${id}st`} x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#e7927d" /><stop offset="0.6" stopColor="#c9705e" /><stop offset="1" stopColor="#8a4541" /></linearGradient>
        <pattern id={`${id}ck`} width="16" height="16" patternUnits="userSpaceOnUse">
          <rect width="8" height="8" fill="#b5604f" /><rect x="8" y="8" width="8" height="8" fill="#b5604f" />
          <rect x="8" width="8" height="8" fill="#7a3a37" /><rect y="8" width="8" height="8" fill="#7a3a37" />
          <circle cx="4" cy="4" r="1.6" fill="#f2b09a" /><circle cx="12" cy="12" r="1.6" fill="#f2b09a" />
        </pattern>
        <linearGradient id={`${id}l`} x1="0" x2="1"><stop offset="0" stopColor="#fff" stopOpacity="0" /><stop offset="0.5" stopColor="#fff1e6" stopOpacity="0.35" /><stop offset="1" stopColor="#fff" stopOpacity="0" /></linearGradient>
        <clipPath id={`${id}c`}><path d="M290 162 Q 334 84 420 22 Q 506 84 550 162 Z" /></clipPath>
      </defs>
      <rect width="600" height="200" fill={`url(#${id}s)`} />
      {/* pediment */}
      <path d="M280 166 Q 326 78 420 10 Q 514 78 560 166 Z" fill="#7a3a37" />
      <path d="M290 162 Q 334 84 420 22 Q 506 84 550 162 Z" fill={`url(#${id}st)`} />
      <g clipPath={`url(#${id}c)`}>
        {/* tympanum: central medallion flanked by flame scrolls */}
        <g transform="translate(420 112)" fill="none" stroke="#5a2a2e" strokeWidth="2">
          <circle r="30" /><circle r="22" strokeDasharray="3 3" />
          {Array.from({ length: 12 }, (_, i) => <path key={i} transform={`rotate(${i * 30})`} d="M0 -21 C 5 -15, 5 -9, 0 -5 C -5 -9, -5 -15, 0 -21 Z" fill="#f2b09a" />)}
          <circle r="5" fill="#5a2a2e" />
        </g>
        <g fill="#5a2a2e" opacity="0.85">
          {flamesAlong([[352, 136], [374, 110], [380, 150], [334, 154], [392, 88]], 2)}
          {flamesAlong([[488, 136], [466, 110], [460, 150], [506, 154], [448, 88]], 2, true)}
        </g>
        <rect className="kw-rake" x="200" y="0" width="90" height="200" fill={`url(#${id}l)`} />
      </g>
      {/* flame crest along the edges */}
      <g fill="#f2b09a">{flamesAlong(left, 1.5)}{flamesAlong(right, 1.5, true)}</g>
      {/* curled ends (naga heads, abstracted) */}
      <g fill="none" stroke="#f2b09a" strokeWidth="4" strokeLinecap="round"><path d="M284 164 C 268 160, 266 144, 278 140" /><path d="M556 164 C 572 160, 574 144, 562 140" /></g>
      {/* lintel with the alternating checkerboard carving */}
      <rect x="270" y="166" width="300" height="34" fill={`url(#${id}ck)`} />
      <rect x="270" y="166" width="300" height="3" fill="#f2b09a" />
    </svg>
  )
}

function TaProhmScene({ className }: { className?: string }) {
  const id = useId().replace(/:/g, '')
  const blocks: ReactNode[] = []
  for (let r = 0; r < 6; r++) for (let c = 0; c < 9; c++) {
    const w = 38, x = 260 + c * w + (r % 2 ? 19 : 0), y = 60 + r * 23
    const g = 70 + ((r * 7 + c * 13) % 5) * 7
    blocks.push(<rect key={`${r}-${c}`} x={x} y={y} width={w - 2} height="21" rx="2" fill={`rgb(${g - 8},${g + 4},${g - 12})`} />)
  }
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#0d2418" /><stop offset="1" stopColor="#1d3a26" /></linearGradient>
        <radialGradient id={`${id}sun`} cx="0.7" cy="0.1" r="0.6"><stop offset="0" stopColor="#e9f5b0" stopOpacity="0.45" /><stop offset="1" stopColor="#e9f5b0" stopOpacity="0" /></radialGradient>
      </defs>
      <rect width="600" height="200" fill={`url(#${id}s)`} />
      <g opacity="0.9">{blocks}</g>
      {/* doorway */}
      <path d="M392 198 V 108 L 404 96 H 436 L 448 108 V 198 Z" fill="#06100b" />
      <path d="M384 108 H 456 M 388 100 H 452" stroke="#9aa486" strokeWidth="4" />
      {/* circular medallions carved into the stones */}
      {[[300, 85], [530, 131], [339, 154], [490, 85]].map(([x, y], i) => (
        <g key={i} transform={`translate(${x} ${y})`} fill="none" stroke="#b9c2a0" strokeWidth="1.2" opacity="0.75">
          <circle r="8" /><circle r="4" /><path d="M0 -8 V 8 M -8 0 H 8" />
        </g>
      ))}
      {/* silk-cotton tree roots spilling over the wall */}
      <g fill="none" stroke="#cdb98d" strokeLinecap="round" className="kw-roots">
        <path d="M470 0 C 470 40, 440 60, 446 110 C 450 150, 420 170, 410 200" strokeWidth="9" />
        <path d="M486 0 C 500 50, 530 70, 520 120 C 512 160, 540 180, 560 200" strokeWidth="7" />
        <path d="M455 30 C 420 50, 380 50, 350 80 C 330 100, 320 140, 300 200" strokeWidth="5" />
        <path d="M500 40 C 560 60, 590 90, 600 140" strokeWidth="4" />
        <path d="M446 110 C 470 130, 480 160, 474 200" strokeWidth="3.5" />
        <path d="M350 80 C 340 60, 310 60, 290 70" strokeWidth="2.5" />
      </g>
      <g fill="#14321f">
        <circle cx="470" cy="-10" r="60" /><circle cx="540" cy="0" r="50" /><circle cx="400" cy="-6" r="44" /><circle cx="590" cy="30" r="36" />
      </g>
      <rect className="kw-dapple" width="600" height="200" fill={`url(#${id}sun)`} />
    </svg>
  )
}

function SbekScene({ className }: { className?: string }) {
  const id = useId().replace(/:/g, '')
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <radialGradient id={`${id}scr`} cx="0.7" cy="0.62" r="0.7">
          <stop offset="0" stopColor="#ffe2b0" /><stop offset="0.35" stopColor="#f29a45" /><stop offset="0.75" stopColor="#7a3410" /><stop offset="1" stopColor="#1a0c05" />
        </radialGradient>
        <pattern id={`${id}holes`} width="9" height="9" patternUnits="userSpaceOnUse">
          <rect width="9" height="9" fill="#fff" /><circle cx="4.5" cy="4.5" r="1.3" fill="#000" />
        </pattern>
        <mask id={`${id}m`}>
          <rect width="600" height="200" fill={`url(#${id}holes)`} />
          {/* solid outlines stay uncut */}
          <g fill="none" stroke="#fff" strokeWidth="5">
            <path d="M330 150 V 70 Q 420 10 510 70 V 150 Z" />
          </g>
        </mask>
      </defs>
      <rect width="600" height="200" fill="#0e0a07" />
      <rect className="kw-flicker" x="230" y="0" width="370" height="178" fill={`url(#${id}scr)`} />
      <rect x="230" y="0" width="370" height="178" fill="none" stroke="#2a1408" strokeWidth="6" />
      {/* the leather panel: palace roof under a sacred tree, perforated */}
      <g className="kw-sway" style={{ transformOrigin: '420px 200px' }}>
        <g mask={`url(#${id}m)`} fill="#2a1408">
          <path d="M330 150 V 70 Q 420 10 510 70 V 150 Z" />
        </g>
        <g fill="none" stroke="#ffcf8a" strokeOpacity="0.85" strokeWidth="2.4" strokeLinecap="round">
          <path d="M360 140 L 360 108 L 420 72 L 480 108 L 480 140" />
          <path d="M380 140 V 116 M 400 140 V 112 M 440 140 V 112 M 460 140 V 116" strokeWidth="1.6" />
          <path d="M404 140 V 124 Q 420 112 436 124 V 140" />
          <path d="M350 112 L 420 66 L 490 112" strokeWidth="5" />
          <path d="M412 66 L 420 46 L 428 66" />
          <path d="M346 112 C 340 104, 340 98, 346 96 M 494 112 C 500 104, 500 98, 494 96" strokeWidth="3" />
          {/* bodhi leaves either side */}
          <path d="M342 64 C 334 54, 338 42, 348 40 C 352 50, 352 58, 342 64 Z M 498 64 C 506 54, 502 42, 492 40 C 488 50, 488 58, 498 64 Z" strokeWidth="1.8" />
          <path d="M336 146 H 504" strokeWidth="2" />
        </g>
        <rect x="402" y="150" width="4" height="60" fill="#5a3a1c" /><rect x="434" y="150" width="4" height="60" fill="#5a3a1c" />
      </g>
      {/* coconut-husk fire at the foot of the screen */}
      <g className="kw-fire">
        <path d="M400 200 C 396 186, 404 180, 402 168 C 410 178, 416 182, 414 200 Z" fill="#ffb347" />
        <path d="M420 200 C 418 190, 426 184, 426 172 C 434 184, 436 190, 432 200 Z" fill="#ff7a29" />
        <path d="M440 200 C 438 192, 444 188, 444 180 C 450 188, 452 194, 448 200 Z" fill="#ffd59a" />
      </g>
      <rect y="178" width="600" height="22" fill="#120a05" />
    </svg>
  )
}

function PalaceScene({ className }: { className?: string }) {
  const id = useId().replace(/:/g, '')
  // one tier of roof with chovear horns at both ends
  const tier = (cx: number, y: number, w: number, h: number) => (
    <g>
      <path d={`M${cx - w} ${y} L ${cx} ${y - h} L ${cx + w} ${y} Z`} fill="#d9822b" />
      <path d={`M${cx - w} ${y} L ${cx} ${y - h} L ${cx + w} ${y}`} fill="none" stroke="#f2c94c" strokeWidth="2.5" />
      <path d={`M${cx - w} ${y} c -6 0, -8 -6, -4 -12 M ${cx + w} ${y} c 6 0, 8 -6, 4 -12`} fill="none" stroke="#f2c94c" strokeWidth="2.4" strokeLinecap="round" />
    </g>
  )
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#0d1230" /><stop offset="0.7" stopColor="#2a2f6a" /><stop offset="1" stopColor="#6b4a6a" /></linearGradient>
        <linearGradient id={`${id}g`} x1="0" x2="1"><stop offset="0" stopColor="#fff6c8" stopOpacity="0" /><stop offset="0.5" stopColor="#fff6c8" stopOpacity="0.5" /><stop offset="1" stopColor="#fff6c8" stopOpacity="0" /></linearGradient>
        <clipPath id={`${id}c`}><rect x="280" y="0" width="300" height="200" /></clipPath>
      </defs>
      <rect width="600" height="200" fill={`url(#${id}s)`} />
      {Array.from({ length: 16 }, (_, i) => <circle key={i} className="kw-star" cx={250 + (i * 97) % 350} cy={10 + (i * 41) % 70} r={i % 3 ? 0.9 : 1.4} fill="#f0ead8" style={{ animationDelay: `${(i % 5) * 0.8}s` }} />)}
      <g clipPath={`url(#${id}c)`}>
        {/* central spire */}
        <path d="M420 14 C 424 30, 428 44, 430 60 H 410 C 412 44, 416 30, 420 14 Z" fill="#f2c94c" />
        <rect x="408" y="58" width="24" height="10" fill="#f2c94c" />
        {tier(420, 92, 46, 24)}{tier(420, 116, 76, 30)}{tier(420, 142, 112, 34)}
        {/* white walls, gold columns */}
        <rect x="322" y="142" width="196" height="40" fill="#efe4cf" />
        {Array.from({ length: 8 }, (_, i) => <rect key={i} x={330 + i * 24} y="146" width="6" height="36" fill="#d9b24a" />)}
        <rect x="300" y="182" width="240" height="8" fill="#c9b9a0" /><rect x="286" y="190" width="268" height="10" fill="#a8977c" />
        {/* naga eave line */}
        <path d="M308 142 H 532" stroke="#2f7a4a" strokeWidth="3" />
        <rect className="kw-shine" x="150" y="0" width="60" height="200" fill={`url(#${id}g)`} />
      </g>
      {/* parasol */}
      <g fill="#f2c94c"><rect x="560" y="80" width="2" height="110" />
        <path d="M548 92 H 574 L 570 86 H 552 Z M 551 104 H 571 L 568 99 H 554 Z M 553 115 H 569 L 566 111 H 556 Z" /></g>
    </svg>
  )
}

function TonleSapScene({ className }: { className?: string }) {
  const id = useId().replace(/:/g, '')
  const house = (x: number, y: number, w: number, c: string) => (
    <g>
      {[0.15, 0.5, 0.85].map(p => <rect key={p} x={x + w * p - 1} y={y} width="2" height={150 - y + 8} fill="#2a1a12" />)}
      <rect x={x} y={y - 18} width={w} height="18" fill={c} />
      <path d={`M${x - 5} ${y - 18} L ${x + w / 2} ${y - 34} L ${x + w + 5} ${y - 18} Z`} fill="#3a2418" />
      <rect x={x + w * 0.4} y={y - 12} width={w * 0.2} height="8" fill="#f7c46a" className="kw-lamp" />
    </g>
  )
  return (
    <svg className={className} viewBox="0 0 600 200" preserveAspectRatio="xMidYMid slice" aria-hidden>
      <defs>
        <linearGradient id={`${id}s`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#1b2350" /><stop offset="0.55" stopColor="#b9566a" /><stop offset="0.75" stopColor="#f39a5a" /></linearGradient>
        <linearGradient id={`${id}w`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#c96a5a" /><stop offset="1" stopColor="#0a1620" /></linearGradient>
      </defs>
      <rect width="600" height="150" fill={`url(#${id}s)`} />
      <circle cx="500" cy="128" r="26" fill="#ffd59a" opacity="0.9" />
      <rect y="150" width="600" height="50" fill={`url(#${id}w)`} />
      {[0, 1, 2, 3, 4, 5, 6].map(i => <rect key={i} className="kw-glint" x={430 + (i % 4) * 30} y={156 + i * 5} width={40 - i * 4} height="1.4" fill="#ffd59a" style={{ animationDelay: `${i * 0.5}s` }} />)}
      {house(270, 120, 44, '#6b4430')}{house(326, 112, 52, '#7a5034')}{house(560, 118, 40, '#6b4430')}
      {/* birds */}
      <g fill="none" stroke="#1b1424" strokeWidth="1.6" className="kw-birds">
        <path d="M330 50 q 5 -5 10 0 q 5 -5 10 0" /><path d="M360 38 q 4 -4 8 0 q 4 -4 8 0" /><path d="M390 56 q 4 -4 8 0 q 4 -4 8 0" />
      </g>
      {/* Bon Om Touk dragon boat crossing the lake */}
      <g className="kw-boat">
        <path d="M0 0 H 150 C 162 0, 168 -10, 170 -22 C 174 -16, 172 -6, 166 2 C 160 8, 140 8, 120 8 H 10 C 4 8, -4 4, -8 -6 C -2 -2, 0 -1, 0 0 Z" fill="#c23a2a" stroke="#f2c94c" strokeWidth="1.4" />
        <circle cx="168" cy="-19" r="1.4" fill="#f2c94c" />
        <g className="kw-oars" stroke="#f6e9d6" strokeWidth="1.6" strokeLinecap="round">
          {Array.from({ length: 12 }, (_, i) => <path key={i} d={`M${16 + i * 11} -2 l -5 10`} />)}
        </g>
        <g fill="#1b1424">{Array.from({ length: 12 }, (_, i) => <circle key={i} cx={18 + i * 11} cy="-5" r="2.4" />)}</g>
      </g>
    </svg>
  )
}

export const WORLDS: World[] = [
  { id: 'angkor', en: 'Angkor Wat', km: 'អង្គរវត្ត', blurb: 'Lotus-bud towers at dawn over the moat',
    vars: v('#100e22', '#17142e', '#1e1a3a', '#282347', '#29234a', '#3b3266', '#efe4cf', '#b9ad9a', '#8f86a3', '#e0a83a', '#ebb952', '224,168,58', '#1d1305', '#ea8fa2', '234,143,162'),
    particles: 'petals', Scene: AngkorScene },
  { id: 'banteay', en: 'Banteay Srei', km: 'បន្ទាយស្រី', blurb: 'Pink sandstone pediments and flame kbach',
    vars: v('#1c0f17', '#261520', '#301b28', '#3d2333', '#3a2130', '#553044', '#f3e6dc', '#cdb3aa', '#a08590', '#f0927c', '#f5a693', '240,146,124', '#2a0e0a', '#f4c28a', '244,194,138'),
    particles: 'roses', Scene: BanteayScene },
  { id: 'taprohm', en: 'Ta Prohm', km: 'តាព្រហ្ម', blurb: 'Jungle temple held by silk-cotton roots',
    vars: v('#0b1612', '#111f19', '#172a21', '#1f362b', '#1f342a', '#2e4a3c', '#e8eedc', '#b4c2ab', '#83977f', '#a6d17a', '#b8de8f', '166,209,122', '#13200a', '#e0be7a', '224,190,122'),
    particles: 'leaves', Scene: TaProhmScene },
  { id: 'sbek', en: 'Sbek Thom', km: 'ស្បែកធំ', blurb: 'Leather shadow panels before a fire-lit screen',
    vars: v('#0e0a07', '#17110c', '#211810', '#2c2016', '#2a1f15', '#3d2d1e', '#f6e9d6', '#c9b49a', '#9a8670', '#f5a04a', '#f8b469', '245,160,74', '#2a1203', '#ffd59a', '255,213,154'),
    particles: 'embers', Scene: SbekScene },
  { id: 'palace', en: 'Royal Palace', km: 'ព្រះបរមរាជវាំង', blurb: 'Tiered roofs with chovear finials',
    vars: v('#0d1230', '#141a3d', '#1b2249', '#252d5a', '#232b55', '#34407a', '#f0ead8', '#bdb8c9', '#8e8db0', '#f2c94c', '#f6d670', '242,201,76', '#241a02', '#ef8a5a', '239,138,90'),
    particles: 'gold', Scene: PalaceScene },
  { id: 'tonlesap', en: 'Tonle Sap', km: 'ទន្លេសាប', blurb: 'Stilt houses and the dragon-boat race',
    vars: v('#0a1620', '#10202c', '#162a38', '#1e3646', '#1d3242', '#2b4a5e', '#e6f0ee', '#a9c0c2', '#7e999e', '#5fd0c6', '#7fdcd4', '95,208,198', '#062220', '#f39a5a', '243,154,90'),
    particles: 'fireflies', Scene: TonleSapScene },
]

export const worldById = (id: string | null | undefined) => WORLDS.find(w => w.id === id)

const KEY = (pid: string | number) => `kh-world-${pid}`
const listeners = new Set<() => void>()

export function defaultWorld(projectId: string | number): World {
  // numeric ids cycle through the worlds so neighbouring projects always differ
  const n = Number(projectId)
  if (Number.isInteger(n) && n > 0) return WORLDS[(n - 1) % WORLDS.length]
  return WORLDS[Math.floor(hashSeed(`world-${projectId}`) * WORLDS.length) % WORLDS.length]
}

export function getWorld(projectId: string | number): World {
  try { const w = worldById(localStorage.getItem(KEY(projectId))); if (w) return w } catch { /* storage unavailable */ }
  return defaultWorld(projectId)
}

export function setWorld(projectId: string | number, id: WorldId) {
  try { localStorage.setItem(KEY(projectId), id) } catch { /* storage unavailable */ }
  listeners.forEach(l => l())
}

export function useWorld(projectId: string | number | undefined): World | null {
  const [, bump] = useState(0)
  useEffect(() => {
    const l = () => bump(n => n + 1)
    listeners.add(l)
    return () => { listeners.delete(l) }
  }, [])
  return projectId === undefined ? null : getWorld(projectId)
}

/** Paint the whole app in a world's palette while it is mounted. */
export function useWorldPalette(world: World | null) {
  useEffect(() => {
    if (!world) return
    const root = document.documentElement
    Object.entries(world.vars).forEach(([k, val]) => root.style.setProperty(k, val))
    root.dataset.world = world.id
    return () => {
      Object.keys(world.vars).forEach(k => root.style.removeProperty(k))
      delete root.dataset.world
    }
  }, [world])
}

/** The world of the project currently open (null outside projects). */
export const WorldContext = createContext<World | null>(null)
export const useCurrentWorld = () => useContext(WorldContext)
