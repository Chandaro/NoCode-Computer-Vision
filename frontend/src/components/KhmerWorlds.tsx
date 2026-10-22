import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { AngkorScene, BanteayScene, TaProhmScene, SbekScene, PalaceScene, TonleSapScene, BayonScene, MuralScene } from './WorldScenes'
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
 *   bayon     Bayon: the serene carved faces on its towers
 *   mural     Reamker murals of the Silver Pagoda: flat red earth, blue sky, gold pavilions
 */

export type WorldId = 'angkor' | 'banteay' | 'taprohm' | 'sbek' | 'palace' | 'tonlesap' | 'bayon' | 'mural'

export interface SceneFx {
  ripple?: [number, number]        // rows that shimmer like water
  sweep?: 'light' | 'shadow'       // a band of light (or shade) crossing the carving
  flicker?: boolean                // firelight brightness
  twinkle?: [number, number]       // rows where single pixels glint
  sprite?: number                  // seconds for the sprite layer to cross the scene
}

export interface World {
  id: WorldId
  en: string
  km: string
  blurb: string
  vars: Record<string, string>
  particles: 'petals' | 'roses' | 'leaves' | 'embers' | 'gold' | 'fireflies' | 'motes'
  Scene: (p: { className?: string; layer?: 'base' | 'sprite' }) => ReactNode
  /** pixel-art motion for the dithered scene (y values are fractions of the scene height) */
  fx: SceneFx
}

const v = (bg: string, s: string, s2: string, s3: string, b: string, b2: string, t: string, t2: string, t3: string,
  a: string, ah: string, aRgb: string, on: string, sec: string, secRgb: string) => ({
  '--bg': bg, '--surface': s, '--surface2': s2, '--surface3': s3, '--border': b, '--border2': b2,
  '--text': t, '--text2': t2, '--text3': t3,
  '--accent': a, '--accent-h': ah, '--accent-s': `rgba(${aRgb},0.15)`, '--accent-t': `rgba(${aRgb},0.07)`, '--on-accent': on,
  '--secondary': sec, '--secondary-h': sec, '--secondary-s': `rgba(${secRgb},0.13)`, '--secondary-t': `rgba(${secRgb},0.06)`,
  '--world-rgb': aRgb,
})

export const WORLDS: World[] = [
  { id: 'angkor', en: 'Angkor Wat', km: 'អង្គរវត្ត', blurb: 'Lotus-bud towers at dawn over the moat',
    vars: v('#100e22', '#17142e', '#1e1a3a', '#282347', '#29234a', '#3b3266', '#efe4cf', '#b9ad9a', '#8f86a3', '#e0a83a', '#ebb952', '224,168,58', '#1d1305', '#ea8fa2', '234,143,162'),
    particles: 'petals', Scene: AngkorScene , fx: { ripple: [0.755, 1], twinkle: [0.02, 0.4] } },
  { id: 'banteay', en: 'Banteay Srei', km: 'បន្ទាយស្រី', blurb: 'Pink sandstone pediments and flame kbach',
    vars: v('#1c0f17', '#261520', '#301b28', '#3d2333', '#3a2130', '#553044', '#f3e6dc', '#cdb3aa', '#a08590', '#f0927c', '#f5a693', '240,146,124', '#2a0e0a', '#f4c28a', '244,194,138'),
    particles: 'roses', Scene: BanteayScene , fx: { sweep: 'light' } },
  { id: 'taprohm', en: 'Ta Prohm', km: 'តាព្រហ្ម', blurb: 'Jungle temple held by silk-cotton roots',
    vars: v('#0b1612', '#111f19', '#172a21', '#1f362b', '#1f342a', '#2e4a3c', '#e8eedc', '#b4c2ab', '#83977f', '#a6d17a', '#b8de8f', '166,209,122', '#13200a', '#e0be7a', '224,190,122'),
    particles: 'leaves', Scene: TaProhmScene , fx: { sweep: 'light', twinkle: [0, 0.25] } },
  { id: 'sbek', en: 'Sbek Thom', km: 'ស្បែកធំ', blurb: 'Leather shadow panels before a fire-lit screen',
    vars: v('#0e0a07', '#17110c', '#211810', '#2c2016', '#2a1f15', '#3d2d1e', '#f6e9d6', '#c9b49a', '#9a8670', '#f5a04a', '#f8b469', '245,160,74', '#2a1203', '#ffd59a', '255,213,154'),
    particles: 'embers', Scene: SbekScene , fx: { flicker: true } },
  { id: 'palace', en: 'Royal Palace', km: 'ព្រះបរមរាជវាំង', blurb: 'Tiered roofs with chovear finials',
    vars: v('#0d1230', '#141a3d', '#1b2249', '#252d5a', '#232b55', '#34407a', '#f0ead8', '#bdb8c9', '#8e8db0', '#f2c94c', '#f6d670', '242,201,76', '#241a02', '#ef8a5a', '239,138,90'),
    particles: 'gold', Scene: PalaceScene , fx: { sweep: 'light', twinkle: [0, 0.35] } },
  { id: 'tonlesap', en: 'Tonle Sap', km: 'ទន្លេសាប', blurb: 'Stilt houses and the dragon-boat race',
    vars: v('#0a1620', '#10202c', '#162a38', '#1e3646', '#1d3242', '#2b4a5e', '#e6f0ee', '#a9c0c2', '#7e999e', '#5fd0c6', '#7fdcd4', '95,208,198', '#062220', '#f39a5a', '243,154,90'),
    particles: 'fireflies', Scene: TonleSapScene , fx: { ripple: [0.74, 1], sprite: 22 } },
  { id: 'bayon', en: 'Bayon', km: 'បាយ័ន', blurb: 'Serene stone faces watching every direction',
    vars: v('#10141a', '#171d25', '#1e2630', '#28323e', '#26303c', '#384656', '#ecefe8', '#b5bcb4', '#87918c', '#8fc4ff', '#a9d2ff', '143,196,255', '#0a1a2e', '#c9c2a4', '201,194,164'),
    particles: 'motes', Scene: BayonScene , fx: { sweep: 'shadow' } },
  { id: 'mural', en: 'Reamker mural', km: 'រឿងរាមកេរ្តិ៍', blurb: 'Silver Pagoda gallery paintings: red earth, gold pavilions',
    vars: v('#140f24', '#1c1530', '#241b3c', '#2f234c', '#2e2448', '#43356a', '#f2e9dc', '#c4b6c2', '#958aa8', '#ff8f6b', '#ffa587', '255,143,107', '#2a0d05', '#e0a83a', '224,168,58'),
    particles: 'gold', Scene: MuralScene , fx: { ripple: [0.56, 0.69] } },
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
