import { useEffect, useRef } from 'react'
import * as THREE from 'three'

/*
 * SplatViewer: renders a real-world Gaussian-splat scene with Spark (MIT) inside
 * three.js. Scenes come from MapAnything in OpenCV axes (x right, y down,
 * z forward), so the group is turned 180° about X to make y point up.
 *
 * One renderer lives for the whole life of the viewer; changing the scene or
 * the selected object only swaps the splat meshes (re-creating the WebGL
 * renderer is slow and Spark does not survive it). With an object selected,
 * the object and the rest of the scene are separate meshes, so the object can
 * be highlighted, shown alone, or removed.
 */
export type ViewMode = 'highlight' | 'object' | 'hide' | 'all'

type Mesh = THREE.Object3D & { opacity: number; dispose: () => void }
interface Ctx {
  group: THREE.Group; camera: THREE.PerspectiveCamera
  controls: { target: THREE.Vector3; update: () => void }
  SplatMesh: new (o: { url: string }) => Mesh
  meshes: { object?: Mesh; rest?: Mesh; all?: Mesh }
}

export default function SplatViewer({ sceneUrl, objectUrl, restUrl, mode = 'all', center, radius, height = 460 }: {
  sceneUrl: string; objectUrl?: string; restUrl?: string; mode?: ViewMode
  center: number[]; radius: number; height?: number
}) {
  const hostRef = useRef<HTMLDivElement>(null)
  const ctxRef = useRef<Ctx | null>(null)
  const latest = useRef({ sceneUrl, objectUrl, restUrl, mode, center, radius })
  latest.current = { sceneUrl, objectUrl, restUrl, mode, center, radius }

  // ── renderer, camera and controls: created once ──
  useEffect(() => {
    const host = hostRef.current
    if (!host) return
    let disposed = false
    let cleanup = () => {}
    ;(async () => {
      const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark')
      const { OrbitControls } = await import('three/examples/jsm/controls/OrbitControls.js')
      if (disposed) return
      const w = host.clientWidth
      const renderer = new THREE.WebGLRenderer({ antialias: false })
      renderer.setPixelRatio(Math.min(2, window.devicePixelRatio))
      renderer.setSize(w, height)
      renderer.setClearColor(0x07060f, 1)
      host.appendChild(renderer.domElement)
      const scene = new THREE.Scene()
      const camera = new THREE.PerspectiveCamera(55, w / height, 0.01, 1000)
      scene.add(new SparkRenderer({ renderer }))
      const group = new THREE.Group()
      group.quaternion.set(1, 0, 0, 0)          // OpenCV -> three.js (flip Y and Z)
      scene.add(group)
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true
      controls.dampingFactor = 0.08

      ctxRef.current = { group, camera, controls, SplatMesh: SplatMesh as unknown as Ctx['SplatMesh'], meshes: {} }
      loadMeshes(ctxRef.current, latest.current)
      frameCamera(ctxRef.current, latest.current.center, latest.current.radius)

      let visible = true
      const io = new IntersectionObserver(([e]) => { visible = e.isIntersecting })
      io.observe(host)
      renderer.setAnimationLoop(() => {
        if (!visible || document.hidden) return
        controls.update()
        renderer.render(scene, camera)
      })
      const ro = new ResizeObserver(() => {
        const nw = host.clientWidth
        renderer.setSize(nw, height); camera.aspect = nw / height; camera.updateProjectionMatrix()
      })
      ro.observe(host)
      cleanup = () => {
        renderer.setAnimationLoop(null)
        io.disconnect(); ro.disconnect(); controls.dispose()
        if (ctxRef.current) clearMeshes(ctxRef.current)
        ctxRef.current = null
        renderer.dispose()
        renderer.domElement.remove()
      }
    })()
    return () => { disposed = true; cleanup() }
  }, [height])

  // ── swap meshes when the scene or selected object changes ──
  useEffect(() => {
    if (ctxRef.current) loadMeshes(ctxRef.current, latest.current)
  }, [sceneUrl, objectUrl, restUrl])

  // ── re-aim the camera when the focus changes ──
  const [cx, cy, cz] = center
  useEffect(() => {
    if (ctxRef.current) frameCamera(ctxRef.current, [cx, cy, cz], radius)
  }, [cx, cy, cz, radius])

  useEffect(() => { if (ctxRef.current) applyMode(ctxRef.current, mode) }, [mode])

  return <div ref={hostRef} style={{ width: '100%', height, borderRadius: 'var(--radius-lg)', overflow: 'hidden', cursor: 'grab', background: '#07060f' }} />
}

function clearMeshes(c: Ctx) {
  for (const m of Object.values(c.meshes)) if (m) { c.group.remove(m); m.dispose() }
  c.meshes = {}
}

function loadMeshes(c: Ctx, p: { sceneUrl: string; objectUrl?: string; restUrl?: string; mode: ViewMode }) {
  clearMeshes(c)
  if (p.objectUrl && p.restUrl) {
    const rest = new c.SplatMesh({ url: p.restUrl })
    const object = new c.SplatMesh({ url: p.objectUrl })
    c.group.add(rest, object)
    c.meshes = { rest, object }
  } else {
    const all = new c.SplatMesh({ url: p.sceneUrl })
    c.group.add(all)
    c.meshes = { all }
  }
  applyMode(c, p.mode)
}

function frameCamera(c: Ctx, center: number[], radius: number) {
  // start roughly where the first camera stood, looking into the scene
  const target = new THREE.Vector3(center[0], -center[1], -center[2])
  c.camera.near = Math.max(0.001, radius / 500); c.camera.far = radius * 60; c.camera.updateProjectionMatrix()
  c.camera.position.copy(target).add(new THREE.Vector3(0, radius * 0.15, radius * 1.15))
  c.controls.target.copy(target)
  c.controls.update()
}

function applyMode(c: Ctx, mode: ViewMode) {
  const { object, rest } = c.meshes
  if (!object || !rest) return
  // Spark draws every SplatMesh in the scene, so hide with opacity, not .visible
  object.opacity = mode === 'hide' ? 0 : 1
  rest.opacity = mode === 'object' ? 0 : mode === 'highlight' ? 0.18 : 1
}
