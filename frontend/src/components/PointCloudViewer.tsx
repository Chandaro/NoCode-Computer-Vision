/**
 * PointCloudViewer — interactive Three.js WebGL viewer for {n, pos, col} JSON.
 *
 * Props:
 *   fetchUrl   — full URL that returns JSON { n, pos: number[], col: number[] }
 *   pointSize  — dot radius (1–6, default 2)
 *
 * Controls: left drag = rotate · scroll = zoom · right drag = pan
 */
import { useEffect, useRef, useState } from 'react'
import { Loader, RotateCcw } from 'lucide-react'
import api from '../api'

interface Props {
  fetchUrl:  string
  pointSize: number
}

export default function PointCloudViewer({ fetchUrl, pointSize }: Props) {
  const mountRef  = useRef<HTMLDivElement>(null)
  const rendRef   = useRef<any>(null)
  const [loading, setLoading]  = useState(true)
  const [count,   setCount]    = useState(0)
  const [error,   setError]    = useState('')

  const resetCamera = () => {
    const r = rendRef.current
    if (r?.camera && r?.controls) {
      r.camera.position.set(0, 0, 2)
      r.controls.target.set(0, 0, 0)
      r.controls.update()
    }
  }

  useEffect(() => {
    const container = mountRef.current
    if (!container) return
    let cancelled = false
    let cleanup: (() => void) | undefined

    ;(async () => {
      try {
        // ── Fetch data ───────────────────────────────────────────────────
        const res = await api.get(fetchUrl)
        if (cancelled) return
        const { n, pos, col } = res.data as { n: number; pos: number[]; col: number[] }
        setCount(Number(n) || 0)
        if (!n || !pos?.length) throw new Error("No point cloud data returned")

        const posArr = new Float32Array(pos)
        const colArr = new Float32Array(col)

        // ── Three.js ──────────────────────────────────────────────────────
        const THREE = await import('three')
        const { OrbitControls } = await import(
          'three/examples/jsm/controls/OrbitControls.js' as any
        )
        if (cancelled) return

        const W = container.clientWidth
        const H = container.clientHeight

        const scene    = new THREE.Scene()
        scene.background = new THREE.Color(0x0d1117)

        const camera   = new THREE.PerspectiveCamera(60, W / H, 0.001, 100)
        camera.position.set(0, 0, 2)

        const renderer = new THREE.WebGLRenderer({ antialias: true })
        renderer.setSize(W, H)
        renderer.setPixelRatio(window.devicePixelRatio)
        container.appendChild(renderer.domElement)

        const controls = new OrbitControls(camera, renderer.domElement)
        controls.enableDamping = true
        controls.dampingFactor = 0.06
        controls.screenSpacePanning = true

        // ── Geometry ──────────────────────────────────────────────────────
        const geo = new THREE.BufferGeometry()
        geo.setAttribute('position', new THREE.BufferAttribute(posArr, 3))
        geo.setAttribute('color',    new THREE.BufferAttribute(colArr, 3))
        geo.computeBoundingBox()
        geo.computeBoundingSphere()

        // Centre at origin
        const box    = geo.boundingBox!
        const centre = new THREE.Vector3()
        box.getCenter(centre)
        geo.translate(-centre.x, -centre.y, -centre.z)

        // Fit camera to bounding sphere
        const radius = geo.boundingSphere?.radius ?? 1
        camera.position.set(0, 0, radius * 2.5)
        controls.minDistance = radius * 0.05
        controls.maxDistance = radius * 12

        const mat = new THREE.PointsMaterial({
          size: pointSize * 0.0015,
          vertexColors: true,
          sizeAttenuation: true,
        })
        scene.add(new THREE.Points(geo, mat))
        scene.add(new THREE.AxesHelper(radius * 0.3))

        // ── Animation ─────────────────────────────────────────────────────
        let animId = 0
        const animate = () => {
          animId = requestAnimationFrame(animate)
          controls.update()
          renderer.render(scene, camera)
        }
        animate()

        const onResize = () => {
          const w = container.clientWidth, h = container.clientHeight
          camera.aspect = w / h
          camera.updateProjectionMatrix()
          renderer.setSize(w, h)
        }
        window.addEventListener('resize', onResize)

        rendRef.current = { camera, controls, mat, renderer, animId }
        setLoading(false)

        cleanup = () => {
          cancelAnimationFrame(animId)
          window.removeEventListener('resize', onResize)
          controls.dispose(); renderer.dispose(); geo.dispose(); mat.dispose()
          if (container.contains(renderer.domElement))
            container.removeChild(renderer.domElement)
        }
      } catch (e: any) {
        if (!cancelled) {
          setError(e?.response?.data?.detail ?? e?.message ?? 'Failed to load')
          setLoading(false)
        }
      }
    })()

    return () => {
      cancelled = true
      cleanup?.()
    }
  }, [fetchUrl])

  // Live point size update
  useEffect(() => {
    if (rendRef.current?.mat)
      rendRef.current.mat.size = pointSize * 0.0015
  }, [pointSize])

  return (
    <div style={{ position: 'relative' }}>
      <div ref={mountRef} style={{
        width: '100%', height: 500, borderRadius: 8, overflow: 'hidden',
        border: '1px solid var(--border)', background: '#0d1117',
      }} />

      {loading && !error && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center',
          background: 'rgba(13,17,23,0.88)', borderRadius: 8, gap: 10 }}>
          <Loader size={22} className="animate-spin" style={{ color: 'var(--accent)' }} />
          <p style={{ fontSize: 12, color: 'var(--text2)' }}>Building 3D view…</p>
        </div>
      )}

      {error && (
        <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center',
          justifyContent: 'center', background: 'rgba(13,17,23,0.88)', borderRadius: 8 }}>
          <p style={{ fontSize: 12, color: '#f85149', padding: 24, textAlign: 'center' }}>{error}</p>
        </div>
      )}

      {!loading && !error && (
        <>
          <div style={{ position: 'absolute', top: 10, left: 12,
            fontSize: 11, color: 'rgba(255,255,255,0.45)',
            fontFamily: 'JetBrains Mono, monospace', pointerEvents: 'none' }}>
            {(count ?? 0).toLocaleString()} pts
          </div>
          <button onClick={resetCamera} title="Reset camera"
            style={{ position: 'absolute', top: 10, right: 10,
              padding: '5px 9px', border: '1px solid rgba(255,255,255,0.15)',
              borderRadius: 5, background: 'rgba(0,0,0,0.55)',
              color: 'rgba(255,255,255,0.7)', cursor: 'pointer',
              fontSize: 11, display: 'flex', alignItems: 'center', gap: 4 }}>
            <RotateCcw size={11} /> Reset
          </button>
          <div style={{ position: 'absolute', bottom: 10, left: 12,
            fontSize: 10, color: 'rgba(255,255,255,0.35)', pointerEvents: 'none' }}>
            🖱 Left drag: rotate · Scroll: zoom · Right drag: pan
          </div>
        </>
      )}
    </div>
  )
}
