import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import * as THREE from 'three'
import api from '../api'
import { PageHeader, Btn, Badge } from '../components/ui'
import { Cpu } from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────

type LayerType =
  | 'conv2d' | 'batchnorm2d' | 'maxpool2d' | 'avgpool2d'
  | 'relu' | 'gelu' | 'sigmoid' | 'dropout' | 'flatten' | 'linear'

interface Layer {
  id: string
  type: LayerType
  params: Record<string, number>
}

interface CustomConfig {
  id: number
  project_id: number
  name: string
  layers: Layer[]
  input_h: number
  input_w: number
  created_at: string
}

interface CustomRun {
  id: number
  config_id: number
  project_id: number
  status: string
  epochs: number
  batch: number
  lr: number
  model_path: string
  results: Record<string, unknown>
  created_at: string
}

// ── Constants ─────────────────────────────────────────────────────────────────

const LAYER_COLORS: Record<string, number> = {
  input:       0x8b8b9a,
  conv2d:      0x5865f2,
  batchnorm2d: 0x22c55e,
  maxpool2d:   0xf97316,
  avgpool2d:   0xfb923c,
  relu:        0xef4444,
  gelu:        0xf43f5e,
  sigmoid:     0xec4899,
  dropout:     0xeab308,
  flatten:     0xa855f7,
  linear:      0x06b6d4,
}

const LAYER_CSS: Record<string, string> = {
  input:       '#8b8b9a',
  conv2d:      '#5865f2',
  batchnorm2d: '#22c55e',
  maxpool2d:   '#f97316',
  avgpool2d:   '#fb923c',
  relu:        '#ef4444',
  gelu:        '#f43f5e',
  sigmoid:     '#ec4899',
  dropout:     '#eab308',
  flatten:     '#a855f7',
  linear:      '#06b6d4',
}

const ALL_LAYER_TYPES: LayerType[] = [
  'conv2d', 'batchnorm2d', 'maxpool2d', 'avgpool2d',
  'relu', 'gelu', 'sigmoid', 'dropout', 'flatten', 'linear',
]

const DEFAULT_PARAMS: Record<LayerType, Record<string, number>> = {
  conv2d:      { filters: 16, kernel_size: 3, stride: 1, padding: 1 },
  batchnorm2d: {},
  maxpool2d:   { kernel_size: 2, stride: 2 },
  avgpool2d:   { kernel_size: 2, stride: 2 },
  relu:        {},
  gelu:        {},
  sigmoid:     {},
  dropout:     { p: 0.5 },
  flatten:     {},
  linear:      { out_features: 128 },
}

const DEFAULT_LAYERS: Layer[] = [
  { id: 'l1', type: 'conv2d',   params: { filters: 16, kernel_size: 3, stride: 1, padding: 1 } },
  { id: 'l2', type: 'relu',     params: {} },
  { id: 'l3', type: 'maxpool2d',params: { kernel_size: 2, stride: 2 } },
  { id: 'l4', type: 'conv2d',   params: { filters: 32, kernel_size: 3, stride: 1, padding: 1 } },
  { id: 'l5', type: 'relu',     params: {} },
  { id: 'l6', type: 'maxpool2d',params: { kernel_size: 2, stride: 2 } },
  { id: 'l7', type: 'flatten',  params: {} },
  { id: 'l8', type: 'linear',   params: { out_features: 128 } },
]

const SPACING = 3.0

// ── Shape computation ─────────────────────────────────────────────────────────

function computeOutputShape(layer: Layer, shape: number[]): number[] {
  const [C, H = 1, W = 1] = shape
  switch (layer.type) {
    case 'conv2d': {
      const { filters = 32, kernel_size = 3, stride = 1, padding = 1 } = layer.params
      return [
        filters,
        Math.floor((H + 2 * padding - kernel_size) / stride + 1),
        Math.floor((W + 2 * padding - kernel_size) / stride + 1),
      ]
    }
    case 'maxpool2d':
    case 'avgpool2d': {
      const { kernel_size = 2 } = layer.params
      const s = layer.params.stride ?? kernel_size
      return [C, Math.max(1, Math.floor(H / s)), Math.max(1, Math.floor(W / s))]
    }
    case 'flatten':
      return [shape.reduce((a, b) => a * b, 1)]
    case 'linear':
      return [layer.params.out_features ?? 128]
    default:
      return shape
  }
}

function isValidShape(shape: number[]): boolean {
  return shape.every(d => d > 0 && isFinite(d))
}

function computeAllShapes(layers: Layer[], inputH: number, inputW: number) {
  const shapes: number[][] = []
  let shape: number[] = [3, inputH, inputW]
  let valid = true
  // Input shape
  shapes.push(shape)
  for (const layer of layers) {
    if (!valid) {
      shapes.push([-1])
      continue
    }
    const next = computeOutputShape(layer, shape)
    if (!isValidShape(next)) { valid = false; shapes.push([-1]); continue }
    shape = next
    shapes.push(shape)
  }
  return shapes
}

function estimateParams(layers: Layer[], inputH: number, inputW: number): number {
  let total = 0
  let shape: number[] = [3, inputH, inputW]
  let valid = true
  for (const layer of layers) {
    if (!valid) break
    const next = computeOutputShape(layer, shape)
    if (!isValidShape(next)) { valid = false; break }
    if (layer.type === 'conv2d') {
      const { filters = 32, kernel_size = 3 } = layer.params
      total += filters * shape[0] * kernel_size * kernel_size + filters
    } else if (layer.type === 'batchnorm2d') {
      total += shape[0] * 4
    } else if (layer.type === 'linear') {
      const inF = shape.reduce((a, b) => a * b, 1)
      const outF = layer.params.out_features ?? 128
      total += inF * outF + outF
    }
    shape = next
  }
  return total
}

function uid(): string {
  return Math.random().toString(36).slice(2, 9)
}

const LAYER_DISPLAY: Record<string, string> = {
  linear: 'linear (FC)',
}

// ── Three.js: create label sprite ─────────────────────────────────────────────

function makeLabel(text1: string, text2: string, color: string): THREE.Sprite {
  const W = 256, H = 72
  const canvas = document.createElement('canvas')
  canvas.width  = W
  canvas.height = H
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, W, H)
  ctx.font      = 'bold 22px "JetBrains Mono", monospace'
  ctx.fillStyle = color
  ctx.textAlign = 'center'
  ctx.fillText(text1, W / 2, 26)
  ctx.font      = '17px "JetBrains Mono", monospace'
  ctx.fillStyle = 'rgba(200,200,220,0.7)'
  ctx.fillText(text2, W / 2, 52)
  const tex    = new THREE.CanvasTexture(canvas)
  const mat    = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false })
  const sprite = new THREE.Sprite(mat)
  sprite.scale.set((W / H) * 0.9, 0.9, 1)
  return sprite
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
}

// ── Three.js scene builder ────────────────────────────────────────────────────

function buildScene(
  scene: THREE.Scene,
  layers: Layer[],
  inputH: number,
  inputW: number,
) {
  // Remove all previous objects
  while (scene.children.length > 0) scene.remove(scene.children[0])

  const shapes = computeAllShapes(layers, inputH, inputW)
  // shapes[0] = input, shapes[i+1] = output of layers[i]
  // We display shapes[0..shapes.length-1] as "feature map slots"
  // Between slot i and slot i+1 there is layers[i]

  const MAX_PLANES = 10
  const NORM_MAX   = 1.5  // max plane size in world units

  for (let si = 0; si < shapes.length; si++) {
    const shape = shapes[si]
    const z     = -si * SPACING

    // Determine layer type for color
    let layerType = si === 0 ? 'input' : layers[si - 1].type
    const colorHex = LAYER_COLORS[layerType] ?? 0x8b8b9a
    const colorCSS = LAYER_CSS[layerType] ?? '#8b8b9a'
    const valid = isValidShape(shape)

    if (!valid) {
      // Red error marker
      const geo = new THREE.BoxGeometry(0.4, 0.4, 0.4)
      const mat = new THREE.MeshBasicMaterial({ color: 0xef4444, wireframe: true })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(0, 0, z)
      scene.add(mesh)
      continue
    }

    if (shape.length === 1) {
      // 1D shape (after flatten/linear) — draw tall thin rectangle
      const N      = shape[0]
      const height = Math.min(N / 64 * 2, 3)
      const width  = 0.15

      const geo = new THREE.PlaneGeometry(width, height)
      const mat = new THREE.MeshBasicMaterial({
        color:       colorHex,
        transparent: true,
        opacity:     0.35,
        side:        THREE.DoubleSide,
      })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(0, 0, z)
      scene.add(mesh)

      const edgesGeo = new THREE.EdgesGeometry(geo)
      const edgesMat = new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.7 })
      const edges    = new THREE.LineSegments(edgesGeo, edgesMat)
      edges.position.copy(mesh.position)
      scene.add(edges)

      // Label
      const labelText = si === 0 ? 'input' : layers[si - 1].type
      const shapeText = `${N}`
      const sprite    = makeLabel(labelText, shapeText, colorCSS)
      sprite.position.set(0, -(height / 2) - 0.5, z)
      scene.add(sprite)

    } else {
      // 3D shape [C, H, W]
      const [C, H, W] = shape
      const maxDim    = Math.max(H, W, 1)
      const scale     = NORM_MAX / maxDim
      const planeH    = H * scale
      const planeW    = W * scale
      const nPlanes   = Math.min(C, MAX_PLANES)

      for (let pi = 0; pi < nPlanes; pi++) {
        const ox  = pi * 0.06
        const oy  = pi * 0.02
        const oz  = pi * 0.05
        const opacity = 0.08 + (pi / nPlanes) * 0.18

        const geo     = new THREE.PlaneGeometry(planeW, planeH)
        const tex     = makeFeatureMapTexture(colorHex)
        const mat     = new THREE.MeshBasicMaterial({
          map:         tex,
          color:       colorHex,
          transparent: true,
          opacity,
          side:        THREE.DoubleSide,
        })
        const mesh = new THREE.Mesh(geo, mat)
        mesh.position.set(ox, oy, z + oz)
        scene.add(mesh)

        // Wireframe edges
        const edgesGeo = new THREE.EdgesGeometry(geo)
        const edgesMat = new THREE.LineBasicMaterial({
          color:       colorHex,
          transparent: true,
          opacity:     0.5,
        })
        const edges = new THREE.LineSegments(edgesGeo, edgesMat)
        edges.position.copy(mesh.position)
        scene.add(edges)
      }

      // Label
      const labelType  = si === 0 ? 'input' : layers[si - 1].type
      const shapeText  = shape.length === 3 ? `${C}×${H}×${W}` : `${shape[0]}`
      const sprite     = makeLabel(labelType, shapeText, colorCSS)
      sprite.position.set(0, -(planeH / 2) - 0.55, z)
      scene.add(sprite)
    }

    // Connector lines to previous slot
    if (si > 0) {
      const prevShape = shapes[si - 1]
      const prevZ     = -(si - 1) * SPACING
      const connColorHex = LAYER_COLORS[layers[si - 1].type] ?? 0x8b8b9a

      if (isValidShape(prevShape) && valid) {
        const getPlaneDims = (sh: number[]) => {
          if (sh.length === 1) return { w: 0.15, h: Math.min(sh[0] / 64 * 2, 3) }
          const [, H2, W2] = sh
          const maxD = Math.max(H2, W2, 1)
          const sc   = NORM_MAX / maxD
          return { w: W2 * sc, h: H2 * sc }
        }
        const prev = getPlaneDims(prevShape)
        const curr = getPlaneDims(shape)

        const corners: [THREE.Vector3, THREE.Vector3][] = [
          [
            new THREE.Vector3(-prev.w / 2, -prev.h / 2, prevZ),
            new THREE.Vector3(-curr.w / 2, -curr.h / 2, z),
          ],
          [
            new THREE.Vector3( prev.w / 2, -prev.h / 2, prevZ),
            new THREE.Vector3( curr.w / 2, -curr.h / 2, z),
          ],
          [
            new THREE.Vector3(-prev.w / 2,  prev.h / 2, prevZ),
            new THREE.Vector3(-curr.w / 2,  curr.h / 2, z),
          ],
          [
            new THREE.Vector3( prev.w / 2,  prev.h / 2, prevZ),
            new THREE.Vector3( curr.w / 2,  curr.h / 2, z),
          ],
        ]

        for (const [a, b] of corners) {
          const geo  = new THREE.BufferGeometry().setFromPoints([a, b])
          const mat  = new THREE.LineBasicMaterial({
            color:       connColorHex,
            transparent: true,
            opacity:     0.15,
          })
          scene.add(new THREE.Line(geo, mat))
        }
      }
    }
  }
}

// ── Layer Param Editor ────────────────────────────────────────────────────────

function ParamEditor({
  layer,
  onChange,
}: {
  layer: Layer
  onChange: (params: Record<string, number>) => void
}) {
  const p = layer.params

  const inp = (
    key: string, label: string,
    min: number, max: number, step = 1,
  ) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <label style={{ fontSize: 11, color: 'var(--text2)', width: 80, flexShrink: 0 }}>{label}</label>
      <input
        type="number" min={min} max={max} step={step}
        value={p[key] ?? 0}
        onChange={e => onChange({ ...p, [key]: Number(e.target.value) })}
        style={{
          flex: 1, padding: '4px 8px', background: 'var(--surface3)',
          border: '1px solid var(--border2)', borderRadius: 4,
          color: 'var(--text)', fontSize: 12, fontFamily: 'inherit',
        }}
      />
    </div>
  )

  if (layer.type === 'conv2d') return (
    <div>
      {inp('filters',     'Filters',     1, 512)}
      {inp('kernel_size', 'Kernel',      1, 7)}
      {inp('stride',      'Stride',      1, 4)}
      {inp('padding',     'Padding',     0, 3)}
    </div>
  )
  if (layer.type === 'maxpool2d' || layer.type === 'avgpool2d') return (
    <div>
      {inp('kernel_size', 'Kernel', 2, 4)}
      {inp('stride',      'Stride', 1, 4)}
    </div>
  )
  if (layer.type === 'dropout') return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <label style={{ fontSize: 11, color: 'var(--text2)', width: 80 }}>Dropout p</label>
        <input
          type="range" min={0} max={1} step={0.05}
          value={p.p ?? 0.5}
          onChange={e => onChange({ ...p, p: Number(e.target.value) })}
          style={{ flex: 1, accentColor: 'var(--accent)' }}
        />
        <span style={{ fontSize: 11, color: 'var(--text)', fontFamily: 'monospace', width: 32 }}>
          {(p.p ?? 0.5).toFixed(2)}
        </span>
      </div>
    </div>
  )
  if (layer.type === 'linear') return (
    <div>{inp('out_features', 'Out features', 1, 4096)}</div>
  )
  return <p style={{ fontSize: 11, color: 'var(--text3)', fontStyle: 'italic' }}>No parameters</p>
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function CustomModel() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  // Project info
  const [projectName, setProjectName] = useState('…')

  // Config state
  const [layers,   setLayers]   = useState<Layer[]>(DEFAULT_LAYERS)
  const [inputH,   setInputH]   = useState(64)
  const [inputW,   setInputW]   = useState(64)
  const [modelName, setModelName] = useState('My Model')
  const [savedConfig, setSavedConfig] = useState<CustomConfig | null>(null)
  const [loading, setLoading] = useState(false)

  // Selected layer
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Training state
  const [epochs, setEpochs] = useState(20)
  const [batch,  setBatch]  = useState(32)
  const [lr,     setLr]     = useState(0.001)
  const [activeRun, setActiveRun] = useState<CustomRun | null>(null)
  const [logs,   setLogs]   = useState<string[]>([])
  const logRef  = useRef<HTMLDivElement>(null)
  const sseRef  = useRef<EventSource | null>(null)

  // Add layer dropdown
  const [showAddMenu, setShowAddMenu] = useState(false)

  // Three.js refs
  const canvasRef    = useRef<HTMLDivElement>(null)
  const rendererRef  = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef     = useRef<THREE.Scene | null>(null)
  const cameraRef    = useRef<THREE.PerspectiveCamera | null>(null)
  const animFrameRef = useRef<number>(0)

  // Orbit state
  const orbitRef = useRef({ theta: 0.5, phi: 1.0, radius: 10, dragging: false, lastX: 0, lastY: 0 })

  // Debounce timer
  const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── Load project + configs ────────────────────────────────────────────────

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [pRes, cfgRes] = await Promise.all([
          api.get(`/projects/${projectId}`),
          api.get(`/projects/${projectId}/custom/configs`),
        ])
        setProjectName(pRes.data.name)
        const cfgs: CustomConfig[] = cfgRes.data
        if (cfgs.length > 0) {
          const cfg = cfgs[cfgs.length - 1]
          setSavedConfig(cfg)
          setModelName(cfg.name)
          setLayers(cfg.layers.length > 0 ? cfg.layers : DEFAULT_LAYERS)
          setInputH(cfg.input_h)
          setInputW(cfg.input_w)
        }
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [projectId])

  // ── Auto-save debounce ────────────────────────────────────────────────────

  const triggerAutoSave = useCallback(() => {
    if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current)
    autoSaveTimer.current = setTimeout(async () => {
      try {
        const body = { name: modelName, layers, input_h: inputH, input_w: inputW }
        if (savedConfig) {
          const res = await api.put(`/projects/${projectId}/custom/configs/${savedConfig.id}`, body)
          setSavedConfig(res.data)
        } else {
          const res = await api.post(`/projects/${projectId}/custom/configs`, body)
          setSavedConfig(res.data)
        }
      } catch (e) {
        // Silently ignore
      }
    }, 1000)
  }, [projectId, modelName, layers, inputH, inputW, savedConfig])

  useEffect(() => {
    if (!loading) triggerAutoSave()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, inputH, inputW, modelName])

  // ── Three.js setup ────────────────────────────────────────────────────────

  useEffect(() => {
    if (!canvasRef.current) return

    const container = canvasRef.current
    const w = container.clientWidth
    const h = container.clientHeight

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(w, h)
    renderer.setClearColor(0x0d0d0f, 1)
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const scene = new THREE.Scene()
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 500)
    cameraRef.current = camera

    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate)
      renderer.render(scene, camera)
    }
    animate()

    // Resize observer
    const ro = new ResizeObserver(() => {
      const nw = container.clientWidth
      const nh = container.clientHeight
      renderer.setSize(nw, nh)
      camera.aspect = nw / nh
      camera.updateProjectionMatrix()
    })
    ro.observe(container)

    return () => {
      cancelAnimationFrame(animFrameRef.current)
      ro.disconnect()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  // Rebuild scene when layers / input size change
  useEffect(() => {
    if (!sceneRef.current || !cameraRef.current) return
    buildScene(sceneRef.current, layers, inputH, inputW)
    updateCamera()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, inputH, inputW])

  // ── Camera orbit ──────────────────────────────────────────────────────────

  const updateCamera = useCallback(() => {
    const camera = cameraRef.current
    if (!camera) return
    const { theta, phi, radius } = orbitRef.current
    const totalSlots = layers.length + 1
    const target = new THREE.Vector3(0, 0, -(totalSlots - 1) * SPACING / 2)
    camera.position.set(
      target.x + radius * Math.sin(phi) * Math.sin(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.cos(theta),
    )
    camera.lookAt(target)
  }, [layers.length])

  // Re-run when layers.length changes
  useEffect(() => { updateCamera() }, [layers.length, updateCamera])

  // ── Mouse orbit handlers ──────────────────────────────────────────────────

  useEffect(() => {
    const container = canvasRef.current
    if (!container) return

    const onDown = (e: MouseEvent) => {
      orbitRef.current.dragging = true
      orbitRef.current.lastX    = e.clientX
      orbitRef.current.lastY    = e.clientY
    }
    const onMove = (e: MouseEvent) => {
      if (!orbitRef.current.dragging) return
      const dx = e.clientX - orbitRef.current.lastX
      const dy = e.clientY - orbitRef.current.lastY
      orbitRef.current.lastX  = e.clientX
      orbitRef.current.lastY  = e.clientY
      orbitRef.current.theta -= dx * 0.005
      orbitRef.current.phi    = Math.max(0.1, Math.min(Math.PI - 0.1, orbitRef.current.phi + dy * 0.005))
      updateCamera()
    }
    const onUp   = () => { orbitRef.current.dragging = false }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      orbitRef.current.radius = Math.max(3, Math.min(25, orbitRef.current.radius + e.deltaY * 0.01))
      updateCamera()
    }

    container.addEventListener('mousedown', onDown)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    container.addEventListener('wheel', onWheel, { passive: false })

    return () => {
      container.removeEventListener('mousedown', onDown)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      container.removeEventListener('wheel', onWheel)
    }
  }, [updateCamera])

  // ── Log auto-scroll ───────────────────────────────────────────────────────

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  // ── Training ──────────────────────────────────────────────────────────────

  const startTraining = async () => {
    if (!savedConfig) {
      alert('Config is still saving, please wait a moment.')
      return
    }
    setLogs([])
    try {
      const res = await api.post(`/projects/${projectId}/custom/runs`, {
        config_id: savedConfig.id,
        epochs,
        batch,
        lr,
      })
      const run: CustomRun = res.data
      setActiveRun(run)
      openSSE(run.id)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      setLogs([`Error: ${msg}`])
    }
  }

  const openSSE = (runId: number) => {
    sseRef.current?.close()
    const es = new EventSource(`/api/projects/${projectId}/custom/runs/${runId}/stream`)
    sseRef.current = es
    es.onmessage = (e) => {
      const data = e.data as string
      if (data === '__END__') {
        es.close()
        // Refresh run status
        api.get(`/projects/${projectId}/custom/runs/${runId}`)
          .then(r => setActiveRun(r.data))
          .catch(() => {})
        return
      }
      if (data.startsWith('__PROGRESS__:')) {
        const parts = data.split(':')
        const [ep, total] = parts[1].split('/')
        const acc = parseFloat(parts[2])
        setActiveRun(prev => prev ? { ...prev, status: 'running' } : prev)
        setLogs(prev => [...prev.filter(l => !l.startsWith('[PROG]')),
          `[PROG] Epoch ${ep}/${total} — acc ${(acc * 100).toFixed(1)}%`])
        return
      }
      if (data === '__FAILED__') {
        setActiveRun(prev => prev ? { ...prev, status: 'failed' } : prev)
        return
      }
      if (data.startsWith('__DONE__:')) {
        setActiveRun(prev => prev ? { ...prev, status: 'done' } : prev)
        return
      }
      setLogs(prev => [...prev, data])
    }
    es.onerror = () => {
      es.close()
      setLogs(prev => [...prev, 'Stream disconnected'])
    }
  }

  // Cleanup SSE on unmount
  useEffect(() => () => { sseRef.current?.close() }, [])

  // ── Layer editor helpers ──────────────────────────────────────────────────

  const addLayer = (type: LayerType) => {
    const layer: Layer = { id: uid(), type, params: { ...DEFAULT_PARAMS[type] } }
    setLayers(prev => [...prev, layer])
    setSelectedId(layer.id)
    setShowAddMenu(false)
  }

  const deleteLayer = (layerId: string) => {
    setLayers(prev => prev.filter(l => l.id !== layerId))
    setSelectedId(prev => prev === layerId ? null : prev)
  }

  const moveLayer = (index: number, dir: -1 | 1) => {
    setLayers(prev => {
      const next = [...prev]
      const target = index + dir
      if (target < 0 || target >= next.length) return prev
      ;[next[index], next[target]] = [next[target], next[index]]
      return next
    })
  }

  const updateLayerParams = (layerId: string, params: Record<string, number>) => {
    setLayers(prev => prev.map(l => l.id === layerId ? { ...l, params } : l))
  }

  // Compute shapes for all layers (for display)
  const allShapes = computeAllShapes(layers, inputH, inputW)
  // allShapes[0] = input, allShapes[i+1] = output of layers[i]
  const totalParams = estimateParams(layers, inputH, inputW)

  const selectedLayer = layers.find(l => l.id === selectedId) ?? null

  // Status badge color
  const statusColor = (s: string): 'green' | 'yellow' | 'red' | 'gray' | 'blue' => {
    if (s === 'done')    return 'green'
    if (s === 'running') return 'blue'
    if (s === 'failed')  return 'red'
    if (s === 'pending') return 'yellow'
    return 'gray'
  }

  // ── Render ────────────────────────────────────────────────────────────────

  const panelStyle = {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    display: 'flex',
    flexDirection: 'column' as const,
    overflow: 'hidden',
  }

  const sectionLabel = (text: string) => (
    <p style={{
      fontSize: 10, fontWeight: 600, letterSpacing: '0.08em',
      textTransform: 'uppercase' as const, color: 'var(--text3)',
      padding: '10px 12px 4px', flexShrink: 0,
    }}>{text}</p>
  )

  return (
    <div style={{ maxWidth: '100%', height: 'calc(100vh - 80px)', display: 'flex', flexDirection: 'column' }}>
      <PageHeader
        back={() => navigate(`/projects/${projectId}/images`)}
        title={`Conv Builder — ${projectName}`}
        subtitle="Build a custom CNN architecture and train it"
        actions={
          <Btn variant="ghost" size="sm" onClick={() => navigate(`/projects/${projectId}/images`)}>
            <Cpu size={13} /> Back to images
          </Btn>
        }
      />

      <div style={{ flex: 1, display: 'flex', gap: 8, overflow: 'hidden', minHeight: 0 }}>

        {/* ── Left panel: Layer Editor ───────────────────────────────────── */}
        <div style={{ ...panelStyle, width: 220, flexShrink: 0 }}>
          {sectionLabel('Model')}

          {/* Model name */}
          <div style={{ padding: '0 10px 8px' }}>
            <input
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              placeholder="Model name"
              style={{
                width: '100%', padding: '5px 8px',
                background: 'var(--surface2)', border: '1px solid var(--border)',
                borderRadius: 5, color: 'var(--text)', fontSize: 12,
                fontFamily: 'inherit',
              }}
            />
          </div>

          {/* Input size */}
          {sectionLabel('Input size')}
          <div style={{ padding: '0 10px 8px', display: 'flex', gap: 6 }}>
            {([['H', inputH, setInputH], ['W', inputW, setInputW]] as const).map(([lbl, val, setter]) => (
              <div key={lbl} style={{ flex: 1 }}>
                <label style={{ fontSize: 10, color: 'var(--text3)', display: 'block', marginBottom: 3 }}>{lbl}</label>
                <input
                  type="number" min={32} max={256} step={32}
                  value={val}
                  onChange={e => setter(Number(e.target.value))}
                  style={{
                    width: '100%', padding: '4px 6px',
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 4, color: 'var(--text)', fontSize: 12,
                  }}
                />
              </div>
            ))}
          </div>

          {sectionLabel('Layers')}

          {/* Layer list */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '0 6px' }}>
            {/* Input node */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '5px 6px', borderRadius: 5, marginBottom: 2,
              background: 'var(--surface2)',
            }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: LAYER_CSS.input, flexShrink: 0 }} />
              <span style={{ fontSize: 11, color: 'var(--text2)', flex: 1 }}>input</span>
              <span style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'monospace' }}>
                {`${3}×${inputH}×${inputW}`}
              </span>
            </div>

            {layers.map((layer, index) => {
              const outShape = allShapes[index + 1]
              const valid    = isValidShape(outShape)
              const isSelected = layer.id === selectedId
              const shapeStr  = !valid ? '⚠ invalid'
                : outShape.length === 1 ? `${outShape[0]}`
                : `${outShape[0]}×${outShape[1]}×${outShape[2]}`

              return (
                <div
                  key={layer.id}
                  onClick={() => setSelectedId(isSelected ? null : layer.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    padding: '5px 6px', borderRadius: 5, marginBottom: 2,
                    cursor: 'pointer',
                    background: isSelected ? 'rgba(88,101,242,0.12)' : 'transparent',
                    border: `1px solid ${isSelected ? 'var(--accent)' : 'transparent'}`,
                    transition: 'background 0.1s',
                  }}
                >
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                    background: valid ? (LAYER_CSS[layer.type] ?? '#8b8b9a') : '#ef4444',
                  }} />
                  <span style={{ fontSize: 11, color: valid ? 'var(--text)' : 'var(--danger)', flex: 1, minWidth: 0 }}>
                    {LAYER_DISPLAY[layer.type] ?? layer.type}
                  </span>
                  <span style={{ fontSize: 9, color: valid ? 'var(--text3)' : 'var(--danger)', fontFamily: 'monospace', maxWidth: 55, textAlign: 'right' }}>
                    {shapeStr}
                  </span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <button
                      onClick={e => { e.stopPropagation(); moveLayer(index, -1) }}
                      disabled={index === 0}
                      style={{ fontSize: 8, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '1px 2px', lineHeight: 1 }}
                    >▲</button>
                    <button
                      onClick={e => { e.stopPropagation(); moveLayer(index, 1) }}
                      disabled={index === layers.length - 1}
                      style={{ fontSize: 8, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '1px 2px', lineHeight: 1 }}
                    >▼</button>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); deleteLayer(layer.id) }}
                    style={{ fontSize: 10, color: 'var(--danger)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 2px', lineHeight: 1 }}
                  >✕</button>
                </div>
              )
            })}
          </div>

          {/* Param editor for selected layer */}
          {selectedLayer && (
            <div style={{ borderTop: '1px solid var(--border)', padding: '8px 10px', flexShrink: 0 }}>
              <p style={{ fontSize: 10, color: 'var(--text3)', marginBottom: 6, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                {LAYER_DISPLAY[selectedLayer.type] ?? selectedLayer.type} params
              </p>
              <ParamEditor
                layer={selectedLayer}
                onChange={params => updateLayerParams(selectedLayer.id, params)}
              />
            </div>
          )}

          {/* Total params */}
          <div style={{ borderTop: '1px solid var(--border)', padding: '6px 10px', flexShrink: 0 }}>
            <p style={{ fontSize: 10, color: 'var(--text3)' }}>
              Est. params: <span style={{ color: 'var(--text)', fontFamily: 'monospace' }}>{totalParams.toLocaleString()}</span>
            </p>
          </div>

          {/* Add layer button */}
          <div style={{ padding: '8px 10px', borderTop: '1px solid var(--border)', flexShrink: 0, position: 'relative' }}>
            <button
              onClick={() => setShowAddMenu(prev => !prev)}
              style={{
                width: '100%', padding: '6px', borderRadius: 5,
                background: 'var(--surface2)', border: '1px solid var(--border2)',
                color: 'var(--text2)', fontSize: 12, cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
              }}
            >
              + Add Layer
            </button>
            {showAddMenu && (
              <div style={{
                position: 'absolute', bottom: '100%', left: 10, right: 10,
                background: 'var(--surface2)', border: '1px solid var(--border2)',
                borderRadius: 6, overflow: 'hidden', zIndex: 100,
              }}>
                {ALL_LAYER_TYPES.map(lt => (
                  <button
                    key={lt}
                    onClick={() => addLayer(lt)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 7,
                      width: '100%', padding: '6px 10px',
                      background: 'transparent', border: 'none',
                      color: 'var(--text)', fontSize: 12, cursor: 'pointer',
                      textAlign: 'left',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'var(--surface3)')}
                    onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}
                  >
                    <span style={{ width: 7, height: 7, borderRadius: '50%', background: LAYER_CSS[lt] ?? '#8b8b9a', flexShrink: 0 }} />
                    {LAYER_DISPLAY[lt] ?? lt}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── Center: Three.js Canvas ─────────────────────────────────────── */}
        <div style={{ flex: 1, position: 'relative', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border)', background: '#0d0d0f' }}>
          <div ref={canvasRef} style={{ width: '100%', height: '100%', cursor: 'grab' }} />

          {/* Reset view button */}
          <button
            onClick={() => {
              orbitRef.current = { ...orbitRef.current, theta: 0.5, phi: 1.0, radius: 10 }
              updateCamera()
            }}
            style={{
              position: 'absolute', top: 10, right: 10,
              padding: '4px 10px', borderRadius: 5,
              background: 'rgba(15,15,18,0.85)', border: '1px solid var(--border2)',
              color: 'var(--text2)', fontSize: 11, cursor: 'pointer',
              backdropFilter: 'blur(4px)',
            }}
          >
            Reset View
          </button>

          {/* Help hint */}
          <div style={{
            position: 'absolute', bottom: 10, left: '50%', transform: 'translateX(-50%)',
            fontSize: 10, color: 'rgba(140,140,158,0.5)',
            pointerEvents: 'none',
          }}>
            Drag to orbit · Scroll to zoom
          </div>
        </div>

        {/* ── Right panel: Training ───────────────────────────────────────── */}
        <div style={{ ...panelStyle, width: 260, flexShrink: 0 }}>
          {sectionLabel('Training')}

          <div style={{ padding: '0 12px 10px', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { label: 'Epochs', val: epochs, set: setEpochs, min: 1,   max: 200, step: 1    },
              { label: 'Batch',  val: batch,  set: setBatch,  min: 4,   max: 128, step: 4    },
            ].map(({ label, val, set, min, max, step }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <label style={{ fontSize: 12, color: 'var(--text2)', width: 50, flexShrink: 0 }}>{label}</label>
                <input
                  type="number" min={min} max={max} step={step}
                  value={val}
                  onChange={e => set(Number(e.target.value))}
                  style={{
                    flex: 1, padding: '5px 8px',
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 5, color: 'var(--text)', fontSize: 12,
                  }}
                />
              </div>
            ))}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: 12, color: 'var(--text2)', width: 50, flexShrink: 0 }}>LR</label>
              <input
                type="number" min={0.00001} max={0.1} step={0.0001}
                value={lr}
                onChange={e => setLr(Number(e.target.value))}
                style={{
                  flex: 1, padding: '5px 8px',
                  background: 'var(--surface2)', border: '1px solid var(--border)',
                  borderRadius: 5, color: 'var(--text)', fontSize: 12,
                }}
              />
            </div>
          </div>

          <div style={{ padding: '0 12px 10px' }}>
            <Btn
              variant="primary"
              onClick={startTraining}
              disabled={activeRun?.status === 'running' || activeRun?.status === 'pending'}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {activeRun?.status === 'running' ? 'Training…' : 'Train'}
            </Btn>
          </div>

          {/* Status */}
          {activeRun && (
            <div style={{ padding: '0 12px 10px', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, color: 'var(--text2)' }}>Status:</span>
              <Badge color={statusColor(activeRun.status)}>{activeRun.status}</Badge>
            </div>
          )}

          {sectionLabel('Logs')}

          {/* Log terminal */}
          <div
            ref={logRef}
            style={{
              flex: 1, overflowY: 'auto', margin: '0 10px 10px',
              background: '#07070a', border: '1px solid var(--border)',
              borderRadius: 5, padding: '8px 10px',
              fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 1.65,
              minHeight: 0,
            }}
          >
            {logs.length === 0
              ? <span style={{ color: 'var(--text3)' }}>Waiting…</span>
              : logs.map((l, i) => (
                <div
                  key={i}
                  style={{
                    color: l.startsWith('Error') || l.includes('❌') ? 'var(--danger)'
                      : l.startsWith('[PROG]') ? 'var(--accent)'
                      : l.includes('Done') || l.includes('✅') ? 'var(--success)'
                      : 'var(--text2)',
                  }}
                >
                  {l}
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}
