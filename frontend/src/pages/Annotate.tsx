import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  ChevronLeft, ChevronRight, Save, Trash2, RotateCcw,
  MousePointer2, Square, Hexagon, Crosshair,
  ZoomIn, ZoomOut, Maximize2, Copy
} from 'lucide-react'
import api, { type AnnData, type ImageItem, type Project } from '../api'

// ─── Constants ────────────────────────────────────────────���──────────────────
const COLORS = ['#ef4444','#f97316','#eab308','#22c55e','#06b6d4','#6366f1','#a855f7','#ec4899']
const H = 7       // handle size in screen px
const SNAP_PX = 14 // polygon close-snap distance in screen px
const DOT_PX  = 8  // point dot radius in screen px

// ─── Shape types ────────────────────────────────���──────────────────────────���─
type Tool = 'select' | 'bbox' | 'polygon' | 'point'

interface BBoxShape   { type: 'bbox';    class_id: number; x: number; y: number; w: number; h: number }
interface PolygonShape{ type: 'polygon'; class_id: number; pts: [number,number][] }
interface PointShape  { type: 'point';   class_id: number; x: number; y: number }
type Shape = BBoxShape | PolygonShape | PointShape

// ─── Drag state (stored in ref, not state, to avoid stale closures) ───────────
type Drag =
  | { kind: 'none' }
  | { kind: 'bbox-draw';   start: [number,number] }
  | { kind: 'move-shape';  idx: number; mx0: number; my0: number; orig: Shape }
  | { kind: 'move-vertex'; idx: number; vi: number; mx0: number; my0: number; orig: PolygonShape }
  | { kind: 'bbox-handle'; idx: number; handle: string; mx0: number; my0: number; orig: BBoxShape }
  | { kind: 'pan';         cx0: number; cy0: number; px0: number; py0: number }

// ─── Helpers ──────────────────────────���─────────────────────���────────────────
function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)) }

function pointInPolygon(px: number, py: number, pts: [number,number][]) {
  let inside = false
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i]; const [xj, yj] = pts[j]
    if ((yi > py) !== (yj > py) && px < (xj - xi) * (py - yi) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

// Convert between API format and internal shape
function apiToShape(a: AnnData): Shape {
  if (a.shape_type === 'polygon') return { type: 'polygon', class_id: a.class_id, pts: a.points }
  if (a.shape_type === 'point')   return { type: 'point',   class_id: a.class_id, x: a.points[0]?.[0] ?? 0, y: a.points[0]?.[1] ?? 0 }
  return { type: 'bbox', class_id: a.class_id, x: a.x_center - a.width/2, y: a.y_center - a.height/2, w: a.width, h: a.height }
}

function shapeToApi(s: Shape): Omit<AnnData, 'id'> {
  if (s.type === 'polygon') return { class_id: s.class_id, shape_type: 'polygon', x_center: 0, y_center: 0, width: 0, height: 0, points: s.pts }
  if (s.type === 'point')   return { class_id: s.class_id, shape_type: 'point',   x_center: 0, y_center: 0, width: 0, height: 0, points: [[s.x, s.y]] }
  return { class_id: s.class_id, shape_type: 'bbox', x_center: s.x + s.w/2, y_center: s.y + s.h/2, width: s.w, height: s.h, points: [] }
}

// ─── Component ────────────────────────────���───────────────────────────────────
export default function Annotate() {
  const { id, imageId } = useParams<{ id: string; imageId: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project, setProject]   = useState<Project | null>(null)
  const [images, setImages]     = useState<ImageItem[]>([])
  const [currentIdx, setCurrentIdx] = useState(0)
  const [shapes, setShapes]     = useState<Shape[]>([])
  const [selected, setSelected] = useState<number | null>(null)
  const [tool, setTool]         = useState<Tool>('bbox')
  const [activeClass, setActiveClass] = useState(0)
  const [zoom, setZoom]         = useState(1)
  const [pan, setPan]           = useState({ x: 0, y: 0 })
  const [saved, setSaved]       = useState(false)

  // In-progress polygon points + live mouse pos for preview
  const [polyPts, setPolyPts]   = useState<[number,number][]>([])
  const [mouse, setMouse]       = useState<[number,number] | null>(null)

  // Live bbox while drawing — must be STATE (not ref) so draw re-runs when it changes
  const [liveBbox, setLiveBbox] = useState<BBoxShape | null>(null)
  const drag          = useRef<Drag>({ kind: 'none' })
  const navigatingRef = useRef(false)
  const canvasRef    = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imgRef       = useRef<HTMLImageElement | null>(null)
  const spaceHeld    = useRef(false)

  // Keep latest state accessible in event handlers via refs
  const shapesRef   = useRef(shapes)
  const selectedRef = useRef(selected)
  const toolRef     = useRef(tool)
  const polyPtsRef  = useRef(polyPts)
  const zoomRef     = useRef(zoom)
  const panRef      = useRef(pan)
  useEffect(() => { shapesRef.current   = shapes   }, [shapes])
  useEffect(() => { selectedRef.current = selected }, [selected])
  useEffect(() => { toolRef.current     = tool     }, [tool])
  useEffect(() => { polyPtsRef.current  = polyPts  }, [polyPts])
  useEffect(() => { zoomRef.current     = zoom     }, [zoom])
  useEffect(() => { panRef.current      = pan      }, [pan])

  // ── Undo / Redo ───────────────────────────────────────────────────────────
  const historyRef = useRef<Shape[][]>([])
  const histIdxRef = useRef(-1)

  const snapshotHistory = useCallback((snap: Shape[]) => {
    historyRef.current = historyRef.current.slice(0, histIdxRef.current + 1)
    historyRef.current.push([...snap])
    histIdxRef.current = historyRef.current.length - 1
  }, [])

  const undo = useCallback(() => {
    if (histIdxRef.current > 0) {
      histIdxRef.current--
      setShapes([...historyRef.current[histIdxRef.current]])
      setSelected(null)
    }
  }, [])

  const redo = useCallback(() => {
    if (histIdxRef.current < historyRef.current.length - 1) {
      histIdxRef.current++
      setShapes([...historyRef.current[histIdxRef.current]])
      setSelected(null)
    }
  }, [])

  // ── Auto-annotate ─────────────────────────────────────────────────────────
  const [trainingRuns, setTrainingRuns] = useState<{id:number;model_base:string;status:string}[]>([])
  const [autoRunId,    setAutoRunId]    = useState('')
  const [autoConf,     setAutoConf]     = useState(0.25)
  const [autoLoading,  setAutoLoading]  = useState(false)
  const [autoMsg,      setAutoMsg]      = useState<{text:string;ok:boolean}|null>(null)

  // ─── Load project + image list ────────��─────────────────────────────────
  useEffect(() => {
    Promise.all([
      api.get(`/projects/${projectId}`),
      api.get(`/projects/${projectId}/images`),
      api.get(`/projects/${projectId}/training/runs`),
    ]).then(([pRes, iRes, rRes]) => {
      setProject(pRes.data)
      const imgs: ImageItem[] = iRes.data
      setImages(imgs)
      const idx = imgs.findIndex(i => i.id === Number(imageId))
      setCurrentIdx(idx >= 0 ? idx : 0)
      const doneRuns = (rRes.data as {id:number;model_base:string;status:string}[])
        .filter(r => r.status === 'done')
      setTrainingRuns(doneRuns)
      if (doneRuns.length > 0) setAutoRunId(String(doneRuns[doneRuns.length - 1].id))
    })
  }, [projectId, imageId])

  const currentImage = images[currentIdx]

  // ─── Load annotations on image change ──────────────────────────────────
  useEffect(() => {
    if (!currentImage) return
    setSaved(false); setSelected(null); setPolyPts([]); setLiveBbox(null)
    api.get(`/projects/${projectId}/images/${currentImage.id}/annotations`)
      .then(r => {
        const loaded = (r.data as AnnData[]).map(apiToShape)
        setShapes(loaded)
        historyRef.current = [[...loaded]]
        histIdxRef.current = 0
      })
  }, [currentImage?.id])

  // ─── Load image ─────────────────────────────────────────────────────────
  const fitToContainer = useCallback(() => {
    const canvas = canvasRef.current
    const cont   = containerRef.current
    if (!canvas || !cont || !imgRef.current) return
    const img    = imgRef.current
    const scale  = Math.min((cont.clientWidth - 4) / img.width, (cont.clientHeight - 4) / img.height, 1)
    canvas.width  = img.width  * scale
    canvas.height = img.height * scale
    setZoom(1); setPan({ x: 0, y: 0 })
  }, [])

  useEffect(() => {
    if (!currentImage) return
    const img = new Image()
    img.onload = () => { imgRef.current = img; fitToContainer() }
    img.src = `/api/projects/${projectId}/images/${currentImage.id}/file`
  }, [currentImage?.id])

  // ─── Draw canvas ──────────────────────────────────��──────────────────��──
  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const img    = imgRef.current
    if (!canvas || !img) return
    const ctx = canvas.getContext('2d')!
    const cw  = canvas.width, ch = canvas.height

    ctx.clearRect(0, 0, cw, ch)
    ctx.save()
    ctx.translate(pan.x, pan.y)
    ctx.scale(zoom, zoom)
    ctx.drawImage(img, 0, 0, cw, ch)

    // ── Helper: canvas coords from normalized ──
    const cx = (nx: number) => nx * cw
    const cy = (ny: number) => ny * ch
    const hSz = H * 2 / zoom  // handle size in zoomed space

    // ── Draw all shapes ──
    const allShapes: Shape[] = liveBbox ? [...shapes, liveBbox] : shapes
    allShapes.forEach((s, i) => {
      const isSel = selected === i && tool === 'select'
      const color = COLORS[s.class_id % COLORS.length]
      ctx.strokeStyle = color
      ctx.lineWidth   = (isSel ? 2.5 : 1.8) / zoom

      if (s.type === 'bbox') {
        const px = cx(s.x), py = cy(s.y), pw = cx(s.w), ph = cy(s.h)
        ctx.fillStyle = color + '25'; ctx.fillRect(px, py, pw, ph)
        ctx.strokeRect(px, py, pw, ph)
        drawLabel(ctx, project?.classes?.[s.class_id] ?? `cls${s.class_id}`, color, cx(s.x), cy(s.y), zoom)
        if (isSel) drawBBoxHandles(ctx, s, cw, ch, zoom, color, hSz)
      }

      if (s.type === 'polygon') {
        if (s.pts.length < 2) return
        ctx.beginPath()
        ctx.moveTo(cx(s.pts[0][0]), cy(s.pts[0][1]))
        s.pts.slice(1).forEach(p => ctx.lineTo(cx(p[0]), cy(p[1])))
        ctx.closePath()
        ctx.fillStyle = color + '25'; ctx.fill()
        ctx.stroke()
        drawLabel(ctx, project?.classes?.[s.class_id] ?? `cls${s.class_id}`, color, cx(s.pts[0][0]), cy(s.pts[0][1]), zoom)
        if (isSel) s.pts.forEach(p => drawDot(ctx, cx(p[0]), cy(p[1]), H / zoom, '#fff', color))
      }

      if (s.type === 'point') {
        const r = DOT_PX / zoom
        ctx.beginPath(); ctx.arc(cx(s.x), cy(s.y), r, 0, Math.PI * 2)
        ctx.fillStyle = color + '80'; ctx.fill(); ctx.stroke()
        drawLabel(ctx, project?.classes?.[s.class_id] ?? `cls${s.class_id}`, color, cx(s.x) + r, cy(s.y) - r, zoom)
        if (isSel) drawDot(ctx, cx(s.x), cy(s.y), (H + 2) / zoom, 'transparent', '#fff')
      }
    })

    // ── Draw in-progress polygon ──
    if (polyPts.length > 0) {
      const color = COLORS[activeClass % COLORS.length]
      ctx.strokeStyle = color; ctx.lineWidth = 1.8 / zoom
      ctx.setLineDash([6 / zoom, 3 / zoom])
      ctx.beginPath()
      ctx.moveTo(cx(polyPts[0][0]), cy(polyPts[0][1]))
      polyPts.slice(1).forEach(p => ctx.lineTo(cx(p[0]), cy(p[1])))
      // Preview line to mouse
      if (mouse) ctx.lineTo(cx(mouse[0]), cy(mouse[1]))
      ctx.stroke()
      ctx.setLineDash([])

      // Placed vertex dots
      polyPts.forEach((p, vi) => {
        const isFirst = vi === 0
        // Snap ring on first point
        if (isFirst && mouse && willSnapClose(polyPts, mouse, cw, ch, zoom, pan)) {
          ctx.beginPath(); ctx.arc(cx(p[0]), cy(p[1]), (H + 4) / zoom, 0, Math.PI * 2)
          ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 2 / zoom; ctx.stroke()
        }
        drawDot(ctx, cx(p[0]), cy(p[1]), H / zoom, isFirst ? color : '#fff', color)
      })
    }

    ctx.restore()
  }, [shapes, liveBbox, selected, tool, polyPts, mouse, activeClass, zoom, pan, project])

  useEffect(() => { draw() }, [draw])

  // ─── Coordinate conversion ────────────────────────────────��──────────────
  const canvasPx = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!
    const rect   = canvas.getBoundingClientRect()
    return {
      cx: (e.clientX - rect.left) * (canvas.width  / rect.width),
      cy: (e.clientY - rect.top)  * (canvas.height / rect.height),
    }
  }

  const toNorm = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] => {
    const { cx, cy } = canvasPx(e)
    const canvas = canvasRef.current!
    return [
      (cx - pan.x) / (zoom * canvas.width),
      (cy - pan.y) / (zoom * canvas.height),
    ]
  }

  // ─── Hit testing ──────────────────────────────────────────────────���──────
  const hitShape = (nx: number, ny: number): number | null => {
    const ss = shapesRef.current
    for (let i = ss.length - 1; i >= 0; i--) {
      const s = ss[i]
      if (s.type === 'bbox')    { if (nx >= s.x && nx <= s.x+s.w && ny >= s.y && ny <= s.y+s.h) return i }
      if (s.type === 'polygon') { if (pointInPolygon(nx, ny, s.pts)) return i }
      if (s.type === 'point')   {
        const canvas = canvasRef.current!
        const r = DOT_PX / (zoomRef.current * canvas.width)
        if (Math.hypot(nx - s.x, ny - s.y) < r * 2) return i
      }
    }
    return null
  }

  const hitVertex = (s: PolygonShape, nx: number, ny: number): number => {
    const canvas = canvasRef.current!
    const thresh = H / (zoomRef.current * canvas.width) * 2
    return s.pts.findIndex(p => Math.hypot(nx - p[0], ny - p[1]) < thresh)
  }

  const hitBBoxHandle = (s: BBoxShape, nx: number, ny: number): string | null => {
    const canvas = canvasRef.current!
    const cw = canvas.width, ch = canvas.height
    const hx = H / (zoomRef.current * cw), hy = H / (zoomRef.current * ch)
    const handles: Record<string, [number, number]> = {
      tl:[s.x,s.y], tc:[s.x+s.w/2,s.y], tr:[s.x+s.w,s.y],
      ml:[s.x,s.y+s.h/2],               mr:[s.x+s.w,s.y+s.h/2],
      bl:[s.x,s.y+s.h], bc:[s.x+s.w/2,s.y+s.h], br:[s.x+s.w,s.y+s.h],
    }
    for (const [hid, [hpx, hpy]] of Object.entries(handles)) {
      if (Math.abs(nx - hpx) < hx * 1.6 && Math.abs(ny - hpy) < hy * 1.6) return hid
    }
    return null
  }

  // ─── Mouse events ─────────────────────────────────────────────────────���───
  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { cx, cy } = canvasPx(e)
    const [nx, ny]   = toNorm(e)

    // Pan: middle-click or Space+LMB
    if (e.button === 1 || spaceHeld.current) {
      drag.current = { kind: 'pan', cx0: cx, cy0: cy, px0: panRef.current.x, py0: panRef.current.y }
      return
    }
    if (e.button !== 0) return

    const currentTool = toolRef.current

    // ── Polygon tool ──
    if (currentTool === 'polygon') {
      const pts = polyPtsRef.current
      if (pts.length >= 3 && willSnapClose(pts, [nx, ny], canvasRef.current!.width, canvasRef.current!.height, zoomRef.current, panRef.current)) {
        const newShapes = [...shapesRef.current, { type: 'polygon' as const, class_id: activeClass, pts: [...pts] }]
        setShapes(newShapes)
        snapshotHistory(newShapes)
        setPolyPts([])
      } else {
        setPolyPts(prev => [...prev, [nx, ny]])
      }
      return
    }

    // ── Point tool ──
    if (currentTool === 'point') {
      const newShapes = [...shapesRef.current, { type: 'point' as const, class_id: activeClass, x: nx, y: ny }]
      setShapes(newShapes)
      snapshotHistory(newShapes)
      return
    }

    // ── BBox draw tool ──
    if (currentTool === 'bbox') {
      setSelected(null)
      drag.current = { kind: 'bbox-draw', start: [nx, ny] }
      setLiveBbox(null)
      return
    }

    // ── Select tool ──
    if (currentTool === 'select') {
      const sel = selectedRef.current
      // Check bbox handles first
      if (sel !== null && shapesRef.current[sel]?.type === 'bbox') {
        const hid = hitBBoxHandle(shapesRef.current[sel] as BBoxShape, nx, ny)
        if (hid) {
          drag.current = { kind: 'bbox-handle', idx: sel, handle: hid, mx0: nx, my0: ny, orig: { ...shapesRef.current[sel] as BBoxShape } }
          return
        }
      }
      // Check polygon vertices
      if (sel !== null && shapesRef.current[sel]?.type === 'polygon') {
        const vi = hitVertex(shapesRef.current[sel] as PolygonShape, nx, ny)
        if (vi >= 0) {
          drag.current = { kind: 'move-vertex', idx: sel, vi, mx0: nx, my0: ny, orig: { ...shapesRef.current[sel] as PolygonShape, pts: [...(shapesRef.current[sel] as PolygonShape).pts] } }
          return
        }
      }
      // Hit shape
      const i = hitShape(nx, ny)
      setSelected(i)
      if (i !== null) {
        drag.current = { kind: 'move-shape', idx: i, mx0: nx, my0: ny, orig: { ...shapesRef.current[i] } as Shape }
      } else {
        drag.current = { kind: 'none' }
      }
    }
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const { cx, cy } = canvasPx(e)
    const [nx, ny]   = toNorm(e)
    const d = drag.current

    // Update cursor
    if (canvasRef.current) canvasRef.current.style.cursor = computeCursor(e, nx, ny)

    // Update mouse for polygon preview
    setMouse([nx, ny])

    if (d.kind === 'pan') {
      setPan({ x: d.px0 + (cx - d.cx0), y: d.py0 + (cy - d.cy0) })
      return
    }
    if (d.kind === 'bbox-draw') {
      const x = Math.min(d.start[0], nx), y = Math.min(d.start[1], ny)
      setLiveBbox({ type: 'bbox', class_id: activeClass, x, y, w: Math.abs(nx - d.start[0]), h: Math.abs(ny - d.start[1]) })
      return
    }
    if (d.kind === 'move-shape') {
      const dx = nx - d.mx0, dy = ny - d.my0
      setShapes(prev => prev.map((s, i) => {
        if (i !== d.idx) return s
        if (s.type === 'bbox')    return { ...s, x: clamp(d.orig.type === 'bbox' ? d.orig.x + dx : 0, 0, 1 - s.w), y: clamp(d.orig.type === 'bbox' ? d.orig.y + dy : 0, 0, 1 - s.h) }
        if (s.type === 'point')   return { ...s, x: clamp((d.orig as PointShape).x + dx, 0, 1), y: clamp((d.orig as PointShape).y + dy, 0, 1) }
        if (s.type === 'polygon') return { ...s, pts: (d.orig as PolygonShape).pts.map(p => [clamp(p[0]+dx,0,1), clamp(p[1]+dy,0,1)] as [number,number]) }
        return s
      }))
      return
    }
    if (d.kind === 'move-vertex') {
      setShapes(prev => prev.map((s, i) => {
        if (i !== d.idx || s.type !== 'polygon') return s
        const pts: [number,number][] = d.orig.pts.map((p, vi) =>
          vi === d.vi ? [clamp(p[0]+(nx-d.mx0),0,1), clamp(p[1]+(ny-d.my0),0,1)] : [...p] as [number,number]
        )
        return { ...s, pts }
      }))
      return
    }
    if (d.kind === 'bbox-handle') {
      setShapes(prev => prev.map((s, i) => {
        if (i !== d.idx || s.type !== 'bbox') return s
        return applyBboxHandle(d.orig, d.handle, nx - d.mx0, ny - d.my0)
      }))
    }
  }

  const onMouseUp = () => {
    const d = drag.current
    drag.current = { kind: 'none' }
    if (d.kind === 'bbox-draw' && liveBbox && liveBbox.w > 0.01 && liveBbox.h > 0.01) {
      const newShapes = [...shapesRef.current, liveBbox]
      setShapes(newShapes)
      snapshotHistory(newShapes)
    } else if (d.kind === 'move-shape' || d.kind === 'move-vertex' || d.kind === 'bbox-handle') {
      snapshotHistory(shapesRef.current)
    }
    setLiveBbox(null)
  }

  // ─── Double-click to close polygon ──────────────────────────────────────
  const onDblClick = () => {
    if (toolRef.current === 'polygon' && polyPtsRef.current.length >= 3) {
      const newShapes = [...shapesRef.current, { type: 'polygon' as const, class_id: activeClass, pts: [...polyPtsRef.current] }]
      setShapes(newShapes)
      snapshotHistory(newShapes)
      setPolyPts([])
    }
  }

  // ──��� Scroll to zoom ────────���─────────────────────────────────────────────
  const onWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const { cx, cy } = canvasPx(e)
    const factor  = e.deltaY < 0 ? 1.12 : 1 / 1.12
    const newZoom = clamp(zoomRef.current * factor, 0.15, 15)
    setPan(p => ({ x: cx - (cx - p.x) * newZoom / zoomRef.current, y: cy - (cy - p.y) * newZoom / zoomRef.current }))
    setZoom(newZoom)
  }

  // ─── Keyboard shortcuts ─────────────────────��────────────────────────────
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === 'Space') { spaceHeld.current = true; e.preventDefault() }
      if (e.key === 'Escape') { setPolyPts([]); setLiveBbox(null); setSelected(null) }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedRef.current !== null && document.activeElement?.tagName !== 'INPUT') {
        const newShapes = shapesRef.current.filter((_, i) => i !== selectedRef.current)
        setShapes(newShapes)
        snapshotHistory(newShapes)
        setSelected(null)
      }
      if (!e.ctrlKey && !e.metaKey) {
        if (e.key === 'v' || e.key === '1') setTool('select')
        if (e.key === 'r' || e.key === '2') { setTool('bbox');    setPolyPts([]) }
        if (e.key === 'p' || e.key === '3') { setTool('polygon'); setPolyPts([]) }
        if (e.key === 'd' || e.key === '4') { setTool('point');   setPolyPts([]) }
      }
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undo() }
      if ((e.ctrlKey || e.metaKey) && e.key === 'y') { e.preventDefault(); redo() }
      if ((e.ctrlKey || e.metaKey) && e.key === 'd' && selectedRef.current !== null) {
        e.preventDefault()
        const s = shapesRef.current[selectedRef.current]
        if (!s) return
        let dup: Shape
        if (s.type === 'bbox')    dup = { ...s, x: Math.min(s.x+0.02,1-s.w), y: Math.min(s.y+0.02,1-s.h) }
        else if (s.type === 'point') dup = { ...s, x: Math.min(s.x+0.02,1), y: Math.min(s.y+0.02,1) }
        else dup = { ...s, pts: s.pts.map(p => [Math.min(p[0]+0.02,1), Math.min(p[1]+0.02,1)] as [number,number]) }
        const newShapes = [...shapesRef.current, dup]
        setShapes(newShapes)
        snapshotHistory(newShapes)
      }
    }
    const up = (e: KeyboardEvent) => { if (e.code === 'Space') spaceHeld.current = false }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup',   up)
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', up) }
  }, [])

  // ─── Cursor ─────────────────────────��────────────────────────────────────
  const computeCursor = (_e: React.MouseEvent<HTMLCanvasElement>, nx: number, ny: number) => {
    if (spaceHeld.current || drag.current.kind === 'pan') return 'grab'
    if (tool === 'bbox' || tool === 'polygon' || tool === 'point') return 'crosshair'
    const sel = selected
    if (sel !== null) {
      const s = shapes[sel]
      if (s?.type === 'bbox') {
        const h = hitBBoxHandle(s, nx, ny)
        if (h) return HANDLE_CURSORS[h] ?? 'pointer'
      }
      if (s?.type === 'polygon') {
        const vi = hitVertex(s, nx, ny)
        if (vi >= 0) return 'crosshair'
      }
    }
    return hitShape(nx, ny) !== null ? 'move' : 'default'
  }

  // ─── Auto-annotate ────────────────────────────────────────────────────────
  const autoAnnotate = async () => {
    if (!currentImage || !autoRunId) return
    setAutoLoading(true)
    setAutoMsg(null)
    try {
      const res = await api.post(
        `/projects/${projectId}/images/${currentImage.id}/auto-annotate`,
        null,
        { params: { run_id: autoRunId, conf: autoConf } }
      )
      const suggested = (res.data.annotations as AnnData[]).map(apiToShape)
      if (suggested.length > 0) {
        const merged = [...shapesRef.current, ...suggested]
        setShapes(merged)
        snapshotHistory(merged)
        setAutoMsg({ text: `+${suggested.length} annotation${suggested.length > 1 ? 's' : ''} added`, ok: true })
      } else {
        setAutoMsg({ text: 'No objects detected — try lowering the confidence threshold', ok: false })
      }
    } catch (err: unknown) {
      const msg = (err as {response?: {data?: {detail?: string}}})?.response?.data?.detail ?? 'Auto-annotate failed'
      setAutoMsg({ text: msg, ok: false })
    } finally {
      setAutoLoading(false)
    }
  }

  const copyFromPrev = async () => {
    if (currentIdx === 0) return
    const prevImg = images[currentIdx - 1]
    if (!prevImg) return
    const res = await api.get(`/projects/${projectId}/images/${prevImg.id}/annotations`)
    const copied = (res.data as AnnData[]).map(apiToShape)
    if (copied.length > 0) {
      const merged = [...shapesRef.current, ...copied]
      setShapes(merged)
      snapshotHistory(merged)
    }
  }

  // ─── Save ─────────────────────────────────────────────────────────────────
  const save = async () => {
    if (!currentImage) return
    await api.post(`/projects/${projectId}/images/${currentImage.id}/annotations`, shapes.map(shapeToApi))
    setSaved(true)
    setImages(prev => prev.map(i => i.id === currentImage.id ? { ...i, annotated: shapes.length > 0 } : i))
  }

  const goTo = async (nextIdx: number) => {
    if (navigatingRef.current) return
    navigatingRef.current = true
    try { await save(); setCurrentIdx(nextIdx) }
    finally { navigatingRef.current = false }
  }

  // ─── Shape label icons ─────────────���─────────────────────────────────��───
  const shapeIcon = (s: Shape) => {
    if (s.type === 'bbox')    return '▭'
    if (s.type === 'polygon') return '⬡'
    return '●'
  }

  // ─── Render ─────────────────────────────────────────────────────────────
  const sidePanel: React.CSSProperties = {
    background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, padding: 12,
  }
  const sideLabel: React.CSSProperties = {
    fontSize: 10, fontWeight: 600, letterSpacing: '0.07em', textTransform: 'uppercase',
    color: 'var(--text3)', marginBottom: 8,
  }
  const iconBtn = (active = false): React.CSSProperties => ({
    display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 3, padding: '8px 6px',
    borderRadius: 6, border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
    background: active ? 'rgba(88,101,242,0.15)' : 'var(--surface2)',
    color: active ? 'var(--accent)' : 'var(--text2)', cursor: 'pointer',
    fontSize: 10, fontWeight: 500, transition: 'all 0.12s',
  })
  const zoomBtn: React.CSSProperties = {
    padding: '5px 7px', border: '1px solid var(--border)', borderRadius: 5,
    background: 'var(--surface2)', color: 'var(--text2)', cursor: 'pointer', display: 'flex',
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: 'calc(100vh - 80px)',
      maxWidth: 1400, margin: '0 auto', gap: 10 }}>

      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
        <button onClick={() => navigate(`/projects/${projectId}/images`)}
          style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', width: 28, height: 28,
            border: '1px solid var(--border)', borderRadius: 6, background: 'transparent',
            color: 'var(--text2)', cursor: 'pointer', flexShrink: 0 }}>
          <ChevronLeft size={15} />
        </button>
        <h1 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', flex: 1,
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {currentImage?.original_name}
        </h1>
        <span style={{ fontSize: 12, color: 'var(--text3)', whiteSpace: 'nowrap' }}>
          {currentIdx + 1} / {images.length}
        </span>
        {/* Undo / Redo */}
        <button onClick={undo} title="Undo (Ctrl+Z)"
          style={{ ...zoomBtn, fontFamily: 'inherit', fontSize: 13 }}>↩</button>
        <button onClick={redo} title="Redo (Ctrl+Y)"
          style={{ ...zoomBtn, fontFamily: 'inherit', fontSize: 13 }}>↪</button>
        {/* Copy from prev */}
        {currentIdx > 0 && (
          <button onClick={copyFromPrev} title="Copy annotations from previous image"
            style={{ ...zoomBtn, fontSize: 11, padding: '5px 8px',
              color: 'var(--text2)', whiteSpace: 'nowrap' }}>
            ← Copy
          </button>
        )}
        <button onClick={() => goTo(currentIdx - 1)} disabled={currentIdx === 0}
          style={{ ...zoomBtn, opacity: currentIdx === 0 ? 0.3 : 1 }}>
          <ChevronLeft size={14} />
        </button>
        <button onClick={() => goTo(currentIdx + 1)} disabled={currentIdx === images.length - 1}
          style={{ ...zoomBtn, opacity: currentIdx === images.length - 1 ? 0.3 : 1 }}>
          <ChevronRight size={14} />
        </button>
        <button onClick={save} style={{
          display: 'flex', alignItems: 'center', gap: 6, padding: '6px 14px',
          borderRadius: 6, border: `1px solid ${saved ? 'var(--success)' : 'var(--accent)'}`,
          background: saved ? 'rgba(34,197,94,0.12)' : 'var(--accent)',
          color: saved ? 'var(--success)' : '#fff', cursor: 'pointer',
          fontSize: 13, fontWeight: 500, transition: 'all 0.15s',
        }}>
          <Save size={13} /> {saved ? 'Saved!' : 'Save'}
        </button>
      </div>

      <div style={{ display: 'flex', gap: 10, flex: 1, minHeight: 0 }}>

        {/* ── Sidebar ── */}
        <div style={{ width: 172, flexShrink: 0, display: 'flex', flexDirection: 'column',
          gap: 8, overflowY: 'auto' }}>

          {/* Tool selector */}
          <div style={sidePanel}>
            <p style={sideLabel}>Tool</p>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 5 }}>
              {TOOLS.map(t => (
                <button key={t.id} onClick={() => { setTool(t.id); setPolyPts([]) }}
                  title={t.hint ? `${t.label} (${t.hint})` : t.label}
                  style={iconBtn(tool === t.id)}>
                  {t.icon}
                  <span>{t.label}</span>
                  {t.hint && <kbd style={{ fontSize: 9, opacity: 0.5, fontFamily: 'JetBrains Mono, monospace' }}>{t.hint}</kbd>}
                </button>
              ))}
            </div>
            {tool === 'polygon' && polyPts.length > 0 && (
              <div style={{ marginTop: 8, padding: '7px 9px',
                background: 'rgba(88,101,242,0.08)', border: '1px solid rgba(88,101,242,0.25)',
                borderRadius: 6, fontSize: 11, color: '#a5b4fc' }}>
                <p>{polyPts.length} pts — click near start or dbl-click to close</p>
                <button onClick={() => setPolyPts([])}
                  style={{ marginTop: 4, border: 'none', background: 'transparent',
                    color: 'var(--danger)', cursor: 'pointer', fontSize: 11, padding: 0 }}>
                  Cancel (Esc)
                </button>
              </div>
            )}
          </div>

          {/* Zoom */}
          <div style={sidePanel}>
            <p style={sideLabel}>Zoom</p>
            <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 6 }}>
              <button onClick={() => setZoom(z => clamp(z / 1.3, 0.15, 15))} style={zoomBtn}>
                <ZoomOut size={13} />
              </button>
              <span style={{ flex: 1, textAlign: 'center', fontSize: 11,
                color: 'var(--text2)', fontFamily: 'JetBrains Mono, monospace' }}>
                {Math.round(zoom * 100)}%
              </span>
              <button onClick={() => setZoom(z => clamp(z * 1.3, 0.15, 15))} style={zoomBtn}>
                <ZoomIn size={13} />
              </button>
            </div>
            <button onClick={fitToContainer} style={{ width: '100%', display: 'flex', alignItems: 'center',
              justifyContent: 'center', gap: 5, padding: '5px 0', background: 'var(--surface2)',
              border: '1px solid var(--border)', borderRadius: 5, color: 'var(--text2)',
              cursor: 'pointer', fontSize: 11 }}>
              <Maximize2 size={11} /> Fit
            </button>
          </div>

          {/* Classes */}
          <div style={sidePanel}>
            <p style={sideLabel}>Active Class</p>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {project?.classes.map((c, i) => (
                <button key={i} onClick={() => setActiveClass(i)}
                  style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 9px',
                    borderRadius: 6, border: `1px solid ${activeClass === i ? 'var(--accent)' : 'var(--border)'}`,
                    background: activeClass === i ? 'rgba(88,101,242,0.12)' : 'var(--surface2)',
                    color: activeClass === i ? 'var(--accent)' : 'var(--text2)',
                    cursor: 'pointer', fontSize: 12, textAlign: 'left', transition: 'all 0.12s' }}>
                  <span style={{ width: 9, height: 9, borderRadius: '50%', flexShrink: 0,
                    background: COLORS[i % COLORS.length] }} />
                  <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{c}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Shapes list */}
          <div style={{ ...sidePanel, flex: 1, minHeight: 0, display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <p style={{ ...sideLabel, marginBottom: 0 }}>Shapes ({shapes.length})</p>
              {shapes.length > 0 && (
                <button onClick={() => { const n: Shape[] = []; setShapes(n); snapshotHistory(n); setSelected(null) }}
                  style={{ border: 'none', background: 'transparent', color: 'var(--text3)',
                    cursor: 'pointer', display: 'flex', padding: 2, borderRadius: 3 }}>
                  <RotateCcw size={11} />
                </button>
              )}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3,
              overflowY: 'auto', maxHeight: 200 }}>
              {shapes.map((s, i) => (
                <div key={i} onClick={() => { setSelected(i); setTool('select') }}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                    padding: '5px 8px', borderRadius: 5, cursor: 'pointer',
                    border: `1px solid ${selected === i ? 'var(--accent)' : 'var(--border)'}`,
                    background: selected === i ? 'rgba(88,101,242,0.10)' : 'var(--surface2)',
                    transition: 'all 0.1s' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0 }}>
                    <span style={{ width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                      background: COLORS[s.class_id % COLORS.length] }} />
                    <span style={{ fontSize: 11, color: 'var(--text3)', marginRight: 1 }}>{shapeIcon(s)}</span>
                    <span style={{ fontSize: 11, color: 'var(--text2)',
                      overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {project?.classes[s.class_id]}
                    </span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 3, flexShrink: 0 }}>
                    <button onClick={e => { e.stopPropagation()
                      const sh = shapesRef.current[i]; if (!sh) return
                      let dup: Shape
                      if (sh.type === 'bbox')    dup = { ...sh, x: Math.min(sh.x+0.02,1-sh.w) }
                      else if (sh.type === 'point') dup = { ...sh, x: Math.min(sh.x+0.02,1) }
                      else dup = { ...sh, pts: sh.pts.map(p => [Math.min(p[0]+0.02,1), p[1]] as [number,number]) }
                      const ns = [...shapesRef.current, dup]
                      setShapes(ns); snapshotHistory(ns)
                    }} style={{ border: 'none', background: 'transparent',
                      color: 'var(--text3)', cursor: 'pointer', padding: 1, display: 'flex' }}>
                      <Copy size={10} />
                    </button>
                    <button onClick={e => { e.stopPropagation()
                      const ns = shapesRef.current.filter((_,j)=>j!==i)
                      setShapes(ns); snapshotHistory(ns)
                      if(selectedRef.current===i) setSelected(null)
                    }} style={{ border: 'none', background: 'transparent',
                        color: 'var(--text3)', cursor: 'pointer', padding: 1, display: 'flex' }}>
                      <Trash2 size={10} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
          {/* Auto-annotate */}
          {trainingRuns.length > 0 && (
            <div style={sidePanel}>
              <p style={sideLabel}>Auto-Annotate</p>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
                <select value={autoRunId} onChange={e => setAutoRunId(e.target.value)}
                  style={{ width: '100%', padding: '5px 8px', background: 'var(--surface2)',
                    border: '1px solid var(--border)', borderRadius: 5, color: 'var(--text)',
                    fontSize: 11, fontFamily: 'inherit', outline: 'none', cursor: 'pointer' }}>
                  {trainingRuns.map(r => (
                    <option key={r.id} value={r.id}>#{r.id} {r.model_base}</option>
                  ))}
                </select>
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
                    <span style={{ fontSize: 10, color: 'var(--text2)' }}>Conf</span>
                    <span style={{ fontSize: 10, color: 'var(--text)', fontFamily: 'JetBrains Mono, monospace' }}>
                      {Math.round(autoConf * 100)}%
                    </span>
                  </div>
                  <input type="range" min={0.05} max={0.95} step={0.05} value={autoConf}
                    onChange={e => setAutoConf(Number(e.target.value))}
                    style={{ width: '100%', accentColor: 'var(--accent)', cursor: 'pointer' }} />
                </div>
                <button onClick={autoAnnotate} disabled={autoLoading || !autoRunId}
                  style={{ padding: '6px 0', background: 'var(--accent)',
                    border: '1px solid var(--accent)', borderRadius: 5,
                    color: '#fff', fontSize: 11, fontFamily: 'inherit',
                    cursor: autoLoading ? 'wait' : 'pointer',
                    opacity: autoLoading ? 0.6 : 1 }}>
                  {autoLoading ? 'Running…' : '⚡ Auto-Annotate'}
                </button>
                {autoMsg && (
                  <p style={{ fontSize: 10, margin: 0, lineHeight: 1.4,
                    color: autoMsg.ok ? '#22c55e' : '#f97316' }}>
                    {autoMsg.text}
                  </p>
                )}
              </div>
            </div>
          )}

        </div>

        {/* ── Canvas area ── */}
        <div ref={containerRef} style={{ flex: 1, background: 'var(--surface)',
          border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden',
          display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <div style={{ flex: 1, overflow: 'hidden', display: 'flex', alignItems: 'center',
            justifyContent: 'center', background: '#07070a', minHeight: 0 }}>
            <canvas ref={canvasRef} style={{ display: 'block' }}
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={() => { setMouse(null); onMouseUp() }}
              onWheel={onWheel}
              onDoubleClick={onDblClick}
              onContextMenu={e => e.preventDefault()}
            />
          </div>
          {/* Hint bar */}
          <div style={{ flexShrink: 0, padding: '7px 14px', borderTop: '1px solid var(--border)',
            display: 'flex', flexWrap: 'wrap', gap: '4px 16px',
            fontSize: 10, color: 'var(--text3)', fontFamily: 'JetBrains Mono, monospace' }}>
            {[['1','Select'],['2','Rect'],['3','Polygon'],['4','Point'],['Del','Delete'],['Ctrl+D','Dupe'],['Esc','Cancel']].map(([k,v]) => (
              <span key={k}><kbd style={{ background: 'var(--surface3)', border: '1px solid var(--border2)',
                borderRadius: 3, padding: '1px 4px', fontSize: 9 }}>{k}</kbd> {v}</span>
            ))}
            <span style={{ color: 'var(--text3)' }}>Scroll=Zoom · Space+drag=Pan · Dbl-click=Close poly</span>
          </div>
        </div>

      </div>
    </div>
  )
}

// ─── Tool definitions ────────────────────────────��────────────────────────────
const TOOLS: { id: Tool; label: string; hint?: string; icon: React.ReactNode }[] = [
  { id: 'select',  label: 'Select',   hint: '1', icon: <MousePointer2 size={16}/> },
  { id: 'bbox',    label: 'Rect',     hint: '2', icon: <Square size={16}/> },
  { id: 'polygon', label: 'Polygon',  hint: '3', icon: <Hexagon size={16}/> },
  { id: 'point',   label: 'Point',    hint: '4', icon: <Crosshair size={16}/> },
]

const HANDLE_CURSORS: Record<string, string> = {
  tl:'nwse-resize', tr:'nesw-resize', bl:'nesw-resize', br:'nwse-resize',
  tc:'ns-resize',   bc:'ns-resize',   ml:'ew-resize',   mr:'ew-resize',
}

// ─── Canvas drawing helpers ─────────────────────────────���─────────────────────
function drawLabel(ctx: CanvasRenderingContext2D, text: string, color: string, x: number, y: number, zoom: number) {
  ctx.font = `${12 / zoom}px system-ui`
  const tw = ctx.measureText(text).width
  ctx.fillStyle = color
  ctx.fillRect(x, y - 18 / zoom, tw + 8 / zoom, 18 / zoom)
  ctx.fillStyle = '#fff'
  ctx.fillText(text, x + 4 / zoom, y - 5 / zoom)
}

function drawDot(ctx: CanvasRenderingContext2D, x: number, y: number, r: number, fill: string, stroke: string) {
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2)
  ctx.fillStyle   = fill;   ctx.fill()
  ctx.strokeStyle = stroke; ctx.stroke()
}

function drawBBoxHandles(ctx: CanvasRenderingContext2D, s: BBoxShape, cw: number, ch: number, zoom: number, color: string, hSz: number) {
  const handles: [number,number][] = [
    [s.x,s.y], [s.x+s.w/2,s.y], [s.x+s.w,s.y],
    [s.x,s.y+s.h/2],             [s.x+s.w,s.y+s.h/2],
    [s.x,s.y+s.h], [s.x+s.w/2,s.y+s.h], [s.x+s.w,s.y+s.h],
  ]
  handles.forEach(([nx, ny]) => {
    ctx.fillStyle   = '#fff'
    ctx.fillRect(nx*cw - hSz/2, ny*ch - hSz/2, hSz, hSz)
    ctx.strokeStyle = color; ctx.lineWidth = 1.5 / zoom
    ctx.strokeRect(nx*cw - hSz/2, ny*ch - hSz/2, hSz, hSz)
  })
}

function applyBboxHandle(orig: BBoxShape, h: string, dx: number, dy: number): BBoxShape {
  let { x, y, w, h: bh } = orig
  if (h.includes('l')) { x += dx; w -= dx }
  if (h.includes('r')) { w += dx }
  if (h.includes('t')) { y += dy; bh -= dy }
  if (h.includes('b')) { bh += dy }
  return { ...orig, x: Math.max(0, Math.min(x, x+w)), y: Math.max(0, Math.min(y, y+bh)), w: Math.abs(w), h: Math.abs(bh) }
}

function willSnapClose(
  pts: [number,number][],
  mouse: [number,number],
  cw: number, ch: number,
  zoom: number,
  pan: { x: number; y: number }
): boolean {
  if (pts.length < 3) return false
  const [fpx, fpy] = pts[0]
  const screenFx = fpx * cw * zoom + pan.x
  const screenFy = fpy * ch * zoom + pan.y
  const screenMx = mouse[0] * cw * zoom + pan.x
  const screenMy = mouse[1] * ch * zoom + pan.y
  return Math.hypot(screenMx - screenFx, screenMy - screenFy) < SNAP_PX
}
