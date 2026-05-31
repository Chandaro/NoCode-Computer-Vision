import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Upload, Pencil, Trash2, BarChart2, Zap, Brain, CheckSquare, Square, X, Cpu, FolderInput, Tags, Camera, Activity, Layers, Info, ChevronDown, ChevronUp, Folder, FileText, Image as ImageIcon, FileArchive, ImagePlus, ScanLine, Crosshair } from 'lucide-react'
import api, { type ImageItem, type Project } from '../api'
import { PageHeader, Btn, Badge, Empty } from '../components/ui'

interface ImportResult {
  imported: number
  annotated: number
  skipped_duplicates: number
  classes_updated: boolean
}

export default function ProjectImages() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const [project, setProject]   = useState<Project | null>(null)
  const [images, setImages]     = useState<ImageItem[]>([])
  const [loading, setLoading]   = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [importing, setImporting] = useState(false)
  const [importResult, setImportResult] = useState<ImportResult | null>(null)
  const [importProgress, setImportProgress] = useState<{ current: number; total: number } | null>(null)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [selectMode, setSelectMode] = useState(false)
  const [editingClasses, setEditingClasses] = useState(false)
  const [classInput, setClassInput] = useState('')
  const [savingClasses, setSavingClasses] = useState(false)
  const [showImportGuide, setShowImportGuide] = useState(false)
  // ── Dataset mode ──
  const [datasetMode, setDatasetMode] = useState<'detection' | 'classification'>('detection')
  // ── Classification dataset state ──
  const [datasetStats,   setDatasetStats]   = useState<Record<string, number>>({})
  const [uploadingCls,   setUploadingCls]   = useState<string | null>(null)
  const [clearingCls,    setClearingCls]    = useState<string | null>(null)
  const [clsImporting,   setClsImporting]   = useState(false)
  const [showClsGuide,   setShowClsGuide]   = useState(false)
  const [newClassName,   setNewClassName]   = useState('')
  const fileRef        = useRef<HTMLInputElement>(null)
  const importRef      = useRef<HTMLInputElement>(null)
  const importFolderRef = useRef<HTMLInputElement>(null)
  const newClassFileRef   = useRef<HTMLInputElement>(null)
  const zipImportRef      = useRef<HTMLInputElement>(null)
  const clsFolderInputRef = useRef<HTMLInputElement | null>(null)
  const navigate = useNavigate()

  const load = async () => {
    const [pRes, iRes] = await Promise.all([
      api.get(`/projects/${projectId}`),
      api.get(`/projects/${projectId}/images`),
    ])
    setProject(pRes.data)
    setImages(iRes.data)
  }

  useEffect(() => { load() }, [projectId])

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return
    setLoading(true)
    try {
      const fd = new FormData()
      Array.from(e.target.files).forEach(f => fd.append('files', f))
      await api.post(`/projects/${projectId}/images`, fd)
      await load()
    } finally { setLoading(false); e.target.value = '' }
  }

  const saveClasses = async (newClasses: string[]) => {
    if (!project) return
    setSavingClasses(true)
    try {
      await api.put(`/projects/${projectId}`, {
        name: project.name,
        description: project.description,
        classes: newClasses,
      })
      await load()
    } finally { setSavingClasses(false) }
  }

  const addClass = () => {
    const t = classInput.trim()
    if (!t || project?.classes.includes(t)) return
    const updated = [...(project?.classes ?? []), t]
    setClassInput('')
    saveClasses(updated)
  }

  const removeClass = (idx: number) => {
    const updated = (project?.classes ?? []).filter((_, i) => i !== idx)
    saveClasses(updated)
  }

  const handleImportYolo = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const allFiles = Array.from(e.target.files || [])
    if (!allFiles.length) return
    e.target.value = ''
    setImporting(true)
    setImportResult(null)
    setImportProgress(null)

    try {
      const imageMap = new Map<string, File>()  // matchKey → file
      const labelMap = new Map<string, File>()  // matchKey → file
      let classesFile: File | null = null

      const matchKey = (f: File): string => {
        // Use webkitRelativePath when available (folder upload)
        const rel = ((f as any).webkitRelativePath as string || f.name).replace(/\\/g, '/')
        const parts = rel.split('/')
        const filename = parts[parts.length - 1]
        const dot = filename.lastIndexOf('.')
        const stem = (dot > 0 ? filename.slice(0, dot) : filename).toLowerCase()

        // Match by subpath AFTER the images/ or labels/ folder, so nested dirs stay aligned
        // e.g. "dataset/images/train/fire001.jpg" → key "train/fire001"
        //      "dataset/labels/train/fire001.txt" → key "train/fire001"
        const imgIdx = parts.findIndex(p => p.toLowerCase() === 'images')
        const lblIdx = parts.findIndex(p => p.toLowerCase() === 'labels')
        const baseIdx = imgIdx >= 0 ? imgIdx : lblIdx >= 0 ? lblIdx : -1
        if (baseIdx >= 0 && baseIdx < parts.length - 1) {
          const sub = parts.slice(baseIdx + 1, -1)  // subdirs between images/ and filename
          return [...sub, stem].join('/').toLowerCase()
        }
        return stem
      }

      for (const f of allFiles) {
        const filename = f.name
        const dot = filename.lastIndexOf('.')
        const ext = (dot > 0 ? filename.slice(dot + 1) : '').toLowerCase()
        const stem = (dot > 0 ? filename.slice(0, dot) : filename).toLowerCase()
        const key = matchKey(f)

        if (['jpg','jpeg','png','bmp','webp'].includes(ext)) {
          imageMap.set(key, f)
        } else if (ext === 'txt') {
          if (stem === 'classes') classesFile = f
          else labelMap.set(key, f)
        }
      }

      // Pair each image with its matching label using the same key
      const pairs = Array.from(imageMap.entries()).map(([key, img]) => ({
        img, label: labelMap.get(key) ?? null
      }))

      const BATCH = 50
      const total = pairs.length
      let imported = 0, annotated = 0, skipped = 0, classesUpdated = false

      setImportProgress({ current: 0, total })

      for (let i = 0; i < pairs.length; i += BATCH) {
        const chunk = pairs.slice(i, i + BATCH)
        const fd = new FormData()
        if (i === 0 && classesFile) fd.append('files', classesFile, 'classes.txt')
        for (const { img, label } of chunk) {
          fd.append('files', img, img.name)
          if (label) fd.append('files', label, label.name)
        }
        const res = await api.post(`/projects/${projectId}/images/import-yolo`, fd)
        imported += res.data.imported
        annotated += res.data.annotated
        skipped   += res.data.skipped_duplicates
        if (res.data.classes_updated) classesUpdated = true
        setImportProgress({ current: Math.min(i + BATCH, total), total })
      }

      setImportResult({ imported, annotated, skipped_duplicates: skipped, classes_updated: classesUpdated })
      await load()
    } finally {
      setImporting(false)
      setImportProgress(null)
    }
  }

  const deleteImage = async (imgId: number) => {
    await api.delete(`/projects/${projectId}/images/${imgId}`)
    setImages(prev => prev.filter(i => i.id !== imgId))
  }

  const toggleSelect = (imgId: number) => {
    setSelected(prev => {
      const next = new Set(prev)
      next.has(imgId) ? next.delete(imgId) : next.add(imgId)
      return next
    })
  }

  const selectAll = () => setSelected(new Set(filteredImages.map(i => i.id)))
  const deselectAll = () => setSelected(new Set())

  const bulkDelete = async () => {
    if (selected.size === 0) return
    const confirmed = window.confirm(`Delete ${selected.size} image${selected.size > 1 ? 's' : ''}? This cannot be undone.`)
    if (!confirmed) return
    setDeleting(true)
    try {
      await api.delete(`/projects/${projectId}/images`, { data: { ids: Array.from(selected) } })
      setImages(prev => prev.filter(i => !selected.has(i.id)))
      setSelected(new Set())
      setSelectMode(false)
    } finally {
      setDeleting(false)
    }
  }

  const exitSelectMode = () => { setSelectMode(false); setSelected(new Set()) }

  // ── Classification dataset helpers ──
  const loadDatasetStats = async () => {
    try {
      const res = await api.get(`/projects/${projectId}/classification/dataset/stats`)
      setDatasetStats(res.data)
    } catch { setDatasetStats({}) }
  }

  useEffect(() => {
    if (datasetMode === 'classification') loadDatasetStats()
  }, [datasetMode, projectId])

  const _refreshAfterImport = async () => {
    await Promise.all([load(), loadDatasetStats()])
  }

  const uploadToClass = async (cls: string, files: FileList) => {
    setUploadingCls(cls)
    try {
      const fd = new FormData()
      Array.from(files).forEach(f => fd.append('files', f))
      await api.post(`/projects/${projectId}/classification/dataset/upload/${encodeURIComponent(cls)}`, fd)
      await _refreshAfterImport()
    } finally { setUploadingCls(null) }
  }

  const uploadNewClass = async (files: FileList) => {
    const cls = newClassName.trim()
    if (!cls || !files.length) return
    setNewClassName('')
    await uploadToClass(cls, files)
  }

  const clearClass = async (cls: string) => {
    if (!window.confirm(`Delete all images for class "${cls}"?`)) return
    setClearingCls(cls)
    try {
      await api.delete(`/projects/${projectId}/classification/dataset/${encodeURIComponent(cls)}`)
      await _refreshAfterImport()
    } finally { setClearingCls(null) }
  }

  const importClsZip = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    e.target.value = ''
    setClsImporting(true)
    try {
      const fd = new FormData()
      fd.append('file', f)
      await api.post(`/projects/${projectId}/classification/dataset/import-zip`, fd)
      await _refreshAfterImport()
    } finally { setClsImporting(false) }
  }

  const importClsFolder = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const allFiles = Array.from(e.target.files || [])
    if (!allFiles.length) return
    e.target.value = ''
    setClsImporting(true)
    try {
      const fd = new FormData()
      for (const f of allFiles) {
        // Send webkitRelativePath as filename so the backend can parse class structure
        const relPath = (f as any).webkitRelativePath || f.name
        fd.append('files', f, relPath)
      }
      await api.post(`/projects/${projectId}/classification/dataset/import-folder`, fd)
      await _refreshAfterImport()
    } finally { setClsImporting(false) }
  }

  const [filter, setFilter] = useState<'all'|'annotated'|'unannotated'|'corrupt'>('all')

  const annotated = images.filter(i => i.annotated).length
  const total     = images.length
  const pct       = total > 0 ? Math.round(annotated / total * 100) : 0

  const filteredImages = images.filter(img => {
    if (filter === 'annotated')   return img.annotated
    if (filter === 'unannotated') return !img.annotated
    if (filter === 'corrupt')     return img.is_corrupt
    return true
  })

  const allFilteredSelected = filteredImages.length > 0 && filteredImages.every(i => selected.has(i.id))

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto' }}>

      {/* Import progress overlay */}
      {importing && importProgress && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.7)', zIndex: 200,
          display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(4px)' }}>
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12,
            padding: 32, width: 380, display: 'flex', flexDirection: 'column', gap: 16 }}>
            <p style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>Importing YOLO Dataset…</p>
            <div style={{ background: 'var(--surface2)', borderRadius: 6, height: 8, overflow: 'hidden' }}>
              <div style={{
                height: '100%', borderRadius: 6, background: 'var(--accent)',
                width: `${Math.round(importProgress.current / importProgress.total * 100)}%`,
                transition: 'width 0.3s ease'
              }} />
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: 'var(--text2)' }}>
              <span>{importProgress.current.toLocaleString()} of {importProgress.total.toLocaleString()} files</span>
              <span>{Math.round(importProgress.current / importProgress.total * 100)}%</span>
            </div>
            <p style={{ fontSize: 11, color: 'var(--text3)' }}>Please wait, uploading in batches…</p>
          </div>
        </div>
      )}
      <PageHeader
        back={() => navigate('/')}
        title={project?.name ?? '…'}
        subtitle={`${total} images · ${annotated} annotated · ${pct}%`}
        actions={<>
          <Btn variant="secondary" size="sm" onClick={() => navigate(`/projects/${projectId}/analytics`)}>
            <BarChart2 size={13} /> Analytics
          </Btn>
          <Btn variant="secondary" size="sm" onClick={() => navigate(`/projects/${projectId}/classify`)}>
            <Brain size={13} /> Classify
          </Btn>
          <Btn variant="secondary" size="sm" onClick={() => navigate(`/projects/${projectId}/custom`)}>
            <Cpu size={13} /> Conv Builder
          </Btn>
          <Btn variant="secondary" size="sm" onClick={() => navigate(`/projects/${projectId}/segment`)}>
            <Layers size={13} /> Segment
          </Btn>
          <Btn variant="secondary" size="sm" onClick={() => navigate(`/projects/${projectId}/pose`)}>
            <Activity size={13} /> Pose
          </Btn>
          <Btn variant="secondary" size="sm" onClick={() => navigate(`/projects/${projectId}/depth`)}>
            <Layers size={13} /> Depth
          </Btn>
          <Btn variant="secondary" size="sm" onClick={() => navigate(`/projects/${projectId}/webcam`)}>
            <Camera size={13} /> Webcam
          </Btn>
          <Btn variant="primary" size="sm" onClick={() => navigate(`/projects/${projectId}/train`)}>
            <Zap size={13} strokeWidth={2.5} /> Detect & Train
          </Btn>
        </>}
      />

      {/* Class labels editor */}
      <div style={{ marginBottom: 20, padding: '12px 16px', borderRadius: 8,
        background: 'var(--surface)', border: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: editingClasses || (project?.classes.length ?? 0) > 0 ? 10 : 0 }}>
          <Tags size={13} style={{ color: 'var(--text3)' }} />
          <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text2)' }}>Class labels</span>
          <span style={{ fontSize: 11, color: 'var(--text3)', marginLeft: 2 }}>
            {project?.classes.length ? `${project.classes.length} defined` : 'none defined'}
          </span>
          <div style={{ flex: 1 }} />
          <button onClick={() => setEditingClasses(v => !v)}
            style={{ fontSize: 11, color: 'var(--accent)', background: 'none', border: 'none',
              cursor: 'pointer', padding: '2px 6px', borderRadius: 4 }}>
            {editingClasses ? 'Done' : 'Edit'}
          </button>
        </div>

        {/* Tag list */}
        {(project?.classes.length ?? 0) > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginBottom: editingClasses ? 10 : 0 }}>
            {project!.classes.map((c, i) => (
              <span key={i} style={{ display: 'inline-flex', alignItems: 'center', gap: 4,
                fontSize: 11, padding: '2px 8px', borderRadius: 4,
                background: 'var(--surface2)', color: 'var(--text2)',
                border: '1px solid var(--border)', fontFamily: 'monospace' }}>
                <span style={{ color: 'var(--text3)' }}>{i}:</span> {c}
                {editingClasses && (
                  <button onClick={() => removeClass(i)}
                    style={{ background: 'none', border: 'none', color: 'var(--text3)',
                      cursor: 'pointer', padding: 0, lineHeight: 1, fontSize: 13,
                      display: 'flex', alignItems: 'center' }}>
                    <X size={10} />
                  </button>
                )}
              </span>
            ))}
          </div>
        )}

        {/* Add input — visible when editing */}
        {editingClasses && (
          <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <input
              value={classInput}
              onChange={e => setClassInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addClass()}
              placeholder="Add class name…"
              style={{ flex: 1, fontSize: 12, padding: '5px 10px', borderRadius: 6,
                background: 'var(--surface2)', border: '1px solid var(--border)',
                color: 'var(--text)', outline: 'none' }}
            />
            <Btn variant="secondary" size="sm" onClick={addClass} disabled={savingClasses || !classInput.trim()}>
              Add
            </Btn>
            <span style={{ fontSize: 11, color: 'var(--text3)' }}>
              or auto-populate by importing a dataset with <code style={{ fontFamily: 'monospace' }}>classes.txt</code>
            </span>
          </div>
        )}
      </div>

      {/* ── Dataset Mode Toggle ── */}
      <div style={{ display: 'flex', gap: 0, marginBottom: 20, borderRadius: 8, overflow: 'hidden',
        border: '1px solid var(--border)', width: 'fit-content' }}>
        {([
          { key: 'detection',      icon: <Crosshair size={13} />,  label: 'Object Detection' },
          { key: 'classification', icon: <ScanLine  size={13} />,  label: 'Image Classification' },
        ] as const).map(({ key, icon, label }) => (
          <button key={key} onClick={() => setDatasetMode(key)}
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '8px 18px', border: 'none', cursor: 'pointer',
              fontSize: 12, fontWeight: datasetMode === key ? 600 : 400,
              background: datasetMode === key ? 'var(--accent)' : 'var(--surface)',
              color: datasetMode === key ? '#fff' : 'var(--text2)',
              transition: 'all 0.15s',
            }}>
            {icon} {label}
          </button>
        ))}
      </div>

      {/* ═══════════════════════════════════════════════════════════
          CLASSIFICATION DATASET PANEL
          ═══════════════════════════════════════════════════════════ */}
      {datasetMode === 'classification' && (
        <div style={{ marginBottom: 24 }}>

          {/* Toolbar */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 14, flexWrap: 'wrap' }}>
            <Btn variant="primary" onClick={() => clsFolderInputRef.current?.click()} disabled={clsImporting}>
              <Folder size={13} /> {clsImporting ? 'Importing…' : 'Import Folder'}
            </Btn>
            <Btn variant="secondary" onClick={() => zipImportRef.current?.click()} disabled={clsImporting}>
              <FileArchive size={13} /> {clsImporting ? 'Importing…' : 'Import ZIP'}
            </Btn>
            <button onClick={() => setShowClsGuide(v => !v)}
              style={{ display: 'flex', alignItems: 'center', gap: 5,
                padding: '5px 10px', border: '1px solid var(--border)',
                borderRadius: 'var(--radius)', background: showClsGuide ? 'var(--accent-s)' : 'transparent',
                color: showClsGuide ? 'var(--accent)' : 'var(--text3)',
                fontSize: 12, cursor: 'pointer', transition: 'all 0.1s' }}>
              <Info size={12} /> Format guide {showClsGuide ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>
            <input ref={zipImportRef} type="file" accept=".zip" style={{ display: 'none' }} onChange={importClsZip} />
            <input ref={newClassFileRef} type="file" multiple accept="image/*" style={{ display: 'none' }}
              onChange={e => { if (e.target.files) uploadNewClass(e.target.files); e.target.value = '' }} />
            {/* webkitdirectory must be set via DOM ref — JSX won't pass it through */}
            <input
              ref={el => {
                clsFolderInputRef.current = el
                if (el) {
                  el.setAttribute('webkitdirectory', '')
                  el.setAttribute('multiple', '')
                }
              }}
              type="file"
              style={{ display: 'none' }}
              onChange={importClsFolder}
            />
          </div>

          {/* Format guide */}
          {showClsGuide && (
            <div style={{ marginBottom: 16, border: '1px solid var(--border)', borderRadius: 8,
              background: 'var(--surface)', overflow: 'hidden' }}>
              <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--border)',
                background: 'var(--surface2)', display: 'flex', alignItems: 'center', gap: 8 }}>
                <Info size={13} style={{ color: 'var(--accent)' }} />
                <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Classification Dataset Format</span>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0 }}>
                <div style={{ padding: '14px 16px', borderRight: '1px solid var(--border)' }}>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 6 }}>📁 Folder / ZIP structure</p>
                  <div style={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6,
                    padding: '10px 12px', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 2 }}>
                    <div style={{ color: 'var(--secondary)' }}>📁 dataset/</div>
                    <div style={{ paddingLeft: 16, color: 'var(--accent)' }}>📁 cats/</div>
                    <div style={{ paddingLeft: 32, color: 'var(--text3)' }}>🖼 cat1.jpg</div>
                    <div style={{ paddingLeft: 32, color: 'var(--text3)' }}>🖼 cat2.jpg</div>
                    <div style={{ paddingLeft: 16, color: 'var(--accent)' }}>📁 dogs/</div>
                    <div style={{ paddingLeft: 32, color: 'var(--text3)' }}>🖼 dog1.jpg</div>
                    <div style={{ paddingLeft: 32, color: 'var(--text3)' }}>🖼 dog2.jpg</div>
                  </div>
                </div>
                <div style={{ padding: '14px 16px' }}>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)', marginBottom: 6 }}>📌 Rules</p>
                  <ul style={{ fontSize: 11, color: 'var(--text2)', lineHeight: 2.1, paddingLeft: 16, margin: 0 }}>
                    <li>One folder per class — folder name = class label</li>
                    <li>Supported images: <code>.jpg .jpeg .png .bmp .webp</code></li>
                    <li>Minimum 2 classes required for training</li>
                    <li>Recommended ≥ 20 images per class</li>
                    <li>ZIP must contain the class folders at the root or one level deep</li>
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Per-class rows */}
          <div style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto',
              padding: '8px 14px', background: 'var(--surface2)',
              borderBottom: '1px solid var(--border)',
              fontSize: 11, fontWeight: 600, color: 'var(--text3)',
              textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              <span>Class</span><span>Images</span><span>Actions</span>
            </div>

            {(project?.classes ?? []).length === 0 ? (
              <div style={{ padding: '28px 20px', textAlign: 'center', color: 'var(--text3)', fontSize: 12 }}>
                No classes yet — add class labels above or import a ZIP dataset
              </div>
            ) : (
              (project?.classes ?? []).map(cls => {
                const count = datasetStats[cls] ?? 0
                const dot   = count >= 10 ? '#3fb950' : count >= 2 ? '#f0a500' : '#666'
                return (
                  <div key={cls} style={{ display: 'grid', gridTemplateColumns: '1fr auto auto',
                    alignItems: 'center', gap: 12,
                    padding: '10px 14px', borderBottom: '1px solid var(--border)',
                    background: 'var(--surface)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ width: 8, height: 8, borderRadius: '50%',
                        background: dot, display: 'inline-block', flexShrink: 0 }} />
                      <span style={{ fontSize: 13, color: 'var(--text)', fontFamily: 'monospace' }}>{cls}</span>
                    </div>
                    <span style={{ fontSize: 12, color: 'var(--text3)', minWidth: 60, textAlign: 'right' }}>
                      {count} image{count !== 1 ? 's' : ''}
                    </span>
                    <div style={{ display: 'flex', gap: 6 }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: 5,
                        padding: '4px 10px', borderRadius: 6, fontSize: 12,
                        background: 'var(--surface2)', border: '1px solid var(--border)',
                        color: uploadingCls === cls ? 'var(--text3)' : 'var(--text2)',
                        cursor: uploadingCls === cls ? 'wait' : 'pointer' }}>
                        <input type="file" multiple accept="image/*" style={{ display: 'none' }}
                          disabled={uploadingCls === cls}
                          onChange={e => { if (e.target.files) uploadToClass(cls, e.target.files); e.target.value = '' }} />
                        <ImagePlus size={12} />
                        {uploadingCls === cls ? 'Uploading…' : 'Add'}
                      </label>
                      <button onClick={() => clearClass(cls)}
                        disabled={clearingCls === cls || count === 0}
                        style={{ display: 'flex', alignItems: 'center', gap: 5,
                          padding: '4px 10px', borderRadius: 6, fontSize: 12,
                          background: 'rgba(248,113,113,0.08)', border: '1px solid rgba(248,113,113,0.25)',
                          color: 'var(--danger)', cursor: clearingCls === cls || count === 0 ? 'not-allowed' : 'pointer',
                          opacity: count === 0 ? 0.4 : 1 }}>
                        <Trash2 size={12} />
                        {clearingCls === cls ? 'Clearing…' : 'Clear'}
                      </button>
                    </div>
                  </div>
                )
              })
            )}

            {/* New class row */}
            <div style={{ padding: '10px 14px', background: 'var(--surface)',
              borderTop: '1px solid var(--border)', display: 'flex', gap: 8, alignItems: 'center' }}>
              <input value={newClassName} onChange={e => setNewClassName(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && newClassFileRef.current?.click()}
                placeholder="New class name…"
                style={{ flex: 1, fontSize: 12, padding: '5px 10px', borderRadius: 6,
                  background: 'var(--surface2)', border: '1px solid var(--border)',
                  color: 'var(--text)', outline: 'none' }} />
              <Btn variant="secondary" size="sm"
                disabled={!newClassName.trim() || uploadingCls !== null}
                onClick={() => newClassFileRef.current?.click()}>
                <ImagePlus size={12} /> Choose images
              </Btn>
              <span style={{ fontSize: 11, color: 'var(--text3)' }}>Enter class name then pick images</span>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════════════════════════════════════════════
          OBJECT DETECTION PANEL (filter bar + upload + image grid)
          ═══════════════════════════════════════════════════════════ */}
      {datasetMode === 'detection' && <>

      {/* Filter + export bar */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {(['all','annotated','unannotated','corrupt'] as const).map(f => (
          <button key={f} onClick={() => setFilter(f)}
            style={{ padding: '4px 12px', borderRadius: 6, fontSize: 12, cursor: 'pointer',
              border: `1px solid ${filter === f ? 'var(--accent)' : 'var(--border)'}`,
              background: filter === f ? 'rgba(88,101,242,0.12)' : 'transparent',
              color: filter === f ? 'var(--accent)' : 'var(--text2)',
              fontWeight: filter === f ? 500 : 400, transition: 'all 0.12s' }}>
            {f.charAt(0).toUpperCase() + f.slice(1)}
            {f === 'all' && <span style={{ marginLeft: 5, color: 'var(--text3)' }}>{total}</span>}
            {f === 'annotated' && <span style={{ marginLeft: 5, color: 'var(--text3)' }}>{annotated}</span>}
            {f === 'unannotated' && <span style={{ marginLeft: 5, color: 'var(--text3)' }}>{total - annotated}</span>}
            {f === 'corrupt' && <span style={{ marginLeft: 5, color: 'var(--text3)' }}>{images.filter(i => i.is_corrupt).length}</span>}
          </button>
        ))}
        <div style={{ flex: 1 }} />
        <Btn variant="ghost" size="sm" href={`/api/projects/${projectId}/export/dataset?format=yolo`}>↓ YOLO</Btn>
        <Btn variant="ghost" size="sm" href={`/api/projects/${projectId}/export/dataset?format=coco`}>↓ COCO</Btn>
      </div>

      {/* Import result banner */}
      {importResult && (
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14,
          padding: '10px 16px', borderRadius: 8,
          background: 'rgba(59,165,93,0.1)', border: '1px solid rgba(59,165,93,0.3)',
        }}>
          <span style={{ fontSize: 13, color: 'var(--success)', fontWeight: 600 }}>
            ✓ Import complete
          </span>
          <span style={{ fontSize: 12, color: 'var(--text2)' }}>
            {importResult.imported} image{importResult.imported !== 1 ? 's' : ''} imported
            · {importResult.annotated} annotated
            {importResult.skipped_duplicates > 0 && ` · ${importResult.skipped_duplicates} duplicates skipped`}
            {importResult.classes_updated && ' · class names updated from classes.txt'}
          </span>
          <button onClick={() => setImportResult(null)}
            style={{ marginLeft: 'auto', background: 'none', border: 'none',
              color: 'var(--text3)', cursor: 'pointer', padding: 2 }}>
            <X size={13} />
          </button>
        </div>
      )}

      {/* Upload + select bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 12, alignItems: 'center', flexWrap: 'wrap' }}>
        {!selectMode ? (
          <>
            <Btn variant="primary" onClick={() => fileRef.current?.click()} disabled={loading}>
              <Upload size={13} /> {loading ? 'Uploading…' : 'Upload Images'}
            </Btn>
            <Btn variant="secondary" onClick={() => importFolderRef.current?.click()} disabled={importing}>
              <FolderInput size={13} /> {importing ? 'Importing…' : 'Import YOLO Folder'}
            </Btn>
            <Btn variant="ghost" onClick={() => importRef.current?.click()} disabled={importing}>
              <FolderInput size={13} /> {importing ? 'Importing…' : 'Import YOLO Files'}
            </Btn>
            {images.length > 0 && (
              <Btn variant="secondary" onClick={() => setSelectMode(true)}>
                <CheckSquare size={13} /> Select
              </Btn>
            )}
            {/* Format guide toggle */}
            <button
              onClick={() => setShowImportGuide(v => !v)}
              title="Show import format guide"
              style={{
                display: 'flex', alignItems: 'center', gap: 5,
                padding: '5px 10px', border: '1px solid var(--border)',
                borderRadius: 'var(--radius)', background: showImportGuide ? 'var(--accent-s)' : 'transparent',
                color: showImportGuide ? 'var(--accent)' : 'var(--text3)',
                fontSize: 12, cursor: 'pointer', transition: 'all 0.1s',
              }}
            >
              <Info size={12} />
              Format guide
              {showImportGuide ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
            </button>
          </>
        ) : (
          <>
            <button onClick={allFilteredSelected ? deselectAll : selectAll}
              style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px',
                border: '1px solid var(--border)', borderRadius: 6, background: 'var(--surface2)',
                color: 'var(--text)', fontSize: 12, cursor: 'pointer' }}>
              {allFilteredSelected
                ? <><CheckSquare size={13} /> Deselect all</>
                : <><Square size={13} /> Select all</>}
            </button>
            {selected.size > 0 && (
              <button onClick={bulkDelete} disabled={deleting}
                style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px',
                  border: '1px solid rgba(248,113,113,0.4)', borderRadius: 6,
                  background: 'rgba(248,113,113,0.1)', color: 'var(--danger)',
                  fontSize: 12, cursor: deleting ? 'wait' : 'pointer',
                  opacity: deleting ? 0.6 : 1 }}>
                <Trash2 size={13} />
                {deleting ? 'Deleting…' : `Delete ${selected.size}`}
              </button>
            )}
            <button onClick={exitSelectMode}
              style={{ display: 'flex', alignItems: 'center', gap: 4, padding: '6px 10px',
                border: '1px solid var(--border)', borderRadius: 6, background: 'transparent',
                color: 'var(--text2)', fontSize: 12, cursor: 'pointer' }}>
              <X size={13} /> Cancel
            </button>
            {selected.size > 0 && (
              <span style={{ fontSize: 12, color: 'var(--text2)' }}>
                {selected.size} selected
              </span>
            )}
          </>
        )}
        <input ref={fileRef} type="file" multiple accept="image/*" style={{ display: 'none' }} onChange={handleUpload} />
        <input ref={importRef} type="file" multiple accept="image/*,.txt" style={{ display: 'none' }} onChange={handleImportYolo} />
        {/* @ts-ignore */}
        <input ref={importFolderRef} type="file" multiple webkitdirectory="" style={{ display: 'none' }} onChange={handleImportYolo} />
      </div>

      {/* ── Import format guide ─────────────────────────────────────────── */}
      {showImportGuide && (
        <div style={{
          marginBottom: 20,
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)',
          background: 'var(--surface)',
          overflow: 'hidden',
        }}>
          {/* Header */}
          <div style={{
            padding: '10px 14px',
            borderBottom: '1px solid var(--border)',
            background: 'var(--surface2)',
            display: 'flex', alignItems: 'center', gap: 8,
          }}>
            <Info size={13} style={{ color: 'var(--accent)' }} />
            <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Dataset Import Formats</span>
            <span style={{ fontSize: 11, color: 'var(--text3)', marginLeft: 4 }}>— supported formats explained</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 0 }}>

            {/* ── Card 1: YOLO Folder ── */}
            <div style={{ padding: '14px 16px', borderRight: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 10 }}>
                <div style={{ width: 26, height: 26, borderRadius: 'var(--radius)', background: 'var(--secondary-s)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Folder size={13} style={{ color: 'var(--secondary)' }} />
                </div>
                <div>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Import YOLO Folder</p>
                  <p style={{ fontSize: 11, color: 'var(--text3)' }}>Select an entire dataset folder</p>
                </div>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text2)', marginBottom: 8, lineHeight: 1.6 }}>
                Your folder must contain two sub-folders: <code style={{ color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace' }}>images/</code> and <code style={{ color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace' }}>labels/</code> with matching filenames.
              </p>
              {/* Folder tree */}
              <div style={{
                background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
                padding: '10px 12px', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 1.9,
                color: 'var(--text2)',
              }}>
                <div style={{ color: 'var(--secondary)' }}>📁 my-dataset/</div>
                <div style={{ paddingLeft: 16 }}>📁 <span style={{ color: 'var(--text)' }}>images/</span></div>
                <div style={{ paddingLeft: 32, color: 'var(--text3)' }}>🖼 dog_001.jpg</div>
                <div style={{ paddingLeft: 32, color: 'var(--text3)' }}>🖼 dog_002.jpg</div>
                <div style={{ paddingLeft: 16 }}>📁 <span style={{ color: 'var(--text)' }}>labels/</span></div>
                <div style={{ paddingLeft: 32, color: 'var(--success)' }}>📄 dog_001.txt</div>
                <div style={{ paddingLeft: 32, color: 'var(--success)' }}>📄 dog_002.txt</div>
                <div style={{ paddingLeft: 16, color: 'var(--text3)', fontSize: 10, marginTop: 2 }}>obj.names or classes.txt (optional)</div>
              </div>
              <div style={{ marginTop: 8, padding: '6px 10px', background: 'var(--success-s)', borderRadius: 'var(--radius)', fontSize: 11, color: 'var(--success)' }}>
                ✓ Annotations are imported automatically
              </div>
            </div>

            {/* ── Card 2: YOLO Files ── */}
            <div style={{ padding: '14px 16px', borderRight: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 10 }}>
                <div style={{ width: 26, height: 26, borderRadius: 'var(--radius)', background: 'var(--accent-s)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <FileText size={13} style={{ color: 'var(--accent)' }} />
                </div>
                <div>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>Import YOLO Files</p>
                  <p style={{ fontSize: 11, color: 'var(--text3)' }}>Select image + label files together</p>
                </div>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text2)', marginBottom: 8, lineHeight: 1.6 }}>
                Select both your <code style={{ color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace' }}>.jpg/.png</code> images and their matching <code style={{ color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace' }}>.txt</code> label files in one file-picker selection.
              </p>
              <div style={{
                background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
                padding: '10px 12px', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 1.9,
                color: 'var(--text2)',
              }}>
                <div style={{ color: 'var(--text3)', fontSize: 10, marginBottom: 4 }}>// Select all of these at once:</div>
                <div>🖼 <span style={{ color: 'var(--text)' }}>cat_001.jpg</span></div>
                <div>📄 <span style={{ color: 'var(--success)' }}>cat_001.txt</span></div>
                <div>🖼 <span style={{ color: 'var(--text)' }}>cat_002.png</span></div>
                <div>📄 <span style={{ color: 'var(--success)' }}>cat_002.txt</span></div>
              </div>
              <div style={{ marginTop: 8, padding: '6px 10px', background: 'var(--warn-s)', borderRadius: 'var(--radius)', fontSize: 11, color: 'var(--warn)' }}>
                ⚠ Image and .txt must share the same filename
              </div>
            </div>

            {/* ── Card 3: YOLO label format ── */}
            <div style={{ padding: '14px 16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 10 }}>
                <div style={{ width: 26, height: 26, borderRadius: 'var(--radius)', background: 'rgba(63,185,80,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <FileText size={13} style={{ color: 'var(--success)' }} />
                </div>
                <div>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text)' }}>YOLO Label Format</p>
                  <p style={{ fontSize: 11, color: 'var(--text3)' }}>What's inside each .txt file</p>
                </div>
              </div>
              <p style={{ fontSize: 11, color: 'var(--text2)', marginBottom: 8, lineHeight: 1.6 }}>
                One object per line. All values are <strong>0–1 normalised</strong> relative to image width/height.
              </p>
              <div style={{
                background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 'var(--radius)',
                padding: '10px 12px', fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 2,
              }}>
                <div style={{ color: 'var(--text3)', fontSize: 10 }}>// class_id  cx  cy  width  height</div>
                <div>
                  <span style={{ color: 'var(--danger)' }}>0 </span>
                  <span style={{ color: 'var(--secondary)' }}>0.512 0.480 </span>
                  <span style={{ color: 'var(--success)' }}>0.300 0.420</span>
                </div>
                <div>
                  <span style={{ color: 'var(--danger)' }}>1 </span>
                  <span style={{ color: 'var(--secondary)' }}>0.210 0.340 </span>
                  <span style={{ color: 'var(--success)' }}>0.150 0.200</span>
                </div>
                <div style={{ marginTop: 8, borderTop: '1px solid var(--border)', paddingTop: 6, fontSize: 10, color: 'var(--text3)', lineHeight: 1.8 }}>
                  <span style={{ color: 'var(--danger)' }}>class_id</span> = index from classes list<br />
                  <span style={{ color: 'var(--secondary)' }}>cx cy</span> = box center (0–1)<br />
                  <span style={{ color: 'var(--success)' }}>w h</span> = box size (0–1)
                </div>
              </div>
              <div style={{ marginTop: 8, padding: '6px 10px', background: 'var(--secondary-s)', borderRadius: 'var(--radius)', fontSize: 11, color: 'var(--secondary)' }}>
                ✓ Works with Roboflow, CVAT, LabelImg exports
              </div>
            </div>

          </div>
        </div>
      )}

      {images.length === 0 ? (
        <div style={{ border: '1px dashed var(--border2)', borderRadius: 10 }}>
          <Empty icon={ImageIcon} message="No images yet" sub="Upload images to start annotating" />
        </div>
      ) : filteredImages.length === 0 ? (
        <div style={{ border: '1px dashed var(--border2)', borderRadius: 10 }}>
          <Empty icon={ImageIcon} message={`No ${filter} images`} sub="Try a different filter" />
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 8 }}>
          {filteredImages.map(img => {
            const isSelected = selected.has(img.id)
            return (
              <div key={img.id}
                onClick={selectMode ? () => toggleSelect(img.id) : undefined}
                style={{ background: 'var(--surface)', borderRadius: 8, overflow: 'hidden',
                  position: 'relative', cursor: selectMode ? 'pointer' : 'default',
                  border: `1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                  boxShadow: isSelected ? '0 0 0 2px rgba(88,101,242,0.35)' : 'none',
                  transition: 'border-color 0.1s, box-shadow 0.1s' }}
                onMouseEnter={e => { if (!selectMode) e.currentTarget.querySelector<HTMLDivElement>('.overlay')!.style.opacity = '1' }}
                onMouseLeave={e => { if (!selectMode) e.currentTarget.querySelector<HTMLDivElement>('.overlay')!.style.opacity = '0' }}>

                <img src={`/api/projects/${projectId}/images/${img.id}/file`} alt={img.original_name}
                  style={{ width: '100%', height: 120, objectFit: 'cover', display: 'block' }} />

                {/* Select mode checkbox */}
                {selectMode && (
                  <div style={{ position: 'absolute', top: 7, left: 7,
                    width: 18, height: 18, borderRadius: 4,
                    background: isSelected ? 'var(--accent)' : 'rgba(0,0,0,0.55)',
                    border: `2px solid ${isSelected ? 'var(--accent)' : 'rgba(255,255,255,0.5)'}`,
                    display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    {isSelected && <span style={{ color: '#fff', fontSize: 11, lineHeight: 1 }}>✓</span>}
                  </div>
                )}

                {/* Status dot */}
                <div style={{ position: 'absolute', top: 7, right: 7, width: 8, height: 8, borderRadius: '50%',
                  background: img.annotated ? 'var(--success)' : 'var(--surface3)',
                  border: '1.5px solid rgba(0,0,0,0.5)', boxShadow: '0 1px 3px rgba(0,0,0,0.5)' }} />

                {/* Hover overlay (non-select mode only) */}
                <div className="overlay" style={{ position: 'absolute', inset: 0, background: 'rgba(0,0,0,0.65)',
                  opacity: 0, transition: 'opacity 0.15s', display: 'flex', alignItems: 'center',
                  justifyContent: 'center', gap: 8 }}>
                  <button onClick={() => navigate(`/projects/${projectId}/annotate/${img.id}`)}
                    style={{ padding: '7px 7px', border: '1px solid rgba(255,255,255,0.2)', borderRadius: 6,
                      background: 'rgba(255,255,255,0.1)', color: '#fff', cursor: 'pointer' }}>
                    <Pencil size={13} />
                  </button>
                  <button onClick={() => deleteImage(img.id)}
                    style={{ padding: '7px 7px', border: '1px solid rgba(248,113,113,0.3)', borderRadius: 6,
                      background: 'rgba(248,113,113,0.1)', color: 'var(--danger)', cursor: 'pointer' }}>
                    <Trash2 size={13} />
                  </button>
                </div>

                {/* Filename */}
                <div style={{ padding: '6px 8px', borderTop: '1px solid var(--border)' }}>
                  <p style={{ fontSize: 11, color: 'var(--text2)', whiteSpace: 'nowrap',
                    overflow: 'hidden', textOverflow: 'ellipsis' }}>{img.original_name}</p>
                  {img.is_corrupt && <Badge color="red">corrupt</Badge>}
                </div>
              </div>
            )
          })}
        </div>
      )}

      </> /* end detection panel */}

    </div>
  )
}
