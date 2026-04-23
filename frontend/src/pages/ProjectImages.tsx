import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Upload, Pencil, Trash2, BarChart2, Zap, Brain, CheckSquare, Square, X, Cpu } from 'lucide-react'
import api, { type ImageItem, type Project } from '../api'
import { PageHeader, Btn, Badge, Empty } from '../components/ui'
import { Image as ImageIcon } from 'lucide-react'

export default function ProjectImages() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const [project, setProject]   = useState<Project | null>(null)
  const [images, setImages]     = useState<ImageItem[]>([])
  const [loading, setLoading]   = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [selectMode, setSelectMode] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)
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
          <Btn variant="primary" size="sm" onClick={() => navigate(`/projects/${projectId}/train`)}>
            <Zap size={13} strokeWidth={2.5} /> Detect & Train
          </Btn>
        </>}
      />

      {/* Class tags */}
      {project?.classes && project.classes.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 20 }}>
          {project.classes.map((c, i) => (
            <span key={i} style={{ fontSize: 11, padding: '2px 8px', borderRadius: 4,
              background: 'var(--surface2)', color: 'var(--text2)', border: '1px solid var(--border)',
              fontFamily: 'JetBrains Mono, monospace' }}>
              {i}: {c}
            </span>
          ))}
        </div>
      )}

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

      {/* Upload + select bar */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 20, alignItems: 'center', flexWrap: 'wrap' }}>
        {!selectMode ? (
          <>
            <Btn variant="primary" onClick={() => fileRef.current?.click()} disabled={loading}>
              <Upload size={13} /> {loading ? 'Uploading…' : 'Upload Images'}
            </Btn>
            {images.length > 0 && (
              <Btn variant="secondary" onClick={() => setSelectMode(true)}>
                <CheckSquare size={13} /> Select
              </Btn>
            )}
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
      </div>

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
    </div>
  )
}
