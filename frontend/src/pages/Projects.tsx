import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Plus, Trash2, X, ChevronRight } from 'lucide-react'
import api, { type Project } from '../api'
import { Btn, TextInput, Field, Badge } from '../components/ui'

export default function Projects() {
  const [projects, setProjects]     = useState<Project[]>([])
  const [showForm, setShowForm]     = useState(false)
  const [name, setName]             = useState('')
  const [desc, setDesc]             = useState('')
  const [classInput, setClassInput] = useState('')
  const [classes, setClasses]       = useState<string[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [error, setError]           = useState('')
  const navigate = useNavigate()

  const load = () => api.get('/projects').then(r => setProjects(r.data)).catch(() => setError('Failed to load projects. Is the server running?'))
  useEffect(() => { load() }, [])

  const addClass = () => {
    const t = classInput.trim()
    if (t && !classes.includes(t)) { setClasses([...classes, t]); setClassInput('') }
  }

  const submit = async () => {
    if (!name.trim() || submitting) return
    setSubmitting(true)
    setError('')
    try {
      await api.post('/projects', { name: name.trim(), description: desc, classes })
      setName(''); setDesc(''); setClasses([]); setShowForm(false); load()
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? 'Failed to create project.')
    } finally { setSubmitting(false) }
  }

  const remove = async (id: number, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm('Delete this project and all its data?')) {
      await api.delete(`/projects/${id}`); load()
    }
  }

  return (
    <div style={{ maxWidth: 860, margin: '0 auto' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 32 }}>
        <div>
          <h1 style={{ fontSize: 22, fontWeight: 600, color: 'var(--text)', letterSpacing: '-0.02em' }}>Projects</h1>
          <p style={{ fontSize: 13, color: 'var(--text2)', marginTop: 4 }}>Each project trains one computer vision model</p>
        </div>
        <Btn variant="primary" onClick={() => setShowForm(true)}>
          <Plus size={14} strokeWidth={2.5} /> New Project
        </Btn>
      </div>

      {/* Load error */}
      {error && !showForm && (
        <div style={{ fontSize: 13, color: 'var(--red, #f87171)', marginBottom: 16 }}>{error}</div>
      )}

      {/* Project grid */}
      {projects.length === 0 ? (
        <div style={{ border: '1px dashed var(--border2)', borderRadius: 10, padding: '64px 24px',
          textAlign: 'center', color: 'var(--text3)' }}>
          <p style={{ fontSize: 14, marginBottom: 4 }}>No projects yet</p>
          <p style={{ fontSize: 12 }}>Create your first project to get started</p>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12 }}>
          {projects.map(p => (
            <div key={p.id} onClick={() => navigate(`/projects/${p.id}/images`)}
              style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10,
                padding: '18px 20px', cursor: 'pointer', transition: 'border-color 0.15s',
                display: 'flex', flexDirection: 'column', gap: 12 }}
              onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--border2)')}
              onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <h3 style={{ fontSize: 14, fontWeight: 600, color: 'var(--text)', letterSpacing: '-0.01em',
                    whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {p.name}
                  </h3>
                  {p.description && (
                    <p style={{ fontSize: 12, color: 'var(--text2)', marginTop: 2,
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {p.description}
                    </p>
                  )}
                </div>
                <div style={{ display: 'flex', gap: 4, marginLeft: 8 }}>
                  <button onClick={e => remove(p.id, e)}
                    style={{ padding: '4px', border: 'none', background: 'transparent',
                      color: 'var(--text3)', cursor: 'pointer', borderRadius: 4,
                      display: 'flex', alignItems: 'center' }}>
                    <Trash2 size={13} />
                  </button>
                  <ChevronRight size={14} style={{ color: 'var(--text3)', marginTop: 2 }} />
                </div>
              </div>
              {p.classes.length > 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
                  {p.classes.slice(0, 6).map((c, i) => (
                    <span key={i} style={{ fontSize: 11, padding: '2px 7px', borderRadius: 4,
                      background: 'var(--surface3)', color: 'var(--text2)', border: '1px solid var(--border)' }}>
                      {c}
                    </span>
                  ))}
                  {p.classes.length > 6 && (
                    <span style={{ fontSize: 11, color: 'var(--text3)' }}>+{p.classes.length - 6}</span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Create modal */}
      {showForm && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)', display: 'flex',
          alignItems: 'center', justifyContent: 'center', zIndex: 100, backdropFilter: 'blur(4px)' }}>
          <div style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 12,
            padding: 24, width: '100%', maxWidth: 440 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
              <h2 style={{ fontSize: 16, fontWeight: 600, color: 'var(--text)', letterSpacing: '-0.01em' }}>New Project</h2>
              <button onClick={() => setShowForm(false)}
                style={{ padding: 4, border: 'none', background: 'transparent', color: 'var(--text2)', cursor: 'pointer', borderRadius: 4 }}>
                <X size={16} />
              </button>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              <Field label="Project name *">
                <TextInput value={name} onChange={setName} placeholder="e.g. Fire Detection" />
              </Field>
              <Field label="Description">
                <TextInput value={desc} onChange={setDesc} placeholder="Optional" />
              </Field>
              <Field label="Detection classes">
                <div style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
                  <TextInput value={classInput} onChange={setClassInput} placeholder="e.g. smoke"
                    onKeyDown={e => e.key === 'Enter' && addClass()} />
                  <Btn variant="secondary" onClick={addClass} size="sm" style={{ flexShrink: 0 }}>Add</Btn>
                </div>
                {classes.length > 0 && (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                    {classes.map((c, i) => (
                      <Badge key={i} color="blue">
                        {c}
                        <button onClick={() => setClasses(classes.filter((_, j) => j !== i))}
                          style={{ border: 'none', background: 'transparent', color: 'inherit', cursor: 'pointer',
                            padding: 0, lineHeight: 1, marginLeft: 2, fontSize: 14 }}>×</button>
                      </Badge>
                    ))}
                  </div>
                )}
              </Field>

              {error && (
                <p style={{ fontSize: 12, color: 'var(--red, #f87171)', marginTop: -4 }}>{error}</p>
              )}
              <Btn variant="primary" onClick={submit} disabled={!name.trim() || submitting}
                style={{ width: '100%', justifyContent: 'center', marginTop: 4 }}>
                {submitting ? 'Creating…' : 'Create Project'}
              </Btn>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
