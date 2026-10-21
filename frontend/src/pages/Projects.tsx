import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Plus, Trash2, X, Sparkles } from 'lucide-react'
import api, { type Project } from '../api'
import { Btn, TextInput, Field } from '../components/ui'
import AngkorField from '../components/AngkorField'
import { Lotus } from '../components/Sigil'
import { getWorld } from '../components/KhmerWorlds'

export default function Projects() {
  const [projects, setProjects] = useState<Project[] | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [name, setName]         = useState('')
  const [desc, setDesc]         = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [seeding, setSeeding]       = useState(false)
  const [error, setError]       = useState('')
  const [formError, setFormError] = useState('')
  const navigate = useNavigate()

  const load = () => api.get('/projects')
    .then(r => { setProjects(r.data); setError('') })
    .catch(() => { setProjects([]); setError("Can't reach the NoCode CV server. Start it with “NoCode CV.bat”, then reload this page.") })
  useEffect(() => { load() }, [])

  const submit = async () => {
    if (!name.trim() || submitting) return
    setSubmitting(true)
    setFormError('')
    try {
      const r = await api.post('/projects', { name: name.trim(), description: desc, classes: [] })
      setName(''); setDesc(''); setShowForm(false)
      if (r.data?.id) navigate(`/projects/${r.data.id}/images`)
      else load()
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setFormError(detail ?? "Couldn't create the project. Check that the server is running.")
    } finally { setSubmitting(false) }
  }

  // One-click demo: the server builds a sample dataset and model config.
  const loadSample = async () => {
    if (seeding) return
    setSeeding(true); setError('')
    try {
      const r = await api.post('/sample/project')
      navigate(`/projects/${r.data.project_id}/custom`)
    } catch (e: unknown) {
      const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail
      setError(detail ?? "Couldn't create the sample project. Check that the server is running.")
    } finally { setSeeding(false) }
  }

  const remove = async (p: Project, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm(`Delete “${p.name}” with all its images, labels and trained models?`)) {
      await api.delete(`/projects/${p.id}`); load()
    }
  }

  const count = projects?.length ?? 0

  return (
    <div>
      {/* ── Hero: Angkor at dawn, seen by a model ───────────────────────── */}
      <section style={{ position: 'relative', margin: '-28px -24px 0', height: 440, overflow: 'hidden' }}>
        <AngkorField height={440} />
        <div aria-hidden className="kh-hero-veil" />
        <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none', display: 'flex', alignItems: 'flex-end' }}>
          <div className="kh-rise" style={{ width: '100%', maxWidth: 1228, margin: '0 auto', padding: '0 24px 46px' }}>
            <h1 className="font-display" style={{ fontSize: 'clamp(46px, 7vw, 92px)', lineHeight: 0.92,
              color: 'var(--text)', maxWidth: 560 }}>
              Teach a machine to see
            </h1>
            <p className="font-khmer" lang="km" style={{ fontSize: 'clamp(17px, 2vw, 23px)', color: 'var(--accent)', marginTop: 8 }}>
              បង្រៀនម៉ាស៊ីនឱ្យមើលឃើញ
            </p>
            <p style={{ marginTop: 14, maxWidth: 440, fontSize: 15.5, lineHeight: 1.6, color: 'var(--text2)' }}>
              Label your images, train a detector or classifier, and test it on your webcam.
              Everything runs on this computer. Move your pointer over the temple to see it the way a model does.
            </p>
            <div style={{ marginTop: 22, pointerEvents: 'auto', display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <Btn variant="primary" onClick={() => { setFormError(''); setShowForm(true) }}>
                <Plus size={15} strokeWidth={2.5} /> New project
              </Btn>
              <Btn variant="secondary" onClick={loadSample} disabled={seeding}>
                <Sparkles size={15} strokeWidth={2.2} /> {seeding ? 'Preparing…' : 'Try a sample'}
              </Btn>
            </div>
          </div>
        </div>
      </section>
      <div className="kh-frieze" aria-hidden style={{ margin: '0 -24px' }} />

      {/* ── Projects ─────────────────────────────────────────────────────── */}
      <section style={{ maxWidth: 1180, margin: '0 auto', paddingTop: 40 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 16, marginBottom: 18 }}>
          <h2 className="font-display" style={{ fontSize: 34, color: 'var(--text)', lineHeight: 1 }}>Your projects</h2>
          {count > 0 && <span style={{ fontSize: 13, color: 'var(--text3)' }}>{count} {count === 1 ? 'project' : 'projects'}</span>}
        </div>

        {error && (
          <div role="alert" style={{ fontSize: 13.5, color: 'var(--danger)', marginBottom: 18, padding: '12px 16px',
            background: 'var(--danger-s)', border: '1px solid rgba(239,106,85,0.25)', borderRadius: 'var(--radius)' }}>
            {error}
          </div>
        )}

        {projects === null ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: 14 }}>
            {[0, 1, 2].map(i => <div key={i} style={{ height: 138, borderRadius: 'var(--radius-lg)', background: 'var(--surface)', border: '1px solid var(--border)', opacity: 0.6 }} />)}
          </div>
        ) : count === 0 && !error ? (
          <FirstProject onCreate={() => setShowForm(true)} onSample={loadSample} seeding={seeding} />
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(270px, 1fr))', gap: 14 }}>
            {projects.map(p => { const w = getWorld(p.id); return (
              <div key={p.id} className="project-card" role="link" tabIndex={0}
                aria-label={`Open ${p.name}`}
                onClick={() => navigate(`/projects/${p.id}/images`)}
                onKeyDown={e => { if (e.key === 'Enter') navigate(`/projects/${p.id}/images`) }}
                style={{ ['--world-accent' as string]: w.vars['--accent'], ['--world-rgb' as string]: w.vars['--world-rgb'] }}>
                <div className="kw-card-art">
                  <w.Scene className="kw-fill" />
                  <span className="kw-card-tag"><span lang="km" className="font-khmer">{w.km}</span> {w.en}</span>
                </div>
                <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
                  <Lotus size={40} seed={p.id} />
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <h3 style={{ fontSize: 15.5, fontWeight: 600, color: 'var(--text)',
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {p.name}
                    </h3>
                    <p style={{ fontSize: 12.5, color: p.description ? 'var(--text2)' : 'var(--text3)', marginTop: 2,
                      whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {p.description || 'No description'}
                    </p>
                  </div>
                  <button className="card-delete" onClick={e => remove(p, e)} aria-label={`Delete ${p.name}`}
                    title="Delete project"
                    style={{ padding: 6, border: 'none', background: 'transparent',
                      color: 'var(--text3)', cursor: 'pointer', borderRadius: 6, display: 'flex' }}>
                    <Trash2 size={14} />
                  </button>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, minHeight: 22 }}>
                  {p.classes.length === 0 && <span style={{ fontSize: 12, color: 'var(--text3)' }}>No classes yet</span>}
                  {p.classes.slice(0, 6).map((c, i) => (
                    <span key={i} style={{ fontSize: 11.5, padding: '2px 9px', borderRadius: 999,
                      background: 'var(--surface2)', color: 'var(--text2)', border: '1px solid var(--border)' }}>
                      {c}
                    </span>
                  ))}
                  {p.classes.length > 6 && (
                    <span style={{ fontSize: 11.5, color: 'var(--text3)', alignSelf: 'center' }}>+{p.classes.length - 6} more</span>
                  )}
                </div>
              </div>
            ) })}
          </div>
        )}
      </section>

      {showForm && (
        <CreateDialog
          name={name} desc={desc} setName={setName} setDesc={setDesc}
          error={formError} submitting={submitting}
          onSubmit={submit} onClose={() => setShowForm(false)} />
      )}
    </div>
  )
}

function FirstProject({ onCreate, onSample, seeding }: { onCreate: () => void; onSample: () => void; seeding: boolean }) {
  const steps = [
    ['Add images', 'Drop in photos, a folder, a video, or a YOLO dataset you already have.'],
    ['Label them', 'Draw boxes or polygons around each thing you want the model to find.'],
    ['Train and test', 'Pick a model size, train, then try it on new images or your webcam.'],
  ]
  return (
    <div style={{ border: '1px dashed var(--border2)', borderRadius: 'var(--radius-xl)', padding: '28px 28px 26px' }}>
      <p style={{ fontSize: 15, color: 'var(--text)', fontWeight: 600 }}>Start your first project</p>
      <p style={{ fontSize: 13.5, color: 'var(--text2)', marginTop: 4, maxWidth: 520 }}>
        A project holds one dataset and the models you train on it.
      </p>
      <ol style={{ listStyle: 'none', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 18, margin: '22px 0 24px' }}>
        {steps.map(([t, d], i) => (
          <li key={t} style={{ display: 'flex', gap: 12 }}>
            <span style={{ fontSize: 28, fontWeight: 700, lineHeight: 1, color: 'var(--accent)', width: 24 }}>{i + 1}</span>
            <div>
              <p style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text)' }}>{t}</p>
              <p style={{ fontSize: 13, color: 'var(--text2)', marginTop: 2, lineHeight: 1.55 }}>{d}</p>
            </div>
          </li>
        ))}
      </ol>
      <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
        <Btn variant="primary" onClick={onCreate}><Plus size={15} strokeWidth={2.5} /> New project</Btn>
        <Btn variant="secondary" onClick={onSample} disabled={seeding}>
          <Sparkles size={15} strokeWidth={2.2} /> {seeding ? 'Preparing…' : 'Load a sample project'}
        </Btn>
      </div>
    </div>
  )
}

function CreateDialog({ name, desc, setName, setDesc, error, submitting, onSubmit, onClose }: {
  name: string; desc: string; setName: (v: string) => void; setDesc: (v: string) => void
  error: string; submitting: boolean; onSubmit: () => void; onClose: () => void
}) {
  const boxRef = useRef<HTMLDivElement>(null)
  const closeRef = useRef(onClose)
  useEffect(() => { closeRef.current = onClose })
  useEffect(() => {
    boxRef.current?.querySelector('input')?.focus()
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeRef.current() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return (
    <div className="guide-backdrop-in" onMouseDown={e => { if (e.target === e.currentTarget) onClose() }}
      style={{ position: 'fixed', inset: 0, background: 'rgba(8,6,20,0.75)', display: 'flex',
        alignItems: 'center', justifyContent: 'center', zIndex: 100, backdropFilter: 'blur(6px)', padding: 16 }}>
      <div ref={boxRef} role="dialog" aria-modal="true" aria-labelledby="new-project-title" className="guide-card-in"
        style={{ background: 'var(--surface)', border: '1px solid var(--border2)', borderRadius: 'var(--radius-xl)',
          padding: 26, width: '100%', maxWidth: 460, boxShadow: 'var(--shadow-lg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <h2 id="new-project-title" className="font-display" style={{ fontSize: 30, color: 'var(--text)', lineHeight: 1 }}>New project</h2>
          <button onClick={onClose} aria-label="Close"
            style={{ padding: 6, border: 'none', background: 'transparent', color: 'var(--text2)', cursor: 'pointer', borderRadius: 6, display: 'flex' }}>
            <X size={17} />
          </button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Field label="Name">
            <TextInput value={name} onChange={setName} placeholder="e.g. Fire detection"
              onKeyDown={e => { if (e.key === 'Enter') onSubmit() }} />
          </Field>
          <Field label="Description" hint="optional">
            <TextInput value={desc} onChange={setDesc} placeholder="What should the model find?"
              onKeyDown={e => { if (e.key === 'Enter') onSubmit() }} />
          </Field>

          <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.55 }}>
            You'll add class labels on the next page, or they're filled in for you when you import a
            YOLO dataset with a <code style={{ fontSize: 11.5, color: 'var(--secondary)' }}>classes.txt</code>.
          </p>

          {error && <p role="alert" style={{ fontSize: 12.5, color: 'var(--danger)' }}>{error}</p>}
          <Btn variant="primary" onClick={onSubmit} disabled={!name.trim() || submitting}
            style={{ width: '100%', justifyContent: 'center', marginTop: 4, padding: '10px 18px' }}>
            {submitting ? 'Creating…' : 'Create project'}
          </Btn>
        </div>
      </div>
    </div>
  )
}
