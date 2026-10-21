import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import api, { type Project } from '../api'
import { Lotus } from './Sigil'
import { LotusPip } from './Kbach'
import { WORLDS, setWorld, useCurrentWorld } from './KhmerWorlds'

// Left-hand tool rail for every page inside a project. Groups follow the
// dataset each tool needs, so it is clear which annotations feed which model.
const GROUPS: { title: string; items: { to: string; label: string; match?: string[] }[] }[] = [
  { title: 'Dataset', items: [
    { to: 'images', label: 'Images & labels' },
    { to: 'analytics', label: 'Analytics' },
  ]},
  { title: 'Detection', items: [
    { to: 'train', label: 'Train detector', match: ['eval'] },
    { to: 'segment', label: 'Segmentation' },
    { to: 'pose', label: 'Pose' },
  ]},
  { title: 'Classification', items: [
    { to: 'classify', label: 'Image classifier' },
    { to: 'custom', label: 'CNN builder' },
  ]},
  { title: 'Try a model', items: [
    { to: 'webcam', label: 'Webcam' },
    { to: 'road-safety', label: 'Road safety' },
    { to: 'guidance', label: 'Steering guide' },
    { to: 'depth', label: 'Depth & 3D' },
  ]},
]

const cache = new Map<string, Project>()

export function useProject(id: string | undefined) {
  const [p, setP] = useState<Project | null>(id ? cache.get(id) ?? null : null)
  useEffect(() => {
    if (!id) return
    let live = true
    api.get(`/projects/${id}`).then(r => { cache.set(id, r.data); if (live) setP(r.data) }).catch(() => {})
    return () => { live = false }
  }, [id])
  return p
}

export default function ProjectRail({ projectId, section }: { projectId: string; section: string }) {
  const project = useProject(projectId)
  const world = useCurrentWorld()
  return (
    <aside className="kh-rail" aria-label="Project tools">
      <div className="kh-rail-sky" style={{ height: 116 }}>
        {world && <world.Scene className="kw-fill" />}
        <div style={{ position: 'absolute', left: 0, right: 0, bottom: 0, display: 'flex', alignItems: 'center', gap: 9,
          padding: '22px 10px 8px', background: 'linear-gradient(0deg, rgba(0,0,0,0.78), rgba(0,0,0,0))' }}>
          <Lotus size={28} seed={projectId} />
          <div style={{ minWidth: 0 }}>
            <p className="font-display" style={{ fontSize: 19, lineHeight: 1.05, paddingTop: 2, color: 'var(--text)',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {project?.name ?? 'Project'}
            </p>
            <p style={{ fontSize: 11.5, color: 'var(--text2)' }}>
              {project ? `${project.classes.length} ${project.classes.length === 1 ? 'class' : 'classes'}` : '\u00a0'}
            </p>
          </div>
        </div>
      </div>
      {world && (
        <div style={{ padding: '0 4px 14px' }}>
          <p style={{ fontSize: 12, color: 'var(--text2)' }}>
            <span lang="km" className="font-khmer" style={{ color: 'var(--accent)', fontSize: 12, marginRight: 6 }}>{world.km}</span>
            {world.en}
          </p>
          <p style={{ fontSize: 11.5, color: 'var(--text3)', lineHeight: 1.45 }}>{world.blurb}</p>
          <div className="kw-picker" role="group" aria-label="Art style for this project">
            {WORLDS.map(w => (
              <button key={w.id} className="kw-swatch" aria-pressed={w.id === world.id}
                title={`${w.en}: ${w.blurb}`} aria-label={`Use ${w.en} art`}
                onClick={() => setWorld(projectId, w.id)}>
                <w.Scene className="kw-fill" />
              </button>
            ))}
          </div>
        </div>
      )}
      <nav className="kh-rail-groups" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {GROUPS.map((g, gi) => (
          <div key={g.title}>
            {gi > 0 && <div className="kh-frieze kh-rail-divider" aria-hidden />}
            <h3 style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text3)', padding: '0 10px 4px' }}>{g.title}</h3>
            {g.items.map(it => {
              const active = section === it.to || (it.match?.includes(section) ?? false)
              return (
                <NavLink key={it.to} to={`/projects/${projectId}/${it.to}`}
                  className={`kh-nav-link${active ? ' active' : ''}`}
                  aria-current={active ? 'page' : undefined}>
                  <LotusPip />
                  {it.label}
                </NavLink>
              )
            })}
          </div>
        ))}
      </nav>
    </aside>
  )
}
