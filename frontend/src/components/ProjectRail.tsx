import { useEffect, useState } from 'react'
import { NavLink } from 'react-router-dom'
import api, { type Project } from '../api'
import { Lotus } from './Sigil'

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
  return (
    <aside className="kh-rail" aria-label="Project tools">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '4px 10px 16px' }}>
        <Lotus size={32} seed={projectId} />
        <div style={{ minWidth: 0 }}>
          <p className="font-display" style={{ fontSize: 20, lineHeight: 1.1, paddingTop: 2, color: 'var(--text)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {project?.name ?? 'Project'}
          </p>
          <p style={{ fontSize: 11.5, color: 'var(--text3)' }}>
            {project ? `${project.classes.length} ${project.classes.length === 1 ? 'class' : 'classes'}` : ' '}
          </p>
        </div>
      </div>
      <nav className="kh-rail-groups" style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {GROUPS.map(g => (
          <div key={g.title}>
            <h3 style={{ fontSize: 11.5, fontWeight: 600, color: 'var(--text3)', padding: '0 10px 4px' }}>{g.title}</h3>
            {g.items.map(it => {
              const active = section === it.to || (it.match?.includes(section) ?? false)
              return (
                <NavLink key={it.to} to={`/projects/${projectId}/${it.to}`}
                  className={`kh-nav-link${active ? ' active' : ''}`}
                  aria-current={active ? 'page' : undefined}>
                  <span className="pip" />
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
