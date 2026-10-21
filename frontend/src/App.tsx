import { BrowserRouter, Routes, Route, NavLink, useLocation, matchPath } from 'react-router-dom'
import { lazy, Suspense, type ReactNode } from 'react'
import Projects from './pages/Projects'   // landing page — keep eager for instant first paint
import { PrasatMark, Lotus } from './components/Sigil'
import ProjectRail, { useProject } from './components/ProjectRail'
import AmbientPetals from './components/AmbientPetals'

// Heavy / less-frequent pages are code-split so the initial bundle stays small.
// Each becomes its own chunk, fetched on first navigation.
const ProjectImages = lazy(() => import('./pages/ProjectImages'))
const Annotate      = lazy(() => import('./pages/Annotate'))
const Train         = lazy(() => import('./pages/Train'))
const Analytics     = lazy(() => import('./pages/Analytics'))
const Evaluation    = lazy(() => import('./pages/Evaluation'))
const Classification = lazy(() => import('./pages/Classification'))
const CustomModel   = lazy(() => import('./pages/CustomModel'))
const Webcam        = lazy(() => import('./pages/Webcam'))
const Pose          = lazy(() => import('./pages/Pose'))
const Segmentation  = lazy(() => import('./pages/Segmentation'))
const Depth         = lazy(() => import('./pages/Depth'))

// Pages that need the full width (canvas editors) skip the tool rail.
const FULL_BLEED = new Set(['annotate', 'custom'])

function useProjectRoute() {
  const { pathname } = useLocation()
  const m = matchPath('/projects/:id/:section/*', pathname)
  return m ? { id: m.params.id!, section: m.params.section! } : null
}

function TopBar() {
  const route = useProjectRoute()
  const project = useProject(route?.id)
  return (
    <header className="sticky top-0 z-40" style={{
      background: 'rgba(16, 14, 34, 0.84)', backdropFilter: 'blur(10px)',
      borderBottom: '1px solid var(--border)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '0 24px', height: 56 }}>
        <NavLink to="/" aria-label="NoCode CV home" style={{ display: 'flex', alignItems: 'center', gap: 10, textDecoration: 'none' }}>
          <PrasatMark size={28} />
          <span className="font-display" style={{ color: 'var(--text)', fontSize: 23, lineHeight: 1, paddingTop: 3 }}>
            NoCode CV
          </span>
        </NavLink>
        <nav aria-label="Breadcrumb" style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0, fontSize: 13 }}>
          <span aria-hidden style={{ color: 'var(--border2)' }}>/</span>
          <NavLink to="/" end style={({ isActive }) => ({
            color: isActive ? 'var(--text)' : 'var(--text2)', textDecoration: 'none',
          })}>Projects</NavLink>
          {route && <>
            <span aria-hidden style={{ color: 'var(--border2)' }}>/</span>
            <NavLink to={`/projects/${route.id}/images`} style={{
              color: 'var(--text)', textDecoration: 'none',
              overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 260,
            }}>{project?.name ?? '…'}</NavLink>
          </>}
        </nav>
      </div>
    </header>
  )
}

function PageFallback() {
  return (
    <div role="status" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '96px 0', color: 'var(--text3)', gap: 12 }}>
      <span className="kh-blink"><Lotus size={24} seed="loading" /></span>
      <span style={{ fontSize: 13 }}>Opening…</span>
    </div>
  )
}

function Frame({ children }: { children: ReactNode }) {
  const route = useProjectRoute()
  const { pathname } = useLocation()
  const withRail = route && !FULL_BLEED.has(route.section)
  return (
    <main style={{ flex: 1, padding: '28px 24px 48px', width: '100%' }}>
      {withRail ? (
        <div className="kh-layout" style={{ maxWidth: 1340, margin: '0 auto' }}>
          <ProjectRail projectId={route.id} section={route.section} />
          <div key={pathname} className="kh-page" style={{ minWidth: 0 }}>{children}</div>
        </div>
      ) : children}
    </main>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <AmbientRoute />
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative', zIndex: 1 }}>
        <TopBar />
        <Frame>
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/"                               element={<Projects />} />
              <Route path="/projects/:id/images"            element={<ProjectImages />} />
              <Route path="/projects/:id/annotate/:imageId" element={<Annotate />} />
              <Route path="/projects/:id/train"             element={<Train />} />
              <Route path="/projects/:id/analytics"         element={<Analytics />} />
              <Route path="/projects/:id/eval/:runId"       element={<Evaluation />} />
              <Route path="/projects/:id/classify"          element={<Classification />} />
              <Route path="/projects/:id/custom"            element={<CustomModel />} />
              <Route path="/projects/:id/webcam"            element={<Webcam />} />
              <Route path="/projects/:id/pose"              element={<Pose />} />
              <Route path="/projects/:id/segment"           element={<Segmentation />} />
              <Route path="/projects/:id/depth"             element={<Depth />} />
              <Route path="*"                               element={<NotFound />} />
            </Routes>
          </Suspense>
        </Frame>
      </div>
    </BrowserRouter>
  )
}

function NotFound() {
  return (
    <div style={{ maxWidth: 520, margin: '80px auto', textAlign: 'center' }}>
      <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 18 }}><Lotus size={60} seed="lost" /></div>
      <h1 className="font-display" style={{ fontSize: 40 }}>Nothing to see here</h1>
      <p style={{ color: 'var(--text2)', marginTop: 8 }}>This page doesn't exist. Go back to your projects to pick up where you left off.</p>
      <NavLink to="/" style={{ display: 'inline-block', marginTop: 20, color: 'var(--accent)' }}>Open projects</NavLink>
    </div>
  )
}

// Drifting petals everywhere except the canvas editors, where nothing should move behind the work.
function AmbientRoute() {
  const route = useProjectRoute()
  if (route && FULL_BLEED.has(route.section)) return null
  return <AmbientPetals density={route ? 0.7 : 1} />
}
