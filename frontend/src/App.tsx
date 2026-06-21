import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { Eye, Loader } from 'lucide-react'
import Projects from './pages/Projects'   // landing page — keep eager for instant first paint

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

function Nav() {
  return (
    <header style={{ borderBottom: '1px solid var(--border)', background: 'var(--surface)' }}
      className="sticky top-0 z-40">
      <div className="flex items-center gap-6 px-6 h-12">
        <NavLink to="/" className="flex items-center gap-2 no-underline">
          <Eye size={16} style={{ color: 'var(--accent)' }} strokeWidth={1.8} />
          <span className="font-display" style={{ color: 'var(--text)', fontWeight: 700, fontSize: 15, letterSpacing: '0.04em' }}>
            NoCode CV
          </span>
        </NavLink>
        <div style={{ width: 1, height: 16, background: 'var(--border2)' }} />
        <NavLink to="/" end style={({ isActive }) => ({
          fontSize: 13,
          color: isActive ? 'var(--text)' : 'var(--text2)',
          textDecoration: 'none',
          fontWeight: isActive ? 500 : 400,
        })}>
          Projects
        </NavLink>
      </div>
    </header>
  )
}

function PageFallback() {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center',
      padding: '80px 0', color: 'var(--text3)', gap: 10 }}>
      <Loader size={18} className="animate-spin" style={{ color: 'var(--accent)' }} />
      <span style={{ fontSize: 13 }}>Loading…</span>
    </div>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ minHeight: '100vh', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>
        <Nav />
        <main style={{ flex: 1, padding: '32px 24px' }}>
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
            </Routes>
          </Suspense>
        </main>
      </div>
    </BrowserRouter>
  )
}
