import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { Eye } from 'lucide-react'
import Projects from './pages/Projects'
import ProjectImages from './pages/ProjectImages'
import Annotate from './pages/Annotate'
import Train from './pages/Train'
import Analytics from './pages/Analytics'
import Evaluation from './pages/Evaluation'
import Classification from './pages/Classification'
import CustomModel from './pages/CustomModel'
import Webcam from './pages/Webcam'
import Pose from './pages/Pose'
import Segmentation from './pages/Segmentation'
import Depth from './pages/Depth'

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

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ minHeight: '100vh', background: 'var(--bg)', display: 'flex', flexDirection: 'column' }}>
        <Nav />
        <main style={{ flex: 1, padding: '32px 24px' }}>
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
        </main>
      </div>
    </BrowserRouter>
  )
}
