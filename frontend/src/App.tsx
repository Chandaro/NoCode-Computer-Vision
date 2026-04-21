import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { Layers } from 'lucide-react'
import Projects from './pages/Projects'
import ProjectImages from './pages/ProjectImages'
import Annotate from './pages/Annotate'
import Train from './pages/Train'
import Analytics from './pages/Analytics'
import Evaluation from './pages/Evaluation'
import Classification from './pages/Classification'

function Nav() {
  return (
    <header style={{ borderBottom: '1px solid var(--border)', background: 'var(--surface)' }}
      className="sticky top-0 z-40">
      <div className="flex items-center gap-6 px-6 h-12">
        <NavLink to="/" className="flex items-center gap-2 no-underline">
          <Layers size={16} style={{ color: 'var(--accent)' }} strokeWidth={1.8} />
          <span style={{ color: 'var(--text)', fontWeight: 600, fontSize: 14, letterSpacing: '-0.01em' }}>
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
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
