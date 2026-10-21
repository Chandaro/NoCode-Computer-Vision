import { StrictMode, Component } from 'react'
import type { ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource/gloock/400.css'
import '@fontsource-variable/schibsted-grotesk/index.css'
import '@fontsource-variable/jetbrains-mono/index.css'
import './index.css'
import App from './App.tsx'

// ── Global error boundary – shows error details instead of blank page ──────────
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state: { error: Error | null } = { error: null }
  static getDerivedStateFromError(error: Error) { return { error } }
  render() {
    const { error } = this.state
    if (!error) return this.props.children
    return (
      <div style={{
        padding: 40, fontFamily: 'JetBrains Mono, monospace', color: '#f06a5e',
        background: '#071315', minHeight: '100vh',
      }}>
        <h2 style={{ color: '#ecf1e7', marginBottom: 16 }}>⚠ App failed to render</h2>
        <p style={{ marginBottom: 8, color: '#e4b43f' }}>{error.message}</p>
        <pre style={{
          background: '#0c1d20', padding: 16, borderRadius: 8,
          color: '#a3b9b3', fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-all',
        }}>{error.stack}</pre>
        <button onClick={() => window.location.reload()}
          style={{ marginTop: 24, padding: '8px 20px', background: '#e4b43f',
            color: '#1a1405', border: 'none', borderRadius: 999, cursor: 'pointer', fontSize: 14 }}>
          Reload
        </button>
      </div>
    )
  }
}

// ── Catch JS errors before React even mounts ───────────────────────────────────
window.addEventListener('error', (e) => {
  const root = document.getElementById('root')
  if (root && !root.hasChildNodes()) {
    root.innerHTML = `
      <div style="padding:40px;font-family:monospace;background:#071315;color:#f06a5e;min-height:100vh">
        <h2 style="color:#ecf1e7;margin-bottom:16px">⚠ Failed to load app</h2>
        <p style="color:#e4b43f;margin-bottom:8px">${e.message}</p>
        <p style="color:#a3b9b3;font-size:12px">${e.filename} line ${e.lineno}</p>
        <button onclick="location.reload()"
          style="margin-top:24px;padding:8px 20px;background:#e4b43f;color:#1a1405;
                 border:none;border-radius:6px;cursor:pointer;font-size:14px">
          Reload
        </button>
      </div>`
  }
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
