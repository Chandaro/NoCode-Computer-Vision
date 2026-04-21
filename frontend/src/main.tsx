import { StrictMode, Component } from 'react'
import type { ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
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
        padding: 40, fontFamily: 'JetBrains Mono, monospace', color: '#ed4245',
        background: '#0d0d0f', minHeight: '100vh',
      }}>
        <h2 style={{ color: '#e2e2e2', marginBottom: 16 }}>⚠ App failed to render</h2>
        <p style={{ marginBottom: 8, color: '#faa61a' }}>{error.message}</p>
        <pre style={{
          background: '#16161a', padding: 16, borderRadius: 8,
          color: '#8b8b9a', fontSize: 12, whiteSpace: 'pre-wrap', wordBreak: 'break-all',
        }}>{error.stack}</pre>
        <button onClick={() => window.location.reload()}
          style={{ marginTop: 24, padding: '8px 20px', background: '#5865f2',
            color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer', fontSize: 14 }}>
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
      <div style="padding:40px;font-family:monospace;background:#0d0d0f;color:#ed4245;min-height:100vh">
        <h2 style="color:#e2e2e2;margin-bottom:16px">⚠ Failed to load app</h2>
        <p style="color:#faa61a;margin-bottom:8px">${e.message}</p>
        <p style="color:#8b8b9a;font-size:12px">${e.filename} line ${e.lineno}</p>
        <button onclick="location.reload()"
          style="margin-top:24px;padding:8px 20px;background:#5865f2;color:#fff;
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
