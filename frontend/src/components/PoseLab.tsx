/**
 * PoseLab — educational analysis panel for one detected person.
 * Shows recognised gestures, joint angles, a keypoint data inspector,
 * and lets students export the data (JSON / CSV) to use elsewhere.
 */
import { useState } from 'react'
import { ChevronDown, ChevronUp, Download, Activity, Tag, Table, Scale } from 'lucide-react'
import {
  COCO_KEYPOINTS, computeAngles, detectGestures, computeSymmetry,
  buildExport, keypointsToCSV,
} from '../lib/poseAnalysis'

interface Props {
  keypoints: number[][]
  kpConf:    number[]
}

const download = (text: string, filename: string, mime: string) => {
  const blob = new Blob([text], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

export default function PoseLab({ keypoints, kpConf }: Props) {
  const [showInspector, setShowInspector] = useState(false)
  const angles   = computeAngles(keypoints, kpConf)
  const gestures = detectGestures(keypoints, kpConf)
  const symmetry = computeSymmetry(keypoints, kpConf)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, marginTop: 12 }}>

      {/* ── Gestures ── */}
      <div style={{ background: 'var(--surface2)', borderRadius: 8, padding: 12 }}>
        <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
          textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8,
          display: 'flex', alignItems: 'center', gap: 6 }}>
          <Tag size={11} /> Detected Pose / Gesture
        </p>
        {gestures.length ? (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
            {gestures.map(g => (
              <span key={g} style={{ fontSize: 13, padding: '4px 10px', borderRadius: 6,
                background: 'rgba(94,106,210,0.15)', border: '1px solid rgba(94,106,210,0.3)',
                color: 'var(--text)' }}>{g}</span>
            ))}
          </div>
        ) : (
          <p style={{ fontSize: 12, color: 'var(--text3)' }}>No distinct gesture recognised</p>
        )}
      </div>

      {/* ── Joint angles ── */}
      <div style={{ background: 'var(--surface2)', borderRadius: 8, padding: 12 }}>
        <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
          textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8,
          display: 'flex', alignItems: 'center', gap: 6 }}>
          <Activity size={11} /> Joint Angles
        </p>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(110px, 1fr))', gap: 6 }}>
          {angles.map(a => (
            <div key={a.name} style={{ display: 'flex', justifyContent: 'space-between',
              alignItems: 'baseline', padding: '5px 8px', borderRadius: 5,
              background: 'var(--surface)', opacity: a.valid ? 1 : 0.4 }}>
              <span style={{ fontSize: 11, color: 'var(--text3)' }}>{a.name}</span>
              <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--accent)',
                fontFamily: 'JetBrains Mono, monospace' }}>
                {a.valid ? `${Math.round(a.angle)}°` : '—'}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* ── Left / right symmetry ── */}
      {symmetry.rows.length > 0 && (() => {
        const sc = symmetry.score
        const col = sc >= 85 ? '#22c55e' : sc >= 70 ? '#f59e0b' : '#ef4444'
        return (
          <div style={{ background: 'var(--surface2)', borderRadius: 8, padding: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <p style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
                textTransform: 'uppercase', letterSpacing: '0.07em',
                display: 'flex', alignItems: 'center', gap: 6, margin: 0 }}>
                <Scale size={11} /> Left / Right Symmetry
              </p>
              <span style={{ fontSize: 13, fontWeight: 700, color: col,
                fontFamily: 'JetBrains Mono, monospace' }}>{sc}/100</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
              {symmetry.rows.map(r => (
                <div key={r.name} style={{ display: 'flex', alignItems: 'center', gap: 8,
                  padding: '5px 8px', borderRadius: 5, background: 'var(--surface)' }}>
                  <span style={{ width: 7, height: 7, borderRadius: '50%', flexShrink: 0,
                    background: r.balanced ? '#22c55e' : '#f59e0b' }} />
                  <span style={{ fontSize: 11, color: 'var(--text2)', flex: 1 }}>{r.name}</span>
                  <span style={{ fontSize: 11, color: 'var(--text3)', fontFamily: 'JetBrains Mono, monospace' }}>
                    {Math.round(r.left)}° / {Math.round(r.right)}°
                  </span>
                  <span style={{ fontSize: 11, fontWeight: 600, minWidth: 38, textAlign: 'right',
                    fontFamily: 'JetBrains Mono, monospace',
                    color: r.balanced ? 'var(--text3)' : '#f59e0b' }}>
                    Δ{Math.round(r.diff)}°
                  </span>
                </div>
              ))}
            </div>
            <p style={{ fontSize: 10, color: 'var(--text3)', marginTop: 8, lineHeight: 1.4 }}>
              Compares matching left/right joint angles. A large Δ (amber dot) means that side is
              bent differently — useful for spotting form imbalance. Side-on angles can read
              uneven; face the camera for the cleanest read.
            </p>
          </div>
        )
      })()}

      {/* ── Keypoint inspector ── */}
      <div style={{ background: 'var(--surface2)', borderRadius: 8, overflow: 'hidden' }}>
        <button onClick={() => setShowInspector(v => !v)}
          style={{ width: '100%', display: 'flex', justifyContent: 'space-between',
            alignItems: 'center', padding: '10px 12px', background: 'transparent',
            border: 'none', cursor: 'pointer', color: 'var(--text2)' }}>
          <span style={{ fontSize: 10, fontWeight: 600, color: 'var(--text3)',
            textTransform: 'uppercase', letterSpacing: '0.07em',
            display: 'flex', alignItems: 'center', gap: 6 }}>
            <Table size={11} /> Keypoint Inspector (17 COCO points)
          </span>
          {showInspector ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>
        {showInspector && (
          <div style={{ padding: '0 12px 12px', maxHeight: 280, overflowY: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11,
              fontFamily: 'JetBrains Mono, monospace' }}>
              <thead>
                <tr style={{ color: 'var(--text3)', textAlign: 'left' }}>
                  <th style={{ padding: '4px 6px', position: 'sticky', top: 0, background: 'var(--surface2)' }}>#</th>
                  <th style={{ padding: '4px 6px', position: 'sticky', top: 0, background: 'var(--surface2)' }}>name</th>
                  <th style={{ padding: '4px 6px', position: 'sticky', top: 0, background: 'var(--surface2)' }}>x</th>
                  <th style={{ padding: '4px 6px', position: 'sticky', top: 0, background: 'var(--surface2)' }}>y</th>
                  <th style={{ padding: '4px 6px', position: 'sticky', top: 0, background: 'var(--surface2)' }}>conf</th>
                </tr>
              </thead>
              <tbody>
                {COCO_KEYPOINTS.map((name, i) => {
                  const c = kpConf[i] ?? 0
                  return (
                    <tr key={i} style={{ borderTop: '1px solid var(--border)',
                      color: c >= 0.5 ? 'var(--text2)' : 'var(--text3)' }}>
                      <td style={{ padding: '3px 6px' }}>{i}</td>
                      <td style={{ padding: '3px 6px', color: 'var(--text2)' }}>{name}</td>
                      <td style={{ padding: '3px 6px' }}>{keypoints[i]?.[0]?.toFixed(3) ?? '—'}</td>
                      <td style={{ padding: '3px 6px' }}>{keypoints[i]?.[1]?.toFixed(3) ?? '—'}</td>
                      <td style={{ padding: '3px 6px',
                        color: c >= 0.5 ? 'var(--success)' : c >= 0.3 ? 'var(--warn)' : 'var(--text3)' }}>
                        {c.toFixed(2)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Export ── */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button onClick={() => download(
          JSON.stringify(buildExport(keypoints, kpConf), null, 2),
          'pose_data.json', 'application/json')}
          style={btnStyle}>
          <Download size={12} /> Export JSON
        </button>
        <button onClick={() => download(
          keypointsToCSV(keypoints, kpConf), 'pose_keypoints.csv', 'text/csv')}
          style={btnStyle}>
          <Download size={12} /> Export CSV
        </button>
        <span style={{ fontSize: 11, color: 'var(--text3)', alignSelf: 'center' }}>
          Use the data in Excel, Python, or your own project
        </span>
      </div>
    </div>
  )
}

const btnStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', gap: 5, padding: '6px 12px',
  borderRadius: 6, border: '1px solid var(--border)', background: 'var(--surface2)',
  color: 'var(--text2)', fontSize: 12, cursor: 'pointer', fontFamily: 'inherit',
}
