import { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Loader } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  ScatterChart, Scatter, PieChart, Pie, Cell,
} from 'recharts'
import api, { type AnalyticsData, type Project } from '../api'
import { Card, StatTile, PageHeader, Label } from '../components/ui'

const PALETTE = ['#5865f2','#22c55e','#f59e0b','#06b6d4','#a855f7','#ec4899','#64748b','#ef4444']
const TT = { contentStyle: { background: 'var(--surface2)', border: '1px solid var(--border)', borderRadius: 6, fontSize: 11 } }

export default function Analytics() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  const [project, setProject] = useState<Project | null>(null)
  const [data, setData]       = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      api.get(`/projects/${projectId}`),
      api.get(`/projects/${projectId}/analytics`),
    ]).then(([p, a]) => { setProject(p.data); setData(a.data) })
      .finally(() => setLoading(false))
  }, [projectId])

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 260, gap: 8, color: 'var(--text2)' }}>
      <Loader size={16} className="animate-spin" /> Computing analytics…
    </div>
  )
  if (!data) return null

  const classDist  = Object.entries(data.class_distribution).map(([name, count]) => ({ name, count }))
  const shapeDist  = [
    { name: 'BBox',    value: data.shape_breakdown.bbox    },
    { name: 'Polygon', value: data.shape_breakdown.polygon },
    { name: 'Point',   value: data.shape_breakdown.point   },
  ].filter(s => s.value > 0)
  const annHist    = Object.entries(data.ann_histogram).map(([range, count]) => ({ range, count }))
  const aspectDist = Object.entries(data.aspect_buckets).map(([label, count]) => ({ label, count }))

  return (
    <div style={{ maxWidth: 1080, margin: '0 auto' }}>
      <PageHeader back={() => navigate(`/projects/${projectId}/images`)}
        title="Dataset Analytics" subtitle={project?.name} />

      {/* Overview */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 10, marginBottom: 20 }}>
        <StatTile label="Total Images"      value={data.total_images} />
        <StatTile label="Annotated"         value={data.annotated_images}
          sub={`${Math.round(data.annotated_images / Math.max(data.total_images,1)*100)}%`} />
        <StatTile label="Annotations"       value={data.total_annotations} />
        <StatTile label="Corrupt"           value={data.corrupt_images} />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>

        {/* Class distribution */}
        <Card style={{ padding: 16 }}>
          <Label>Class Distribution</Label>
          {classDist.length === 0
            ? <p style={{ color: 'var(--text3)', fontSize: 12 }}>No annotations yet.</p>
            : <ResponsiveContainer width="100%" height={190}>
                <BarChart data={classDist} margin={{ left: -16 }}>
                  <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
                  <XAxis dataKey="name" stroke="var(--text3)" tick={{ fontSize: 11 }} />
                  <YAxis stroke="var(--text3)" tick={{ fontSize: 11 }} />
                  <Tooltip {...TT} />
                  <Bar dataKey="count" radius={[3,3,0,0]}>
                    {classDist.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>}
        </Card>

        {/* Shape types */}
        <Card style={{ padding: 16 }}>
          <Label>Annotation Types</Label>
          {shapeDist.length === 0
            ? <p style={{ color: 'var(--text3)', fontSize: 12 }}>No annotations yet.</p>
            : <ResponsiveContainer width="100%" height={190}>
                <PieChart>
                  <Pie data={shapeDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={72}
                    label={({ name, percent }: { name?: string; percent?: number }) => `${name ?? ''} ${Math.round((percent ?? 0)*100)}%`}
                    labelLine={{ stroke: 'var(--text3)', strokeWidth: 0.5 }}>
                    {shapeDist.map((_, i) => <Cell key={i} fill={PALETTE[i % PALETTE.length]} />)}
                  </Pie>
                  <Tooltip {...TT} />
                </PieChart>
              </ResponsiveContainer>}
        </Card>

        {/* Annotations per image */}
        <Card style={{ padding: 16 }}>
          <Label>Annotations per Image</Label>
          <ResponsiveContainer width="100%" height={170}>
            <BarChart data={annHist} margin={{ left: -16 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
              <XAxis dataKey="range" stroke="var(--text3)" tick={{ fontSize: 11 }} />
              <YAxis stroke="var(--text3)" tick={{ fontSize: 11 }} />
              <Tooltip {...TT} />
              <Bar dataKey="count" fill={PALETTE[0]} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Aspect ratio */}
        <Card style={{ padding: 16 }}>
          <Label>Aspect Ratio Distribution</Label>
          <ResponsiveContainer width="100%" height={170}>
            <BarChart data={aspectDist} margin={{ left: -16 }}>
              <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
              <XAxis dataKey="label" stroke="var(--text3)" tick={{ fontSize: 10 }} />
              <YAxis stroke="var(--text3)" tick={{ fontSize: 11 }} />
              <Tooltip {...TT} />
              <Bar dataKey="count" fill={PALETTE[1]} radius={[3,3,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        {/* Image size scatter */}
        {data.size_samples.length > 0 && (
          <Card style={{ padding: 16 }}>
            <Label>Image Dimensions</Label>
            <ResponsiveContainer width="100%" height={190}>
              <ScatterChart margin={{ left: -16 }}>
                <CartesianGrid strokeDasharray="2 2" stroke="var(--border)" />
                <XAxis dataKey="w" name="Width"  stroke="var(--text3)" tick={{ fontSize: 10 }} />
                <YAxis dataKey="h" name="Height" stroke="var(--text3)" tick={{ fontSize: 10 }} />
                <Tooltip {...TT} formatter={(v: unknown) => [`${v}px`]} />
                <Scatter data={data.size_samples} fill={PALETTE[2]} opacity={0.5} />
              </ScatterChart>
            </ResponsiveContainer>
          </Card>
        )}

        {/* Channel stats */}
        {data.channel_stats && (
          <Card style={{ padding: 16 }}>
            <Label>Channel Statistics</Label>
            <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 8 }}>
              <thead>
                <tr>
                  {['Channel','Mean','Std Dev'].map(h => (
                    <th key={h} style={{ fontSize: 11, fontWeight: 500, color: 'var(--text3)', textAlign: h === 'Channel' ? 'left' : 'right',
                      paddingBottom: 8, letterSpacing: '0.04em', textTransform: 'uppercase' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {(['R','G','B'] as const).map((ch, i) => (
                  <tr key={ch} style={{ borderTop: '1px solid var(--border)' }}>
                    <td style={{ padding: '8px 0', display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2,
                        background: ['#ef4444','#22c55e','#3b82f6'][i], flexShrink: 0 }} />
                      <span style={{ fontSize: 13, color: 'var(--text2)', fontWeight: 500 }}>{ch}</span>
                    </td>
                    <td style={{ padding: '8px 0', textAlign: 'right', fontSize: 13,
                      fontFamily: 'JetBrains Mono, monospace', color: 'var(--text)' }}>
                      {data.channel_stats!.mean[ch].toFixed(4)}
                    </td>
                    <td style={{ padding: '8px 0', textAlign: 'right', fontSize: 13,
                      fontFamily: 'JetBrains Mono, monospace', color: 'var(--text2)' }}>
                      {data.channel_stats!.std[ch].toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 10 }}>Sampled from up to 100 images · values in [0, 1]</p>
          </Card>
        )}
      </div>

      {/* Color space */}
      <Card style={{ padding: 16 }}>
        <Label>Color Space</Label>
        <div style={{ display: 'flex', gap: 10, marginTop: 4 }}>
          {Object.entries(data.color_space_counts).map(([cs, n]) => (
            <div key={cs} style={{ background: 'var(--surface2)', border: '1px solid var(--border)',
              borderRadius: 6, padding: '8px 14px', textAlign: 'center' }}>
              <p style={{ fontSize: 18, fontWeight: 600, color: 'var(--text)' }}>{n}</p>
              <p style={{ fontSize: 11, color: 'var(--text3)', marginTop: 2 }}>{cs}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
