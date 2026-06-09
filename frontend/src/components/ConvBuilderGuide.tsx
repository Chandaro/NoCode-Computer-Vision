/**
 * ConvBuilderGuide: a plain-language guide that explains what a CNN is and how
 * to build one on this page. Written for students with no prior knowledge.
 * Opens as an animated popup (same style as the Pose guide).
 */
import { useEffect, useState } from 'react'
import { BookOpen, X, ArrowRight } from 'lucide-react'

const LS_KEY = 'conv_guide_seen'

// Stable example photos (Unsplash CDN). onError hides them so a broken link
// never shows a broken-image icon.
const DOG_PHOTO = 'https://images.unsplash.com/photo-1543466835-00a7907e9de1?w=400&q=80'
const CAT_PHOTO = 'https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&q=80'

// ── CNN pipeline diagram (input image → conv → pool → flatten → output) ───────
function PipelineDiagram() {
  const [photoOk, setPhotoOk] = useState(true)
  const stages = [
    { label: 'Conv',    sub: 'find features', c: '#a78bfa' },
    { label: 'Pool',    sub: 'shrink',        c: '#38bdf8' },
    { label: 'Conv',    sub: 'find more',     c: '#a78bfa' },
    { label: 'Pool',    sub: 'shrink',        c: '#38bdf8' },
    { label: 'Flatten', sub: 'to a list',     c: '#fb923c' },
    { label: 'Output',  sub: 'cat or dog?',   c: '#3fb950' },
  ]
  return (
    <div style={{
      background: 'var(--surface2)', border: '1px solid var(--border)',
      borderRadius: 12, padding: 18,
    }}>
      <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: '0 0 14px' }}>
        A picture goes in on the left. Each layer changes it a little, pulling out more and
        more useful information, until the last layer makes a decision such as <strong>cat</strong>
        {' '}or <strong>dog</strong>.
      </p>

      <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
        {/* Input photo */}
        {photoOk && (
          <div style={{ textAlign: 'center', flexShrink: 0 }}>
            <img src={DOG_PHOTO} alt="input" onError={() => setPhotoOk(false)}
              style={{ width: 84, height: 84, objectFit: 'cover', borderRadius: 8,
                border: '1px solid var(--border2)', display: 'block' }} />
            <span style={{ fontSize: 10, color: 'var(--text3)' }}>input image</span>
          </div>
        )}
        <ArrowRight size={16} style={{ color: 'var(--text3)', flexShrink: 0 }} />

        {/* Stages */}
        {stages.map((s, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ textAlign: 'center', flexShrink: 0 }}>
              <div style={{
                width: 54, height: 54, borderRadius: 8,
                background: `${s.c}22`, border: `1.5px solid ${s.c}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 11, fontWeight: 700, color: s.c,
              }}>{s.label}</div>
              <span style={{ fontSize: 9.5, color: 'var(--text3)', display: 'block', marginTop: 2 }}>{s.sub}</span>
            </div>
            {i < stages.length - 1 && <ArrowRight size={14} style={{ color: 'var(--text3)', flexShrink: 0 }} />}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Small building blocks ─────────────────────────────────────────────────────
const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <p style={{
    fontSize: 11, fontWeight: 600, color: 'var(--text3)',
    textTransform: 'uppercase', letterSpacing: '0.08em', margin: '22px 0 10px',
  }}>{children}</p>
)

const Step = ({ n, children }: { n: number; children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 11, marginBottom: 9 }}>
    <span style={{
      flexShrink: 0, width: 20, height: 20, borderRadius: '50%',
      background: 'var(--accent-s)', color: 'var(--accent)', fontSize: 11, fontWeight: 700,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
      fontFamily: 'JetBrains Mono, monospace',
    }}>{n}</span>
    <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.55, margin: 0 }}>{children}</p>
  </div>
)

// One layer entry: a coloured chip + name + clear description
const LayerRow = ({ name, color, children }:
  { name: string; color: string; children: React.ReactNode }) => (
  <div style={{ display: 'flex', gap: 11, marginBottom: 11 }}>
    <span style={{
      flexShrink: 0, marginTop: 1, padding: '2px 8px', borderRadius: 5, height: 'fit-content',
      fontSize: 10, fontWeight: 700, color, background: `${color}22`, border: `1px solid ${color}55`,
      fontFamily: 'JetBrains Mono, monospace', minWidth: 64, textAlign: 'center',
    }}>{name}</span>
    <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.6, margin: 0 }}>{children}</p>
  </div>
)

const Term = ({ children }: { children: React.ReactNode }) => (
  <span style={{ color: 'var(--text)', fontWeight: 500 }}>{children}</span>
)

// ── Guide content ─────────────────────────────────────────────────────────────
function GuideContent() {
  const [catOk, setCatOk] = useState(true)
  return (
    <div style={{ padding: '4px 24px 24px' }}>

      {/* Intro */}
      <p style={{ fontSize: 13.5, color: 'var(--text2)', lineHeight: 1.7, margin: '16px 0 0' }}>
        A <Term>Convolutional Neural Network</Term> (CNN) is the kind of AI that looks at
        images and learns to tell them apart, for example a <Term>cat</Term> from a{' '}
        <Term>dog</Term>. On this page you build one yourself by stacking layers like building
        blocks, watch your image flow through it in 3D, and then train it on your own pictures.
        No code is needed.
      </p>

      <div style={{ display: 'flex', gap: 14, alignItems: 'center', margin: '14px 0' }}>
        {catOk && (
          <img src={CAT_PHOTO} alt="cat" onError={() => setCatOk(false)}
            style={{ width: 96, height: 96, objectFit: 'cover', borderRadius: 10,
              border: '1px solid var(--border2)', flexShrink: 0 }} />
        )}
        <p style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.7, margin: 0 }}>
          The network never sees the word &ldquo;cat&rdquo;. It only sees numbers (the pixel
          colours). Its job is to learn which patterns of numbers usually mean cat and which
          usually mean dog, all on its own, by looking at many labelled examples.
        </p>
      </div>

      {/* Pipeline */}
      <PipelineDiagram />

      {/* The layers */}
      <SectionLabel>The layers, and what each one does</SectionLabel>
      <LayerRow name="Conv 2D" color="#a78bfa">
        The most important layer. It slides small <Term>filters</Term> across the image to find
        patterns. Early conv layers find simple things like edges and colours. Later ones
        combine those into shapes, then whole objects like an ear or an eye.
      </LayerRow>
      <LayerRow name="ReLU" color="#fb923c">
        A simple rule that keeps the useful (positive) signals and sets everything else to
        zero. It lets the network learn complicated patterns instead of only straight lines.
      </LayerRow>
      <LayerRow name="Max Pool" color="#38bdf8">
        Shrinks the image by keeping only the strongest signal in each small area. This makes
        the network faster and helps it care about <Term>what</Term> is in the image rather
        than exactly <Term>where</Term>.
      </LayerRow>
      <LayerRow name="Batch Norm" color="#22d3ee">
        Keeps the numbers flowing through the network steady and balanced, which makes training
        faster and more reliable. A helper layer you usually place right after a Conv.
      </LayerRow>
      <LayerRow name="Dropout" color="#94a3b8">
        During training it randomly switches off some connections. This stops the network from
        simply memorising the training photos and helps it work on new, unseen images.
      </LayerRow>
      <LayerRow name="Flatten" color="#fb923c">
        Turns the 2D picture of features into one long list of numbers, so the final decision
        layers can read it.
      </LayerRow>
      <LayerRow name="Linear" color="#5e6ad2">
        Also called a dense layer. Every number connects to every output, which lets the
        network weigh up all the features and move toward a final answer.
      </LayerRow>
      <LayerRow name="Output" color="#3fb950">
        The last layer. It gives one score per class. The highest score is the network's
        guess, for example 0.92 cat and 0.08 dog.
      </LayerRow>

      {/* How to build */}
      <SectionLabel>How to build a network</SectionLabel>
      <Step n={1}>
        Start with a <Term>preset</Term> (Minimal, LeNet, VGG-mini) to get a working network,
        or build your own from the input down.
      </Step>
      <Step n={2}>
        Repeat the common pattern: <Term>Conv → ReLU → Max Pool</Term>. Each repeat lets the
        network find more detailed features while shrinking the image.
      </Step>
      <Step n={3}>
        Finish with <Term>Flatten → Linear → Output</Term>. The output classes are filled in
        automatically from your dataset.
      </Step>
      <Step n={4}>
        Watch the layer list. If a layer shows <Term>⚠ bad</Term>, the image has shrunk to
        nothing. Remove a pooling layer or make the input bigger.
      </Step>
      <Step n={5}>
        Set the training options on the right (epochs, batch, learning rate), then press{' '}
        <Term>Train</Term>.
      </Step>

      {/* Reading the 3D view */}
      <SectionLabel>Reading the 3D view</SectionLabel>
      <ul style={{ margin: 0, paddingLeft: 18, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {[
          'Each block is one layer. The image flows from left (input) to right (output).',
          'Turn on Activations to see what each layer actually produces from a real image.',
          'Shuffle Image loads a different picture from your dataset so you can compare.',
          'Drag to orbit, scroll to zoom, or use Fly Mode to move through the network.',
          'The numbers under each block (like 16×64×64) are channels × height × width.',
        ].map((t, i) => (
          <li key={i} style={{ fontSize: 13, color: 'var(--text2)', lineHeight: 1.55 }}>{t}</li>
        ))}
      </ul>

      {/* Tips */}
      <SectionLabel>Tips</SectionLabel>
      <div style={{
        background: 'var(--surface2)', border: '1px solid var(--border)',
        borderRadius: 8, padding: '12px 16px',
      }}>
        {[
          ['Use the Conv, ReLU, Pool pattern', 'It is the proven building block of almost every image CNN.'],
          ['Do not stack too many pools', 'Each one halves the image. Too many and it disappears.'],
          ['More filters find more features', 'But they also use more memory and train slower.'],
          ['Bigger input keeps more detail', 'A 128×128 input sees more than 32×32, but costs more memory.'],
          ['Give each class enough images', 'Aim for at least 20 per class so the network has plenty to learn from.'],
        ].map(([t, d]) => (
          <div key={t} style={{ display: 'flex', gap: 8, marginBottom: 6 }}>
            <span style={{ color: 'var(--success)', flexShrink: 0, fontSize: 13 }}>•</span>
            <p style={{ fontSize: 12.5, color: 'var(--text2)', lineHeight: 1.55, margin: 0 }}>
              <Term>{t}.</Term> {d}
            </p>
          </div>
        ))}
      </div>

      <p style={{ fontSize: 11.5, color: 'var(--text3)', lineHeight: 1.6, margin: '16px 0 0' }}>
        Note: this page uses the same images as the Image Classification page. Upload pictures
        for each class there first, then come back to build and train your CNN.
      </p>
    </div>
  )
}

// ── Trigger banner + animated popup modal ────────────────────────────────────
export default function ConvBuilderGuide() {
  const [mounted, setMounted] = useState(false)
  const [closing, setClosing] = useState(false)

  const open = () => { setClosing(false); setMounted(true) }
  const close = () => {
    setClosing(true)
    setTimeout(() => { setMounted(false); setClosing(false) }, 170)
  }

  useEffect(() => {
    if (localStorage.getItem(LS_KEY) !== 'true') {
      localStorage.setItem(LS_KEY, 'true')
      const t = setTimeout(open, 350)
      return () => clearTimeout(t)
    }
  }, [])

  useEffect(() => {
    if (!mounted) return
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close() }
    document.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.removeEventListener('keydown', onKey); document.body.style.overflow = prev }
  }, [mounted])

  return (
    <>
      {/* Trigger banner */}
      <button onClick={open} style={{
        width: '100%', display: 'flex', alignItems: 'center', gap: 11,
        padding: '13px 18px', marginBottom: 16, textAlign: 'left',
        background: 'var(--surface)', border: '1px solid var(--border)',
        borderRadius: 10, cursor: 'pointer', transition: 'border-color 0.15s',
      }}
        onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--accent)' }}
        onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border)' }}>
        <span style={{
          width: 32, height: 32, borderRadius: 8, flexShrink: 0,
          background: 'var(--accent-s)', display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <BookOpen size={16} style={{ color: 'var(--accent)' }} />
        </span>
        <span style={{ flex: 1 }}>
          <span style={{ display: 'block', fontSize: 14, fontWeight: 600, color: 'var(--text)' }}>
            What is a CNN, and how do I build one here?
          </span>
          <span style={{ display: 'block', fontSize: 12, color: 'var(--text3)', marginTop: 1 }}>
            New to neural networks? Open the quick guide.
          </span>
        </span>
        <span style={{ display: 'flex', alignItems: 'center', gap: 5, flexShrink: 0,
          fontSize: 12, fontWeight: 500, color: 'var(--accent)' }}>
          Open guide <ArrowRight size={14} />
        </span>
      </button>

      {/* Modal */}
      {mounted && (
        <div onClick={close}
          className={closing ? 'guide-backdrop-out' : 'guide-backdrop-in'}
          style={{
            position: 'fixed', inset: 0, zIndex: 200,
            background: 'rgba(8, 10, 14, 0.62)', backdropFilter: 'blur(3px)',
            display: 'flex', alignItems: 'flex-start', justifyContent: 'center',
            padding: '6vh 16px 16px', overflowY: 'auto',
          }}>
          <div onClick={e => e.stopPropagation()}
            className={closing ? 'guide-card-out' : 'guide-card-in'}
            style={{
              width: '100%', maxWidth: 780, background: 'var(--surface)',
              border: '1px solid var(--border2)', borderRadius: 16,
              boxShadow: '0 24px 64px rgba(0,0,0,0.5)', overflow: 'hidden',
              maxHeight: '88vh', display: 'flex', flexDirection: 'column',
            }}>
            {/* Header */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 11,
              padding: '16px 22px', borderBottom: '1px solid var(--border)',
              flexShrink: 0, background: 'var(--surface)',
            }}>
              <BookOpen size={17} style={{ color: 'var(--accent)', flexShrink: 0 }} />
              <div style={{ flex: 1 }}>
                <p style={{ fontSize: 15, fontWeight: 600, color: 'var(--text)', margin: 0 }}>
                  Conv Builder Guide
                </p>
                <p style={{ fontSize: 12, color: 'var(--text3)', margin: '1px 0 0' }}>
                  How CNNs work, and how to build one here, explained simply.
                </p>
              </div>
              <button onClick={close} title="Close (Esc)" style={{
                width: 30, height: 30, borderRadius: 7, border: '1px solid var(--border)',
                background: 'var(--surface2)', color: 'var(--text3)', cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}
                onMouseEnter={e => { e.currentTarget.style.color = 'var(--text)' }}
                onMouseLeave={e => { e.currentTarget.style.color = 'var(--text3)' }}>
                <X size={16} />
              </button>
            </div>

            {/* Scrollable body */}
            <div style={{ overflowY: 'auto', flex: 1 }}>
              <GuideContent />
            </div>
          </div>
        </div>
      )}
    </>
  )
}
