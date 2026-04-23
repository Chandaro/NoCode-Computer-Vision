import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import * as THREE from 'three'
import api from '../api'
import { PageHeader, Btn, Badge } from '../components/ui'
import { Cpu } from 'lucide-react'

// ── Types ─────────────────────────────────────────────────────────────────────

type LayerType =
  | 'conv2d' | 'batchnorm2d' | 'maxpool2d' | 'avgpool2d'
  | 'relu' | 'gelu' | 'sigmoid' | 'dropout' | 'flatten' | 'linear'

interface Layer {
  id: string
  type: LayerType
  params: Record<string, number>
}

interface CustomConfig {
  id: number
  project_id: number
  name: string
  layers: Layer[]
  input_h: number
  input_w: number
  created_at: string
}

interface CustomRun {
  id: number
  config_id: number
  project_id: number
  status: string
  epochs: number
  batch: number
  lr: number
  model_path: string
  results: Record<string, unknown>
  created_at: string
}

// ── Constants ─────────────────────────────────────────────────────────────────

const LAYER_COLORS: Record<string, number> = {
  input:       0x8b8b9a,
  conv2d:      0x5865f2,
  batchnorm2d: 0x22c55e,
  maxpool2d:   0xf97316,
  avgpool2d:   0xfb923c,
  relu:        0xef4444,
  gelu:        0xf43f5e,
  sigmoid:     0xec4899,
  dropout:     0xeab308,
  flatten:     0xa855f7,
  linear:      0x06b6d4,
}

const LAYER_CSS: Record<string, string> = {
  input:       '#8b8b9a',
  conv2d:      '#5865f2',
  batchnorm2d: '#22c55e',
  maxpool2d:   '#f97316',
  avgpool2d:   '#fb923c',
  relu:        '#ef4444',
  gelu:        '#f43f5e',
  sigmoid:     '#ec4899',
  dropout:     '#eab308',
  flatten:     '#a855f7',
  linear:      '#06b6d4',
}

const ALL_LAYER_TYPES: LayerType[] = [
  'conv2d', 'batchnorm2d', 'maxpool2d', 'avgpool2d',
  'relu', 'gelu', 'sigmoid', 'dropout', 'flatten', 'linear',
]

const DEFAULT_PARAMS: Record<LayerType, Record<string, number>> = {
  conv2d:      { filters: 16, kernel_size: 3, stride: 1, padding: 1 },
  batchnorm2d: {},
  maxpool2d:   { kernel_size: 2, stride: 2 },
  avgpool2d:   { kernel_size: 2, stride: 2 },
  relu:        {},
  gelu:        {},
  sigmoid:     {},
  dropout:     { p: 0.5 },
  flatten:     {},
  linear:      { out_features: 128 },
}

const DEFAULT_LAYERS: Layer[] = [
  { id: 'l1', type: 'conv2d',   params: { filters: 16, kernel_size: 3, stride: 1, padding: 1 } },
  { id: 'l2', type: 'relu',     params: {} },
  { id: 'l3', type: 'maxpool2d',params: { kernel_size: 2, stride: 2 } },
  { id: 'l4', type: 'conv2d',   params: { filters: 32, kernel_size: 3, stride: 1, padding: 1 } },
  { id: 'l5', type: 'relu',     params: {} },
  { id: 'l6', type: 'maxpool2d',params: { kernel_size: 2, stride: 2 } },
  { id: 'l7', type: 'flatten',  params: {} },
  { id: 'l8', type: 'linear',   params: { out_features: 128 } },
]

const SPACING = 3.0

// ── Shape computation ─────────────────────────────────────────────────────────

function computeOutputShape(layer: Layer, shape: number[]): number[] {
  const [C, H = 1, W = 1] = shape
  switch (layer.type) {
    case 'conv2d': {
      const { filters = 32, kernel_size = 3, stride = 1, padding = 1 } = layer.params
      return [
        filters,
        Math.floor((H + 2 * padding - kernel_size) / stride + 1),
        Math.floor((W + 2 * padding - kernel_size) / stride + 1),
      ]
    }
    case 'maxpool2d':
    case 'avgpool2d': {
      const { kernel_size = 2 } = layer.params
      const s = layer.params.stride ?? kernel_size
      return [C, Math.max(1, Math.floor(H / s)), Math.max(1, Math.floor(W / s))]
    }
    case 'flatten':
      return [shape.reduce((a, b) => a * b, 1)]
    case 'linear':
      return [layer.params.out_features ?? 128]
    default:
      return shape
  }
}

function isValidShape(shape: number[]): boolean {
  return shape.every(d => d > 0 && isFinite(d))
}

function computeAllShapes(layers: Layer[], inputH: number, inputW: number) {
  const shapes: number[][] = []
  let shape: number[] = [3, inputH, inputW]
  let valid = true
  // Input shape
  shapes.push(shape)
  for (const layer of layers) {
    if (!valid) {
      shapes.push([-1])
      continue
    }
    const next = computeOutputShape(layer, shape)
    if (!isValidShape(next)) { valid = false; shapes.push([-1]); continue }
    shape = next
    shapes.push(shape)
  }
  return shapes
}

function estimateParams(layers: Layer[], inputH: number, inputW: number): number {
  let total = 0
  let shape: number[] = [3, inputH, inputW]
  let valid = true
  for (const layer of layers) {
    if (!valid) break
    const next = computeOutputShape(layer, shape)
    if (!isValidShape(next)) { valid = false; break }
    if (layer.type === 'conv2d') {
      const { filters = 32, kernel_size = 3 } = layer.params
      total += filters * shape[0] * kernel_size * kernel_size + filters
    } else if (layer.type === 'batchnorm2d') {
      total += shape[0] * 4
    } else if (layer.type === 'linear') {
      const inF = shape.reduce((a, b) => a * b, 1)
      const outF = layer.params.out_features ?? 128
      total += inF * outF + outF
    }
    shape = next
  }
  return total
}

function uid(): string {
  return Math.random().toString(36).slice(2, 9)
}

const LAYER_DISPLAY: Record<string, string> = {
  linear: 'linear (FC)',
}

const LAYER_DESCRIPTIONS: Record<string, { label: string; what: string; why: string; analogy: string }> = {
  conv2d: {
    label: 'Convolutional Layer',
    what: 'Slides a small learnable filter (kernel) over the image, computing dot products at each position to detect local patterns.',
    why: 'Each filter learns to detect one feature — edges, curves, textures. More filters = richer feature extraction.',
    analogy: '🔍 Like scanning a photo with a magnifying glass that lights up when it sees a specific shape.',
  },
  batchnorm2d: {
    label: 'Batch Normalization',
    what: 'Normalizes each batch of activations to have zero mean and unit variance, then scales and shifts with learned parameters.',
    why: 'Prevents exploding/vanishing gradients, allows higher learning rates, and speeds up training significantly.',
    analogy: '⚖️ Like a referee ensuring all players play at the same energy level before each round.',
  },
  maxpool2d: {
    label: 'Max Pooling',
    what: 'Divides the feature map into small windows and keeps only the maximum value from each window.',
    why: 'Reduces spatial size, cuts computation, and makes features robust to small shifts/translations.',
    analogy: '📸 Like thumbnail-ing an image — keeps the most prominent signal and throws away fine detail.',
  },
  avgpool2d: {
    label: 'Average Pooling',
    what: 'Like max pooling but takes the average of values in each window instead of the maximum.',
    why: 'Produces smoother feature maps. Often used in the final stage before classification.',
    analogy: '🌫️ Like blurring a photo — takes a soft summary of a region rather than the sharpest peak.',
  },
  relu: {
    label: 'ReLU Activation',
    what: 'Sets all negative values to zero, keeps positive values unchanged: f(x) = max(0, x).',
    why: 'Adds non-linearity so the network can learn complex patterns. Simple yet very effective.',
    analogy: '💡 Like a switch — neuron is "silent" when negative, "fires" when positive.',
  },
  gelu: {
    label: 'GELU Activation',
    what: 'A smooth, differentiable approximation of ReLU that slightly activates for small negative values.',
    why: 'Used in modern architectures (BERT, GPT). Tends to train slightly better than ReLU.',
    analogy: '🎚️ A smoother dimmer switch instead of a hard on/off toggle.',
  },
  sigmoid: {
    label: 'Sigmoid Activation',
    what: 'Squashes any input to a value between 0 and 1 using f(x) = 1 / (1 + e⁻ˣ).',
    why: 'Useful for binary classification outputs. Rarely used in hidden layers due to vanishing gradients.',
    analogy: '📊 Like a confidence meter that always reads between 0% and 100%.',
  },
  dropout: {
    label: 'Dropout',
    what: 'Randomly zeros out a fraction (p) of neurons during training. Disabled at inference time.',
    why: 'Prevents overfitting by forcing the network not to rely on any single neuron.',
    analogy: '🎲 Like randomly benching players during practice — the team learns to work without any one star.',
  },
  flatten: {
    label: 'Flatten',
    what: 'Reshapes a 3D tensor (Channels × Height × Width) into a single 1D vector.',
    why: 'Required to bridge spatial feature maps with fully connected layers.',
    analogy: '📄 Like unrolling a folded map into a single long strip.',
  },
  linear: {
    label: 'Fully Connected (Linear)',
    what: 'Every input neuron connects to every output neuron with a learned weight.',
    why: 'Combines all extracted features to make a final decision or produce class scores.',
    analogy: '🗳️ Like a committee vote — every piece of evidence contributes to the final answer.',
  },
}

// ── Three.js: create label sprite ─────────────────────────────────────────────

function makeLabel(text1: string, text2: string, color: string): THREE.Sprite {
  const W = 256, H = 72
  const canvas = document.createElement('canvas')
  canvas.width  = W
  canvas.height = H
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, W, H)
  ctx.font      = 'bold 22px "JetBrains Mono", monospace'
  ctx.fillStyle = color
  ctx.textAlign = 'center'
  ctx.fillText(text1, W / 2, 26)
  ctx.font      = '17px "JetBrains Mono", monospace'
  ctx.fillStyle = 'rgba(200,200,220,0.7)'
  ctx.fillText(text2, W / 2, 52)
  const tex    = new THREE.CanvasTexture(canvas)
  const mat    = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false })
  const sprite = new THREE.Sprite(mat)
  sprite.scale.set((W / H) * 0.9, 0.9, 1)
  return sprite
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t
}

// ── Real image processing pipeline for 2D Activations view ──────────────────

function createBaseImage(size: number): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = c.height = size
  const ctx = c.getContext('2d')!

  const sky = ctx.createLinearGradient(0, 0, 0, size * 0.45)
  sky.addColorStop(0, '#1a3a6e'); sky.addColorStop(1, '#4a90d9')
  ctx.fillStyle = sky; ctx.fillRect(0, 0, size, size * 0.45)

  const gnd = ctx.createLinearGradient(0, size * 0.45, 0, size)
  gnd.addColorStop(0, '#4a7c3f'); gnd.addColorStop(1, '#2d4a1e')
  ctx.fillStyle = gnd; ctx.fillRect(0, size * 0.45, size, size)

  ctx.fillStyle = '#c0784a'; ctx.fillRect(size * 0.3, size * 0.3, size * 0.4, size * 0.25)
  ctx.fillStyle = '#8b3a2a'
  ctx.beginPath(); ctx.moveTo(size * 0.25, size * 0.3)
  ctx.lineTo(size * 0.5, size * 0.08); ctx.lineTo(size * 0.75, size * 0.3)
  ctx.closePath(); ctx.fill()

  ctx.fillStyle = '#f0d080'
  ctx.fillRect(size * 0.38, size * 0.37, size * 0.09, size * 0.09)
  ctx.fillRect(size * 0.53, size * 0.37, size * 0.09, size * 0.09)
  ctx.fillStyle = '#4a2010'; ctx.fillRect(size * 0.44, size * 0.42, size * 0.12, size * 0.13)

  ctx.fillStyle = '#ffe066'
  ctx.beginPath(); ctx.arc(size * 0.12, size * 0.12, size * 0.06, 0, Math.PI * 2); ctx.fill()
  return c
}

function cloneCanvas(src: HTMLCanvasElement): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = src.width; c.height = src.height
  c.getContext('2d')!.drawImage(src, 0, 0)
  return c
}

function applyConv2dOp(src: HTMLCanvasElement, kernelIdx: number): HTMLCanvasElement {
  const W = src.width, H = src.height
  const inp = src.getContext('2d')!.getImageData(0, 0, W, H).data
  const kernels = [
    [-1, 0, 1, -2, 0, 2, -1, 0, 1],   // Sobel-X
    [-1, -2, -1, 0, 0, 0, 1, 2, 1],   // Sobel-Y
    [0, -1, 0, -1, 4, -1, 0, -1, 0],  // Laplacian
    [0, -1, 0, -1, 5, -1, 0, -1, 0],  // Sharpen
    [-2, -1, 0, -1, 1, 1, 0, 1, 2],   // Emboss
  ]
  const kernel = kernels[kernelIdx % kernels.length]
  const dst = document.createElement('canvas')
  dst.width = W; dst.height = H
  const dstCtx = dst.getContext('2d')!
  const dstData = dstCtx.createImageData(W, H)
  const out = dstData.data
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let r = 0, g = 0, b = 0
      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const px = Math.max(0, Math.min(W - 1, x + kx))
          const py = Math.max(0, Math.min(H - 1, y + ky))
          const ki = (ky + 1) * 3 + (kx + 1)
          const pi = (py * W + px) * 4
          r += inp[pi] * kernel[ki]; g += inp[pi+1] * kernel[ki]; b += inp[pi+2] * kernel[ki]
        }
      }
      const i = (y * W + x) * 4
      out[i] = Math.max(0, Math.min(255, r + 128))
      out[i+1] = Math.max(0, Math.min(255, g + 128))
      out[i+2] = Math.max(0, Math.min(255, b + 128))
      out[i+3] = 255
    }
  }
  dstCtx.putImageData(dstData, 0, 0)
  return dst
}

function applyReLUOp(src: HTMLCanvasElement): HTMLCanvasElement {
  const W = src.width, H = src.height
  const inp = src.getContext('2d')!.getImageData(0, 0, W, H).data
  const dst = document.createElement('canvas')
  dst.width = W; dst.height = H
  const dstCtx = dst.getContext('2d')!
  const dstData = dstCtx.createImageData(W, H)
  const out = dstData.data
  for (let i = 0; i < inp.length; i += 4) {
    out[i]   = inp[i]   > 128 ? inp[i]   : 0
    out[i+1] = inp[i+1] > 128 ? inp[i+1] : 0
    out[i+2] = inp[i+2] > 128 ? inp[i+2] : 0
    out[i+3] = 255
  }
  dstCtx.putImageData(dstData, 0, 0)
  return dst
}

function applyGELUOp(src: HTMLCanvasElement): HTMLCanvasElement {
  const W = src.width, H = src.height
  const inp = src.getContext('2d')!.getImageData(0, 0, W, H).data
  const dst = document.createElement('canvas')
  dst.width = W; dst.height = H
  const dstCtx = dst.getContext('2d')!
  const dstData = dstCtx.createImageData(W, H)
  const out = dstData.data
  for (let i = 0; i < inp.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const x = (inp[i+c] / 128) - 1
      const gelu = x * 0.5 * (1 + Math.tanh(0.7978845 * (x + 0.044715 * x ** 3)))
      out[i+c] = Math.max(0, Math.min(255, (gelu + 1) * 128))
    }
    out[i+3] = 255
  }
  dstCtx.putImageData(dstData, 0, 0)
  return dst
}

function applySigmoidOp(src: HTMLCanvasElement): HTMLCanvasElement {
  const W = src.width, H = src.height
  const inp = src.getContext('2d')!.getImageData(0, 0, W, H).data
  const dst = document.createElement('canvas')
  dst.width = W; dst.height = H
  const dstCtx = dst.getContext('2d')!
  const dstData = dstCtx.createImageData(W, H)
  const out = dstData.data
  for (let i = 0; i < inp.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const x = (inp[i+c] / 128) - 1
      const sig = 1 / (1 + Math.exp(-x * 4))
      out[i+c] = Math.max(0, Math.min(255, sig * 255))
    }
    out[i+3] = 255
  }
  dstCtx.putImageData(dstData, 0, 0)
  return dst
}

function applyMaxPoolOp(src: HTMLCanvasElement, stride: number): HTMLCanvasElement {
  const W = src.width, H = src.height
  const inp = src.getContext('2d')!.getImageData(0, 0, W, H).data
  const dW = Math.max(1, Math.floor(W / stride)), dH = Math.max(1, Math.floor(H / stride))
  const dst = document.createElement('canvas'); dst.width = dW; dst.height = dH
  const dstCtx = dst.getContext('2d')!
  const dstData = dstCtx.createImageData(dW, dH); const out = dstData.data
  for (let y = 0; y < dH; y++) {
    for (let x = 0; x < dW; x++) {
      let mR = 0, mG = 0, mB = 0
      for (let ky = 0; ky < stride; ky++) for (let kx = 0; kx < stride; kx++) {
        const pi = (Math.min(H-1, y*stride+ky) * W + Math.min(W-1, x*stride+kx)) * 4
        mR = Math.max(mR, inp[pi]); mG = Math.max(mG, inp[pi+1]); mB = Math.max(mB, inp[pi+2])
      }
      const i = (y * dW + x) * 4
      out[i] = mR; out[i+1] = mG; out[i+2] = mB; out[i+3] = 255
    }
  }
  dstCtx.putImageData(dstData, 0, 0)
  const res = document.createElement('canvas'); res.width = W; res.height = H
  const rctx = res.getContext('2d')!; rctx.imageSmoothingEnabled = false
  rctx.drawImage(dst, 0, 0, W, H); return res
}

function applyAvgPoolOp(src: HTMLCanvasElement, stride: number): HTMLCanvasElement {
  const W = src.width, H = src.height
  const inp = src.getContext('2d')!.getImageData(0, 0, W, H).data
  const dW = Math.max(1, Math.floor(W / stride)), dH = Math.max(1, Math.floor(H / stride))
  const dst = document.createElement('canvas'); dst.width = dW; dst.height = dH
  const dstCtx = dst.getContext('2d')!
  const dstData = dstCtx.createImageData(dW, dH); const out = dstData.data
  const n = stride * stride
  for (let y = 0; y < dH; y++) {
    for (let x = 0; x < dW; x++) {
      let sR = 0, sG = 0, sB = 0
      for (let ky = 0; ky < stride; ky++) for (let kx = 0; kx < stride; kx++) {
        const pi = (Math.min(H-1, y*stride+ky) * W + Math.min(W-1, x*stride+kx)) * 4
        sR += inp[pi]; sG += inp[pi+1]; sB += inp[pi+2]
      }
      const i = (y * dW + x) * 4
      out[i] = sR/n; out[i+1] = sG/n; out[i+2] = sB/n; out[i+3] = 255
    }
  }
  dstCtx.putImageData(dstData, 0, 0)
  const res = document.createElement('canvas'); res.width = W; res.height = H
  const rctx = res.getContext('2d')!; rctx.imageSmoothingEnabled = true
  rctx.drawImage(dst, 0, 0, W, H); return res
}

function applyBatchNormOp(src: HTMLCanvasElement): HTMLCanvasElement {
  const W = src.width, H = src.height
  const inp = src.getContext('2d')!.getImageData(0, 0, W, H).data
  const N = W * H
  const mean = [0, 0, 0]
  for (let i = 0; i < inp.length; i += 4) { mean[0] += inp[i]; mean[1] += inp[i+1]; mean[2] += inp[i+2] }
  mean[0] /= N; mean[1] /= N; mean[2] /= N
  const variance = [0, 0, 0]
  for (let i = 0; i < inp.length; i += 4) {
    variance[0] += (inp[i]-mean[0])**2; variance[1] += (inp[i+1]-mean[1])**2; variance[2] += (inp[i+2]-mean[2])**2
  }
  const std = variance.map(v => Math.sqrt(v / N) + 1e-5)
  const dst = document.createElement('canvas'); dst.width = W; dst.height = H
  const dstCtx = dst.getContext('2d')!
  const dstData = dstCtx.createImageData(W, H); const out = dstData.data
  for (let i = 0; i < inp.length; i += 4) {
    for (let c = 0; c < 3; c++) out[i+c] = Math.max(0, Math.min(255, ((inp[i+c]-mean[c])/std[c]) * 40 + 128))
    out[i+3] = 255
  }
  dstCtx.putImageData(dstData, 0, 0); return dst
}

function applyDropoutOp(src: HTMLCanvasElement, p: number, seed: number): HTMLCanvasElement {
  const W = src.width, H = src.height
  const inp = src.getContext('2d')!.getImageData(0, 0, W, H).data
  let _s = seed ^ 0xdeadbeef
  const rng = () => {
    _s |= 0; _s = _s + 0x6D2B79F5 | 0
    let t = Math.imul(_s ^ (_s >>> 15), 1 | _s)
    t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const dst = document.createElement('canvas'); dst.width = W; dst.height = H
  const dstCtx = dst.getContext('2d')!
  const dstData = dstCtx.createImageData(W, H); const out = dstData.data
  for (let i = 0; i < inp.length; i += 4) {
    const drop = rng() < p
    out[i] = drop ? 0 : inp[i]; out[i+1] = drop ? 0 : inp[i+1]
    out[i+2] = drop ? 0 : inp[i+2]; out[i+3] = 255
  }
  dstCtx.putImageData(dstData, 0, 0); return dst
}

function buildActivationSequence(layers: Layer[], numClasses: number): HTMLCanvasElement[] {
  const SIZE = 64
  const canvases: HTMLCanvasElement[] = []
  let current = createBaseImage(SIZE)
  canvases.push(cloneCanvas(current))

  let convIdx = 0
  for (const layer of layers) {
    switch (layer.type) {
      case 'conv2d':    current = applyConv2dOp(current, convIdx++); break
      case 'relu':      current = applyReLUOp(current); break
      case 'gelu':      current = applyGELUOp(current); break
      case 'sigmoid':   current = applySigmoidOp(current); break
      case 'maxpool2d': current = applyMaxPoolOp(current, layer.params.stride ?? layer.params.kernel_size ?? 2); break
      case 'avgpool2d': current = applyAvgPoolOp(current, layer.params.stride ?? layer.params.kernel_size ?? 2); break
      case 'batchnorm2d': current = applyBatchNormOp(current); break
      case 'dropout':   current = applyDropoutOp(current, layer.params.p ?? 0.5, convIdx * 37); break
      case 'flatten': {
        const flat = document.createElement('canvas'); flat.width = SIZE; flat.height = SIZE
        const fctx = flat.getContext('2d')!
        fctx.fillStyle = '#04040a'; fctx.fillRect(0, 0, SIZE, SIZE)
        for (let s = 0; s < 4; s++) fctx.drawImage(current, 0, s * SIZE / 4, SIZE, SIZE / 4)
        current = flat; break
      }
      case 'linear': {
        const lin = document.createElement('canvas'); lin.width = SIZE; lin.height = SIZE
        const lctx = lin.getContext('2d')!
        lctx.fillStyle = '#04040a'; lctx.fillRect(0, 0, SIZE, SIZE)
        const srcD = current.getContext('2d')!.getImageData(0, 0, current.width, current.height).data
        const nBars = Math.min(layer.params.out_features ?? 128, 48)
        for (let n = 0; n < nBars; n++) {
          const pIdx = Math.floor(n / nBars * srcD.length / 4) * 4
          const avg = (srcD[pIdx] + srcD[pIdx+1] + srcD[pIdx+2]) / 3
          const barH = (avg / 255) * (SIZE - 6)
          const bx = (n / nBars) * SIZE; const bw = SIZE / nBars
          lctx.fillStyle = `rgb(${srcD[pIdx]},${srcD[pIdx+1]},${srcD[pIdx+2]})`
          lctx.fillRect(bx, SIZE - barH - 3, Math.max(1, bw - 0.5), barH)
        }
        current = lin; break
      }
      default: current = cloneCanvas(current)
    }
    canvases.push(cloneCanvas(current))
  }

  // Output node: softmax probability bars
  const out = document.createElement('canvas'); out.width = SIZE; out.height = SIZE
  const octx = out.getContext('2d')!
  octx.fillStyle = '#04040a'; octx.fillRect(0, 0, SIZE, SIZE)
  const bw = SIZE / numClasses
  for (let c = 0; c < numClasses; c++) {
    const v = c === 0 ? 0.82 : 0.04 + Math.sin(c * 1.3) * 0.04 + 0.04
    octx.fillStyle = `hsl(${(c / numClasses) * 280 + 120},70%,52%)`
    octx.fillRect(c * bw + 1, SIZE * (1 - v) - 2, bw - 2, SIZE * v)
  }
  canvases.push(out)
  return canvases
}

// ── 2D architecture diagram renderer ─────────────────────────────────────────

function draw2DScene(
  canvas: HTMLCanvasElement,
  layers: Layer[],
  inputH: number,
  inputW: number,
  numClasses: number,
  activationCanvases: HTMLCanvasElement[] | null = null,
) {
  const ctx  = canvas.getContext('2d')!
  const CW   = canvas.width
  const CH   = canvas.height

  ctx.fillStyle = '#0d0d0f'
  ctx.fillRect(0, 0, CW, CH)

  const shapes   = [...computeAllShapes(layers, inputH, inputW), [numClasses]]
  const N        = shapes.length
  const CENTER_Y = CH * 0.44
  const MAX_VIS_H = CH * 0.52
  const PLANE_W   = 20
  const DX        = 3.8   // depth offset per channel-plane, x
  const DY        = -2.8  // depth offset per channel-plane, y
  const MAX_PL    = 9
  const MARGIN    = 44

  // ── Compute visual params ────────────────────────────────────────────────────
  const vis = shapes.map((shape, i) => {
    const layerType = i === 0 ? 'input' : i <= layers.length ? layers[i - 1].type : 'output'
    const color     = layerType === 'output' ? '#22c55e' : (LAYER_CSS[layerType] ?? '#8b8b9a')
    const label     = layerType === 'output' ? 'output' : (LAYER_DISPLAY[layerType] ?? layerType)

    if (!isValidShape(shape)) return { visH: 20, nPl: 1, color, label, shapeStr: '⚠', layerType }

    let visH: number, nPl: number, shapeStr: string
    if (shape.length === 3) {
      const [C, H, W] = shape
      visH     = Math.max(18, Math.min((Math.max(H, W) / inputH) * MAX_VIS_H * 1.6, MAX_VIS_H))
      nPl      = Math.min(C, MAX_PL)
      shapeStr = `${C}×${H}×${W}`
    } else {
      visH     = Math.max(18, Math.min((shape[0] / 128) * MAX_VIS_H, MAX_VIS_H))
      nPl      = 1
      shapeStr = `${shape[0]}`
    }
    return { visH, nPl, color, label, shapeStr, layerType }
  })

  // ── Horizontal layout ────────────────────────────────────────────────────────
  const blobW    = vis.map(v => PLANE_W + (v.nPl - 1) * DX)
  const totalBW  = blobW.reduce((a, b) => a + b, 0)
  const gap      = Math.max(14, (CW - 2 * MARGIN - totalBW) / Math.max(N - 1, 1))
  const blobX: number[] = []
  let cx = MARGIN
  for (let i = 0; i < N; i++) { blobX.push(cx); cx += blobW[i] + gap }

  // ── Draw trapezoid connectors (behind blobs) ─────────────────────────────────
  for (let i = 1; i < N; i++) {
    const pv = vis[i - 1], cv = vis[i]
    const x1 = blobX[i - 1] + PLANE_W
    const x2 = blobX[i]
    if (x2 <= x1 + 2) continue
    const connColor = cv.layerType === 'output' ? '#22c55e' : (LAYER_CSS[cv.layerType] ?? '#8b8b9a')

    ctx.beginPath()
    ctx.moveTo(x1, CENTER_Y - pv.visH / 2)
    ctx.lineTo(x2, CENTER_Y - cv.visH / 2)
    ctx.lineTo(x2, CENTER_Y + cv.visH / 2)
    ctx.lineTo(x1, CENTER_Y + pv.visH / 2)
    ctx.closePath()
    const grd = ctx.createLinearGradient(x1, 0, x2, 0)
    grd.addColorStop(0, connColor + '28')
    grd.addColorStop(1, connColor + '10')
    ctx.fillStyle = grd
    ctx.fill()
    ctx.strokeStyle = connColor + '44'
    ctx.lineWidth = 0.5
    ctx.stroke()
  }

  // ── Draw blobs ───────────────────────────────────────────────────────────────
  for (let i = 0; i < N; i++) {
    const { visH, nPl, color } = vis[i]
    const x = blobX[i]

    // Top cap
    if (nPl > 1) {
      const lox = (nPl - 1) * DX, loy = (nPl - 1) * DY
      ctx.beginPath()
      ctx.moveTo(x,          CENTER_Y - visH / 2)
      ctx.lineTo(x + PLANE_W, CENTER_Y - visH / 2)
      ctx.lineTo(x + PLANE_W + lox, CENTER_Y - visH / 2 + loy)
      ctx.lineTo(x + lox,          CENTER_Y - visH / 2 + loy)
      ctx.closePath()
      ctx.fillStyle = color + '25'
      ctx.fill()
      ctx.strokeStyle = color + '55'
      ctx.lineWidth = 0.8
      ctx.stroke()

      // Right cap
      ctx.beginPath()
      ctx.moveTo(x + PLANE_W,       CENTER_Y - visH / 2)
      ctx.lineTo(x + PLANE_W + lox, CENTER_Y - visH / 2 + loy)
      ctx.lineTo(x + PLANE_W + lox, CENTER_Y + visH / 2 + loy)
      ctx.lineTo(x + PLANE_W,       CENTER_Y + visH / 2)
      ctx.closePath()
      ctx.fillStyle = color + '1a'
      ctx.fill()
      ctx.strokeStyle = color + '44'
      ctx.stroke()
    }

    // Planes back → front
    for (let pi = nPl - 1; pi >= 0; pi--) {
      const ox      = pi * DX
      const oy      = pi * DY
      const t       = nPl > 1 ? pi / (nPl - 1) : 0
      const isFront = pi === 0
      const px      = x + ox
      const py      = CENTER_Y - visH / 2 + oy

      ctx.shadowColor = isFront ? color : 'transparent'
      ctx.shadowBlur  = isFront ? 12 : 0

      const actCanvas = activationCanvases?.[i]
      if (actCanvas && isFront) {
        ctx.save()
        ctx.beginPath()
        ctx.rect(px, py, PLANE_W, visH)
        ctx.clip()
        ctx.globalAlpha = 0.94
        ctx.drawImage(actCanvas, px, py, PLANE_W, visH)
        ctx.globalAlpha = 1.0
        ctx.restore()
      } else {
        const grd = ctx.createLinearGradient(px, py, px + PLANE_W, py + visH)
        const a1  = isFront ? 'cc' : (0x30 + Math.floor(t * 0x40)).toString(16).padStart(2, '0')
        const a2  = isFront ? '88' : (0x20 + Math.floor(t * 0x30)).toString(16).padStart(2, '0')
        grd.addColorStop(0, color + a1)
        grd.addColorStop(1, color + a2)
        ctx.fillStyle = grd
        ctx.fillRect(px, py, PLANE_W, visH)
      }

      ctx.shadowBlur  = 0
      ctx.strokeStyle = color + (isFront ? 'ee' : '40')
      ctx.lineWidth   = isFront ? 1.2 : 0.5
      ctx.strokeRect(px, py, PLANE_W, visH)
    }
  }

  // ── Labels ───────────────────────────────────────────────────────────────────
  for (let i = 0; i < N; i++) {
    const { visH, nPl, color, label, shapeStr } = vis[i]
    const lx = blobX[i] + PLANE_W / 2 + ((nPl - 1) * DX) / 2
    const ly = CENTER_Y + visH / 2 + 18

    ctx.font      = 'bold 10px "JetBrains Mono", monospace'
    ctx.fillStyle = color
    ctx.textAlign = 'center'
    ctx.shadowColor = color
    ctx.shadowBlur  = 6
    ctx.fillText(label, lx, ly)
    ctx.shadowBlur  = 0

    ctx.font      = '9px "JetBrains Mono", monospace'
    ctx.fillStyle = 'rgba(140,140,160,0.65)'
    ctx.fillText(shapeStr, lx, ly + 13)
  }
}

// ── Educational texture generator ─────────────────────────────────────────────
// Each texture visually simulates what that layer type does to the data.
// makeLayerTextureCanvas returns the raw canvas (shared by 2D and 3D views).
// makeLayerTexture wraps it as a Three.js texture for the 3D renderer.

function makeLayerTextureCanvas(layerType: string, colorHex: number, seed: number): HTMLCanvasElement {
  const S = 64
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = S
  const ctx = canvas.getContext('2d')!
  const r = (colorHex >> 16) & 0xff
  const g = (colorHex >> 8)  & 0xff
  const b =  colorHex        & 0xff

  // Seeded deterministic RNG (mulberry32)
  let _s = seed ^ 0xdeadbeef
  const rng = () => {
    _s |= 0; _s = _s + 0x6D2B79F5 | 0
    let t = Math.imul(_s ^ (_s >>> 15), 1 | _s)
    t = t + Math.imul(t ^ (t >>> 7), 61 | t) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }

  ctx.fillStyle = '#04040a'
  ctx.fillRect(0, 0, S, S)

  switch (layerType) {
    case 'input': {
      // Simulate a natural image: sky/object/ground gradient with texture
      const gy = ctx.createLinearGradient(0, 0, 0, S)
      gy.addColorStop(0,    'rgba(25,70,160,1)')
      gy.addColorStop(0.35, 'rgba(70,170,70,1)')
      gy.addColorStop(0.65, 'rgba(90,130,50,1)')
      gy.addColorStop(1,    'rgba(55,35,15,1)')
      ctx.fillStyle = gy
      ctx.fillRect(0, 0, S, S)
      for (let i = 0; i < 500; i++) {
        const x = rng() * S, y = rng() * S, v = (rng() - 0.5) * 0.4
        ctx.fillStyle = `rgba(${Math.min(255, 128 + v * 255 | 0)},${Math.min(255, 100 + v * 200 | 0)},${Math.min(255, 40 + v * 100 | 0)},0.2)`
        ctx.fillRect(x, y, 2, 2)
      }
      break
    }
    case 'conv2d': {
      // Gabor-like edge-detection activation map
      const angle = rng() * Math.PI, freq = 4 + rng() * 4
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x += 2) {
          const nx = x / S - 0.5, ny = y / S - 0.5
          const v = Math.sin(freq * (nx * Math.cos(angle) + ny * Math.sin(angle)) * Math.PI * 6) * 0.5 + 0.5
          const pv = Math.pow(v, 1.8)
          if (pv > 0.05) {
            ctx.fillStyle = `rgba(${r * pv | 0},${g * pv | 0},${b * pv | 0},${0.2 + pv * 0.8})`
            ctx.fillRect(x, y, 2, 1)
          }
        }
      }
      break
    }
    case 'relu': {
      // ReLU zeros negatives — sparse bright patches on dark
      const angle = rng() * Math.PI, freq = 4 + rng() * 4
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x += 2) {
          const nx = x / S - 0.5, ny = y / S - 0.5
          const raw = Math.sin(freq * (nx * Math.cos(angle) + ny * Math.sin(angle)) * Math.PI * 6)
          const v = Math.max(0, raw)   // ReLU: kill negatives
          if (v > 0.05) {
            ctx.fillStyle = `rgba(${r * v | 0},${g * v | 0},${b * v | 0},${v * 0.95})`
            ctx.fillRect(x, y, 2, 1)
          }
        }
      }
      break
    }
    case 'maxpool2d': {
      // Blocky pixels — spatial compression with bright "winner" highlights
      const bs = 8
      for (let by = 0; by < S; by += bs) {
        for (let bx = 0; bx < S; bx += bs) {
          const v = 0.25 + rng() * 0.75
          ctx.fillStyle = `rgba(${r * v | 0},${g * v | 0},${b * v | 0},${0.5 + v * 0.4})`
          ctx.fillRect(bx + 1, by + 1, bs - 2, bs - 2)
          // Max pixel highlight
          ctx.fillStyle = 'rgba(255,255,255,0.45)'
          ctx.fillRect(bx + (rng() * (bs - 3) | 0), by + (rng() * (bs - 3) | 0), 2, 2)
        }
      }
      break
    }
    case 'avgpool2d': {
      // Smooth blurry blocks — averaging effect
      const bs = 7
      for (let by = 0; by < S; by += bs) {
        for (let bx = 0; bx < S; bx += bs) {
          const v = 0.2 + rng() * 0.55
          for (let dy = 0; dy < bs; dy++) {
            for (let dx = 0; dx < bs; dx++) {
              const d = Math.sqrt((dx - bs/2)**2 + (dy - bs/2)**2) / bs
              const lv = v * (1 - d * 0.4)
              ctx.fillStyle = `rgba(${r * lv | 0},${g * lv | 0},${b * lv | 0},0.7)`
              ctx.fillRect(bx + dx, by + dy, 1, 1)
            }
          }
        }
      }
      break
    }
    case 'batchnorm2d': {
      // Normalized: even distribution across the full range
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          const v = 0.15 + ((x + y) / (S * 2)) * 0.75
          ctx.fillStyle = `rgba(${r * v | 0},${g * v | 0},${b * v | 0},0.8)`
          ctx.fillRect(x, y, 1, 1)
        }
      }
      break
    }
    case 'dropout': {
      // ~50% neurons zeroed — visible sparse gaps
      for (let y = 0; y < S; y += 3) {
        for (let x = 0; x < S; x += 3) {
          if (rng() > 0.5) {
            const v = 0.4 + rng() * 0.6
            ctx.fillStyle = `rgba(${Math.min(255, r * v * 2 | 0)},${Math.min(255, g * v * 2 | 0)},${Math.min(255, b * v * 2 | 0)},0.9)`
            ctx.fillRect(x, y, 2, 2)
          }
        }
      }
      break
    }
    case 'sigmoid': {
      // S-curve mapping: values squashed to (0,1) — horizontal gradient
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          const raw = (x / S - 0.5) * 8
          const v = 0.05 + (1 / (1 + Math.exp(-raw))) * 0.95
          ctx.fillStyle = `rgba(${r * v | 0},${g * v | 0},${b * v | 0},0.85)`
          ctx.fillRect(x, y, 1, 1)
        }
      }
      break
    }
    case 'gelu': {
      // GELU: similar to sigmoid but slightly asymmetric smooth curve
      for (let y = 0; y < S; y++) {
        for (let x = 0; x < S; x++) {
          const raw = (x / S - 0.5) * 8
          const gelu = raw * 0.5 * (1 + Math.tanh(0.7978845 * (raw + 0.044715 * raw ** 3)))
          const v = Math.max(0, Math.min(1, gelu / 4 + 0.5))
          ctx.fillStyle = `rgba(${r * v | 0},${g * v | 0},${b * v | 0},0.85)`
          ctx.fillRect(x, y, 1, 1)
        }
      }
      break
    }
    default: {
      for (let i = 0; i < S * S / 4; i++) {
        const x = rng() * S, y = rng() * S, v = rng()
        ctx.fillStyle = `rgba(${r * v | 0},${g * v | 0},${b * v | 0},0.6)`
        ctx.fillRect(x, y, 2, 2)
      }
    }
  }

  return canvas
}

function makeLayerTexture(layerType: string, colorHex: number, seed: number): THREE.CanvasTexture {
  return new THREE.CanvasTexture(makeLayerTextureCanvas(layerType, colorHex, seed))
}

// ── Three.js scene builder ────────────────────────────────────────────────────

function buildScene(
  scene: THREE.Scene,
  layers: Layer[],
  inputH: number,
  inputW: number,
  particlesRef: React.MutableRefObject<THREE.Mesh[]>,
  numClasses: number,
  viewMode: 'color' | 'image',
  activationCanvases: HTMLCanvasElement[] | null = null,
) {
  while (scene.children.length > 0) scene.remove(scene.children[0])
  particlesRef.current = []

  // Global lighting
  scene.add(new THREE.AmbientLight(0x0a0a1a, 6))
  const sun = new THREE.DirectionalLight(0xffffff, 0.5)
  sun.position.set(4, 10, 6)
  scene.add(sun)

  // Depth fog
  scene.fog = new THREE.FogExp2(0x0d0d0f, 0.016)

  const shapes   = computeAllShapes(layers, inputH, inputW)
  const MAX_PLANES = 10
  const NORM_MAX   = 1.4
  const BOX_DEPTH  = 0.055

  for (let si = 0; si < shapes.length; si++) {
    const shape     = shapes[si]
    const z         = -si * SPACING
    const layerType = si === 0 ? 'input' : layers[si - 1].type
    const colorHex  = LAYER_COLORS[layerType] ?? 0x8b8b9a
    const colorCSS  = LAYER_CSS[layerType]    ?? '#8b8b9a'
    const valid     = isValidShape(shape)

    if (!valid) {
      const geo  = new THREE.BoxGeometry(0.5, 0.5, 0.5)
      const mat  = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 0.9 })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(0, 0, z)
      scene.add(mesh)
      continue
    }

    // Colored point light per layer — illuminates nearby planes
    const pl = new THREE.PointLight(colorHex, 1.8, 7)
    pl.position.set(0.6, 0.4, z)
    scene.add(pl)

    if (shape.length === 1) {
      const N      = shape[0]
      const height = Math.min(N / 64 * 2.2, 3.4)
      const width  = 0.2

      const geo = new THREE.BoxGeometry(width, height, BOX_DEPTH)
      const mat = new THREE.MeshStandardMaterial({
        color: colorHex, emissive: colorHex, emissiveIntensity: 0.55,
        transparent: true, opacity: 0.88, roughness: 0.2, metalness: 0.7,
      })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(0, 0, z)
      scene.add(mesh)

      const edgesGeo  = new THREE.EdgesGeometry(geo)
      const edgesMesh = new THREE.LineSegments(edgesGeo,
        new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 1.0 }))
      edgesMesh.position.copy(mesh.position)
      scene.add(edgesMesh)

      const labelType = si === 0 ? 'input' : layers[si - 1].type
      const sprite    = makeLabel(LAYER_DISPLAY[labelType] ?? labelType, `${N}`, colorCSS)
      sprite.position.set(0, -(height / 2) - 0.62, z)
      scene.add(sprite)

    } else {
      const [C, H, W] = shape
      const maxDim    = Math.max(H, W, 1)
      const sc        = NORM_MAX / maxDim
      const planeH    = H * sc
      const planeW    = W * sc
      const nPlanes   = Math.min(C, MAX_PLANES)

      for (let pi = 0; pi < nPlanes; pi++) {
        const t   = nPlanes > 1 ? pi / (nPlanes - 1) : 0
        const ox  = pi * 0.09
        const oy  = pi * 0.032
        const oz  = pi * 0.075

        const geo = new THREE.BoxGeometry(planeW, planeH, BOX_DEPTH)
        const actCanvas = activationCanvases?.[si]
        const mat = (viewMode === 'image' && actCanvas)
          ? new THREE.MeshBasicMaterial({
              map: new THREE.CanvasTexture(actCanvas),
              transparent: true, opacity: 0.2 + t * 0.7,
              side: THREE.DoubleSide,
            })
          : viewMode === 'image'
          ? new THREE.MeshBasicMaterial({
              map: makeLayerTexture(layerType, colorHex, si * 100 + pi),
              transparent: true, opacity: 0.2 + t * 0.7,
              side: THREE.DoubleSide,
            })
          : new THREE.MeshStandardMaterial({
              color: colorHex, emissive: colorHex,
              emissiveIntensity: 0.18 + t * 0.45,
              transparent: true, opacity: 0.14 + t * 0.38,
              roughness: 0.15, metalness: 0.55, side: THREE.DoubleSide,
            })
        const mesh = new THREE.Mesh(geo, mat)
        mesh.position.set(ox, oy, z + oz)
        scene.add(mesh)

        const edgesGeo = new THREE.EdgesGeometry(geo)
        const edges    = new THREE.LineSegments(edgesGeo,
          new THREE.LineBasicMaterial({ color: colorHex, transparent: true, opacity: 0.35 + t * 0.6 }))
        edges.position.copy(mesh.position)
        scene.add(edges)
      }

      const labelType = si === 0 ? 'input' : layers[si - 1].type
      const shapeText = `${C}×${H}×${W}`
      const sprite    = makeLabel(LAYER_DISPLAY[labelType] ?? labelType, shapeText, colorCSS)
      sprite.position.set(0, -(planeH / 2) - 0.65, z)
      scene.add(sprite)
    }

    // Connectors + flowing particles to previous slot
    if (si > 0 && isValidShape(shapes[si - 1]) && valid) {
      const prevShape    = shapes[si - 1]
      const prevZ        = -(si - 1) * SPACING
      const connColorHex = LAYER_COLORS[layers[si - 1].type] ?? 0x8b8b9a

      const getPlaneDims = (sh: number[]) => {
        if (sh.length === 1) return { w: 0.2, h: Math.min(sh[0] / 64 * 2.2, 3.4) }
        const [, H2, W2] = sh
        const maxD = Math.max(H2, W2, 1)
        const s    = NORM_MAX / maxD
        return { w: W2 * s, h: H2 * s }
      }
      const prev = getPlaneDims(prevShape)
      const curr = getPlaneDims(shape)

      const corners: [THREE.Vector3, THREE.Vector3][] = [
        [new THREE.Vector3(-prev.w / 2, -prev.h / 2, prevZ), new THREE.Vector3(-curr.w / 2, -curr.h / 2, z)],
        [new THREE.Vector3( prev.w / 2, -prev.h / 2, prevZ), new THREE.Vector3( curr.w / 2, -curr.h / 2, z)],
        [new THREE.Vector3(-prev.w / 2,  prev.h / 2, prevZ), new THREE.Vector3(-curr.w / 2,  curr.h / 2, z)],
        [new THREE.Vector3( prev.w / 2,  prev.h / 2, prevZ), new THREE.Vector3( curr.w / 2,  curr.h / 2, z)],
      ]
      for (const [a, b] of corners) {
        const geo = new THREE.BufferGeometry().setFromPoints([a, b])
        scene.add(new THREE.Line(geo,
          new THREE.LineBasicMaterial({ color: connColorHex, transparent: true, opacity: 0.28 })))
      }

      // Animated signal particles
      const pGeo = new THREE.SphereGeometry(0.048, 7, 7)
      const pMat = new THREE.MeshStandardMaterial({
        color: connColorHex, emissive: connColorHex,
        emissiveIntensity: 1.8, roughness: 0.0, metalness: 1.0,
      })
      for (let p = 0; p < 4; p++) {
        const particle      = new THREE.Mesh(pGeo, pMat)
        particle.userData   = {
          startZ: prevZ, endZ: z,
          progress: p / 4,
          speed: 0.006 + Math.random() * 0.003,
          xPhase: Math.random() * Math.PI * 2,
          yPhase: Math.random() * Math.PI * 2,
        }
        scene.add(particle)
        particlesRef.current.push(particle)
      }
    }
  }

  // ── Output classifier head (always appended by backend) ───────────────────
  const outColorHex = 0x22c55e
  const outColorCSS = '#22c55e'
  const lastShape   = shapes[shapes.length - 1]
  const outZ        = -shapes.length * SPACING
  const outHeight   = Math.min(numClasses / 4 * 0.8, 3.4)
  const outWidth    = 0.2

  const outPl = new THREE.PointLight(outColorHex, 2.0, 7)
  outPl.position.set(0.6, 0.4, outZ)
  scene.add(outPl)

  const outGeo  = new THREE.BoxGeometry(outWidth, outHeight, BOX_DEPTH)
  const outMat  = new THREE.MeshStandardMaterial({
    color: outColorHex, emissive: outColorHex, emissiveIntensity: 0.7,
    transparent: true, opacity: 0.9, roughness: 0.15, metalness: 0.75,
  })
  const outMesh = new THREE.Mesh(outGeo, outMat)
  outMesh.position.set(0, 0, outZ)
  scene.add(outMesh)

  const outEdgesGeo  = new THREE.EdgesGeometry(outGeo)
  const outEdgesMesh = new THREE.LineSegments(outEdgesGeo,
    new THREE.LineBasicMaterial({ color: outColorHex, transparent: true, opacity: 1.0 }))
  outEdgesMesh.position.copy(outMesh.position)
  scene.add(outEdgesMesh)

  const outSprite = makeLabel('output', `${numClasses} classes`, outColorCSS)
  outSprite.position.set(0, -(outHeight / 2) - 0.62, outZ)
  scene.add(outSprite)

  // Connector + particles from last user layer to output
  if (isValidShape(lastShape)) {
    const prevZ        = -(shapes.length - 1) * SPACING
    const connColorHex = layers.length > 0 ? (LAYER_COLORS[layers[layers.length - 1].type] ?? outColorHex) : outColorHex

    const getPlaneDims = (sh: number[]) => {
      if (sh.length === 1) return { w: 0.2, h: Math.min(sh[0] / 64 * 2.2, 3.4) }
      const [, H2, W2] = sh
      const maxD = Math.max(H2, W2, 1)
      const s    = NORM_MAX / maxD
      return { w: W2 * s, h: H2 * s }
    }
    const prev = getPlaneDims(lastShape)
    const curr = { w: outWidth, h: outHeight }

    const corners: [THREE.Vector3, THREE.Vector3][] = [
      [new THREE.Vector3(-prev.w / 2, -prev.h / 2, prevZ), new THREE.Vector3(-curr.w / 2, -curr.h / 2, outZ)],
      [new THREE.Vector3( prev.w / 2, -prev.h / 2, prevZ), new THREE.Vector3( curr.w / 2, -curr.h / 2, outZ)],
      [new THREE.Vector3(-prev.w / 2,  prev.h / 2, prevZ), new THREE.Vector3(-curr.w / 2,  curr.h / 2, outZ)],
      [new THREE.Vector3( prev.w / 2,  prev.h / 2, prevZ), new THREE.Vector3( curr.w / 2,  curr.h / 2, outZ)],
    ]
    for (const [a, b] of corners) {
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]),
        new THREE.LineBasicMaterial({ color: outColorHex, transparent: true, opacity: 0.28 })))
    }

    const pGeo2 = new THREE.SphereGeometry(0.048, 7, 7)
    const pMat2 = new THREE.MeshStandardMaterial({
      color: outColorHex, emissive: outColorHex, emissiveIntensity: 1.8, roughness: 0, metalness: 1,
    })
    for (let p = 0; p < 4; p++) {
      const particle    = new THREE.Mesh(pGeo2, pMat2)
      particle.userData = {
        startZ: prevZ, endZ: outZ,
        progress: p / 4,
        speed: 0.006 + Math.random() * 0.003,
        xPhase: Math.random() * Math.PI * 2,
        yPhase: Math.random() * Math.PI * 2,
      }
      scene.add(particle)
      particlesRef.current.push(particle)
    }
    void connColorHex
  }
}

// ── Layer Param Editor ────────────────────────────────────────────────────────

function ParamEditor({
  layer,
  onChange,
}: {
  layer: Layer
  onChange: (params: Record<string, number>) => void
}) {
  const p = layer.params

  const inp = (
    key: string, label: string,
    min: number, max: number, step = 1,
  ) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <label style={{ fontSize: 11, color: 'var(--text2)', width: 80, flexShrink: 0 }}>{label}</label>
      <input
        type="number" min={min} max={max} step={step}
        value={p[key] ?? 0}
        onChange={e => onChange({ ...p, [key]: Number(e.target.value) })}
        style={{
          flex: 1, padding: '4px 8px', background: 'var(--surface3)',
          border: '1px solid var(--border2)', borderRadius: 4,
          color: 'var(--text)', fontSize: 12, fontFamily: 'inherit',
        }}
      />
    </div>
  )

  if (layer.type === 'conv2d') return (
    <div>
      {inp('filters',     'Filters',     1, 512)}
      {inp('kernel_size', 'Kernel',      1, 7)}
      {inp('stride',      'Stride',      1, 4)}
      {inp('padding',     'Padding',     0, 3)}
    </div>
  )
  if (layer.type === 'maxpool2d' || layer.type === 'avgpool2d') return (
    <div>
      {inp('kernel_size', 'Kernel', 2, 4)}
      {inp('stride',      'Stride', 1, 4)}
    </div>
  )
  if (layer.type === 'dropout') return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <label style={{ fontSize: 11, color: 'var(--text2)', width: 80 }}>Dropout p</label>
        <input
          type="range" min={0} max={1} step={0.05}
          value={p.p ?? 0.5}
          onChange={e => onChange({ ...p, p: Number(e.target.value) })}
          style={{ flex: 1, accentColor: 'var(--accent)' }}
        />
        <span style={{ fontSize: 11, color: 'var(--text)', fontFamily: 'monospace', width: 32 }}>
          {(p.p ?? 0.5).toFixed(2)}
        </span>
      </div>
    </div>
  )
  if (layer.type === 'linear') return (
    <div>{inp('out_features', 'Out features', 1, 4096)}</div>
  )
  return <p style={{ fontSize: 11, color: 'var(--text3)', fontStyle: 'italic' }}>No parameters</p>
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function CustomModel() {
  const { id } = useParams<{ id: string }>()
  const projectId = Number(id)
  const navigate  = useNavigate()

  // Project info
  const [projectName, setProjectName] = useState('…')

  // Config state
  const [layers,   setLayers]   = useState<Layer[]>(DEFAULT_LAYERS)
  const [inputH,   setInputH]   = useState(64)
  const [inputW,   setInputW]   = useState(64)
  const [modelName, setModelName] = useState('My Model')
  const [savedConfig, setSavedConfig] = useState<CustomConfig | null>(null)
  const [loading, setLoading] = useState(false)
  const [numClasses, setNumClasses] = useState(2)

  // Selected layer
  const [selectedId, setSelectedId] = useState<string | null>(null)

  // Training state
  const [epochs, setEpochs] = useState(20)
  const [batch,  setBatch]  = useState(32)
  const [lr,     setLr]     = useState(0.001)
  const [activeRun, setActiveRun] = useState<CustomRun | null>(null)
  const [logs,   setLogs]   = useState<string[]>([])
  const logRef  = useRef<HTMLDivElement>(null)
  const sseRef  = useRef<EventSource | null>(null)

  // Add layer dropdown
  const [showAddMenu, setShowAddMenu] = useState(false)
  const [hoveredAddType, setHoveredAddType] = useState<LayerType | null>(null)

  // Three.js refs
  const canvasRef       = useRef<HTMLDivElement>(null)
  const rendererRef     = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef        = useRef<THREE.Scene | null>(null)
  const cameraRef       = useRef<THREE.PerspectiveCamera | null>(null)
  const animFrameRef    = useRef<number>(0)
  const particlesRef    = useRef<THREE.Mesh[]>([])
  const cameraUpdateRef = useRef<() => void>(() => {})
  const [autoRotate, setAutoRotate] = useState(true)
  const autoRotateRef = useRef(true)
  const [viewMode, setViewMode] = useState<'color' | 'image'>('color')
  const [viewDim,  setViewDim]  = useState<'3d' | '2d'>('3d')
  const canvas2DRef = useRef<HTMLCanvasElement>(null)

  // Orbit state — slightly dramatic angle by default
  const orbitRef = useRef({ theta: 0.45, phi: 0.88, radius: 11, dragging: false, lastX: 0, lastY: 0 })

  // Free-fly camera (Blender/UE style)
  const [flyMode, setFlyMode] = useState(false)
  const flyModeRef = useRef(false)
  const flyRef = useRef({
    yaw: Math.PI,
    pitch: -0.15,
    pos: new THREE.Vector3(0, 1.5, 8),
    keys: new Set<string>(),
    speed: 0.12,
  })

  // Debounce timer
  const autoSaveTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // ── Load project + configs ────────────────────────────────────────────────

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const [pRes, cfgRes] = await Promise.all([
          api.get(`/projects/${projectId}`),
          api.get(`/projects/${projectId}/custom/configs`),
        ])
        setProjectName(pRes.data.name)
        setNumClasses(pRes.data.classes?.length ?? 2)
        const cfgs: CustomConfig[] = cfgRes.data
        if (cfgs.length > 0) {
          const cfg = cfgs[cfgs.length - 1]
          setSavedConfig(cfg)
          setModelName(cfg.name)
          setLayers(cfg.layers.length > 0 ? cfg.layers : DEFAULT_LAYERS)
          setInputH(cfg.input_h)
          setInputW(cfg.input_w)
        }
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [projectId])

  // ── Auto-save debounce ────────────────────────────────────────────────────

  const triggerAutoSave = useCallback(() => {
    if (autoSaveTimer.current) clearTimeout(autoSaveTimer.current)
    autoSaveTimer.current = setTimeout(async () => {
      try {
        const body = { name: modelName, layers, input_h: inputH, input_w: inputW }
        if (savedConfig) {
          const res = await api.put(`/projects/${projectId}/custom/configs/${savedConfig.id}`, body)
          setSavedConfig(res.data)
        } else {
          const res = await api.post(`/projects/${projectId}/custom/configs`, body)
          setSavedConfig(res.data)
        }
      } catch (e) {
        // Silently ignore
      }
    }, 1000)
  }, [projectId, modelName, layers, inputH, inputW, savedConfig])

  useEffect(() => {
    if (!loading) triggerAutoSave()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, inputH, inputW, modelName])

  // ── Three.js setup ────────────────────────────────────────────────────────

  useEffect(() => {
    if (!canvasRef.current) return

    const container = canvasRef.current
    const w = container.clientWidth
    const h = container.clientHeight

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(window.devicePixelRatio)
    renderer.setSize(w, h)
    renderer.setClearColor(0x0d0d0f, 1)
    renderer.toneMapping        = THREE.ACESFilmicToneMapping
    renderer.toneMappingExposure = 1.3
    container.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const scene = new THREE.Scene()
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(58, w / h, 0.01, 500)
    cameraRef.current = camera

    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate)

      if (flyModeRef.current) {
        // Free-fly movement
        const { yaw, pitch, pos, keys, speed } = flyRef.current
        const cosP = Math.cos(pitch), sinP = Math.sin(pitch)
        const sinY = Math.sin(yaw),   cosY = Math.cos(yaw)
        const forward = new THREE.Vector3(cosP * sinY, sinP, cosP * cosY)
        const right   = new THREE.Vector3(Math.sin(yaw - Math.PI / 2), 0, Math.cos(yaw - Math.PI / 2))
        if (keys.has('w') || keys.has('arrowup'))    pos.addScaledVector(forward,  speed)
        if (keys.has('s') || keys.has('arrowdown'))  pos.addScaledVector(forward, -speed)
        if (keys.has('a') || keys.has('arrowleft'))  pos.addScaledVector(right,   -speed)
        if (keys.has('d') || keys.has('arrowright')) pos.addScaledVector(right,    speed)
        if (keys.has('e'))                           pos.y += speed
        if (keys.has('q'))                           pos.y -= speed
        camera.position.copy(pos)
        camera.lookAt(pos.clone().add(forward))
      } else {
        // Orbit auto-rotate
        if (!orbitRef.current.dragging && autoRotateRef.current) {
          orbitRef.current.theta += 0.004
          cameraUpdateRef.current()
        }
      }

      // Animate signal particles along connections
      for (const particle of particlesRef.current) {
        const { startZ, endZ, speed, xPhase, yPhase } = particle.userData
        particle.userData.progress = (particle.userData.progress + speed) % 1
        const t = easeInOut(particle.userData.progress)
        particle.position.z = startZ + (endZ - startZ) * t
        particle.position.x = Math.sin(particle.userData.progress * Math.PI * 2 + xPhase) * 0.08
        particle.position.y = Math.sin(particle.userData.progress * Math.PI     + yPhase) * 0.12
      }

      renderer.render(scene, camera)
    }
    animate()

    // Resize observer
    const ro = new ResizeObserver(() => {
      const nw = container.clientWidth
      const nh = container.clientHeight
      renderer.setSize(nw, nh)
      camera.aspect = nw / nh
      camera.updateProjectionMatrix()
    })
    ro.observe(container)

    return () => {
      cancelAnimationFrame(animFrameRef.current)
      ro.disconnect()
      renderer.dispose()
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement)
      }
    }
  }, [])

  // Rebuild scene when layers / input size change
  useEffect(() => {
    if (!sceneRef.current || !cameraRef.current) return
    const activations = viewMode === 'image' ? buildActivationSequence(layers, numClasses) : null
    buildScene(sceneRef.current, layers, inputH, inputW, particlesRef, numClasses, viewMode, activations)
    updateCamera()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layers, inputH, inputW, numClasses, viewMode])

  // ── Camera orbit ──────────────────────────────────────────────────────────

  const updateCamera = useCallback(() => {
    const camera = cameraRef.current
    if (!camera) return
    const { theta, phi, radius } = orbitRef.current
    const totalSlots = layers.length + 2  // +1 input, +1 output head
    const target = new THREE.Vector3(0, 0, -(totalSlots - 1) * SPACING / 2)
    camera.position.set(
      target.x + radius * Math.sin(phi) * Math.sin(theta),
      target.y + radius * Math.cos(phi),
      target.z + radius * Math.sin(phi) * Math.cos(theta),
    )
    camera.lookAt(target)
  }, [layers.length])

  // ── Free-fly camera ────────────────────────────────────────────────────────

  const enterFlyMode = useCallback(() => {
    const cam = cameraRef.current
    if (!cam) return
    // Sync position + orientation from current orbit camera
    flyRef.current.pos.copy(cam.position)
    const dir = new THREE.Vector3()
    cam.getWorldDirection(dir)
    flyRef.current.yaw   = Math.atan2(dir.x, dir.z)
    flyRef.current.pitch = Math.asin(Math.max(-1, Math.min(1, dir.y)))
    flyRef.current.keys.clear()
    flyModeRef.current = true
    setFlyMode(true)
    canvasRef.current?.requestPointerLock()
  }, [])

  const exitFlyMode = useCallback(() => {
    flyModeRef.current = false
    flyRef.current.keys.clear()
    setFlyMode(false)
    if (document.pointerLockElement) document.exitPointerLock()
  }, [])

  // Fly mode: keyboard + pointer-lock mouse look
  useEffect(() => {
    if (!flyMode) return

    const onKeyDown = (e: KeyboardEvent) => {
      flyRef.current.keys.add(e.key.toLowerCase())
      if (e.key === 'Escape') exitFlyMode()
      // Prevent WASD from scrolling the page
      if (['w','a','s','d','q','e','arrowup','arrowdown','arrowleft','arrowright'].includes(e.key.toLowerCase()))
        e.preventDefault()
    }
    const onKeyUp = (e: KeyboardEvent) => flyRef.current.keys.delete(e.key.toLowerCase())

    const onMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement !== canvasRef.current) return
      flyRef.current.yaw  -= e.movementX * 0.002
      flyRef.current.pitch = Math.max(-Math.PI / 2 + 0.05,
        Math.min(Math.PI / 2 - 0.05, flyRef.current.pitch - e.movementY * 0.002))
    }

    const onPointerLockChange = () => {
      if (!document.pointerLockElement) exitFlyMode()
    }

    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      flyRef.current.speed = Math.max(0.02, Math.min(1.5, flyRef.current.speed * (e.deltaY > 0 ? 1.15 : 0.87)))
    }

    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    window.addEventListener('mousemove', onMouseMove)
    document.addEventListener('pointerlockchange', onPointerLockChange)
    canvasRef.current?.addEventListener('wheel', onWheel, { passive: false })

    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
      window.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('pointerlockchange', onPointerLockChange)
      canvasRef.current?.removeEventListener('wheel', onWheel)
      flyRef.current.keys.clear()
    }
  }, [flyMode, exitFlyMode])

  // 2D diagram: redraw whenever layers, input size, mode, or view switches to 2D
  useEffect(() => {
    if (viewDim !== '2d' || !canvas2DRef.current) return
    const c   = canvas2DRef.current
    const par = c.parentElement!
    const w   = par.clientWidth  || 800
    const h   = par.clientHeight || 420
    c.width   = w
    c.height  = h
    if (w > 0 && h > 0) {
      const activations = viewMode === 'image' ? buildActivationSequence(layers, numClasses) : null
      draw2DScene(c, layers, inputH, inputW, numClasses, activations)
    }
  }, [viewDim, layers, inputH, inputW, numClasses, viewMode])

  // Keep cameraUpdateRef current so the animation loop can call it without stale closure
  useEffect(() => {
    cameraUpdateRef.current = updateCamera
  }, [updateCamera])

  // Re-run when layers.length changes
  useEffect(() => { updateCamera() }, [layers.length, updateCamera])

  // ── Mouse orbit handlers ──────────────────────────────────────────────────

  useEffect(() => {
    const container = canvasRef.current
    if (!container) return

    const onDown = (e: MouseEvent) => {
      if (flyModeRef.current) return
      orbitRef.current.dragging = true
      orbitRef.current.lastX    = e.clientX
      orbitRef.current.lastY    = e.clientY
    }
    const onMove = (e: MouseEvent) => {
      if (flyModeRef.current || !orbitRef.current.dragging) return
      const dx = e.clientX - orbitRef.current.lastX
      const dy = e.clientY - orbitRef.current.lastY
      orbitRef.current.lastX  = e.clientX
      orbitRef.current.lastY  = e.clientY
      orbitRef.current.theta -= dx * 0.005
      orbitRef.current.phi    = Math.max(0.1, Math.min(Math.PI - 0.1, orbitRef.current.phi + dy * 0.005))
      updateCamera()
    }
    const onUp   = () => { orbitRef.current.dragging = false }
    const onWheel = (e: WheelEvent) => {
      e.preventDefault()
      orbitRef.current.radius = Math.max(3, Math.min(25, orbitRef.current.radius + e.deltaY * 0.01))
      updateCamera()
    }

    container.addEventListener('mousedown', onDown)
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    container.addEventListener('wheel', onWheel, { passive: false })

    return () => {
      container.removeEventListener('mousedown', onDown)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
      container.removeEventListener('wheel', onWheel)
    }
  }, [updateCamera])

  // ── Log auto-scroll ───────────────────────────────────────────────────────

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight
    }
  }, [logs])

  // ── Training ──────────────────────────────────────────────────────────────

  const startTraining = async () => {
    if (!savedConfig) {
      alert('Config is still saving, please wait a moment.')
      return
    }
    setLogs([])
    try {
      const res = await api.post(`/projects/${projectId}/custom/runs`, {
        config_id: savedConfig.id,
        epochs,
        batch,
        lr,
      })
      const run: CustomRun = res.data
      setActiveRun(run)
      openSSE(run.id)
    } catch (e: unknown) {
      const msg = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? String(e)
      setLogs([`Error: ${msg}`])
    }
  }

  const openSSE = (runId: number) => {
    sseRef.current?.close()
    const es = new EventSource(`/api/projects/${projectId}/custom/runs/${runId}/stream`)
    sseRef.current = es
    es.onmessage = (e) => {
      const data = e.data as string
      if (data === '__END__') {
        es.close()
        // Refresh run status
        api.get(`/projects/${projectId}/custom/runs/${runId}`)
          .then(r => setActiveRun(r.data))
          .catch(() => {})
        return
      }
      if (data.startsWith('__PROGRESS__:')) {
        const parts = data.split(':')
        const [ep, total] = parts[1].split('/')
        const acc = parseFloat(parts[2])
        setActiveRun(prev => prev ? { ...prev, status: 'running' } : prev)
        setLogs(prev => [...prev.filter(l => !l.startsWith('[PROG]')),
          `[PROG] Epoch ${ep}/${total} — acc ${(acc * 100).toFixed(1)}%`])
        return
      }
      if (data === '__FAILED__') {
        setActiveRun(prev => prev ? { ...prev, status: 'failed' } : prev)
        return
      }
      if (data.startsWith('__DONE__:')) {
        setActiveRun(prev => prev ? { ...prev, status: 'done' } : prev)
        return
      }
      setLogs(prev => [...prev, data])
    }
    es.onerror = () => {
      es.close()
      setLogs(prev => [...prev, 'Stream disconnected'])
    }
  }

  // Cleanup SSE on unmount
  useEffect(() => () => { sseRef.current?.close() }, [])

  // ── Layer editor helpers ──────────────────────────────────────────────────

  const addLayer = (type: LayerType) => {
    const layer: Layer = { id: uid(), type, params: { ...DEFAULT_PARAMS[type] } }
    setLayers(prev => [...prev, layer])
    setSelectedId(layer.id)
    setShowAddMenu(false)
  }

  const deleteLayer = (layerId: string) => {
    setLayers(prev => prev.filter(l => l.id !== layerId))
    setSelectedId(prev => prev === layerId ? null : prev)
  }

  const moveLayer = (index: number, dir: -1 | 1) => {
    setLayers(prev => {
      const next = [...prev]
      const target = index + dir
      if (target < 0 || target >= next.length) return prev
      ;[next[index], next[target]] = [next[target], next[index]]
      return next
    })
  }

  const updateLayerParams = (layerId: string, params: Record<string, number>) => {
    setLayers(prev => prev.map(l => l.id === layerId ? { ...l, params } : l))
  }

  // Compute shapes for all layers (for display)
  const allShapes = computeAllShapes(layers, inputH, inputW)
  // allShapes[0] = input, allShapes[i+1] = output of layers[i]
  const totalParams = estimateParams(layers, inputH, inputW)

  const selectedLayer = layers.find(l => l.id === selectedId) ?? null

  // Status badge color
  const statusColor = (s: string): 'green' | 'yellow' | 'red' | 'gray' | 'blue' => {
    if (s === 'done')    return 'green'
    if (s === 'running') return 'blue'
    if (s === 'failed')  return 'red'
    if (s === 'pending') return 'yellow'
    return 'gray'
  }

  // ── Render ────────────────────────────────────────────────────────────────

  const panelStyle = {
    background: 'var(--surface)',
    border: '1px solid var(--border)',
    borderRadius: 8,
    display: 'flex',
    flexDirection: 'column' as const,
    overflow: 'hidden',
  }

  const sectionLabel = (text: string) => (
    <p style={{
      fontSize: 10, fontWeight: 600, letterSpacing: '0.08em',
      textTransform: 'uppercase' as const, color: 'var(--text3)',
      padding: '10px 12px 4px', flexShrink: 0,
    }}>{text}</p>
  )

  return (
    <div style={{ maxWidth: '100%', height: 'calc(100vh - 80px)', display: 'flex', flexDirection: 'column' }}>
      <PageHeader
        back={() => navigate(`/projects/${projectId}/images`)}
        title={`Conv Builder — ${projectName}`}
        subtitle="Build a custom CNN architecture and train it"
        actions={
          <Btn variant="ghost" size="sm" onClick={() => navigate(`/projects/${projectId}/images`)}>
            <Cpu size={13} /> Back to images
          </Btn>
        }
      />

      <div style={{ flex: 1, display: 'flex', gap: 8, overflow: 'hidden', minHeight: 0 }}>

        {/* ── Left panel: Layer Editor ───────────────────────────────────── */}
        <div style={{ ...panelStyle, width: 220, flexShrink: 0 }}>
          {sectionLabel('Model')}

          {/* Model name */}
          <div style={{ padding: '0 10px 8px' }}>
            <input
              value={modelName}
              onChange={e => setModelName(e.target.value)}
              placeholder="Model name"
              style={{
                width: '100%', padding: '5px 8px',
                background: 'var(--surface2)', border: '1px solid var(--border)',
                borderRadius: 5, color: 'var(--text)', fontSize: 12,
                fontFamily: 'inherit',
              }}
            />
          </div>

          {/* Input size */}
          {sectionLabel('Input size')}
          <div style={{ padding: '0 10px 8px', display: 'flex', gap: 6 }}>
            {([['H', inputH, setInputH], ['W', inputW, setInputW]] as const).map(([lbl, val, setter]) => (
              <div key={lbl} style={{ flex: 1 }}>
                <label style={{ fontSize: 10, color: 'var(--text3)', display: 'block', marginBottom: 3 }}>{lbl}</label>
                <input
                  type="number" min={32} max={256} step={32}
                  value={val}
                  onChange={e => setter(Number(e.target.value))}
                  style={{
                    width: '100%', padding: '4px 6px',
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 4, color: 'var(--text)', fontSize: 12,
                  }}
                />
              </div>
            ))}
          </div>

          {sectionLabel('Layers')}

          {/* Layer list */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '0 6px' }}>
            {/* Input node */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '5px 6px', borderRadius: 5, marginBottom: 2,
              background: 'var(--surface2)',
            }}>
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: LAYER_CSS.input, flexShrink: 0 }} />
              <span style={{ fontSize: 11, color: 'var(--text2)', flex: 1 }}>input</span>
              <span style={{ fontSize: 10, color: 'var(--text3)', fontFamily: 'monospace' }}>
                {`${3}×${inputH}×${inputW}`}
              </span>
            </div>

            {layers.map((layer, index) => {
              const outShape = allShapes[index + 1]
              const valid    = isValidShape(outShape)
              const isSelected = layer.id === selectedId
              const shapeStr  = !valid ? '⚠ invalid'
                : outShape.length === 1 ? `${outShape[0]}`
                : `${outShape[0]}×${outShape[1]}×${outShape[2]}`

              return (
                <div
                  key={layer.id}
                  onClick={() => setSelectedId(isSelected ? null : layer.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: 5,
                    padding: '5px 6px', borderRadius: 5, marginBottom: 2,
                    cursor: 'pointer',
                    background: isSelected ? 'rgba(88,101,242,0.12)' : 'transparent',
                    border: `1px solid ${isSelected ? 'var(--accent)' : 'transparent'}`,
                    transition: 'background 0.1s',
                  }}
                >
                  <span style={{
                    width: 8, height: 8, borderRadius: '50%', flexShrink: 0,
                    background: valid ? (LAYER_CSS[layer.type] ?? '#8b8b9a') : '#ef4444',
                  }} />
                  <span style={{ fontSize: 11, color: valid ? 'var(--text)' : 'var(--danger)', flex: 1, minWidth: 0 }}>
                    {LAYER_DISPLAY[layer.type] ?? layer.type}
                  </span>
                  <span style={{ fontSize: 9, color: valid ? 'var(--text3)' : 'var(--danger)', fontFamily: 'monospace', maxWidth: 55, textAlign: 'right' }}>
                    {shapeStr}
                  </span>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                    <button
                      onClick={e => { e.stopPropagation(); moveLayer(index, -1) }}
                      disabled={index === 0}
                      style={{ fontSize: 8, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '1px 2px', lineHeight: 1 }}
                    >▲</button>
                    <button
                      onClick={e => { e.stopPropagation(); moveLayer(index, 1) }}
                      disabled={index === layers.length - 1}
                      style={{ fontSize: 8, color: 'var(--text3)', background: 'none', border: 'none', cursor: 'pointer', padding: '1px 2px', lineHeight: 1 }}
                    >▼</button>
                  </div>
                  <button
                    onClick={e => { e.stopPropagation(); deleteLayer(layer.id) }}
                    style={{ fontSize: 10, color: 'var(--danger)', background: 'none', border: 'none', cursor: 'pointer', padding: '0 2px', lineHeight: 1 }}
                  >✕</button>
                </div>
              )
            })}
          </div>

          {/* Param editor for selected layer */}
          {selectedLayer && (
            <div style={{ borderTop: '1px solid var(--border)', padding: '8px 10px', flexShrink: 0 }}>
              {/* Educational description card */}
              {LAYER_DESCRIPTIONS[selectedLayer.type] && (() => {
                const desc = LAYER_DESCRIPTIONS[selectedLayer.type]
                const color = LAYER_CSS[selectedLayer.type] ?? 'var(--accent)'
                return (
                  <div style={{
                    background: 'rgba(255,255,255,0.03)',
                    border: `1px solid ${color}40`,
                    borderLeft: `3px solid ${color}`,
                    borderRadius: 5, padding: '8px 10px', marginBottom: 10,
                  }}>
                    <p style={{ fontSize: 11, color, fontWeight: 700, marginBottom: 5 }}>{desc.label}</p>
                    <p style={{ fontSize: 11, color: 'var(--text2)', lineHeight: 1.55, marginBottom: 5 }}>{desc.what}</p>
                    <p style={{ fontSize: 10, color: 'var(--text3)', lineHeight: 1.5, marginBottom: 5 }}>
                      <span style={{ color: 'var(--text2)', fontWeight: 600 }}>Why: </span>{desc.why}
                    </p>
                    <p style={{ fontSize: 10, color: 'var(--text3)', lineHeight: 1.5 }}>{desc.analogy}</p>
                  </div>
                )
              })()}
              <p style={{ fontSize: 10, color: 'var(--text3)', marginBottom: 6, fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase' }}>
                {LAYER_DISPLAY[selectedLayer.type] ?? selectedLayer.type} params
              </p>
              <ParamEditor
                layer={selectedLayer}
                onChange={params => updateLayerParams(selectedLayer.id, params)}
              />
            </div>
          )}

          {/* Total params */}
          <div style={{ borderTop: '1px solid var(--border)', padding: '6px 10px', flexShrink: 0 }}>
            <p style={{ fontSize: 10, color: 'var(--text3)' }}>
              Est. params: <span style={{ color: 'var(--text)', fontFamily: 'monospace' }}>{totalParams.toLocaleString()}</span>
            </p>
          </div>

          {/* Add layer button */}
          <div style={{ padding: '8px 10px', borderTop: '1px solid var(--border)', flexShrink: 0, position: 'relative' }}>
            <button
              onClick={() => setShowAddMenu(prev => !prev)}
              style={{
                width: '100%', padding: '6px', borderRadius: 5,
                background: 'var(--surface2)', border: '1px solid var(--border2)',
                color: 'var(--text2)', fontSize: 12, cursor: 'pointer',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 4,
              }}
            >
              + Add Layer
            </button>
            {showAddMenu && (
              <div style={{
                position: 'absolute', bottom: '100%', left: 10, right: 10,
                background: 'var(--surface2)', border: '1px solid var(--border2)',
                borderRadius: 6, overflow: 'hidden', zIndex: 100,
              }}>
                {/* Hover preview card */}
                {hoveredAddType && LAYER_DESCRIPTIONS[hoveredAddType] && (
                  <div style={{
                    padding: '8px 10px', borderBottom: '1px solid var(--border)',
                    background: 'var(--surface3)',
                  }}>
                    <p style={{ fontSize: 10, fontWeight: 700, color: LAYER_CSS[hoveredAddType] ?? 'var(--accent)', marginBottom: 3 }}>
                      {LAYER_DESCRIPTIONS[hoveredAddType].label}
                    </p>
                    <p style={{ fontSize: 10, color: 'var(--text3)', lineHeight: 1.5 }}>
                      {LAYER_DESCRIPTIONS[hoveredAddType].what}
                    </p>
                    <p style={{ fontSize: 10, color: 'var(--text3)', marginTop: 3, lineHeight: 1.4 }}>
                      {LAYER_DESCRIPTIONS[hoveredAddType].analogy}
                    </p>
                  </div>
                )}
                {ALL_LAYER_TYPES.map(lt => (
                  <button
                    key={lt}
                    onClick={() => addLayer(lt)}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 7,
                      width: '100%', padding: '6px 10px',
                      background: 'transparent', border: 'none',
                      color: 'var(--text)', fontSize: 12, cursor: 'pointer',
                      textAlign: 'left',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.background = 'var(--surface3)'; setHoveredAddType(lt) }}
                    onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; setHoveredAddType(null) }}
                  >
                    <span style={{ width: 7, height: 7, borderRadius: '50%', background: LAYER_CSS[lt] ?? '#8b8b9a', flexShrink: 0 }} />
                    {LAYER_DISPLAY[lt] ?? lt}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* ── Center: Canvas (3D or 2D) ───────────────────────────────────── */}
        <div style={{ flex: 1, position: 'relative', borderRadius: 8, overflow: 'hidden', border: '1px solid var(--border)', background: '#0d0d0f' }}>
          <div ref={canvasRef} style={{ width: '100%', height: '100%', cursor: 'grab', display: viewDim === '3d' ? 'block' : 'none' }} />
          <canvas ref={canvas2DRef} style={{ width: '100%', height: '100%', display: viewDim === '2d' ? 'block' : 'none' }} />

          {/* Canvas controls */}
          <div style={{ position: 'absolute', top: 10, right: 10, display: 'flex', gap: 6 }}>
            <button
              onClick={() => setViewDim(v => v === '3d' ? '2d' : '3d')}
              style={{
                padding: '4px 10px', borderRadius: 5,
                background: viewDim === '2d' ? 'rgba(168,85,247,0.25)' : 'rgba(15,15,18,0.85)',
                border: `1px solid ${viewDim === '2d' ? '#a855f7' : 'var(--border2)'}`,
                color: viewDim === '2d' ? '#a855f7' : 'var(--text2)',
                fontSize: 11, cursor: 'pointer', backdropFilter: 'blur(4px)',
              }}
            >
              {viewDim === '3d' ? '▦ 2D View' : '⬡ 3D View'}
            </button>
            <button
              onClick={() => setViewMode(v => v === 'color' ? 'image' : 'color')}
              style={{
                padding: '4px 10px', borderRadius: 5,
                background: viewMode === 'image' ? 'rgba(251,146,60,0.25)' : 'rgba(15,15,18,0.85)',
                border: `1px solid ${viewMode === 'image' ? '#fb923c' : 'var(--border2)'}`,
                color: viewMode === 'image' ? '#fb923c' : 'var(--text2)',
                fontSize: 11, cursor: 'pointer', backdropFilter: 'blur(4px)',
              }}
              title="Toggle between color blocks and educational activation textures"
            >
              {viewMode === 'image' ? '◉ Activations' : '◎ Activations'}
            </button>
            {viewDim === '3d' && <>
              <button
                onClick={() => { const n = !autoRotate; setAutoRotate(n); autoRotateRef.current = n }}
                disabled={flyMode}
                style={{
                  padding: '4px 10px', borderRadius: 5,
                  background: autoRotate && !flyMode ? 'rgba(88,101,242,0.25)' : 'rgba(15,15,18,0.85)',
                  border: `1px solid ${autoRotate && !flyMode ? 'var(--accent)' : 'var(--border2)'}`,
                  color: autoRotate && !flyMode ? 'var(--accent)' : 'var(--text2)',
                  fontSize: 11, cursor: flyMode ? 'default' : 'pointer',
                  backdropFilter: 'blur(4px)', opacity: flyMode ? 0.4 : 1,
                }}
              >
                {autoRotate ? '⟳ Auto-rotate' : '⟳ Rotate off'}
              </button>
              <button
                onClick={() => flyMode ? exitFlyMode() : enterFlyMode()}
                style={{
                  padding: '4px 10px', borderRadius: 5,
                  background: flyMode ? 'rgba(34,197,94,0.25)' : 'rgba(15,15,18,0.85)',
                  border: `1px solid ${flyMode ? '#22c55e' : 'var(--border2)'}`,
                  color: flyMode ? '#22c55e' : 'var(--text2)',
                  fontSize: 11, cursor: 'pointer', backdropFilter: 'blur(4px)',
                }}
                title="Free fly — WASD move · mouse look · Q/E up/down · scroll speed · Esc exit"
              >
                {flyMode ? '✈ Exit Fly' : '✈ Fly Mode'}
              </button>
              <button
                onClick={() => { if (flyMode) exitFlyMode(); orbitRef.current = { ...orbitRef.current, theta: 0.45, phi: 0.88, radius: 11 }; updateCamera() }}
                style={{
                  padding: '4px 10px', borderRadius: 5,
                  background: 'rgba(15,15,18,0.85)', border: '1px solid var(--border2)',
                  color: 'var(--text2)', fontSize: 11, cursor: 'pointer', backdropFilter: 'blur(4px)',
                }}
              >
                Reset View
              </button>
            </>}
          </div>

          {/* Help hint */}
          <div style={{
            position: 'absolute', bottom: 10, left: '50%', transform: 'translateX(-50%)',
            fontSize: 10, color: 'rgba(140,140,158,0.5)',
            pointerEvents: 'none',
          }}>
            {viewDim === '2d' ? 'Architecture diagram — 2D view'
              : flyMode ? 'WASD move · mouse look · Q/E up/down · scroll speed · Esc exit'
              : 'Drag to orbit · Scroll to zoom · ✈ Fly Mode to move freely'}
          </div>
        </div>

        {/* ── Right panel: Training ───────────────────────────────────────── */}
        <div style={{ ...panelStyle, width: 260, flexShrink: 0 }}>
          {sectionLabel('Training')}

          <div style={{ padding: '0 12px 10px', display: 'flex', flexDirection: 'column', gap: 8 }}>
            {[
              { label: 'Epochs', val: epochs, set: setEpochs, min: 1,   max: 200, step: 1    },
              { label: 'Batch',  val: batch,  set: setBatch,  min: 4,   max: 128, step: 4    },
            ].map(({ label, val, set, min, max, step }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <label style={{ fontSize: 12, color: 'var(--text2)', width: 50, flexShrink: 0 }}>{label}</label>
                <input
                  type="number" min={min} max={max} step={step}
                  value={val}
                  onChange={e => set(Number(e.target.value))}
                  style={{
                    flex: 1, padding: '5px 8px',
                    background: 'var(--surface2)', border: '1px solid var(--border)',
                    borderRadius: 5, color: 'var(--text)', fontSize: 12,
                  }}
                />
              </div>
            ))}
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <label style={{ fontSize: 12, color: 'var(--text2)', width: 50, flexShrink: 0 }}>LR</label>
              <input
                type="number" min={0.00001} max={0.1} step={0.0001}
                value={lr}
                onChange={e => setLr(Number(e.target.value))}
                style={{
                  flex: 1, padding: '5px 8px',
                  background: 'var(--surface2)', border: '1px solid var(--border)',
                  borderRadius: 5, color: 'var(--text)', fontSize: 12,
                }}
              />
            </div>
          </div>

          <div style={{ padding: '0 12px 10px' }}>
            <Btn
              variant="primary"
              onClick={startTraining}
              disabled={activeRun?.status === 'running' || activeRun?.status === 'pending'}
              style={{ width: '100%', justifyContent: 'center' }}
            >
              {activeRun?.status === 'running' ? 'Training…' : 'Train'}
            </Btn>
          </div>

          {/* Status */}
          {activeRun && (
            <div style={{ padding: '0 12px 10px', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 12, color: 'var(--text2)' }}>Status:</span>
              <Badge color={statusColor(activeRun.status)}>{activeRun.status}</Badge>
            </div>
          )}

          {sectionLabel('Logs')}

          {/* Log terminal */}
          <div
            ref={logRef}
            style={{
              flex: 1, overflowY: 'auto', margin: '0 10px 10px',
              background: '#07070a', border: '1px solid var(--border)',
              borderRadius: 5, padding: '8px 10px',
              fontFamily: 'JetBrains Mono, monospace', fontSize: 11, lineHeight: 1.65,
              minHeight: 0,
            }}
          >
            {logs.length === 0
              ? <span style={{ color: 'var(--text3)' }}>Waiting…</span>
              : logs.map((l, i) => (
                <div
                  key={i}
                  style={{
                    color: l.startsWith('Error') || l.includes('❌') ? 'var(--danger)'
                      : l.startsWith('[PROG]') ? 'var(--accent)'
                      : l.includes('Done') || l.includes('✅') ? 'var(--success)'
                      : 'var(--text2)',
                  }}
                >
                  {l}
                </div>
              ))}
          </div>
        </div>
      </div>
    </div>
  )
}
