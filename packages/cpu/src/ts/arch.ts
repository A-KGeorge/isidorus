/**
 * arch.ts — Architecture building blocks for frozen graph construction.
 *
 * Two layers:
 *
 * 1. Weight buffer generators (pure JS, no graph dependency):
 *      glorotBuffer, heNormalBuffer, zerosBuffer, onesBuffer
 *
 * 2. Frozen graph helpers (accept a Graph, return Tensors):
 *      constGlorot, constHe, constZeros, constOnes
 *      frozenBatchNorm
 *      convBnRelu, convBnRelu6
 *      residualBlock         — bottleneck identity block (ResNet-style)
 *      projectionBlock       — bottleneck block with projection shortcut
 *      invertedResidual      — expansion + depthwise + projection (MobileNet-style)
 *
 * These complement the trainable Layer API (Sequential, Dense, Conv2D, ...) for
 * cases where you want to build a frozen inference graph directly — custom
 * architectures, pre-trained weights baked in, no Python conversion needed.
 *
 * @example
 * import { Graph }                from "@isidorus/cpu";
 * import { constHe, constZeros, convBnRelu, residualBlock } from "@isidorus/cpu/arch";
 *
 * const g  = new Graph(...);
 * const [x] = g.addOp("Placeholder", [], { dtype: ..., shape: ... }, "inputs");
 * let [h, c] = convBnRelu(g, x, 3, 64, 7, 2, "SAME", "conv1");
 * [h, c]    = residualBlock(g, h, 64, [64, 64, 256], "res2a");
 * // ... etc
 */

import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "./graph.js";
import { constant } from "./ops/array_ops.js";
import {
  conv2d,
  depthwiseConv2d,
  maxPool,
  globalAvgPool,
  batchNorm,
  relu,
  relu6,
} from "./ops/nn_ops.js";
import { add, biasAdd } from "./ops/math_ops.js";

// ---------------------------------------------------------------------------
// Weight buffer generators — pure JS, no graph dependency.
//
// These produce flat float32 Buffers with the appropriate statistical
// distribution for each initializer. Use them with constant() to bake
// pre-initialised weights directly into a frozen graph.
// ---------------------------------------------------------------------------

/** Glorot uniform: samples from Uniform(−limit, +limit) where limit = √(6/(fanIn+fanOut)). */
export function glorotBuffer(shape: number[]): Buffer {
  const n = shape.reduce((a, b) => a * b, 1);
  const fanIn = shape.length >= 2 ? shape[shape.length - 2] : shape[0];
  const fanOut = shape[shape.length - 1];
  // For conv kernels [kH, kW, inC, outC] use receptive-field-scaled fans.
  const receptive =
    shape.length > 2 ? shape.slice(0, -2).reduce((a, b) => a * b, 1) : 1;
  const limit = Math.sqrt(6 / (fanIn * receptive + fanOut * receptive));
  const buf = Buffer.allocUnsafe(n * 4);
  for (let i = 0; i < n; i++)
    buf.writeFloatLE((Math.random() * 2 - 1) * limit, i * 4);
  return buf;
}

/**
 * He normal: samples from N(0, √(2/fanIn)).
 * Preferred for conv layers followed by ReLU.
 */
export function heNormalBuffer(shape: number[]): Buffer {
  const n = shape.reduce((a, b) => a * b, 1);
  const fanIn = shape.slice(0, -1).reduce((a, b) => a * b, 1);
  const std = Math.sqrt(2 / fanIn);
  const buf = Buffer.allocUnsafe(n * 4);
  // Box-Muller transform for normally distributed samples.
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.random() || 1e-10; // guard against log(0)
    const u2 = Math.random();
    const r = std * Math.sqrt(-2 * Math.log(u1));
    buf.writeFloatLE(r * Math.cos(2 * Math.PI * u2), i * 4);
    if (i + 1 < n)
      buf.writeFloatLE(r * Math.sin(2 * Math.PI * u2), (i + 1) * 4);
  }
  return buf;
}

/** Zeros buffer: n float32 values all set to 0. Useful for bias init. */
export function zerosBuffer(n: number): Buffer {
  return Buffer.alloc(n * 4); // alloc zero-fills
}

/** Ones buffer: n float32 values all set to 1. Useful for BN gamma init. */
export function onesBuffer(n: number): Buffer {
  const buf = Buffer.allocUnsafe(n * 4);
  for (let i = 0; i < n; i++) buf.writeFloatLE(1.0, i * 4);
  return buf;
}

// ---------------------------------------------------------------------------
// Const op shortcuts — frozen graph weight initialisation.
//
// Each function generates a weight buffer and embeds it as a Const op
// in the graph. The name is required so the frozen graph has legible op names
// and weights can be identified during debugging.
// ---------------------------------------------------------------------------

/** Add a Glorot-uniform initialised weight as a Const op. */
export function constGlorot(g: Graph, shape: number[], name: string): Tensor {
  return constant(g, glorotBuffer(shape), shape, DType.FLOAT32, name);
}

/** Add a He-normal initialised weight as a Const op. */
export function constHe(g: Graph, shape: number[], name: string): Tensor {
  return constant(g, heNormalBuffer(shape), shape, DType.FLOAT32, name);
}

/** Add a zeros Const op with shape [n] (bias init, BN beta init). */
export function constZeros(g: Graph, n: number, name: string): Tensor {
  return constant(g, zerosBuffer(n), [n], DType.FLOAT32, name);
}

/** Add a ones Const op with shape [n] (BN gamma init). */
export function constOnes(g: Graph, n: number, name: string): Tensor {
  return constant(g, onesBuffer(n), [n], DType.FLOAT32, name);
}

// ---------------------------------------------------------------------------
// frozenBatchNorm — batch normalisation with all parameters as Const ops.
//
// Applies: y = gamma * (x - mean) / sqrt(var + epsilon) + beta
// via FusedBatchNormV3 with is_training=false.
//
// All four parameters (gamma, beta, mean, var) are initialised as constants
// using the standard identity initialisation (gamma=1, beta=0, mean=0, var=1)
// unless you pass pre-trained buffers explicitly.
//
// This is the frozen-graph equivalent of the trainable BatchNormalization layer.
// Use when building a frozen graph from scratch (e.g. for InferencePool) rather
// than training — for training, use the BatchNormalization Layer instead.
// ---------------------------------------------------------------------------
export function frozenBatchNorm(
  g: Graph,
  x: Tensor,
  channels: number,
  name: string,
  options: {
    gamma?: Buffer; // [channels] float32 — default ones
    beta?: Buffer; // [channels] float32 — default zeros
    mean?: Buffer; // [channels] float32 — default zeros
    variance?: Buffer; // [channels] float32 — default ones
    epsilon?: number;
  } = {},
): Tensor {
  const gamma = constant(
    g,
    options.gamma ?? onesBuffer(channels),
    [channels],
    DType.FLOAT32,
    `${name}/gamma`,
  );
  const beta = constant(
    g,
    options.beta ?? zerosBuffer(channels),
    [channels],
    DType.FLOAT32,
    `${name}/beta`,
  );
  const mean = constant(
    g,
    options.mean ?? zerosBuffer(channels),
    [channels],
    DType.FLOAT32,
    `${name}/mean`,
  );
  const vari = constant(
    g,
    options.variance ?? onesBuffer(channels),
    [channels],
    DType.FLOAT32,
    `${name}/var`,
  );
  return batchNorm(
    g,
    x,
    gamma,
    beta,
    mean,
    vari,
    { epsilon: options.epsilon },
    name,
  );
}

// ---------------------------------------------------------------------------
// Common CNN patterns — functional building blocks for frozen graph construction.
//
// All functions follow the same convention:
//   - First args: g (Graph), input (Tensor), channel counts
//   - Last arg:   name (string) — used as op name prefix; all sub-ops are
//                 named `${name}/…` for legible frozen graphs
//   - Return:     [output Tensor, output channel count]
//
// The [Tensor, number] return makes it easy to chain blocks where the channel
// count feeds into the next block's inC argument:
//   let [h, c] = convBnRelu(g, x, 3, 64, 7, 2, "SAME", "conv1");
//   [h, c]    = convBnRelu(g, h, c, 128, 3, 2, "SAME", "conv2");
// ---------------------------------------------------------------------------

/**
 * convBnRelu — Conv2D → BatchNorm → ReLU.
 *
 * The standard building block for ResNet-style architectures.
 * Uses He-normal initialisation for the conv kernel (appropriate for ReLU).
 *
 * @param inC      Input channels
 * @param outC     Output channels
 * @param kSize    Kernel size (square)
 * @param stride   Stride (square)
 * @param padding  "SAME" or "VALID"
 */
export function convBnRelu(
  g: Graph,
  input: Tensor,
  inC: number,
  outC: number,
  kSize: number,
  stride: number,
  padding: "SAME" | "VALID",
  name: string,
): [Tensor, number] {
  const w = constHe(g, [kSize, kSize, inC, outC], `${name}/w`);
  const conv = conv2d(
    g,
    input,
    w,
    [1, stride, stride, 1],
    padding,
    `${name}/conv`,
  );
  const bn = frozenBatchNorm(g, conv, outC, `${name}/bn`);
  return [relu(g, bn, `${name}/relu`), outC];
}

/**
 * convBnRelu6 — Conv2D → BatchNorm → ReLU6.
 *
 * Used in MobileNet-family architectures where activations are clipped to [0, 6]
 * to improve fixed-point quantisation friendliness.
 */
export function convBnRelu6(
  g: Graph,
  input: Tensor,
  inC: number,
  outC: number,
  kSize: number,
  stride: number,
  name: string,
): [Tensor, number] {
  const w = constHe(g, [kSize, kSize, inC, outC], `${name}/w`);
  const conv = conv2d(
    g,
    input,
    w,
    [1, stride, stride, 1],
    "SAME",
    `${name}/conv`,
  );
  const bn = frozenBatchNorm(g, conv, outC, `${name}/bn`);
  return [relu6(g, bn, `${name}/relu6`), outC];
}

/**
 * depthwiseBnRelu6 — DepthwiseConv2D → BatchNorm → ReLU6.
 *
 * Depthwise step of a separable convolution (MobileNet-style).
 * Applies one filter per input channel — channel count is preserved.
 */
export function depthwiseBnRelu6(
  g: Graph,
  input: Tensor,
  inC: number,
  stride: number,
  name: string,
): [Tensor, number] {
  const w = constHe(g, [3, 3, inC, 1], `${name}/dw`);
  const conv = depthwiseConv2d(
    g,
    input,
    w,
    [1, stride, stride, 1],
    "SAME",
    `${name}/dw_conv`,
  );
  const bn = frozenBatchNorm(g, conv, inC, `${name}/bn`);
  return [relu6(g, bn, `${name}/relu6`), inC];
}

/**
 * pointwiseBn — 1×1 Conv2D (pointwise) → BatchNorm, no activation.
 *
 * Projection step of a separable convolution (MobileNet-style).
 * Linear activation is intentional — the MobileNetV2 paper shows that
 * applying ReLU to low-dimensional projections destroys information.
 */
export function pointwiseBn(
  g: Graph,
  input: Tensor,
  inC: number,
  outC: number,
  name: string,
): [Tensor, number] {
  const w = constGlorot(g, [1, 1, inC, outC], `${name}/pw`);
  const conv = conv2d(g, input, w, [1, 1, 1, 1], "SAME", `${name}/conv`);
  return [frozenBatchNorm(g, conv, outC, `${name}/bn`), outC];
}

/**
 * residualBlock — bottleneck identity block (ResNet-style).
 *
 * Three-layer bottleneck: 1×1 → 3×3 → 1×1 with a skip connection.
 * Used when input channels equal the final filter count (no downsampling).
 *
 *   input ──────────────────────────── AddV2 → ReLU → output
 *             1×1 conv → 3×3 conv → 1×1 conv ──┘
 *
 * @param inC      Input (and output) channels — must equal filters[2]
 * @param filters  [bottleneck, bottleneck, expansion] e.g. [64, 64, 256]
 */
export function residualBlock(
  g: Graph,
  input: Tensor,
  inC: number,
  filters: [number, number, number],
  name: string,
): [Tensor, number] {
  const [f1, f2, f3] = filters;
  let [h] = convBnRelu(g, input, inC, f1, 1, 1, "SAME", `${name}/a`);
  [h] = convBnRelu(g, h, f1, f2, 3, 1, "SAME", `${name}/b`);

  const wC = constHe(g, [1, 1, f2, f3], `${name}/c/w`);
  const cOut = conv2d(g, h, wC, [1, 1, 1, 1], "SAME", `${name}/c/conv`);
  const bOut = frozenBatchNorm(g, cOut, f3, `${name}/c/bn`);

  const [added] = g.addOp("AddV2", [input, bOut], {}, `${name}/add`);
  return [relu(g, added, `${name}/relu`), f3];
}

/**
 * projectionBlock — bottleneck block with projection shortcut (ResNet-style).
 *
 * Used when the spatial dimensions or channel count changes (stride > 1 or
 * inC ≠ filters[2]). A 1×1 conv projection on the shortcut aligns the shapes.
 *
 *   input ── 1×1 proj (stride) ─────────────────── AddV2 → ReLU → output
 *            1×1 conv (stride) → 3×3 conv → 1×1 conv ──┘
 *
 * @param inC      Input channels
 * @param filters  [bottleneck, bottleneck, expansion] e.g. [64, 64, 256]
 * @param stride   Stride applied to the first conv and the shortcut projection
 */
export function projectionBlock(
  g: Graph,
  input: Tensor,
  inC: number,
  filters: [number, number, number],
  stride: number,
  name: string,
): [Tensor, number] {
  const [f1, f2, f3] = filters;
  let [h] = convBnRelu(g, input, inC, f1, 1, stride, "SAME", `${name}/a`);
  [h] = convBnRelu(g, h, f1, f2, 3, 1, "SAME", `${name}/b`);

  const wC = constHe(g, [1, 1, f2, f3], `${name}/c/w`);
  const cOut = conv2d(g, h, wC, [1, 1, 1, 1], "SAME", `${name}/c/conv`);
  const bOut = frozenBatchNorm(g, cOut, f3, `${name}/c/bn`);

  // Projection shortcut — 1×1 conv with same stride to match spatial dims.
  const wS = constHe(g, [1, 1, inC, f3], `${name}/shortcut/w`);
  const sC = conv2d(
    g,
    input,
    wS,
    [1, stride, stride, 1],
    "SAME",
    `${name}/shortcut/conv`,
  );
  const sB = frozenBatchNorm(g, sC, f3, `${name}/shortcut/bn`);

  const [added] = g.addOp("AddV2", [sB, bOut], {}, `${name}/add`);
  return [relu(g, added, `${name}/relu`), f3];
}

/**
 * invertedResidual — inverted residual block (MobileNetV2-style).
 *
 * Expands channels by expandRatio, applies depthwise conv, then projects
 * back down. Skip connection only when stride=1 and inC === outC.
 *
 *   input ── [expand] ── depthwise ── project ── [+ input] ── output
 *
 * @param inC         Input channels
 * @param outC        Output channels
 * @param stride      Depthwise stride (1 or 2)
 * @param expandRatio Channel expansion factor (1 = no expansion for first block)
 */
export function invertedResidual(
  g: Graph,
  input: Tensor,
  inC: number,
  outC: number,
  stride: number,
  expandRatio: number,
  name: string,
): [Tensor, number] {
  const expandC = inC * expandRatio;
  let h: Tensor = input;

  // Expansion: 1×1 convBnRelu6 — skipped when expandRatio=1 (first block).
  if (expandRatio > 1) {
    [h] = convBnRelu6(g, input, inC, expandC, 1, 1, `${name}/expand`);
  }

  // Depthwise: 3×3 per-channel convolution.
  [h] = depthwiseBnRelu6(g, h, expandC, stride, `${name}/dw`);

  // Projection: linear 1×1 (no activation — preserves low-dim information).
  [h] = pointwiseBn(g, h, expandC, outC, `${name}/project`);

  // Skip connection only when spatial dims and channels are preserved.
  if (stride === 1 && inC === outC) {
    const [out] = g.addOp("AddV2", [input, h], {}, `${name}/add`);
    return [out, outC];
  }
  return [h, outC];
}
