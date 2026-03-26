import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import { constant } from "./array_ops.js";
import {
  sub,
  mul,
  div,
  add,
  exp,
  log,
  sum,
  mean,
  sqrt,
  square,
} from "./math_ops.js";

// ---------------------------------------------------------------------------
// nn_ops — neural network operations
// ---------------------------------------------------------------------------

// ── Activations ──────────────────────────────────────────────────────────────

/** Rectified linear unit: max(0, x). */
export function relu(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Relu", [x], {}, name);
  return t;
}

/** Leaky ReLU: max(alpha * x, x). Default alpha = 0.2. */
export function leakyRelu(
  g: Graph,
  x: Tensor,
  alpha = 0.2,
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "LeakyRelu",
    [x],
    {
      alpha: { kind: "float", value: alpha },
    },
    name,
  );
  return t;
}

/** ReLU6: min(max(0, x), 6). Common in MobileNet. */
export function relu6(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Relu6", [x], {}, name);
  return t;
}

/** Sigmoid: 1 / (1 + e^-x). */
export function sigmoid(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Sigmoid", [x], {}, name);
  return t;
}

/** Hyperbolic tangent. */
export function tanh(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Tanh", [x], {}, name);
  return t;
}

/**
 * Softmax along the last axis.
 * For other axes use softmaxAxis().
 */
export function softmax(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Softmax", [x], {}, name);
  return t;
}

/** ELU: x if x > 0, else e^x - 1. */
export function elu(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Elu", [x], {}, name);
  return t;
}

/** SELU: scaled ELU. */
export function selu(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Selu", [x], {}, name);
  return t;
}

/**
 * Swish: x * sigmoid(x).
 * Built from primitives — no single TF op for this.
 */
export function swish(g: Graph, x: Tensor, name?: string): Tensor {
  return mul(g, x, sigmoid(g, x), name);
}

/**
 * GELU (approximate): x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 * Used in BERT, GPT.
 * Built from primitives.
 */
export function gelu(g: Graph, x: Tensor, name?: string): Tensor {
  // 0.044715
  const c1 = constant(
    g,
    (() => {
      const b = Buffer.allocUnsafe(4);
      b.writeFloatLE(0.044715, 0);
      return b;
    })(),
    [],
    DType.FLOAT32,
  );
  // sqrt(2 / pi) ≈ 0.7978845608
  const c2 = constant(
    g,
    (() => {
      const b = Buffer.allocUnsafe(4);
      b.writeFloatLE(0.7978845608, 0);
      return b;
    })(),
    [],
    DType.FLOAT32,
  );
  const half = constant(
    g,
    (() => {
      const b = Buffer.allocUnsafe(4);
      b.writeFloatLE(0.5, 0);
      return b;
    })(),
    [],
    DType.FLOAT32,
  );
  const one = constant(
    g,
    (() => {
      const b = Buffer.allocUnsafe(4);
      b.writeFloatLE(1.0, 0);
      return b;
    })(),
    [],
    DType.FLOAT32,
  );

  const x3 = mul(g, mul(g, x, x), x);
  const inner = mul(g, c2, add(g, x, mul(g, c1, x3)));
  const tanhInner = tanh(g, inner);
  return mul(g, mul(g, x, half), add(g, one, tanhInner), name);
}

/** Log-softmax: log(softmax(x)). More numerically stable for cross-entropy. */
export function logSoftmax(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("LogSoftmax", [x], {}, name);
  return t;
}

// ── Normalisation ─────────────────────────────────────────────────────────────

/**
 * Batch normalisation (inference mode).
 *
 * Uses pre-computed running mean/variance from training.
 * For training-mode BN (with FusedBatchNormV3 + gradient), use batchNormTraining().
 */
export function batchNorm(
  g: Graph,
  x: Tensor,
  scale: Tensor,
  offset: Tensor,
  mean_: Tensor,
  variance_: Tensor,
  options: { epsilon?: number; dataFormat?: "NHWC" | "NCHW" } = {},
  name?: string,
): Tensor {
  const [y] = g.addOp(
    "FusedBatchNormV3",
    [x, scale, offset, mean_, variance_],
    {
      epsilon: { kind: "float", value: options.epsilon ?? 1e-3 },
      data_format: { kind: "bool", value: false }, // NHWC
      is_training: { kind: "bool", value: false },
    },
    name,
  );
  return y; // index 0 is the normalised output; other outputs are for training
}

/**
 * Layer normalisation — normalises across the last `numAxes` dimensions.
 * Built from primitives (TF has no single LayerNorm C op in the public API).
 *
 * @param numAxes  Number of trailing axes to normalise over (default 1).
 */
export function layerNorm(
  g: Graph,
  x: Tensor,
  scale: Tensor,
  offset: Tensor,
  options: { epsilon?: number; axes?: number[] } = {},
  name?: string,
): Tensor {
  const eps = options.epsilon ?? 1e-5;
  // Default: normalise over last axis only. Caller should specify axes
  // explicitly when normalising over multiple dims (e.g. for transformers).
  const axes = options.axes ?? [-1];

  const mu = mean(g, x, axes, /* keepDims */ true);
  const diff = sub(g, x, mu);
  const var_ = mean(g, square(g, diff), axes, true);

  const epsBuf = Buffer.allocUnsafe(4);
  epsBuf.writeFloatLE(eps, 0);
  const epsT = constant(g, epsBuf, [], DType.FLOAT32);

  const denom = sqrt(g, add(g, var_, epsT));
  const xNorm = div(g, diff, denom);
  return add(g, mul(g, scale, xNorm), offset, name);
}

/**
 * Dropout — applies inverted dropout during training.
 *
 * rate: fraction of units to drop (e.g. 0.5 = drop 50%).
 * training: if false, returns x unchanged.
 *
 * NOTE: TF's Dropout op requires a random seed and noise shape tensor.
 * This implementation wraps the functional form for inference-time use
 * (rate ignored when training=false).
 *
 * TODO: wire up training-mode dropout once variable_ops seeds are in place.
 */
export function dropout(
  g: Graph,
  x: Tensor,
  rate: number,
  training: boolean,
  name?: string,
): Tensor {
  if (!training || rate === 0) {
    // Inference: identity pass-through.
    const [t] = g.addOp("Identity", [x], {}, name);
    return t;
  }

  // Training: scale = 1 / (1 - rate), then apply random mask.
  const keepProb = 1 - rate;
  const scaleBuf = Buffer.allocUnsafe(4);
  scaleBuf.writeFloatLE(1 / keepProb, 0);
  const scaleT = constant(g, scaleBuf, [], DType.FLOAT32);

  const [t] = g.addOp(
    "Dropout",
    [x, scaleT],
    {
      // noise_shape defaults to shape of x
    },
    name,
  );
  return t;
}

// ── Convolution ───────────────────────────────────────────────────────────────

/**
 * 2D convolution (NHWC).
 *
 * @param x        Input tensor [batch, height, width, in_channels]
 * @param filter   Filter tensor [filter_height, filter_width, in_channels, out_channels]
 * @param strides  [1, stride_h, stride_w, 1]
 * @param padding  "SAME" or "VALID"
 */
export function conv2d(
  g: Graph,
  x: Tensor,
  filter: Tensor,
  strides: [number, number, number, number] = [1, 1, 1, 1],
  padding: "SAME" | "VALID" = "SAME",
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "Conv2D",
    [x, filter],
    {
      strides: { kind: "list_int", value: strides },
      padding: { kind: "bool", value: padding === "SAME" },
      data_format: { kind: "bool", value: false }, // NHWC
    },
    name,
  );
  return t;
}

/**
 * Depthwise 2D convolution.
 * Each input channel is convolved with its own filter of depth channel_multiplier.
 */
export function depthwiseConv2d(
  g: Graph,
  x: Tensor,
  filter: Tensor,
  strides: [number, number, number, number] = [1, 1, 1, 1],
  padding: "SAME" | "VALID" = "SAME",
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "DepthwiseConv2dNative",
    [x, filter],
    {
      strides: { kind: "list_int", value: strides },
      padding: { kind: "bool", value: padding === "SAME" },
      data_format: { kind: "bool", value: false }, // NHWC
    },
    name,
  );
  return t;
}

// ── Pooling ───────────────────────────────────────────────────────────────────

/** Max pooling. */
export function maxPool(
  g: Graph,
  x: Tensor,
  ksize: [number, number, number, number] = [1, 2, 2, 1],
  strides: [number, number, number, number] = [1, 2, 2, 1],
  padding: "SAME" | "VALID" = "VALID",
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "MaxPool",
    [x],
    {
      ksize: { kind: "list_int", value: ksize },
      strides: { kind: "list_int", value: strides },
      padding: { kind: "bool", value: padding === "SAME" },
      data_format: { kind: "bool", value: false },
    },
    name,
  );
  return t;
}

/** Average pooling. */
export function avgPool(
  g: Graph,
  x: Tensor,
  ksize: [number, number, number, number] = [1, 2, 2, 1],
  strides: [number, number, number, number] = [1, 2, 2, 1],
  padding: "SAME" | "VALID" = "VALID",
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "AvgPool",
    [x],
    {
      ksize: { kind: "list_int", value: ksize },
      strides: { kind: "list_int", value: strides },
      padding: { kind: "bool", value: padding === "SAME" },
      data_format: { kind: "bool", value: false },
    },
    name,
  );
  return t;
}

/** Global average pooling — reduces spatial dims to [batch, channels]. */
export function globalAvgPool(g: Graph, x: Tensor, name?: string): Tensor {
  // Reduce over H and W (axes 1 and 2 for NHWC).
  return mean(g, x, [1, 2], /* keepDims */ false, name);
}

// ── Loss functions ────────────────────────────────────────────────────────────

/**
 * Sparse softmax cross-entropy.
 * labels: int32/int64 class indices [batch]
 * logits: float32 [batch, num_classes]
 * Returns per-example loss [batch].
 */
export function sparseSoftmaxCrossEntropyWithLogits(
  g: Graph,
  labels: Tensor,
  logits: Tensor,
  name?: string,
): Tensor {
  const [loss] = g.addOp(
    "SparseSoftmaxCrossEntropyWithLogits",
    [logits, labels],
    {},
    name,
  );
  return loss; // output 0 is per-example loss; output 1 is backprop grad
}

/**
 * Sigmoid cross-entropy with logits (binary classification).
 * labels: float32 {0, 1} [batch]
 * logits: float32 [batch]
 */
export function sigmoidCrossEntropyWithLogits(
  g: Graph,
  labels: Tensor,
  logits: Tensor,
  name?: string,
): Tensor {
  // max(logits, 0) - logits * labels + log(1 + exp(-abs(logits)))
  // Built from primitives for numerical stability.
  const zeroBuf = Buffer.allocUnsafe(4);
  zeroBuf.writeFloatLE(0, 0);
  const zero = constant(g, zeroBuf, [], DType.FLOAT32);
  const oneBuf = Buffer.allocUnsafe(4);
  oneBuf.writeFloatLE(1, 0);
  const one = constant(g, oneBuf, [], DType.FLOAT32);

  const [relu_logits] = g.addOp("Relu", [logits], {});
  const [abs_logits] = g.addOp("Abs", [logits], {});
  const [neg_abs] = g.addOp("Neg", [abs_logits], {});
  const [exp_neg] = g.addOp("Exp", [neg_abs], {});
  const [one_plus] = g.addOp("AddV2", [one, exp_neg], {});
  const [log_part] = g.addOp("Log", [one_plus], {});
  const [prod] = g.addOp("Mul", [logits, labels], {});
  const [t] = g.addOp(
    "AddV2",
    [g.addOp("Sub", [relu_logits, prod], {})[0], log_part],
    {},
    name,
  );
  return t;
}

/**
 * L2 loss: 0.5 * sum(t^2).
 * Used as a regularisation term.
 */
export function l2Loss(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("L2Loss", [x], {}, name);
  return t;
}
