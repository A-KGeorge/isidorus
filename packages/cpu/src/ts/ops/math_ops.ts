import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import { constant } from "./array_ops.js";

// ---------------------------------------------------------------------------
// math_ops — elementwise and reduction ops
//
// All ops follow the same pattern:
//   function opName(g: Graph, ...inputs: Tensor[], options?, name?: string): Tensor
//
// The graph is always the first argument so ops can be composed without
// needing a "default graph" global — consistent with the graph-first design.
// ---------------------------------------------------------------------------

// ── Binary elementwise ───────────────────────────────────────────────────────

/**
 * matmul — matrix multiplication.
 * Supports optional transpose of either operand.
 */
export function matmul(
  g: Graph,
  a: Tensor,
  b: Tensor,
  options: { transposeA?: boolean; transposeB?: boolean } = {},
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "MatMul",
    [a, b],
    {
      transpose_a: { kind: "bool", value: options.transposeA ?? false },
      transpose_b: { kind: "bool", value: options.transposeB ?? false },
    },
    name,
  );
  return t;
}

/**
 * batchMatmul — batched matrix multiplication (BatchMatMulV2).
 * Inputs must have rank >= 2. Leading dims are treated as batch dims.
 */
export function batchMatmul(
  g: Graph,
  a: Tensor,
  b: Tensor,
  options: { adjX?: boolean; adjY?: boolean } = {},
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "BatchMatMulV2",
    [a, b],
    {
      adj_x: { kind: "bool", value: options.adjX ?? false },
      adj_y: { kind: "bool", value: options.adjY ?? false },
    },
    name,
  );
  return t;
}

/** Element-wise addition. Supports broadcasting. */
export function add(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("AddV2", [a, b], {}, name);
  return t;
}

/** Element-wise subtraction. Supports broadcasting. */
export function sub(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Sub", [a, b], {}, name);
  return t;
}

/** Element-wise multiplication. Supports broadcasting. */
export function mul(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Mul", [a, b], {}, name);
  return t;
}

/** Element-wise (real) division. Supports broadcasting. */
export function div(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("RealDiv", [a, b], {}, name);
  return t;
}

/**
 * biasAdd — adds a 1-D bias tensor to a value tensor.
 * The bias is added to the last dimension of value.
 * Equivalent to add() but semantically clearer for neural network layers.
 */
export function biasAdd(
  g: Graph,
  value: Tensor,
  bias: Tensor,
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "BiasAdd",
    [value, bias],
    {
      data_format: { kind: "bool", value: false }, // NHWC default
    },
    name,
  );
  return t;
}

/** Element-wise maximum. Supports broadcasting. */
export function maximum(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Maximum", [a, b], {}, name);
  return t;
}

/** Element-wise minimum. Supports broadcasting. */
export function minimum(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Minimum", [a, b], {}, name);
  return t;
}

/** Element-wise x^y. Supports broadcasting. */
export function pow(
  g: Graph,
  base: Tensor,
  exp: Tensor,
  name?: string,
): Tensor {
  const [t] = g.addOp("Pow", [base, exp], {}, name);
  return t;
}

// ── Unary elementwise ────────────────────────────────────────────────────────

/** Element-wise negation. */
export function neg(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Neg", [x], {}, name);
  return t;
}

/** Element-wise absolute value. */
export function abs(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Abs", [x], {}, name);
  return t;
}

/** Element-wise e^x. */
export function exp(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Exp", [x], {}, name);
  return t;
}

/** Element-wise natural log. */
export function log(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Log", [x], {}, name);
  return t;
}

/** Element-wise square root. */
export function sqrt(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Sqrt", [x], {}, name);
  return t;
}

/** Element-wise square. */
export function square(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Square", [x], {}, name);
  return t;
}

/** Element-wise reciprocal (1/x). */
export function reciprocal(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Reciprocal", [x], {}, name);
  return t;
}

/** Element-wise floor. */
export function floor(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Floor", [x], {}, name);
  return t;
}

/** Element-wise ceiling. */
export function ceil(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Ceil", [x], {}, name);
  return t;
}

/** Element-wise round (ties to even). */
export function round(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Round", [x], {}, name);
  return t;
}

/** Element-wise sign (-1, 0, or 1). */
export function sign(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Sign", [x], {}, name);
  return t;
}

/** Cast tensor to a different dtype. */
export function cast(g: Graph, x: Tensor, dtype: DType, name?: string): Tensor {
  const [t] = g.addOp(
    "Cast",
    [x],
    {
      DstT: { kind: "type", value: dtype },
    },
    name,
  );
  return t;
}

// ── Reductions ────────────────────────────────────────────────────────────────

/**
 * Build a constant int32 axis tensor for reduction ops.
 * axes: array of axis indices to reduce over.
 */
function makeAxisConst(g: Graph, axes: number[]): Tensor {
  const buf = Buffer.allocUnsafe(axes.length * 4);
  axes.forEach((a, i) => buf.writeInt32LE(a, i * 4));
  return constant(g, buf, [axes.length], DType.INT32);
}

/** Sum over the given axes. keepDims preserves reduced dimensions as size 1. */
export function sum(
  g: Graph,
  x: Tensor,
  axes: number[],
  keepDims = false,
  name?: string,
): Tensor {
  const axisT = makeAxisConst(g, axes);
  const [t] = g.addOp(
    "Sum",
    [x, axisT],
    {
      keep_dims: { kind: "bool", value: keepDims },
    },
    name,
  );
  return t;
}

/** Mean over the given axes. */
export function mean(
  g: Graph,
  x: Tensor,
  axes: number[],
  keepDims = false,
  name?: string,
): Tensor {
  const axisT = makeAxisConst(g, axes);
  const [t] = g.addOp(
    "Mean",
    [x, axisT],
    {
      keep_dims: { kind: "bool", value: keepDims },
    },
    name,
  );
  return t;
}

/** Max over the given axes. */
export function reduceMax(
  g: Graph,
  x: Tensor,
  axes: number[],
  keepDims = false,
  name?: string,
): Tensor {
  const axisT = makeAxisConst(g, axes);
  const [t] = g.addOp(
    "Max",
    [x, axisT],
    {
      keep_dims: { kind: "bool", value: keepDims },
    },
    name,
  );
  return t;
}

/** Min over the given axes. */
export function reduceMin(
  g: Graph,
  x: Tensor,
  axes: number[],
  keepDims = false,
  name?: string,
): Tensor {
  const axisT = makeAxisConst(g, axes);
  const [t] = g.addOp(
    "Min",
    [x, axisT],
    {
      keep_dims: { kind: "bool", value: keepDims },
    },
    name,
  );
  return t;
}

/** Product over the given axes. */
export function prod(
  g: Graph,
  x: Tensor,
  axes: number[],
  keepDims = false,
  name?: string,
): Tensor {
  const axisT = makeAxisConst(g, axes);
  const [t] = g.addOp(
    "Prod",
    [x, axisT],
    {
      keep_dims: { kind: "bool", value: keepDims },
    },
    name,
  );
  return t;
}

/** Variance over the given axes (naive: E[x^2] - E[x]^2). */
export function variance(
  g: Graph,
  x: Tensor,
  axes: number[],
  keepDims = false,
  name?: string,
): Tensor {
  const xMean = mean(g, x, axes, /* keepDims */ true);
  const diff = sub(g, x, xMean);
  const sq = square(g, diff);
  return mean(g, sq, axes, keepDims, name);
}

/** Standard deviation over the given axes. */
export function std(
  g: Graph,
  x: Tensor,
  axes: number[],
  keepDims = false,
  name?: string,
): Tensor {
  return sqrt(g, variance(g, x, axes, keepDims), name);
}

// ── Comparison ────────────────────────────────────────────────────────────────

/** Element-wise a == b → bool tensor. */
export function equal(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Equal", [a, b], {}, name);
  return t;
}

/** Element-wise a != b → bool tensor. */
export function notEqual(
  g: Graph,
  a: Tensor,
  b: Tensor,
  name?: string,
): Tensor {
  const [t] = g.addOp("NotEqual", [a, b], {}, name);
  return t;
}

/** Element-wise a > b → bool tensor. */
export function greater(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Greater", [a, b], {}, name);
  return t;
}

/** Element-wise a >= b → bool tensor. */
export function greaterEqual(
  g: Graph,
  a: Tensor,
  b: Tensor,
  name?: string,
): Tensor {
  const [t] = g.addOp("GreaterEqual", [a, b], {}, name);
  return t;
}

/** Element-wise a < b → bool tensor. */
export function less(g: Graph, a: Tensor, b: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Less", [a, b], {}, name);
  return t;
}

/** Element-wise a <= b → bool tensor. */
export function lessEqual(
  g: Graph,
  a: Tensor,
  b: Tensor,
  name?: string,
): Tensor {
  const [t] = g.addOp("LessEqual", [a, b], {}, name);
  return t;
}

// ── Clipping ─────────────────────────────────────────────────────────────────

/** Clip values to [minVal, maxVal]. */
export function clipByValue(
  g: Graph,
  x: Tensor,
  minVal: number,
  maxVal: number,
  name?: string,
): Tensor {
  const minT = constant(
    g,
    (() => {
      const b = Buffer.allocUnsafe(4);
      b.writeFloatLE(minVal, 0);
      return b;
    })(),
    [],
    DType.FLOAT32,
  );
  const maxT = constant(
    g,
    (() => {
      const b = Buffer.allocUnsafe(4);
      b.writeFloatLE(maxVal, 0);
      return b;
    })(),
    [],
    DType.FLOAT32,
  );
  const [t] = g.addOp("ClipByValue", [x, minT, maxT], {}, name);
  return t;
}

// ── Index ops ────────────────────────────────────────────────────────────────

/** Index of the maximum value along an axis. */
export function argMax(
  g: Graph,
  x: Tensor,
  axis: number,
  name?: string,
): Tensor {
  const axisT = constant(
    g,
    (() => {
      const b = Buffer.allocUnsafe(4);
      b.writeInt32LE(axis, 0);
      return b;
    })(),
    [],
    DType.INT32,
  );
  const [t] = g.addOp(
    "ArgMax",
    [x, axisT],
    {
      output_type: { kind: "type", value: DType.INT64 },
    },
    name,
  );
  return t;
}

/** Index of the minimum value along an axis. */
export function argMin(
  g: Graph,
  x: Tensor,
  axis: number,
  name?: string,
): Tensor {
  const axisT = constant(
    g,
    (() => {
      const b = Buffer.allocUnsafe(4);
      b.writeInt32LE(axis, 0);
      return b;
    })(),
    [],
    DType.INT32,
  );
  const [t] = g.addOp(
    "ArgMin",
    [x, axisT],
    {
      output_type: { kind: "type", value: DType.INT64 },
    },
    name,
  );
  return t;
}
