import type { Tensor, Shape } from "@isidorus/core";
import { DType, shapeToTF } from "@isidorus/core";
import type { Graph } from "../graph.js";
import { constant } from "./array_ops.js";

// ---------------------------------------------------------------------------
// variable_ops — variable lifecycle and initialisation
//
// TF variable lifecycle in the graph-first model:
//
//   1. variable()       → adds a VarHandleOp to the graph
//                         returns a Tensor (the variable handle)
//   2. initializer()    → adds an AssignVariableOp feeding the handle
//                         returns the assign op name (run as a target)
//   3. readVariable()   → adds a ReadVariableOp
//                         returns the current value as a Tensor
//   4. assignVariable() → adds an AssignVariableOp for a new value
//                         returns the assign op name (run as a target)
//
// In a Session:
//   sess.run([], [], ["init_all"])     ← run initializer targets
//   sess.run([[x, feed]], [y])         ← inference
//   sess.run([[x, feed]], [], ["step"]) ← training step (assign ops)
// ---------------------------------------------------------------------------

// ── Variable handle ───────────────────────────────────────────────────────────

/**
 * Variable handle — a VarHandleOp that identifies a resource variable.
 *
 * The handle itself holds no data. Use initializer() to set the initial
 * value, readVariable() to read it, and assignVariable() to update it.
 *
 * @param shape     Shape of the variable (null dims = dynamic)
 * @param dtype     Element type
 * @param varName   Logical name for the variable (used in checkpoints)
 */
export function variable(
  g: Graph,
  shape: Shape,
  dtype: DType,
  varName: string,
): Tensor {
  const [handle] = g.addOp(
    "VarHandleOp",
    [],
    {
      dtype: { kind: "type", value: dtype },
      shape: { kind: "shape", value: shapeToTF(shape) },
      shared_name: { kind: "string", value: varName },
      // container and shared_name are string attrs — we use the op name
      // as the variable name which TF uses for checkpoint key resolution.
    },
    varName,
  );
  return handle;
}

/**
 * readVariable — emits a ReadVariableOp to read the current value.
 * The returned Tensor can be used as input to any op.
 */
export function readVariable(
  g: Graph,
  handle: Tensor,
  dtype: DType,
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "ReadVariableOp",
    [handle],
    {
      dtype: { kind: "type", value: dtype },
    },
    name,
  );
  return t;
}

/**
 * assignVariable — adds an AssignVariableOp that writes `value` into
 * the variable identified by `handle`.
 *
 * Returns the op name — pass it as a target in sess.run() to execute
 * the assignment as a side-effect.
 *
 * @example
 * const updateOp = ops.assignVariable(g, wHandle, newW, DType.FLOAT32);
 * await sess.run(feeds, [], [updateOp]);
 */
export function assignVariable(
  g: Graph,
  handle: Tensor,
  value: Tensor,
  dtype: DType,
  name?: string,
): string {
  const opName = name ?? `AssignVariableOp_${handle.opName}`;
  g.addOp(
    "AssignVariableOp",
    [handle, value],
    {
      dtype: { kind: "type", value: dtype },
    },
    opName,
  );
  return opName;
}

/**
 * assignAdd — adds an AssignAddVariableOp: variable += delta.
 * Returns the op name.
 */
export function assignAdd(
  g: Graph,
  handle: Tensor,
  delta: Tensor,
  dtype: DType,
  name?: string,
): string {
  const opName = name ?? `AssignAddVariableOp_${handle.opName}`;
  g.addOp(
    "AssignAddVariableOp",
    [handle, delta],
    {
      dtype: { kind: "type", value: dtype },
    },
    opName,
  );
  return opName;
}

/**
 * assignSub — adds an AssignSubVariableOp: variable -= delta.
 * Returns the op name.
 */
export function assignSub(
  g: Graph,
  handle: Tensor,
  delta: Tensor,
  dtype: DType,
  name?: string,
): string {
  const opName = name ?? `AssignSubVariableOp_${handle.opName}`;
  g.addOp(
    "AssignSubVariableOp",
    [handle, delta],
    {
      dtype: { kind: "type", value: dtype },
    },
    opName,
  );
  return opName;
}

// ── Initializers ──────────────────────────────────────────────────────────────

/**
 * zerosInitializer — creates a constant zero tensor matching `shape` and `dtype`.
 * Use with initializer() to zero-initialise a variable.
 */
export function zerosInitializer(
  g: Graph,
  shape: number[],
  dtype: DType,
  name?: string,
): Tensor {
  const itemSize =
    dtype === DType.FLOAT64
      ? 8
      : dtype === DType.INT64 || dtype === DType.UINT64
      ? 8
      : dtype === DType.INT16 || dtype === DType.UINT16
      ? 2
      : dtype === DType.INT8 || dtype === DType.UINT8
      ? 1
      : 4;
  const n = shape.reduce((a, b) => a * b, 1);
  const buf = Buffer.alloc(n * itemSize, 0);
  return constant(g, buf, shape, dtype, name);
}

/**
 * onesInitializer — constant one tensor.
 */
export function onesInitializer(
  g: Graph,
  shape: number[],
  dtype: DType,
  name?: string,
): Tensor {
  const n = shape.reduce((a, b) => a * b, 1);
  const buf = Buffer.allocUnsafe(n * 4);
  for (let i = 0; i < n; i++) buf.writeFloatLE(1, i * 4);
  return constant(g, buf, shape, dtype, name);
}

/**
 * truncatedNormalInitializer — samples from a truncated normal distribution.
 * Values more than 2 stddevs from the mean are resampled.
 * This is Xavier/Glorot-style initialisation when stddev = sqrt(2/fan_in).
 *
 * NOTE: TF's TruncatedNormal op produces a different value on every run
 * unless a fixed seed is provided. For reproducible initialisation, run
 * the init op once and save the checkpoint.
 */
export function truncatedNormalInitializer(
  g: Graph,
  shape: number[],
  dtype: DType = DType.FLOAT32,
  options: { mean?: number; stddev?: number; seed?: number } = {},
  name?: string,
): Tensor {
  const shapeBuf = Buffer.allocUnsafe(shape.length * 4);
  shape.forEach((d, i) => shapeBuf.writeInt32LE(d, i * 4));
  const shapeConst = constant(g, shapeBuf, [shape.length], DType.INT32);

  const [t] = g.addOp(
    "TruncatedNormal",
    [shapeConst],
    {
      dtype: { kind: "type", value: dtype },
      mean: { kind: "float", value: options.mean ?? 0.0 },
      stddev: { kind: "float", value: options.stddev ?? 1.0 },
      seed: { kind: "int", value: options.seed ?? 0 },
      seed2: { kind: "int", value: 0 },
    },
    name,
  );
  return t;
}

/**
 * glorotUniformInitializer — samples from Uniform(-limit, limit)
 * where limit = sqrt(6 / (fan_in + fan_out)).
 *
 * For a weight matrix [fan_in, fan_out]:
 *   fan_in  = shape[0]
 *   fan_out = shape[1]
 */
export function glorotUniformInitializer(
  g: Graph,
  shape: [number, number], // [fan_in, fan_out]
  dtype: DType = DType.FLOAT32,
  name?: string,
): Tensor {
  const [fanIn, fanOut] = shape;
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  const n = fanIn * fanOut;
  const buf = Buffer.allocUnsafe(n * 4);
  for (let i = 0; i < n; i++) {
    buf.writeFloatLE(Math.random() * 2 * limit - limit, i * 4);
  }
  return constant(g, buf, shape, dtype, name);
}

// ── Variable + initializer convenience ───────────────────────────────────────

/**
 * variableWithInit — creates a variable handle and its initialisation op
 * in one call.
 *
 * @returns { handle, initOp }
 *   handle:  the VarHandleOp tensor (use with readVariable / assignVariable)
 *   initOp:  op name to pass as a target in sess.run() to initialise
 *
 * @example
 * const { handle: w, initOp: wInit } = ops.variableWithInit(
 *   g, [784, 128], DType.FLOAT32, "weights",
 *   ops.glorotUniformInitializer(g, [784, 128])
 * );
 * const { handle: b, initOp: bInit } = ops.variableWithInit(
 *   g, [128], DType.FLOAT32, "bias",
 *   ops.zerosInitializer(g, [128], DType.FLOAT32)
 * );
 *
 * // Run init once before first training step:
 * await sess.run([], [], [wInit, bInit]);
 */
export function variableWithInit(
  g: Graph,
  shape: Shape,
  dtype: DType,
  varName: string,
  initialValue: Tensor,
): { handle: Tensor; initOp: string } {
  const handle = variable(g, shape, dtype, varName);
  const initOp = assignVariable(
    g,
    handle,
    initialValue,
    dtype,
    `${varName}/init`,
  );
  return { handle, initOp };
}

/**
 * globalVariablesInitializer — groups all init ops into a single NoOp target.
 *
 * @param initOps  Array of init op names returned by variableWithInit()
 * @returns        Op name to pass as a single target to sess.run()
 *
 * @example
 * const initAll = ops.globalVariablesInitializer(g, [wInit, bInit]);
 * await sess.run([], [], [initAll]);
 */
export function globalVariablesInitializer(
  g: Graph,
  initOps: string[],
  name = "init_all_variables",
): string {
  // NoOp with control edges to every init AssignVariableOp.
  // TF_AddControlInput guarantees the NoOp will not execute until all
  // listed ops have completed, so running this target from a Session
  // atomically initialises all variables before any training step proceeds.
  g.addOp(
    "NoOp",
    [],
    {},
    name,
    initOps, // controlInputs — wired via TF_AddControlInput in graph.cc
  );
  return name;
}

// ── Optimizer update ops ──────────────────────────────────────────────────────
//
// These are low-level building blocks. High-level optimizers (SGD, Adam)
// are built on top in a separate optimizers/ module once the gradient
// infrastructure is in place.
//
// Each returns an op name to run as a target (side-effect, no output).

/**
 * applyGradientDescent — w -= lr * grad.
 * The simplest parameter update step.
 */
export function applyGradientDescent(
  g: Graph,
  handle: Tensor,
  lr: Tensor, // scalar learning rate
  grad: Tensor,
  dtype: DType,
  name?: string,
): string {
  const opName = name ?? `ApplyGradientDescent_${handle.opName}`;
  g.addOp(
    "ResourceApplyGradientDescent",
    [handle, lr, grad],
    {
      T: { kind: "type", value: dtype },
      use_locking: { kind: "bool", value: false },
    },
    opName,
  );
  return opName;
}

/**
 * applyAdam — one Adam parameter update step.
 *
 * Requires four variable handles: var, m (first moment), v (second moment),
 * and beta1_power, beta2_power scalars (updated separately).
 */
export function applyAdam(
  g: Graph,
  handle: Tensor,
  mHandle: Tensor,
  vHandle: Tensor,
  beta1Power: Tensor,
  beta2Power: Tensor,
  lr: Tensor,
  beta1: Tensor,
  beta2: Tensor,
  epsilon: Tensor,
  grad: Tensor,
  dtype: DType,
  name?: string,
): string {
  const opName = name ?? `ApplyAdam_${handle.opName}`;
  g.addOp(
    "ResourceApplyAdam",
    [
      handle,
      mHandle,
      vHandle,
      beta1Power,
      beta2Power,
      lr,
      beta1,
      beta2,
      epsilon,
      grad,
    ],
    {
      T: { kind: "type", value: dtype },
      use_locking: { kind: "bool", value: false },
      use_nesterov: { kind: "bool", value: false },
    },
    opName,
  );
  return opName;
}

/**
 * applyRMSProp — one RMSProp parameter update step.
 */
export function applyRMSProp(
  g: Graph,
  handle: Tensor,
  msHandle: Tensor,
  momHandle: Tensor,
  lr: Tensor,
  rho: Tensor,
  momentum: Tensor,
  epsilon: Tensor,
  grad: Tensor,
  dtype: DType,
  name?: string,
): string {
  const opName = name ?? `ApplyRMSProp_${handle.opName}`;
  g.addOp(
    "ResourceApplyRMSProp",
    [handle, msHandle, momHandle, lr, rho, momentum, epsilon, grad],
    {
      T: { kind: "type", value: dtype },
      use_locking: { kind: "bool", value: false },
    },
    opName,
  );
  return opName;
}
