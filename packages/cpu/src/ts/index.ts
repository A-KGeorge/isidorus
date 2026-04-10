/**
 * @isidorus/cpu entry point.
 *
 * Resolves libtensorflow before loading the native addon, so users
 * get a clear message and an install prompt rather than a cryptic
 * "cannot find module" or DLL load error.
 */

import { ensureTf } from "./install-libtensorflow.js";
import nodeGypBuild from "node-gyp-build";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// ── Ensure libtensorflow is present before touching the native addon ─────────
// This is the only await at module top level — it resolves synchronously
// on the fast path (library already present) and only blocks for the
// install prompt if the library is genuinely missing.
await ensureTf();

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

let IsidorusAddon: any;
try {
  const addon = nodeGypBuild(join(__dirname, "..")) as any;
  IsidorusAddon = addon.SharedTensor ?? addon;
} catch (e) {
  try {
    const addon = nodeGypBuild(join(__dirname, "..", "..")) as any;
    IsidorusAddon = addon.SharedTensor ?? addon;
  } catch (err: any) {
    console.error("Failed to load native SharedTensor module.");
    console.error(
      "Attempt 1 error (installed path ../):",
      (e as Error).message,
    );
    console.error("Attempt 2 error (local path ../../):", err.message);
    throw new Error(
      `Could not load native module. Is the build complete? ` +
        `Search paths tried: ${join(__dirname, "..")} and ${join(
          __dirname,
          "..",
          "..",
        )}`,
    );
  }
}

import { setAddon } from "./_native.js";
setAddon(IsidorusAddon);
export { getAddon } from "./_native.js";
export * from "./ops/index.js";
export * from "./optimizers/index.js";
export * from "./model/index.js";

// ── Re-export core types ─────────────────────────────────────────────────────
export type { Tensor, Shape } from "@isidorus/core";
export {
  DType,
  dtypeItemSize,
  dtypeName,
  makeTensor,
  tensorId,
} from "@isidorus/core";

// ── Graph + Session ───────────────────────────────────────────────────────────
export type { AttrValue, InlineTensor, TFOutput } from "./graph.js";
export { Graph } from "./graph.js";
export type { FeedValue, TensorValue } from "./session.js";
export { Session } from "./session.js";

// ── Factory functions ─────────────────────────────────────────────────────────

import { Graph } from "./graph.js";
import { Session } from "./session.js";

/**
 * Create a new empty Graph.
 *
 * @example
 * const g = graph();
 * const x = ops.placeholder(g, "x", [null, 784], DType.FLOAT32);
 */
export function graph(): Graph {
  return new Graph(new IsidorusAddon.Graph());
}

/**
 * Create a Session backed by the given Graph.
 *
 * @param g        The graph to execute.
 * @param options  Thread and affinity configuration:
 *   strategy?:       "worker-pool" | "tf-parallel"
 *   intraOpThreads?: number   (explicit override)
 *   interOpThreads?: number
 *   reserveCores?:   number   (reserve N cores for event loop / other libs)
 *
 * @example
 * // Leave 2 cores for the event loop and opencv — TF gets the rest
 * const sess = session(g, { strategy: "tf-parallel", reserveCores: 2 });
 */
export function session(
  g: Graph,
  options?: {
    strategy?: "worker-pool" | "tf-parallel";
    intraOpThreads?: number;
    interOpThreads?: number;
    reserveCores?: number;
  },
): Session {
  return new Session(new IsidorusAddon.Session(g._native, options));
}

// ── Ops namespace ─────────────────────────────────────────────────────────────
export * as ops from "./ops/index.js";

// ── InferencePool ─────────────────────────────────────────────────────────────
export type {
  PoolOptions,
  PoolResult,
  ExecutionStrategy,
} from "./inference-pool.js";
export { InferencePool } from "./inference-pool.js";

// ── Model layers and optimizers ───────────────────────────────────────────────
export * as optimizers from "./optimizers/index.js";
export type {
  ActivationFn,
  Layer,
  LayerParam,
  LossFn,
  TrainStepResult,
} from "./model/index.js";
export { Dense, Flatten, Conv2D, Sequential } from "./model/index.js";
