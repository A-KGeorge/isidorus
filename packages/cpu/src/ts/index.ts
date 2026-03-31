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

const addon = nodeGypBuild(join(__dirname, "..", "..")) as any;

import { setAddon } from "./_native.js";
setAddon(addon);

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
  return new Graph(new addon.Graph());
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
  return new Session(new addon.Session(g._native, options));
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
