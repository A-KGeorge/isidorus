/**
 * InferencePool — concurrent TF inference via a single session and async queue.
 *
 * Architecture:
 *   A single TF session handles all concurrent requests. TF's internal thread
 *   scheduler (intra/interOpThreads) handles CPU parallelism — it knows which
 *   ops can run in parallel and avoids false sharing across cores. An async JS
 *   queue manages concurrent callers; maxQueueDepth provides backpressure.
 *
 *   The autotuner benchmarks candidate intra/inter/concurrency configs during
 *   create() and selects the best for this model on this machine (~300ms cold
 *   start). Override with explicit intraOpThreads/maxConcurrent to skip it.
 *
 * Benchmark results (AMD Ryzen 9 5900X, 24 threads):
 *   bench_small (MobileNetV2, 224x224x3):  1.36-3.5x faster than tfjs-node
 *   bench_medium (ResNet50, 224x224x3):    1.30-2.35x faster than tfjs-node
 *   bench_large (Dense ~44M params):       tfjs-node wins on memory-bw workloads
 */

import { readFileSync, statSync } from "fs";
import { performance } from "perf_hooks";
import { availableParallelism } from "os";

import { getAddon } from "./_native.js";
import { Graph } from "./graph.js";
import { Session } from "./session.js";
import { debug, warn } from "./_log.js";
import { DType } from "@isidorus/core";
import type { TensorValue, FeedValue } from "./session.js";
import type { DataLike } from "./model/easy.js";
import { toFloat32Array, toInt32Array } from "./model/easy.js";

// ── Public types ─────────────────────────────────────────────────────────────

/**
 * Optimisation profile — controls how InferencePool chooses thread counts
 * and concurrency without exposing raw knobs.
 *
 *   "auto"       (default) — benchmarks candidate configs during create() and
 *                picks the one with the best throughput for this model on this
 *                machine. Adds ~300ms to startup. Recommended for production.
 *
 *   "latency"    — minimise per-request latency. Gives all cores to one request
 *                at a time (intra=usable, maxConcurrent=1). Best for
 *                interactive workloads and large memory-bandwidth-bound models.
 *
 *   "throughput" — maximise requests/second. Splits cores across concurrent
 *                requests (intra=4, maxConcurrent=floor(usable/4)). Best for
 *                batch serving of small/medium compute-bound models.
 */
export type PoolProfile = "auto" | "latency" | "throughput";

export interface PoolOptions {
  /** Path to a frozen .pb model file. */
  modelPath: string;

  /**
   * Optimisation profile. Default: "auto".
   * @see PoolProfile
   */
  profile?: PoolProfile;

  /** Input op name. Auto-discovered from Placeholder ops if not specified. */
  inputOp?: string;

  /** Output op names. Auto-discovered from sink ops if not specified. */
  outputOps?: string[];

  /**
   * Maximum number of requests queued waiting for a slot.
   * Requests beyond this limit are rejected with QueueFullError.
   * Default: 128.
   */
  maxQueueDepth?: number;

  /**
   * Advanced: number of CPU cores to reserve for non-TF work.
   * Default: 0. Ignored when explicit thread counts are set.
   */
  reserveCores?: number;

  // ── Expert overrides — skip autotuning when set ───────────────────────────
  // Only set these if you have profiled the model and know better than auto.
  // Setting any of these disables the autotuner for that dimension.

  /** Expert: intra-op parallelism (threads per op). */
  intraOpThreads?: number;
  /** Expert: inter-op parallelism (concurrent independent graph branches). */
  interOpThreads?: number;
  /** Expert: max concurrent runAsync() calls in flight. */
  maxConcurrent?: number;
}

export interface PoolResult {
  outputs: TensorValue[];
  inferenceMs: number;
}

/** Thrown when a request arrives but the queue is at maxQueueDepth. */
export class QueueFullError extends Error {
  constructor(depth: number) {
    super(
      `InferencePool queue full (depth=${depth}). ` +
        `Apply backpressure or increase maxQueueDepth.`,
    );
    this.name = "QueueFullError";
  }
}

// ── Constants ────────────────────────────────────────────────────────────────

// Autotuner candidate intraOpThreads values.
// Candidates are filtered to ≤ usable cores and deduped before testing.
const AUTOTUNE_INTRA_CANDIDATES = [1, 2, 4, 8, 16, 32];

// Iterations per candidate. More = stable but slower startup.
// 3 warmup + 10 timed ≈ 300–500ms total on typical hardware.
// For large models (>= AUTOTUNE_LARGE_MODEL_BYTES) we halve this to reduce
// cold-start time — the oneDNN primitive cache dominates for large models so
// fewer iterations still give a reliable winner.
const AUTOTUNE_WARMUP = 3;
const AUTOTUNE_ITERS = 10;
const AUTOTUNE_ITERS_LARGE = 5;
const AUTOTUNE_LARGE_MODEL_BYTES = 50 * 1024 * 1024; // 50 MB

// When two configs are within this fraction of each other in throughput score,
// prefer the one with higher intraOpThreads (= lower per-request latency).
// Without this tiebreaker, noise in 10-iter timing can cause the autotuner to
// pick a high-concurrency/low-intra config that has similar throughput but
// much worse latency (e.g. 57ms vs 18ms for ResNet50 on a 24-core machine).
const AUTOTUNE_LATENCY_PREFERENCE_MARGIN = 0.08; // 8%

// ── Internal ──────────────────────────────────────────────────────────────────

interface QueueEntry {
  inputData: Float32Array | Int32Array;
  inputShape: number[];
  inputDtype: number;
  resolve: (r: PoolResult) => void;
  reject: (e: Error) => void;
}

// ── InferencePool ─────────────────────────────────────────────────────────────

export class InferencePool {
  private readonly graph: Graph;
  private readonly sess: Session;
  private readonly inputOp: string;
  private readonly outputOps: string[];
  private readonly maxQueueDepth: number;
  private readonly _inputShape: (number | null)[];
  private readonly modelPath: string;

  private active = 0;
  private destroyed = false;
  private readonly queue: QueueEntry[] = [];
  private readonly maxConcurrent: number;

  private constructor(params: {
    graph: Graph;
    sess: Session;
    inputOp: string;
    outputOps: string[];
    maxQueueDepth: number;
    maxConcurrent: number;
    inputShape: (number | null)[];
    modelPath: string;
  }) {
    this.graph = params.graph;
    this.sess = params.sess;
    this.inputOp = params.inputOp;
    this.outputOps = params.outputOps;
    this.maxQueueDepth = params.maxQueueDepth;
    this.maxConcurrent = params.maxConcurrent;
    this._inputShape = params.inputShape;
    this.modelPath = params.modelPath;
  }

  // ── Factory ───────────────────────────────────────────────────────────────

  static async create(opts: PoolOptions): Promise<InferencePool> {
    const addon = getAddon();
    const g = new Graph(new addon.Graph());

    // Read model size before importing — used for autotuner iteration scaling.
    const modelBytes = statSync(opts.modelPath).size;
    g.importGraphDef(readFileSync(opts.modelPath));

    // Auto-discover inputOp and outputOps.
    let inputOp = opts.inputOp;
    let outputOps = opts.outputOps?.slice();

    if (!inputOp || !outputOps?.length) {
      if (!inputOp) {
        const placeholders = g.listOpsOfType("Placeholder");
        if (!placeholders.length)
          throw new Error(`No Placeholder ops found in ${opts.modelPath}`);
        inputOp = placeholders[0];
      }
      if (!outputOps?.length) {
        const sinks = g.listSinkOps();
        if (!sinks.length)
          throw new Error(`No sink ops found in ${opts.modelPath}`);
        outputOps = sinks;
      }
    }

    // Resolve input shape from the graph's Placeholder type info.
    const rawShape = g._native.opOutputShape(inputOp, 0) as number[] | null;
    const inputShape = rawShape
      ? rawShape.map((d: number) => (d < 0 ? null : d))
      : [null];
    debug(
      `resolvedInputShape: op="${inputOp}" rawShape=${JSON.stringify(
        rawShape,
      )} -> ${JSON.stringify(inputShape)}`,
    );

    const hw = availableParallelism();
    const reserveCores = opts.reserveCores ?? 0;
    const usable = Math.max(1, hw - reserveCores);
    const profile = opts.profile ?? "auto";

    // ── libuv thread pool check ───────────────────────────────────────────────
    // Each runAsync() call occupies one libuv worker thread for the full
    // duration of TF_SessionRun. Node's default UV_THREADPOOL_SIZE is 4 —
    // meaning maxConcurrent > 4 has no effect without raising it first.
    // UV_THREADPOOL_SIZE must be set BEFORE Node starts (in the environment
    // or via a loader), not at runtime. We clamp maxConcurrent to the actual
    // pool size so the autotuner doesn't pick a concurrency the pool can't serve.
    const uvPoolSize = parseInt(process.env["UV_THREADPOOL_SIZE"] ?? "4", 10);
    if (uvPoolSize < 8)
      warn(
        `UV_THREADPOOL_SIZE=${uvPoolSize} limits runAsync concurrency to ${uvPoolSize}. ` +
          `Set UV_THREADPOOL_SIZE=<num_cores> before starting Node for full throughput. ` +
          `Example: UV_THREADPOOL_SIZE=24 node server.js`,
      );
    // Effective concurrency ceiling — we can't run more than the pool allows.
    const poolCap = uvPoolSize;

    // ── Helper: create a session + warm oneDNN ────────────────────────────────
    const dummyShape = (inputShape as (number | null)[]).map((d) => d ?? 1);
    const nElems = dummyShape.reduce((a, b) => a * b, 1);
    const dummyBuf = Buffer.alloc(nElems * 4);
    const dummyFeeds = [
      [
        { opName: inputOp!, index: 0 },
        { dtype: 1 as any, shape: dummyShape, data: dummyBuf },
      ],
    ] as any;
    const dummyFetches = (outputOps as string[]).map((op) => ({
      opName: op,
      index: 0,
    }));

    const makeSession = (intra: number, inter: number) =>
      new Session(
        new addon.Session(g._native, {
          reserveCores,
          intraOpThreads: intra,
          interOpThreads: inter,
        }),
      );

    const warmSession = async (s: Session) => {
      // Three warmup passes: first populates oneDNN cache, rest stabilise
      // TF's internal thread pool and memory allocator.
      for (let i = 0; i < AUTOTUNE_WARMUP; i++)
        await s.runAsync(dummyFeeds, dummyFetches as any).catch(() => {});
    };

    // ── Resolve expert overrides (bypass autotuner if set) ────────────────────
    let intra: number;
    let inter: number;
    let maxConcurrent: number;
    let sess: Session;

    const hasExplicitIntra = opts.intraOpThreads !== undefined;
    const hasExplicitConc = opts.maxConcurrent !== undefined;

    if (hasExplicitIntra || hasExplicitConc) {
      // Expert mode: honour explicit values, derive the missing one.
      // Always clamp maxConcurrent to poolCap — concurrency above the UV
      // thread pool size queues work rather than running it in parallel.
      if (hasExplicitIntra && hasExplicitConc) {
        intra = opts.intraOpThreads!;
        maxConcurrent = Math.min(opts.maxConcurrent!, poolCap);
        if (intra * maxConcurrent > usable)
          warn(
            `intra(${intra}) × maxConcurrent(${maxConcurrent}) = ${
              intra * maxConcurrent
            } > usable(${usable}) — thread over-subscription will hurt latency`,
          );
      } else if (hasExplicitIntra) {
        intra = opts.intraOpThreads!;
        maxConcurrent = Math.min(
          Math.max(1, Math.floor(usable / intra)),
          poolCap,
        );
      } else {
        maxConcurrent = Math.min(opts.maxConcurrent!, poolCap);
        intra = Math.max(1, Math.floor(usable / maxConcurrent));
      }
      inter = opts.interOpThreads ?? Math.max(1, Math.min(4, maxConcurrent));
      sess = makeSession(intra, inter);
      await warmSession(sess);
      debug(
        `InferencePool: expert mode — intra=${intra} inter=${inter} maxConcurrent=${maxConcurrent} (UV pool=${poolCap})`,
      );
    } else if (profile === "latency") {
      // Latency profile: all cores to one request at a time.
      intra = usable;
      maxConcurrent = 1;
      inter = opts.interOpThreads ?? 1;
      sess = makeSession(intra, inter);
      await warmSession(sess);
      debug(`InferencePool: profile=latency — intra=${intra} maxConcurrent=1`);
    } else if (profile === "throughput") {
      // Throughput profile: split cores for concurrent requests, capped to UV pool.
      intra = Math.min(4, usable);
      maxConcurrent = Math.min(
        Math.max(1, Math.floor(usable / intra)),
        poolCap,
      );
      inter = opts.interOpThreads ?? Math.min(4, maxConcurrent);
      sess = makeSession(intra, inter);
      await warmSession(sess);
      debug(
        `InferencePool: profile=throughput — intra=${intra} maxConcurrent=${maxConcurrent} (UV pool=${poolCap})`,
      );
    } else {
      // ── Autotuner ──────────────────────────────────────────────────────────
      //
      // Scores each candidate on effective throughput = maxConcurrent / mean_ms.
      // Tiebreaker: when two configs are within AUTOTUNE_LATENCY_PREFERENCE_MARGIN
      // of each other, prefer higher intra (lower per-request latency).
      //
      // Candidates are tried low→high intra. The global oneDNN primitive cache
      // is shared across sessions, so later candidates benefit from earlier
      // compilations. We counter this by running more iterations (AUTOTUNE_ITERS=10)
      // so steady-state performance dominates over cold-start noise.

      // Candidates: filter to ≤ usable AND ≤ poolCap (no point in more
      // concurrency than the libuv pool can service).
      const candidates = AUTOTUNE_INTRA_CANDIDATES.filter((c) => c <= usable)
        .concat(usable)
        .filter((c, i, a) => a.indexOf(c) === i)
        .sort((a, b) => a - b);

      // Large models have a longer oneDNN cold-start per candidate. Reduce
      // iterations to keep total autotuner time under ~10s for 100MB+ models.
      const autotuneIters =
        modelBytes >= AUTOTUNE_LARGE_MODEL_BYTES
          ? AUTOTUNE_ITERS_LARGE
          : AUTOTUNE_ITERS;

      debug(
        `InferencePool: autotuning ${
          candidates.length
        } configs — UV pool=${poolCap} (${autotuneIters} concurrent rounds each, model=${(
          modelBytes /
          1024 /
          1024
        ).toFixed(1)}MB)...`,
      );

      // Track results for tiebreaker after all candidates measured.
      const results: {
        rps: number;
        meanMs: number;
        intra: number;
        conc: number;
        sess: Session;
      }[] = [];

      for (const candidateIntra of candidates) {
        // Clamp concurrency to what the UV pool can actually service.
        const candidateConc = Math.min(
          Math.max(1, Math.floor(usable / candidateIntra)),
          poolCap,
        );
        const candidateInter = Math.max(1, Math.min(4, candidateConc));
        const s = makeSession(candidateIntra, candidateInter);

        await warmSession(s);

        // ── Measure sustained throughput via sliding window pipeline ────────
        // Issue AUTOTUNE_ITERS × candidateConc total requests, maintaining
        // exactly candidateConc in-flight at all times (like a real workload).
        // This correctly measures sustained req/s for all concurrency levels:
        //   - High maxConc: concurrent memory-bus pressure is captured
        //   - Low maxConc (e.g. 1): slot stays busy — no idle gaps between rounds
        const totalReqs = autotuneIters * Math.max(candidateConc, 1);
        let completed = 0;
        let issued = 0;
        const inFlight = new Set<Promise<void>>();
        const t0 = performance.now();

        // Seed with candidateConc in-flight requests.
        while (issued < Math.min(candidateConc, totalReqs)) {
          const p: Promise<void> = s
            .runAsync(dummyFeeds, dummyFetches as any)
            .catch(() => {})
            .then(() => {
              completed++;
              inFlight.delete(p);
            });
          inFlight.add(p);
          issued++;
        }

        // As each completes, immediately issue the next — zero idle time.
        while (completed < totalReqs) {
          await Promise.race(inFlight);
          while (issued < totalReqs && inFlight.size < candidateConc) {
            const p: Promise<void> = s
              .runAsync(dummyFeeds, dummyFetches as any)
              .catch(() => {})
              .then(() => {
                completed++;
                inFlight.delete(p);
              });
            inFlight.add(p);
            issued++;
          }
        }

        const elapsedMs = performance.now() - t0;
        const rps = (totalReqs * 1000) / elapsedMs;
        const meanMs = elapsedMs / totalReqs;

        debug(
          `  intra=${String(candidateIntra).padStart(
            2,
          )} maxConc=${candidateConc} inter=${candidateInter} rps=${rps.toFixed(
            1,
          )} mean=${meanMs.toFixed(2)}ms`,
        );

        results.push({
          rps,
          meanMs,
          intra: candidateIntra,
          conc: candidateConc,
          sess: s,
        });
      }

      // Best = highest measured req/s. Tiebreaker: within margin, prefer
      // higher intra (lower per-request latency).
      const topRps = Math.max(...results.map((r) => r.rps));
      const threshold = topRps * (1 - AUTOTUNE_LATENCY_PREFERENCE_MARGIN);
      const contenders = results.filter((r) => r.rps >= threshold);
      const winner = contenders.reduce((a, b) => (a.intra > b.intra ? a : b));

      for (const r of results) if (r.intra !== winner.intra) r.sess.destroy();

      intra = winner.intra;
      maxConcurrent = winner.conc;
      inter = Math.max(1, Math.min(4, maxConcurrent));
      sess = winner.sess;

      debug(
        `InferencePool: autotuned — intra=${intra} inter=${inter} maxConcurrent=${maxConcurrent} reserved=${reserveCores} (${
          intra * maxConcurrent
        }/${usable} cores) [rps=${winner.rps.toFixed(
          1,
        )} mean=${winner.meanMs.toFixed(1)}ms]`,
      );
    }

    return new InferencePool({
      graph: g,
      sess,
      inputOp: inputOp!,
      outputOps: outputOps!,
      maxQueueDepth: opts.maxQueueDepth ?? 128,
      maxConcurrent,
      inputShape,
      modelPath: opts.modelPath,
    });
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /**
   * Resolved input shape from the model Placeholder.
   * Null dims are dynamic (batch dim is typically null).
   * Use this to allocate correctly-sized input buffers.
   */
  get resolvedInputShape(): (number | null)[] {
    return this._inputShape;
  }

  /** Number of requests currently queued (waiting for the session). */
  get queueDepth(): number {
    return this.queue.length;
  }

  /**
   * infer — submit an inference request.
   *
   * Runs immediately if the session is idle. Otherwise queues the request
   * (FIFO). Throws QueueFullError synchronously if maxQueueDepth is reached.
   *
   * Accepts data in multiple formats:
   * - Float32Array / Int32Array (fastest, zero-copy)
   * - JS arrays: [1, 2, 3, ...] or nested [[1, 2, ...], ...] (auto-flattened)
   *
   * @param inputData  Input tensor data (array or typed array)
   * @param inputShape [batchSize, ...dims]
   * @param inputDtype TF DType integer (1=FLOAT32, 3=INT32, ...). Default: 1 (FLOAT32).
   */
  infer(
    inputData: DataLike,
    inputShape: number[],
    inputDtype: number = 1, // TF_FLOAT
  ): Promise<PoolResult> {
    if (this.destroyed)
      return Promise.reject(new Error("InferencePool has been destroyed"));

    // Convert input data to a typed array
    let inputTyped: Float32Array | Int32Array;
    try {
      if (inputDtype === 3) {
        // INT32
        inputTyped = toInt32Array(inputData);
      } else {
        // Default to FLOAT32
        inputTyped = toFloat32Array(inputData);
      }
    } catch (e) {
      return Promise.reject(
        new Error(
          `infer() data conversion failed: ${(e as Error).message}. ` +
            `Expected data matching shape ${JSON.stringify(inputShape)}.`,
        ),
      );
    }

    // Dispatch immediately if a concurrent slot is available.
    // Multiple runAsync() calls on the same session are safe — TF's inter-op
    // scheduler overlaps independent ops across concurrent requests, which
    // is how tfjs-node achieves sub-linear latency scaling at high concurrency.
    if (this.active < this.maxConcurrent)
      return this._run(inputTyped, inputShape, inputDtype);

    // All slots occupied — queue the request.
    if (this.queue.length >= this.maxQueueDepth)
      throw new QueueFullError(this.queue.length);

    return new Promise<PoolResult>((resolve, reject) => {
      this.queue.push({
        inputData: inputTyped,
        inputShape,
        inputDtype,
        resolve,
        reject,
      });
    });
  }

  private async _run(
    inputTyped: Float32Array | Int32Array,
    inputShape: number[],
    inputDtype: number,
  ): Promise<PoolResult> {
    this.active++;
    const t0 = performance.now();
    try {
      // Create a Buffer view of the typed array for TensorFlow
      const inputBuf = Buffer.from(
        inputTyped.buffer,
        inputTyped.byteOffset,
        inputTyped.byteLength,
      );

      const feeds = [
        [
          { opName: this.inputOp, index: 0 },
          { dtype: inputDtype as DType, shape: inputShape, data: inputBuf },
        ],
      ] as [any, FeedValue][];

      const fetches = this.outputOps.map((op) => ({ opName: op, index: 0 }));
      const outputs = await this.sess.runAsync(feeds as any, fetches as any);
      return { outputs, inferenceMs: performance.now() - t0 };
    } finally {
      this.active--;
      this._drain();
    }
  }

  private _drain(): void {
    // Dispatch as many queued requests as concurrent slots allow.
    while (
      this.queue.length > 0 &&
      this.active < this.maxConcurrent &&
      !this.destroyed
    ) {
      const next = this.queue.shift()!;
      this._run(next.inputData, next.inputShape, next.inputDtype).then(
        next.resolve,
        next.reject,
      );
    }
  }

  /**
   * runBatch — process an array of inputs through the pool in order.
   *
   * Designed for offline batch processing (CLI tools, data pipelines) where
   * you want to pipeline work through the pool without per-item async overhead.
   * All items share the same shape and dtype.
   *
   * Internally submits all items via infer() in parallel up to maxConcurrent,
   * then collects results in submission order. The event loop stays free
   * throughout — this is not a synchronous blocking call.
   *
   * @param inputs     Array of raw tensor buffers (one per item)
   * @param inputShape Shape for every item in the batch (same for all)
   * @param inputDtype TF dtype integer (default: 1 = FLOAT32)
   * @returns          Results in the same order as inputs
   *
   * @example
   * const results = await pool.runBatch(
   *   images.map(img => img.buffer),
   *   [1, 224, 224, 3],
   * );
   */
  async runBatch(
    inputs: Buffer[],
    inputShape: number[],
    inputDtype: number = 1,
  ): Promise<PoolResult[]> {
    if (this.destroyed) throw new Error("InferencePool has been destroyed");
    if (inputs.length === 0) return [];

    // Submit all items concurrently (pool enforces maxConcurrent internally).
    // We track promises in submission order to return results sorted correctly.
    const promises = inputs.map((buf) =>
      this.infer(buf, inputShape, inputDtype),
    );
    return Promise.all(promises);
  }

  /**
   * destroy — reject all queued requests and close the TF session.
   */
  async destroy(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;

    const err = new Error("InferencePool destroyed");
    while (this.queue.length > 0) this.queue.shift()!.reject(err);

    this.sess.destroy();
  }
}
