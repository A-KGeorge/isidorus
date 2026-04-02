/**
 * InferencePool — strategy-aware inference execution with CPU affinity.
 *
 * Strategies:
 *   worker-pool  N Workers × native Session(intra=1, inter=1)
 *                JS controls parallelism. Each Worker owns one Session.
 *                N concurrent requests run on N cores simultaneously.
 *                Best: small/medium models, high concurrency.
 *
 *   tf-parallel  1 native Session × (intra=hw−reserveCores, inter=1)
 *                TF's eigen threadpool owns all TF cores for one request.
 *                Concurrent requests queue behind each other.
 *                Best: large models where one matmul fills all cores.
 *
 *   auto         Probe-based selection:
 *                  model < 150 MB                 → worker-pool (no probe)
 *                  model ≥ 150 MB + probeShape     → warm probe → threshold
 *                  model ≥ 150 MB, no probeShape   → tf-parallel (fallback)
 *
 * CPU affinity (reserveCores):
 *   Pins TF compute to the LAST (N−reserveCores) cores via OS affinity fence
 *   applied in OnRunWork immediately before/after TF_SessionRun.
 *
 * Transport:
 *   Control plane (state machine) — 4-slot Int32Array over a SharedArrayBuffer.
 *     Main stores WORK, Worker Atomics.wait()s and observes it. Tiny, exact.
 *
 *   Data plane (tensor bytes) — one SharedTensorSegment (jude-map) per slot.
 *     Main calls seg.write(shape, dtype, bytes) — seqlock write, zero copy.
 *     Worker calls seg.read()                  — seqlock read, zero copy.
 *     No postMessage for input data. No SAB size limit concern (control SAB
 *     is 16 bytes per worker; data segments are sized to the model's input).
 *     Atomics are slower than seqlocks — jude-map's seqlock handles the data
 *     plane; Atomics handle only the 4-slot state machine.
 */

import { Worker, isMainThread, parentPort, workerData } from "worker_threads";
import { availableParallelism } from "os";
import { statSync, readFileSync } from "fs";
import { performance } from "perf_hooks";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import nodeGypBuild from "node-gyp-build";
import { SharedTensorSegment, DType as JudeMapDType } from "jude-map";
import type { Graph } from "./graph.js";
import type { Session } from "./session.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ─── Constants ──────────────────────────────────────────────────────────────

const IDLE = 0;
const WORK = 1;
const DONE = 2;
const SHUTDOWN = 3;
const CTRL_SLOTS = 4; // state-machine only — IDLE/WORK/DONE/SHUTDOWN

const SIZE_THRESHOLD_BYTES = 150 * 1024 * 1024; // 150 MB
const DEFAULT_AUTO_THRESHOLD = 20; // ms

// Default SharedTensorSegment capacity per Worker slot (bytes).
// 4 MB covers MobileNetV2 (224×224×3×4 ≈ 602 KB) and ResNet50 single-image
// inputs with comfortable headroom. Increase via PoolOptions.maxInputBytes
// for large batch sizes or video frame inputs.
const DEFAULT_MAX_INPUT_BYTES = 4 * 1024 * 1024;

// ─── Types ──────────────────────────────────────────────────────────────────

export type ExecutionStrategy = "worker-pool" | "tf-parallel" | "auto";

export interface PoolOptions {
  modelPath: string;
  /**
   * Input op name. Optional — if omitted the first Placeholder op in the
   * frozen graph is used (auto-discover via native graph scan).
   */
  inputOp?: string;
  /**
   * Output op names. Optional — if omitted the sink ops (ops whose outputs
   * are not consumed by any other op) are used.
   */
  outputOps?: string[];
  /**
   * Reserve the first R cores for the event loop and other native libs.
   * TF inference is pinned to the last (N−R) cores via OS affinity.
   * Default: 0 (no reservation).
   */
  reserveCores?: number;
  strategy?: ExecutionStrategy;
  /** Worker thread count (worker-pool). Default: os.availableParallelism(). */
  concurrency?: number;
  /** Input shape for auto probe. Required when model >= 150 MB. */
  probeShape?: number[];
  /** Crossover threshold for auto strategy (ms). Default: 20. */
  autoThresholdMs?: number;
  /**
   * Capacity of each Worker's SharedTensorSegment data buffer (bytes).
   * Must be >= the byte size of the largest input tensor you will feed.
   * Default: 4 MB.
   */
  maxInputBytes?: number;
}

export interface PoolResult {
  workerId: number;
  strategy: "worker-pool" | "tf-parallel";
  outputs: { dtype: number; shape: number[]; data: Buffer }[];
  inferenceMs: number;
}

// ─── jude-map DType bridge ───────────────────────────────────────────────────
// TF_DataType integers used by the native Session match jude-map's DType enum
// values — both mirror the TensorFlow wire format. We cast directly.
function tfDtypeToJudeMap(dtype: number): JudeMapDType {
  return dtype as unknown as JudeMapDType;
}

// ─── Worker-side logic ──────────────────────────────────────────────────────
//
// Runs when this file is loaded by inference-pool-worker.mjs (tsx bootstrap)
// or by the compiled .js entry as a Worker thread.
//
// Transport:
//   Control plane  — Int32Array over ctrlSab (4 slots, state machine only)
//   Data plane     — SharedTensorSegment reconstructed from segSab (jude-map)
//
// Init:
//   Loads the native addon directly (no import from @isidorus/cpu — that would
//   re-run ensureTf() and create a circular module reference).
//   Loads the frozen graph via importGraphDef(readFileSync(modelPath)).
//
// Work loop:
//   Atomics.wait(ctrl, 0, IDLE)         ← park
//   const { data, shape, dtype } = seg.read()  ← seqlock read, zero copy
//   results = await sess.runAsync(feeds, fetches)
//   Atomics.store(ctrl, 0, DONE) + notify
//   postMessage({ type: "result", ... })

if (!isMainThread) {
  const {
    ctrlSab,
    segSab,
    maxInputBytes,
    workerIndex,
    modelPath,
    inputOp,
    outputOps,
    reserveCores,
  } = workerData as {
    ctrlSab: SharedArrayBuffer;
    segSab: SharedArrayBuffer;
    maxInputBytes: number;
    workerIndex: number;
    modelPath: string;
    inputOp: string;
    outputOps: string[];
    reserveCores: number;
  };

  const ctrl = new Int32Array(
    ctrlSab,
    workerIndex * CTRL_SLOTS * 4,
    CTRL_SLOTS,
  );

  // ── Init ─────────────────────────────────────────────────────────────────
  let sess: any;
  try {
    // Load the native addon from the package root. Workers inherit
    // LIBTENSORFLOW_PATH and PATH so the addon finds libtensorflow without
    // re-running ensureTf().

    let workerAddon: any;
    try {
      const addon = nodeGypBuild(join(__dirname, "..")) as any;
      workerAddon = addon.SharedTensor ?? addon;
    } catch (e) {
      try {
        const addon = nodeGypBuild(join(__dirname, "..", "..")) as any;
        workerAddon = addon.SharedTensor ?? addon;
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

    const nativeGraph = new workerAddon.Graph();
    nativeGraph.importGraphDef(readFileSync(modelPath));
    sess = new workerAddon.Session(nativeGraph, {
      strategy: "worker-pool", // intra=1, inter=1
      reserveCores,
    });
  } catch (err: any) {
    parentPort!.postMessage({
      type: "init_error",
      error: err?.stack ?? String(err),
    });
    process.exit(1);
  }

  // Reconstruct the SharedTensorSegment from the SAB passed via workerData.
  // jude-map segments are SAB-backed — the same underlying memory is accessible
  // from both the main thread and this Worker.
  const seg = SharedTensorSegment.fromSharedBuffer(segSab, maxInputBytes);

  Atomics.store(ctrl, 0, IDLE);
  parentPort!.postMessage({ type: "ready" });

  // ── Work loop ─────────────────────────────────────────────────────────────
  // Atomics.wait BLOCKS — intentional. This Worker is dedicated to inference
  // and has no other async work to process while parked.
  while (true) {
    Atomics.wait(ctrl, 0, IDLE);
    const state = Atomics.load(ctrl, 0);

    if (state === SHUTDOWN) {
      sess.destroy();
      seg.destroy();
      parentPort!.postMessage({ type: "shutdown_ack" });
      break;
    }

    if (state === WORK) {
      try {
        const t0 = performance.now();

        // Zero-copy read — seqlock guarantees a consistent snapshot.
        // Main thread called seg.write() before storing WORK, so the data
        // is fully committed before we observe WORK here.
        const tensor = seg.read();
        if (!tensor)
          throw new Error("seg.read() returned null — segment not written");

        const feeds = [
          {
            opName: inputOp,
            index: 0,
            tensor: {
              dtype: tensor.dtype as unknown as number,
              shape: tensor.shape,
              data: Buffer.from(
                (tensor.data as ArrayBufferView).buffer,
                (tensor.data as ArrayBufferView).byteOffset,
                (tensor.data as ArrayBufferView).byteLength,
              ),
            },
          },
        ];
        const fetches = outputOps.map((op: string) => ({
          opName: op,
          index: 0,
        }));

        // runAsync pushes TF_SessionRun onto the Worker's own libuv thread
        // pool, keeping the Worker's event loop free for Atomics signalling.
        const rawOutputs = (await sess.runAsync(feeds, fetches)) as Array<{
          dtype: number;
          shape: number[];
          data: Buffer;
        }>;
        const inferenceMs = performance.now() - t0;

        // Signal done before postMessage so main thread can unblock.
        Atomics.store(ctrl, 0, DONE);
        Atomics.notify(ctrl, 0, 1);

        parentPort!.postMessage({
          type: "result",
          outputs: outputOps.map((_: string, i: number) => rawOutputs[i]),
          inferenceMs,
        });
      } catch (err: any) {
        Atomics.store(ctrl, 0, DONE);
        Atomics.notify(ctrl, 0, 1);
        parentPort!.postMessage({
          type: "work_error",
          error: err?.stack ?? String(err),
        });
      }

      // compareExchange: if main wrote SHUTDOWN between DONE and here, keep it.
      Atomics.compareExchange(ctrl, 0, DONE, IDLE);
    }
  }
}

// ─── Main-thread InferencePool ──────────────────────────────────────────────

interface WorkerSlot {
  worker: Worker;
  ctrl: Int32Array; // 4-slot state machine
  seg: SharedTensorSegment; // jude-map data transport
  busy: boolean;
  resolve: ((r: PoolResult) => void) | null;
  reject: ((e: Error) => void) | null;
}

interface QueueEntry {
  inputBuf: Buffer;
  inputShape: number[];
  inputDtype: number;
  resolve: (r: PoolResult) => void;
  reject: (e: Error) => void;
}

export class InferencePool {
  readonly strategy: "worker-pool" | "tf-parallel";
  readonly reserveCores: number;

  private readonly workerSlots: WorkerSlot[];
  private readonly queue: QueueEntry[];
  private readonly ctrlSab: SharedArrayBuffer | null;
  private tfParallelGraph: Graph | null;
  private tfParallelSess: Session | null;
  private tfParallelBusy: boolean;
  private readonly tfParallelQueue: QueueEntry[];
  private readonly modelPath: string;
  private readonly inputOp: string;
  private readonly outputOps: string[];

  private constructor(params: {
    strategy: "worker-pool" | "tf-parallel";
    reserveCores: number;
    workerSlots: WorkerSlot[];
    queue: QueueEntry[];
    ctrlSab: SharedArrayBuffer | null;
    tfParallelGraph: Graph | null;
    tfParallelSess: Session | null;
    modelPath: string;
    inputOp: string;
    outputOps: string[];
  }) {
    this.strategy = params.strategy;
    this.reserveCores = params.reserveCores;
    this.workerSlots = params.workerSlots;
    this.queue = params.queue;
    this.ctrlSab = params.ctrlSab;
    this.tfParallelGraph = params.tfParallelGraph;
    this.tfParallelSess = params.tfParallelSess;
    this.tfParallelBusy = false;
    this.tfParallelQueue = [];
    this.modelPath = params.modelPath;
    this.inputOp = params.inputOp;
    this.outputOps = params.outputOps;
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  static async create(opts: PoolOptions): Promise<InferencePool> {
    // ── Auto-discover inputOp / outputOps if not provided ──────────────────
    // Load the graph once, scan for Placeholder ops (inputs) and sink ops
    // (outputs whose results nothing else consumes). No jude-tf needed.
    if (!opts.inputOp || !opts.outputOps?.length) {
      const { getAddon } = await import("./_native.js");
      const { Graph: GCls } = await import("./graph.js");
      const addon = getAddon();
      const g = new GCls(new addon.Graph());
      g.importGraphDef(readFileSync(opts.modelPath));

      if (!opts.inputOp) {
        const placeholders = g.listOpsOfType("Placeholder");
        if (!placeholders.length)
          throw new Error(`No Placeholder ops found in ${opts.modelPath}`);
        opts.inputOp = placeholders[0];
      }
      if (!opts.outputOps?.length) {
        const sinks = g.listSinkOps();
        if (!sinks.length)
          throw new Error(`No sink ops found in ${opts.modelPath}`);
        opts.outputOps = sinks;
      }
      // g is garbage-collected — no explicit destroy needed for a probe graph.
    }

    const requestedStrategy = opts.strategy ?? "auto";
    const concurrency = opts.concurrency ?? availableParallelism();
    const autoThreshold = opts.autoThresholdMs ?? DEFAULT_AUTO_THRESHOLD;
    const reserveCores = opts.reserveCores ?? 0;
    const maxInputBytes = opts.maxInputBytes ?? DEFAULT_MAX_INPUT_BYTES;

    let resolved: "worker-pool" | "tf-parallel";

    if (requestedStrategy === "worker-pool") {
      resolved = "worker-pool";
    } else if (requestedStrategy === "tf-parallel") {
      resolved = "tf-parallel";
    } else {
      const modelBytes = statSync(opts.modelPath).size;

      if (modelBytes < SIZE_THRESHOLD_BYTES) {
        resolved = "worker-pool";
      } else if (!opts.probeShape) {
        resolved = "tf-parallel";
        process.stderr.write(
          `[isidorus] auto: ${(modelBytes / 1024 / 1024).toFixed(1)}MB >= ` +
            `threshold, no probeShape → tf-parallel\n`,
        );
      } else {
        // Warm probe via native Session (intra=1) to measure single-core time.
        const { getAddon } = await import("./_native.js");
        const { Graph: GCls } = await import("./graph.js");
        const addon = getAddon();
        const probeG = new GCls(new addon.Graph());
        probeG.importGraphDef(readFileSync(opts.modelPath));
        const probeSess = new addon.Session(probeG._native, {
          strategy: "worker-pool",
          reserveCores: 0,
        });

        const probeElems = opts.probeShape.reduce(
          (a: number, b: number) => a * b,
          1,
        );
        const probeInput = Buffer.alloc(probeElems * 4);
        const probeFeeds = [
          {
            opName: opts.inputOp,
            index: 0,
            tensor: { dtype: 1, shape: opts.probeShape, data: probeInput },
          },
        ];
        const probeFetches = opts.outputOps!.map((op: string) => ({
          opName: op,
          index: 0,
        }));

        // Warmup
        await probeSess.runAsync(probeFeeds, probeFetches);
        const t0 = performance.now();
        await probeSess.runAsync(probeFeeds, probeFetches);
        const probeMs = performance.now() - t0;
        probeSess.destroy();

        resolved = probeMs >= autoThreshold ? "tf-parallel" : "worker-pool";
        process.stderr.write(
          `[isidorus] auto: probe=${probeMs.toFixed(2)}ms ` +
            `threshold=${autoThreshold}ms → ${resolved}\n`,
        );
      }
    }

    if (reserveCores > 0) {
      const tfCores = Math.max(1, availableParallelism() - reserveCores);
      process.stderr.write(
        `[isidorus] CPU affinity: reserving ${reserveCores} core(s), ` +
          `TF gets ${tfCores} core(s)\n`,
      );
    }

    return resolved === "worker-pool"
      ? InferencePool.createWorkerPool(
          opts,
          concurrency,
          reserveCores,
          maxInputBytes,
        )
      : InferencePool.createTfParallel(opts, reserveCores);
  }

  // ── worker-pool init ───────────────────────────────────────────────────────

  private static async createWorkerPool(
    opts: PoolOptions,
    concurrency: number,
    reserveCores: number,
    maxInputBytes: number,
  ): Promise<InferencePool> {
    const ctrlSab = new SharedArrayBuffer(concurrency * CTRL_SLOTS * 4);
    const slots: WorkerSlot[] = [];
    const startedWorkers: Worker[] = [];

    // In dev/test we run TypeScript source directly via tsx.
    // inference-pool-worker.mjs calls register() from tsx/esm/api before
    // importing this .ts file. In production the compiled .js is used directly.
    const isTsSource = import.meta.url.endsWith(".ts");
    const workerEntry = isTsSource
      ? new URL("./inference-pool-worker.mjs", import.meta.url)
      : new URL("./inference-pool.js", import.meta.url);

    try {
      for (let i = 0; i < concurrency; i++) {
        const ctrl = new Int32Array(ctrlSab, i * CTRL_SLOTS * 4, CTRL_SLOTS);
        Atomics.store(ctrl, 0, IDLE);

        // One SharedTensorSegment per slot — zero-copy data transport.
        // createShared() allocates a SharedArrayBuffer backing the seqlock
        // + data region. The SAB is passed to the Worker so both sides
        // share the same physical memory with no cross-thread copy.
        // Must use createShared(), not new SharedTensorSegment() — the
        // mmap constructor produces a process-local mapping that has no SAB
        // and cannot be transferred to a Worker via workerData.
        const seg = SharedTensorSegment.createShared(maxInputBytes);
        const segSab = seg.sharedBuffer; // the backing SAB

        const worker = new Worker(workerEntry, {
          workerData: {
            ctrlSab,
            segSab,
            maxInputBytes,
            workerIndex: i,
            modelPath: opts.modelPath,
            inputOp: opts.inputOp,
            outputOps: opts.outputOps,
            reserveCores,
          },
        });
        startedWorkers.push(worker);

        await new Promise<void>((resolve, reject) => {
          worker.once("message", (msg: any) => {
            if (msg.type === "ready") resolve();
            else if (msg.type === "init_error")
              reject(new Error(`Worker ${i} init failed: ${msg.error}`));
            else
              reject(
                new Error(`Worker ${i} unexpected init message: ${msg.type}`),
              );
          });
          worker.once("error", reject);
        });

        slots.push({
          worker,
          ctrl,
          seg,
          busy: false,
          resolve: null,
          reject: null,
        });
      }
    } catch (err) {
      await Promise.allSettled(startedWorkers.map((w) => w.terminate()));
      throw err;
    }

    return new InferencePool({
      strategy: "worker-pool",
      reserveCores,
      workerSlots: slots,
      queue: [],
      ctrlSab,
      tfParallelGraph: null,
      tfParallelSess: null,
      modelPath: opts.modelPath,
      inputOp: opts.inputOp!,
      outputOps: opts.outputOps!,
    });
  }

  // ── tf-parallel init ───────────────────────────────────────────────────────

  private static async createTfParallel(
    opts: PoolOptions,
    reserveCores: number,
  ): Promise<InferencePool> {
    const hw = availableParallelism();
    const tfCores = Math.max(1, hw - reserveCores);

    const { getAddon } = await import("./_native.js");
    const { Graph: GCls } = await import("./graph.js");
    const { Session: SCls } = await import("./session.js");
    const addon = getAddon();

    const g = new GCls(new addon.Graph());
    g.importGraphDef(readFileSync(opts.modelPath));

    const sess = new SCls(
      new addon.Session(g._native, {
        strategy: "tf-parallel",
        reserveCores,
      }),
    );

    process.stderr.write(
      `[isidorus] tf-parallel: intra_op=${tfCores} ` +
        `(${reserveCores} core(s) reserved, native Session)\n`,
    );

    return new InferencePool({
      strategy: "tf-parallel",
      reserveCores,
      workerSlots: [],
      queue: [],
      ctrlSab: null,
      tfParallelGraph: g,
      tfParallelSess: sess,
      modelPath: opts.modelPath,
      inputOp: opts.inputOp!,
      outputOps: opts.outputOps!,
    });
  }

  // ── Inference ──────────────────────────────────────────────────────────────

  infer(
    inputBuf: Buffer,
    inputShape: number[],
    inputDtype = 1,
  ): Promise<PoolResult> {
    return this.strategy === "worker-pool"
      ? this.inferWorkerPool(inputBuf, inputShape, inputDtype)
      : this.inferTfParallel(inputBuf, inputShape, inputDtype);
  }

  private inferWorkerPool(
    inputBuf: Buffer,
    inputShape: number[],
    inputDtype: number,
  ): Promise<PoolResult> {
    return new Promise((resolve, reject) => {
      const slot = this.workerSlots.find((w) => !w.busy);
      if (slot)
        this.dispatchToWorker(
          slot,
          inputBuf,
          inputShape,
          inputDtype,
          resolve,
          reject,
        );
      else
        this.queue.push({ inputBuf, inputShape, inputDtype, resolve, reject });
    });
  }

  private dispatchToWorker(
    slot: WorkerSlot,
    inputBuf: Buffer,
    inputShape: number[],
    inputDtype: number,
    resolve: (r: PoolResult) => void,
    reject: (e: Error) => void,
  ) {
    const workerId = this.workerSlots.indexOf(slot);
    slot.busy = true;
    slot.resolve = resolve;
    slot.reject = reject;

    // Register result listener BEFORE writing data or signalling WORK.
    const handleMessage = (msg: any) => {
      if (msg.type === "work_error") {
        this.settleSlot(slot, null, new Error(msg.error));
        return;
      }
      if (msg.type !== "result") {
        slot.worker.once("message", handleMessage);
        return;
      }
      this.settleSlot(
        slot,
        {
          workerId,
          strategy: "worker-pool",
          outputs: msg.outputs.map((o: any) => ({
            dtype: o.dtype,
            shape: o.shape,
            // postMessage structured-clones Buffer as Uint8Array — rewrap.
            data: Buffer.isBuffer(o.data) ? o.data : Buffer.from(o.data),
          })),
          inferenceMs: msg.inferenceMs,
        },
        null,
      );
    };
    slot.worker.once("message", handleMessage);
    slot.worker.once("error", (err: Error) => this.settleSlot(slot, null, err));

    // Zero-copy write — seqlock ensures the Worker sees a consistent snapshot.
    // No postMessage for input data. The Worker reads via seg.read() after
    // observing WORK on the control SAB.
    slot.seg.write(inputShape, tfDtypeToJudeMap(inputDtype), inputBuf);

    Atomics.store(slot.ctrl, 0, WORK);
    Atomics.notify(slot.ctrl, 0, 1);
  }

  private settleSlot(
    slot: WorkerSlot,
    result: PoolResult | null,
    err: Error | null,
  ) {
    const resolve = slot.resolve;
    const reject = slot.reject;
    slot.busy = false;
    slot.resolve = null;
    slot.reject = null;

    if (err) reject?.(err);
    else resolve?.(result!);

    const next = this.queue.shift();
    if (next)
      this.dispatchToWorker(
        slot,
        next.inputBuf,
        next.inputShape,
        next.inputDtype,
        next.resolve,
        next.reject,
      );
  }

  // ── tf-parallel path ───────────────────────────────────────────────────────

  private inferTfParallel(
    inputBuf: Buffer,
    inputShape: number[],
    inputDtype: number,
  ): Promise<PoolResult> {
    return new Promise((resolve, reject) => {
      if (this.tfParallelBusy) {
        this.tfParallelQueue.push({
          inputBuf,
          inputShape,
          inputDtype,
          resolve,
          reject,
        });
        return;
      }
      this.runTfParallel(inputBuf, inputShape, inputDtype, resolve, reject);
    });
  }

  private runTfParallel(
    inputBuf: Buffer,
    inputShape: number[],
    inputDtype: number,
    resolve: (r: PoolResult) => void,
    reject: (e: Error) => void,
  ) {
    this.tfParallelBusy = true;
    const t0 = performance.now();

    // Native Session feed/fetch format — Graph.getOp() resolves op names to
    // Tensor descriptors that Session.runAsync() expects.
    const g = this.tfParallelGraph!;
    const inputTensor = g.getOp(this.inputOp);
    if (!inputTensor) {
      this.tfParallelBusy = false;
      reject(new Error(`tf-parallel: input op not found: ${this.inputOp}`));
      return;
    }
    const outputTensors = this.outputOps.map((name) => {
      const t = g.getOp(name);
      if (!t) throw new Error(`tf-parallel: output op not found: ${name}`);
      return t;
    });

    const feedValue = {
      dtype: inputDtype,
      shape: inputShape,
      data: inputBuf,
    };

    this.tfParallelSess!.runAsync([[inputTensor, feedValue]], outputTensors)
      .then((outputs: any[]) => {
        const inferenceMs = performance.now() - t0;
        this.tfParallelBusy = false;
        resolve({
          workerId: 0,
          strategy: "tf-parallel",
          outputs: outputs.map((o) => ({
            dtype: o.dtype,
            shape: o.shape,
            data: Buffer.isBuffer(o.data) ? o.data : Buffer.from(o.data),
          })),
          inferenceMs,
        });
        const next = this.tfParallelQueue.shift();
        if (next)
          this.runTfParallel(
            next.inputBuf,
            next.inputShape,
            next.inputDtype,
            next.resolve,
            next.reject,
          );
      })
      .catch((err: Error) => {
        this.tfParallelBusy = false;
        reject(err);
        const next = this.tfParallelQueue.shift();
        if (next)
          this.runTfParallel(
            next.inputBuf,
            next.inputShape,
            next.inputDtype,
            next.resolve,
            next.reject,
          );
      });
  }

  // ── Introspection ──────────────────────────────────────────────────────────

  get busyCount(): number {
    if (this.strategy === "worker-pool")
      return this.workerSlots.filter((w) => w.busy).length;
    return this.tfParallelBusy ? 1 : 0;
  }

  get queueDepth(): number {
    return this.strategy === "worker-pool"
      ? this.queue.length
      : this.tfParallelQueue.length;
  }

  get size(): number {
    return this.strategy === "worker-pool" ? this.workerSlots.length : 1;
  }

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  async destroy(): Promise<void> {
    if (this.strategy === "worker-pool") {
      await Promise.all(
        this.workerSlots.map(
          (slot) =>
            new Promise<void>((resolve, reject) => {
              const doShutdown = () => {
                slot.worker.once("message", (msg: any) => {
                  if (msg.type === "shutdown_ack") {
                    slot.seg.destroy(); // release jude-map segment after Worker exits
                    resolve();
                  }
                });
                slot.worker.once("error", reject);
                Atomics.store(slot.ctrl, 0, SHUTDOWN);
                Atomics.notify(slot.ctrl, 0, 1);
              };
              if (slot.busy) {
                const origResolve = slot.resolve;
                const origReject = slot.reject;
                slot.resolve = (r) => {
                  origResolve?.(r);
                  doShutdown();
                };
                slot.reject = (e) => {
                  origReject?.(e);
                  doShutdown();
                };
              } else {
                doShutdown();
              }
            }),
        ),
      );
    } else {
      if (this.tfParallelBusy) {
        await new Promise<void>((res) => {
          const t = setInterval(() => {
            if (!this.tfParallelBusy) {
              clearInterval(t);
              res();
            }
          }, 1);
        });
      }
      this.tfParallelSess?.destroy();
      this.tfParallelSess = null;
      this.tfParallelGraph = null;
    }
  }
}
