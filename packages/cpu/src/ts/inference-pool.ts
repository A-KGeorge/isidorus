/**
 * InferencePool — strategy-aware inference execution with CPU affinity.
 *
 * Strategies:
 *
 *   worker-pool  N Workers × Session(intra=1, inter=1)
 *                JS controls parallelism. Each Worker owns one Session.
 *                N concurrent requests run on N cores simultaneously.
 *                Best: small/medium models, high concurrency.
 *
 *   tf-parallel  1 Session × Session(intra=hw, inter=1)
 *                TF's eigen threadpool owns all TF cores for one request.
 *                Concurrent requests queue behind each other.
 *                Best: large models where one matmul fills all cores.
 *
 *   auto         Probe-based selection:
 *                  model < 150 MB                   → worker-pool (no probe)
 *                  model ≥ 150 MB + probeShape       → warm probe → threshold
 *                  model ≥ 150 MB, no probeShape     → tf-parallel (fallback)
 *
 * CPU affinity (reserveCores):
 *   reserveCores = R pins TF computation to the LAST (N-R) cores.
 *   The FIRST R cores stay free for the event loop, libuv I/O, opencv, etc.
 *   The fence is applied in OnRunWork immediately before/after TF_SessionRun.
 */

import { Worker, isMainThread, parentPort, workerData } from "worker_threads";
import { availableParallelism } from "os";
import { statSync } from "fs";
import { performance } from "perf_hooks";
import type { Graph } from "./graph.js";
import type { Session } from "./session.js";

// ─── Constants ──────────────────────────────────────────────────────────────
const IDLE = 0;
const WORK = 1;
const DONE = 2;
const SHUTDOWN = 3;
const CTRL_SLOTS = 4; // Int32 slots per worker in the control SAB

const SIZE_THRESHOLD_BYTES = 150 * 1024 * 1024; // 150 MB
const DEFAULT_AUTO_THRESHOLD = 20; // ms

// ─── Types ──────────────────────────────────────────────────────────────────

export type ExecutionStrategy = "worker-pool" | "tf-parallel" | "auto";

export interface PoolOptions {
  modelPath: string;
  inputOp?: string;
  outputOps?: string[];
  /**
   * Reserve the first R cores for the event loop and other native libs.
   * TF inference is pinned to the last (N-R) cores via OS affinity.
   * Default: 0 (no reservation).
   */
  reserveCores?: number;
  strategy?: ExecutionStrategy;
  /** Worker thread count (worker-pool only). Default: os.availableParallelism(). */
  concurrency?: number;
  /** Input shape for auto probe. Required when model >= 150 MB. */
  probeShape?: number[];
  /** Crossover threshold for auto strategy (ms). Default: 20. */
  autoThresholdMs?: number;
}

export interface PoolResult {
  workerId: number;
  strategy: "worker-pool" | "tf-parallel";
  outputs: { dtype: number; shape: number[]; data: Buffer }[];
  inferenceMs: number;
}

// ─── Worker-side logic ──────────────────────────────────────────────────────
//
// This block runs when the same file is loaded as a Worker thread.
// The worker owns exactly one TFSession with intra_op=1, so all parallelism
// is expressed at the Worker level (N workers = N cores).
//
// Control protocol (per-worker Int32Array over a SharedArrayBuffer):
//   slot[0] = IDLE      → parked, waiting for work
//             WORK      → main thread has a request ready
//             DONE      → worker finished, result sent via postMessage
//             SHUTDOWN  → main thread requests exit
//
// Message protocol (postMessage, ordered relative to Atomics):
//   main → worker:  { inputData: Buffer, inputShape: number[], inputDtype: number }
//   worker → main:  { type: "ready" }
//                   { type: "result", outputs: TensorValue[], inferenceMs: number }
//                   { type: "work_error", error: string }
//                   { type: "shutdown_ack" }
//
// Ordering guarantee:
//   Main posts the input message BEFORE storing WORK + notifying,
//   so the worker's Atomics.wait wakes AFTER the message is queued.
//   Node.js buffers port messages until a listener is registered,
//   so parentPort.once("message", ...) safely receives the queued message.

if (!isMainThread) {
  const { ctrlSab, workerIndex, modelPath, inputOp, outputOps } =
    workerData as {
      ctrlSab: SharedArrayBuffer;
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

  // ── Init ──────────────────────────────────────────────────────────────────
  let sess: any;
  try {
    const { TFSession } = await import("jude-tf");
    sess = await TFSession.loadFrozenGraph(modelPath);
  } catch (err: any) {
    parentPort!.postMessage({
      type: "init_error",
      error: err?.stack ?? String(err),
    });
    process.exit(1);
  }

  Atomics.store(ctrl, 0, IDLE);
  parentPort!.postMessage({ type: "ready" });

  // Reinterpret raw bytes as the correct typed array for jude-tf.
  // postMessage structured-clone strips the Buffer prototype, so the worker
  // receives a plain Uint8Array. We reinterpret the underlying bytes as the
  // correct typed array so jude-tf sees the right TF_DataType.
  function asTypedArray(buf: Buffer, dtype: number): ArrayBufferView {
    const ab = buf.buffer.slice(
      buf.byteOffset,
      buf.byteOffset + buf.byteLength,
    );
    switch (dtype) {
      case 1:
        return new Float32Array(ab); // TF_FLOAT
      case 2:
        return new Float64Array(ab); // TF_DOUBLE
      case 3:
        return new Int32Array(ab); // TF_INT32
      case 4:
        return new Uint8Array(ab); // TF_UINT8
      case 9:
        return new BigInt64Array(ab); // TF_INT64
      default:
        return new Uint8Array(ab);
    }
  }

  // ── Work loop ─────────────────────────────────────────────────────────────
  // Atomics.wait BLOCKS the worker thread (allowed in Worker threads, not in
  // the main thread). The block is intentional — the worker is dedicated to
  // inference and has no other async work to process while parked.
  while (true) {
    // Park until main thread sets ctrl to WORK or SHUTDOWN.
    Atomics.wait(ctrl, 0, IDLE);
    const state = Atomics.load(ctrl, 0);

    if (state === SHUTDOWN) {
      sess.destroy();
      parentPort!.postMessage({ type: "shutdown_ack" });
      break;
    }

    if (state === WORK) {
      // Receive input data. Main posted it BEFORE storing WORK,
      // so the message is already queued in the port's buffer.
      const msg: any = await new Promise((resolve) =>
        parentPort!.once("message", resolve),
      );

      try {
        const t0 = performance.now();
        const inputArray = asTypedArray(
          msg.inputData as Buffer,
          msg.inputDtype as number,
        );

        const results = await sess.run({ [inputOp]: inputArray }, outputOps);

        const inferenceMs = performance.now() - t0;

        Atomics.store(ctrl, 0, DONE);
        Atomics.notify(ctrl, 0, 1);

        parentPort!.postMessage({
          type: "result",
          outputs: outputOps.map((k) => {
            const r = results[k];
            const view = r.data as ArrayBufferView;
            return {
              dtype: r.dtype,
              shape: r.shape,
              // Copy into a Buffer — postMessage will structured-clone it as
              // Uint8Array on the receiving end; the main thread wraps it back
              // into a Buffer in handleMessage.
              data: Buffer.from(view.buffer, view.byteOffset, view.byteLength),
            };
          }),
          inferenceMs,
        });
      } catch (err: any) {
        console.error(
          `[worker ${workerIndex}] error:`,
          err?.stack ?? String(err),
        );
        Atomics.store(ctrl, 0, DONE);
        Atomics.notify(ctrl, 0, 1);
        parentPort!.postMessage({
          type: "work_error",
          error: err?.stack ?? String(err),
        });
      }

      // Use compareExchange instead of store — if the main thread wrote
      // SHUTDOWN between our DONE store and here, don't overwrite it.
      // Otherwise the next Atomics.wait(ctrl, 0, IDLE) would block forever
      // waiting for a notify that never comes.
      Atomics.compareExchange(ctrl, 0, DONE, IDLE);
    }
  }
}

// ─── Main-thread InferencePool ──────────────────────────────────────────────

interface WorkerSlot {
  worker: Worker;
  ctrl: Int32Array;
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

  // tf-parallel path — exactly one of these pairs is non-null:
  //   (tfParallelGraph, tfParallelSess) — native @isidorus/cpu Session
  //                                       uses Graph.getOp() for feed/fetch
  //   (null, tfParallelSess)            — jude-tf TFSession fallback
  //                                       uses dict API { [opName]: data }
  private tfParallelGraph: Graph | null;
  private tfParallelSess: Session | any | null;
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
    tfParallelSess: Session | any | null;
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
    // Auto-discover input/output op names if not provided.
    // Loads the frozen graph once via jude-tf, reads inferred Placeholder
    // names and sink op names, then destroys the probe session.
    if (!opts.inputOp || !opts.outputOps?.length) {
      const { TFSession } = await import("jude-tf");
      const probe = await TFSession.loadFrozenGraph(opts.modelPath);
      opts.inputOp ??= probe.inputs[0];
      opts.outputOps ??= probe.outputs;
      probe.destroy();
      if (!opts.inputOp)
        throw new Error(`Could not infer inputOp from ${opts.modelPath}`);
      if (!opts.outputOps?.length)
        throw new Error(`Could not infer outputOps from ${opts.modelPath}`);
    }

    const requestedStrategy = opts.strategy ?? "auto";
    const concurrency = opts.concurrency ?? availableParallelism();
    const autoThreshold = opts.autoThresholdMs ?? DEFAULT_AUTO_THRESHOLD;
    const reserveCores = opts.reserveCores ?? 0;

    let resolved: "worker-pool" | "tf-parallel";

    if (requestedStrategy === "worker-pool") {
      resolved = "worker-pool";
    } else if (requestedStrategy === "tf-parallel") {
      resolved = "tf-parallel";
    } else {
      // auto — file size as cheap first signal, probe if ambiguous
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
        // Warm probe with intra=1 — measure single-core inference time
        const { TFSession } = await import("jude-tf");
        const probeSess = await TFSession.loadFrozenGraph(opts.modelPath);
        const probeInput = Buffer.alloc(
          opts.probeShape.reduce((a, b) => a * b, 1) * 4,
        );

        await probeSess.runAsync(
          { [opts.inputOp]: probeInput },
          opts.outputOps,
        );
        const t0 = performance.now();
        await probeSess.runAsync(
          { [opts.inputOp]: probeInput },
          opts.outputOps,
        );
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
      const hw = availableParallelism();
      const tfCores = Math.max(1, hw - reserveCores);
      process.stderr.write(
        `[isidorus] CPU affinity: reserving ${reserveCores} core(s), ` +
          `TF gets ${tfCores} core(s)\n`,
      );
    }

    return resolved === "worker-pool"
      ? InferencePool.createWorkerPool(opts, concurrency, reserveCores)
      : InferencePool.createTfParallel(opts, reserveCores);
  }

  // ── worker-pool init ───────────────────────────────────────────────────────

  private static async createWorkerPool(
    opts: PoolOptions,
    concurrency: number,
    reserveCores: number,
  ): Promise<InferencePool> {
    const ctrlSab = new SharedArrayBuffer(concurrency * CTRL_SLOTS * 4);
    const slots: WorkerSlot[] = [];
    const startedWorkers: Worker[] = [];

    // In dev/test we run TypeScript source directly via tsx. Workers don't
    // inherit --import tsx from the parent, so we use a small .mjs bootstrap
    // (inference-pool-worker.mjs) that calls register() from tsx/esm/api
    // before importing this .ts file. In production the compiled .js entry
    // is used directly with no extra loader needed.
    const isTsSource = import.meta.url.endsWith(".ts");
    const workerEntry = isTsSource
      ? new URL("./inference-pool-worker.mjs", import.meta.url)
      : new URL("./inference-pool.js", import.meta.url);

    try {
      for (let i = 0; i < concurrency; i++) {
        const ctrl = new Int32Array(ctrlSab, i * CTRL_SLOTS * 4, CTRL_SLOTS);
        Atomics.store(ctrl, 0, IDLE);

        const worker = new Worker(workerEntry, {
          workerData: {
            ctrlSab,
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

        slots.push({ worker, ctrl, busy: false, resolve: null, reject: null });
      }
    } catch (err) {
      // Terminate any workers that were already started before the failure.
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

    // Try the native @isidorus/cpu Session path first (ConfigProto thread
    // config + OnRunWork affinity fence). Falls back to jude-tf if the addon
    // hasn't been initialised (e.g. when called from outside @isidorus/cpu).
    //
    // We import _native.js rather than "@isidorus/cpu" to avoid a circular
    // dependency — this file IS part of @isidorus/cpu, so importing the
    // package entry point would re-run ensureTf() + node-gyp-build.
    let tfParallelGraph: Graph | null = null;
    let tfParallelSess: Session | any | null = null;

    try {
      const { getAddon } = await import("./_native.js");
      const { readFileSync } = await import("fs");
      const { Graph: GraphClass } = await import("./graph.js");
      const { Session: SessionClass } = await import("./session.js");
      const addon = getAddon();

      const g = new GraphClass(new addon.Graph());
      g.importGraphDef(readFileSync(opts.modelPath));

      tfParallelGraph = g;
      tfParallelSess = new SessionClass(
        new addon.Session(g._native, {
          strategy: "tf-parallel",
          reserveCores,
        }),
      );

      process.stderr.write(
        `[isidorus] tf-parallel: intra_op=${tfCores} ` +
          `(${reserveCores} core(s) reserved, native Session)\n`,
      );
    } catch {
      // Native addon not available — fall back to jude-tf TFSession.
      // This path lacks the affinity fence but is otherwise correct.
      const { TFSession } = await import("jude-tf");
      tfParallelSess = await TFSession.loadFrozenGraph(opts.modelPath);

      process.stderr.write(
        `[isidorus] tf-parallel: intra_op=${tfCores} ` +
          `(${reserveCores} core(s) reserved, jude-tf fallback)\n`,
      );
    }

    return new InferencePool({
      strategy: "tf-parallel",
      reserveCores,
      workerSlots: [],
      queue: [],
      ctrlSab: null,
      tfParallelGraph,
      tfParallelSess,
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
      if (slot) {
        this.dispatchToWorker(
          slot,
          inputBuf,
          inputShape,
          inputDtype,
          resolve,
          reject,
        );
      } else {
        this.queue.push({ inputBuf, inputShape, inputDtype, resolve, reject });
      }
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

    // ── ORDERING CRITICAL ──────────────────────────────────────────────────
    // 1. Register the result listener FIRST — before the worker can possibly
    //    send a response. Node.js buffers port messages until a listener is
    //    registered, but registering after notify introduces a thread-level
    //    race where an extremely fast result could be missed.
    // 2. Post the input data SECOND — the worker awaits this message after
    //    waking from Atomics.wait, so it must arrive before WORK is stored.
    // 3. Store WORK + notify LAST — wakes the worker.
    const handleMessage = (msg: any) => {
      if (msg.type === "work_error") {
        this.settleSlot(slot, null, new Error(msg.error));
        return;
      }
      if (msg.type !== "result") {
        // Unexpected message type — re-register to wait for the actual result.
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
            // postMessage structured-clones Buffer as plain Uint8Array —
            // wrap it back into a Buffer so callers can use Buffer.isBuffer().
            data: Buffer.isBuffer(o.data) ? o.data : Buffer.from(o.data),
          })),
          inferenceMs: msg.inferenceMs,
        },
        null,
      );
    };

    // Register listener before waking the worker.
    slot.worker.once("message", handleMessage);

    // Register a one-shot error listener so an uncaught worker crash rejects
    // the promise instead of leaving it hanging.
    slot.worker.once("error", (err: Error) => {
      this.settleSlot(slot, null, err);
    });

    // Post input, then wake worker.
    slot.worker.postMessage({ inputData: inputBuf, inputShape, inputDtype });
    Atomics.store(slot.ctrl, 0, WORK);
    Atomics.notify(slot.ctrl, 0, 1);
  }

  /** Settle a worker slot's in-flight promise and drain the queue. */
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

    // Drain one queued request now that the slot is free.
    const next = this.queue.shift();
    if (next) {
      this.dispatchToWorker(
        slot,
        next.inputBuf,
        next.inputShape,
        next.inputDtype,
        next.resolve,
        next.reject,
      );
    }
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

    let inferencePromise: Promise<any>;

    if (this.tfParallelGraph) {
      // ── Native @isidorus/cpu Session path ──────────────────────────────
      // Build feed/fetch arrays using Graph.getOp() to resolve op names to
      // Tensor references, which Session.runAsync expects.
      const g = this.tfParallelGraph;
      const inputTensor = g.getOp(this.inputOp);
      if (!inputTensor) {
        this.tfParallelBusy = false;
        reject(
          new Error(
            `tf-parallel: input op not found in graph: ${this.inputOp}`,
          ),
        );
        return;
      }

      const outputTensors = this.outputOps.map((name) => {
        const t = g.getOp(name);
        if (!t)
          throw new Error(`tf-parallel: output op not found in graph: ${name}`);
        return t;
      });

      // Reinterpret the raw Buffer bytes as the correct TypedArray dtype.
      // asTypedArray is only defined inside the !isMainThread block, so we
      // inline the same logic here for the main-thread tf-parallel path.
      const ab = inputBuf.buffer.slice(
        inputBuf.byteOffset,
        inputBuf.byteOffset + inputBuf.byteLength,
      );
      let typedInput: ArrayBufferView;
      switch (inputDtype) {
        case 1:
          typedInput = new Float32Array(ab);
          break;
        case 2:
          typedInput = new Float64Array(ab);
          break;
        case 3:
          typedInput = new Int32Array(ab);
          break;
        case 4:
          typedInput = new Uint8Array(ab);
          break;
        case 9:
          typedInput = new BigInt64Array(ab);
          break;
        default:
          typedInput = new Uint8Array(ab);
      }

      const feedValue = {
        dtype: inputDtype,
        shape: inputShape,
        data: Buffer.from(
          typedInput.buffer,
          typedInput.byteOffset,
          typedInput.byteLength,
        ),
      };

      inferencePromise = this.tfParallelSess!.runAsync(
        [[inputTensor, feedValue]],
        outputTensors,
      ).then((outputs: any[]) => {
        // Map back to { [outputKey]: TensorResult } for uniform handling below
        const result: Record<string, any> = {};
        this.outputOps.forEach((key, i) => {
          result[key] = outputs[i];
        });
        return result;
      });
    } else {
      // ── jude-tf TFSession fallback path ───────────────────────────────
      inferencePromise = this.tfParallelSess!.runAsync(
        { [this.inputOp]: inputBuf },
        this.outputOps,
      );
    }

    inferencePromise
      .then((results: any) => {
        const inferenceMs = performance.now() - t0;
        this.tfParallelBusy = false;
        resolve({
          workerId: 0,
          strategy: "tf-parallel",
          outputs: this.outputOps.map((k: string) => {
            const r = results[k];
            if (!r) return { dtype: 0, shape: [], data: Buffer.alloc(0) };
            const view = r.data as ArrayBufferView;
            return {
              dtype: r.dtype,
              shape: r.shape,
              data: Buffer.isBuffer(r.data)
                ? r.data
                : Buffer.from(view.buffer, view.byteOffset, view.byteLength),
            };
          }),
          inferenceMs,
        });
        const next = this.tfParallelQueue.shift();
        if (next) {
          this.runTfParallel(
            next.inputBuf,
            next.inputShape,
            next.inputDtype,
            next.resolve,
            next.reject,
          );
        }
      })
      .catch((err: Error) => {
        this.tfParallelBusy = false;
        reject(err);
        const next = this.tfParallelQueue.shift();
        if (next) {
          this.runTfParallel(
            next.inputBuf,
            next.inputShape,
            next.inputDtype,
            next.resolve,
            next.reject,
          );
        }
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
                // Register shutdown_ack listener before storing SHUTDOWN.
                slot.worker.once("message", (msg: any) => {
                  if (msg.type === "shutdown_ack") resolve();
                });
                slot.worker.once("error", reject);
                Atomics.store(slot.ctrl, 0, SHUTDOWN);
                Atomics.notify(slot.ctrl, 0, 1);
              };

              if (slot.busy) {
                // Wait for the current in-flight request to finish first.
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
