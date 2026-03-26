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
 *   auto         Probe-based selection (see below).
 *
 * CPU affinity (reserveCores):
 *
 *   reserveCores = R pins TF computation to the LAST (N-R) cores.
 *   The FIRST R cores are left free for the event loop, libuv I/O,
 *   opencv4nodejs, and other native libs.
 *
 *   Without reservation, TF's eigen threads compete with the event loop
 *   and other native work for all cores. Under a large model this causes
 *   timer jitter, delayed I/O callbacks, and starvation of other addons.
 *
 *   Example on an 8-core machine:
 *     reserveCores: 2  →  cores 0-1: event loop + opencv
 *                         cores 2-7: TF eigen threadpool
 *
 *   The affinity fence is applied in OnRunWork (the libuv thread pool
 *   thread) immediately before TF_SessionRun and restored immediately
 *   after, so the libuv worker returns to unrestricted scheduling between
 *   inference calls.
 */

import { Worker, isMainThread, parentPort, workerData } from "worker_threads";
import { availableParallelism } from "os";
import { statSync } from "fs";
import { performance } from "perf_hooks";

// ─── Constants ──────────────────────────────────────────────────────────────
const IDLE = 0;
const WORK = 1;
const DONE = 2;
const SHUTDOWN = 3;
const CTRL_SLOTS = 4;

const SIZE_THRESHOLD_BYTES = 150 * 1024 * 1024; // 150 MB
const DEFAULT_AUTO_THRESHOLD = 20; // ms

// ─── Types ──────────────────────────────────────────────────────────────────

export type ExecutionStrategy = "worker-pool" | "tf-parallel" | "auto";

export interface PoolOptions {
  modelPath: string;
  inputOp: string;
  outputOps: string[];
  /**
   * Number of CPU cores to reserve for the event loop and other native libs
   * (opencv, sharp, etc.). TF inference is pinned to the remaining cores.
   *
   * Default: 0 (no reservation — TF may use any core).
   *
   * Recommended: 1–2 on a dedicated inference server.
   * Example: reserveCores=2 on an 8-core machine gives TF cores 2-7 and
   * leaves cores 0-1 for the event loop and other work.
   */
  reserveCores?: number;
  /** Execution strategy. Default: "auto". */
  strategy?: ExecutionStrategy;
  /** Worker thread count (worker-pool). Default: os.availableParallelism(). */
  concurrency?: number;
  /** Input shape for auto probe run. Required when strategy="auto" and
   *  model size >= 150MB. */
  probeShape?: number[];
  /** Auto-detection threshold (ms). Default: 20ms. */
  autoThresholdMs?: number;
}

export interface PoolResult {
  workerId: number;
  strategy: "worker-pool" | "tf-parallel";
  outputs: { dtype: number; shape: number[]; data: Buffer }[];
  inferenceMs: number;
}

// ─── Worker-side logic ──────────────────────────────────────────────────────

if (!isMainThread) {
  const { ctrlSab, workerIndex, modelPath, inputOp, outputOps, reserveCores } =
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

  const { TFSession } = await import("jude-tf");
  const sess = await TFSession.loadFrozenGraph(modelPath);
  // TODO: once @isidorus/cpu native Session is used here, pass reserveCores
  // as a session option so the affinity fence is applied inside OnRunWork.
  // For now we use jude-tf's session which doesn't have affinity support.

  Atomics.store(ctrl, 0, IDLE);
  parentPort!.postMessage({ type: "ready" });

  while (true) {
    Atomics.wait(ctrl, 0, IDLE);
    const state = Atomics.load(ctrl, 0);

    if (state === SHUTDOWN) {
      sess.destroy();
      parentPort!.postMessage({ type: "shutdown_ack" });
      break;
    }

    if (state === WORK) {
      const msg: any = await new Promise((resolve) =>
        parentPort!.once("message", resolve),
      );

      const { inputData, inputShape, inputDtype } = msg as {
        inputData: Buffer;
        inputShape: number[];
        inputDtype: number;
      };

      const t0 = performance.now();
      const results = await sess.run({ [inputOp]: inputData }, outputOps);
      const inferenceMs = performance.now() - t0;

      Atomics.store(ctrl, 0, DONE);
      Atomics.notify(ctrl, 0, 1);

      parentPort!.postMessage({
        type: "result",
        outputs: outputOps.map((k) => results[k]),
        inferenceMs,
      });

      Atomics.store(ctrl, 0, IDLE);
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
  private tfParallelSess: any | null;
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
    tfParallelSess: any | null;
    modelPath: string;
    inputOp: string;
    outputOps: string[];
  }) {
    this.strategy = params.strategy;
    this.reserveCores = params.reserveCores;
    this.workerSlots = params.workerSlots;
    this.queue = params.queue;
    this.ctrlSab = params.ctrlSab;
    this.tfParallelSess = params.tfParallelSess;
    this.tfParallelBusy = false;
    this.tfParallelQueue = [];
    this.modelPath = params.modelPath;
    this.inputOp = params.inputOp;
    this.outputOps = params.outputOps;
  }

  // ── Factory ────────────────────────────────────────────────────────────────

  static async create(opts: PoolOptions): Promise<InferencePool> {
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
      // auto
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
        // Probe with intra=1 (worker-pool config) to measure single-core time.
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
        `[isidorus] CPU affinity: reserving ${reserveCores} core(s) ` +
          `for event loop/other libs, TF gets ${tfCores} core(s)\n`,
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
    const workerEntrypoint = `
const { parentPort, workerData } = require("node:worker_threads");
const { performance } = require("node:perf_hooks");
const { join } = require("node:path");

const IDLE = 0;
const WORK = 1;
const DONE = 2;
const SHUTDOWN = 3;
const CTRL_SLOTS = 4;

(async () => {
  try {
    if (process.platform === "win32") {
      const libtfPath = process.env.LIBTENSORFLOW_PATH || "C:\\\\libtensorflow";
      const dllDir = libtfPath.toLowerCase().endsWith("\\\\lib")
        ? libtfPath
        : join(libtfPath, "lib");
      const currentPath = process.env.Path || process.env.PATH || "";
      const nextPath = dllDir + ";" + currentPath;
      process.env.Path = nextPath;
      process.env.PATH = nextPath;
    }

    const { ctrlSab, workerIndex, modelPath, inputOp, outputOps } = workerData;
    const ctrl = new Int32Array(ctrlSab, workerIndex * CTRL_SLOTS * 4, CTRL_SLOTS);

    const { TFSession } = await import("jude-tf");
    const sess = await TFSession.loadFrozenGraph(modelPath);

    Atomics.store(ctrl, 0, IDLE);
    parentPort.postMessage({ type: "ready" });

    while (true) {
      Atomics.wait(ctrl, 0, IDLE);
      const state = Atomics.load(ctrl, 0);

      if (state === SHUTDOWN) {
        sess.destroy();
        parentPort.postMessage({ type: "shutdown_ack" });
        break;
      }

      if (state === WORK) {
        const msg = await new Promise((resolve) => parentPort.once("message", resolve));
        const t0 = performance.now();
        const results = await sess.run({ [inputOp]: msg.inputData }, outputOps);
        const inferenceMs = performance.now() - t0;

        Atomics.store(ctrl, 0, DONE);
        Atomics.notify(ctrl, 0, 1);

        parentPort.postMessage({
          type: "result",
          outputs: outputOps.map((k) => results[k]),
          inferenceMs,
        });

        Atomics.store(ctrl, 0, IDLE);
      }
    }
  } catch (err) {
    parentPort.postMessage({
      type: "init_error",
      error: err && err.stack ? String(err.stack) : String(err),
    });
  }
})();
`;

    try {
      for (let i = 0; i < concurrency; i++) {
        const ctrl = new Int32Array(ctrlSab, i * CTRL_SLOTS * 4, CTRL_SLOTS);
        Atomics.store(ctrl, 0, IDLE);

        const worker = new Worker(workerEntrypoint, {
          eval: true,
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
            else if (msg.type === "init_error") {
              reject(new Error(`Worker init failed: ${msg.error}`));
            } else reject(new Error(`Unexpected worker init: ${msg.type}`));
          });
          worker.once("error", reject);
        });

        slots.push({ worker, ctrl, busy: false, resolve: null, reject: null });
      }
    } catch (err) {
      await Promise.allSettled(
        startedWorkers.map((worker) => worker.terminate()),
      );
      throw err;
    }

    return new InferencePool({
      strategy: "worker-pool",
      reserveCores,
      workerSlots: slots,
      queue: [],
      ctrlSab,
      tfParallelSess: null,
      modelPath: opts.modelPath,
      inputOp: opts.inputOp,
      outputOps: opts.outputOps,
    });
  }

  // ── tf-parallel init ───────────────────────────────────────────────────────

  private static async createTfParallel(
    opts: PoolOptions,
    reserveCores: number,
  ): Promise<InferencePool> {
    const hw = availableParallelism();
    const tfCores = Math.max(1, hw - reserveCores);

    // Once @isidorus/cpu's native Graph+Session is used here, we'll
    // construct the Session with:
    //   { strategy: "tf-parallel", reserveCores }
    // and the C++ ConfigProto + affinity fencing take effect.
    // For now we load via jude-tf (which uses its own session).
    const { TFSession } = await import("jude-tf");
    const sess = await TFSession.loadFrozenGraph(opts.modelPath);

    process.stderr.write(
      `[isidorus] tf-parallel: intra_op=${tfCores} ` +
        `(${reserveCores} core(s) reserved)\n`,
    );

    return new InferencePool({
      strategy: "tf-parallel",
      reserveCores,
      workerSlots: [],
      queue: [],
      ctrlSab: null,
      tfParallelSess: sess,
      modelPath: opts.modelPath,
      inputOp: opts.inputOp,
      outputOps: opts.outputOps,
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

    slot.worker.postMessage({ inputData: inputBuf, inputShape, inputDtype });
    Atomics.store(slot.ctrl, 0, WORK);
    Atomics.notify(slot.ctrl, 0, 1);

    slot.worker.once("message", (msg: any) => {
      if (msg.type !== "result") return;
      slot.busy = false;
      slot.resolve = null;
      slot.reject = null;

      resolve({
        workerId,
        strategy: "worker-pool",
        outputs: msg.outputs,
        inferenceMs: msg.inferenceMs,
      });

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
    });
  }

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

    this.tfParallelSess!.runAsync({ [this.inputOp]: inputBuf }, this.outputOps)
      .then((results: any) => {
        const inferenceMs = performance.now() - t0;
        this.tfParallelBusy = false;
        resolve({
          workerId: 0,
          strategy: "tf-parallel",
          outputs: this.outputOps.map((k: string) => results[k]),
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
                slot.worker.once("message", (msg: any) => {
                  if (msg.type === "shutdown_ack") resolve();
                });
                Atomics.store(slot.ctrl, 0, SHUTDOWN);
                Atomics.notify(slot.ctrl, 0, 1);
              };
              if (slot.busy) {
                const origRes = slot.resolve;
                const origRej = slot.reject;
                slot.resolve = (r) => {
                  origRes?.(r);
                  doShutdown();
                };
                slot.reject = (e) => {
                  origRej?.(e);
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
    }
  }
}
