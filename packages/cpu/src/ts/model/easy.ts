/**
 * model/easy.ts — High-level "Keras-like" Model API
 *
 * Wraps the low-level Sequential + Graph + Session + Optimizer lifecycle into a
 * single class so beginners can train and deploy without touching any of those
 * primitives directly.
 *
 * @example
 * import { Model, layers } from "@isidorus/cpu/easy";
 *
 * const model = new Model([28, 28, 1], [
 *   new layers.Conv2D(32, 3, { activation: "relu" }),
 *   new layers.MaxPooling2D(),
 *   new layers.Flatten(),
 *   new layers.Dense(10, { activation: "softmax" }),
 * ]);
 *
 * model.compile({ loss: "sparse_categorical_crossentropy", optimizer: "adam" });
 * await model.fit(xTrain, yTrain, { epochs: 5, batchSize: 64 });
 * const preds = await model.predict(xTest);
 * await model.save("./model.pb");
 */

import { Graph } from "../graph.js";
import { Session } from "../session.js";
import { getAddon } from "../_native.js";
import { SGD } from "../optimizers/sgd.js";
import { Adam } from "../optimizers/adam.js";
import { RMSProp } from "../optimizers/rmsprop.js";
import { Sequential } from "./sequential.js";
import type { Layer } from "./layer.js";
import type { LossFn, Optimizer } from "./sequential.js";
import { DType } from "@isidorus/core";
import { debug } from "../_log.js";

// ── Public types ──────────────────────────────────────────────────────────────

/** Built-in optimizer shorthands accepted by compile(). */
export type OptimizerName = "adam" | "sgd" | "rmsprop";

export interface CompileOptions {
  /** Loss function. */
  loss: LossFn;
  /**
   * Optimizer — either a name string or a pre-constructed optimizer instance.
   * When a string is given, lr (and momentum for "sgd") are applied.
   * Default: "adam".
   */
  optimizer?: OptimizerName | Optimizer;
  /** Learning rate. Ignored when an Optimizer instance is passed. Default: 0.001. */
  lr?: number;
  /** SGD momentum. Only used when optimizer === "sgd". Default: 0.0. */
  momentum?: number;
  /**
   * TF intra-op thread count (threads per op / kernel).
   * Defaults to hardware_concurrency() — same as InferencePool.
   * Set explicitly to match a known-good InferencePool autotuner result.
   */
  intraOpThreads?: number;
  /**
   * TF inter-op thread count (concurrent independent graph branches).
   * Default: 1.
   */
  interOpThreads?: number;
}

export interface FitOptions {
  /** Number of full passes over the dataset. Default: 1. */
  epochs?: number;
  /** Number of samples per gradient update. Default: 32. */
  batchSize?: number;
  /**
   * Fraction of training data held out for validation (0–1).
   * Validation loss is computed but does not affect weights.
   * Default: 0 (no validation).
   */
  validationSplit?: number;
  /**
   * Whether to print epoch/step progress to stdout.
   * Default: true.
   */
  verbose?: boolean;
  /**
   * Called at the end of each epoch.
   * @param epoch   0-indexed epoch number.
   * @param logs    Training (and optional validation) metrics for the epoch.
   */
  onEpochEnd?: (epoch: number, logs: EpochLogs) => void;
  /**
   * Whether to shuffle the dataset before each epoch.
   * Default: true.
   */
  shuffle?: boolean;
  /**
   * Whether to run each training step synchronously on the calling thread.
   * Default: true (recommended for training loops).
   *
   * sync: true  — uses trainStepSync(): one blocking TF_SessionRun per batch.
   *               Eliminates Promise + libuv thread-pool dispatch overhead.
   *               Matches Python's sess.run() throughput.
   *               Event loop is blocked during each step (acceptable in training).
   *
   * sync: false — uses trainStep(): dispatches to the libuv thread pool.
   *               Use only when the event loop must remain live during training
   *               (e.g. serving health-check requests while training).
   */
  sync?: boolean;
}

export interface EpochLogs {
  loss: number;
  valLoss?: number;
}

export interface FitResult {
  history: EpochLogs[];
}

// ── Data conversion helpers ──────────────────────────────────────────────────

/**
 * User-friendly data types accepted by fit(), predict(), and InferencePool.infer().
 * - Float32Array / Int32Array / Uint8Array / Buffer: use directly (zero-copy)
 * - Any TypedArray: auto-interpreted as bytes and converted
 * - number[] or nested number[][]: auto-flatten and convert
 * - Promise<...>: awaited to support lazy-loaded datasets
 */
export type DataLike =
  | Float32Array
  | Int32Array
  | Uint8Array
  | number[]
  | number[][]
  | number[][][]
  | number[][][][];

/**
 * Convert various data formats into a flat Float32Array.
 * Automatically flattens nested arrays and validates total size.
 *
 * @param data        Input data (array, typed array, or Buffer)
 * @param expectedElems  Expected total elements (for validation)
 * @returns           Flat Float32Array
 * @throws Error      If size doesn't match expectedElems
 */
export function toFloat32Array(
  data: DataLike,
  expectedElems?: number,
): Float32Array {
  let result: Float32Array;

  if (data instanceof Float32Array) {
    result = data;
  } else if (data instanceof Int32Array) {
    // Convert Int32Array to Float32Array
    result = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) result[i] = data[i];
  } else if (
    data instanceof Uint8Array ||
    (typeof data === "object" && data !== null && "buffer" in data)
  ) {
    // Handle Buffer (extends Uint8Array) and other TypedArrays
    // Interpret bytes as 32-bit floats: 4 bytes = 1 float
    const byteView = data as
      | Uint8Array
      | { buffer: ArrayBuffer; byteOffset?: number; byteLength?: number };
    const buffer = byteView instanceof ArrayBuffer ? byteView : byteView.buffer;
    const byteOffset =
      "byteOffset" in byteView && byteView.byteOffset ? byteView.byteOffset : 0;
    const byteLength =
      "byteLength" in byteView && byteView.byteLength
        ? byteView.byteLength
        : buffer.byteLength - byteOffset;
    result = new Float32Array(buffer, byteOffset, byteLength / 4);
  } else if (Array.isArray(data)) {
    // Recursively flatten nested arrays
    const flat = flattenArray(data);
    result = new Float32Array(flat);
  } else {
    throw new TypeError(
      `toFloat32Array: expected array or typed array, got ${typeof data}`,
    );
  }

  if (expectedElems !== undefined && result.length !== expectedElems) {
    throw new Error(
      `Data size mismatch: expected ${expectedElems} elements, got ${result.length}`,
    );
  }

  return result;
}

/**
 * Convert various data formats into a flat Int32Array.
 * For use with sparse_categorical_crossentropy (class indices).
 */
export function toInt32Array(
  data: DataLike,
  expectedElems?: number,
): Int32Array {
  let result: Int32Array;

  if (data instanceof Int32Array) {
    result = data;
  } else if (data instanceof Float32Array) {
    result = new Int32Array(data.length);
    for (let i = 0; i < data.length; i++) result[i] = Math.floor(data[i]);
  } else if (
    data instanceof Uint8Array ||
    (typeof data === "object" && data !== null && "buffer" in data)
  ) {
    // Handle Buffer (extends Uint8Array) and other TypedArrays
    // Interpret bytes as 32-bit ints: 4 bytes = 1 int
    const byteView = data as
      | Uint8Array
      | { buffer: ArrayBuffer; byteOffset?: number; byteLength?: number };
    const buffer = byteView instanceof ArrayBuffer ? byteView : byteView.buffer;
    const byteOffset =
      "byteOffset" in byteView && byteView.byteOffset ? byteView.byteOffset : 0;
    const byteLength =
      "byteLength" in byteView && byteView.byteLength
        ? byteView.byteLength
        : buffer.byteLength - byteOffset;
    result = new Int32Array(buffer, byteOffset, byteLength / 4);
  } else if (Array.isArray(data)) {
    const flat = flattenArray(data);
    result = new Int32Array(flat.map((x) => Math.floor(x)));
  } else {
    throw new TypeError(
      `toInt32Array: expected array or typed array, got ${typeof data}`,
    );
  }

  if (expectedElems !== undefined && result.length !== expectedElems) {
    throw new Error(
      `Data size mismatch: expected ${expectedElems} elements, got ${result.length}`,
    );
  }

  return result;
}

/**
 * Recursively flatten any nesting of arrays into a single flat array of numbers.
 */
function flattenArray(arr: any): number[] {
  const result: number[] = [];

  function flatten(item: any): void {
    if (Array.isArray(item)) {
      for (const elem of item) {
        flatten(elem);
      }
    } else if (typeof item === "number") {
      result.push(item);
    } else {
      throw new TypeError(
        `Expected number or array, got ${typeof item}: ${item}`,
      );
    }
  }

  flatten(arr);
  return result;
}

// ── Internal helpers ──────────────────────────────────────────────────────────

function makeOptimizer(
  name: OptimizerName,
  g: Graph,
  seq: Sequential,
  lr: number,
  momentum: number,
): Optimizer {
  switch (name) {
    case "sgd":
      return new SGD(g, seq.params, lr, { momentum });
    case "rmsprop":
      return new RMSProp(g, seq.params, lr);
    case "adam":
    default:
      return new Adam(g, seq.params, lr);
  }
}

/** Infer the number of elements per label sample from loss + output shape. */
function yElemsPerSample(loss: LossFn, outputShape: (number | null)[]): number {
  if (loss === "sparse_categorical_crossentropy") return 1; // single int32 class index
  if (loss === "binary_crossentropy") return 1; // single float32 logit/prob
  // mse: full output vector
  return outputShape.slice(1).reduce<number>((a, b) => (a ?? 1) * (b ?? 1), 1);
}

/** Label dtype from loss function. */
function labelDtype(loss: LossFn): DType {
  return loss === "sparse_categorical_crossentropy"
    ? DType.INT32
    : DType.FLOAT32;
}

/**
 * Shuffle xBuf and yBuf in-place using Fisher-Yates.
 * xElems and yElems are the element counts *per sample*.
 */
function shuffleDataset(
  x: Float32Array,
  xElems: number,
  y: Float32Array | Int32Array,
  yElems: number,
  nSamples: number,
): void {
  const xTmp = new Float32Array(xElems);
  const yTmp =
    y instanceof Int32Array ? new Int32Array(yElems) : new Float32Array(yElems);
  for (let i = nSamples - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    if (i === j) continue;
    // swap x[i] ↔ x[j]
    xTmp.set(x.subarray(i * xElems, (i + 1) * xElems));
    x.copyWithin(i * xElems, j * xElems, (j + 1) * xElems);
    x.set(xTmp, j * xElems);
    // swap y[i] ↔ y[j]
    (yTmp as any).set(y.subarray(i * yElems, (i + 1) * yElems));
    y.copyWithin(i * yElems, j * yElems, (j + 1) * yElems);
    (y as any).set(yTmp, j * yElems);
  }
}

// ── Model class ───────────────────────────────────────────────────────────────

/**
 * High-level model container.
 *
 * Lifecycle:
 *   new Model(inputShape, layers)
 *   model.compile({ loss, optimizer })
 *   await model.fit(xTrain, yTrain, { epochs, batchSize })
 *   const preds = await model.predict(xTest)
 *   await model.save("./model.pb")
 */
export class Model {
  private readonly inputShape: number[];
  private readonly layerList: Layer[];

  // Populated by compile()
  private _g: Graph | null = null;
  private _seq: Sequential | null = null;
  private _opt: Optimizer | null = null;
  private _loss: LossFn | null = null;
  private _intraOpThreads: number = 0; // 0 = TF default (hw_concurrency)
  private _interOpThreads: number = 1;
  private _predictCallCount: number = 0; // for one-shot debug log
  // WeakMap<source array, Float32Array> — caches toFloat32Array() results so
  // repeated predict() calls with the same non-typed-array source pay the
  // allocation cost once only. Automatically GC'd when the source is released.
  private readonly _convertCache = new WeakMap<object, Float32Array>();
  private readonly _convertCacheInt = new WeakMap<object, Int32Array>();

  // Populated on first fit() or predict()
  private _sess: Session | null = null;
  private _initialized: boolean = false;

  constructor(inputShape: number[], layers: Layer[]) {
    if (inputShape.length === 0)
      throw new Error("Model: inputShape must be non-empty");
    if (layers.length === 0)
      throw new Error("Model: layers array must be non-empty");
    this.inputShape = inputShape;
    this.layerList = layers;
  }

  // ── compile ───────────────────────────────────────────────────────────────

  /**
   * Wire the computation graph and initialise the optimizer.
   *
   * Must be called once before fit() or predict().
   */
  compile(opts: CompileOptions = { loss: "mse" }): void {
    if (this._seq) throw new Error("Model.compile() has already been called.");

    const {
      loss,
      optimizer = "adam",
      lr = 0.001,
      momentum = 0.0,
      intraOpThreads = 0,
      interOpThreads = 1,
    } = opts;
    this._intraOpThreads = intraOpThreads;
    this._interOpThreads = interOpThreads;

    this._loss = loss;
    const addon = getAddon();
    this._g = new Graph(new addon.Graph());
    this._seq = new Sequential(this._g, this.layerList);

    // seq.compile() wires the graph and computes gradient tensors.
    // The optimizer MUST be created after this because it needs grad tensors.
    this._seq.compile({
      loss,
      inputShape: this.inputShape,
      labelDtype: labelDtype(loss),
    });

    if (typeof optimizer === "string") {
      this._opt = makeOptimizer(optimizer, this._g, this._seq, lr, momentum);
    } else {
      // User passed a pre-built optimizer — trust it.
      this._opt = optimizer;
    }
    debug(
      `[Model] compiled  loss=${loss}  optimizer=${
        typeof optimizer === "string" ? optimizer : "custom"
      }` +
        `  lr=${lr}  inputShape=${JSON.stringify(this.inputShape)}` +
        `  intraOpThreads=${
          this._intraOpThreads || "hw_default"
        }  interOpThreads=${this._interOpThreads}`,
    );
  }

  // ── dispose ──────────────────────────────────────────────────────────────

  /**
   * Synchronously close the TF session and release all native resources.
   *
   * Idempotent — safe to call multiple times.
   *
   * Recommended: call this explicitly after you're done with the model,
   * especially in production servers. In test environments, call dispose()
   * at the end of each test.
   *
   * NOTE: We do NOT hook process.once("beforeExit") because it interferes
   * with Node's test runner (causes "Promise resolution is still pending"
   * errors). Call dispose() explicitly instead.
   */
  dispose(): void {
    if (!this._sess) return; // already disposed or never initialized
    debug("[Model] dispose  closing TF session");
    this._sess.destroy(); // → TF_CloseSession + TF_DeleteSession
    this._sess = null;
    this._initialized = false;
    // Keep _g, _seq, _opt alive so their C++ finalizers run AFTER the session
    // is already closed (session finalizer calls graph_ref_.Reset() which
    // then allows the Graph to be collected cleanly).
  }

  // ── Internal: lazy session init ──────────────────────────────────────────

  private assertCompiled(method: string): void {
    if (!this._seq || !this._opt || !this._g || !this._loss)
      throw new Error(`Model.${method}(): call compile() first.`);
  }

  private async ensureSession(): Promise<Session> {
    this.assertCompiled("ensureSession");
    if (this._sess === null && this._initialized) {
      throw new Error("Model has been disposed and cannot be used again.");
    }
    if (!this._sess) {
      // Create native Session directly — no free-function factory in session.ts.
      const addon = getAddon();
      const sessOpts: Record<string, number> = {
        interOpThreads: this._interOpThreads,
      };
      if (this._intraOpThreads > 0) {
        // 0 means "let TF default to hardware_concurrency()" (session.cc default).
        sessOpts.intraOpThreads = this._intraOpThreads;
      }
      this._sess = new Session(new addon.Session(this._g!._native, sessOpts));
      debug("[DEBUG] Session created");
    }
    if (!this._initialized) {
      debug(
        "[DEBUG] Initializing sequence with init op:",
        this._seq!["_allInitOp"],
      );
      await this._seq!.init(this._sess, this._opt!);
      debug("[DEBUG] Sequence initialized");
      this._initialized = true;
    }
    return this._sess;
  }

  // ── fit ───────────────────────────────────────────────────────────────────

  /**
   * Train the model on the provided dataset.
   *
   * Accepts data in multiple formats:
   * - Float32Array / Int32Array (fastest, zero-copy)
   * - JS arrays: [1, 2, 3, ...] or nested [[1, 2, ...], ...] (auto-flattened)
   * - Labels for sparse_categorical_crossentropy: class indices (0, 1, 2, ...)
   *
   * @param xData  Training inputs. Size must match nSamples × prod(inputShape).
   * @param yData  Training labels. Size must match nSamples × prod(outputShape) or nSamples for sparse labels.
   * @param opts   Training options (epochs, batchSize, validationSplit, …).
   */
  async fit(
    xData: DataLike,
    yData: DataLike,
    opts: FitOptions = {},
  ): Promise<FitResult> {
    this.assertCompiled("fit");

    const {
      epochs = 1,
      batchSize = 32,
      validationSplit = 0,
      verbose = true,
      onEpochEnd,
      shuffle = true,
      sync = true,
    } = opts;

    const seq = this._seq!;
    const opt = this._opt!;
    const loss = this._loss!;
    const xElems = this.inputShape.reduce((a, b) => a * b, 1);
    const yElems = yElemsPerSample(loss, seq.outputShape);

    // Auto-convert input and label data
    let xDataTyped: Float32Array;
    let yDataTyped: Float32Array | Int32Array;

    if (xData instanceof Float32Array) {
      xDataTyped = xData;
    } else {
      const cached = this._convertCache.get(xData as object);
      if (cached) {
        xDataTyped = cached;
      } else {
        try {
          xDataTyped = toFloat32Array(xData);
        } catch (e) {
          throw new Error(
            `Model.fit() xData conversion failed: ${(e as Error).message}. ` +
              `Expected data matching inputShape ${JSON.stringify(
                this.inputShape,
              )}.`,
          );
        }
        this._convertCache.set(xData as object, xDataTyped);
      }
    }

    if (loss === "sparse_categorical_crossentropy") {
      if (yData instanceof Int32Array) {
        yDataTyped = yData;
      } else {
        const cached = this._convertCacheInt.get(yData as object);
        if (cached) {
          yDataTyped = cached;
        } else {
          try {
            yDataTyped = toInt32Array(yData);
          } catch (e) {
            throw new Error(
              `Model.fit() yData conversion failed: ${(e as Error).message}. ` +
                `Expected class indices (0, 1, 2, ...).`,
            );
          }
          this._convertCacheInt.set(yData as object, yDataTyped as Int32Array);
        }
      }
    } else {
      if (yData instanceof Float32Array) {
        yDataTyped = yData;
      } else {
        const cached = this._convertCache.get(yData as object);
        if (cached) {
          yDataTyped = cached;
        } else {
          try {
            yDataTyped = toFloat32Array(yData);
          } catch (e) {
            throw new Error(
              `Model.fit() yData conversion failed: ${(e as Error).message}. ` +
                `Expected float values.`,
            );
          }
          this._convertCache.set(yData as object, yDataTyped as Float32Array);
        }
      }
    }

    const nTotal = Math.floor(xDataTyped.length / xElems);
    const lDtype = labelDtype(loss);

    if (nTotal === 0) {
      throw new Error(
        `Model.fit(): xData is empty or too small for inputShape ${JSON.stringify(
          this.inputShape,
        )}. ` +
          `Expected at least ${xElems} elements for one sample, got ${xDataTyped.length}.`,
      );
    }

    const expectedYElems = nTotal * yElems;
    if (yDataTyped.length !== expectedYElems) {
      throw new Error(
        `Model.fit() shape mismatch: xData has ${nTotal} samples, ` +
          `but yData has ${yDataTyped.length} elements (expected ${expectedYElems}).`,
      );
    }

    // Validation split — take from the tail so we don't need to copy.
    const nVal = Math.floor(nTotal * validationSplit);
    const nTrain = nTotal - nVal;

    const sess = await this.ensureSession();
    debug(
      `[Model] fit  samples=${nTotal}  epochs=${epochs}  batchSize=${batchSize}  sync=${sync}  nTrain=${
        nTotal - Math.floor(nTotal * validationSplit)
      }  nVal=${Math.floor(nTotal * validationSplit)}`,
    );

    const history: EpochLogs[] = [];

    for (let ep = 0; ep < epochs; ep++) {
      // When shuffle=true: slice() creates a mutable copy for in-place Fisher-Yates.
      // When shuffle=false: subarray() creates a zero-copy view — saves a full
      // dataset allocation per epoch (can be 10s of MB for large datasets).
      const xTrain = shuffle
        ? xDataTyped.slice(0, nTrain * xElems)
        : xDataTyped.subarray(0, nTrain * xElems);
      const yTrain = shuffle
        ? yDataTyped instanceof Int32Array
          ? (yDataTyped.slice(0, nTrain * yElems) as Int32Array)
          : (yDataTyped.slice(0, nTrain * yElems) as Float32Array)
        : ((yDataTyped instanceof Int32Array
            ? yDataTyped.subarray(0, nTrain * yElems)
            : (yDataTyped as Float32Array).subarray(
                0,
                nTrain * yElems,
              )) as typeof yDataTyped);

      if (shuffle)
        shuffleDataset(xTrain as Float32Array, xElems, yTrain, yElems, nTrain);

      let totalLoss = 0;
      let totalSteps = 0;

      const epochStart = Date.now();

      // Wrap the epoch copy once in a Buffer. Subarray views into it are
      // zero-allocation per batch and safe because xTrain / yTrain (the
      // TypedArrays whose ArrayBuffer these alias) stay alive as const locals
      // for the entire epoch loop. This eliminates one Buffer allocation +
      // copy per batch — down to one Buffer.from() per epoch.
      const xEpochBuf = Buffer.from(
        xTrain.buffer,
        xTrain.byteOffset,
        xTrain.byteLength,
      );
      const yEpochBuf = Buffer.from(
        (yTrain as any).buffer,
        (yTrain as any).byteOffset,
        (yTrain as any).byteLength,
      );
      void xTrain;
      void yTrain; // keep TypedArrays (and their backing memory) live

      for (let start = 0; start < nTrain; start += batchSize) {
        const end = Math.min(start + batchSize, nTrain);
        const batchN = end - start;
        const xShape = [batchN, ...this.inputShape];
        const yShape =
          loss === "sparse_categorical_crossentropy"
            ? [batchN]
            : [batchN, ...seq.outputShape.slice(1).map((s) => s ?? 1)];

        // Zero-allocation subarray views into the epoch Buffer.
        // xEpochBuf shares the backing ArrayBuffer with xTrain, which is kept
        // alive by the void ref above and by being in the outer for(ep) scope.
        // Subarray() returns a Buffer view — no copy, no GC allocation.
        const xBuf = xEpochBuf.subarray(start * xElems * 4, end * xElems * 4);
        const yBuf = yEpochBuf.subarray(start * yElems * 4, end * yElems * 4);
        const { loss: stepLoss } = sync
          ? seq.trainStepSync(sess, opt, xBuf, yBuf, xShape, yShape, lDtype)
          : await seq.trainStep(sess, opt, xBuf, yBuf, xShape, yShape, lDtype);
        void xBuf;
        void yBuf;
        void xEpochBuf;
        void yEpochBuf; // keep refs live
        totalLoss += stepLoss;
        totalSteps += 1;

        if (verbose && process.stdout.isTTY) {
          const pct = Math.floor((end / nTrain) * 100);
          const bars = Math.floor(pct / 5);
          process.stdout.write(
            `\r  Epoch ${ep + 1}/${epochs}  [${"=".repeat(bars)}${" ".repeat(
              20 - bars,
            )}] ` +
              `${end}/${nTrain}  loss: ${(totalLoss / totalSteps).toFixed(4)}`,
          );
        }
      }

      const meanLoss = totalLoss / totalSteps;
      const elapsed = ((Date.now() - epochStart) / 1000).toFixed(1);
      debug(
        `[Model] fit  epoch=${ep + 1}/${epochs}  loss=${meanLoss.toFixed(
          4,
        )}  elapsed=${elapsed}s`,
      );

      // Optional validation pass.
      let valLoss: number | undefined;
      if (nVal > 0) {
        const xVal = xDataTyped.slice(nTrain * xElems);
        const yVal = yDataTyped.slice(nTrain * yElems);
        const xValBuf = Buffer.from(
          xVal.buffer,
          xVal.byteOffset,
          xVal.byteLength,
        );
        const valPreds = await seq.predict(sess, xValBuf, [
          nVal,
          ...this.inputShape,
        ]);
        void xVal;
        void xValBuf; // keep backing memory alive past the await
        // Compute mean squared difference as a proxy val loss (good enough for monitoring).
        valLoss = computeValLoss(
          loss,
          new Float32Array(
            valPreds.data.buffer,
            valPreds.data.byteOffset,
            valPreds.data.byteLength / 4,
          ),
          yVal,
          yElems,
        );
      }

      const logs: EpochLogs = {
        loss: meanLoss,
        ...(valLoss !== undefined ? { valLoss } : {}),
      };
      history.push(logs);
      onEpochEnd?.(ep, logs);

      if (verbose) {
        const valStr =
          valLoss !== undefined ? `  val_loss: ${valLoss.toFixed(4)}` : "";
        process.stdout.write(
          `\r  Epoch ${ep + 1}/${epochs}  loss: ${meanLoss.toFixed(
            4,
          )}${valStr}  (${elapsed}s)\n`,
        );
      }
    }

    return { history };
  }

  // ── predict ───────────────────────────────────────────────────────────────

  /**
   * Run a forward pass and return the output tensor values.
   *
   * Accepts data in multiple formats:
   * - Float32Array (fastest, zero-copy)
   * - JS arrays: [1, 2, 3, ...] or nested [[1, 2, ...], ...] (auto-flattened)
   *
   * @param xData       Input data. Size must match nSamples × prod(inputShape).
   * @param batchSize   Process in chunks to avoid OOM for large datasets. Default: 128.
   * @returns           Float32Array of model outputs (nSamples × prod(outputShape)).
   */
  async predict(xData: DataLike, batchSize = 128): Promise<Float32Array> {
    this.assertCompiled("predict");
    const seq = this._seq!;
    const sess = await this.ensureSession();

    // Auto-convert input data.
    // Cache the Float32Array conversion keyed on the source object reference so
    // that repeated calls with the same array (e.g. benchmark loops) pay the
    // flattenArray + Float32Array allocation cost only once instead of every call.
    // Float32Array inputs are always zero-copy and bypass the cache entirely.
    let xDataTyped: Float32Array;
    if (xData instanceof Float32Array) {
      xDataTyped = xData;
    } else {
      const cached = this._convertCache.get(xData as object);
      if (cached) {
        xDataTyped = cached;
      } else {
        try {
          xDataTyped = toFloat32Array(xData);
        } catch (e) {
          throw new Error(
            `Model.predict() xData conversion failed: ${
              (e as Error).message
            }. ` +
              `Expected data matching inputShape ${JSON.stringify(
                this.inputShape,
              )}.`,
          );
        }
        this._convertCache.set(xData as object, xDataTyped);
      }
    }

    const xElems = this.inputShape.reduce((a, b) => a * b, 1);
    const nTotal = Math.floor(xDataTyped.length / xElems);
    if (nTotal === 0) {
      throw new Error(
        `Model.predict(): xData is empty or too small for inputShape ${JSON.stringify(
          this.inputShape,
        )}. ` +
          `Expected at least ${xElems} elements for one sample, got ${xDataTyped.length}.`,
      );
    }

    // Infer output size from a single forward pass if unknown.
    const outShape = seq.outputShape;
    const outElems = outShape.slice(1).reduce((a, b) => (a ?? 1) * (b ?? 1), 1);
    // One-shot debug: fires only on the first predict() call per model instance,
    // so it never appears in hot-path benchmark iterations.
    if (this._predictCallCount === 0) {
      debug(
        `[Model] predict  inputShape=${JSON.stringify(
          this.inputShape,
        )}  outElems/sample=${outElems}  nTotal=${nTotal}`,
      );
    }
    this._predictCallCount++;

    const result = new Float32Array(nTotal * (outElems ?? 1));
    let written = 0;

    for (let start = 0; start < nTotal; start += batchSize) {
      const end = Math.min(start + batchSize, nTotal);
      const n = end - start;
      const xSlice = xDataTyped.subarray(start * xElems, end * xElems);
      // Same zero-copy race applies on the input side: allocate an owned copy.
      const inBuf = Buffer.from(
        xSlice.buffer,
        xSlice.byteOffset,
        xSlice.byteLength,
      );
      const out = await seq.predict(sess, inBuf, [n, ...this.inputShape]);
      void inBuf; // keep alive past the await
      // out.data is a Buffer owned by the native layer; read it synchronously
      // before anything else can trigger GC.
      const outBuf = out.data as Buffer;
      result.set(
        new Float32Array(
          outBuf.buffer,
          outBuf.byteOffset,
          outBuf.byteLength / 4,
        ),
        written,
      );
      written += outBuf.byteLength / 4;
    }

    return result;
  }

  // ── save ─────────────────────────────────────────────────────────────────

  /**
   * Export a frozen inference graph (.pb) at the given path.
   *
   * The resulting file can be loaded directly by InferencePool:
   *   const pool = await InferencePool.create({ modelPath: "./model.pb" });
   *
   * @param path   File path for the .pb output. The directory must exist.
   */
  async save(path: string): Promise<void> {
    this.assertCompiled("save");
    const sess = await this.ensureSession();
    await this._seq!.exportFrozen(sess, path);
  }

  // ── saveWeights / loadWeights ─────────────────────────────────────────────

  /** Save weights (manifest.toon + weights.bin) to a directory. */
  async saveWeights(dir: string): Promise<void> {
    this.assertCompiled("saveWeights");
    const sess = await this.ensureSession();
    await this._seq!.saveWeights(sess, dir);
  }

  /** Restore weights from a directory previously written by saveWeights(). */
  async loadWeights(dir: string): Promise<void> {
    this.assertCompiled("loadWeights");
    const sess = await this.ensureSession();
    await this._seq!.loadWeights(sess, dir);
  }

  // ── Model inspection ─────────────────────────────────────────────────────

  /**
   * Count the total number of trainable parameters in the model.
   * Only available after compile().
   *
   * @example
   * const model = new Model([28, 28, 1], [...]);
   * model.compile({ loss: "mse" });
   * console.log(model.countParams()); // 123456
   */
  countParams(): number {
    this.assertCompiled("countParams");
    let total = 0;
    for (const layer of this.layerList) {
      if (layer.paramCount) {
        total += layer.paramCount();
      }
    }
    return total;
  }

  // ── Low-level escape hatch ────────────────────────────────────────────────

  /**
   * Print a Keras-style model summary to the console.
   *
   * Displays layer names, types, output shapes, and parameter counts in a
   * formatted table, plus total parameters and estimated FP32 memory usage.
   *
   * Only available after compile().
   *
   * @example
   * model.compile({ loss: "sparse_categorical_crossentropy", optimizer: "adam" });
   * model.summary();
   * // Model Summary
   * // ──────────────────────────────────┬──────────────────────────────────┬───────────┐
   * // Layer (type)                      │ Output Shape                     │ Params    │
   * // ──────────────────────────────────┼──────────────────────────────────┼───────────┤
   * // conv2d (Conv2D)                   │ [null, 224, 224, 32]             │ 896       │
   * // flatten (Flatten)                 │ [null, 1605632]                  │ 0         │
   * // dense_1 (Dense)                   │ [null, 1000]                     │ 1,605,633,000 │
   * // ══════════════════════════════════╪══════════════════════════════════╪═══════════╡
   * // Total params: 1,605,633,896
   * // Trainable params: 1,605,633,896
   * // Non-trainable params: 0
   * // Estimated size (fp32): 6,422.54 MB
   */
  summary(): void {
    this.assertCompiled("summary");

    const output = this.getOutputShape();
    const total = this.countParams();
    const estimatedMB = (total * 4) / 1024 / 1024;

    // Helper to format numbers with thousand separators
    const fmtNum = (n: number): string => {
      return n.toLocaleString("en-US");
    };

    // Helper to format shape
    const fmtShape = (shape: (number | null)[]): string => {
      return "[" + shape.map((d) => (d === null ? "null" : d)).join(", ") + "]";
    };

    // Column widths for alignment
    const colLayer = 32;
    const colShape = 28;
    const colParams = 14;
    const lineWidth = colLayer + colShape + colParams + 6;

    console.log("\nModel Summary");
    console.log("-".repeat(lineWidth));
    console.log(
      String("Layer (type)").padEnd(colLayer) +
        " | " +
        String("Output Shape").padEnd(colShape) +
        " | " +
        String("Params").padEnd(colParams),
    );
    console.log("-".repeat(lineWidth));

    // Print each layer
    for (let i = 0; i < this.layerList.length; i++) {
      const layer = this.layerList[i];
      const config = layer.getConfig();
      const type = config.type;
      const params = layer.paramCount ? layer.paramCount() : 0;

      // Use final output shape for last layer, otherwise "?"
      const shapeStr = i === this.layerList.length - 1 ? fmtShape(output) : "?";

      console.log(
        String(layer.name + " (" + type + ")").padEnd(colLayer) +
          " | " +
          String(shapeStr).padEnd(colShape) +
          " | " +
          String(fmtNum(params)).padStart(colParams),
      );
    }

    console.log("=".repeat(lineWidth));
    console.log(`Total params: ${fmtNum(total)}`);
    console.log(`Estimated size (fp32): ${estimatedMB.toFixed(2)} MB\n`);
  }

  // ── Low-level escape hatch ────────────────────────────────────────────────

  /**
   * Get the input shape defined during Model initialization.
   *
   * @example
   * const model = new Model([28, 28, 1], [...]);
   * console.log(model.getInputShape()); // [28, 28, 1]
   */
  getInputShape(): number[] {
    return [...this.inputShape];
  }

  /**
   * Get the output shape after all layers have been applied.
   * Only available after compile().
   *
   * @example
   * model.compile({ loss: "sparse_categorical_crossentropy", optimizer: "adam" });
   * console.log(model.getOutputShape()); // [null, 10] for Dense(10) output
   *
   * Returns [null, ...] where null represents the batch dimension.
   */
  getOutputShape(): (number | null)[] {
    this.assertCompiled("getOutputShape");
    return [...this._seq!.outputShape];
  }

  /**
   * Direct access to the underlying Sequential, Graph, and Session.
   * Useful when the high-level API doesn't cover a specific use-case.
   */
  get sequential(): Sequential {
    this.assertCompiled("sequential");
    return this._seq!;
  }
  get graph(): Graph {
    this.assertCompiled("graph");
    return this._g!;
  }
  /** Null until the first fit() or predict() call. */
  get sess(): Session | null {
    return this._sess;
  }
}

// ── Validation loss helper ────────────────────────────────────────────────────

/**
 * A quick scalar loss for monitoring validation — not the same as the training
 * loss (which is computed natively inside the graph), but close enough for
 * early-stopping decisions.
 */
function computeValLoss(
  loss: LossFn,
  preds: Float32Array,
  labels: Float32Array | Int32Array,
  yElems: number,
): number {
  const n = preds.length / yElems;
  let total = 0;

  if (loss === "mse") {
    for (let i = 0; i < preds.length; i++) {
      const d = preds[i] - (labels as Float32Array)[i];
      total += d * d;
    }
    return total / preds.length;
  }

  if (loss === "binary_crossentropy") {
    const EPS = 1e-7;
    for (let i = 0; i < preds.length; i++) {
      const p = Math.max(EPS, Math.min(1 - EPS, preds[i]));
      const y = (labels as Float32Array)[i];
      total += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));
    }
    return total / preds.length;
  }

  if (loss === "sparse_categorical_crossentropy") {
    const nClasses = yElems; // preds per sample = nClasses
    const EPS = 1e-7;
    for (let i = 0; i < n; i++) {
      const cls = (labels as Int32Array)[i];
      const p = Math.max(EPS, preds[i * nClasses + cls]);
      total += -Math.log(p);
    }
    return total / n;
  }

  return 0;
}

// ── Pre-built model factories ────────────────────────────────────────────────
//
// Keras-like factory functions for common architectures.
// Matches tf.keras.applications API.

import {
  Conv2D,
  Dense,
  Flatten,
  GlobalAveragePooling2D,
  MaxPooling2D,
} from "./layers.js";
import {
  ConvBnRelu,
  ConvBnRelu6,
  InvertedResidual,
  ResidualBlock,
} from "./easy-layers.js";

/**
 * MobileNetV2 pre-built model.
 *
 * Mobile architecture using inverted residual blocks with skip connections.
 * Efficient for inference on mobile/edge devices.
 *
 * Proper Architecture:
 * - Initial Conv(32, 3×3, stride=2)
 * - Inverted residual blocks (expand → depthwise → project + skip)
 * - Skip connections applied when stride=1 and input/output channels match
 * - Classification head: Conv(1280) → GlobalAvgPool → Dense(classes)
 *
 * Reference: MobileNetV2 (https://arxiv.org/abs/1801.04381)
 *
 * @param inputShape Input spatial dimensions. Default: [224, 224, 3].
 * @param classes    Number of output classes. Default: 1000.
 * @returns          Uncompiled Model instance.
 *
 * @example
 * const model = mobilenetv2([224, 224, 3], 1000);
 * model.compile({ loss: "sparse_categorical_crossentropy" });
 * const nParams = model.countParams(); // ~3.5M
 */
export function mobilenetv2(
  inputShape: [number, number, number] = [224, 224, 3],
  classes: number = 1000,
): Model {
  const layers: Layer[] = [
    // Initial convolution with BN + ReLU6: 224×224×3 → 112×112×32 (stride=2)
    // MobileNetV2 uses ReLU6 everywhere for better quantization
    new ConvBnRelu6(32, 3, {
      stride: 2,
      padding: "SAME",
      name: "conv_stem",
    }),

    // Inverted residual blocks: (inCh, outCh, {expandRatio, stride})
    // t=1, c=16, n=1, s=1
    new InvertedResidual(32, 16, {
      expandRatio: 1,
      stride: 1,
      name: "ir1",
    }),

    // t=6, c=24, n=2, s=2 (downsampling + skip block)
    new InvertedResidual(16, 24, {
      expandRatio: 6,
      stride: 2,
      name: "ir2_0",
    }),
    new InvertedResidual(24, 24, {
      expandRatio: 6,
      stride: 1,
      name: "ir2_1",
    }),

    // t=6, c=32, n=3, s=2
    new InvertedResidual(24, 32, {
      expandRatio: 6,
      stride: 2,
      name: "ir3_0",
    }),
    new InvertedResidual(32, 32, {
      expandRatio: 6,
      stride: 1,
      name: "ir3_1",
    }),
    new InvertedResidual(32, 32, {
      expandRatio: 6,
      stride: 1,
      name: "ir3_2",
    }),

    // t=6, c=64, n=4, s=2
    new InvertedResidual(32, 64, {
      expandRatio: 6,
      stride: 2,
      name: "ir4_0",
    }),
    new InvertedResidual(64, 64, {
      expandRatio: 6,
      stride: 1,
      name: "ir4_1",
    }),
    new InvertedResidual(64, 64, {
      expandRatio: 6,
      stride: 1,
      name: "ir4_2",
    }),
    new InvertedResidual(64, 64, {
      expandRatio: 6,
      stride: 1,
      name: "ir4_3",
    }),

    // t=6, c=96, n=3, s=1 (no downsampling)
    new InvertedResidual(64, 96, {
      expandRatio: 6,
      stride: 1,
      name: "ir5_0",
    }),
    new InvertedResidual(96, 96, {
      expandRatio: 6,
      stride: 1,
      name: "ir5_1",
    }),
    new InvertedResidual(96, 96, {
      expandRatio: 6,
      stride: 1,
      name: "ir5_2",
    }),

    // t=6, c=160, n=3, s=2 (downsampling)
    new InvertedResidual(96, 160, {
      expandRatio: 6,
      stride: 2,
      name: "ir6_0",
    }),
    new InvertedResidual(160, 160, {
      expandRatio: 6,
      stride: 1,
      name: "ir6_1",
    }),
    new InvertedResidual(160, 160, {
      expandRatio: 6,
      stride: 1,
      name: "ir6_2",
    }),

    // t=6, c=320, n=1, s=1 (no downsampling)
    new InvertedResidual(160, 320, {
      expandRatio: 6,
      stride: 1,
      name: "ir7",
    }),

    // Classification head: Conv1280 → BN → ReLU6 → GlobalAvgPool → Dense
    // Using ConvBnRelu6 for strict paper compliance (some variants omit final ReLU6,
    // but TensorFlow/Keras includes it for better training dynamics)
    new ConvBnRelu6(1280, 1, { name: "conv_head" }),
    new GlobalAveragePooling2D(),
    new Dense(classes, { name: "predictions" }),
  ];

  return new Model(inputShape, layers);
}

/**
 * ResNet50 pre-built model.
 *
 * Deep residual network with 50 layers using bottleneck blocks.
 * Each residual block contains skip connections that allow gradients to
 * flow directly, enabling training of very deep networks.
 *
 * Proper Architecture:
 * - Initial Conv(64, 7×7, stride=2) + MaxPool
 * - Layer1: 3× identity blocks [64→64→256]
 * - Layer2: 1× projection (stride=2) + 3× identity blocks [128→128→512]
 * - Layer3: 1× projection (stride=2) + 5× identity blocks [256→256→1024]
 * - Layer4: 1× projection (stride=2) + 2× identity blocks [512→512→2048]
 * - Classification head: GlobalAvgPool → Dense(classes)
 *
 * Skip connections reduce vanishing gradient problem and enable training
 * of very deep networks. Each bottleneck block has 1×1 (reduce) → 3×3 (process)
 * → 1×1 (expand) structure within a skip connection.
 *
 * Reference: ResNet (https://arxiv.org/abs/1512.03385)
 *
 * @param inputShape Input spatial dimensions. Default: [224, 224, 3].
 * @param classes    Number of output classes. Default: 1000.
 * @returns          Uncompiled Model instance.
 *
 * @example
 * const model = resnet50([224, 224, 3], 1000);
 * model.compile({ loss: "sparse_categorical_crossentropy" });
 * const nParams = model.countParams(); // ~25.6M
 */
export function resnet50(
  inputShape: [number, number, number] = [224, 224, 3],
  classes: number = 1000,
): Model {
  const layers: Layer[] = [
    // ── Initial convolution + pooling ────────────────────────────────────
    // 224×224×3 → 112×112×64 → 56×56×64
    // Per original ResNet paper: Conv → BN → ReLU → MaxPool
    new ConvBnRelu(64, 7, {
      stride: 2,
      padding: "SAME",
      name: "conv1",
    }),
    new MaxPooling2D({
      poolSize: 3,
      strides: 2,
      padding: "SAME",
      name: "pool1",
    }),

    // ── Layer1: 56×56 spatial (3 blocks, bottleneck=[64, 64, 256]) ─────────
    new ResidualBlock(64, 64, 256, {
      stride: 1,
      name: "layer1_block0",
    }),
    new ResidualBlock(256, 64, 256, {
      stride: 1,
      name: "layer1_block1",
    }),
    new ResidualBlock(256, 64, 256, {
      stride: 1,
      name: "layer1_block2",
    }),

    // ── Layer2: 56×56 → 28×28 (4 blocks, bottleneck=[128, 128, 512]) ──────
    // First block uses stride=2 (projection block)
    new ResidualBlock(256, 128, 512, {
      stride: 2,
      name: "layer2_block0",
    }),
    // Remaining blocks are identity blocks
    new ResidualBlock(512, 128, 512, {
      stride: 1,
      name: "layer2_block1",
    }),
    new ResidualBlock(512, 128, 512, {
      stride: 1,
      name: "layer2_block2",
    }),
    new ResidualBlock(512, 128, 512, {
      stride: 1,
      name: "layer2_block3",
    }),

    // ── Layer3: 28×28 → 14×14 (6 blocks, bottleneck=[256, 256, 1024]) ──────
    new ResidualBlock(512, 256, 1024, {
      stride: 2,
      name: "layer3_block0",
    }),
    new ResidualBlock(1024, 256, 1024, {
      stride: 1,
      name: "layer3_block1",
    }),
    new ResidualBlock(1024, 256, 1024, {
      stride: 1,
      name: "layer3_block2",
    }),
    new ResidualBlock(1024, 256, 1024, {
      stride: 1,
      name: "layer3_block3",
    }),
    new ResidualBlock(1024, 256, 1024, {
      stride: 1,
      name: "layer3_block4",
    }),
    new ResidualBlock(1024, 256, 1024, {
      stride: 1,
      name: "layer3_block5",
    }),

    // ── Layer4: 14×14 → 7×7 (3 blocks, bottleneck=[512, 512, 2048]) ────────
    new ResidualBlock(1024, 512, 2048, {
      stride: 2,
      name: "layer4_block0",
    }),
    new ResidualBlock(2048, 512, 2048, {
      stride: 1,
      name: "layer4_block1",
    }),
    new ResidualBlock(2048, 512, 2048, {
      stride: 1,
      name: "layer4_block2",
    }),

    // ── Classification head ──────────────────────────────────────────────
    // 7×7×2048 → 2048 → classes
    new GlobalAveragePooling2D(),
    new Dense(classes, { name: "predictions" }),
  ];

  return new Model(inputShape, layers);
}
