import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { toonEncode, toonDecode } from "../toon.js";
import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import { Graph } from "../graph.js";
import type { Session } from "../session.js";
import type { Layer, LayerParam, WeightMap, LayerConfig } from "./layer.js";
import {
  Dense,
  Flatten,
  Conv2D,
  DepthwiseConv2D,
  SeparableConv2D,
  MaxPooling2D,
  GlobalAveragePooling2D,
  ZeroPadding2D,
  BatchNormalization,
} from "./layers.js";
import type { ParamSpec, FeedEntry } from "../optimizers/sgd.js";
import { placeholder, constant } from "../ops/array_ops.js";
import { mean } from "../ops/math_ops.js";
import {
  globalVariablesInitializer,
  assignVariable,
} from "../ops/variable_ops.js";
import { sigmoidCrossEntropyWithLogits } from "../ops/nn_ops.js";
import { getAddon } from "../_native.js";

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------
export type LossFn =
  | "sparse_categorical_crossentropy"
  | "binary_crossentropy"
  | "mse";

// ---------------------------------------------------------------------------
// Optimizer duck-type — any optimizer with init() + applyGradients() works.
// ---------------------------------------------------------------------------
export interface Optimizer {
  init(sess: Session): Promise<void>;
  applyGradients(sess: Session, feeds: FeedEntry[]): Promise<void>;
  /** Op names that constitute one optimizer step. Used to merge forward+backward into one runAsync. */
  readonly targetOps: string[];
}

export interface TrainStepResult {
  loss: number;
}

// ---------------------------------------------------------------------------
// Sequential
//
// Correct usage:
//
//   // 1. Build graph
//   const model = new Sequential(g, [
//     new Dense(128, { activation: "relu" }),
//     new Dense(10,  { activation: "softmax" }),
//   ]);
//
//   // 2. compile() — no optimizer here. Builds graph, exposes params.
//   model.compile({ loss: "sparse_categorical_crossentropy", inputShape: [784] });
//
//   // 3. Construct optimizer from model.params — grad tensors exist now.
//   const opt = new Adam(g, model.params, 0.001);
//
//   // 4. init(sess, opt) — run variable inits + optimizer state init.
//   await model.init(sess, opt);
//
//   // 5. Training loop — pass opt explicitly each step.
//   for (const [xBuf, yBuf] of batches) {
//     const { loss } = await model.trainStep(sess, opt, xBuf, yBuf, ...);
//   }
//
//   // 6. Inference — no optimizer needed.
//   const preds = await model.predict(sess, xBuf, xShape);
//
// Why optimizer is not in compile():
//   compile() calls g.addGradients() which is what populates model.params
//   with real gradient tensors. The optimizer constructor needs those grad
//   tensors to wire update ops. So the optimizer must be created AFTER
//   compile(), not passed into it.
// ---------------------------------------------------------------------------

export class Sequential {
  private readonly g: Graph;
  private readonly layers: Layer[];

  // Populated during compile()
  private _xPlaceholder!: Tensor;
  private _yPlaceholder!: Tensor;
  private _compiledInputShape!: number[];
  private _layerParams!: LayerParam[];
  private _outputTensor!: Tensor;
  private _lossTensor!: Tensor;
  private _allParams!: ParamSpec[];
  private _allInitOp!: string;
  private _labelDtype!: DType;
  private _loss!: string;
  private _extraUpdateOps!: string[]; // EMA ops from BN layers
  private _nonTrainable!: LayerParam[]; // moving_mean/var — saved but not trained
  private _hasBatchNorm!: boolean; // true if any layer is BatchNormalization
  private compiled = false;

  constructor(g: Graph, layers: Layer[]) {
    this.g = g;
    this.layers = layers;
  }

  /**
   * compile — wire the full computation graph.
   *
   * After this returns, model.params contains ParamSpec entries with
   * real gradient tensors from g.addGradients(). Use those to construct
   * your optimizer before calling init().
   */
  compile(opts: {
    loss: LossFn;
    inputShape: number[];
    labelDtype?: DType;
  }): void {
    if (this.compiled) throw new Error("Sequential.compile() called twice");

    const { loss, inputShape } = opts;
    this._compiledInputShape = inputShape;
    this._loss = loss;
    this._labelDtype = opts.labelDtype ?? DType.INT32;

    // ── Input placeholder ─────────────────────────────────────────────────
    this._xPlaceholder = placeholder(
      this.g,
      "x",
      [null, ...inputShape],
      DType.FLOAT32,
    );

    // ── Thread layers ─────────────────────────────────────────────────────
    let current: Tensor = this._xPlaceholder;
    let shape: (number | null)[] = [null, ...inputShape];

    for (const layer of this.layers) {
      shape = layer.build(this.g, current, shape);
      current = layer.output;
    }
    this._outputTensor = current;

    // ── Collect all layer params ──────────────────────────────────────────
    const allLayerParams = this.layers.flatMap((l) => l.layerParams);

    // ── Label placeholder ─────────────────────────────────────────────────
    const yShape: (number | null)[] =
      loss === "mse" ? [null, shape[shape.length - 1] ?? null] : [null];
    this._yPlaceholder = placeholder(this.g, "y", yShape, this._labelDtype);

    // ── Loss op + gradient computation ────────────────────────────────────
    //
    // SparseSoftmaxCrossEntropyWithLogits has no registered gradient in the
    // TF C API gradient registry, so calling addGradients([meanLoss], params)
    // fails with "No gradient defined for op: SparseSoftmaxCrossEntropyWithLogits".
    //
    // The fix: the op exposes output[1] — the exact analytical gradient
    // of the loss w.r.t. logits (= softmax(logits) - one_hot(labels)).
    // We scale it by 1/batch_size to get dMeanLoss/dLogits, then pass it
    // as the initial dx to addGradients([outputTensor], params, [dLogits]).
    // TF differentiates backward through matmul/relu/biasAdd/conv just fine.
    //
    // binary_crossentropy and mse are built entirely from primitives that
    // TF can differentiate through, so they use addGradients([loss], params)
    // directly without this workaround.

    const readTensors = allLayerParams.map((p) => p.read);
    let grads: Tensor[];

    switch (loss) {
      case "sparse_categorical_crossentropy": {
        // Output[0]: per-example loss [batch]
        // Output[1]: dLoss/dLogits per example [batch, num_classes]
        const [lossPerEx, backprop] = this.g.addOp(
          "SparseSoftmaxCrossEntropyWithLogits",
          [this._outputTensor, this._yPlaceholder],
          {},
          "loss/xent",
        );

        this._lossTensor = mean(this.g, lossPerEx, [0], false, "loss/mean");

        // Dynamic batch size — Size on a 1-D label tensor = batch size.
        const [batchN] = this.g.addOp(
          "Size",
          [this._yPlaceholder],
          {
            out_type: { kind: "type", value: DType.INT32 },
          },
          "loss/batch_n",
        );
        const [batchNf] = this.g.addOp(
          "Cast",
          [batchN],
          {
            DstT: { kind: "type", value: DType.FLOAT32 },
          },
          "loss/batch_n_f32",
        );

        // dMeanLoss/dLogits = backprop / batch_size  [batch, num_classes]
        const [dLogits] = this.g.addOp(
          "RealDiv",
          [backprop, batchNf],
          {},
          "loss/dL_dlogits",
        );

        // Differentiate through the network layers only — not through
        // the loss op which has no registered C API gradient.
        grads = this.g.addGradients([this._outputTensor], readTensors, [
          dLogits,
        ]);
        break;
      }

      case "binary_crossentropy": {
        // Built from primitives — TF differentiates through all of them.
        const lossPerEx = sigmoidCrossEntropyWithLogits(
          this.g,
          this._yPlaceholder,
          this._outputTensor,
          "loss/bce",
        );
        this._lossTensor = mean(this.g, lossPerEx, [0], false, "loss/mean");
        grads = this.g.addGradients([this._lossTensor], readTensors);
        break;
      }

      case "mse": {
        const [diff] = this.g.addOp(
          "Sub",
          [this._outputTensor, this._yPlaceholder],
          {},
          "loss/diff",
        );
        const [sq] = this.g.addOp("Square", [diff], {}, "loss/sq");
        const lossPerEx = mean(this.g, sq, [1], false, "loss/mse_per_sample");
        this._lossTensor = mean(this.g, lossPerEx, [0], false, "loss/mean");
        grads = this.g.addGradients([this._lossTensor], readTensors);
        break;
      }

      default:
        throw new Error(`Unknown loss: ${loss}`);
    }
    // ── Assemble ParamSpec with real grad tensors ─────────────────────────
    this._layerParams = allLayerParams;
    this._extraUpdateOps = this.layers.flatMap((l) => l.updateOps?.() ?? []);
    this._nonTrainable = this.layers.flatMap(
      (l) => l.nonTrainableParams?.() ?? [],
    );
    this._hasBatchNorm = this.layers.some(
      (l) => l instanceof BatchNormalization,
    );
    this._allParams = allLayerParams.map((p, i) => ({
      handle: p.handle,
      grad: grads[i],
      dtype: p.dtype,
      name: p.name,
    }));

    // ── Global variable init op ───────────────────────────────────────────
    const initOpNames = allLayerParams.map((p) => p.initOp);
    // Include init ops for non-trainable BN moving stats alongside trainable params.
    const allInitOps = [
      ...initOpNames,
      ...this.layers.flatMap(
        (l) => l.nonTrainableParams?.().map((p) => p.initOp) ?? [],
      ),
    ];
    this._allInitOp = globalVariablesInitializer(
      this.g,
      allInitOps,
      "model_init",
    );

    this.compiled = true;
  }

  /**
   * init — run all variable initialisations and the optimizer's state init.
   *
   * @param sess The session to run on
   * @param opt  The optimizer (must be constructed from model.params)
   */
  async init(sess: Session, opt: Optimizer): Promise<void> {
    this.assertCompiled("init");
    await sess.run([], [], [this._allInitOp]);
    await opt.init(sess);
  }

  /**
   * trainStep — one forward pass, gradient computation, and weight update.
   *
   * Two sequential TF_SessionRun calls:
   *   1. Fetch loss (forward pass runs as part of this)
   *   2. Run optimizer update ops (backward pass + weight update)
   *
   * Keeping them separate avoids the optimizer needing to expose its
   * internal step op name. The overhead is one extra C++ call per step,
   * which is negligible compared to the matmul/conv cost.
   *
   * @param sess       Session to run on
   * @param opt        Optimizer (same instance used in init)
   * @param xBuf       Float32 input bytes [batchSize, ...inputShape]
   * @param yBuf       Label bytes (INT32 class indices or FLOAT32 values)
   * @param xShape     [batchSize, ...inputShape]
   * @param yShape     [batchSize] for classification, [batchSize, units] for mse
   * @param labelDtype DType for labels — defaults to what was set in compile()
   */
  async trainStep(
    sess: Session,
    opt: Optimizer,
    xBuf: Buffer,
    yBuf: Buffer,
    xShape: number[],
    yShape: number[],
    labelDtype?: DType,
  ): Promise<TrainStepResult> {
    this.assertCompiled("trainStep");

    const lDtype = labelDtype ?? this._labelDtype;
    const feeds: FeedEntry[] = [
      [this._xPlaceholder, { dtype: DType.FLOAT32, shape: xShape, data: xBuf }],
      [this._yPlaceholder, { dtype: lDtype, shape: yShape, data: yBuf }],
    ];

    // Forward pass + backward pass + EMA updates in a single TF_SessionRun.
    // opt.targetOps    = optimizer update ops (backward pass + weight update)
    // _extraUpdateOps  = BN EMA updates for moving_mean / moving_var
    // All run atomically — no round-trips between them.
    const [lossOut] = await sess.runAsync(
      feeds,
      [this._lossTensor],
      [...opt.targetOps, ...this._extraUpdateOps],
    );

    const lossVal = new Float32Array(
      lossOut.data.buffer,
      lossOut.data.byteOffset,
      1,
    )[0];
    return { loss: lossVal };
  }

  /**
   * trainStepSync — fully synchronous training step.
   *
   * Calls TF_SessionRun directly on the calling thread via sess.runSync(),
   * bypassing libuv scheduling, TSFN signalling, and Promise overhead entirely.
   * TF's eigen thread pool still provides internal op parallelism via intraOpThreads.
   *
   * This matches Python TF's sess.run([train_op, loss_op]) exactly — the calling
   * thread blocks for the full forward + backward pass with no async machinery.
   *
   * Only appropriate when you intentionally own the thread — training loops,
   * CLI scripts, batch jobs. Never use this in an HTTP request handler.
   *
   * @param sess       Session (create with intraOpThreads = availableParallelism())
   * @param opt        Optimizer (same instance used in init)
   * @param xBuf       Float32 input bytes
   * @param yBuf       Label bytes
   * @param xShape     [batchSize, ...inputShape]
   * @param yShape     [batchSize] for classification
   * @param labelDtype DType for labels — defaults to what was set in compile()
   */
  trainStepSync(
    sess: Session,
    opt: Optimizer,
    xBuf: Buffer,
    yBuf: Buffer,
    xShape: number[],
    yShape: number[],
    labelDtype?: DType,
  ): TrainStepResult {
    this.assertCompiled("trainStepSync");

    const lDtype = labelDtype ?? this._labelDtype;
    const feeds: FeedEntry[] = [
      [this._xPlaceholder, { dtype: DType.FLOAT32, shape: xShape, data: xBuf }],
      [this._yPlaceholder, { dtype: lDtype, shape: yShape, data: yBuf }],
    ];

    // Single synchronous TF_SessionRun: forward pass (loss fetch) + backward
    // pass (opt.targetOps) + BN EMA updates (_extraUpdateOps), all in one shot.
    const [lossOut] = sess.runSync(
      feeds,
      [this._lossTensor],
      [...opt.targetOps, ...this._extraUpdateOps],
    );

    const lossVal = new Float32Array(
      lossOut.data.buffer,
      lossOut.data.byteOffset,
      1,
    )[0];
    return { loss: lossVal };
  }

  /**
   * predict — forward pass only, no gradient computation or update.
   *
   * Uses runAsync() so the event loop stays free during inference.
   * For Worker threads where blocking is acceptable, call sess.run() directly.
   *
   * ⚠ BatchNormalization warning:
   *   This model contains BatchNormalization layers. predict() runs the training
   *   graph which uses *batch statistics*, not the learned moving_mean/moving_var.
   *   For correct inference results, use exportFrozen() → InferencePool instead:
   *
   *     model.exportFrozen(sess, "model.pb");
   *     const pool = await InferencePool.create({ modelPath: "model.pb" });
   */
  async predict(
    sess: Session,
    xBuf: Buffer,
    xShape: number[],
  ): Promise<{ data: Buffer; shape: number[]; dtype: DType }> {
    this.assertCompiled("predict");
    if (this._hasBatchNorm) {
      process.emitWarning(
        "Sequential.predict() uses batch statistics for BatchNormalization layers. " +
          "For correct inference use exportFrozen() → InferencePool.",
        { code: "ISIDORUS_BN_PREDICT" },
      );
    }
    const [out] = await sess.runAsync(
      [
        [
          this._xPlaceholder,
          { dtype: DType.FLOAT32, shape: xShape, data: xBuf },
        ],
      ],
      [this._outputTensor],
    );
    return out;
  }

  // ── Accessors ─────────────────────────────────────────────────────────────

  /** Input placeholder tensor — available after compile(). */
  get xPlaceholder(): Tensor {
    this.assertCompiled("xPlaceholder");
    return this._xPlaceholder;
  }

  /** Label placeholder tensor — available after compile(). */
  get yPlaceholder(): Tensor {
    this.assertCompiled("yPlaceholder");
    return this._yPlaceholder;
  }

  /** Final layer output tensor — available after compile(). */
  get output(): Tensor {
    this.assertCompiled("output");
    return this._outputTensor;
  }

  /** Scalar mean loss tensor — available after compile(). */
  get loss(): Tensor {
    this.assertCompiled("loss");
    return this._lossTensor;
  }

  /**
   * All parameter specs with real gradient tensors.
   * Available after compile(). Use these to construct your optimizer.
   */
  get params(): ParamSpec[] {
    this.assertCompiled("params");
    return this._allParams;
  }

  /** Label dtype resolved during compile(). */
  get labelDtype(): DType {
    this.assertCompiled("labelDtype");
    return this._labelDtype;
  }

  // ── Model I/O ─────────────────────────────────────────────────────────────

  /**
   * readWeights — read all current parameter values from the session.
   *
   * Uses runSync (direct TF_SessionRun, no libuv) since this is called from
   * save/export contexts where blocking the event loop is acceptable.
   *
   * @returns Map from param name → {data, shape, dtype}
   */
  readWeights(sess: Session): WeightMap {
    this.assertCompiled("readWeights");
    // Include both trainable params (gamma, beta, kernels, biases) and
    // non-trainable moving statistics (moving_mean, moving_var) so that
    // saveWeights/exportFrozen capture the full model state.
    const allParams = [...this._layerParams, ...this._nonTrainable];
    const values = sess.runSync(
      [],
      allParams.map((p) => p.read),
      [],
    );
    const map = new Map<
      string,
      { data: Buffer; shape: number[]; dtype: DType }
    >();
    for (let i = 0; i < allParams.length; i++) {
      const p = allParams[i];
      const v = values[i];
      map.set(p.name, {
        data: v.data,
        shape: v.shape,
        dtype: v.dtype as DType,
      });
    }
    return map;
  }

  /**
   * saveWeights — persist trained weights to disk.
   *
   * Writes two files into `dir`:
   *   manifest.json  — param names, shapes, dtypes, byte offsets
   *   weights.bin    — raw float32 bytes for all params concatenated
   *
   * Compatible with loadWeights(). For inference deployment use
   * exportFrozen() instead, which produces a .pb loadable by InferencePool.
   *
   * @param sess  Session holding the current variable values
   * @param dir   Output directory (created if it does not exist)
   */
  saveWeights(sess: Session, dir: string): void {
    this.assertCompiled("saveWeights");
    mkdirSync(dir, { recursive: true });

    const weights = this.readWeights(sess);
    const parts: Buffer[] = [];
    let offset = 0;

    type ParamRecord = {
      name: string;
      shape: number[];
      dtype: number;
      byteOffset: number;
      byteLength: number;
    };
    const params: ParamRecord[] = [];

    for (const [name, entry] of weights) {
      parts.push(entry.data);
      params.push({
        name,
        shape: entry.shape,
        dtype: Number(entry.dtype),
        byteOffset: offset,
        byteLength: entry.data.byteLength,
      });
      offset += entry.data.byteLength;
    }

    const manifest = {
      version: 1,
      inputShape: this._compiledInputShape,
      params,
    };

    // TOON encodes the manifest as a compact binary — faster to parse than JSON,
    // no repeated key strings, integers stored as 4-byte LE rather than decimal text.
    writeFileSync(join(dir, "manifest.toon"), toonEncode(manifest));
    writeFileSync(join(dir, "weights.bin"), Buffer.concat(parts));
  }

  /**
   * loadWeights — restore weights from a saveWeights() checkpoint.
   *
   * Reads manifest.json + weights.bin from `dir` and assigns each param
   * back to its variable using AssignVariableOp, matching by name.
   *
   * @param sess  Session to restore into
   * @param dir   Checkpoint directory written by saveWeights()
   */
  loadWeights(sess: Session, dir: string): void {
    this.assertCompiled("loadWeights");

    const manifestBuf = readFileSync(join(dir, "manifest.toon"));
    const manifest = toonDecode(manifestBuf) as {
      version: number;
      inputShape: number[];
      params: {
        name: string;
        shape: number[];
        dtype: number;
        byteOffset: number;
        byteLength: number;
      }[];
    };

    if (manifest.version !== 1)
      throw new Error(
        `loadWeights: unsupported checkpoint version ${manifest.version}`,
      );

    const bin = readFileSync(join(dir, "weights.bin"));

    // Build name → LayerParam map covering trainable + non-trainable params.
    const paramByName = new Map(
      [...this._layerParams, ...this._nonTrainable].map((p) => [p.name, p]),
    );

    for (const rec of manifest.params) {
      const lp = paramByName.get(rec.name);
      if (!lp) continue; // checkpoint has a param not in this model — skip

      const slice = bin.slice(rec.byteOffset, rec.byteOffset + rec.byteLength);

      // Const → AssignVariable → run to update the variable.
      const valueConst = constant(
        this.g,
        slice,
        rec.shape,
        rec.dtype as DType,
        `restore/${rec.name}/value`,
      );
      const assignOp = assignVariable(
        this.g,
        lp.handle,
        valueConst,
        rec.dtype as DType,
        `restore/${rec.name}/assign`,
      );
      sess.runSync([], [], [assignOp]);
    }
  }

  /**
   * exportFrozen — export a frozen inference graph (.pb) from trained weights.
   *
   * Builds a brand-new Graph where:
   *   • All weights are Const ops with current variable values baked in
   *   • No VarHandleOp, ReadVariableOp, gradient, or optimizer ops present
   *   • Input is a Placeholder named "inputs" (matches InferencePool convention)
   *
   * The resulting .pb is directly loadable by InferencePool — no Python,
   * no freeze_graph script, no SavedModel conversion step needed.
   *
   * @example
   * // Train for N steps, then export for production inference:
   * for (let i = 0; i < N; i++)
   *   model.trainStepSync(sess, opt, xBuf, yBuf, xShape, yShape);
   * model.exportFrozen(sess, "model.pb");
   *
   * // Later, in the inference server:
   * const pool = await InferencePool.create({ modelPath: "model.pb" });
   *
   * @param sess        Session holding current variable values
   * @param outputPath  Destination .pb file path
   */
  exportFrozen(sess: Session, outputPath: string): void {
    this.assertCompiled("exportFrozen");

    // 1. Read current weight values from the training session.
    const weights = this.readWeights(sess);

    // 2. Build a fresh inference-only graph — no variables, no gradients.
    const frozenG = new Graph(new (getAddon().Graph)());
    const inputShape = this._compiledInputShape;

    // 3. Add input Placeholder — name "inputs" matches InferencePool discovery.
    const [phInput] = frozenG.addOp(
      "Placeholder",
      [],
      {
        dtype: { kind: "type", value: DType.FLOAT32 },
        shape: { kind: "shape", value: [1, ...inputShape] },
      },
      "inputs",
    );

    // 4. Walk the layers in order, building the frozen forward pass.
    let current: Tensor = phInput;
    let currentShape: (number | null)[] = [1, ...inputShape];

    for (const layer of this.layers) {
      const result = layer.buildFrozen(frozenG, current, currentShape, weights);
      current = result.tensor;
      currentShape = result.shape;
    }

    // 5. Serialize the frozen graph and write to disk.
    const graphDef = frozenG.toGraphDef();
    writeFileSync(outputPath, graphDef);
  }

  /**
   * initVariables — initialise all model variables without an optimizer.
   *
   * Use this when loading a model for inference or to restore a checkpoint:
   *   const { model, g } = Sequential.loadModel("./saved");
   *   const sess = session(g, { intraOpThreads: availableParallelism() });
   *   model.restore(sess, "./saved");   // calls initVariables + loadWeights
   *
   * For training, use model.init(sess, opt) instead — it also initialises the
   * optimizer state (Adam moments, RMSProp variances, etc.).
   */
  initVariables(sess: Session): void {
    this.assertCompiled("initVariables");
    sess.runSync([], [], [this._allInitOp]);
  }

  /**
   * restore — initialise variables and load weights from a saveModel() / saveWeights()
   * checkpoint in one call.
   *
   * Convenience wrapper around initVariables() + loadWeights().
   * Use this when loading a model for inference.
   *
   * @param sess  Freshly created session (variables not yet initialised)
   * @param dir   Checkpoint directory from saveModel() or saveWeights()
   */
  restore(sess: Session, dir: string): void {
    this.initVariables(sess);
    this.loadWeights(sess, dir);
  }

  /**
   * saveModel — persist the full model (architecture + weights) to disk.
   *
   * Writes two files into `dir`:
   *   model.json   — layer configs, loss, inputShape, param byte-offset manifest
   *   weights.bin  — raw float32 bytes for all parameters concatenated
   *
   * The saved model can be reconstructed without any .pb file or Python:
   *   const { model, g } = Sequential.loadModel("./saved");
   *   const sess = session(g, { intraOpThreads: availableParallelism() });
   *   model.restore(sess, "./saved");
   *
   * For deployment behind InferencePool, use exportFrozen() which produces
   * a .pb compatible with the autotuner and concurrency machinery.
   *
   * @param sess  Session holding current variable values
   * @param dir   Output directory (created if it does not exist)
   */
  saveModel(sess: Session, dir: string): void {
    this.assertCompiled("saveModel");
    mkdirSync(dir, { recursive: true });

    const weights = this.readWeights(sess);
    const parts: Buffer[] = [];
    let offset = 0;

    type ParamRecord = {
      name: string;
      shape: number[];
      dtype: number;
      byteOffset: number;
      byteLength: number;
    };
    const params: ParamRecord[] = [];

    for (const [name, entry] of weights) {
      parts.push(entry.data);
      params.push({
        name,
        shape: entry.shape,
        dtype: Number(entry.dtype),
        byteOffset: offset,
        byteLength: entry.data.byteLength,
      });
      offset += entry.data.byteLength;
    }

    const modelJson = {
      version: 1,
      loss: this._loss,
      inputShape: this._compiledInputShape,
      layers: this.layers.map((l) => l.getConfig()),
      params,
    };

    // TOON for architecture + manifest — faster parse, more compact than JSON.
    writeFileSync(join(dir, "model.toon"), toonEncode(modelJson));
    writeFileSync(join(dir, "weights.bin"), Buffer.concat(parts));
  }

  /**
   * Sequential.loadModel — reconstruct a model saved with saveModel().
   *
   * Reads model.json from `dir`, rebuilds all layers from their configs,
   * creates a new Graph, and calls compile() with the stored loss and
   * inputShape. Returns the compiled model and its graph — both needed
   * to create a Session.
   *
   * @example
   * const { model, g } = Sequential.loadModel("./saved");
   * const sess = session(g, { intraOpThreads: availableParallelism() });
   * model.restore(sess, "./saved");      // initVariables + loadWeights
   * const out = model.predict(sess, xBuf, xShape);
   *
   * @param dir  Directory written by saveModel()
   * @returns    { model: Sequential, g: Graph }
   */
  static loadModel(dir: string): { model: Sequential; g: Graph } {
    const raw = toonDecode(readFileSync(join(dir, "model.toon"))) as {
      version: number;
      loss: string;
      inputShape: number[];
      layers: LayerConfig[];
      params: unknown[];
    };

    if (raw.version !== 1)
      throw new Error(
        `Sequential.loadModel: unsupported version ${raw.version}`,
      );

    // Reconstruct layers from their configs.
    const layers = raw.layers.map((cfg) => Sequential._layerFromConfig(cfg));

    // Build a new graph and compile.
    const g = new Graph(new (getAddon().Graph)());
    const model = new Sequential(g, layers);
    model.compile({ loss: raw.loss as LossFn, inputShape: raw.inputShape });

    return { model, g };
  }

  /** @internal Reconstruct a single layer from a LayerConfig. */
  private static _layerFromConfig(cfg: LayerConfig): Layer {
    switch (cfg.type) {
      case "Dense":
        return new Dense(cfg.units, {
          activation: cfg.activation,
          useBias: cfg.useBias,
          name: cfg.name,
        });
      case "Conv2D":
        return new Conv2D(cfg.filters, {
          kernelSize: cfg.kernelSize,
          strides: cfg.strides,
          padding: cfg.padding,
          activation: cfg.activation,
          useBias: cfg.useBias,
          name: cfg.name,
        });
      case "Flatten":
        return new Flatten({ name: cfg.name });
      case "DepthwiseConv2D":
        return new DepthwiseConv2D({
          kernelSize: cfg.kernelSize,
          strides: cfg.strides,
          padding: cfg.padding,
          depthMultiplier: cfg.depthMultiplier,
          activation: cfg.activation,
          useBias: cfg.useBias,
          name: cfg.name,
        });
      case "SeparableConv2D":
        return new SeparableConv2D(cfg.filters, {
          kernelSize: cfg.kernelSize,
          strides: cfg.strides,
          padding: cfg.padding,
          depthMultiplier: cfg.depthMultiplier,
          activation: cfg.activation,
          useBias: cfg.useBias,
          name: cfg.name,
        });
      case "MaxPooling2D":
        return new MaxPooling2D({
          poolSize: cfg.poolSize,
          strides: cfg.strides,
          padding: cfg.padding,
          name: cfg.name,
        });
      case "GlobalAveragePooling2D":
        return new GlobalAveragePooling2D({ name: cfg.name });
      case "ZeroPadding2D":
        return new ZeroPadding2D({ padding: cfg.padding, name: cfg.name });
      case "BatchNormalization":
        return new BatchNormalization({
          epsilon: cfg.epsilon,
          momentum: cfg.momentum,
          name: cfg.name,
        });
      default:
        throw new Error(
          `Sequential.loadModel: unknown layer type "${(cfg as any).type}"`,
        );
    }
  }

  private assertCompiled(caller: string): void {
    if (!this.compiled)
      throw new Error(`Sequential.${caller}(): call compile() first`);
  }
}
