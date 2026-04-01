import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";
import type { Session } from "../session.js";
import type { Layer } from "./layer.js";
import type { ParamSpec, FeedEntry } from "../optimizers/sgd.js";
import { placeholder } from "../ops/array_ops.js";
import { mean } from "../ops/math_ops.js";
import { globalVariablesInitializer } from "../ops/variable_ops.js";
import { sigmoidCrossEntropyWithLogits } from "../ops/nn_ops.js";

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
  private _outputTensor!: Tensor;
  private _lossTensor!: Tensor;
  private _allParams!: ParamSpec[];
  private _allInitOp!: string;
  private _labelDtype!: DType;
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
    this._allParams = allLayerParams.map((p, i) => ({
      handle: p.handle,
      grad: grads[i],
      dtype: p.dtype,
      name: p.name,
    }));

    // ── Global variable init op ───────────────────────────────────────────
    const initOpNames = allLayerParams.map((p) => p.initOp);
    this._allInitOp = globalVariablesInitializer(
      this.g,
      initOpNames,
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

    // Forward pass + fetch loss.
    const [lossOut] = await sess.run(feeds, [this._lossTensor], []);

    // Backward pass + weight update.
    await opt.applyGradients(sess, feeds);

    const lossVal = new Float32Array(
      lossOut.data.buffer,
      lossOut.data.byteOffset,
      1,
    )[0];
    return { loss: lossVal };
  }

  /**
   * predict — forward pass only, no gradient computation or update.
   */
  async predict(
    sess: Session,
    xBuf: Buffer,
    xShape: number[],
  ): Promise<{ data: Buffer; shape: number[]; dtype: DType }> {
    this.assertCompiled("predict");
    const [out] = await sess.run(
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

  private assertCompiled(caller: string): void {
    if (!this.compiled)
      throw new Error(`Sequential.${caller}(): call compile() first`);
  }
}
