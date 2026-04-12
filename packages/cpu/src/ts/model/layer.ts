import type { Tensor } from "@isidorus/core";
import { DType } from "@isidorus/core";
import type { Graph } from "../graph.js";

// ---------------------------------------------------------------------------
// Layer — the interface all model layers must implement.
//
// Build protocol:
//   Sequential calls layer.build(g, input, inputShape).
//   The layer adds its ops to g and populates:
//     output       — symbolic tensor fed into the next layer
//     paramHandles — VarHandleOp tensors (for the optimizer)
//     paramReads   — ReadVariableOp tensors (inputs to addGradients)
//     paramDtypes  — matching dtypes
//     paramNames   — matching names for op-name prefixes
//     initOpNames  — op names to pass to globalVariablesInitializer
//
// Sequential wires everything together after all layers are built.
// ---------------------------------------------------------------------------

export type ActivationFn =
  | "relu"
  | "leaky_relu"
  | "relu6"
  | "sigmoid"
  | "tanh"
  | "softmax"
  | "log_softmax"
  | "elu"
  | "selu"
  | "swish"
  | "gelu"
  | "linear";

/** Weight value map used by buildFrozen — maps param name to its data. */
export type WeightMap = ReadonlyMap<
  string,
  {
    data: Buffer;
    shape: number[];
    dtype: DType;
  }
>;

export interface LayerParam {
  handle: Tensor; // VarHandleOp
  read: Tensor; // ReadVariableOp — used as x in addGradients
  dtype: DType;
  name: string;
  initOp: string; // AssignVariableOp name from variableWithInit
}

// ---------------------------------------------------------------------------
// LayerConfig — serialisable description of a layer's constructor arguments.
// Used by Sequential.saveModel() / loadModel() for pure-JS model I/O.
// ---------------------------------------------------------------------------

export type DenseConfig = {
  type: "Dense";
  name: string;
  units: number;
  activation: ActivationFn;
  useBias: boolean;
};

export type Conv2DConfig = {
  type: "Conv2D";
  name: string;
  filters: number;
  kernelSize: [number, number];
  strides: [number, number]; // [sH, sW] — user-facing, not NHWC [1,sH,sW,1]
  padding: "SAME" | "VALID";
  activation: ActivationFn;
  useBias: boolean;
};

export type FlattenConfig = {
  type: "Flatten";
  name: string;
};

export type DepthwiseConv2DConfig = {
  type: "DepthwiseConv2D";
  name: string;
  depthMultiplier: number;
  kernelSize: [number, number];
  strides: [number, number];
  padding: "SAME" | "VALID";
  activation: ActivationFn;
  useBias: boolean;
};

export type SeparableConv2DConfig = {
  type: "SeparableConv2D";
  name: string;
  filters: number;
  kernelSize: [number, number];
  strides: [number, number];
  padding: "SAME" | "VALID";
  depthMultiplier: number;
  activation: ActivationFn;
  useBias: boolean;
};

export type MaxPooling2DConfig = {
  type: "MaxPooling2D";
  name: string;
  poolSize: [number, number];
  strides: [number, number];
  padding: "SAME" | "VALID";
};

export type GlobalAveragePooling2DConfig = {
  type: "GlobalAveragePooling2D";
  name: string;
};

export type ZeroPadding2DConfig = {
  type: "ZeroPadding2D";
  name: string;
  padding: [[number, number], [number, number]]; // [[top,bot],[left,right]]
};

export type BatchNormalizationConfig = {
  type: "BatchNormalization";
  name: string;
  epsilon: number;
  momentum: number;
};

/** Discriminated union of all layer configurations. */
export type LayerConfig =
  | DenseConfig
  | Conv2DConfig
  | FlattenConfig
  | DepthwiseConv2DConfig
  | SeparableConv2DConfig
  | MaxPooling2DConfig
  | GlobalAveragePooling2DConfig
  | ZeroPadding2DConfig
  | BatchNormalizationConfig;

export interface Layer {
  readonly name: string;

  /**
   * Build — add ops to the training graph.
   * Called once by Sequential in order.
   * @returns output shape after this layer (null dims = dynamic)
   */
  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[];

  /**
   * buildFrozen — add ops to a frozen inference graph.
   *
   * Identical computation to build() but uses Const tensors from the
   * provided weight map instead of ResourceVariable ops. Called by
   * Sequential.exportFrozen() to produce a .pb file compatible with
   * InferencePool without any Python dependency.
   *
   * @param g          The new (empty) inference graph
   * @param input      Input tensor from the previous frozen layer
   * @param inputShape Shape of the input tensor
   * @param weights    Map of param name → {data, shape, dtype}
   * @returns          {tensor: output tensor, shape: output shape}
   */
  buildFrozen(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
    weights: WeightMap,
  ): { tensor: Tensor; shape: (number | null)[] };

  /** Symbolic output tensor. Set during build(). */
  readonly output: Tensor;

  /** Parameters exposed to the optimizer. Set during build(). */
  readonly layerParams: LayerParam[];

  /** Return a serialisable description of this layer's configuration. */
  getConfig(): LayerConfig;

  /**
   * Optional: return op names that must run as targets on every training step.
   * Used by BatchNormalization to update moving_mean and moving_var via EMA.
   * Sequential.compile() collects these from all layers into _extraUpdateOps.
   */
  updateOps?(): string[];

  /**
   * Optional: return non-trainable parameters (e.g. BN moving statistics)
   * that should be saved/restored by saveWeights/loadWeights but are excluded
   * from model.params and therefore have no gradients.
   */
  nonTrainableParams?(): LayerParam[];

  /**
   * Optional: return the number of trainable parameters in this layer.
   * Used by Model.countParams() for parameter counting.
   * If not implemented, defaults to 0.
   */
  paramCount?(): number;
}
