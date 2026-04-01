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

export interface LayerParam {
  handle: Tensor; // VarHandleOp
  read: Tensor; // ReadVariableOp — used as x in addGradients
  dtype: DType;
  name: string;
  initOp: string; // AssignVariableOp name from variableWithInit
}

export interface Layer {
  readonly name: string;

  /**
   * Build — add ops to the graph.
   * Called once by Sequential in order.
   * @returns output shape after this layer (null dims = dynamic)
   */
  build(
    g: Graph,
    input: Tensor,
    inputShape: (number | null)[],
  ): (number | null)[];

  /** Symbolic output tensor. Set during build(). */
  readonly output: Tensor;

  /** Parameters exposed to the optimizer. Set during build(). */
  readonly layerParams: LayerParam[];
}
