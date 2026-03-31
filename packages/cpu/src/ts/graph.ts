import type { Tensor } from "@isidorus/core";
import { DType, makeTensor, ShapeFromTF } from "@isidorus/core";

// Attribute value union — mirrors the native AttrValue handling in graph.cc
export type AttrValue =
  | { kind: "int"; value: number }
  | { kind: "float"; value: number }
  | { kind: "bool"; value: boolean }
  | { kind: "type"; value: DType }
  | { kind: "shape"; value: number[] } // -1 = unknown dim
  | { kind: "list_type"; value: DType[] }
  | { kind: "list_int"; value: number[] }
  | { kind: "string"; value: string }
  | { kind: "tensor"; value: InlineTensor };

export interface InlineTensor {
  dtype: DType;
  shape: number[];
  data: Buffer;
}

export type TFOutput = { opName: string; index: number };

/**
 * Graph — builds a TensorFlow graph via the C API.
 *
 * This is the ground-level primitive. Every operation (placeholder, variable,
 * matmul, relu, …) adds one or more ops to this graph. Symbolic Tensors
 * reference op outputs but hold no data until a Session executes the graph.
 *
 * @example
 * const g = new Graph();
 * const x = g.placeholder("x", [null, 784], DType.FLOAT32);
 * const w = g.variable("w", [784, 128], DType.FLOAT32);
 * const y = g.matmul(x, g.readVariable(w));
 */
export class Graph {
  /** @internal */
  readonly _native: any;

  constructor(native: any) {
    this._native = native;
  }

  /**
   * Add a raw op to the graph.
   *
   * @param type      TF op type string, e.g. "MatMul", "Placeholder"
   * @param inputs    Output references from prior ops
   * @param attrs     Op attributes
   * @param name      Optional explicit op name (auto-generated if omitted)
   * @param controlInputs Op names that must complete before this op runs.
   *                      Used by globalVariablesInitializer to sequence init
   *                      ops before the NoOp target that callers wait on.
   * @returns         Array of output Tensors (one per op output)
   */
  addOp(
    type: string,
    inputs: TFOutput[],
    attrs: Record<string, AttrValue> = {},
    name?: string,
    controlInputs: string[] = [],
  ): Tensor[] {
    const nativeAttrs: Record<string, any> = {};
    for (const [k, v] of Object.entries(attrs)) {
      if (v.kind === "list_type") {
        nativeAttrs[k] = { kind: "list_type", value: v.value.map(Number) };
      } else if (v.kind === "type") {
        nativeAttrs[k] = { kind: "type", value: Number(v.value) };
      } else {
        nativeAttrs[k] = v;
      }
    }

    const result = this._native.addOp(
      type,
      inputs,
      nativeAttrs,
      name,
      controlInputs,
    );
    const { opName, numOutputs } = result as {
      opName: string;
      numOutputs: number;
    };

    const tensors: Tensor[] = [];
    for (let i = 0; i < numOutputs; i++) {
      const tfDtype = this._native.opOutputType(opName, i) as number | null;
      const tfShape = this._native.opOutputShape(opName, i) as number[] | null;
      tensors.push(
        makeTensor(
          opName,
          i,
          tfDtype != null ? (tfDtype as DType) : null,
          tfShape != null ? ShapeFromTF(tfShape) : null,
        ),
      );
    }
    return tensors;
  }

  /**
   * addGradients — compute symbolic gradients via TF_AddGradients.
   *
   * Injects gradient ops into the graph.  Returns one gradient Tensor per
   * entry in `x` — the partial derivative dSum(y)/dx_i.
   *
   * @param y   Loss outputs to differentiate (typically a scalar loss tensor)
   * @param x   Parameters to differentiate with respect to
   * @param dx  Initial upstream gradients (default: ones, i.e. dL/dy = 1)
   *
   * @example
   * const loss = ops.sparseSoftmaxCrossEntropyWithLogits(g, labels, logits);
   * const wVal = ops.readVariable(g, wHandle, DType.FLOAT32);
   * const [dw] = g.addGradients([loss], [wVal]);
   * // dw is now a Tensor representing dLoss/dW
   * // pass it to applyGradientDescent or applyAdam
   */
  addGradients(y: TFOutput[], x: TFOutput[], dx?: TFOutput[]): Tensor[] {
    const raw = this._native.addGradients(y, x, dx ?? null) as TFOutput[];
    return raw.map(({ opName, index }) => {
      const tfDtype = this._native.opOutputType(opName, index) as number | null;
      const tfShape = this._native.opOutputShape(opName, index) as
        | number[]
        | null;
      return makeTensor(
        opName,
        index,
        tfDtype != null ? (tfDtype as DType) : null,
        tfShape != null ? ShapeFromTF(tfShape) : null,
      );
    });
  }

  /** Whether an op with the given name exists in this graph. */
  hasOp(name: string): boolean {
    return this._native.hasOp(name) as boolean;
  }

  /** Total number of ops in the graph. */
  get numOps(): number {
    return this._native.numOps() as number;
  }

  /**
   * Serialise the graph to a binary GraphDef proto.
   * Useful for saving, inspecting with Netron, or freezing.
   */
  toGraphDef(): Buffer {
    return this._native.toGraphDef() as Buffer;
  }

  /**
   * importGraphDef — deserialise a binary GraphDef proto into this graph.
   *
   * Used to load a frozen .pb model so it can be executed via the native
   * Session (which applies ConfigProto thread config and CPU affinity).
   * The graph must be empty before calling this.
   *
   * @param buffer  Raw bytes of a frozen GraphDef proto (.pb file contents)
   *
   * @example
   * import { readFileSync } from "fs";
   * const g = graph();
   * g.importGraphDef(readFileSync("model.pb"));
   * const sess = session(g, { strategy: "tf-parallel", reserveCores: 2 });
   */
  importGraphDef(buffer: Buffer): void {
    this._native.importGraphDef(buffer);
  }

  // graph.ts
  getOp(name: string, index = 0): Tensor | null {
    if (!this._native.hasOp(name)) return null;
    const tfDtype = this._native.opOutputType(name, index) as number | null;
    const tfShape = this._native.opOutputShape(name, index) as number[] | null;
    return makeTensor(
      name,
      index,
      tfDtype != null ? (tfDtype as DType) : null,
      tfShape != null ? ShapeFromTF(tfShape) : null,
    );
  }
}
