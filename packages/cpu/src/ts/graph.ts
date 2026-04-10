import type { Shape, Tensor } from "@isidorus/core";
import { DType, makeTensor, ShapeFromTF } from "@isidorus/core";

export type AttrValue =
  | { kind: "int"; value: number }
  | { kind: "float"; value: number }
  | { kind: "bool"; value: boolean }
  | { kind: "string"; value: string }
  | { kind: "type"; value: DType }
  | { kind: "shape"; value: number[] } // -1 = unknown dim
  | { kind: "list_type"; value: DType[] }
  | { kind: "list_int"; value: number[] }
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
   * @param type          TF op type string, e.g. "MatMul", "Placeholder"
   * @param inputs        Data inputs — output references from prior ops
   * @param attrs         Op attributes
   * @param name          Optional explicit op name (auto-generated if omitted)
   * @param controlInputs Op names that must complete before this op runs.
   *                      Used by globalVariablesInitializer to sequence init
   *                      ops before the NoOp target that callers wait on.
   * @returns             Array of output Tensors (one per op output)
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
      if (v.kind === "list_type")
        nativeAttrs[k] = { kind: "list_type", value: v.value.map(Number) };
      else if (v.kind === "type")
        nativeAttrs[k] = { kind: "type", value: Number(v.value) };
      else nativeAttrs[k] = v;
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

    // Side-effect-only ops (NoOp, AssignVariableOp, ResourceApply*) have
    // numOutputs === 0 so the tensors array is empty. Callers that use the
    // pattern `const [t] = g.addOp(...)` to get the op name would see t=undefined.
    // Provide a sentinel at index -1 carrying just the opName so those callers work:
    //   const [t] = g.addOp("NoOp", ...);  t.opName → the NoOp's name ✓
    if (numOutputs === 0) {
      tensors.push(makeTensor(opName, -1, null, null));
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

  /**
   * getOp — look up a Tensor descriptor by op name.
   *
   * Returns the Tensor at the specified output index, or null if the op
   * does not exist in the graph. Use this to reconnect to an op after
   * importGraphDef() or to build feeds/fetches by name rather than by
   * reference.
   *
   * @param name         Op name (e.g. "inputs", "conv1/relu")
   * @param outputIndex  Which output to return (default 0)
   */
  getOp(name: string, outputIndex = 0): Tensor | null {
    if (!this.hasOp(name)) return null;
    const tfDtype = this._native.opOutputType(name, outputIndex) as
      | number
      | null;
    const tfShape = this._native.opOutputShape(name, outputIndex) as
      | number[]
      | null;
    return makeTensor(
      name,
      outputIndex,
      tfDtype != null ? (tfDtype as DType) : null,
      tfShape != null ? ShapeFromTF(tfShape) : null,
    );
  }

  /** Total number of ops in the graph. */
  get numOps(): number {
    return this._native.numOps() as number;
  }

  /** Serialise the graph to a binary GraphDef proto. */
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

  /**
   * listOpsOfType — return the names of all ops whose type matches `type`.
   * Used to auto-discover Placeholder (input) op names in frozen graphs.
   */
  listOpsOfType(type: string): string[] {
    return this._native.listOpsOfType(type) as string[];
  }

  /**
   * listSinkOps — return op names whose outputs are not consumed by any
   * other op in the graph. These are the natural output ops of a frozen graph.
   */
  listSinkOps(): string[] {
    return this._native.listSinkOps() as string[];
  }
}
