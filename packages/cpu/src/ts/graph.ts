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
   * @returns         Array of output Tensors (one per op output)
   */
  addOp(
    type: string,
    inputs: TFOutput[],
    attrs: Record<string, AttrValue> = {},
    name?: string,
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

    const result = this._native.addOp(type, inputs, nativeAttrs, name);
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

  /** Whether an op with the given name exists in this graph. */
  hasOp(name: string): boolean {
    return this._native.hasOp(name) as boolean;
  }

  /** Total number of ops in the graph. */
  get numOps(): number {
    return this._native.numOps as number;
  }

  /**
   * Serialise the graph to a binary GraphDef proto.
   * Useful for saving, inspecting with Netron, or freezing.
   */
  toGraphDef(): Buffer {
    return this._native.toGraphDef() as Buffer;
  }
}
