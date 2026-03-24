import type { DType } from "./dtype.js";
import type { Shape } from "./shape.js";

/**
 * Tensor - a symbolic reference to one output of a graph op.
 *
 * A Tensor does not hold data. It is an address: "output index 'index'
 * of the op named 'opName' in the graph 'graphId'". data flows through it
 * only when a Session executes the graph.
 *
 * This mirrors TF_Output in the C API: { TF_Operation* oper, int index }.
 *
 * The native layer (@isidorus/cpu) stores TF_Output as { opName, index }
 * and resolves back to TF_Operation* via TF_GraphOperationByName at
 * Session.run() time.
 */
export interface Tensor {
  /** Unique op name within the graph, e.g. "MatMul_1", "Variable/read" */
  readonly opName: string;

  /** Output index on that op (0 for ops with a single output). */
  readonly index: number;

  /** Element type. nuyll = unknown until graph analysis */
  readonly dtype: DType | null;

  /**
   * Shape. May contain null dims for dynamic axes.
   * null = shape entirely unknown until graph analysis;
   */
  readonly shape: Shape | null;

  /**
   * Human-readable name assigned by the user or auto-generated.
   * Used for debugging and feed-dict keys in user-facing APIs
   */
  readonly name: string;
}

/** Construct a Tensor descriptor. */
export function makeTensor(
  opName: string,
  index: number,
  dtype: DType | null,
  shape: Shape | null,
  name?: string,
): Tensor {
  return {
    opName,
    index,
    dtype,
    shape,
    name: name ?? (index === 0 ? opName : `${opName}:${index}`),
  };
}

/**
 * TF wire format for a tensor output: "op_name:index",
 * Used as feed-dict keys and fetch specs in TF_SessionRun.
 */
export function tensorId(t: Tensor): string {
  return `${t.opName}:${t.index}`;
}
