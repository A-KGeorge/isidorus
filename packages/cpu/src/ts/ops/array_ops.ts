import type { Tensor, Shape } from "@isidorus/core";
import { DType, shapeToTF } from "@isidorus/core";
import type { Graph } from "../graph.js";

/**
 * placeholder — feed input into the graph at session run time.
 * Shape may contain null for dynamic (unknown-at-build-time) dimensions.
 */
export function placeholder(
  g: Graph,
  name: string,
  shape: Shape,
  dtype: DType = DType.FLOAT32,
): Tensor {
  const [t] = g.addOp(
    "Placeholder",
    [],
    {
      dtype: { kind: "type", value: dtype },
      shape: { kind: "shape", value: shapeToTF(shape) },
    },
    name,
  );
  return t;
}

/**
 * constant — embed a fixed value as a Const op in the graph.
 * `data` should be a Buffer containing the raw little-endian bytes.
 */
export function constant(
  g: Graph,
  data: Buffer,
  shape: number[],
  dtype: DType,
  name?: string,
): Tensor {
  const [t] = g.addOp(
    "Const",
    [],
    {
      dtype: { kind: "type", value: dtype },
      value: { kind: "tensor", value: { dtype, shape, data } },
    },
    name,
  );
  return t;
}

/** Convenience: scalar float32 constant. */
export function scalar(g: Graph, value: number, name?: string): Tensor {
  const buf = Buffer.allocUnsafe(4);
  buf.writeFloatLE(value, 0);
  return constant(g, buf, [], DType.FLOAT32, name);
}

/**
 * reshape — reshape a tensor to a new shape.
 * `newShape` must be a 1-D int32 constant already in the graph,
 * or use reshapeTo() for the convenience form.
 */
export function reshape(
  g: Graph,
  x: Tensor,
  shape: Tensor,
  name?: string,
): Tensor {
  const [t] = g.addOp("Reshape", [x, shape], {}, name);
  return t;
}

/**
 * reshapeTo — convenience wrapper that creates the shape constant inline.
 * @param newShape  Array of dimension sizes (-1 = infer)
 */
export function reshapeTo(
  g: Graph,
  x: Tensor,
  newShape: number[],
  name?: string,
): Tensor {
  const shapeBuf = Buffer.allocUnsafe(newShape.length * 4);
  newShape.forEach((d, i) => shapeBuf.writeInt32LE(d, i * 4));
  const shapeConst = constant(g, shapeBuf, [newShape.length], DType.INT32);
  return reshape(g, x, shapeConst, name);
}

/** identity — pass-through (useful for naming intermediate tensors). */
export function identity(g: Graph, x: Tensor, name?: string): Tensor {
  const [t] = g.addOp("Identity", [x], {}, name);
  return t;
}
