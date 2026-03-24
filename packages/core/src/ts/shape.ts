/**
 * Shape - dimension sizes for a tensor.
 *
 * null means "unknown at graph construction time" (dynamic dim).
 * -1 is the TF wire representation of unknonwn: we use null in TypeScript
 * for clarity and convert at the N-API boundary.
 *
 * Examples:
 *      []                  scalar
 *      [4]                 1-D vector of length 4 elements
 *      [null, 784]         2-D matrix, batch size unknown, 784 columns
 *      [1, 224, 224, 3]    NHWC image batch of 1, 224x224 pixels, 3 channels
 */

export type Shape = (number | null)[];

/** Number of dimensions. */
export function rank(shape: Shape): number {
  return shape.length;
}

/**
 * Number of elements. Returns null if any dimensions is unknown.
 * Returns 1 for a scalar (rank 0).
 */
export function numElements(shape: Shape): number | null {
  if (shape.length === 0) return 1; // Scalar
  let n = 1;
  for (const d of shape) {
    if (d === null) return null; // Unknown dimension
    n *= d;
  }
  return n;
}

/**
 * Convert Shape to the TF C API representation:
 * null dims -> -1, known dims -> their value.
 */
export function shapeToTF(shape: Shape): number[] {
  return shape.map((d) => (d === null ? -1 : d));
}

/**
 * Convert TF wire representation back to Shape.
 */
export function ShapeFromTF(dims: number[]): Shape {
  return dims.map((d) => (d === -1 ? null : d));
}

/** Are two shapes compatible for element-wise operations? */
export function shapesCompatible(a: Shape, b: Shape): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== null && b[i] !== null && a[i] !== b[i]) return false;
  }
  return true;
}

export function shapeToString(shape: Shape): string {
  if (shape.length === 0) return "()";
  return `[${shape.map((d) => (d === null ? "?" : d)).join(", ")}]`;
}
