import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, session, ops, DType } from "../index.js";

// ---------------------------------------------------------------------------
// addGradients
//
// These tests verify:
//   1. addGradients adds gradient ops without throwing
//   2. Gradient tensors are present and runnable in the same session
//   3. Gradient values are numerically correct for simple differentiable ops
//   4. Correct number of gradient outputs (one per x entry)
//   5. Custom dx (upstream gradient scaling) works correctly
//   6. Non-differentiable ops produce a clear error
// ---------------------------------------------------------------------------

describe("addGradients", () => {
  it("returns one gradient tensor per x entry", () => {
    const g = graph();
    const x = ops.placeholder(g, "x", [3], DType.FLOAT32);
    const y = ops.sum(g, x, [0]); // scalar

    const [dx] = g.addGradients([y], [x]);

    assert.ok(dx !== undefined, "gradient tensor should be defined");
    assert.ok(g.hasOp(dx.opName), "gradient op should exist in graph");
  });

  it("dy/dx for y = sum(x) is all-ones", async () => {
    // y = sum(x)  →  dy/dx_i = 1 for all i
    const g = graph();
    const x = ops.placeholder(g, "x", [4], DType.FLOAT32);
    const y = ops.sum(g, x, [0]);

    const [dx] = g.addGradients([y], [x]);

    const sess = session(g);
    const inputX = {
      dtype: DType.FLOAT32,
      shape: [4],
      data: Buffer.from(new Float32Array([1, 2, 3, 4]).buffer),
    };

    const [gradOut] = await sess.run([[x, inputX]], [dx]);
    const vals = new Float32Array(
      gradOut.data.buffer,
      gradOut.data.byteOffset,
      4,
    );

    assert.deepStrictEqual(Array.from(vals), [1, 1, 1, 1]);
    sess.destroy();
  });

  it("dy/dx for y = 2*x is 2 everywhere", async () => {
    // y = 2x  →  dy/dx_i = 2
    const g = graph();
    const x = ops.placeholder(g, "x", [3], DType.FLOAT32);
    const twoBuf = Buffer.allocUnsafe(4);
    twoBuf.writeFloatLE(2.0, 0);
    const two = ops.constant(g, twoBuf, [], DType.FLOAT32);
    const y = ops.mul(g, x, two);
    const ySum = ops.sum(g, y, [0]); // reduce to scalar for addGradients

    const [dx] = g.addGradients([ySum], [x]);

    const sess = session(g);
    const inputX = {
      dtype: DType.FLOAT32,
      shape: [3],
      data: Buffer.from(new Float32Array([1, 2, 3]).buffer),
    };

    const [gradOut] = await sess.run([[x, inputX]], [dx]);
    const vals = new Float32Array(
      gradOut.data.buffer,
      gradOut.data.byteOffset,
      3,
    );

    assert.deepStrictEqual(Array.from(vals), [2, 2, 2]);
    sess.destroy();
  });

  it("gradient of matmul: dL/dW = x^T for L = sum(W*x)", async () => {
    // L = sum(W @ x),  dL/dW_ij = x_j
    // W: [1, 3], x: [3, 1]  →  W@x: [1, 1]  →  L: scalar
    // dL/dW = x^T = [x0, x1, x2]
    const g = graph();
    const W = ops.placeholder(g, "W", [1, 3], DType.FLOAT32);
    const x = ops.placeholder(g, "x", [3, 1], DType.FLOAT32);
    const Wx = ops.matmul(g, W, x);
    const L = ops.sum(g, Wx, [0, 1]);

    const [dW] = g.addGradients([L], [W]);

    const sess = session(g);
    const [gradOut] = await sess.run(
      [
        [
          W,
          {
            dtype: DType.FLOAT32,
            shape: [1, 3],
            data: Buffer.from(new Float32Array([1, 0, 0]).buffer),
          },
        ],
        [
          x,
          {
            dtype: DType.FLOAT32,
            shape: [3, 1],
            data: Buffer.from(new Float32Array([2, 3, 4]).buffer),
          },
        ],
      ],
      [dW],
    );

    // dL/dW should be x^T = [[2, 3, 4]]
    const vals = new Float32Array(
      gradOut.data.buffer,
      gradOut.data.byteOffset,
      3,
    );
    assert.deepStrictEqual(Array.from(vals), [2, 3, 4]);
    sess.destroy();
  });

  it("multiple x entries produce correct number of gradients", () => {
    const g = graph();
    const a = ops.placeholder(g, "a", [2], DType.FLOAT32);
    const b = ops.placeholder(g, "b", [2], DType.FLOAT32);
    const y = ops.sum(g, ops.add(g, a, b), [0]);

    const grads = g.addGradients([y], [a, b]);

    assert.strictEqual(grads.length, 2, "one gradient per x");
    assert.ok(g.hasOp(grads[0].opName));
    assert.ok(g.hasOp(grads[1].opName));
  });

  it("custom dx scales the gradient", async () => {
    // y = sum(x),  dx_upstream = 3.0  →  dy/dx = 3.0 (not 1.0)
    const g = graph();
    const x = ops.placeholder(g, "x", [2], DType.FLOAT32);
    const y = ops.sum(g, x, [0]);

    // dx: a placeholder we'll feed as 3.0
    const dxPlaceholder = ops.placeholder(g, "dy", [], DType.FLOAT32);

    const [dx] = g.addGradients([y], [x], [dxPlaceholder]);

    const sess = session(g);
    const [gradOut] = await sess.run(
      [
        [
          x,
          {
            dtype: DType.FLOAT32,
            shape: [2],
            data: Buffer.from(new Float32Array([1, 1]).buffer),
          },
        ],
        [
          dxPlaceholder,
          {
            dtype: DType.FLOAT32,
            shape: [],
            data: Buffer.from(new Float32Array([3]).buffer),
          },
        ],
      ],
      [dx],
    );

    const vals = new Float32Array(
      gradOut.data.buffer,
      gradOut.data.byteOffset,
      2,
    );
    assert.deepStrictEqual(Array.from(vals), [3, 3]);
    sess.destroy();
  });

  it("mismatched dx length throws a clear error", () => {
    const g = graph();
    const x = ops.placeholder(g, "x", [2], DType.FLOAT32);
    const y = ops.sum(g, x, [0]);
    const fakeGrad = ops.placeholder(g, "g1", [], DType.FLOAT32);
    const fakeGrad2 = ops.placeholder(g, "g2", [], DType.FLOAT32);

    // y has 1 output but we provide 2 dx entries
    assert.throws(
      () => g.addGradients([y], [x], [fakeGrad, fakeGrad2]),
      /dx length must equal y length/,
    );
  });

  it("non-differentiable op throws at graph construction", () => {
    const g = graph();
    const x = ops.placeholder(g, "x", [3], DType.FLOAT32);
    const idx = ops.argMax(g, x, 0); // ArgMax has no gradient

    assert.throws(() => g.addGradients([idx], [x]), /TF_AddGradients failed/);
  });
});
