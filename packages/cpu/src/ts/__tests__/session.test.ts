import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, session, ops, DType } from "../index.js";

describe("Session Execution", () => {
  it("should execute a simple addition synchronously", async () => {
    const g = graph();
    // Using high-level ops to build a simple a + b graph
    const a = g.addOp(
      "Placeholder",
      [],
      { dtype: { kind: "type", value: DType.FLOAT32 } },
      "a",
    )[0];
    const b = g.addOp(
      "Placeholder",
      [],
      { dtype: { kind: "type", value: DType.FLOAT32 } },
      "b",
    )[0];
    const res = ops.add(g, a, b, "res");

    const sess = session(g);

    const inputA = {
      dtype: DType.FLOAT32,
      shape: [1],
      data: Buffer.from(new Float32Array([10]).buffer),
    };
    const inputB = {
      dtype: DType.FLOAT32,
      shape: [1],
      data: Buffer.from(new Float32Array([5]).buffer),
    };

    const [output] = await sess.run(
      [
        [a, inputA],
        [b, inputB],
      ],
      [res],
    );

    assert.strictEqual(output.dtype, DType.FLOAT32);
    const val = new Float32Array(
      output.data.buffer,
      output.data.byteOffset,
      1,
    )[0];
    assert.strictEqual(val, 15);

    sess.destroy();
  });

  it("should execute non-blocking inference via runAsync", async () => {
    const g = graph();
    const x = g.addOp(
      "Placeholder",
      [],
      { dtype: { kind: "type", value: DType.FLOAT32 } },
      "x",
    )[0];
    const y = ops.square(g, x, "y");

    const sess = session(g);
    const inputX = {
      dtype: DType.FLOAT32,
      shape: [2],
      data: Buffer.from(new Float32Array([2, 4]).buffer),
    };

    const [output] = await sess.runAsync([[x, inputX]], [y]);

    const results = new Float32Array(
      output.data.buffer,
      output.data.byteOffset,
      2,
    );
    assert.strictEqual(results[0], 4);
    assert.strictEqual(results[1], 16);

    sess.destroy();
  });
});
