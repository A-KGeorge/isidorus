import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, session, ops, DType } from "../index.js";

describe("Math Broadcasting", () => {
  it("should broadcast a scalar to a 1D tensor", async () => {
    const g = graph();
    const a = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
      shape: { kind: "shape", value: [3] },
    })[0];

    const b = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
      shape: { kind: "shape", value: [] }, // Scalar
    })[0];

    const res = ops.add(g, a, b);
    const sess = session(g);

    const inputA = {
      dtype: DType.FLOAT32,
      shape: [3],
      data: Buffer.from(new Float32Array([1, 2, 3]).buffer),
    };
    const inputB = {
      dtype: DType.FLOAT32,
      shape: [],
      data: Buffer.from(new Float32Array([10]).buffer),
    };

    const [output] = await sess.run(
      [
        [a, inputA],
        [b, inputB],
      ],
      [res],
    );
    const vals = new Float32Array(
      output.data.buffer,
      output.data.byteOffset,
      3,
    );

    assert.deepStrictEqual(Array.from(vals), [11, 12, 13]);
    sess.destroy();
  });

  it("should support complex operation chaining (Wx + b)", async () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    const w = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    const b = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    // y = matmul(x, w) + b
    const xw = ops.matmul(g, x, w);
    const y = ops.add(g, xw, b);

    assert.strictEqual(g.numOps, 5); // 3 placeholders + 1 matmul + 1 add
    assert.ok(g.hasOp(y.opName));
  });
});
