import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, session, DType } from "../index.js";

describe("Validation and Errors", () => {
  it("should throw when running a destroyed session", async () => {
    const g = graph();
    const sess = session(g);
    sess.destroy();

    await assert.rejects(async () => await sess.run([], []), {
      message: /Session destroyed/,
    });
  });

  it("should throw error for non-existent feed ops", async () => {
    const g = graph();
    const sess = session(g);

    // Create a fake tensor that doesn't exist in this graph's native layer
    const fakeTensor = {
      opName: "invalid_op",
      index: 0,
      dtype: DType.FLOAT32,
      shape: null,
    };
    const dummyFeed = {
      dtype: DType.FLOAT32,
      shape: [1],
      data: Buffer.alloc(4),
    };

    await assert.rejects(
      async () => await sess.run([[fakeTensor as any, dummyFeed]], []),
      { message: /Feed op not found/ },
    );

    sess.destroy();
  });

  it("should throw if addOp is called with missing inputs", () => {
    const g = graph();
    // MatMul requires 2 inputs
    assert.throws(() => g.addOp("MatMul", []), /TF_FinishOperation failed/);
  });
});
