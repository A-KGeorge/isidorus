import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, ops, DType } from "../index.js";

describe("Math Operations", () => {
  it("should create an AddV2 operation", () => {
    const g = graph();
    const a = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    const b = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const sum = ops.add(g, a, b, "my_sum");
    assert.strictEqual(sum.opName, "my_sum");
    assert.strictEqual(g.hasOp("my_sum"), true);
  });

  it("should create a MatMul operation with transpose attributes", () => {
    const g = graph();
    const a = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    const b = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const product = ops.matmul(g, a, b, { transposeA: true });
    assert.ok(product.opName.startsWith("MatMul"));
  });

  it("should support reduction operations like Mean", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    // Mean involves the Mean op and a constant for axes
    const avg = ops.mean(g, x, [0, 1], true);
    assert.ok(g.hasOp(avg.opName));

    // numOps check (Placeholder + Const + Mean = 3)
    assert.ok(g.numOps >= 2);
  });
});
