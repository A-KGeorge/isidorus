import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, DType } from "../index.js";

describe("Graph Construction", () => {
  it("should initialize an empty graph", () => {
    const g = graph();
    assert.strictEqual(g.numOps, 0);
  });

  it("should add a raw operation and track its presence", () => {
    const g = graph();
    // Manually adding a placeholder op via the low-level addOp
    g.addOp(
      "Placeholder",
      [],
      {
        dtype: { kind: "type", value: DType.FLOAT32 },
        shape: { kind: "shape", value: [1, 10] },
      },
      "input_0",
    );

    assert.strictEqual(g.numOps, 1);
    assert.strictEqual(g.hasOp("input_0"), true);
  });

  it("should generate a GraphDef buffer", () => {
    const g = graph();
    g.addOp("NoOp", [], {}, "noop");
    const def = g.toGraphDef();
    assert.ok(Buffer.isBuffer(def));
    assert.ok(def.length > 0);
  });

  it("should correctly map attribute types in addOp", () => {
    const g = graph();
    // IdentityN requires at least one input if T has one element.
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const tensors = g.addOp("IdentityN", [x], {
      T: { kind: "list_type", value: [DType.FLOAT32] },
    });

    assert.strictEqual(tensors.length, 1);
  });
});
