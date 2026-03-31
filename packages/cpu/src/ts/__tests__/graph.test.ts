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
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const tensors = g.addOp("IdentityN", [x], {
      T: { kind: "list_type", value: [DType.FLOAT32] },
    });

    assert.strictEqual(tensors.length, 1);
  });
});

describe("Graph.getOp", () => {
  it("returns null for a non-existent op", () => {
    const g = graph();
    assert.strictEqual(g.getOp("does_not_exist"), null);
  });

  it("returns a Tensor descriptor for an existing op", () => {
    const g = graph();
    g.addOp(
      "Placeholder",
      [],
      {
        dtype: { kind: "type", value: DType.FLOAT32 },
        shape: { kind: "shape", value: [1, 784] },
      },
      "my_input",
    );

    const t = g.getOp("my_input");
    assert.ok(t !== null);
    assert.strictEqual(t!.opName, "my_input");
    assert.strictEqual(t!.index, 0);
    assert.strictEqual(t!.dtype, DType.FLOAT32);
  });

  it("respects the output index parameter", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    // IdentityN has one output per input — output index 0 is valid
    g.addOp("IdentityN", [x], {
      T: { kind: "list_type", value: [DType.FLOAT32] },
    });

    // Index 0 should resolve, but we can verify the tensor id is correct
    const t = g.getOp(x.opName, 0);
    assert.ok(t !== null);
    assert.strictEqual(t!.index, 0);
  });

  it("can be used to build Session feeds/fetches after addOp", () => {
    const g = graph();
    g.addOp(
      "Placeholder",
      [],
      {
        dtype: { kind: "type", value: DType.FLOAT32 },
        shape: { kind: "shape", value: [4] },
      },
      "x",
    );

    const x = g.getOp("x");
    assert.ok(x !== null);

    // Verify properties needed by Session.runAsync
    assert.strictEqual(typeof x!.opName, "string");
    assert.strictEqual(typeof x!.index, "number");
  });
});

describe("Graph.importGraphDef", () => {
  it("imports a round-tripped GraphDef and restores all ops", () => {
    // Build a small graph and export it
    const g1 = graph();
    const x = g1.addOp(
      "Placeholder",
      [],
      { dtype: { kind: "type", value: DType.FLOAT32 } },
      "x",
    )[0];
    g1.addOp("Square", [x], {}, "x_squared");

    const graphDef = g1.toGraphDef();
    assert.ok(graphDef.length > 0);

    // Import into a fresh graph
    const g2 = graph();
    assert.strictEqual(g2.numOps, 0);
    g2.importGraphDef(graphDef);

    // All ops from g1 should now exist in g2
    assert.strictEqual(g2.hasOp("x"), true);
    assert.strictEqual(g2.hasOp("x_squared"), true);
    assert.ok(g2.numOps >= 2);
  });

  it("imported ops are resolvable via getOp", () => {
    const g1 = graph();
    g1.addOp(
      "Placeholder",
      [],
      {
        dtype: { kind: "type", value: DType.FLOAT32 },
        shape: { kind: "shape", value: [1, 4] },
      },
      "input",
    );
    const graphDef = g1.toGraphDef();

    const g2 = graph();
    g2.importGraphDef(graphDef);

    const t = g2.getOp("input");
    assert.ok(t !== null);
    assert.strictEqual(t!.opName, "input");
    assert.strictEqual(t!.dtype, DType.FLOAT32);
  });

  it("throws if given invalid bytes", () => {
    const g = graph();
    // Random garbage bytes — not a valid GraphDef proto
    assert.throws(() => g.importGraphDef(Buffer.from([0xff, 0xfe, 0xfd])));
  });
});
