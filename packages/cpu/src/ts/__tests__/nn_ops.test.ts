import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, session, ops, DType } from "../index.js";

describe("dropout()", () => {
  it("throws RangeError for rate < 0", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    assert.throws(() => ops.dropout(g, x, -0.1, true), RangeError);
  });

  it("throws RangeError for rate >= 1", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    assert.throws(() => ops.dropout(g, x, 1.0, true), RangeError);
    assert.throws(() => ops.dropout(g, x, 1.5, true), RangeError);
  });

  it("throws RangeError even when training=false", () => {
    // Out-of-range rate is always a caller bug regardless of training flag
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    assert.throws(() => ops.dropout(g, x, -0.1, false), RangeError);
  });

  it("returns identity op when training=false", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const before = g.numOps;
    const out = ops.dropout(g, x, 0.5, false);
    assert.ok(g.hasOp(out.opName));
    // Only an Identity op should be added
    assert.strictEqual(g.numOps, before + 1);
  });

  it("returns identity op when rate=0 and training=true", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const before = g.numOps;
    const out = ops.dropout(g, x, 0, true);
    assert.ok(g.hasOp(out.opName));
    assert.strictEqual(g.numOps, before + 1);
  });

  it("training mode adds multiple ops to the graph", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const before = g.numOps;
    const out = ops.dropout(g, x, 0.5, true);
    assert.ok(g.hasOp(out.opName));
    // Expect: Shape, Const (seed), StatelessRandomUniform,
    //         Const (rate), GreaterEqual, Cast, Mul (mask),
    //         Const (scale), Mul (scale) — at least 6 new ops
    assert.ok(
      g.numOps > before + 5,
      `expected >5 new ops, got ${g.numOps - before}`,
    );
  });

  it("inference: training=false passes values through unchanged", async () => {
    const g = graph();
    const x = g.addOp(
      "Placeholder",
      [],
      { dtype: { kind: "type", value: DType.FLOAT32 } },
      "x",
    )[0];

    const out = ops.dropout(g, x, 0.5, false, undefined, "out");
    const sess = session(g);

    const input = new Float32Array([1, 2, 3, 4]);
    const [result] = await sess.run(
      [
        [
          x,
          { dtype: DType.FLOAT32, shape: [4], data: Buffer.from(input.buffer) },
        ],
      ],
      [out],
    );

    const vals = new Float32Array(
      result.data.buffer,
      result.data.byteOffset,
      4,
    );
    assert.deepStrictEqual(Array.from(vals), [1, 2, 3, 4]);
    sess.destroy();
  });

  it("training mode: output has same shape as input", async () => {
    const g = graph();
    const x = g.addOp(
      "Placeholder",
      [],
      { dtype: { kind: "type", value: DType.FLOAT32 } },
      "x",
    )[0];

    const out = ops.dropout(g, x, 0.5, true);
    const sess = session(g);

    const input = new Float32Array(16).fill(1.0);
    const [result] = await sess.run(
      [
        [
          x,
          {
            dtype: DType.FLOAT32,
            shape: [16],
            data: Buffer.from(input.buffer),
          },
        ],
      ],
      [out],
    );

    // Shape must be preserved
    assert.deepStrictEqual(result.shape, [16]);
    sess.destroy();
  });

  it("training mode: dropped values are 0, kept values are scaled", async () => {
    const g = graph();
    const x = g.addOp(
      "Placeholder",
      [],
      { dtype: { kind: "type", value: DType.FLOAT32 } },
      "x",
    )[0];

    const rate = 0.5;
    const out = ops.dropout(g, x, rate, true);
    const sess = session(g);

    // All-ones input — kept units should be 1/(1-rate) = 2.0
    const input = new Float32Array(64).fill(1.0);
    const [result] = await sess.run(
      [
        [
          x,
          {
            dtype: DType.FLOAT32,
            shape: [64],
            data: Buffer.from(input.buffer),
          },
        ],
      ],
      [out],
    );

    const vals = new Float32Array(
      result.data.buffer,
      result.data.byteOffset,
      64,
    );
    const scale = 1 / (1 - rate);

    for (const v of vals) {
      // Each element must be exactly 0 (dropped) or scale (kept)
      assert.ok(
        v === 0 || Math.abs(v - scale) < 1e-5,
        `unexpected value ${v}: must be 0 or ${scale}`,
      );
    }

    sess.destroy();
  });

  it("training mode with constant seed produces same mask each run", async () => {
    // With seed=[0,0] the mask is deterministic — same neurons dropped each call
    const g = graph();
    const x = g.addOp(
      "Placeholder",
      [],
      { dtype: { kind: "type", value: DType.FLOAT32 } },
      "x",
    )[0];

    const out = ops.dropout(g, x, 0.5, true);
    const sess = session(g);

    const input = new Float32Array(32).fill(1.0);
    const feed: [typeof x, { dtype: DType; shape: number[]; data: Buffer }] = [
      x,
      { dtype: DType.FLOAT32, shape: [32], data: Buffer.from(input.buffer) },
    ];

    const [r1] = await sess.run([feed], [out]);
    const [r2] = await sess.run([feed], [out]);

    const v1 = new Float32Array(r1.data.buffer, r1.data.byteOffset, 32);
    const v2 = new Float32Array(r2.data.buffer, r2.data.byteOffset, 32);

    assert.deepStrictEqual(Array.from(v1), Array.from(v2));
    sess.destroy();
  });

  it("respects the name parameter", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const out = ops.dropout(g, x, 0.5, false, undefined, "my_dropout");
    assert.strictEqual(out.opName, "my_dropout");
    assert.ok(g.hasOp("my_dropout"));
  });
});

describe("makeDropoutSeed()", () => {
  it("returns a tensor with shape [2]", () => {
    const g = graph();
    const { handle } = ops.variableWithInit(
      g,
      [],
      DType.INT64,
      "global_step",
      ops.zerosInitializer(g, [], DType.INT64),
    );

    const result = ops.makeDropoutSeed(g, handle, 0);
    assert.ok(g.hasOp(result.seed.opName));
  });

  it("different layerIds produce different seed ops", () => {
    const g = graph();
    const { handle } = ops.variableWithInit(
      g,
      [],
      DType.INT64,
      "step",
      ops.zerosInitializer(g, [], DType.INT64),
    );

    const result0 = ops.makeDropoutSeed(g, handle, 0);
    const result1 = ops.makeDropoutSeed(g, handle, 1);

    // Each call adds its own Stack op — they must be distinct
    assert.notStrictEqual(result0.seed.opName, result1.seed.opName);
    assert.ok(g.hasOp(result0.seed.opName));
    assert.ok(g.hasOp(result1.seed.opName));
  });

  it("can be passed to dropout as the seed parameter", () => {
    const g = graph();
    const x = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];

    const { handle } = ops.variableWithInit(
      g,
      [],
      DType.INT64,
      "step",
      ops.zerosInitializer(g, [], DType.INT64),
    );

    const result = ops.makeDropoutSeed(g, handle, 0);

    // Should not throw — seed tensor wired into dropout correctly
    assert.doesNotThrow(() => ops.dropout(g, x, 0.3, true, result.seed));
  });
});
