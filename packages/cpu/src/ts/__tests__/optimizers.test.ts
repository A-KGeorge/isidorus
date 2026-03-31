import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { graph, session, ops, optimizers, DType } from "../index.js";

// ---------------------------------------------------------------------------
// Optimizer tests
//
// All tests use L = 0.5 * w^2 (scalar, one parameter) so dL/dw = w.
// Starting from w = 1.0, every optimizer should move w toward 0.
//
// The key design: gradient tensors from g.addGradients() are passed to the
// optimizer at construction time and wired directly into the update ops.
// applyGradients() just runs those ops — TF computes the gradients as part
// of the same TF_SessionRun call. No manual gradient evaluation needed.
// ---------------------------------------------------------------------------

function f32(value: number): Buffer {
  const b = Buffer.allocUnsafe(4);
  b.writeFloatLE(value, 0);
  return b;
}

async function readScalar(
  sess: ReturnType<typeof session>,
  t: ReturnType<typeof ops.readVariable>,
): Promise<number> {
  const [out] = await sess.run([], [t]);
  return new Float32Array(out.data.buffer, out.data.byteOffset, 1)[0];
}

/**
 * Build a minimal training graph: L = 0.5 * w^2, dL/dw = w.
 * w is initialised to 1.0 so every optimizer should decrease |w| each step.
 */
function buildQuadraticGraph() {
  const g = graph();

  const { handle: wHandle, initOp: wInit } = ops.variableWithInit(
    g,
    [],
    DType.FLOAT32,
    "w",
    ops.constant(g, f32(1.0), [], DType.FLOAT32),
  );

  const wRead = ops.readVariable(g, wHandle, DType.FLOAT32);
  const loss = ops.mul(
    g,
    ops.constant(g, f32(0.5), [], DType.FLOAT32),
    ops.square(g, wRead),
  );

  // addGradients returns the grad tensor — passed to optimizer at construction.
  const [dw] = g.addGradients([loss], [wRead]);

  // Init target for w itself.
  const paramInit = ops.globalVariablesInitializer(g, [wInit], "param_init");

  return { g, wHandle, wRead, dw, paramInit };
}

// ---------------------------------------------------------------------------

describe("SGD", () => {
  it("one step: w moves from 1.0 to ~0.9 (lr=0.1, no momentum)", async () => {
    const { g, wHandle, wRead, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.SGD(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.1,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);

    await opt.applyGradients(sess, []);
    const w = await readScalar(sess, wRead);

    // w = 1.0 - 0.1 * 1.0 = 0.9
    assert.ok(Math.abs(w - 0.9) < 1e-5, `expected ~0.9, got ${w}`);
    sess.destroy();
  });

  it("20 steps converge toward 0 (lr=0.1)", async () => {
    const { g, wHandle, wRead, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.SGD(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.1,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);

    for (let i = 0; i < 20; i++) await opt.applyGradients(sess, []);

    const w = await readScalar(sess, wRead);
    // (0.9)^20 ≈ 0.1216
    assert.ok(
      Math.abs(w) < 0.15,
      `expected |w| < 0.15 after 20 steps, got ${w}`,
    );
    sess.destroy();
  });

  it("init is idempotent", async () => {
    const { g, wHandle, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.SGD(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.1,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);
    await assert.doesNotReject(() => opt.init(sess));
    sess.destroy();
  });

  it("momentum > 0: w still decreases each step from 1.0", async () => {
    const { g, wHandle, wRead, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.SGD(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.01,
      { momentum: 0.9 },
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);

    await opt.applyGradients(sess, []);
    const w = await readScalar(sess, wRead);
    assert.ok(w < 1.0, `w should decrease below 1.0, got ${w}`);
    sess.destroy();
  });
});

// ---------------------------------------------------------------------------

describe("Adam", () => {
  it("one step: w moves toward 0", async () => {
    const { g, wHandle, wRead, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.Adam(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.1,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);

    const before = await readScalar(sess, wRead);
    await opt.applyGradients(sess, []);
    const after = await readScalar(sess, wRead);

    assert.ok(
      Math.abs(after) < Math.abs(before),
      `Adam: w should decrease — before=${before}, after=${after}`,
    );
    sess.destroy();
  });

  it("10 steps: w decreases monotonically from 1.0", async () => {
    const { g, wHandle, wRead, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.Adam(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.1,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);

    let prev = await readScalar(sess, wRead);
    for (let i = 0; i < 10; i++) {
      await opt.applyGradients(sess, []);
      const cur = await readScalar(sess, wRead);
      assert.ok(
        cur < prev,
        `step ${i + 1}: w should decrease — prev=${prev}, cur=${cur}`,
      );
      prev = cur;
    }
    sess.destroy();
  });

  it("init is idempotent", async () => {
    const { g, wHandle, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.Adam(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.001,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);
    await assert.doesNotReject(() => opt.init(sess));
    sess.destroy();
  });
});

// ---------------------------------------------------------------------------

describe("RMSProp", () => {
  it("one step: w moves toward 0", async () => {
    const { g, wHandle, wRead, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.RMSProp(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.1,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);

    const before = await readScalar(sess, wRead);
    await opt.applyGradients(sess, []);
    const after = await readScalar(sess, wRead);

    assert.ok(
      Math.abs(after) < Math.abs(before),
      `RMSProp: w should decrease — before=${before}, after=${after}`,
    );
    sess.destroy();
  });

  it("10 steps converge toward 0", async () => {
    const { g, wHandle, wRead, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.RMSProp(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.1,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);

    for (let i = 0; i < 10; i++) await opt.applyGradients(sess, []);

    const w = await readScalar(sess, wRead);
    assert.ok(Math.abs(w) < 0.5, `expected |w| < 0.5 after 10 steps, got ${w}`);
    sess.destroy();
  });

  it("init is idempotent", async () => {
    const { g, wHandle, dw, paramInit } = buildQuadraticGraph();
    const opt = new optimizers.RMSProp(
      g,
      [{ handle: wHandle, grad: dw, dtype: DType.FLOAT32, name: "w" }],
      0.001,
    );
    const sess = session(g);
    await sess.run([], [], [paramInit]);
    await opt.init(sess);
    await assert.doesNotReject(() => opt.init(sess));
    sess.destroy();
  });
});
