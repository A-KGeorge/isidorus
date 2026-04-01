import { describe, it } from "node:test";
import assert from "node:assert/strict";
import {
  graph,
  session,
  optimizers,
  Sequential,
  Dense,
  Flatten,
  Conv2D,
  DType,
} from "../index.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function f32(values: number[]): Buffer {
  return Buffer.from(new Float32Array(values).buffer);
}

function i32(values: number[]): Buffer {
  return Buffer.from(new Int32Array(values).buffer);
}

// ---------------------------------------------------------------------------
// Dense layer
// ---------------------------------------------------------------------------
describe("Dense layer", () => {
  it("output shape [null, units]", () => {
    const g = graph();
    const layer = new Dense(16, { activation: "relu", name: "fc1" });
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    const outShape = layer.build(g, input, [null, 8]);
    assert.deepStrictEqual(outShape, [null, 16]);
  });

  it("W + b params by default", () => {
    const g = graph();
    const layer = new Dense(8, { name: "fc" });
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    layer.build(g, input, [null, 4]);
    assert.strictEqual(layer.layerParams.length, 2);
  });

  it("useBias=false — W param only", () => {
    const g = graph();
    const layer = new Dense(4, { useBias: false, name: "no_bias" });
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    layer.build(g, input, [null, 4]);
    assert.strictEqual(layer.layerParams.length, 1);
  });

  it("throws for unknown last input dim", () => {
    const g = graph();
    const layer = new Dense(4, { name: "bad" });
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    assert.throws(
      () => layer.build(g, input, [null, null]),
      /last input dim must be known/,
    );
  });
});

// ---------------------------------------------------------------------------
// Flatten layer
// ---------------------------------------------------------------------------
describe("Flatten layer", () => {
  it("known dims: [null,4,4,8] → [null,128]", () => {
    const g = graph();
    const layer = new Flatten();
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    assert.deepStrictEqual(layer.build(g, input, [null, 4, 4, 8]), [null, 128]);
  });

  it("unknown dims: flat size is null", () => {
    const g = graph();
    const layer = new Flatten();
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    assert.strictEqual(layer.build(g, input, [null, null, null, 3])[1], null);
  });

  it("no params", () => {
    const g = graph();
    const layer = new Flatten();
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    layer.build(g, input, [null, 4, 4, 2]);
    assert.strictEqual(layer.layerParams.length, 0);
  });
});

// ---------------------------------------------------------------------------
// Conv2D layer
// ---------------------------------------------------------------------------
describe("Conv2D layer", () => {
  it("SAME padding stride 1: spatial dims preserved", () => {
    const g = graph();
    const layer = new Conv2D(32, {
      kernelSize: 3,
      padding: "SAME",
      name: "c1",
    });
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    assert.deepStrictEqual(layer.build(g, input, [null, 8, 8, 3]), [
      null,
      8,
      8,
      32,
    ]);
  });

  it("VALID padding stride 2: spatial dims reduced", () => {
    const g = graph();
    const layer = new Conv2D(16, {
      kernelSize: 3,
      strides: 2,
      padding: "VALID",
      name: "c2",
    });
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    // H' = ceil((8 - 3 + 1) / 2) = 3
    assert.deepStrictEqual(layer.build(g, input, [null, 8, 8, 1]), [
      null,
      3,
      3,
      16,
    ]);
  });

  it("W + b params", () => {
    const g = graph();
    const layer = new Conv2D(8, { name: "c3" });
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    layer.build(g, input, [null, 4, 4, 1]);
    assert.strictEqual(layer.layerParams.length, 2);
  });

  it("throws for non-4D input", () => {
    const g = graph();
    const layer = new Conv2D(8, { name: "c4" });
    const input = g.addOp("Placeholder", [], {
      dtype: { kind: "type", value: DType.FLOAT32 },
    })[0];
    assert.throws(() => layer.build(g, input, [null, 32]), /expects 4D input/);
  });
});

// ---------------------------------------------------------------------------
// Sequential compile
// ---------------------------------------------------------------------------
describe("Sequential compile", () => {
  it("Dense stack compiles without error", () => {
    const g = graph();
    const model = new Sequential(g, [
      new Dense(8, { activation: "relu", name: "fc1" }),
      new Dense(2, { activation: "softmax", name: "out" }),
    ]);
    assert.doesNotThrow(() =>
      model.compile({
        loss: "sparse_categorical_crossentropy",
        inputShape: [8],
      }),
    );
  });

  it("after compile: params has correct count (W + b per Dense)", () => {
    const g = graph();
    const model = new Sequential(g, [
      new Dense(4, { name: "fc1" }), // W + b = 2 params
      new Dense(2, { name: "out" }), // W + b = 2 params
    ]);
    model.compile({ loss: "sparse_categorical_crossentropy", inputShape: [4] });
    assert.strictEqual(model.params.length, 4);
  });

  it("all params have non-null grad tensors after compile", () => {
    const g = graph();
    const model = new Sequential(g, [
      new Dense(4, { name: "fc1" }),
      new Dense(2, { name: "out" }),
    ]);
    model.compile({ loss: "sparse_categorical_crossentropy", inputShape: [4] });
    for (const p of model.params) {
      assert.ok(
        p.grad !== null && p.grad !== undefined,
        `param ${p.name} should have a grad tensor`,
      );
      assert.ok(
        typeof p.grad.opName === "string",
        `param ${p.name} grad should have an opName`,
      );
    }
  });

  it("Conv2D + Flatten + Dense compiles", () => {
    const g = graph();
    const model = new Sequential(g, [
      new Conv2D(4, { kernelSize: 3, activation: "relu", name: "conv1" }),
      new Flatten(),
      new Dense(2, { activation: "softmax", name: "out" }),
    ]);
    assert.doesNotThrow(() =>
      model.compile({
        loss: "sparse_categorical_crossentropy",
        inputShape: [4, 4, 1],
      }),
    );
  });

  it("compile twice throws", () => {
    const g = graph();
    const model = new Sequential(g, [new Dense(4, { name: "fc" })]);
    model.compile({ loss: "mse", inputShape: [4], labelDtype: DType.FLOAT32 });
    assert.throws(
      () =>
        model.compile({
          loss: "mse",
          inputShape: [4],
          labelDtype: DType.FLOAT32,
        }),
      /called twice/,
    );
  });

  it("pre-compile accessors throw", () => {
    const g = graph();
    const model = new Sequential(g, [new Dense(4, { name: "fc" })]);
    assert.throws(() => model.output, /call compile\(\) first/);
    assert.throws(() => model.loss, /call compile\(\) first/);
    assert.throws(() => model.params, /call compile\(\) first/);
    assert.throws(() => model.xPlaceholder, /call compile\(\) first/);
  });
});

// ---------------------------------------------------------------------------
// Sequential end-to-end — Dense MLP
// ---------------------------------------------------------------------------
describe("Sequential — Dense MLP training", () => {
  it("trainStep returns a finite positive loss", async () => {
    const g = graph();
    const model = new Sequential(g, [
      new Dense(8, { activation: "relu", name: "fc1" }),
      new Dense(2, { activation: "softmax", name: "out" }),
    ]);
    model.compile({ loss: "sparse_categorical_crossentropy", inputShape: [4] });

    const opt = new optimizers.Adam(g, model.params, 0.01);
    const sess = session(g);
    await model.init(sess, opt);

    const { loss } = await model.trainStep(
      sess,
      opt,
      f32([0.1, 0.2, 0.3, 0.4]),
      i32([1]),
      [1, 4],
      [1],
    );

    assert.ok(Number.isFinite(loss), `loss should be finite, got ${loss}`);
    assert.ok(loss > 0, `loss should be positive, got ${loss}`);
    sess.destroy();
  });

  it("loss decreases over 20 steps on a fixed batch", async () => {
    const g = graph();
    // 4-sample 2-class dataset
    const xData = f32([1, 0, 0, 1, 1, 1, 0, 0]);
    const yData = i32([0, 1, 0, 1]);

    const model = new Sequential(g, [
      new Dense(16, { activation: "relu", name: "fc1" }),
      new Dense(2, { activation: "softmax", name: "out" }),
    ]);
    model.compile({ loss: "sparse_categorical_crossentropy", inputShape: [2] });

    const opt = new optimizers.Adam(g, model.params, 0.1);
    const sess = session(g);
    await model.init(sess, opt);

    const { loss: first } = await model.trainStep(
      sess,
      opt,
      xData,
      yData,
      [4, 2],
      [4],
    );

    for (let i = 0; i < 19; i++)
      await model.trainStep(sess, opt, xData, yData, [4, 2], [4]);

    const { loss: last } = await model.trainStep(
      sess,
      opt,
      xData,
      yData,
      [4, 2],
      [4],
    );

    assert.ok(
      last < first,
      `loss should decrease: first=${first.toFixed(4)}, last=${last.toFixed(
        4,
      )}`,
    );
    sess.destroy();
  });

  it("predict: output shape [batch, classes] and each row sums to ~1", async () => {
    const g = graph();
    const model = new Sequential(g, [
      new Dense(4, { activation: "relu", name: "fc1" }),
      new Dense(3, { activation: "softmax", name: "out" }),
    ]);
    model.compile({ loss: "sparse_categorical_crossentropy", inputShape: [6] });

    const opt = new optimizers.SGD(g, model.params, 0.01);
    const sess = session(g);
    await model.init(sess, opt);

    const out = await model.predict(
      sess,
      f32(new Array(2 * 6).fill(0.5)),
      [2, 6],
    );

    assert.deepStrictEqual(out.shape, [2, 3]);
    assert.strictEqual(out.dtype, DType.FLOAT32);

    const vals = new Float32Array(out.data.buffer, out.data.byteOffset, 6);
    const row0 = vals[0] + vals[1] + vals[2];
    assert.ok(
      Math.abs(row0 - 1.0) < 1e-4,
      `softmax row 0 should sum to 1, got ${row0}`,
    );
    sess.destroy();
  });

  it("MSE loss on regression model", async () => {
    const g = graph();
    const model = new Sequential(g, [
      new Dense(8, { activation: "relu", name: "fc1" }),
      new Dense(1, { activation: "linear", name: "out" }),
    ]);
    model.compile({
      loss: "mse",
      inputShape: [4],
      labelDtype: DType.FLOAT32,
    });

    const opt = new optimizers.Adam(g, model.params, 0.01);
    const sess = session(g);
    await model.init(sess, opt);

    // Target: output 0.5 for input [0.1, 0.2, 0.3, 0.4]
    const { loss } = await model.trainStep(
      sess,
      opt,
      f32([0.1, 0.2, 0.3, 0.4]),
      f32([0.5]),
      [1, 4],
      [1, 1],
      DType.FLOAT32,
    );

    assert.ok(Number.isFinite(loss), `MSE loss should be finite, got ${loss}`);
    sess.destroy();
  });
});

// ---------------------------------------------------------------------------
// Sequential end-to-end — Conv2D
// ---------------------------------------------------------------------------
describe("Sequential — Conv2D", () => {
  it("predict: correct output shape", async () => {
    const g = graph();
    const model = new Sequential(g, [
      new Conv2D(4, { kernelSize: 3, activation: "relu", name: "conv1" }),
      new Flatten(),
      new Dense(2, { activation: "softmax", name: "out" }),
    ]);
    model.compile({
      loss: "sparse_categorical_crossentropy",
      inputShape: [4, 4, 1],
    });

    const opt = new optimizers.SGD(g, model.params, 0.01);
    const sess = session(g);
    await model.init(sess, opt);

    const out = await model.predict(
      sess,
      f32(new Array(4 * 4 * 1).fill(0.1)),
      [1, 4, 4, 1],
    );

    assert.deepStrictEqual(out.shape, [1, 2]);
    sess.destroy();
  });

  it("trainStep: finite loss", async () => {
    const g = graph();
    const model = new Sequential(g, [
      new Conv2D(4, { kernelSize: 3, activation: "relu", name: "conv1" }),
      new Flatten(),
      new Dense(2, { activation: "softmax", name: "out" }),
    ]);
    model.compile({
      loss: "sparse_categorical_crossentropy",
      inputShape: [4, 4, 1],
    });

    const opt = new optimizers.Adam(g, model.params, 0.001);
    const sess = session(g);
    await model.init(sess, opt);

    const { loss } = await model.trainStep(
      sess,
      opt,
      f32(new Array(4 * 4 * 1).fill(0.1)),
      i32([0]),
      [1, 4, 4, 1],
      [1],
    );

    assert.ok(Number.isFinite(loss), `loss should be finite, got ${loss}`);
    sess.destroy();
  });

  it("loss decreases over 10 steps on a fixed image", async () => {
    const g = graph();
    const model = new Sequential(g, [
      new Conv2D(8, { kernelSize: 3, activation: "relu", name: "conv1" }),
      new Flatten(),
      new Dense(2, { activation: "softmax", name: "out" }),
    ]);
    model.compile({
      loss: "sparse_categorical_crossentropy",
      inputShape: [4, 4, 1],
    });

    const opt = new optimizers.Adam(g, model.params, 0.01);
    const sess = session(g);
    await model.init(sess, opt);

    const xBuf = f32(new Array(4 * 4 * 1).fill(0.5));
    const yBuf = i32([1]);
    const xShape = [1, 4, 4, 1];
    const yShape = [1];

    const { loss: first } = await model.trainStep(
      sess,
      opt,
      xBuf,
      yBuf,
      xShape,
      yShape,
    );

    for (let i = 0; i < 9; i++)
      await model.trainStep(sess, opt, xBuf, yBuf, xShape, yShape);

    const { loss: last } = await model.trainStep(
      sess,
      opt,
      xBuf,
      yBuf,
      xShape,
      yShape,
    );

    assert.ok(
      last < first,
      `conv loss should decrease: first=${first.toFixed(
        4,
      )}, last=${last.toFixed(4)}`,
    );
    sess.destroy();
  });
});
