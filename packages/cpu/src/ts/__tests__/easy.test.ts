import { describe, it, afterEach } from "node:test";
import assert from "node:assert/strict";
import {
  Model,
  Dense,
  Conv2D,
  ConvBnRelu,
  Flatten,
  ResidualBlock,
  toFloat32Array,
  toInt32Array,
  type DataLike,
} from "../index.js";

// Global list to track all models created in tests
const modelsToCleanup: Model[] = [];

// Cleanup after each test
afterEach(async () => {
  // Yield to the event loop to allow any lingering promises to settle
  await new Promise((resolve) => setImmediate(resolve));

  for (const m of modelsToCleanup) {
    m.dispose();
  }
  modelsToCleanup.length = 0;
});

// Helper function to create and track models
function trackModel<M extends Model>(model: M): M {
  modelsToCleanup.push(model);
  return model;
}

describe("easy api (Model)", () => {
  it("should compile and predict successfully", async () => {
    const model = trackModel(
      new Model(
        [2],
        [
          new Dense(4, { activation: "relu" }),
          new Dense(1, { activation: "linear" }),
        ],
      ),
    );

    model.compile({ loss: "mse", optimizer: "adam" });

    const x = new Float32Array([1.0, 2.0, 3.0, 4.0]); // 2 samples
    const preds = await model.predict(x);

    assert.equal(preds.length, 2);
  });

  it("should fit successfully", async () => {
    const model = trackModel(
      new Model([1], [new Dense(1, { activation: "linear" })]),
    );

    model.compile({ loss: "mse", optimizer: "sgd", lr: 0.1 });

    const x = new Float32Array([1.0, 2.0, 3.0]);
    const y = new Float32Array([2.0, 4.0, 6.0]); // Simple y = 2x

    const result = await model.fit(x, y, {
      epochs: 2,
      batchSize: 3,
      verbose: false,
    });
    assert.equal(result.history.length, 2);

    // Prediction should be somewhat close to y = 2x after training, but we just verify it runs.
    const preds = await model.predict(new Float32Array([4.0]));
    assert.equal(preds.length, 1);
  });

  it("should support compound layers", async () => {
    const model = trackModel(
      new Model(
        [8, 8, 3],
        [
          new ConvBnRelu(16, 3, { padding: "SAME" }),
          new ResidualBlock(16, 4, 16),
          new Flatten(),
          new Dense(2),
        ],
      ),
    );

    model.compile({ loss: "mse", optimizer: "adam" });

    const x = new Float32Array(8 * 8 * 3); // 1 sample
    x.fill(0.1);

    const preds = await model.predict(x);
    assert.equal(preds.length, 2);
  });
});

// ── Data Conversion Tests ────────────────────────────────────────────────────

describe("toFloat32Array()", () => {
  it("should accept Float32Array as-is", () => {
    const input = new Float32Array([1, 2, 3, 4]);
    const result = toFloat32Array(input);
    assert.deepEqual(result, input);
  });

  it("should convert Int32Array to Float32Array", () => {
    const input = new Int32Array([1, 2, 3, 4]);
    const result = toFloat32Array(input);
    assert.deepEqual(Array.from(result), [1, 2, 3, 4]);
  });

  it("should flatten simple number arrays", () => {
    const input = [1, 2, 3, 4];
    const result = toFloat32Array(input);
    assert.deepEqual(Array.from(result), [1, 2, 3, 4]);
  });

  it("should flatten 2D nested arrays", () => {
    const input = [
      [1, 2],
      [3, 4],
    ];
    const result = toFloat32Array(input);
    assert.deepEqual(Array.from(result), [1, 2, 3, 4]);
  });

  it("should flatten 3D nested arrays", () => {
    const input = [
      [
        [1, 2],
        [3, 4],
      ],
      [
        [5, 6],
        [7, 8],
      ],
    ];
    const result = toFloat32Array(input);
    assert.deepEqual(Array.from(result), [1, 2, 3, 4, 5, 6, 7, 8]);
  });

  it("should validate expectedElems parameter", () => {
    const input = [1, 2, 3, 4];
    assert.doesNotThrow(() => toFloat32Array(input, 4));
    assert.throws(() => toFloat32Array(input, 5), {
      message: /expected 5 elements, got 4/,
    });
  });

  it("should throw on invalid input types", () => {
    assert.throws(() => toFloat32Array("invalid" as any), {
      message: /expected array or typed array/,
    });
  });
});

describe("toInt32Array()", () => {
  it("should accept Int32Array as-is", () => {
    const input = new Int32Array([1, 2, 3, 4]);
    const result = toInt32Array(input);
    assert.deepEqual(result, input);
  });

  it("should convert Float32Array to Int32Array (flooring)", () => {
    const input = new Float32Array([1.7, 2.3, 3.9, 4.1]);
    const result = toInt32Array(input);
    assert.deepEqual(Array.from(result), [1, 2, 3, 4]);
  });

  it("should flatten and floor number arrays", () => {
    const input = [1.5, 2.5, 3.5, 4.5];
    const result = toInt32Array(input);
    assert.deepEqual(Array.from(result), [1, 2, 3, 4]);
  });

  it("should flatten 2D arrays for class indices", () => {
    const input = [
      [0, 1],
      [2, 1],
      [0, 3],
    ]; // shape [3, 2] if needed
    const result = toInt32Array(input);
    assert.deepEqual(Array.from(result), [0, 1, 2, 1, 0, 3]);
  });

  it("should validate expectedElems parameter", () => {
    const input = [0, 1, 2];
    assert.doesNotThrow(() => toInt32Array(input, 3));
    assert.throws(() => toInt32Array(input, 2), {
      message: /expected 2 elements, got 3/,
    });
  });
});

// ── Model Shape Introspection Tests ──────────────────────────────────────────

describe("Model shape introspection", () => {
  it("getInputShape() returns correct shape", async () => {
    const inputShape = [28, 28, 1];
    const model = trackModel(new Model(inputShape, [new Dense(10)]));
    assert.deepEqual(model.getInputShape(), inputShape);
  });

  it("getInputShape() returns a copy (not reference)", async () => {
    const inputShape = [28, 28, 1];
    const model = trackModel(new Model(inputShape, [new Dense(10)]));
    const retrieved = model.getInputShape();
    retrieved.push(999); // mutate retrieved
    assert.deepEqual(model.getInputShape(), inputShape); // unchanged
  });

  it("getOutputShape() throws before compile()", async () => {
    const model = trackModel(new Model([10], [new Dense(5)]));
    assert.throws(() => model.getOutputShape(), {
      message: /call compile\(\) first/,
    });
  });

  it("getOutputShape() returns correct shape after compile()", async () => {
    const model = trackModel(
      new Model(
        [10],
        [
          new Dense(8, { activation: "relu" }),
          new Dense(5, { activation: "softmax" }),
        ],
      ),
    );
    model.compile({ loss: "mse", optimizer: "adam" });
    const outShape = model.getOutputShape();
    assert.equal(outShape[0], null); // batch dimension
    assert.equal(outShape[1], 5); // output units from last Dense
  });

  it("getOutputShape() returns a copy (not reference)", async () => {
    const model = trackModel(new Model([10], [new Dense(5)]));
    model.compile({ loss: "mse", optimizer: "adam" });
    const shape1 = model.getOutputShape();
    const shape2 = model.getOutputShape();
    assert.notEqual(shape1, shape2); // different references
    assert.deepEqual(shape1, shape2); // but same values
  });
});

// ── Model Data Conversion Integration Tests ─────────────────────────────────

describe("Model with auto data conversion", () => {
  it("predict() accepts JS arrays", async () => {
    const model = trackModel(
      new Model([2], [new Dense(1, { activation: "linear" })]),
    );
    model.compile({ loss: "mse", optimizer: "adam" });

    // Pass plain JS arrays instead of Float32Array
    const preds = await model.predict([1, 2, 3, 4]); // 2 samples of [1,2] and [3,4]
    assert.equal(preds.length, 2);
  });

  it("fit() accepts JS arrays for x and y", async () => {
    const model = trackModel(
      new Model([1], [new Dense(1, { activation: "linear" })]),
    );
    model.compile({ loss: "mse", optimizer: "sgd", lr: 0.1 });

    // Use plain JS arrays
    const result = await model.fit([1, 2, 3], [2, 4, 6], {
      epochs: 1,
      batchSize: 3,
      verbose: false,
    });
    assert.equal(result.history.length, 1);
  });

  it("predict() efficiently converts Float32Array (zero-copy)", async () => {
    const model = trackModel(
      new Model([2], [new Dense(1, { activation: "linear" })]),
    );
    model.compile({ loss: "mse", optimizer: "adam" });

    // Pass typed array directly (should be zero-copy)
    const typed = new Float32Array([1, 2, 3, 4]);
    const preds = await model.predict(typed);
    assert.equal(preds.length, 2);
  });
});
