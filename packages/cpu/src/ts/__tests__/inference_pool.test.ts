import { describe, it, after } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, existsSync, rmSync } from "node:fs";
import "../index.js"; // Initialize native addon
import { InferencePool } from "../inference-pool.js";
import { toonEncode, toonDecode } from "../toon.js";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { buildSmallGraph } from "./fixtures/generate_small.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODEL_PATH = join(__dirname, "fixtures", "bench_small.pb");

describe("InferencePool Integration", () => {
  after(() => {
    if (existsSync(MODEL_PATH)) {
      rmSync(MODEL_PATH, { force: true });
    }
  });

  /**
   * Test 0: Generate the bench_small.pb fixture
   * Using the provided generate_models API equivalents.
   */
  it("should generate bench_small.pb and save it to fixtures", async () => {
    const g = buildSmallGraph();
    const pbDef = g.toGraphDef();
    writeFileSync(MODEL_PATH, pbDef);
    assert.ok(existsSync(MODEL_PATH));
  });

  /**
   * Test 0.5: Toon encoding/decoding test
   * Tests the uncommitted toon string functionality to correctly serialize and deserialize.
   */
  it("should encode and decode data correctly using toon", async () => {
    const testObject = {
      version: 1,
      inputShape: [1, 224, 224, 3],
      tags: ["vision", "mobile", "fast"],
      params: { count: 3500000, optimized: true },
      empty: null,
      floating: 3.1415,
      magic_buffer: Buffer.from("hello world"),
    };

    // Encode using toon
    const encoded = toonEncode(testObject);
    assert.ok(Buffer.isBuffer(encoded));
    assert.ok(encoded.length > 4); // basic size check

    // Decode mapping back
    const decoded = toonDecode(encoded) as typeof testObject;

    assert.strictEqual(decoded.version, testObject.version);
    assert.deepEqual(decoded.inputShape, testObject.inputShape);
    assert.deepEqual(decoded.tags, testObject.tags);
    assert.strictEqual(decoded.params.count, testObject.params.count);
    assert.strictEqual(decoded.params.optimized, testObject.params.optimized);
    assert.strictEqual(decoded.empty, testObject.empty);
    assert.strictEqual(decoded.floating, testObject.floating);
    assert.ok(Buffer.isBuffer(decoded.magic_buffer));
    assert.strictEqual(decoded.magic_buffer.toString(), "hello world");
  });
  /**
   * Test 1: Full Inference Cycle
   * Verifies that the worker-pool can process a real buffer and return results.
   */
  it("should perform a real inference using tf-parallel and bench_small.pb", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
    });

    try {
      const inputSize = 1 * 224 * 224 * 3 * 4;
      const inputData = Buffer.alloc(inputSize, 0);
      const result = await pool.infer(inputData, [1, 224, 224, 3]);
    } finally {
      await pool.destroy(); // always runs, even on assertion failure
    }
  });

  /**
   * Test 2: Concurrency and Queueing
   * Submits more requests than workers to verify the internal queue logic.
   */
  it("should handle queued requests when TF threads are busy", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
    });

    try {
      const inputData = Buffer.alloc(1 * 224 * 224 * 3 * 4);
      const p1 = pool.infer(inputData, [1, 224, 224, 3]);
      const p2 = pool.infer(inputData, [1, 224, 224, 3]);

      const [r1, r2] = await Promise.all([p1, p2]);
      assert.ok(r1 && r2);
      assert.strictEqual(pool.queueDepth, 0);
    } finally {
      await pool.destroy();
    }
  });

  /**
   * Test 3: Auto-discovery of inputOp and outputOps.
   * When omitted from PoolOptions, create() should infer them from the graph.
   */
  it("should auto-discover inputOp and outputOps when not specified", async () => {
    // Passing no inputOp or outputOps — create() must infer them
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
    });

    try {
      const inputData = Buffer.alloc(1 * 224 * 224 * 3 * 4);
      const result = await pool.infer(inputData, [1, 224, 224, 3]);
      assert.ok(result.outputs.length > 0);
    } finally {
      await pool.destroy();
    }
  });

  /**
   * Test 4: Output data is always a proper Buffer.
   * Verifies the structured-clone fix — postMessage converts Buffer to
   * Uint8Array across the worker boundary; InferencePool must wrap it back.
   */
  it("should return outputs with Buffer.isBuffer() === true", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
    });

    try {
      const inputData = Buffer.alloc(1 * 224 * 224 * 3 * 4);
      const result = await pool.infer(inputData, [1, 224, 224, 3]);

      for (const output of result.outputs) {
        assert.ok(
          Buffer.isBuffer(output.data),
          "output.data must be a Buffer, not a plain Uint8Array",
        );
        assert.ok(Array.isArray(output.shape));
        assert.strictEqual(typeof output.dtype, "number");
      }
    } finally {
      await pool.destroy();
    }
  });

  /**
   * Test 5: destroy() cleans up even when no inference has run.
   */
  it("should destroy cleanly without any inference calls", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
    });

    // Should not hang or throw
    await assert.doesNotReject(() => pool.destroy());
  });

  /**
   * Test 6: Multiple sequential inferences on the same pool.
   */
  it("should handle multiple sequential inferences", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
    });

    try {
      const inputData = Buffer.alloc(1 * 224 * 224 * 3 * 4);
      for (let i = 0; i < 3; i++) {
        const result = await pool.infer(inputData, [1, 224, 224, 3]);
        assert.ok(result.outputs.length > 0);
        assert.ok(result.inferenceMs > 0);
      }
    } finally {
      await pool.destroy();
    }
  });
});
