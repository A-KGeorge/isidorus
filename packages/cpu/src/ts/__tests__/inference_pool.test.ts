import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { InferencePool } from "../inference-pool.js";
import { availableParallelism } from "os";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// This assumes you've placed the file in packages/cpu/src/ts/__tests__/fixtures/
const MODEL_PATH = join(__dirname, "fixtures", "bench_small.pb");

describe("InferencePool Integration", () => {
  /**
   * Test 1: Full Inference Cycle
   * Verifies that the worker-pool can process a real buffer and return results.
   */
  it("should perform a real inference using worker-pool and bench_small.pb", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      inputOp: "Placeholder",
      outputOps: ["StatefulPartitionedCall"],
      strategy: "worker-pool",
      concurrency: 2,
    });

    // MobileNetV2 input: [1, 224, 224, 3] FLOAT32
    const inputSize = 1 * 224 * 224 * 3 * 4;
    const inputData = Buffer.alloc(inputSize, 0); // Zero-filled "image"

    const result = await pool.infer(inputData, [1, 224, 224, 3]);

    assert.strictEqual(result.strategy, "worker-pool");
    assert.ok(result.outputs.length > 0);
    assert.ok(result.inferenceMs > 0);
    assert.ok(Buffer.isBuffer(result.outputs[0].data));

    await pool.destroy();
  });

  /**
   * Test 2: Concurrency and Queueing
   * Submits more requests than workers to verify the internal queue logic.
   */
  it("should handle queued requests when all workers are busy", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      inputOp: "Placeholder",
      outputOps: ["StatefulPartitionedCall"],
      strategy: "worker-pool",
      concurrency: 1, // Single worker to force queueing
    });

    const inputData = Buffer.alloc(1 * 224 * 224 * 3 * 4);

    // Fire two requests: first occupies the worker, second goes to queue
    const p1 = pool.infer(inputData, [1, 224, 224, 3]);
    const p2 = pool.infer(inputData, [1, 224, 224, 3]);

    // Briefly wait for dispatch
    await new Promise<void>((resolve) => {
      const id = setInterval(() => {
        if (pool.busyCount > 0) {
          clearInterval(id);
          resolve();
        }
      }, 5);
    });

    assert.strictEqual(pool.busyCount, 1);
    assert.strictEqual(pool.queueDepth, 1);

    const [r1, r2] = await Promise.all([p1, p2]);
    assert.ok(r1 && r2);
    assert.strictEqual(pool.queueDepth, 0);

    await pool.destroy();
  });

  /**
   * Test 3: Auto Strategy Selection
   * Verifies the model-size based switching logic.
   */
  it("should select worker-pool for bench_small.pb (Auto strategy)", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      inputOp: "Placeholder",
      outputOps: ["StatefulPartitionedCall"],
      strategy: "auto",
    });

    // bench_small.pb (MobileNetV2) is ~14MB, which is < 150MB threshold
    assert.strictEqual(pool.strategy, "worker-pool");

    await pool.destroy();
  });
});
