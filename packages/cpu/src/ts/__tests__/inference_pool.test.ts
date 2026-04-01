import { describe, it } from "node:test";
import assert from "node:assert/strict";
import "../index.js"; // Initialize native addon
import { InferencePool } from "../inference-pool.js";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const MODEL_PATH = join(__dirname, "fixtures", "bench_small.pb");

describe("InferencePool Integration", () => {
  /**
   * Test 1: Full Inference Cycle
   * Verifies that the worker-pool can process a real buffer and return results.
   */
  it("should perform a real inference using worker-pool and bench_small.pb", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      strategy: "worker-pool",
      concurrency: 2,
    });

    try {
      const inputSize = 1 * 224 * 224 * 3 * 4;
      const inputData = Buffer.alloc(inputSize, 0);
      const result = await pool.infer(inputData, [1, 224, 224, 3]);

      assert.strictEqual(result.strategy, "worker-pool");
      assert.ok(result.outputs.length > 0);
      assert.ok(result.inferenceMs > 0);
      assert.ok(Buffer.isBuffer(result.outputs[0].data));
    } finally {
      await pool.destroy(); // always runs, even on assertion failure
    }
  });

  /**
   * Test 2: Concurrency and Queueing
   * Submits more requests than workers to verify the internal queue logic.
   */
  it("should handle queued requests when all workers are busy", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      strategy: "worker-pool",
      concurrency: 1,
    });

    try {
      const inputData = Buffer.alloc(1 * 224 * 224 * 3 * 4);
      const p1 = pool.infer(inputData, [1, 224, 224, 3]);
      const p2 = pool.infer(inputData, [1, 224, 224, 3]);

      // Observe the busy state — but don't hang if inference was faster than the timer
      const busyObservation = await new Promise<{
        busy: number;
        queue: number;
      }>((resolve) => {
        const id = setInterval(() => {
          const busy = pool.busyCount;
          const queue = pool.queueDepth;
          // Resolve as soon as we see anything, or once everything has settled
          if (busy > 0 || queue === 0) {
            clearInterval(id);
            resolve({ busy, queue });
          }
        }, 1); // 1ms for faster detection
      });

      // Only assert the queue state if we actually caught the mid-flight moment
      if (busyObservation.busy > 0) {
        assert.strictEqual(busyObservation.busy, 1);
        assert.strictEqual(busyObservation.queue, 1);
      }

      const [r1, r2] = await Promise.all([p1, p2]);
      assert.ok(r1 && r2);
      assert.strictEqual(pool.queueDepth, 0);
    } finally {
      await pool.destroy();
    }
  });

  /**
   * Test 3: Auto Strategy Selection
   * Verifies the model-size based switching logic.
   */
  it("should select worker-pool for bench_small.pb (Auto strategy)", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      strategy: "auto",
    });

    try {
      assert.strictEqual(pool.strategy, "worker-pool");
    } finally {
      await pool.destroy();
    }
  });

  /**
   * Test 4: Auto-discovery of inputOp and outputOps.
   * When omitted from PoolOptions, create() should infer them from the graph.
   */
  it("should auto-discover inputOp and outputOps when not specified", async () => {
    // Passing no inputOp or outputOps — create() must infer them
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      strategy: "worker-pool",
      concurrency: 1,
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
   * Test 5: Output data is always a proper Buffer.
   * Verifies the structured-clone fix — postMessage converts Buffer to
   * Uint8Array across the worker boundary; InferencePool must wrap it back.
   */
  it("should return outputs with Buffer.isBuffer() === true", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      strategy: "worker-pool",
      concurrency: 1,
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
   * Test 6: destroy() cleans up even when no inference has run.
   */
  it("should destroy cleanly without any inference calls", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      strategy: "worker-pool",
      concurrency: 1,
    });

    // Should not hang or throw
    await assert.doesNotReject(() => pool.destroy());
  });

  /**
   * Test 7: Multiple sequential inferences on the same pool.
   */
  it("should handle multiple sequential inferences", async () => {
    const pool = await InferencePool.create({
      modelPath: MODEL_PATH,
      strategy: "worker-pool",
      concurrency: 1,
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
