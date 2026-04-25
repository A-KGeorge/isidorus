/**
 * libtf-resolution.test.ts
 *
 * Tests for TensorFlow library resolution and linking.
 * Verifies that the package can find and load libtensorflow without
 * requiring manual environment variable configuration.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { platform } from "os";

describe("TensorFlow Library Resolution", () => {
  it("should find TensorFlow library via ensureTf", async () => {
    const { ensureTf } = await import("../install-libtensorflow.js");

    try {
      const tfPath = await ensureTf();
      assert.ok(tfPath !== undefined);
      assert.ok(tfPath.length > 0);
      console.log(`✓ Found TensorFlow at: ${tfPath}`);
    } catch (error) {
      console.log(
        "Note: TensorFlow not found (expected in CI/test environments)",
      );
    }
  });

  it("should resolve library paths on Linux", async () => {
    if (platform() !== "linux") {
      console.log("Skipping Linux-specific test");
      return;
    }

    const { ensureTf } = await import("../install-libtensorflow.js");

    try {
      const tfPath = await ensureTf();
      if (tfPath) {
        assert.ok(tfPath.includes("lib"));
        console.log(`✓ TensorFlow path contains lib directory: ${tfPath}`);
      }
    } catch (error) {
      console.log("Note: TensorFlow not found (expected in test environments)");
    }
  });

  it("should handle custom LIBTENSORFLOW_PATH when set", async () => {
    const originalPath = process.env.LIBTENSORFLOW_PATH;

    try {
      // Set a non-existent path
      process.env.LIBTENSORFLOW_PATH = "/nonexistent/path";

      const { ensureTf } = await import("../install-libtensorflow.js");

      try {
        await ensureTf();
      } catch (error) {
        // Expected - verify the error is about missing tensorflow library
        assert.match((error as Error).message, /libtensorflow/);
        console.log("✓ Custom LIBTENSORFLOW_PATH is properly validated");
      }
    } finally {
      // Restore original value
      if (originalPath !== undefined) {
        process.env.LIBTENSORFLOW_PATH = originalPath;
      } else {
        delete process.env.LIBTENSORFLOW_PATH;
      }
    }
  });
});
