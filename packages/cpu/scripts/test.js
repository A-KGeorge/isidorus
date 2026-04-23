import { readdirSync } from "fs";
import { join } from "path";
import { spawn } from "child_process";
import { fileURLToPath } from "url";
import { dirname } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const testDir = join(__dirname, "../src/ts/__tests__");

// Get all test files
const testFiles = readdirSync(testDir)
  .filter((file) => file.endsWith(".test.ts"))
  .map((file) => join(testDir, file));

// Run node test runner with all test files serially to avoid cross-file
// native teardown races while still preserving all test coverage.
const args = [
  "--import",
  "tsx",
  "--test",
  "--test-concurrency=1",
  ...testFiles,
];

const env = { ...process.env };

// Suppress TensorFlow startup messages (oneDNN, CPU features, MLIR optimization).
// Level 3 filters out all messages. Set TF_CPP_MIN_LOG_LEVEL=0 to restore.
// The library automatically configures LD_LIBRARY_PATH and other platform-specific
// environment variables on import, so no manual setup needed here.
if (!env.TF_CPP_MIN_LOG_LEVEL) {
  env.TF_CPP_MIN_LOG_LEVEL = "3";
}

const child = spawn("node", args, {
  stdio: "inherit",
  shell: true,
  env,
});

child.on("exit", (code) => {
  process.exit(code);
});
